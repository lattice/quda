#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <kernels/dslash_pack.cuh> // for the packing kernel

namespace quda
{

  /**
     @brief Parameter structure for driving the covariatnt derivative operator
  */
  template <typename Float, int nSpin_, int nColor_, int nDim, typename DDArg, QudaReconstructType reconstruct_>
  struct LaplaceArg : DslashArg<Float, nDim, DDArg> {
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = nSpin_;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, true>::type F;

    using Ghost = typename colorspinor::GhostNOrder<Float, nSpin, nColor, colorspinor::getNative<Float>(nSpin),
                                                    spin_project, spinor_direct_load, false>;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    typedef typename gauge_mapper<Float, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost>::type G;

    typedef typename mapper<Float>::type real;

    F out[MAX_MULTI_RHS];  /** output vector field */
    F in[MAX_MULTI_RHS];   /** input vector field */
    const Ghost halo_pack; /** accessor used for writing the halo field */
    const Ghost halo;      /** accessor used for reading the halo field */
    F x[MAX_MULTI_RHS];    /** input vector field for xpay*/
    const G U;    /** the gauge field */
    const real a; /** xpay scale factor - can be -kappa or -kappa^2 */
    const real b; /** used by Wuppetal smearing kernel */
    int dir;      /** The direction from which to omit the derivative */

    LaplaceArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
               const ColorSpinorField &halo, const GaugeField &U, int dir, double a, double b,
               cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override) :
      DslashArg<Float, nDim, DDArg>(out, in, halo, U, x, parity, dagger, a != 0.0 ? true : false, 1, false,
                                    comm_override),
      halo_pack(halo),
      halo(halo),
      U(U),
      a(a),
      b(b),
      dir(dir)
    {
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->in[i] = in[i];
        this->x[i] = x[i];
      }
      if (dir < 3 || dir > 4) errorQuda("Unsupported laplace direction %d (must be 3 or 4)", dir);
    }
  };

  /**
     Applies the off-diagonal part of the covariant derivative operator

     @param[out] out The out result field
     @param[in,out] arg Parameter struct
     @param[in] U The gauge field
     @param[in] coord Site coordinate struct
     @param[in] parity The site parity
     @param[in] idx Thread index (equal to face index for exterior kernels)
     @param[in] thread_dim Which dimension this thread corresponds to (fused exterior only)

  */
  template <int nParity, bool dagger, KernelType kernel_type, int dir, typename Coord, typename Arg, typename Vector>
  __device__ __host__ inline void applyLaplace(Vector &out, Arg &arg, Coord &coord, int parity, int, int thread_dim,
                                               bool &active, int src_idx)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = (arg.nParity == 2) ? 1 - parity : 0;

#pragma unroll
    for (int d = 0; d < Arg::nDim; d++) { // loop over dimension
      if (d != dir) {
        if (arg.dd_in.doHopping(coord, d, +1)) {
          // Forward gather - compute fwd offset for vector fetch
          const bool ghost = (coord[d] + 1 >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);
	  
          if (doHalo<kernel_type>(d) && ghost) {
	    
            // const int ghost_idx = ghostFaceIndexStaggered<1>(coord, arg.dim, d, 1);
            const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
            const Link U = arg.U(d, coord.x_cb, parity);
            const Vector in = arg.halo.Ghost(d, 1, ghost_idx + src_idx * arg.dc.ghostFaceCB[d], their_spinor_parity);

            out += U * in;
          } else if (doBulk<kernel_type>() && !ghost) {

            const int fwd_idx = linkIndexP1(coord, arg.dim, d);
            const Link U = arg.U(d, coord.x_cb, parity);
            const Vector in = arg.in[src_idx](fwd_idx, their_spinor_parity);

            out += U * in;
          }
        }
        if (arg.dd_in.doHopping(coord, d, -1)) {
          // Backward gather - compute back offset for spinor and gauge fetch

          const int back_idx = linkIndexM1(coord, arg.dim, d);
          const int gauge_idx = back_idx;

          const bool ghost = (coord[d] - 1 < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

          if (doHalo<kernel_type>(d) && ghost) {

            // const int ghost_idx = ghostFaceIndexStaggered<0>(coord, arg.dim, d, 1);
            const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);

            const Link U = arg.U.Ghost(d, ghost_idx, 1 - parity);
            const Vector in = arg.halo.Ghost(d, 0, ghost_idx + src_idx * arg.dc.ghostFaceCB[d], their_spinor_parity);

            out += conj(U) * in;
          } else if (doBulk<kernel_type>() && !ghost) {

            const Link U = arg.U(d, gauge_idx, 1 - parity);
            const Vector in = arg.in[src_idx](back_idx, their_spinor_parity);

            out += conj(U) * in;
          }
        }
      }
    }
  }
  
  // out(x) = M*in
  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg> struct laplace : dslash_default {

    const Arg &arg;
    constexpr laplace(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_idx, int parity)
    {
      using real = typename mapper<typename Arg::Float>::type;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      // is thread active (non-trival for fused kernel only)
      bool active = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true;

      // which dimension is thread working on (fused kernel only)
      int thread_dim;

      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, 0, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;
      if (arg.dd_out.isZero(coord)) {
        if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](coord.x_cb, my_spinor_parity) = out;
        return;
      }

      // We instantiate two kernel types:
      // case 4 is an operator in all x,y,z,t dimensions
      // case 3 is a spatial operator only, the t dimension is omitted.
      switch (arg.dir) {
      case 3:
        applyLaplace<nParity, dagger, mykernel_type, 3>(out, arg, coord, parity, idx, thread_dim, active, src_idx);
        break;
      case 4:
      default:
        applyLaplace<nParity, dagger, mykernel_type, -1>(out, arg, coord, parity, idx, thread_dim, active, src_idx);
        break;
      }

      if (xpay && mykernel_type == INTERIOR_KERNEL && arg.dd_x.isZero(coord)) {
        out = arg.a * out;
      } else if (xpay && mykernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x[src_idx](coord.x_cb, my_spinor_parity);
        out = arg.a * out + arg.b * x;
      } else if (mykernel_type != INTERIOR_KERNEL) {
        Vector x = arg.out[src_idx](coord.x_cb, my_spinor_parity);
        out = x + (xpay ? arg.a * out : out);
      }

      if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](coord.x_cb, my_spinor_parity) = out;
    }
  };

} // namespace quda
