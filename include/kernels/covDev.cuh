#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <kernels/dslash_pack.cuh>

namespace quda
{

  /**
     @brief Parameter structure for driving the covariatnt derivative operator
  */
  template <typename Float, int nColor_, QudaReconstructType reconstruct_, int nDim>
  struct CovDevArg : DslashArg<Float, nDim> {
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    typedef typename gauge_mapper<Float, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost>::type G;

    typedef typename mapper<Float>::type real;

    F out;      /** output vector field */
    const F in; /** input vector field */
    const G U;  /** the gauge field */
    int mu;     /** The direction in which to apply the derivative */

    CovDevArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, int mu, int parity, bool dagger,
              const int *comm_override) :
      DslashArg<Float, nDim>(in, U, parity, dagger, false, 1, spin_project, comm_override),
      out(out),
      in(in),
      U(U),
      mu(mu)
    {
      if (in.V() == out.V()) errorQuda("Aliasing pointers");
      checkOrder(out, in);        // check all orders match
      checkPrecision(out, in, U); // check all precisions match
      checkLocation(out, in, U);  // check all locations match
      if (!out.isNative() || !in.isNative() || !U.isNative())
        errorQuda("Unsupported field order colorspinor(in)=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());
      pushKernelPackT(true); // non-spin projection requires kernel packing
    }

    virtual ~CovDevArg() { popKernelPackT(); }
  };

  /**
     Applies the off-diagonal part of the covariant derivative operator

     @param[out] out The out result field
     @param[in,out] arg Parameter struct
     @param[in] U The gauge field
     @param[in] coord Site coordinate struct
     @param[in] x_cb The checker-boarded site index. This is a 4-d index only
     @param[in] parity The site parity
     @param[in] idx Thread index (equal to face index for exterior kernels)
     @param[in] thread_dim Which dimension this thread corresponds to (fused exterior only)

  */
  template <int nParity, bool dagger, KernelType kernel_type, int mu, typename Coord, typename Arg, typename Vector>
  __device__ __host__ inline void applyCovDev(Vector &out, Arg &arg, Coord &coord, int parity,
                                              int idx, int thread_dim, bool &active)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = (arg.nParity == 2) ? 1 - parity : 0;

    const int d = mu % 4;

    if (mu < 4) { // Forward gather - compute fwd offset for vector fetch

      const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
      const bool ghost = (coord[d] + 1 >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

      const Link U = arg.U(d, coord.x_cb, parity);

      if (doHalo<kernel_type>(d) && ghost) {

        const int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);
        const Vector in = arg.in.Ghost(d, 1, ghost_idx, their_spinor_parity);

        out += U * in;
      } else if (doBulk<kernel_type>() && !ghost) {

        const Vector in = arg.in(fwd_idx, their_spinor_parity);
        out += U * in;
      }

    } else { // Backward gather - compute back offset for spinor and gauge fetch

      const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
      const int gauge_idx = back_idx;

      const bool ghost = (coord[d] - 1 < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

      if (doHalo<kernel_type>(d) && ghost) {

        const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
        const Link U = arg.U.Ghost(d, ghost_idx, 1 - parity);
        const Vector in = arg.in.Ghost(d, 0, ghost_idx, their_spinor_parity);

        out += conj(U) * in;
      } else if (doBulk<kernel_type>() && !ghost) {

        const Link U = arg.U(d, gauge_idx, 1 - parity);
        const Vector in = arg.in(back_idx, their_spinor_parity);

        out += conj(U) * in;
      }
    } // Forward/backward derivative
  }

  // out(x) = M*in
  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg> struct covDev : dslash_default {

    Arg &arg;
    constexpr covDev(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    __device__ __host__ inline void operator()(int idx, int s, int parity)
    {
      using real = typename mapper<typename Arg::Float>::type;
      using Vector = ColorSpinor<real, Arg::nColor, 4>;

      // is thread active (non-trival for fused kernel only)
      bool active = kernel_type == EXTERIOR_KERNEL_ALL ? false : true;

      // which dimension is thread working on (fused kernel only)
      int thread_dim;

      auto coord = getCoords<QUDA_4D_PC, kernel_type, Arg>(arg, idx, s, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;

      switch (arg.mu) { // ensure that mu is known to compiler for indexing in applyCovDev (avoid register spillage)
      case 0:
        applyCovDev<nParity, dagger, kernel_type, 0>(out, arg, coord, parity, idx, thread_dim, active);
        break;
      case 1:
        applyCovDev<nParity, dagger, kernel_type, 1>(out, arg, coord, parity, idx, thread_dim, active);
        break;
      case 2:
        applyCovDev<nParity, dagger, kernel_type, 2>(out, arg, coord, parity, idx, thread_dim, active);
        break;
      case 3:
        applyCovDev<nParity, dagger, kernel_type, 3>(out, arg, coord, parity, idx, thread_dim, active);
        break;
      case 4:
        applyCovDev<nParity, dagger, kernel_type, 4>(out, arg, coord, parity, idx, thread_dim, active);
        break;
      case 5:
        applyCovDev<nParity, dagger, kernel_type, 5>(out, arg, coord, parity, idx, thread_dim, active);
        break;
      case 6:
        applyCovDev<nParity, dagger, kernel_type, 6>(out, arg, coord, parity, idx, thread_dim, active);
        break;
      case 7:
        applyCovDev<nParity, dagger, kernel_type, 7>(out, arg, coord, parity, idx, thread_dim, active);
        break;
      }

      if (kernel_type != INTERIOR_KERNEL && active) {
        Vector x = arg.out(coord.x_cb, my_spinor_parity);
        out += x;
      }

      if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(coord.x_cb, my_spinor_parity) = out;
    }
  };

} // namespace quda
