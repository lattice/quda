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
     @brief Parameter structure for driving the covariant derivative operator
  */
  template <typename Float, int nSpin_, int nColor_, int nDim, QudaReconstructType reconstruct_>
  struct StaggeredQSmearArg : DslashArg<Float, nDim> {
    static constexpr int nColor = 3;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, true>::type;

    using Ghost = typename colorspinor::GhostNOrder<Float, nSpin, nColor, colorspinor::getNative<Float>(nSpin),
                                                    spin_project, spinor_direct_load, false>;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    using G = typename gauge_mapper<Float, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost>::type;

    using real = typename mapper<Float>::type;

    F out[MAX_MULTI_RHS]; /** output vector field */
    F in[MAX_MULTI_RHS];  /** input vector field */
    Ghost halo_pack;      /** input vector field used in packing to be able to independently resetGhost */
    Ghost halo;           /** input vector field used in packing to be able to independently resetGhost */
    const G U;       /** the gauge field */
    int dir;         /** The direction from which to omit the derivative */
    int t0;
    bool is_t0_kernel;
    int t0_offset;
    int t0_face_offset[4];
    int face_size[4];
    int t0_face_size[4];
    int threadDimMapUpper_t0[4];
    int threadDimMapLower_t0[4];

    StaggeredQSmearArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                       const ColorSpinorField &halo, const GaugeField &U, int t0, bool is_t0_kernel, int parity,
                       int dir, bool dagger, const int *comm_override) :
      DslashArg<Float, nDim>(out, in, halo, U, in, parity, dagger, false, 3, false, comm_override),
      halo_pack(halo, 3),
      halo(halo, 3),
      U(U),
      dir(dir),
      t0(t0),
      is_t0_kernel(is_t0_kernel),
      t0_offset(is_t0_kernel ? in.VolumeCB() / in.X(3) : 0)
    {
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->in[i] = in[i];
      }

      if (dir < 3 || dir > 4) errorQuda("Unsupported laplace direction %d (must be 3 or 4)", dir);

      for (int i = 0; i < 4; i++) {
        t0_face_offset[i] = is_t0_kernel ? (int)(this->dc.face_XYZ[i]) / 2 : 0;
        face_size[i] = 3 * this->dc.ghostFaceCB[i]; // 3=Nface
        t0_face_size[i] = (is_t0_kernel && i < 3) ? face_size[i] / in.X(3) : face_size[i];
      }

      // partial replication of dslash::setFusedParam()
      int prev = -1;
      for (int i = 0; i < 4; i++) {
        threadDimMapLower_t0[i] = 0;
        threadDimMapUpper_t0[i] = 0;
        if (!(this->commDim[i])) continue;
        threadDimMapLower_t0[i] = (prev >= 0 ? threadDimMapUpper_t0[prev] : 0);
        threadDimMapUpper_t0[i] = threadDimMapLower_t0[i] + 2 * t0_face_size[i];
        prev = i;
      }
    }
  };

  /**
     Applies the off-diagonal part of the covariant derivative operator

     @param[out] out The out result field
     @param[in,out] arg Parameter struct
     @param[in] coord Site coordinate struct
     @param[in] parity The site parity
     @param[in] thread_dim Which dimension this thread corresponds to (fused exterior only)

  */
  template <int nParity, KernelType kernel_type, int dir, typename Coord, typename Arg, typename Vector>
  __device__ __host__ inline void applyStaggeredQSmear(Vector &out, Arg &arg, Coord &coord, int parity, int,
                                                       int thread_dim, bool &active, int src_idx)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = (arg.nParity == 2) ? parity : 0;

#pragma unroll
    for (int d = 0; d < Arg::nDim; d++) { // loop over dimension
      if (d != dir) {
        {
          // Forward gather - compute fwd offset for vector fetch
          const bool ghost
            = (coord[d] + 2 >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg); // 1=>2

          if (doHalo<kernel_type>(d) && ghost) { //?

            const int ghost_idx
              = ghostFaceIndexStaggered<1>(coord, arg.dim, d, 2); // check nFace=2, requires improved staggered fields
            const Link U = arg.U(d, coord.x_cb, parity);
            const Vector in
              = arg.halo.Ghost(d, 1, ghost_idx + src_idx * arg.nFace * arg.dc.ghostFaceCB[d], their_spinor_parity); //?

            out = mv_add(U, in, out);

          } else if (doBulk<kernel_type>() && !ghost) { // doBulk
            const int _2hop_fwd_idx = linkIndexP2(coord, arg.dim, d);
            const Vector in_2hop = arg.in[src_idx](_2hop_fwd_idx, their_spinor_parity);
            const Link U_2link = arg.U(d, coord.x_cb, parity);
            out = mv_add(U_2link, in_2hop, out);
          }
        }
        {
          // Backward gather - compute back offset for spinor and gauge fetch
          const bool ghost = (coord[d] - 2 < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg); // 1=>2

          if (doHalo<kernel_type>(d) && ghost) {

            // when updating replace arg.nFace with 1 here
            const int ghost_idx
              = ghostFaceIndexStaggered<0>(coord, arg.dim, d, 2); // check nFace=2, requires improved staggered field
            const Link U = arg.U.Ghost(d, ghost_idx, parity);
            const Vector in
              = arg.halo.Ghost(d, 0, ghost_idx + src_idx * arg.nFace * arg.dc.ghostFaceCB[d], their_spinor_parity);

            out = mv_add(conj(U), in, out);

          } else if (doBulk<kernel_type>() && !ghost) { //?

            const int _2hop_back_idx = linkIndexM2(coord, arg.dim, d);
            const int _2hop_gauge_idx = _2hop_back_idx;

            const Link U_2link = arg.U(d, _2hop_gauge_idx, parity);
            const Vector in_2hop = arg.in[src_idx](_2hop_back_idx, their_spinor_parity);
            out = mv_add(conj(U_2link), in_2hop, out);
          }
        }
      }
    }
  }

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct staggered_qsmear : dslash_default {

    const Arg &arg;
    constexpr staggered_qsmear(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_idx, int parity) // Kernel3D_impl
    {
      using real = typename mapper<typename Arg::Float>::type;
      using Vector = ColorSpinor<real, Arg::nColor, 1>;

      // is thread active (non-trival for fused kernel only)
      bool active = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true;

      // which dimension is thread working on (fused kernel only)
      int thread_dim;

      if (arg.is_t0_kernel) {
        if (arg.t0 < 0) return;

        if constexpr (mykernel_type == INTERIOR_KERNEL) {
          idx += arg.t0 * arg.t0_offset;
        } else if constexpr (mykernel_type != EXTERIOR_KERNEL_ALL) {
          if (idx >= arg.t0_face_size[mykernel_type])
            idx += arg.face_size[mykernel_type] - arg.t0_face_size[mykernel_type];
          idx += arg.t0 * arg.t0_face_offset[mykernel_type];
        } else // if( mykernel_type == EXTERIOR_KERNEL_ALL )
        {
          for (int i = 0; i < 4; i++) {
            if (idx < arg.threadDimMapUpper_t0[i]) {
              idx -= arg.threadDimMapLower_t0[i];
              if (idx >= arg.t0_face_size[i]) idx += arg.face_size[i] - arg.t0_face_size[i];
              idx += arg.t0 * arg.t0_face_offset[i];
              idx += arg.threadDimMapLower[i];
              break;
            }
          }
        }
      }

      auto coord = getCoords<QUDA_4D_PC, mykernel_type, Arg, 3>(arg, idx, 0, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;
      // We instantiate two kernel types:
      // case 4 is an operator in all x,y,z,t dimensions
      // case 3 is a spatial operator only, the t dimension is omitted.
      switch (arg.dir) {
      case 3:
        applyStaggeredQSmear<nParity, mykernel_type, 3>(out, arg, coord, parity, idx, thread_dim, active, src_idx);
        break;
      case 4:
      default:
        applyStaggeredQSmear<nParity, mykernel_type, -1>(out, arg, coord, parity, idx, thread_dim, active, src_idx);
        break;
      }

      if (mykernel_type != INTERIOR_KERNEL) {
        Vector x = arg.out[src_idx](coord.x_cb, my_spinor_parity);
        out = x + out;
      }

      if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](coord.x_cb, my_spinor_parity) = out;
    }
  };

} // namespace quda
