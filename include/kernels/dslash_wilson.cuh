#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <kernels/dslash_pack.cuh> // for the packing kernel
#include <kernels/spinor_reweight.cuh>

namespace quda
{

  /**
     @brief Parameter structure for driving the Wilson operator
   */
  template <typename Float, int nColor_, int nDim, QudaReconstructType reconstruct_>
  struct WilsonArg : DslashArg<Float, nDim> {
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type F;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    typedef typename gauge_mapper<Float, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost>::type G;

    typedef typename mapper<Float>::type real;

    static constexpr bool distance = false;

    F out;        /** output vector field */
    const F in;   /** input vector field */
    const F in_pack; /** input vector field used in packing to be able to independently resetGhost */
    const F x;    /** input vector when doing xpay */
    const G U;    /** the gauge field */
    const real a; /** xpay scale factor - can be -kappa or -kappa^2 */

    WilsonArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
              const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
      DslashArg<Float, nDim>(out, in, U, x, parity, dagger, a != 0.0 ? true : false, 1, spin_project, comm_override),
      out(out),
      in(in),
      in_pack(in),
      x(x),
      U(U),
      a(a)
    {
    }
  };

  /**
     @brief Parameter structure for driving the Wilson operator
   */
  template <typename Float, int nColor_, int nDim, QudaReconstructType reconstruct_>
  struct WilsonDistanceArg : WilsonArg<Float, nColor_, nDim, reconstruct_> {
    static constexpr bool distance = true;

    typedef typename mapper<Float>::type real;

    const real alpha;
    const int t0;
    const int comm_dim_3;
    const int comm_coord_3;

    WilsonDistanceArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double alpha,
              int t0, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
      WilsonArg<Float, nColor_, nDim, reconstruct_>(out, in, U, a, x, parity, dagger, comm_override),
      alpha(alpha),
      t0(t0),
      comm_dim_3(comm_dim(3)),
      comm_coord_3(comm_coord(3))
    {
    }
  };

  /**
     @brief Applies the off-diagonal part of the Wilson operator

     @param[out] out The out result field
     @param[in,out] arg Parameter struct
     @param[in] coord Site coordinate struct
     @param[in] s The fifth-dimension index
     @param[in] parity Site parity
     @param[in] idx Thread index (equal to face index for exterior kernels)
     @param[in] thread_dim Which dimension this thread corresponds to (fused exterior only)
  */
  template <int nParity, bool dagger, KernelType kernel_type, typename Coord, typename Arg, typename Vector>
  __device__ __host__ inline std::enable_if_t<!Arg::distance, void> applyWilson(Vector &out, const Arg &arg, Coord &coord, int parity, int idx, int thread_dim, bool &active)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;

    // parity for gauge field - include residual parity from 5-d => 4-d checkerboarding
    const int gauge_parity = (Arg::nDim == 5 ? (coord.x_cb / arg.dc.volume_4d_cb + parity) % 2 : parity);

#pragma unroll
    for (int d = 0; d < 4; d++) { // loop over dimension - 4 and not nDim since this is used for DWF as well
      {                           // Forward gather - compute fwd offset for vector fetch
        const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? coord.x_cb % arg.dc.volume_4d_cb : coord.x_cb);
        constexpr int proj_dir = dagger ? +1 : -1;

        const bool ghost
            = (coord[d] + arg.nFace >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<1, Arg::nDim>(coord, arg.dim, d, arg.nFace) : idx;

          Link U = arg.U(d, gauge_idx, gauge_parity);
          HalfVector in = arg.in.Ghost(d, 1, ghost_idx + coord.s * arg.dc.ghostFaceCB[d], their_spinor_parity);

          out += (U * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, gauge_parity);
          Vector in = arg.in(fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);

          out += (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }

      { // Backward gather - compute back offset for spinor and gauge fetch
        const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? back_idx % arg.dc.volume_4d_cb : back_idx);
        constexpr int proj_dir = dagger ? -1 : +1;

        const bool ghost = (coord[d] - arg.nFace < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<0, Arg::nDim>(coord, arg.dim, d, arg.nFace) : idx;

          const int gauge_ghost_idx = (Arg::nDim == 5 ? ghost_idx % arg.dc.ghostFaceCB[d] : ghost_idx);
          Link U = arg.U.Ghost(d, gauge_ghost_idx, 1 - gauge_parity);
          HalfVector in = arg.in.Ghost(d, 0, ghost_idx + coord.s * arg.dc.ghostFaceCB[d], their_spinor_parity);

          out += (conj(U) * in).reconstruct(d, proj_dir);
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, 1 - gauge_parity);
          Vector in = arg.in(back_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);

          out += (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        }
      }
    } // nDim
  }

  /**
     @brief Applies the off-diagonal part of the Wilson operator

     @param[out] out The out result field
     @param[in,out] arg Parameter struct
     @param[in] coord Site coordinate struct
     @param[in] s The fifth-dimension index
     @param[in] parity Site parity
     @param[in] idx Thread index (equal to face index for exterior kernels)
     @param[in] thread_dim Which dimension this thread corresponds to (fused exterior only)
  */
  template <int nParity, bool dagger, KernelType kernel_type, typename Coord, typename Arg, typename Vector>
  __device__ __host__ inline std::enable_if_t<Arg::distance, void> applyWilson(Vector &out, const Arg &arg, Coord &coord, int parity, int idx, int thread_dim, bool &active)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;

    // parity for gauge field - include residual parity from 5-d => 4-d checkerboarding
    const int gauge_parity = (Arg::nDim == 5 ? (coord.x_cb / arg.dc.volume_4d_cb + parity) % 2 : parity);

    // values for distance preconditioning
    const real alpha = arg.alpha;
    const int t0 = arg.t0;
    const int t = arg.comm_coord_3 * arg.dim[3] + coord[3];
    const int nt = arg.comm_dim_3 * arg.dim[3];
    const real denom = genDistanceWeight<false>(alpha, t0, t, nt);
    const real ratio_fwd = genDistanceWeight<false>(alpha, t0, t + 1, nt) / denom;
    const real ratio_bwd = genDistanceWeight<false>(alpha, t0, t - 1, nt) / denom;

#pragma unroll
    for (int d = 0; d < 4; d++) { // loop over dimension - 4 and not nDim since this is used for DWF as well
      {                           // Forward gather - compute fwd offset for vector fetch
        const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? coord.x_cb % arg.dc.volume_4d_cb : coord.x_cb);
        constexpr int proj_dir = dagger ? +1 : -1;

        const bool ghost
            = (coord[d] + arg.nFace >= arg.dim[d]) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<1, Arg::nDim>(coord, arg.dim, d, arg.nFace) : idx;

          Link U = arg.U(d, gauge_idx, gauge_parity);
          HalfVector in = arg.in.Ghost(d, 1, ghost_idx + coord.s * arg.dc.ghostFaceCB[d], their_spinor_parity);

          if (d == 3) {
            out += ratio_fwd * (U * in).reconstruct(d, proj_dir);
          } else {
            out += (U * in).reconstruct(d, proj_dir);
          }
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, gauge_parity);
          Vector in = arg.in(fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);

          if (d == 3) {
            out += ratio_fwd * (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
          } else {
            out += (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
          }
        }
      }

      { // Backward gather - compute back offset for spinor and gauge fetch
        const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? back_idx % arg.dc.volume_4d_cb : back_idx);
        constexpr int proj_dir = dagger ? -1 : +1;

        const bool ghost = (coord[d] - arg.nFace < 0) && isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<0, Arg::nDim>(coord, arg.dim, d, arg.nFace) : idx;

          const int gauge_ghost_idx = (Arg::nDim == 5 ? ghost_idx % arg.dc.ghostFaceCB[d] : ghost_idx);
          Link U = arg.U.Ghost(d, gauge_ghost_idx, 1 - gauge_parity);
          HalfVector in = arg.in.Ghost(d, 0, ghost_idx + coord.s * arg.dc.ghostFaceCB[d], their_spinor_parity);

          if (d == 3) {
            out += ratio_bwd * (conj(U) * in).reconstruct(d, proj_dir);
          } else {
            out += (conj(U) * in).reconstruct(d, proj_dir);
          }
        } else if (doBulk<kernel_type>() && !ghost) {

          Link U = arg.U(d, gauge_idx, 1 - gauge_parity);
          Vector in = arg.in(back_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);

          if (d == 3) {
            out += ratio_bwd * (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
          } else {
            out += (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
          }
        }
      }
    } // nDim
  }

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg> struct wilson : dslash_default {

    const Arg &arg;
    constexpr wilson(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    // out(x) = M*in = (-D + m) * in(x-mu)
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      
      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, 0, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;
      applyWilson<nParity, dagger, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active);

      int xs = coord.x_cb + coord.s * arg.dc.volume_4d_cb;
      if (xpay && mykernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x(xs, my_spinor_parity);
        out = x + arg.a * out;
      } else if (mykernel_type != INTERIOR_KERNEL && active) {
        Vector x = arg.out(xs, my_spinor_parity);
        out = x + (xpay ? arg.a * out : out);
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(xs, my_spinor_parity) = out;
    }
  };

} // namespace quda
