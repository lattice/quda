#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <index_helper.cuh>
#include <kernels/dslash_pack.cuh> // for the packing kernel
#include <kernels/spinor_reweight.cuh>

namespace quda
{

  /**
     @brief Parameter structure for driving the Wilson operator
   */
  template <typename Float, int nColor_, int nDim, typename DDArg, QudaReconstructType reconstruct_, bool distance_pc_ = false>
  struct WilsonArg : DslashArg<Float, nDim, DDArg> {
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, true>::type F;

    using Ghost = typename colorspinor::GhostNOrder<Float, nSpin, nColor, spin_project, spinor_direct_load, false>;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool distance_pc = distance_pc_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    template <bool shifted>
    using G = typename gauge_mapper<Float, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost, false,
                                    QUDA_NATIVE_GAUGE_ORDER, shifted>::type;

    typedef typename mapper<Float>::type real;

    F out[MAX_MULTI_RHS]; /** output vector field set */
    F in[MAX_MULTI_RHS];  /** input vector field set */
    F x[MAX_MULTI_RHS];   /** input vector set when doing xpay */
    Ghost halo_pack;
    Ghost halo;
    mutable G<false> U;    /** the gauge field */
    mutable G<true> Uback; /** the backwards gauge field */
    const real a; /** xpay scale factor - can be -kappa or -kappa^2 */
    /** parameters for distance preconditioning */
    const real alpha0;
    const int t0;
    static constexpr int prefetch_distance = QUDA_DSLASH_PREFETCH_DISTANCE_WILSON;

    WilsonArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
              const GaugeField &U, double a, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
              const int *comm_override, double alpha0 = 0.0, int t0 = -1) :
      DslashArg<Float, nDim, DDArg>(out, in, halo, U, x, parity, dagger, a != 0.0 ? true : false, spin_project,
                                    comm_override),
      halo_pack(halo),
      halo(halo),
      U(U),
      Uback(dslash_double_store() ? U.shift(1) : U),
      a(a),
      alpha0(alpha0),
      t0(t0)
    {
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->in[i] = in[i];
        this->x[i] = x[i];
      }
    }
  };

  /**
     @tparam distance The distance away we are prefetching
     @param[in] dim The dimension we are presently working on
     @param[in] dir The direction we are presently working on (1 = forwards, 0 = backwards)
     @param[in] coord Coordinates that we are working on
     @param[in] parity Partiry that we are working on
     @param[in] arg Paramter struct
  */
  template <class coord_t, class Arg>
  __device__ __host__ void prefetch(int dim, int dir, const coord_t &coord, int parity, const Arg &arg)
  {
    if constexpr (Arg::prefetch_distance == 0) return;

    int step = 2 * dim + dir + Arg::prefetch_distance;
    if (step >= 8) return;

    int dim2 = step / 2;

    // if using a bulk prefetch we need to use block's first coordinate
    auto x_cb = dslash_prefetch_tma() ? coord.x_cb_0 : coord.x_cb;
    x_cb = (Arg::nDim == 5 ? x_cb % arg.dc.volume_4d_cb : x_cb);

    switch (step % 2) {
    case 0: arg.U.template prefetch<Arg::prefetch_type>(x_cb, dim2, parity); break;
    case 1:
      if constexpr (dslash_double_store()) {
        arg.Uback.template prefetch<Arg::prefetch_type>(x_cb, dim2, parity);
      } else {
        int idx = getNeighborIndexCB(coord, dim2, -1, arg.dc);
        arg.U.template prefetch<Arg::prefetch_type>(Arg::nDim == 5 ? idx % arg.dc.volume_4d_cb : idx, dim2, 1 - parity);
      }
      break;
    }
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
  template <bool dagger, KernelType kernel_type, typename Coord, typename Arg, typename Vector>
  __device__ __host__ inline void applyWilson(Vector &out, const Arg &arg, Coord &coord, int parity, int idx,
                                              int thread_dim, bool &active, int src_idx)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = arg.nParity == 2 ? 1 - parity : 0;

    // parity for gauge field - include residual parity from 5-d => 4-d checkerboarding
    const int gauge_parity = (Arg::nDim == 5 ? (coord.x_cb / arg.dc.volume_4d_cb + parity) % 2 : parity);

    const int t = coord.gx[3];
    const int nt = arg.globalDim3;
    real fwd_coeff_3
      = Arg::distance_pc ? distanceWeight(arg, t + 1, nt) / distanceWeight(arg, t, nt) : static_cast<real>(1.0);
    real bwd_coeff_3
      = Arg::distance_pc ? distanceWeight(arg, t - 1, nt) / distanceWeight(arg, t, nt) : static_cast<real>(1.0);

#pragma unroll
    for (int d = 0; d < 4; d++) { // loop over dimension - 4 and not nDim since this is used for DWF as well
      // Forward gather - compute fwd offset for vector fetch
      if (arg.dd_in.doHopping(coord, d, +1)) {
        const real fwd_coeff = (d < 3) ? 1.0 : fwd_coeff_3;
        const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? coord.x_cb % arg.dc.volume_4d_cb : coord.x_cb);
        constexpr int proj_dir = dagger ? +1 : -1;

        const bool ghost = coord.in_boundary[1][d] & isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<1, Arg::nDim>(coord, arg.dc.X, d, arg.nFace) :
            idx;

          Link U = arg.U(d, gauge_idx, gauge_parity);
          HalfVector in = arg.halo.Ghost(d, 1, ghost_idx + (src_idx * arg.Ls + coord.s) * arg.dc.ghostFaceCB[d],
                                         their_spinor_parity);

          out += fwd_coeff * (U * in).reconstruct(d, proj_dir);
        }

        if constexpr (doBulk<kernel_type>()) {
          if (!ghost) {
            Link U = arg.U(d, gauge_idx, gauge_parity);
            Vector in = arg.in[src_idx](fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
            out += fwd_coeff * (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
          }

          prefetch(d, 0, coord, parity, arg); // prefetch the gauge link Arg::prefetch_distance ahead
        }
      }

      // Backward gather - compute back offset for spinor and gauge fetch
      if (arg.dd_in.doHopping(coord, d, -1)) {
        const real bwd_coeff = (d < 3) ? 1.0 : bwd_coeff_3;
        const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
        int gauge_idx = dslash_double_store() ? coord.x_cb : back_idx;
        if constexpr (Arg::nDim == 5) gauge_idx = gauge_idx % arg.dc.volume_4d_cb;
        constexpr int proj_dir = dagger ? -1 : +1;

        const bool ghost = coord.in_boundary[0][d] & isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<0, Arg::nDim>(coord, arg.dc.X, d, arg.nFace) :
            idx;

          const int gauge_ghost_idx = (Arg::nDim == 5 ? ghost_idx % arg.dc.ghostFaceCB[d] : ghost_idx);
          Link U = dslash_double_store() ? static_cast<const Link&>(arg.Uback(d, gauge_idx, gauge_parity)) :
                                           static_cast<const Link &>(arg.U.Ghost(d, gauge_ghost_idx, 1 - gauge_parity));
          HalfVector in = arg.halo.Ghost(d, 0, ghost_idx + (src_idx * arg.Ls + coord.s) * arg.dc.ghostFaceCB[d],
                                         their_spinor_parity);

          out += bwd_coeff * (conj(U) * in).reconstruct(d, proj_dir);
        }

        if constexpr (doBulk<kernel_type>()) {
          if (!ghost) {
            Link U = dslash_double_store() ? static_cast<const Link &>(arg.Uback(d, gauge_idx, gauge_parity)) :
                                             static_cast<const Link &>(arg.U(d, gauge_idx, 1 - gauge_parity));
            Vector in = arg.in[src_idx](back_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
            out += bwd_coeff * (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
          }

          prefetch(d, 1, coord, parity, arg); // prefetch the gauge link Arg::prefetch_distance ahead
        }
      }
    } // nDim
  }

  template <bool dagger, bool xpay, KernelType kernel_type, typename Arg> struct wilson : dslash_default {

    const Arg &arg;
    template <typename Ftor> constexpr wilson(const Ftor &ftor) : dslash_default {ftor.block_idx}, arg(ftor.arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    // out(x) = M*in = (-D + m) * in(x-mu)
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_idx, int parity) const
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)

      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, 0, parity, thread_dim, block_idx.x);

      const int my_spinor_parity = arg.nParity == 2 ? parity : 0;
      int xs = coord.x_cb + coord.s * arg.dc.volume_4d_cb;
      Vector out;
      if (arg.dd_out.isZero(coord)) {
        if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](xs, my_spinor_parity) = out;
        return;
      }

      applyWilson<dagger, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active, src_idx);

      if (xpay && mykernel_type == INTERIOR_KERNEL && arg.dd_x.isZero(coord)) {
        out = arg.a * out;
      } else if (xpay && mykernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x[src_idx](xs, my_spinor_parity);
        out = x + arg.a * out;
      } else if (mykernel_type != INTERIOR_KERNEL && active) {
        Vector x = arg.out[src_idx](xs, my_spinor_parity);
        out = x + (xpay ? arg.a * out : out);
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](xs, my_spinor_parity) = out;
    }
  };

} // namespace quda
