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
    static constexpr bool work_steal_functor
      = true; // set true to drive request() from prefetch; false = regular work-steal (kernel calls request/complete)

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
  template <class coord_t, class Arg, KernelType kernel_type = INTERIOR_KERNEL>
  __device__ __host__ void
  prefetch(int dim, int dir, const coord_t &coord, int parity, const Arg &arg, work_steal<3> *robber = nullptr,
           const coord_t *next_coord_ptr = nullptr, bool next_is_interior = false, int next_parity_val = 0,
           bool *p_next_steal_valid = nullptr, coord_t *p_next_steal_coord = nullptr, unsigned int block_idx_x = 0)
  {
    if constexpr (Arg::prefetch_distance == 0) return;

    constexpr int pipeline_length
      = 8; // 4 dims * 2 directions; request at pipeline_length-1, complete at pipeline_length
    int step = 2 * dim + dir + Arg::prefetch_distance;
    if constexpr (Arg::work_steal_functor) {
      // Last interior block never participates in request/complete to avoid mbarrier hang when block is ragged
      const bool skip_work_steal = (kernel_type == INTERIOR_KERNEL || kernel_type == UBER_KERNEL)
        && (block_idx_x == target::grid_dim().x - 1 - arg.exterior_blocks);
      if (step == pipeline_length - 1 && robber && !skip_work_steal) robber->request();
      if (step == pipeline_length && robber && p_next_steal_coord && !skip_work_steal) {
        bool success = robber->complete();
        if (success) {
          dim3 nb = robber->get_block_idx();
          if (p_next_steal_valid) *p_next_steal_valid = true;
          int idx_nb = (nb.x - arg.pack_blocks) * target::block_dim().x;
          int dim_nb = 0;
          const int parity_nb = dslash_prefetch_tma() ? nb.z : (nb.z * target::block_dim().z + target::thread_idx().z);
          *p_next_steal_coord = getCoords<QUDA_4D_PC, INTERIOR_KERNEL>(arg, idx_nb, 0, parity_nb, dim_nb, nb.x);
        }
      }
    }

    const bool have_steal_block = (step >= pipeline_length) && p_next_steal_valid && *p_next_steal_valid && robber
      && (robber->next_block_idx().x >= (unsigned)arg.pack_blocks)
      && (arg.exterior_blocks == 0
          || robber->next_block_idx().x < (unsigned)(target::grid_dim().x - arg.exterior_blocks));
    const bool do_next_block = have_steal_block && p_next_steal_coord;
    if (step >= pipeline_length && !do_next_block) return;

    const int step_mod = do_next_block ? (step % pipeline_length) : step;
    const coord_t &prefetch_coord = do_next_block ? *p_next_steal_coord : coord;
    const int prefetch_parity = do_next_block ? (arg.nParity == 1 ? arg.parity : robber->next_block_idx().z) : parity;

    int dim2 = step_mod / 2;

    // if using a bulk prefetch we need to use block's first coordinate
    auto x_cb = dslash_prefetch_tma() ? prefetch_coord.x_cb_0 : prefetch_coord.x_cb;
    x_cb = (Arg::nDim == 5 ? x_cb % arg.dc.volume_4d_cb : x_cb);

    switch (step_mod % 2) {
    case 0: arg.U.template prefetch<Arg::prefetch_type>(x_cb, dim2, prefetch_parity); break;
    case 1:
      if constexpr (dslash_double_store()) {
        arg.Uback.template prefetch<Arg::prefetch_type>(x_cb, dim2, prefetch_parity);
      } else {
        int idx = getNeighborIndexCB(prefetch_coord, dim2, -1, arg.dc);
        arg.U.template prefetch<Arg::prefetch_type>(Arg::nDim == 5 ? idx % arg.dc.volume_4d_cb : idx, dim2,
                                                    1 - prefetch_parity);
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
                                              int thread_dim, bool &active, int src_idx, work_steal<3> *robber = nullptr,
                                              const Coord *next_coord_ptr = nullptr, bool next_is_interior = false,
                                              int next_parity_val = 0, unsigned int block_idx_x = 0)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = arg.nParity == 2 ? 1 - parity : 0;
    bool next_steal_valid = false;
    Coord next_steal_coord;

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

          prefetch<Coord, Arg, kernel_type>(d, 0, coord, parity, arg, robber, next_coord_ptr, next_is_interior,
                                            next_parity_val, &next_steal_valid, &next_steal_coord,
                                            block_idx_x); // prefetch the gauge link Arg::prefetch_distance ahead
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

          prefetch<Coord, Arg, kernel_type>(d, 1, coord, parity, arg, robber, next_coord_ptr, next_is_interior,
                                            next_parity_val, &next_steal_valid, &next_steal_coord,
                                            block_idx_x); // prefetch the gauge link Arg::prefetch_distance ahead
        }
      }
    } // nDim
  }

  template <bool dagger, bool xpay, KernelType kernel_type, typename Arg> struct wilson : dslash_default {

    const Arg &arg;
    work_steal<3> *robber_ptr_ = nullptr;
    template <typename Ftor>
    constexpr wilson(const Ftor &ftor) : dslash_default {ftor.block_idx}, arg(ftor.arg), robber_ptr_(ftor.robber_ptr())
    {
    }
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

      // Next-block prefetch (step >= 8) is driven entirely in prefetch at step 8 (complete/get_block_idx).
      // We do not have the next block id here when work_steal_functor is true, so pass nullptr/false/0.
      const decltype(coord) *next_coord_ptr = nullptr;
      bool next_is_interior = false;
      int next_parity_val = 0;
      applyWilson<dagger, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active, src_idx, robber_ptr_,
                                         next_coord_ptr, next_is_interior, next_parity_val, block_idx.x);

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
