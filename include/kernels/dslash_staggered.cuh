#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <kernels/dslash_pack.cuh> // forthe packing kernel

namespace quda
{

  /**
     @brief Parameter structure for driving the Staggered Dslash operator
  */
  template <typename Float, int nColor_, int nDim, typename DDArg, QudaReconstructType reconstruct_u_,
            QudaReconstructType reconstruct_l_, bool improved_, QudaStaggeredPhase phase_ = QUDA_STAGGERED_PHASE_MILC,
            int n_src_tile = MAX_MULTI_RHS_TILE>
  struct StaggeredArg : DslashArg<Float, nDim, DDArg, improved_ ? 3 : 1, n_src_tile> {
    typedef typename mapper<Float>::type real;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, true>::type;

    using Ghost = typename colorspinor::GhostNOrder<Float, nSpin, nColor, spin_project, spinor_direct_load, false>;

    static constexpr QudaReconstructType reconstruct_u = reconstruct_u_;
    static constexpr QudaReconstructType reconstruct_l = reconstruct_l_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    static constexpr bool use_inphase = improved_ ? false : true;
    static constexpr QudaStaggeredPhase phase = phase_;
    using GU = typename gauge_mapper<Float, reconstruct_u, 18, phase, gauge_direct_load, ghost, use_inphase>::type;
    using GL =
        typename gauge_mapper<Float, reconstruct_l, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost, use_inphase>::type;

    F out[MAX_MULTI_RHS];  /** output vector field */
    F in[MAX_MULTI_RHS];   /** input vector field */
    const Ghost halo_pack; /** accessor for writing the halo */
    const Ghost halo;      /** accessor for reading the halo */
    F x[MAX_MULTI_RHS];    /** input vector when doing xpay */
    const GU U; /** the gauge field */
    const GU Uback; /** the gauge field */
    const GL L; /** the long gauge field */
    const GL Lback; /** the long gauge field */

    const real a; /** xpay scale factor */
    const real tboundary; /** temporal boundary condition */
    const bool is_first_time_slice; /** are we on the first (global) time slice */
    const bool is_last_time_slice; /** are we on the last (global) time slice */
    static constexpr bool improved = improved_;
    static constexpr int prefetch_distance = QUDA_DSLASH_PREFETCH_DISTANCE_STAGGERED;

    const real dagger_scale;

    StaggeredArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                 const ColorSpinorField &halo, const GaugeField &U, const GaugeField &Uback, const GaugeField &L,
                 const GaugeField &Lback, double a, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                 const int *comm_override) :
      DslashArg < Float,
    nDim, DDArg, improved ? 3 : 1, n_src_tile
      > (out, in, halo, U, x, parity, dagger, a == 0.0 ? false : true, spin_project, comm_override),
    halo_pack(halo, improved_ ? 3 : 1), halo(halo, improved_ ? 3 : 1), U(U), Uback(Uback), L(L), Lback(Lback), a(a),
    tboundary(U.TBoundary()), is_first_time_slice(comm_coord(3) == 0 ? true : false),
    is_last_time_slice(comm_coord(3) == comm_dim(3) - 1 ? true : false),
    dagger_scale(dagger ? static_cast<real>(-1.0) : static_cast<real>(1.0))
    {
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->in[i] = in[i];
        this->x[i] = x[i];
      }
    }
  };

  /**
     @brief Prefetch the gauge field into cache.
     @param[in] dim The dimension we are presently working on
     @param[in] dir The direction we are presently working on (1 = forwards, 0 = backwards)
     @param[in] hop The hopping term we are presently working on (0 = 1 - hop, 1 = 3 - hop)
     @param[in] coord Coordinates that we are working on with hop-3 boundary conditions evaluated
     @param[in] coord1 Copy of coordinates that we are working on with hop-1 boundary conditions evaluated
     @param[in] parity Partiry that we are working on
     @param[in] arg Paramter struct
   */
  template <class coord_t, class Arg>
  __device__ __host__ void prefetch(int dim, int dir, int hop, const coord_t &coord, const coord_t &coord1, int parity,
                                    const Arg &arg)
  {
    if constexpr (arg.prefetch_distance == 0) return;

    if constexpr (arg.improved) {
      int step = 4 * dim + 2 * dir + hop + arg.prefetch_distance;
      if (step >= 16) return;

      // if using a bulk prefetch we need to use block's first coordinate
      auto x_cb = arg.prefetch_bulk ? coord.x_cb_0 : coord.x_cb;
      x_cb = (Arg::nDim == 5 ? x_cb % arg.dc.volume_4d_cb : x_cb);

      int dim2 = step / 4;
      switch (step % 4) {
      case 0: arg.U.prefetch<Arg::prefetch_bulk>(x_cb, dim2, parity); break;
      case 1: arg.L.prefetch<Arg::prefetch_bulk>(x_cb, dim2, parity); break;
#ifdef QUDA_DSLASH_DOUBLE_STORE
      case 2: arg.Uback.prefetch<Arg::prefetch_bulk>(x_cb, dim2, parity); break;
      case 3: arg.Lback.prefetch<Arg::prefetch_bulk>(x_cb, dim2, parity); break;
#else
      case 2:
        arg.U.prefetch<Arg::prefetch_bulk>(getNeighborIndexCB<1>(coord1, dim2, -1, arg.dc), dim2, 1 - parity);
        break;
      case 3:
        arg.L.prefetch<Arg::prefetch_bulk>(getNeighborIndexCB<3>(coord, dim2, -1, arg.dc), dim2, 1 - parity);
        break;
#endif
      }
    }
  }

  /**
     @brief Applies the off-diagonal part of the Staggered / Asqtad
     operator.

     @param[out] out The out result field
     @param[in] U The gauge field
     @param[in] in The input field
     @param[in] parity The site parity
     @param[in] x_cb The checkerboarded site index
  */
  template <KernelType kernel_type, int n_src_tile, typename Coord, typename Arg, typename Vector>
  __device__ __host__ inline void applyStaggered(array<Vector, n_src_tile> &out, const Arg &arg, Coord &coord,
                                                 int parity, int, int thread_dim, bool &active, int src_idx)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = (arg.nParity == 2) ? 1 - parity : 0;

    Coord coord1 = coord;
    if constexpr (arg.improved) { // need to compute 1-hop in_boundary
#pragma unroll
      for (int d = 0; d < 4; d++) {
        coord1.in_boundary[1][d] = -(coord[d] + 1 >= arg.dc.X[d]);
        coord1.in_boundary[0][d] = -(coord[d] - 1 < 0);
      }
    }

#pragma unroll
    for (int d = 0; d < 4; d++) { // loop over dimension

      // standard - forward direction
      if (arg.dd_in.doHopping(coord, d, +1)) {
        const bool ghost = coord1.in_boundary[1][d] & isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          const int ghost_idx = ghostFaceIndexStaggered<1>(coord, arg.dc.X, d, 1);
          const Link U = arg.improved ? arg.U(d, coord.x_cb, parity) : arg.U(d, coord.x_cb, parity, StaggeredPhase(coord, d, +1, arg));
#pragma unroll
          for (auto s = 0; s < n_src_tile; s++) {
            Vector in = arg.halo.Ghost(d, 1, ghost_idx + (src_idx + s) * arg.dc.ghostFaceCB[d], their_spinor_parity);
            out[s] = mv_add(U, in, out[s]);
          }
        }

        if constexpr (doBulk<kernel_type>()) {
          if (!ghost) {
            const int fwd_idx = getNeighborIndexCB<1>(coord1, d, 1, arg.dc);
            const Link U = arg.improved ? arg.U(d, coord.x_cb, parity) :
                                          arg.U(d, coord.x_cb, parity, StaggeredPhase(coord, d, +1, arg));
#pragma unroll
            for (auto s = 0; s < n_src_tile; s++) {
              Vector in = arg.in[src_idx + s](fwd_idx, their_spinor_parity);
              out[s] = mv_add(U, in, out[s]);
            }
          }
          prefetch(d, 0, 0, coord, coord1, parity, arg);
        }
      }

      // improved - forward direction
      if (arg.improved && arg.dd_in.doHopping(coord, d, +3)) {
        const bool ghost = coord.in_boundary[1][d] & isActive<kernel_type>(active, thread_dim, d, coord, arg);
        if (doHalo<kernel_type>(d) && ghost) {
          const int ghost_idx = ghostFaceIndexStaggered<1>(coord, arg.dc.X, d, arg.nFace);
          const Link L = arg.L(d, coord.x_cb, parity);
#pragma unroll
          for (auto s = 0; s < n_src_tile; s++) {
            const Vector in
              = arg.halo.Ghost(d, 1, ghost_idx + (src_idx + s) * arg.dc.ghostFaceCB[d], their_spinor_parity);
            out[s] = mv_add(L, in, out[s]);
          }
        }

        if constexpr (doBulk<kernel_type>()) {
          if (!ghost) {
            const int fwd3_idx = getNeighborIndexCB<3>(coord, d, 1, arg.dc);
            const Link L = arg.L(d, coord.x_cb, parity);
#pragma unroll
            for (auto s = 0; s < n_src_tile; s++) {
              const Vector in = arg.in[src_idx + s](fwd3_idx, their_spinor_parity);
              out[s] = mv_add(L, in, out[s]);
            }
          }
          prefetch(d, 0, 1, coord, coord1, parity, arg);
        }
      }

      if (arg.dd_in.doHopping(coord, d, -1)) {
        // Backward gather - compute back offset for spinor and gauge fetch
        const bool ghost = coord1.in_boundary[1][d] & isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          const int ghost_idx2 = ghostFaceIndexStaggered<0>(coord, arg.dc.X, d, 1);
          const int ghost_idx = arg.improved ? ghostFaceIndexStaggered<0>(coord, arg.dc.X, d, 3) : ghost_idx2;
#ifdef QUDA_DSLASH_DOUBLE_STORE
          const Link U = arg.improved ? arg.Uback(d, coord.x_cb, parity) :
                                        arg.Uback(d, coord.x_cb, parity, StaggeredPhase(coord, d, -1, arg));
#else
          const Link U = arg.improved ? arg.U.Ghost(d, ghost_idx2, 1 - parity) :
            arg.U.Ghost(d, ghost_idx2, 1 - parity, StaggeredPhase(coord, d, -1, arg));
#endif
#pragma unroll
          for (auto s = 0; s < n_src_tile; s++) {
            Vector in = arg.halo.Ghost(d, 0, ghost_idx + (src_idx + s) * arg.dc.ghostFaceCB[d], their_spinor_parity);
            out[s] = mv_sub(conj(U), in, out[s]);
          }
        }

        if constexpr (doBulk<kernel_type>()) {
          if (!ghost) {
            const int back_idx = getNeighborIndexCB<1>(coord1, d, -1, arg.dc);
#ifdef QUDA_DSLASH_DOUBLE_STORE
            const Link U = arg.improved ? arg.Uback(d, coord.x_cb, parity) :
                                          arg.Uback(d, coord.x_cb, parity, StaggeredPhase(coord, d, -1, arg));
#else
            const int gauge_idx = back_idx;
            const Link U = arg.improved ? arg.U(d, gauge_idx, 1 - parity) :
                                          arg.U(d, gauge_idx, 1 - parity, StaggeredPhase(coord, d, -1, arg));
#endif
#pragma unroll
            for (auto s = 0; s < n_src_tile; s++) {
              Vector in = arg.in[src_idx + s](back_idx, their_spinor_parity);
              out[s] = mv_sub(conj(U), in, out[s]);
            }
          }
          prefetch(d, 1, 0, coord, coord1, parity, arg);
        }
      }

      // improved - backward direction
      if (arg.improved && arg.dd_in.doHopping(coord, d, -3)) {
        const bool ghost = coord.in_boundary[0][d] & isActive<kernel_type>(active, thread_dim, d, coord, arg);
        if (doHalo<kernel_type>(d) && ghost) {
          const int ghost_idx = ghostFaceIndexStaggered<0>(coord, arg.dc.X, d, 1);
#ifdef QUDA_DSLASH_DOUBLE_STORE
          const Link L = arg.Lback(d, coord.x_cb, parity);
#else
          const Link L = arg.L.Ghost(d, ghost_idx, 1 - parity);
#endif
#pragma unroll
          for (auto s = 0; s < n_src_tile; s++) {
            const Vector in
              = arg.halo.Ghost(d, 0, ghost_idx + (src_idx + s) * arg.dc.ghostFaceCB[d], their_spinor_parity);
            out[s] = mv_sub(conj(L), in, out[s]);
          }
        }

        if constexpr (doBulk<kernel_type>()) {
          if (!ghost) {
            const int back3_idx = getNeighborIndexCB<3>(coord, d, -1, arg.dc);
#ifdef QUDA_DSLASH_DOUBLE_STORE
            const Link L = arg.Lback(d, coord.x_cb, parity);
#else
            const int gauge_idx = back3_idx;
            const Link L = arg.L(d, gauge_idx, 1 - parity);
#endif
#pragma unroll
            for (auto s = 0; s < n_src_tile; s++) {
              const Vector in = arg.in[src_idx + s](back3_idx, their_spinor_parity);
              out[s] = mv_sub(conj(L), in, out[s]);
            }
          }
          prefetch(d, 1, 1, coord, coord1, parity, arg);
        }
      }
    } // nDim
  }

  // out(x) = M*in = (-D + m) * in(x-mu)
  template <bool dummy, bool xpay, KernelType kernel_type, typename Arg> struct staggered : dslash_default {

    const Arg &arg;
    template <typename Ftor> constexpr staggered(const Ftor &ftor) : arg(ftor.arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    template <KernelType mykernel_type, int n_src_tile>
    __device__ __host__ __forceinline__ void apply(int idx, int src_idx, int parity)
    {
      using real = typename mapper<typename Arg::Float>::type;
      using Vector = ColorSpinor<real, Arg::nColor, 1>;

      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, mykernel_type, Arg>(arg, idx, 0, parity, thread_dim);
      const int my_spinor_parity = arg.nParity == 2 ? parity : 0;

      array<Vector, n_src_tile> out;
      if (arg.dd_out.isZero(coord)) {
        if (mykernel_type != EXTERIOR_KERNEL_ALL || active)
#pragma unroll
          for (auto s = 0; s < n_src_tile; s++) { arg.out[src_idx + s](coord.x_cb, my_spinor_parity) = out[s]; }
        return;
      }

      applyStaggered<mykernel_type, n_src_tile>(out, arg, coord, parity, idx, thread_dim, active, src_idx);

#pragma unroll
      for (auto s = 0; s < n_src_tile; s++) out[s] *= arg.dagger_scale;

      if (xpay && mykernel_type == INTERIOR_KERNEL && arg.dd_x.isZero(coord)) {
        for (auto s = 0; s < n_src_tile; s++) { out[s] = -out[s]; }
      } else if (xpay && mykernel_type == INTERIOR_KERNEL) {
#pragma unroll
        for (auto s = 0; s < n_src_tile; s++) {
          Vector x = arg.x[src_idx + s](coord.x_cb, my_spinor_parity);
          out[s] = axpy(arg.a, x, -out[s]);
        }
      } else if (mykernel_type != INTERIOR_KERNEL) {
#pragma unroll
        for (auto s = 0; s < n_src_tile; s++) {
          Vector x = arg.out[src_idx + s](coord.x_cb, my_spinor_parity);
          out[s] = xpay ? x - out[s] : x + out[s];
        }
      }
      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) {
#pragma unroll
        for (auto s = 0; s < n_src_tile; s++) { arg.out[src_idx + s](coord.x_cb, my_spinor_parity) = out[s]; }
      }
    }

    template <KernelType mykernel_type = kernel_type, int n_src_tile = Arg::n_src_tile>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_idx_block_, int parity)
    {
      int src_idx_block = MAX_MULTI_RHS == 1 ? 0 : src_idx_block_;
      int src_idx = src_idx_block * Arg::n_src_tile;
      if (src_idx + n_src_tile <= arg.n_src) {
        apply<mykernel_type, n_src_tile>(idx, src_idx, parity);
      } else if constexpr (n_src_tile - 1 > 0) {
        operator()<mykernel_type, n_src_tile - 1>(idx, src_idx_block, parity);
      }
    }
  };

} // namespace quda
