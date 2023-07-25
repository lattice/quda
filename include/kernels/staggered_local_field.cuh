#pragma once

#include <dslash_helper.cuh>
#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>

namespace quda
{
  /**
     @brief Parameter structure for driving the local Staggered Dslash operator
  */
  template <typename Float, int nColor_, QudaReconstructType reconstruct_u_,
            QudaReconstructType reconstruct_l_, bool improved_, QudaStaggeredPhase phase_ = QUDA_STAGGERED_PHASE_MILC>
  struct StaggeredLocalFieldArg : kernel_param<> {
    typedef typename mapper<Float>::type real;

    static constexpr int nDim = 4;

    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType reconstruct_u = reconstruct_u_;
    static constexpr QudaReconstructType reconstruct_l = reconstruct_l_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    static constexpr bool use_inphase = improved_ ? false : true;
    static constexpr QudaStaggeredPhase phase = phase_;
    using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO, 18>::type;
    using GU = typename gauge_mapper<Float, reconstruct_u, 18, phase, gauge_direct_load, ghost, use_inphase>::type;
    using GL =
        typename gauge_mapper<Float, reconstruct_l, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost, use_inphase>::type;

    G Ulocal;   /** the output local gauge field */
    G Llocal;   /** the output local long gauge field */
    const GU U; /** the input gauge field */
    const GL L; /** the input long gauge field */

    int_fastdiv dim[nDim];    /** Dimensions of fine grid */
    bool is_partitioned[nDim]; /** Whether or not a dimension is partitioned */

    const real tboundary; /** temporal boundary condition */
    const bool is_first_time_slice; /** are we on the first (global) time slice */
    const bool is_last_time_slice; /** are we on the last (global) time slice */
    static constexpr bool improved = improved_; /** whether or not we're applying the improved operator */

    StaggeredLocalFieldArg(GaugeField &Ulocal, GaugeField &Llocal, const GaugeField &U, const GaugeField &L) :
      kernel_param(dim3(Ulocal.VolumeCB(), 2, 1)),
      Ulocal(Ulocal),
      Llocal(Llocal),
      U(U),
      L(L),
      tboundary(U.TBoundary()),
      is_first_time_slice(comm_coord(3) == 0 ? true : false),
      is_last_time_slice(comm_coord(3) == comm_dim(3) - 1 ? true : false)
    {
      if (Ulocal.Gauge_p() == U.Gauge_p() || Llocal.Gauge_p() == L.Gauge_p()) errorQuda("Aliasing pointers");
      checkPrecision(Ulocal, Llocal, U, L); // check all precisions match
      checkLocation(Ulocal, Llocal, U, L);  // check all locations match
      if (Ulocal.Reconstruct() != QUDA_RECONSTRUCT_NO || Llocal.Reconstruct() != QUDA_RECONSTRUCT_NO)
        errorQuda("Unexpected reconstructs %d %d for local U, L fields\n", Ulocal.Reconstruct(), Llocal.Reconstruct());

      for (int i = 0; i < nDim; i++) {
        dim[i] = U.X()[i];
        is_partitioned[i] = comm_dim_partitioned(i) ? true : false;
      }
    }
  };

  template <typename Arg>
  struct ComputeStaggeredLocalField {

    static constexpr int nDim = 4;
    using real = typename Arg::real;
    using Link = Matrix<complex<real>, Arg::nColor>;

    const Arg &arg;
    constexpr ComputeStaggeredLocalField(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    /**
       @brief Driver to apply the full local dslash
       @param[in] x_cb input coordinate
       @param[in] parity site parity
    */
    __device__ __host__ void operator()(int x_cb, int parity) {
      constexpr int nDim = 4;
      Coord<nDim> coord;

      // Get coordinates
      coord.x_cb = x_cb;
      coord.X = getCoords(coord, x_cb, arg.dim, parity);
      coord.s = 0;

      // helper lambda for getting U
      auto getU = [&] (int d_, int x_cb_, int parity_, int sign_) -> Link {
        return arg.improved ? arg.U(d_, x_cb_, parity_) : arg.U(d_, x_cb_, parity_, StaggeredPhase(coord, d_, sign_, arg));
      };

      // helper lambda for getting U from ghost zone
      auto getUGhost = [=] (int d_, int ghost_idx_, int parity_, int sign_) -> Link {
        return arg.improved ? arg.U.Ghost(d_, ghost_idx_, parity_) : arg.U.Ghost(d_, ghost_idx_, parity_, StaggeredPhase(coord, d_, sign_, arg));
      };

#pragma unroll
      for (int d = 0; d < nDim; d++) {

        // standard - forward direction
        if (!arg.is_partitioned[d] || (coord[d] + 1) < arg.dim[d])
        {
          const Link U = getU(d, x_cb, parity, +1);
          arg.Ulocal(d, x_cb, parity) = U;
        } else {
          // we need to construct a custom "clover"
          // Load the U link once for the "self" contribution
          const Link U = getU(d, x_cb, parity, +1);
          Link C = - U * conj(U);

          if constexpr (Arg::improved) {
            const Link L = arg.L(d, x_cb, parity);
            C = C - L * conj(L);

            // We can also compute the -2 contribution
            const int gauge_idx = linkIndexM2(coord, arg.dim, d);
            const Link Lm2 = arg.L(d, gauge_idx, parity);
            Link Cm2 = - U * conj(Lm2);
            arg.Llocal(d, x_cb, parity) = Cm2;
          }

          // store
          arg.Ulocal(d, x_cb, parity) = C;
        }

        // improved - forward direction
        if constexpr (Arg::improved) {
          if (!arg.is_partitioned[d] || (coord[d] + 3) < arg.dim[d]) {
            const Link L = arg.L(d, coord.x_cb, parity);
            arg.Llocal(d, x_cb, parity) = L;
          } else if ((coord[d] + 2 == arg.dim[d]) || coord[d] + 3 == arg.dim[d]) {
            // Load the L link once for the "self" contribution, take product, store
            const Link L = arg.L(d, coord.x_cb, parity);
            Link C = - L * conj(L);
            arg.Llocal(d, x_cb, parity) = C;
          }
        }

        // standard - backward direction
        // we only need to override the backwards link (in the ghost)
        // if we're on the back border
        if (arg.is_partitioned[d] && (coord[d] - 1) < 0) {
          // Load the U link once for the "self" contribution
          const int ghost_idx = ghostFaceIndexStaggered<0>(coord, arg.dim, d, 1);
          const Link U = getUGhost(d, ghost_idx, 1 - parity, -1);
          Link C = - conj(U) * U;

          if constexpr (Arg::improved) {
            const Link L = arg.L.Ghost(d, ghost_idx, 1 - parity);
            C = C - conj(L) * L;

            // we can also compute the +2 contribution
            // need some special sauce to get the index for the ghost L
            // we only need a few specific components of Coord<nDim>
            int coord_copy[nDim];
#pragma unroll
            for (int d_ = 0; d_ < nDim; d_++)
              coord_copy[d_] = coord[d_];
            coord_copy[d] = 2;
            const int ghost_p2_idx = ghostFaceIndexStaggered<0>(coord_copy, arg.dim, d, 1);
            const Link Lp2 = arg.L.Ghost(d, ghost_p2_idx, 1 - parity);
            Link Cp2 = - conj(U) * Lp2;
            arg.Llocal.Ghost(d, ghost_idx, 1 - parity) = Cp2;
          }

          // store
          arg.Ulocal.Ghost(d, ghost_idx, 1 - parity) = C;
        }

        // improved - backward direction
        // we only need to override the backwards link (in the ghost)
        // if we're on the back border
        if constexpr (Arg::improved) {
          if (arg.is_partitioned[d] && ((coord[d] == 1) || (coord[d] == 2))) {
            // Load the L link once for the "self" contribution, take product, store
            const int ghost_idx = ghostFaceIndexStaggered<0>(coord, arg.dim, d, 1);
            const Link L = arg.L.Ghost(d, ghost_idx, 1 - parity);
            Link C = - conj(L) * L;

            arg.Llocal.Ghost(d, ghost_idx, 1 - parity) = C;
          }
        }

      } // dimension

    }
  };

} // namespace quda


