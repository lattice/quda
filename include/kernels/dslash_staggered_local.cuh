#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>

namespace quda
{
  /**
     @brief Parameter structure for driving the local Staggered Dslash operator
  */
  template <typename Float, int nColor_, QudaReconstructType reconstruct_u_,
            QudaReconstructType reconstruct_l_, bool improved_, QudaStaggeredLocalType step_, QudaStaggeredPhase phase_ = QUDA_STAGGERED_PHASE_MILC>
  struct LocalStaggeredArg : kernel_param<> {
    typedef typename mapper<Float>::type real;

    static constexpr int nDim = 4;

    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false; // false means texture load
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load>::type;

    static constexpr QudaReconstructType reconstruct_u = reconstruct_u_;
    static constexpr QudaReconstructType reconstruct_l = reconstruct_l_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    static constexpr bool use_inphase = improved_ ? false : true;
    static constexpr QudaStaggeredPhase phase = phase_;
    using GU = typename gauge_mapper<Float, reconstruct_u, 18, phase, gauge_direct_load, ghost, use_inphase>::type;
    using GL =
        typename gauge_mapper<Float, reconstruct_l, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost, use_inphase>::type;

    F out;      /** output vector field */
    const F in; /** input vector field */
    const F x;  /** input vector when doing xpay */
    const GU U; /** the gauge field */
    const GL L; /** the long gauge field */

    int_fastdiv dim[nDim];    /** Dimensions of fine grid */
    bool is_partitioned[nDim]; /** Whether or not a dimension is partitioned */

    const real a; /** xpay scale factor */
    const int parity; /** parity we're gathering from */

    const real tboundary; /** temporal boundary condition */
    const bool is_first_time_slice; /** are we on the first (global) time slice */
    const bool is_last_time_slice; /** are we on the last (global) time slice */
    static constexpr bool improved = improved_; /** whether or not we're applying the improved operator */
    static constexpr QudaStaggeredLocalType step = step_; /** which step of the local staggered dslash we're applying */

    LocalStaggeredArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L, double a,
                 const ColorSpinorField &x, int parity) :
      kernel_param(dim3(out.VolumeCB(), 1, 1)),
      out(out),
      in(in),
      x(x),
      U(U),
      L(L),
      a(a),
      parity(parity),
      tboundary(U.TBoundary()),
      is_first_time_slice(comm_coord(3) == 0 ? true : false),
      is_last_time_slice(comm_coord(3) == comm_dim(3) - 1 ? true : false)
    {
      if (in.V() == out.V()) errorQuda("Aliasing pointers");
      checkOrder(out, in, x);        // check all orders match
      checkPrecision(out, in, x, U); // check all precisions match
      checkLocation(out, in, x, U);  // check all locations match
      if (!in.isNative() || !U.isNative())
        errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", in.FieldOrder(), U.FieldOrder());

      dim[0] = ((x.SiteSubset() == QUDA_PARITY_SITE_SUBSET) ? 2 : 1) * x.X()[0];
      for (int i = 0; i < nDim; i++) {
        if (i != 0) dim[i] = x.X()[i];
        is_partitioned[i] = comm_dim_partitioned(i) ? true : false;
      }
    }
  };

  template <typename Arg>
  struct LocalStaggeredApply {

    static constexpr int nDim = 4;
    using real = typename Arg::real;
    using Vector = ColorSpinor<real, Arg::nColor, 1>;
    using Link = Matrix<complex<real>, Arg::nColor>;

    const Arg &arg;
    constexpr LocalStaggeredApply(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    /**
       @brief Computes the contribution to a "ghost" site needed for the forwards 1-hop link on the axpy step.
       @param[in] coord Precomputed coordinate structure
       @param[in] U Fat link for self-contribution
       @param[in] in "In" vector at the output site
       @return Accumulated ColorVector for a "ghost" site
    */
    template <int d>
    __device__ __host__ Vector fatLinkForwardContribution(const Coord<nDim> &coord, const Link &U, const Vector &in) const {
      // to be updated if we implement a full-parity version
      constexpr int my_spinor_parity = 0;

      // contribution from [X - 1] (self)
      Vector accum = -conj(U) * in;

      // contribution from [X - 3]
      if constexpr (Arg::improved) {
        const int back2_idx = linkIndexM2(coord, arg.dim, d);
        const int gauge_idx = back2_idx;
        const Link L = arg.L(d, gauge_idx, arg.parity);
        const Vector in_L = arg.in(back2_idx, my_spinor_parity);
        accum = mv_add(conj(L), -in_L, accum);
      }

      return accum;
    }

    /**
       @brief Computes the contribution to a "ghost" site needed for the forwards 3-hop link on the axpy step.
       @param[in] coord Precomputed coordinate structure
       @param[in] L Long link for self-contribution
       @param[in] in "In" vector at the output site
       @return Accumulated ColorVector for a "ghost" site
    */
    template <int d>
    __device__ __host__ Vector longLinkForwardContribution(const Coord<nDim> &coord, const Link &L, const Vector &in) const {
      static_assert(Arg::improved, "Long link contribution function called for unimproved staggered operator");

      // to be updated if we implement a full-parity version
      constexpr int my_spinor_parity = 0;

      // contribution from [X - 1, X - 2, or X - 3] (self)
      Vector accum = -conj(L) * in;

      // contribution from [X - 1]
      if ((coord[d] + 3) == arg.dim[d]) {
        const int fwd2_idx = linkIndexP2(coord, arg.dim, d);
        const int gauge_idx = fwd2_idx;
        // we do not need the "StaggeredPhase" because there's a guarantee this is the improved operator
        const Link U = arg.U(d, gauge_idx, arg.parity);
        const Vector in_U = arg.in(fwd2_idx, my_spinor_parity);
        accum = mv_add(conj(U), -in_U, accum);
      }

      return accum;
    }

    /**
       @brief Computes the contribution to a "ghost" site needed for the backwards 1-hop link on the axpy step
       @param[in] coord Precomputed coordinate structure
       @param[in] U Fat link for self contribution
       @param[in] in "In" vector at the output site
       @return Accumulated ColorVector for a "ghost" site
    */
    template <int d>
    __device__ __host__ Vector fatLinkBackwardContribution(const Coord<nDim> &coord, const Link &U, const Vector &in) const {
      // to be updated if we implement a full-parity version
      constexpr int my_spinor_parity = 0;

      // contribution from [0] (self)
      Vector accum = U * in;

      // contribution from [2]
      if constexpr (Arg::improved) {
        const int fwd2_idx = linkIndexP2(coord, arg.dim, d);
        const Vector in_L = arg.in(fwd2_idx, my_spinor_parity);

        // need some special sauce to get the index for the ghost L
        // we only need a few specific components of Coord<nDim>
        int coord_copy[nDim];
#pragma unroll
        for (int d_ = 0; d_ < nDim; d_++)
          coord_copy[d_] = coord[d_];
        coord_copy[d] = 2;
        const int ghost_idx = ghostFaceIndexStaggered<0>(coord_copy, arg.dim, d, 1);
        const Link L = arg.L.Ghost(d, ghost_idx, 1 - arg.parity);

        accum = mv_add(L, in_L, accum);
      }

      return accum;
    }

    /**
       @brief Computes the self-contribution to a "ghost" site needed for the backwards 3-hop link on the axpy step
       @param[in] coord Precomputed coordinate structure
       @param[in] L Long link for self contribution
       @param[in] in "In" vector at the output site
       @return Accumulated ColorVector for a "ghost" site
    */
    template <int d>
    __device__ __host__ Vector longLinkBackwardContribution(const Coord<nDim> &coord, const Link &L, const Vector &in) const {
      static_assert(Arg::improved, "Long link contribution function called for unimproved staggered operator");

      // for if we ever do a full parity version
      constexpr int my_spinor_parity = 0;

      // contribution from self [0, 1, or 2]
      Vector accum = L * in;

      // contibution from [0]
      if (coord[d] == 2) {
        const int bak2_idx = linkIndexM2(coord, arg.dim, d);
        Vector in_U = arg.in(bak2_idx, my_spinor_parity);

        // need some special sauce to get the index for the ghost L
        // we only need a few specific components of Coord<nDim>
        auto coord_copy = coord;
        coord_copy[d] = 0;
        const int ghost_idx2 = ghostFaceIndexStaggered<0>(coord_copy, arg.dim, d, 1);
        // we do not need an extra "StaggeredPhase" argument because there's a guarantee that this is the improved operator
        const Link U = arg.U.Ghost(d, ghost_idx2, 1 - arg.parity);
        accum = mv_add(U, in_U, accum);
      }

      return accum;
    }

    /**
       @brief Applies step 1 and 2 of the local dslash, which is D_{oe} for parity even and D_{eo} for parity odd
       @param[in] coord Precomputed Coord structure
       @return Accumulated output ColorVector
    */
    __device__ __host__ Vector localDslash(const Coord<nDim> &coord) const {
      static_assert(Arg::step != QUDA_STAGGERED_LOCAL_CLOVER, "localDslash called for a clover argument struct");
      // to be updated if we implement a full-parity version
      constexpr int their_spinor_parity = 0;

      Vector out;

      // helper lambda for getting U
      auto getU = [&] (int d_, int x_cb_, int parity_, int sign_) -> Link {
        return arg.improved ? arg.U(d_, x_cb_, parity_) : arg.U(d_, x_cb_, parity_, StaggeredPhase(coord, d_, sign_, arg));
      };

#pragma unroll
      for (int d = 0; d < nDim; d++) {

        // standard - forward direction
        if (!arg.is_partitioned[d] || (coord[d] + 1) < arg.dim[d])
        {
          const int fwd_idx = linkIndexP1(coord, arg.dim, d);
          const Link U = getU(d, coord.x_cb, arg.parity, +1);
          Vector in = arg.in(fwd_idx, their_spinor_parity);
          out = mv_add(U, in, out);
        }

        // improved - forward direction
        if constexpr (arg.improved) {
          if (!arg.is_partitioned[d] || (coord[d] + 3) < arg.dim[d]) {
            const int fwd3_idx = linkIndexP3(coord, arg.dim, d);
            const Link L = arg.L(d, coord.x_cb, arg.parity);
            const Vector in = arg.in(fwd3_idx, their_spinor_parity);
            out = mv_add(L, in, out);
          }
        }

        // standard - backward direction
        if (!arg.is_partitioned[d] || (coord[d] - 1) >= 0)
        {
          const int back_idx = linkIndexM1(coord, arg.dim, d);
          const int gauge_idx = back_idx;
          const Link U = getU(d, gauge_idx, 1 - arg.parity, -1);
          Vector in = arg.in(back_idx, their_spinor_parity);
          out = mv_add(conj(U), -in, out);
        }

        // improved - backward direction
        if constexpr (arg.improved) {
          if (!arg.is_partitioned[d] || (coord[d] - 3) >= 0) {
            const int back3_idx = linkIndexM3(coord, arg.dim, d);
            const int gauge_idx = back3_idx;
            const Link L = arg.L(d, gauge_idx, 1 - arg.parity);
            const Vector in = arg.in(back3_idx, their_spinor_parity);
            out = mv_add(conj(L), -in, out);
          }
        }

      } // dimension

      return out;
    }

    /**
       @brief Applies the clover portion of the local dslash, which is D_{oe} for parity even and D_{eo} for parity odd. Assumes is_partitioned is true.
       @param[in,out] out Accumulated output
       @param[in] coord Precomputed Coord structure
       @return Accumulated output ColorVector
    */
    template <int d>
    __device__ __host__ void cloverDslashDirection(Vector &out, const Coord<nDim> &coord) const {
      static_assert(Arg::step == QUDA_STAGGERED_LOCAL_CLOVER, "cloverDslash called for a local argument struct");
      // to be updated if we implement a full-parity version
      constexpr int my_spinor_parity = 0;

      // helper lambda for getting U
      auto getU = [=] (int d_, int x_cb_, int parity_, int sign_) -> Link {
        return arg.improved ? arg.U(d_, x_cb_, parity_) : arg.U(d_, x_cb_, parity_, StaggeredPhase(coord, d_, sign_, arg));
      };

      // helper lambda for getting U from ghost zone
      auto getUGhost = [=] (int d_, int ghost_idx_, int parity_, int sign_) -> Link {
        return arg.improved ? arg.U.Ghost(d_, ghost_idx_, parity_) : arg.U.Ghost(d_, ghost_idx_, parity_, StaggeredPhase(coord, d_, sign_, arg));
      };

      // the "in" vector at out's site --- this gets reused many times so (for now) we stick it in registers
      Vector in = arg.in(coord.x_cb, my_spinor_parity);

      // standard - forward direction
      if ((coord[d] + 1) >= arg.dim[d]) {
        // Load the U link once for the "self" contribution
        const Link U = getU(d, coord.x_cb, arg.parity, +1);

        // perform backwards (from previous pass) -- gathering to "X"
        Vector accum = fatLinkForwardContribution<d>(coord, U, in);

        // forwards - standard (gather from X, to X - 1)
        out = mv_add(U, accum, out);
      }

      // improved - forward direction
      if constexpr (Arg::improved) {
	if ((coord[d] + 3) >= arg.dim[d]) {
          // Load the L link once for the "self" contribution
          const Link L = arg.L(d, coord.x_cb, arg.parity);

          // perform backwards (from previous pass) -- gathering to ["X", "X+1", "X+2"]
          Vector accum = longLinkForwardContribution<d>(coord, L, in);

          // forwards - improved (gather from ["X", "X+1", "X+2"] to ["X-3","X-2","X-1"])
          out = mv_add(L, accum, out);
        }
      }

      // standard - backward direction
      if (coord[d] == 0) {
        // Load the U link once for the "self" contribution
        const int ghost_idx2 = ghostFaceIndexStaggered<0>(coord, arg.dim, d, 1);
        const Link U = getUGhost(d, ghost_idx2, 1 - arg.parity, -1);

        // first bit: gather from site [-1]
        Vector accum = fatLinkBackwardContribution<d>(coord, U, in);

        // and now, let's gather what we accumulated at [-1]
        out = mv_add(conj(U), -accum, out);
      }

      // improved - backward direction
      if constexpr (Arg::improved) {
        Vector accum;

        if (coord[d] < 3) {
          const int ghost_idx = ghostFaceIndexStaggered<0>(coord, arg.dim, d, 1);
          const Link L = arg.L.Ghost(d, ghost_idx, 1 - arg.parity);

          // second bit: gather from site [-3]
          accum = longLinkBackwardContribution<d>(coord, L, in);

          // and now, let's gather what we accumulated at [-3]
          out = mv_add(conj(L), -accum, out);
        }
      }
    }

    /**
       @brief Driver to apply the full local dslash
       @param[in] x_cb input coordinate
    */
    __device__ __host__ void operator()(int x_cb, int) {
      constexpr int nDim = 4;
      Coord<nDim> coord;

      // to be updated if we implement a full-parity version
      constexpr int my_spinor_parity = 0;

      Vector out;

      // Get coordinates
      coord.x_cb = x_cb;
      coord.X = getCoords(coord, x_cb, arg.dim, arg.parity);
      coord.s = 0;

      if constexpr (Arg::step == QUDA_STAGGERED_LOCAL_CLOVER) {
        if (arg.is_partitioned[0]) cloverDslashDirection<0>(out, coord);
        if (arg.is_partitioned[1]) cloverDslashDirection<1>(out, coord);
        if (arg.is_partitioned[2]) cloverDslashDirection<2>(out, coord);
        if (arg.is_partitioned[3]) cloverDslashDirection<3>(out, coord);
      } else {
        out = localDslash(coord);
      } // is clover

      if constexpr (Arg::step == QUDA_STAGGERED_LOCAL_STEP2) {
        Vector x = arg.x(coord.x_cb, my_spinor_parity);
        out = arg.a * x - out;
      }

      if constexpr (Arg::step == QUDA_STAGGERED_LOCAL_CLOVER) {
        Vector partial_out = arg.out(coord.x_cb, my_spinor_parity);
        out = partial_out - out;
      }

      arg.out(coord.x_cb, my_spinor_parity) = out;

    }
  };

} // namespace quda


