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
            QudaReconstructType reconstruct_l_, bool improved_, QudaStaggeredPhase phase_ = QUDA_STAGGERED_PHASE_MILC>
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
    const QudaStaggeredLocalType step; /** which step of the local staggered dslash we're applying */

    const real tboundary; /** temporal boundary condition */
    const bool is_first_time_slice; /** are we on the first (global) time slice */
    const bool is_last_time_slice; /** are we on the last (global) time slice */
    static constexpr bool improved = improved_;

    LocalStaggeredArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L, double a,
                 const ColorSpinorField &x, int parity, QudaStaggeredLocalType step) :
      kernel_param(dim3(out.VolumeCB(), 1, 1)),
      out(out),
      in(in),
      x(x),
      U(U),
      L(L),
      a(a),
      parity(parity),
      step(step),
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
  struct LocalStaggeredHoppingApply {
    const Arg &arg;
    constexpr LocalStaggeredHoppingApply(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    /**
       @brief Applies step 1 of the local dslash, which is D_{oe} for parity even and D_{eo} for parity odd
       @param[in] x_cb input coordinate
    */
    __device__ __host__ void operator()(int x_cb, int) {
      constexpr int nDim = 4;
      Coord<nDim> coord;

      using real = typename Arg::real;
      using Vector = ColorSpinor<real, Arg::nColor, 1>;
      using Link = Matrix<complex<real>, Arg::nColor>;

      // to be updated if we implement a full-parity version
      constexpr int my_spinor_parity = 0;
      constexpr int their_spinor_parity = 0;

      Vector out;

      // Get coordinates
      coord.x_cb = x_cb;
      coord.X = getCoords(coord, x_cb, arg.dim, arg.parity);
      coord.s = 0;

#pragma unroll
      for (int d = 0; d < nDim; d++)
      {

        // standard - forward direction
        if (!arg.is_partitioned[d] || (coord[d] + 1) < arg.dim[d])
        {
          const int fwd_idx = linkIndexP1(coord, arg.dim, d);
          const Link U = arg.improved ? arg.U(d, coord.x_cb, arg.parity) : arg.U(d, coord.x_cb, arg.parity, StaggeredPhase(coord, d, +1, arg));
          Vector in = arg.in(fwd_idx, their_spinor_parity);
          out = mv_add(U, in, out);
        }

        // improved - forward direction
        if (arg.improved && (!arg.is_partitioned[d] || (coord[d] + 3) < arg.dim[d]))
        {
          const int fwd3_idx = linkIndexP3(coord, arg.dim, d);
          const Link L = arg.L(d, coord.x_cb, arg.parity);
          const Vector in = arg.in(fwd3_idx, their_spinor_parity);
          out = mv_add(L, in, out);
        }

        // standard - backward direction
        if (!arg.is_partitioned[d] || (coord[d] - 1) >= 0)
        {
          const int back_idx = linkIndexM1(coord, arg.dim, d);
          const int gauge_idx = back_idx;
          const Link U = arg.improved ? arg.U(d, gauge_idx, 1 - arg.parity) :
            arg.U(d, gauge_idx, 1 - arg.parity, StaggeredPhase(coord, d, -1, arg));
          Vector in = arg.in(back_idx, their_spinor_parity);
          out = mv_add(conj(U), -in, out);
        }

        // improved - backward direction
        if (arg.improved && (!arg.is_partitioned[d] || (coord[d] - 3) >= 0))
        {
          const int back3_idx = linkIndexM3(coord, arg.dim, d);
          const int gauge_idx = back3_idx;
          const Link L = arg.L(d, gauge_idx, 1 - arg.parity);
          const Vector in = arg.in(back3_idx, their_spinor_parity);
          out = mv_add(conj(L), -in, out);
        }

      } // dimension

      if (arg.step == QUDA_STAGGERED_LOCAL_STEP2) {
        Vector x = arg.x(coord.x_cb, my_spinor_parity);
        out = arg.a * x - out;
      }

      arg.out(coord.x_cb, my_spinor_parity) = out;

    }
  };

} // namespace quda
