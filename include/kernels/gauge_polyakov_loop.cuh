#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <array.h>
#include <reduction_kernel.h>

namespace quda {

  template <typename store_t, int nColor_, QudaReconstructType recon_>
  struct GaugeInsertTimesliceArg : public kernel_param<> {
    using real = typename::mapper<store_t>::type;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    using Gauge = typename gauge_mapper<real,recon>::type;
    using Link = Matrix<complex<real>, 3>;

    int X_bulk[4];
    int_fastdiv X_slice[4];
    Gauge U;
    Gauge S;
    int timeslice;

    GaugeInsertTimesliceArg(GaugeField &U_, const GaugeField &S_, const int timeslice_) :
      kernel_param(dim3(S_.LocalVolumeCB(), 2, 1)),
      U(U_),
      S(S_),
      timeslice(timeslice_)
    {
      if (U_.Geometry() != QUDA_SCALAR_GEOMETRY || S_.Geometry() != QUDA_SCALAR_GEOMETRY)
        errorQuda("Unexpected geometry pair U %d ; S %d", U_.Geometry(), S_.Geometry());
      for (int dir = 0; dir < 4; dir++) {
        if (U_.R()[dir] > 0 || S_.R()[dir] > 0)
          errorQuda("Unexpected non-zero extended radii %d %d in direction %d", U_.R()[dir], S_.R()[dir], dir);
        X_bulk[dir] = U_.X()[dir];
        X_slice[dir] = S_.X()[dir];
        if (dir != 3 && U_.X()[dir] != S_.X()[dir]) errorQuda("Lengths %d %d in dimension %d do not agree", U_.X()[dir], S_.X()[dir], dir);
      }
      if (S_.X()[3] != 1) errorQuda("Unexpected time extent %d, expected 1", S_.X()[3]);
    }

  };

  template <typename Arg> struct InsertTimeslice {
    const Arg &arg;
    constexpr InsertTimeslice(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      int x[4];
      getCoords(x, x_cb, arg.X_slice, parity);

      // Get bulk x_cb/parity
      x[3] = arg.timeslice;
      int x_bulk_cb = linkIndex(x, arg.X_bulk);
      int parity_bulk = (x[0] + x[1] + x[2] + x[3]) & 1;

      // Load from slice, store into bulk
      typename Arg::Link link = arg.S(0, x_cb, parity);
      arg.U(0, x_bulk_cb, parity_bulk) = link;
    }

  };

  /**
     @brief Calculates the Polyakov loop in a given direction, returning the product matrix

     @return The product of the gauge path
     @param[in] arg Kernel argumnt
     @param[in] x Full index array
     @param[in] parity Parity index
  */
  template <int dir, typename Arg>
  __device__ __host__ inline typename Arg::HighPrecLink
  computePolyakovLoop(const Arg &arg, int x[4], int parity)
  {
    int dx[4] = {0, 0, 0, 0};

    using Link = typename Arg::Link;
    using HighPrecLink = typename Arg::HighPrecLink;

    // polyloop: current matrix
    // link: the loaded matrix in this round
    Link link;
    HighPrecLink hi_link, polyloop;
    setIdentity(&polyloop);

    int nbr_oddbit = parity;

    for (int dt = 0; dt < arg.X[dir]; dt++) {
      dx[dir] = dt;
      link = arg.U(arg.geometry == QUDA_VECTOR_GEOMETRY ? dir : 0, linkIndexShift(x, dx, arg.E), nbr_oddbit);
      hi_link = link; // promote
      polyloop = polyloop * hi_link;
      nbr_oddbit = nbr_oddbit ^ 1;
    } // dt
    return polyloop;
  }

  template <typename store_t, int nColor_, QudaReconstructType recon_>
  struct GaugePolyakovLoopProductArg : public kernel_param<> {
    using real = typename mapper<store_t>::type;
    using AccumFloat = double;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr QudaFieldGeometry geometry = QUDA_VECTOR_GEOMETRY;
    using Gauge = typename gauge_mapper<real,recon>::type;
    using AccumGauge = typename gauge_mapper<AccumFloat,recon>::type;
    using Link = Matrix<complex<real>, 3>;
    using HighPrecLink = Matrix<complex<double>, 3>;

    // While the Polyakov loop doesn't need extended fields, it is a gauge
    // observable, which means it tends to get passed extended fields. This
    // logic keeps it robust to this reality.
    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4];

    AccumGauge P;
    Gauge U;

    GaugePolyakovLoopProductArg(GaugeField &P_, const GaugeField &U_) :
      kernel_param(dim3(P_.LocalVolumeCB(), 2, 1)),
      P(P_),
      U(U_)
    {
      if (U_.Geometry() != QUDA_VECTOR_GEOMETRY || P_.Geometry() != QUDA_SCALAR_GEOMETRY)
        errorQuda("Unexpected geometry pair U %d ; P %d", U_.Geometry(), P_.Geometry());
      for (int dir=0; dir<4; ++dir) {
        if (P_.R()[dir] > 0) errorQuda("Unexpected extended field radius %d in direction %d", P_.R()[dir], dir);
        border[dir] = U_.R()[dir];
        E[dir] = U_.X()[dir];
        X[dir] = U_.X()[dir] - 2 * border[dir];
        if (dir != 3 && X[dir] != P_.X()[dir]) errorQuda("Lengths %d %d in dimension %d do not agree", X[dir], P_.X()[dir], dir);
      }
    }

  };

  template <typename Arg> struct PolyakovLoopProduct {
    const Arg &arg;
    constexpr PolyakovLoopProduct(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      int x[4];
      getCoords(x, x_cb, arg.X, parity);

      // hack
      if (x[3] > 0) return;

#pragma unroll
      for (int dr = 0; dr < 4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

      auto polyloop = computePolyakovLoop<3>(arg, x, parity);

      // save loop
      arg.P(0, x_cb, parity) = polyloop;
    }

  };

  template <typename store_t, int nColor_, QudaReconstructType recon_>
  struct GaugePolyakovLoopTraceArg : public ReduceArg<array<double, 2>> {
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    using Gauge = typename gauge_mapper<real,recon>::type;
    using Link = Matrix<complex<real>, 3>;
    using HighPrecLink = Matrix<complex<double>, 3>;

    // While the Polyakov loop doesn't need extended fields, it is a gauge
    // observable, which means it tends to get passed extended fields. This
    // logic keeps it robust to this reality.
    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4];
    Gauge U;
    QudaFieldGeometry geometry;

    GaugePolyakovLoopTraceArg(const GaugeField &U_) :
      ReduceArg<reduce_t>(dim3(U_.LocalVolumeCB() / (U_.X()[3] - 2 * U_.R()[3]), 2, 1)),
      U(U_),
      geometry(U_.Geometry())
    {
      for (int dir=0; dir<4; ++dir) {
        border[dir] = U_.R()[dir];
        E[dir] = U_.X()[dir];
        X[dir] = U_.X()[dir] - 2 * border[dir];
      }
    }

  };

  template <typename Arg> struct PolyakovLoopTrace : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    constexpr PolyakovLoopTrace(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    // return the (temporal) Polyakov Loop at 3-d site (x_cb, parity)
    __device__ __host__ inline reduce_t operator()(reduce_t &value, int x_cb, int parity)
    {
      using HighPrecLink = typename Arg::HighPrecLink;

      HighPrecLink polyloop;
      reduce_t ploop{0, 0};

      int x[4];
      getCoords(x, x_cb, arg.X, parity);
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

      if (arg.geometry == QUDA_VECTOR_GEOMETRY) {
        // U is the full gauge field, need to contract over the right dimension
        // This codepath is hit when the MPI decomposition isn't split over the
        // loop direction
        polyloop = computePolyakovLoop<3>(arg, x, parity);
      } else {
        // U is a "scalar" gauge field, which is the packed outputs of multi-GPU
        // traces. In this case the "3" direction is the packing dimension
        polyloop = computePolyakovLoop<3>(arg, x, parity);
      }

      // accumulate trace
      auto tr = getTrace( polyloop );
      ploop[0] = tr.real();
      ploop[1] = tr.imag();

      return operator()(ploop, value);
    }

  };

} // namespace quda
