#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <array.h>
#include <reduction_kernel.h>

namespace quda {

  /**
     @brief Calculates the Polyakov loop in a given direction, returning the product matrix

     @return The product of the gauge path
     @param[in] arg Kernel argumnt
     @param[in] x Full index array
     @param[in] parity Parity index
  */
  template <int dir, typename Arg>
  __device__ __host__ inline typename Arg::Link
  computePolyakovLoop(const Arg &arg, int x[4], int parity)
  {
    int dx[4] = {0, 0, 0, 0};

    using Link = typename Arg::Link;

    // polyloop: current matrix
    // link: the loaded matrix in this round
    Link polyloop, link;
    setIdentity(&polyloop);

    int nbr_oddbit = parity;

    for (int dt = 0; dt < arg.X[dir]; dt++) {
      dx[dir] = dt;
      link = arg.U(dir, linkIndexShift(x, dx, arg.X), nbr_oddbit);
      polyloop = polyloop * link;
      nbr_oddbit = nbr_oddbit ^ 1;
    } // dt
    return polyloop;
  }

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct GaugePolyakovLoopSplitArg : public kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    using Gauge = typename gauge_mapper<Float,recon>::type;
    using Link = Matrix<complex<Float>, 3>;

    int X[4];
    Gauge P;
    Gauge U;

    GaugePolyakovLoopSplitArg(GaugeField &P_, const GaugeField &U_) :
      kernel_param(dim3(P_.LocalVolumeCB(), 2, 1)),
      P(P_),
      U(U_)
    {
      for (int dir=0; dir<4; ++dir){
        X[dir] = U_.X()[dir];
        if (dir != 3 && U_.X()[dir] != P_.X()[dir]) errorQuda("Lengths %d %d in dimension %d do not agree", U_.X()[dir], P_.X()[dir], dir);
      }
    }

  };

  template <typename Arg> struct PolyakovLoopSplit {
    const Arg &arg;
    constexpr PolyakovLoopSplit(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      int x[4];
      getCoords(x, x_cb, arg.X, parity);

      auto polyloop = computePolyakovLoop<3>(arg, x, parity);

      // save loop
      arg.P(0, x_cb, parity) = polyloop;

    }

  };

  template <typename Float_, int nColor_, QudaReconstructType recon_, bool compute_loop_>
  struct GaugePolyakovLoopTraceArg : public ReduceArg<array<double, 2>> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr bool compute_loop = compute_loop_;
    using Gauge = typename gauge_mapper<Float,recon>::type;
    using Link = Matrix<complex<Float>, 3>;

    int X[4];
    Gauge U;

    GaugePolyakovLoopTraceArg(const GaugeField &U_) :
      ReduceArg<reduce_t>(dim3(U_.LocalVolumeCB() / U_.X()[3], 2, 1)),
      U(U_)
    {
      for (int dir=0; dir<4; ++dir){
        X[dir] = U_.X()[dir];
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
      using Link = typename Arg::Link;

      Link polyloop;
      reduce_t ploop{0, 0};

      if constexpr (arg.compute_loop) {
        // Dimension is not split, we need to compute the Polyakov loop
        int x[4];
        getCoords(x, x_cb, arg.X, parity);
        polyloop = computePolyakovLoop<3>(arg, x, parity);
      } else {
        // Dimension is split, we're only computing the trace
        polyloop = arg.U(0, x_cb, parity);
      }

      // accumulate trace
      auto tr = getTrace( polyloop );
      ploop[0] = tr.real();
      ploop[1] = tr.imag();

      return operator()(ploop, value);
    }

  };

} // namespace quda
