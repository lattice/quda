#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <array.h>
#include <reduction_kernel.h>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct GaugePolyakovLoopArg : public ReduceArg<array<double, 2>> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    using Gauge = typename gauge_mapper<Float,recon>::type;
    using Link = Matrix<complex<Float>, 3>;

    int X[4];
    Gauge P;
    Gauge U;

    GaugePolyakovLoopArg(GaugeField &P_, const GaugeField &U_) :
      ReduceArg<reduce_t>(dim3(P_.LocalVolumeCB(), 2, 1)),
      P(P_),
      U(U_)
    {
      for (int dir=0; dir<4; ++dir){
        X[dir] = U_.X()[dir];
        if (dir != 3 && U_.X()[dir] != P_.X()[dir]) errorQuda("Lengths %d %d in dimension %d do not agree", U_.X()[dir], P_.X()[dir], dir);
      }
    }

  };

  template <typename Arg> struct PolyakovLoop : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    constexpr PolyakovLoop(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    // return the (temporal) Polyakov Loop at 3-d site (x_cb, parity)
    __device__ __host__ inline reduce_t operator()(reduce_t &value, int x_cb, int parity)
    {
      reduce_t ploop{0, 0};

      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      int dx[4] = {0, 0, 0, 0};

      using Link = typename Arg::Link;

      // polyloop: current matrix
      // link: the loaded matrix in this round
      Link polyloop, link;
      setIdentity(&polyloop);

      int nbr_oddbit = parity;

      for (int dt = 0; dt < arg.X[3]; dt++) {
        dx[3] = dt;
        link = arg.U(3, linkIndexShift(x, dx, arg.X), nbr_oddbit);
        polyloop = polyloop * link;
        nbr_oddbit = nbr_oddbit ^ 1;
      } // t

      // save loop
      arg.P(0, x_cb, parity) = polyloop;

      // accumulate trace
      auto tr = getTrace( polyloop );
      ploop[0] = tr.real();
      ploop[1] = tr.imag();

      return operator()(ploop, value);
    }

  };

} // namespace quda
