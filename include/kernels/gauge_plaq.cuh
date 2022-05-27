#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <array.h>
#include <reduction_kernel.h>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct GaugePlaqArg : public ReduceArg<array<double, 2>> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4];
    Gauge U;

    GaugePlaqArg(const GaugeField &U_) :
      ReduceArg<reduce_t>(dim3(U_.LocalVolumeCB(), 2, 1)),
      U(U_)
    {
      for (int dir=0; dir<4; ++dir){
	border[dir] = U_.R()[dir];
	E[dir] = U_.X()[dir];
	X[dir] = U_.X()[dir] - border[dir]*2;
      }
    }
  };

  template<typename Arg>
  __device__ inline double plaquette(const Arg &arg, int x[], int parity, int mu, int nu)
  {
    using Link = Matrix<complex<typename Arg::Float>,3>;

    int dx[4] = {0, 0, 0, 0};
    Link U1 = arg.U(mu, linkIndexShift(x,dx,arg.E), parity);
    dx[mu]++;
    Link U2 = arg.U(nu, linkIndexShift(x,dx,arg.E), 1-parity);
    dx[mu]--;
    dx[nu]++;
    Link U3 = arg.U(mu, linkIndexShift(x,dx,arg.E), 1-parity);
    dx[nu]--;
    Link U4 = arg.U(nu, linkIndexShift(x,dx,arg.E), parity);

    return getTrace( U1 * U2 * conj(U3) * conj(U4) ).real();
  }

  template <typename Arg> struct Plaquette : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb in x, parity in y
    const Arg &arg;
    constexpr Plaquette(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    // return the plaquette at site (x_cb, parity)
    __device__ __host__ inline reduce_t operator()(reduce_t &value, int x_cb, int parity)
    {
      reduce_t plaq{0, 0};

      int x[4];
      getCoords(x, x_cb, arg.X, parity);
#pragma unroll
      for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

#pragma unroll
      for (int mu = 0; mu < 3; mu++) {
#pragma unroll
        for (int nu = 0; nu < 3; nu++) {
          if (nu >= mu + 1) plaq[0] += plaquette(arg, x, parity, mu, nu);
        }

        plaq[1] += plaquette(arg, x, parity, mu, 3);
      }

      return operator()(plaq, value);
    }

  };

} // namespace quda
