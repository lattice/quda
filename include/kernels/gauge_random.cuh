#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <random_helper.h>
#include <kernel.h>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_, bool group_>
  struct GaugeGaussArg : kernel_param<> {
    using Float = Float_;
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    static constexpr bool group = group_;

    using Gauge = typename gauge_mapper<Float, recon>::type;

    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4];
    Gauge U;
    RNGState *rng;
    real sigma; // where U = exp(sigma * H)

    GaugeGaussArg(const GaugeField &U, RNGState *rng, double sigma) :
      kernel_param(dim3(U.LocalVolumeCB(), 2, 1)),
      U(U),
      rng(rng),
      sigma(sigma)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = U.R()[dir];
        E[dir] = U.X()[dir];
        X[dir] = U.X()[dir] - border[dir] * 2;
      }
    }
  };

  template <typename real, typename Link> __device__ __host__ Link gauss_su3(RNGState &localState)
  {
    Link ret;
    real rand1[4], rand2[4], phi[4], radius[4], temp1[4], temp2[4];

    for (int i = 0; i < 4; ++i) {
      rand1[i] = uniform<real>::rand(localState);
      rand2[i] = uniform<real>::rand(localState);
      phi[i] = 2.0 * rand1[i];
      radius[i] = sqrt(-log(rand2[i]));
      quda::sincospi(phi[i], &temp2[i], &temp1[i]);
      temp1[i] *= radius[i];
      temp2[i] *= radius[i];
    }

    // construct Anti-Hermitian matrix
    const real rsqrt_3 = quda::rsqrt(3.0);
    ret(0, 0) = complex<real>(0.0, temp1[2] + rsqrt_3 * temp2[3]);
    ret(1, 1) = complex<real>(0.0, -temp1[2] + rsqrt_3 * temp2[3]);
    ret(2, 2) = complex<real>(0.0, -2.0 * rsqrt_3 * temp2[3]);
    ret(0, 1) = complex<real>(temp1[0], temp1[1]);
    ret(1, 0) = complex<real>(-temp1[0], temp1[1]);
    ret(0, 2) = complex<real>(temp1[3], temp2[0]);
    ret(2, 0) = complex<real>(-temp1[3], temp2[0]);
    ret(1, 2) = complex<real>(temp2[1], temp2[2]);
    ret(2, 1) = complex<real>(-temp2[1], temp2[2]);

    return ret;
  }

  template <typename Arg> struct GaussGauge
  {
    const Arg &arg;
    constexpr GaussGauge(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      using real = typename mapper<typename Arg::Float>::type;
      using Link = Matrix<complex<real>, Arg::nColor>;

      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      for (int dr = 0; dr < 4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

      if (arg.group and arg.sigma == 0.0) {
        // if sigma = 0 then we just set the output matrix to the identity and finish
        Link I;
        setIdentity(&I);
        for (int mu = 0; mu < 4; mu++) arg.U(mu, linkIndex(x, arg.E), parity) = I;
      } else if (not arg.group and arg.sigma == 0.0) {
        // if sigma = 0 then we just set the output matrix to the zero and finish
        Link O;
        setZero(&O);
        for (int mu = 0; mu < 4; mu++) arg.U(mu, linkIndex(x, arg.E), parity) = O;
      } else {
        for (int mu = 0; mu < 4; mu++) {
          RNGState localState = arg.rng[parity * arg.threads.x + x_cb];

          // generate Gaussian distributed su(n) field
          Link u = arg.sigma * gauss_su3<real, Link>(localState);
          if constexpr (Arg::group) {
            expsu3<real>(u);
          }
          arg.U(mu, linkIndex(x, arg.E), parity) = u;

          arg.rng[parity * arg.threads.x + x_cb] = localState;
        }
      }
    }
  };

}
