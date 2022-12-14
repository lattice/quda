#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <random_helper.h>
#include <kernel.h>

namespace quda {

  template <typename Float_, int nColor_, QudaNoiseType noise_>
  struct GaugeNoiseArg : kernel_param<> {
    using Float = Float_;
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr QudaNoiseType noise = noise_;
    using Gauge = gauge::FieldOrder<real, nColor, 1, QUDA_FLOAT2_GAUGE_ORDER, true, real>;

    int geometry;
    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4];
    Gauge U;
    RNGState *rng;
    real sigma; // where U = exp(sigma * H)

    GaugeNoiseArg(const GaugeField &U, RNGState *rng) :
      kernel_param(dim3(U.LocalVolumeCB(), 2, 1)),
      geometry(U.Geometry()),
      U(U),
      rng(rng)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = U.R()[dir];
        E[dir] = U.X()[dir];
        X[dir] = U.X()[dir] - border[dir] * 2;
      }
    }
  };

  template<typename real, typename Arg> // Gauss
  __device__ __host__ inline void genGauss(Arg &arg, RNGState& localState, int parity, int x_cb, int g, int r, int c)
  {
    real phi = 2.0 * uniform<real>::rand(localState);
    real radius = uniform<real>::rand(localState);
    radius = sqrt(-log(radius));
    real phi_sin, phi_cos;
    quda::sincospi(phi, &phi_sin, &phi_cos);
    arg.U(g, parity, x_cb, r, c) = radius * complex<real>(phi_cos, phi_sin);
  }

  template<typename real, typename Arg> // Uniform
  __device__ __host__ inline void genUniform(Arg &arg, RNGState& localState, int parity, int x_cb, int g, int r, int c)
  {
    real x = uniform<real>::rand(localState);
    real y = uniform<real>::rand(localState);
    arg.U(g, parity, x_cb, r, c) = complex<real>(x, y);
  }

  template <typename Arg> struct NoiseGauge {
    const Arg &arg;
    constexpr NoiseGauge(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      for (int dr = 0; dr < 4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates
      int e_cb = linkIndex(x, arg.E);

      RNGState localState = arg.rng[parity * arg.threads.x + x_cb];
      for (int g = 0; g < arg.geometry; g++) {
        for (int r = 0; r < Arg::nColor; r++) {
          for (int c = 0; c < Arg::nColor; c++) {

            if (Arg::noise == QUDA_NOISE_GAUSS) genGauss<typename Arg::real>(arg, localState, parity, e_cb, g, r, c);
            else if (Arg::noise == QUDA_NOISE_UNIFORM) genUniform<typename Arg::real>(arg, localState, parity, e_cb, g, r, c);

          }
        }
      }
      arg.rng[parity * arg.threads.x + x_cb] = localState;
    }
  };

}
