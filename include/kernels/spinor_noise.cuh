#include <math_helper.cuh>
#include <color_spinor_field_order.h>
#include <random_helper.h>
#include <kernel.h>

namespace quda {

  using namespace colorspinor;

  template <typename real_, int nSpin_, int nColor_, QudaNoiseType noise_>
  struct SpinorNoiseArg : kernel_param<> {
    using real = real_;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    static constexpr QudaFieldOrder order = colorspinor::getNative<real>(nSpin);
    static constexpr QudaNoiseType noise = noise_;
    using V = typename colorspinor::FieldOrderCB<real, nSpin, nColor, 1, order>;
    V v;
    RNGState *rng;
    SpinorNoiseArg(ColorSpinorField &v, RNGState *rng) :
      kernel_param(dim3(v.VolumeCB(), v.SiteSubset(), 1)),
      v(v),
      rng(rng) { }
  };

  template<typename real, typename Arg> // Gauss
  __device__ __host__ inline void genGauss(Arg &arg, RNGState& localState, int parity, int x_cb, int s, int c) {
    real phi = 2.0 * uniform<real>::rand(localState);
    real radius = uniform<real>::rand(localState);
    radius = sqrt(-log(radius));
    real phi_sin, phi_cos;
    quda::sincospi(phi, &phi_sin, &phi_cos);
    arg.v(parity, x_cb, s, c) = radius * complex<real>(phi_cos, phi_sin);
  }

  template<typename real, typename Arg> // Uniform
  __device__ __host__ inline void genUniform(Arg &arg, RNGState& localState, int parity, int x_cb, int s, int c) {
    real x = uniform<real>::rand(localState);
    real y = uniform<real>::rand(localState);
    arg.v(parity, x_cb, s, c) = complex<real>(x, y);
  }

  template <typename Arg> struct NoiseSpinor {
    const Arg &arg;
    constexpr NoiseSpinor(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      RNGState localState = arg.rng[parity * arg.threads.x + x_cb];
      for (int s=0; s<Arg::nSpin; s++) {
        for (int c=0; c<Arg::nColor; c++) {
          if (Arg::noise == QUDA_NOISE_GAUSS) genGauss<typename Arg::real>(arg, localState, parity, x_cb, s, c);
          else if (Arg::noise == QUDA_NOISE_UNIFORM) genUniform<typename Arg::real>(arg, localState, parity, x_cb, s, c);
        }
      }
      arg.rng[parity * arg.threads.x + x_cb] = localState;
    }
  };

}
