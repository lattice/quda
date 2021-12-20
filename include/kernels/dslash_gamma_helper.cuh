#pragma once

#include "color_spinor_field_order.h"
#include "index_helper.cuh"
#include "color_spinor.h"
#include "kernel.h"

namespace quda {

  /**
     @brief Parameter structure for driving the Gamma operator
   */
  template <typename Float, int nColor_>
  struct GammaArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    constexpr static int nColor = nColor_;
    typedef typename colorspinor_mapper<Float,4,nColor>::type F;

    F out;                // output vector field
    const F in;           // input vector field
    const int d;          // which gamma matrix are we applying
    const int nParity;    // number of parities we're working on
    bool doublet;         // whether we applying the operator to a doublet
    const int volumeCB;   // checkerboarded volume
    real a;               // scale factor
    real b;               // chiral twist
    real c;               // flavor twist

    GammaArg(ColorSpinorField &out, const ColorSpinorField &in, int d,
	     real kappa=0.0, real mu=0.0, real epsilon=0.0,
	     bool dagger=false, QudaTwistGamma5Type twist=QUDA_TWIST_GAMMA5_INVALID) :
      out(out), in(in), d(d), nParity(in.SiteSubset()),
      doublet(in.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
      volumeCB(doublet ? in.VolumeCB()/2 : in.VolumeCB()), a(0.0), b(0.0), c(0.0)
    {
      checkPrecision(out, in);
      checkLocation(out, in);
      if (d < 0 || d > 4) errorQuda("Undefined gamma matrix %d", d);
      if (in.Nspin() != 4) errorQuda("Cannot apply gamma5 to nSpin=%d field", in.Nspin());
      if (!in.isNative() || !out.isNative()) errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());

      if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
	if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
          b = 2.0 * kappa * mu;
          a = 1.0;
        } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
          b = -2.0 * kappa * mu;
          a = 1.0 / (1.0 + b * b);
        }
	c = 0.0;
        if (dagger) b *= -1.0;
      } else if (doublet) {
        if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
          b = 2.0 * kappa * mu;
          c = -2.0 * kappa * epsilon;
          a = 1.0;
        } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
          b = -2.0 * kappa * mu;
          c = 2.0 * kappa * epsilon;
          a = 1.0 / (1.0 + b * b - c * c);
          if (a <= 0) errorQuda("Invalid twisted mass parameters (kappa=%e, mu=%e, epsilon=%e)\n", kappa, mu, epsilon);
        }
        if (dagger) b *= -1.0;
      }
      this->threads = dim3(doublet ? in.VolumeCB()/2 : in.VolumeCB(), in.SiteSubset(), 1);
    }
  };

  /**
     @brief Application of Gamma matrix to a color spinor field
  */
  template <typename Arg> struct Gamma {
    const Arg &arg;
    constexpr Gamma(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      ColorSpinor<typename Arg::real, Arg::nColor, 4> in = arg.in(x_cb, parity);
      switch(arg.d) {
      case 0: arg.out(x_cb, parity) = in.gamma(0);
      case 1: arg.out(x_cb, parity) = in.gamma(1);
      case 2: arg.out(x_cb, parity) = in.gamma(2);
      case 3: arg.out(x_cb, parity) = in.gamma(3);
      case 4: arg.out(x_cb, parity) = in.gamma(4);
      }
    }
  };

  /**
     @brief Application of twist to a color spinor field
  */
  template <typename Arg> struct TwistGamma {
    using fermion_t = ColorSpinor<typename Arg::real, Arg::nColor, 4>;
    const Arg &arg;
    constexpr TwistGamma(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      constexpr int d = 4;
      if (!arg.doublet) {
        fermion_t in = arg.in(x_cb, parity);
        arg.out(x_cb, parity) = arg.a * (in + arg.b * in.igamma(d));
      } else {
        fermion_t in_1 = arg.in(x_cb+0*arg.volumeCB, parity);
        fermion_t in_2 = arg.in(x_cb+1*arg.volumeCB, parity);
        arg.out(x_cb + 0 * arg.volumeCB, parity) = arg.a * (in_1 + arg.b * in_1.igamma(d) + arg.c * in_2);
        arg.out(x_cb + 1 * arg.volumeCB, parity) = arg.a * (in_2 - arg.b * in_2.igamma(d) + arg.c * in_1);
      }
    }
  };

}
