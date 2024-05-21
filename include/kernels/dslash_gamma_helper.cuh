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
    using F = typename colorspinor_mapper<Float, 4, nColor, false, false, true>::type;

    static constexpr unsigned int max_n_src = MAX_MULTI_RHS;
    F out[max_n_src];      // output vector field
    F in[max_n_src];      // input vector field
    const int d;          // which gamma matrix are we applying
    const int nParity;    // number of parities we're working on
    const bool doublet;   // whether we applying the operator to a doublet
    const int n_flavor;   // number of flavors
    const int volumeCB;   // checkerboarded volume
    real a;               // scale factor
    real b;               // chiral twist
    real c;               // flavor twist

    GammaArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int d,
	     real kappa = 0.0, real mu = 0.0, real epsilon = 0.0,
	     bool dagger = false, QudaTwistGamma5Type twist = QUDA_TWIST_GAMMA5_INVALID) :
      d(d), nParity(in.SiteSubset()),
      doublet(in.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
      n_flavor(doublet ? 2 : 1),
      volumeCB(doublet ? in.VolumeCB()/2 : in.VolumeCB()), a(0.0), b(0.0), c(0.0)
    {
      if (out.size() > max_n_src) errorQuda("vector set size %lu greater than max size %d", out.size(), max_n_src);
      for (auto i = 0u; i < in.size(); i++) {
        this->in[i] = in[i];
        this->out[i] = out[i];
      }

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

    __device__ __host__ void operator()(int x_cb, int src_idx, int parity)
    {
      for (int i = 0; i < arg.n_flavor; i++) {
        ColorSpinor<typename Arg::real, Arg::nColor, 4> in = arg.in[src_idx](x_cb + i * arg.volumeCB, parity);
        switch(arg.d) {
        case 0: arg.out[src_idx](x_cb + i * arg.volumeCB, parity) = in.gamma(0); break;
        case 1: arg.out[src_idx](x_cb + i * arg.volumeCB, parity) = in.gamma(1); break;
        case 2: arg.out[src_idx](x_cb + i * arg.volumeCB, parity) = in.gamma(2); break;
        case 3: arg.out[src_idx](x_cb + i * arg.volumeCB, parity) = in.gamma(3); break;
        case 4: arg.out[src_idx](x_cb + i * arg.volumeCB, parity) = in.gamma(4); break;
        }
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

    __device__ __host__ void operator()(int x_cb, int src_idx, int parity)
    {
      constexpr int d = 4;
      if (!arg.doublet) {
        fermion_t in = arg.in[src_idx](x_cb, parity);
        arg.out[src_idx](x_cb, parity) = arg.a * (in + arg.b * in.igamma(d));
      } else {
        fermion_t in_1 = arg.in[src_idx](x_cb + 0 * arg.volumeCB, parity);
        fermion_t in_2 = arg.in[src_idx](x_cb + 1 * arg.volumeCB, parity);
        arg.out[src_idx](x_cb + 0 * arg.volumeCB, parity) = arg.a * (in_1 + arg.b * in_1.igamma(d) + arg.c * in_2);
        arg.out[src_idx](x_cb + 1 * arg.volumeCB, parity) = arg.a * (in_2 - arg.b * in_2.igamma(d) + arg.c * in_1);
      }
    }
  };

  /**
     @brief Parameter structure for driving the Tau operator
   */
  template <typename Float, int nColor_> struct TauArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    constexpr static int nColor = nColor_;
    typedef typename colorspinor_mapper<Float, 4, nColor>::type F;

    static constexpr unsigned int max_n_src = MAX_MULTI_RHS;
    F out[max_n_src];      // output vector field
    F in[max_n_src];      // input vector field
    const int d;        // which gamma matrix are we applying
    const int nParity;  // number of parities we're working on
    bool doublet;       // whether we applying the operator to a doublet
    const int volumeCB; // checkerboarded volume

    TauArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int d) :
      d(d),
      nParity(in.SiteSubset()),
      doublet(in.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
      volumeCB(doublet ? in.VolumeCB() / 2 : in.VolumeCB())
    {
      if (out.size() > max_n_src) errorQuda("vector set size %lu greater than max size %d", out.size(), max_n_src);
      for (auto i = 0u; i < in.size(); i++) {
        this->in[i] = in[i];
        this->out[i] = out[i];
      }

      checkPrecision(out, in);
      checkLocation(out, in);
      if (d < 1 || d > 3) errorQuda("Undefined tau matrix %d", d);
      if (in.Nspin() != 4) errorQuda("Cannot apply tau to nSpin=%d field", in.Nspin());
      if (!in.isNative() || !out.isNative())
        errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());
      if (!doublet) errorQuda("tau matrix can be applyed only to spinor doublet");

      this->threads = dim3(doublet ? in.VolumeCB() / 2 : in.VolumeCB(), in.SiteSubset(), 1);
    }
  };
  /**
     @brief Application of Gamma matrix to a color spinor field
  */
  template <typename Arg> struct Tau {
    using fermion_t = ColorSpinor<typename Arg::real, Arg::nColor, 4>;
    const Arg &arg;
    constexpr Tau(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int src_idx, int parity)
    {
      fermion_t in_1 = arg.in[src_idx](x_cb + 0 * arg.volumeCB, parity);
      fermion_t in_2 = arg.in[src_idx](x_cb + 1 * arg.volumeCB, parity);
      const complex<typename Arg::real> j(0.0, 1.0);
      const typename Arg::real m1(-1);

      switch (arg.d) {
      case 1:
        arg.out[src_idx](x_cb + 0 * arg.volumeCB, parity) = in_2;
        arg.out[src_idx](x_cb + 1 * arg.volumeCB, parity) = in_1;
        break;
      case 2:
        arg.out[src_idx](x_cb + 0 * arg.volumeCB, parity) = -j * in_2;
        arg.out[src_idx](x_cb + 1 * arg.volumeCB, parity) = j * in_1;
        break;
      case 3:
        arg.out[src_idx](x_cb + 0 * arg.volumeCB, parity) = in_1;
        arg.out[src_idx](x_cb + 1 * arg.volumeCB, parity) = m1 * in_2;
        break;
      }
    }
  };
}
