#pragma once

#include <color_spinor_field_order.h>
#include <clover_field_order.h>
#include <index_helper.cuh>
#include <color_spinor.h>
#include <linalg.cuh>
#include <kernel.h>

namespace quda {

  /**
     @brief Parameter structure for driving init_dslash_atomic
  */
  template <typename T_> struct init_dslash_atomic_arg : kernel_param<> {
    using T = T_;
    T *count;

    init_dslash_atomic_arg(T *count, unsigned int size) :
      kernel_param(dim3(size, 1, 1)),
      count(count) { }
  };

  /**
     @brief Functor that uses placement new constructor to initialize
     the atomic counters
  */
  template <typename Arg> struct init_dslash_atomic {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr init_dslash_atomic(const Arg &arg) : arg(arg) { }
    __device__ void operator()(int i) { new (arg.count + i) typename Arg::T {0}; }
  };

  /**
     @brief Parameter structure for driving init_dslash_arr
  */
  template <typename T_> struct init_arr_arg : kernel_param<> {
    using T = T_;
    T *arr;
    T val;

    init_arr_arg(T *arr, T val, unsigned int size) :
      kernel_param(dim3(size, 1, 1)),
      arr(arr),
      val(val) { }
  };

  /**
     @brief Functor to initialize the arrive signal
  */
  template <typename Arg> struct init_sync_arr {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr init_sync_arr(const Arg &arg) : arg(arg) { }
    __device__ void operator()(int i) { *(arg.arr + i) = arg.val; }
  };

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
      doublet(in.TwistFlavor() == QUDA_TWIST_DEG_DOUBLET || in.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
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

  /**
     @brief Parameteter structure for driving the clover and twist-clover application kernels
     @tparam Float Underlying storage precision
     @tparam nSpin Number of spin components
     @tparam nColor Number of colors
     @tparam dynamic_clover Whether we are inverting the clover field on the fly
  */
  template <typename Float, int nColor_, bool inverse_ = true>
  struct CloverArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nSpin = 4;
    static constexpr int nColor = nColor_;
    static constexpr int length = (nSpin / (nSpin/2)) * 2 * nColor * nColor * (nSpin/2) * (nSpin/2) / 2;
    static constexpr bool inverse = inverse_;
    static constexpr bool dynamic_clover = dynamic_clover_inverse();

    typedef typename colorspinor_mapper<Float,nSpin,nColor>::type F;
    typedef typename clover_mapper<Float,length>::type C;

    F out;                // output vector field
    const F in;           // input vector field
    const C clover;       // clover field
    const C cloverInv;    // inverse clover field (only set if not dynamic clover and doing twisted clover)
    const int nParity;    // number of parities we're working on
    const int parity;     // which parity we're acting on (if nParity=1)
    bool doublet;         // whether we applying the operator to a doublet
    const int volumeCB;   // checkerboarded volume
    real a;
    real b;
    real c;
    QudaTwistGamma5Type twist;

    CloverArg(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
	      int parity, real kappa=0.0, real mu=0.0, real /*epsilon*/ = 0.0,
	      bool dagger = false, QudaTwistGamma5Type twist=QUDA_TWIST_GAMMA5_INVALID) :
      out(out), in(in), clover(clover, twist == QUDA_TWIST_GAMMA5_INVALID ? inverse : false),
      cloverInv(clover, (twist != QUDA_TWIST_GAMMA5_INVALID && !dynamic_clover) ? true : false),
      nParity(in.SiteSubset()), parity(parity),
      doublet(in.TwistFlavor() == QUDA_TWIST_DEG_DOUBLET || in.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
      volumeCB(doublet ? in.VolumeCB()/2 : in.VolumeCB()), a(0.0), b(0.0), c(0.0), twist(twist)
    {
      checkPrecision(out, in, clover);
      checkLocation(out, in, clover);
      if (in.TwistFlavor() == QUDA_TWIST_SINGLET) {
	if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
	  a = 2.0 * kappa * mu;
	  b = 1.0;
	} else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
	  a = -2.0 * kappa * mu;
	  b = 1.0 / (1.0 + a*a);
	}
	c = 0.0;
	if (dagger) a *= -1.0;
      } else if (doublet) {
	errorQuda("ERROR: Non-degenerated twisted-mass not supported in this regularization\n");
      }
      this->threads = dim3(doublet ? in.VolumeCB()/2 : in.VolumeCB(), in.SiteSubset(), 1);
    }
  };

  template <typename Arg> struct CloverApply {
    static constexpr int N = Arg::nColor * Arg::nSpin / 2;
    using real = typename Arg::real;
    using fermion = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
    using half_fermion = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin / 2>;
    const Arg &arg;
    constexpr CloverApply(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using namespace linalg; // for Cholesky
      int clover_parity = arg.nParity == 2 ? parity : arg.parity;
      int spinor_parity = arg.nParity == 2 ? parity : 0;
      fermion in = arg.in(x_cb, spinor_parity);
      fermion out;

      in.toRel(); // change to chiral basis here

#pragma unroll
      for (int chirality=0; chirality<2; chirality++) {
        HMatrix<real, N> A = arg.clover(x_cb, clover_parity, chirality);
        half_fermion chi = in.chiral_project(chirality);

        if (arg.dynamic_clover) {
          Cholesky<HMatrix, real, N> cholesky(A);
          chi = static_cast<real>(0.25) * cholesky.backward(cholesky.forward(chi));
        } else {
          chi = A * chi;
        }

        out += chi.chiral_reconstruct(chirality);
      }

      out.toNonRel(); // change basis back
      arg.out(x_cb, spinor_parity) = out;
    }
  };

  // if (!inverse) apply (Clover + i*a*gamma_5) to the input spinor
  // else apply (Clover + i*a*gamma_5)/(Clover^2 + a^2) to the input spinor
  template <typename Arg> struct TwistCloverApply {
    static constexpr int N = Arg::nColor * Arg::nSpin / 2;
    using real = typename Arg::real;
    using fermion = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
    using half_fermion = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin / 2>;
    using Mat = HMatrix<typename Arg::real, N>;
    const Arg &arg;
    constexpr TwistCloverApply(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      using namespace linalg; // for Cholesky
      int clover_parity = arg.nParity == 2 ? parity : arg.parity;
      int spinor_parity = arg.nParity == 2 ? parity : 0;
      fermion in = arg.in(x_cb, spinor_parity);
      fermion out;

      in.toRel(); // change to chiral basis here

#pragma unroll
      for (int chirality=0; chirality<2; chirality++) {
        // factor of 2 comes from clover normalization we need to correct for
        const complex<real> j(0.0, chirality == 0 ? static_cast<real>(0.5) : -static_cast<real>(0.5));

        Mat A = arg.clover(x_cb, clover_parity, chirality);

        half_fermion in_chi = in.chiral_project(chirality);
        half_fermion out_chi = A*in_chi + j*arg.a*in_chi;

        if (arg.inverse) {
          if (arg.dynamic_clover) {
            Mat A2 = A.square();
            A2 += arg.a*arg.a*static_cast<real>(0.25);
            Cholesky<HMatrix, real, N> cholesky(A2);
            out_chi = static_cast<real>(0.25)*cholesky.backward(cholesky.forward(out_chi));
          } else {
            Mat Ainv = arg.cloverInv(x_cb, clover_parity, chirality);
            out_chi = static_cast<real>(2.0)*(Ainv*out_chi);
          }
        }

        out += (out_chi).chiral_reconstruct(chirality);
      }

      out.toNonRel(); // change basis back
      arg.out(x_cb, spinor_parity) = out;
    }
  };

}
