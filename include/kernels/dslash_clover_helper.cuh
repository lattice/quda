#pragma once

#include "color_spinor_field_order.h"
#include "clover_field_order.h"
#include "color_spinor.h"
#include "linalg.cuh"
#include "kernel.h"

namespace quda {

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
