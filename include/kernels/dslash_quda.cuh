#pragma once

#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <matrix_field.h>
#include <su3_project.cuh>

namespace quda
{
  
  /**
     @brief Parameter structure for driving the Gamma and chiral projection operator
  */
  template <typename Float, int nColor>
  struct GammaArg {
    typedef typename colorspinor_mapper<Float,4,nColor>::type F;
    typedef typename mapper<Float>::type RegType;

    F out;                // output vector field
    const F in;           // input vector field
    const int d;          // which gamma matrix are we applying
    const int proj;       // performs L(-1) or R(+1) chiral projection
    const int nParity;    // number of parities we're working on
    bool doublet;         // whether we applying the operator to a doublet
    const int volumeCB;   // checkerboarded volume
    RegType a;            // scale factor
    RegType b;            // chiral twist
    RegType c;            // flavor twist

    GammaArg(ColorSpinorField &out, const ColorSpinorField &in, int d, int proj = 0,
	     RegType kappa=0.0, RegType mu=0.0, RegType epsilon=0.0,
	     bool dagger=false, QudaTwistGamma5Type twist=QUDA_TWIST_GAMMA5_INVALID)
      : out(out), in(in), d(d), proj(proj), nParity(in.SiteSubset()),
	doublet(in.TwistFlavor() == QUDA_TWIST_DEG_DOUBLET || in.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
	volumeCB(doublet ? in.VolumeCB()/2 : in.VolumeCB()), a(0.0), b(0.0), c(0.0)
    {
      checkPrecision(out, in);
      checkLocation(out, in);
      if (d < 0 || d > 4) errorQuda("Undefined gamma matrix %d", d);
      if (proj != -1 && proj != 0 && proj != 1) errorQuda("Undefined gamma projection %d", proj);
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
    }
  };
  
  // GPU Kernel for applying the gamma matrix to a colorspinor
  template <typename Float, int nColor, int d, typename Arg>
  __global__ void gammaGPU(Arg arg)
  {
    typedef typename mapper<Float>::type RegType;
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (x_cb >= arg.volumeCB) return;
    if (parity >= arg.nParity) return;

    ColorSpinor<RegType,nColor,4> in = arg.in(x_cb, parity);
    arg.out(x_cb, parity) = in.gamma(d);
  }

  // GPU Kernel for applying a chiral projection to a colorspinor
  template <typename Float, int nColor, typename Arg>
  __global__ void chiralProjGPU(Arg arg)
  {
    typedef typename mapper<Float>::type RegType;
    typedef ColorSpinor<RegType, nColor, 4> Spinor;
    typedef ColorSpinor<RegType, nColor, 2> HalfSpinor;

    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (x_cb >= arg.volumeCB) return;
    if (parity >= arg.nParity) return;
    
    // arg.proj is either +1 or -1
    // chiral_project() expects +1 or 0
    int proj = arg.proj == 1 ? 1 : 0;
    
    Spinor in = arg.in(x_cb, parity);
    Spinor out = arg.out(x_cb, parity);
    HalfSpinor chi;
    
    // out += P_{L/R} * in
    chi = in.chiral_project(proj);
    out += 0.5*chi.chiral_reconstruct(proj);
    
    arg.out(x_cb, parity) = 0.5*out;
  }
  
  
  // GPU Kernel for applying the gamma matrix to a colorspinor
  template <bool doublet, typename Float, int nColor, int d, typename Arg>
  __global__ void twistGammaGPU(Arg arg)
  {
    typedef typename mapper<Float>::type RegType;
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    if (x_cb >= arg.volumeCB) return;
    
    if (!doublet) {
      ColorSpinor<RegType,nColor,4> in = arg.in(x_cb, parity);
      arg.out(x_cb, parity) = arg.a * (in + arg.b * in.igamma(d));
    } else {
      ColorSpinor<RegType,nColor,4> in_1 = arg.in(x_cb+0*arg.volumeCB, parity);
      ColorSpinor<RegType,nColor,4> in_2 = arg.in(x_cb+1*arg.volumeCB, parity);
      arg.out(x_cb + 0 * arg.volumeCB, parity) = arg.a * (in_1 + arg.b * in_1.igamma(d) + arg.c * in_2);
      arg.out(x_cb + 1 * arg.volumeCB, parity) = arg.a * (in_2 - arg.b * in_2.igamma(d) + arg.c * in_1);
    }
  }

  /**
     @brief Parameteter structure for driving the clover and twist-clover application kernels
     @tparam Float Underlying storage precision
     @tparam nSpin Number of spin components
     @tparam nColor Number of colors
     @tparam dynamic_clover Whether we are inverting the clover field on the fly
  */
  template <typename Float, int nSpin, int nColor>
  struct CloverArg {
    static constexpr int length = (nSpin / (nSpin/2)) * 2 * nColor * nColor * (nSpin/2) * (nSpin/2) / 2;
    static constexpr bool dynamic_clover = dynamic_clover_inverse();

    typedef typename colorspinor_mapper<Float,nSpin,nColor>::type F;
    typedef typename clover_mapper<Float,length>::type C;
    typedef typename mapper<Float>::type RegType;

    F out;                // output vector field
    const F in;           // input vector field
    const C clover;       // clover field
    const C cloverInv;    // inverse clover field (only set if not dynamic clover and doing twisted clover)
    const int nParity;    // number of parities we're working on
    const int parity;     // which parity we're acting on (if nParity=1)
    bool inverse;         // whether we are applying the inverse
    bool doublet;         // whether we applying the operator to a doublet
    const int volumeCB;   // checkerboarded volume
    RegType a;
    RegType b;
    RegType c;
    QudaTwistGamma5Type twist;

    CloverArg(ColorSpinorField &out, const ColorSpinorField &in, const CloverField &clover,
	      bool inverse, int parity, RegType kappa=0.0, RegType mu=0.0, RegType epsilon=0.0,
	      bool dagger = false, QudaTwistGamma5Type twist=QUDA_TWIST_GAMMA5_INVALID)
      : out(out), clover(clover, twist == QUDA_TWIST_GAMMA5_INVALID ? inverse : false),
	cloverInv(clover, (twist != QUDA_TWIST_GAMMA5_INVALID && !dynamic_clover) ? true : false),
	in(in), nParity(in.SiteSubset()), parity(parity), inverse(inverse),
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
    }
  };

  template <typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ inline void cloverApply(Arg &arg, int x_cb, int parity) {
    using namespace linalg; // for Cholesky
    typedef typename mapper<Float>::type RegType;
    typedef ColorSpinor<RegType, nColor, nSpin> Spinor;
    typedef ColorSpinor<RegType, nColor, nSpin / 2> HalfSpinor;
    int spinor_parity = arg.nParity == 2 ? parity : 0;
    Spinor in = arg.in(x_cb, spinor_parity);
    Spinor out;

    in.toRel(); // change to chiral basis here

#pragma unroll
    for (int chirality=0; chirality<2; chirality++) {

      HMatrix<RegType,nColor*nSpin/2> A = arg.clover(x_cb, parity, chirality);
      HalfSpinor chi = in.chiral_project(chirality);

      if (arg.dynamic_clover) {
        Cholesky<HMatrix, RegType, nColor * nSpin / 2> cholesky(A);
        chi = static_cast<RegType>(0.25) * cholesky.backward(cholesky.forward(chi));
      } else {
        chi = A * chi;
      }

      out += chi.chiral_reconstruct(chirality);
    }

    out.toNonRel(); // change basis back

    arg.out(x_cb, spinor_parity) = out;
  }

  template <typename Float, int nSpin, int nColor, typename Arg>
  __global__ void cloverGPU(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = (arg.nParity == 2) ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;
    if (x_cb >= arg.volumeCB) return;
    cloverApply<Float,nSpin,nColor>(arg, x_cb, parity);
  }

  // if (!inverse) apply (Clover + i*a*gamma_5) to the input spinor
  // else apply (Clover + i*a*gamma_5)/(Clover^2 + a^2) to the input spinor
  template <bool inverse, typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ inline void twistCloverApply(Arg &arg, int x_cb, int parity) {
    using namespace linalg; // for Cholesky
    constexpr int N = nColor*nSpin/2;
    typedef typename mapper<Float>::type RegType;
    typedef ColorSpinor<RegType,nColor,nSpin> Spinor;
    typedef ColorSpinor<RegType,nColor,nSpin/2> HalfSpinor;
    typedef HMatrix<RegType,N> Mat;
    int spinor_parity = arg.nParity == 2 ? parity : 0;
    Spinor in = arg.in(x_cb, spinor_parity);
    Spinor out;

    in.toRel(); // change to chiral basis here

#pragma unroll
    for (int chirality=0; chirality<2; chirality++) {
      // factor of 2 comes from clover normalization we need to correct for
      const complex<RegType> j(0.0, chirality == 0 ? static_cast<RegType>(0.5) : -static_cast<RegType>(0.5));

      Mat A = arg.clover(x_cb, parity, chirality);

      HalfSpinor in_chi = in.chiral_project(chirality);
      HalfSpinor out_chi = A*in_chi + j*arg.a*in_chi;

      if (inverse) {
	if (arg.dynamic_clover) {
	  Mat A2 = A.square();
	  A2 += arg.a*arg.a*static_cast<RegType>(0.25);
	  Cholesky<HMatrix,RegType,N> cholesky(A2);
	  out_chi = static_cast<RegType>(0.25)*cholesky.backward(cholesky.forward(out_chi));
	} else {
	  Mat Ainv = arg.cloverInv(x_cb, parity, chirality);
	  out_chi = static_cast<RegType>(2.0)*(Ainv*out_chi);
	}
      }

      out += (out_chi).chiral_reconstruct(chirality);
    }

    out.toNonRel(); // change basis back

    arg.out(x_cb, spinor_parity) = out;
  }
}
