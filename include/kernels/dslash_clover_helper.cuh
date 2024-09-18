#pragma once

#include <color_spinor_field_order.h>
#include <clover_field_order.h>
#include "color_spinor.h"
#include <linalg.cuh>
#include "shared_memory_cache_helper.h"
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
    using store_t = Float;
    using real = typename mapper<Float>::type;
    static constexpr int nSpin = 4;
    static constexpr int nColor = nColor_;
    static constexpr int length = (nSpin / (nSpin/2)) * 2 * nColor * nColor * (nSpin/2) * (nSpin/2) / 2;
    static constexpr bool inverse = inverse_;
    static constexpr bool dynamic_clover = clover::dynamic_inverse();

    typedef typename colorspinor_mapper<Float, nSpin, nColor, false, false, true>::type F;
    typedef typename clover_mapper<Float, length>::type C;

    F out[MAX_MULTI_RHS]; // output vector field
    F in[MAX_MULTI_RHS];  // input vector field
    const C clover;       // clover field
    const C cloverInv;    // inverse clover field (only set if not dynamic clover and doing twisted clover)
    const int nParity;    // number of parities we're working on
    const int parity;     // which parity we're acting on (if nParity=1)
    bool doublet;         // whether we applying the operator to a doublet
    const int volumeCB;   // checkerboarded volume
    real a;
    real b;
    real a2_minus_b2;
    QudaTwistGamma5Type twist;

    CloverArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const CloverField &clover,
              int parity, real kappa = 0.0, real mu = 0.0, real epsilon = 0.0, bool dagger = false,
              QudaTwistGamma5Type twist = QUDA_TWIST_GAMMA5_INVALID) :
      kernel_param(dim3(in.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET ? in.VolumeCB() / 2 : in.VolumeCB(),
                        in.size() * (in.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET ? 2 : 1), in.SiteSubset())),
      clover(clover,
             inverse && !dynamic_clover
               && twist == QUDA_TWIST_GAMMA5_INVALID), // only inverse if non-twisted clover and !dynamic
      cloverInv(clover, !dynamic_clover),              // only inverse if !dynamic
      nParity(in.SiteSubset()),
      parity(parity),
      doublet(in.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
      volumeCB(doublet ? in.VolumeCB() / 2 : in.VolumeCB()),
      a(0.0),
      b(0.0),
      twist(twist)
    {
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->in[i] = in[i];
      }
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
        if (dagger) a *= -1.0;
      } else if (doublet) {
        if (twist == QUDA_TWIST_GAMMA5_DIRECT){
          a = 2.0 * kappa * mu;
          b = -2.0 * kappa * epsilon;
        } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
          a = -2.0 * kappa * mu;
          b = 2.0 * kappa * epsilon;
        }
        if (dagger) a *= -1.0;
      }
      // factor of 2 comes from clover normalization we need to correct for
      a *= 0.5;
      b *= 0.5;
      a2_minus_b2 = a * a - b * b;
    }
  };

  template <typename Arg> struct CloverApply {
    static constexpr int N = Arg::nColor * Arg::nSpin / 2;
    using store_t = typename Arg::store_t;
    using real = typename Arg::real;
    using fermion = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
    using half_fermion = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin / 2>;
    const Arg &arg;
    constexpr CloverApply(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int src_idx, int parity)
    {
      using namespace linalg; // for Cholesky
      int clover_parity = arg.nParity == 2 ? parity : arg.parity;
      int spinor_parity = arg.nParity == 2 ? parity : 0;
      fermion in = arg.in[src_idx](x_cb, spinor_parity);
      fermion out;

      in.toRel(); // change to chiral basis here

#pragma unroll
      for (int chirality=0; chirality<2; chirality++) {
        HMatrix<real, N> A = arg.clover(x_cb, clover_parity, chirality);
        half_fermion chi = in.chiral_project(chirality);

        if (arg.dynamic_clover && arg.inverse) {
          Cholesky<HMatrix, clover::cholesky_t<store_t>, N> cholesky(A);
          chi = static_cast<real>(0.25) * cholesky.solve(chi);
        } else {
          chi = A * chi;
        }

        out += chi.chiral_reconstruct(chirality);
      }

      out.toNonRel(); // change basis back
      arg.out[src_idx](x_cb, spinor_parity) = out;
    }
  };

  // if (!inverse) apply (Clover + i*a*gamma_5) to the input spinor
  // else apply (Clover + i*a*gamma_5)/(Clover^2 + a^2) to the input spinor
  template <typename Arg> struct TwistCloverApply {
    static constexpr int N = Arg::nColor * Arg::nSpin / 2;
    using store_t = typename Arg::store_t;
    using real = typename Arg::real;
    using fermion = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
    using half_fermion = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin / 2>;
    using Mat = HMatrix<typename Arg::real, N>;
    const Arg &arg;
    constexpr TwistCloverApply(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int src_idx, int parity)
    {
      using namespace linalg; // for Cholesky
      int clover_parity = arg.nParity == 2 ? parity : arg.parity;
      int spinor_parity = arg.nParity == 2 ? parity : 0;
      fermion in = arg.in[src_idx](x_cb, spinor_parity);
      fermion out;

      in.toRel(); // change to chiral basis here

#pragma unroll
      for (int chirality=0; chirality<2; chirality++) {
        const complex<real> a(0.0, chirality == 0 ? arg.a : -arg.a);
        Mat A = arg.clover(x_cb, clover_parity, chirality);

        half_fermion in_chi = in.chiral_project(chirality);
        half_fermion out_chi = A*in_chi + a * in_chi;

        if (arg.inverse) {
          if (arg.dynamic_clover) {
            Mat A2 = A.square();
            A2 += arg.a * arg.a;
            Cholesky<HMatrix, clover::cholesky_t<store_t>, N> cholesky(A2);
            out_chi = static_cast<real>(0.25)*cholesky.solve(out_chi);
          } else {
            Mat Ainv = arg.cloverInv(x_cb, clover_parity, chirality);
            out_chi = static_cast<real>(2.0)*(Ainv*out_chi);
          }
        }

        out += (out_chi).chiral_reconstruct(chirality);
      }

      out.toNonRel(); // change basis back
      arg.out[src_idx](x_cb, spinor_parity) = out;
    }
  };

  template <typename Arg> using NdegTwistCloverApplyOps =
    KernelOps<SharedMemoryCache<ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin / 2>>>;

  // if (!inverse) apply (Clover + i*a*gamma_5*tau_3 + b*epsilon*tau_1) to the input spinor
  // else apply (Clover + i*a*gamma_5*tau_3 + b*epsilon*tau_1)/(Clover^2 + a^2 - b^2) to the input spinor
  // noting that appropriate signs are carried by a and b depending on inverse
  template <typename Arg> struct NdegTwistCloverApply : NdegTwistCloverApplyOps<Arg> {
    static constexpr int N = Arg::nColor * Arg::nSpin / 2;
    using real = typename Arg::real;
    using fermion = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
    using half_fermion = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin / 2>;
    using Mat = HMatrix<typename Arg::real, N>;
    const Arg &arg;
    using typename NdegTwistCloverApplyOps<Arg>::KernelOpsT;
    template <typename ...Ops>
    constexpr NdegTwistCloverApply(const Arg &arg, const Ops &...ops) : KernelOpsT(ops...), arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    template <bool allthreads = false>
    __device__ __host__ inline void operator()(int x_cb, int src_flavor, int parity, bool active = true)
    {
      using namespace linalg; // for Cholesky
      const int clover_parity = arg.nParity == 2 ? parity : arg.parity;
      const int spinor_parity = arg.nParity == 2 ? parity : 0;
      constexpr int n_flavor = 2;

      const int src_idx = src_flavor / 2;
      const int flavor = src_flavor % 2;

      int my_flavor_idx = x_cb + flavor * arg.volumeCB;
      fermion in;
      int chirality = flavor; // relabel flavor as chirality
      Mat A;
      if (!allthreads || active) {
	in = arg.in[src_idx](my_flavor_idx, spinor_parity);
	in.toRel(); // change to chiral basis here
	A = arg.clover(x_cb, clover_parity, chirality);
      } else {
	in = fermion{};
	A = Mat{};
      }

      // (C + i mu gamma_5 tau_3 - epsilon tau_1 )  [note: appropriate signs carried in arg.a / arg.b]
      const complex<real> a(0.0, chirality == 0 ? arg.a : -arg.a);

      SharedMemoryCache<half_fermion> cache{*this};

      half_fermion in_chi[n_flavor]; // flavor array of chirally projected fermion
#pragma unroll
      for (int i = 0; i < n_flavor; i++) in_chi[i] = in.chiral_project(i);

      auto swizzle = [&](half_fermion x[2], int chirality) {
	if (chirality == 0)
	  cache.save_y(x[1], target::thread_idx().y);
	else
	  cache.save_y(x[0], target::thread_idx().y);
        cache.sync();
	if (chirality == 0)
	  x[1] = cache.load_y(target::thread_idx().y + 1);
	else
	  x[0] = cache.load_y(target::thread_idx().y - 1);
      };

      swizzle(in_chi, chirality); // apply the flavor-chirality swizzle between threads

      half_fermion out_chi[n_flavor];
#pragma unroll
      for (int flavor = 0; flavor < n_flavor; flavor++) {
        out_chi[flavor] = A * in_chi[flavor];
        out_chi[flavor] += (flavor == 0 ? a : -a) * in_chi[flavor];
        out_chi[flavor] += arg.b * in_chi[1 - flavor];
      }

      if (!allthreads || active) {
	if (arg.inverse) {
	  if (arg.dynamic_clover) {
	    Mat A2 = A.square();
	    A2 += arg.a2_minus_b2;
	    Cholesky<HMatrix, clover::cholesky_t<real>, N> cholesky(A2);
#pragma unroll
	    for (int flavor = 0; flavor < n_flavor; flavor++)
	      out_chi[flavor] = static_cast<real>(0.25) * cholesky.backward(cholesky.forward(out_chi[flavor]));
	  } else {
	    Mat Ainv = arg.cloverInv(x_cb, clover_parity, chirality);
#pragma unroll
	    for (int flavor = 0; flavor < n_flavor; flavor++)
	      out_chi[flavor] = static_cast<real>(2.0) * (Ainv * out_chi[flavor]);
	  }
	}
      }

      swizzle(out_chi, chirality); // undo the flavor-chirality swizzle

      if (!allthreads || active) {
	fermion out = out_chi[0].chiral_reconstruct(0) + out_chi[1].chiral_reconstruct(1);
	out.toNonRel(); // change basis back

	arg.out[src_idx](my_flavor_idx, spinor_parity) = out;
      }
    }
  };

}
