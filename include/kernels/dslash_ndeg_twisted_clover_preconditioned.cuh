#pragma once

#include <clover_field_order.h>
#include <kernels/dslash_wilson.cuh>
#include <shared_memory_cache_helper.h>
#include <linalg.cuh>

namespace quda
{
  
  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_>
    struct NdegTwistedCloverPreconditionedArg : WilsonArg<Float, nColor, nDim, reconstruct_> {
    using WilsonArg<Float, nColor, nDim, reconstruct_>::nSpin;
    static constexpr int length = (nSpin / (nSpin / 2)) * 2 * nColor * nColor * (nSpin / 2) * (nSpin / 2) / 2;
    static constexpr bool dynamic_clover = clover::dynamic_inverse();
    
    typedef typename mapper<Float>::type real;
    typedef typename clover_mapper<Float, length>::type C;
    const C A;
    const C A2inv; // A^{-2}
    real a;          /** this is the Wilson-dslash scale factor */
    real b;          /** this is the chiral twist factor */
    real c;          /** this is the flavor twist factor */
    real b2_minus_c2;

  NdegTwistedCloverPreconditionedArg(ColorSpinorField &out, const ColorSpinorField &in,
                                     const GaugeField &U, const CloverField &A,
                                     double a, double b, double c, bool xpay,
                                     const ColorSpinorField &x, int parity, bool dagger,
                                     const int *comm_override) :
    WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, U, xpay ? 1.0 : 0.0, x, parity, dagger, comm_override),
      A(A, false),
      A2inv(A, dynamic_clover ? false : true), // if dynamic clover we don't want the inverse field
      a(a),
      b(dagger ? -0.5 * b : 0.5 * b), // if dagger flip the chiral twist
      c(0.5*c),
      b2_minus_c2(0.25 * (b * b - c * c))
      {
        checkPrecision(U, A);
        checkLocation(U, A);
      }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
    struct nDegTwistedCloverPreconditioned : dslash_default {
    
    const Arg &arg;
    constexpr nDegTwistedCloverPreconditioned(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation
  
    /**
       @brief Apply the preconditioned twisted-clover dslash
       out(x) = M*in = a*(C + i*b*gamma_5*tau_3 + c*tau_1)/(C^2 + b^2 - c^2)*D*x ( xpay == false )
       out(x) = M*in = in + a*(C + i*b*gamma_5*tau_3 + c*tau_1)/(C^2 + b^2 - c^2)*D*x ( xpay == true )
    */
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int flavor, int parity)
    {
      using namespace linalg; // for Cholesky
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;
      typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
      typedef HMatrix<real, Arg::nColor * Arg::nSpin / 2> HMat;

      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                          // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, flavor, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;

      // defined in dslash_wilson.cuh
      applyWilson<nParity, dagger, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active);

      int my_flavor_idx = coord.x_cb + flavor * arg.dc.volume_4d_cb;

      if (mykernel_type != INTERIOR_KERNEL && active) {
        // if we're not the interior kernel, then we must sum the partial
        Vector x = arg.out(my_flavor_idx, my_spinor_parity);
        out += x;
      }

      if (isComplete<mykernel_type>(arg, coord) && active) {
        out.toRel();

        constexpr int n_flavor = 2;
        HalfVector out_chi[n_flavor]; // flavor array of chirally projected fermion
#pragma unroll
        for (int i = 0; i < n_flavor; i++) out_chi[i] = out.chiral_project(i);

        int chirality = flavor; // relabel flavor as chirality

        SharedMemoryCache<HalfVector> cache(target::block_dim());

        enum swizzle_direction {
          FORWARDS = 0,
          BACKWARDS = 1
        };

        auto swizzle = [&](HalfVector x[2], int chirality, swizzle_direction dir) {
          if (chirality == 0) cache.save_y(x[1], dir);
          else                cache.save_y(x[0], 1 - dir);
          cache.sync();
          if (chirality == 0) x[1] = cache.load_y(1 - dir);
          else                x[0] = cache.load_y(dir);
        };

        swizzle(out_chi, chirality, FORWARDS); // apply the flavor-chirality swizzle between threads

        // load in the clover matrix
        HMat A = arg.A(coord.x_cb, parity, chirality);

        HalfVector A_chi[n_flavor];
#pragma unroll
        for (int flavor_ = 0; flavor_ < n_flavor; flavor_++) {
          const complex<real> b(0.0, (chirality^flavor_) == 0 ? arg.b : -arg.b);
          A_chi[flavor_] = A * out_chi[flavor_];
          A_chi[flavor_] += b * out_chi[flavor_];
          A_chi[flavor_] += arg.c * out_chi[1 - flavor_];
        }

        if (arg.dynamic_clover) {
          HMat A2 = A.square();
          A2 += arg.b2_minus_c2;
          Cholesky<HMatrix, clover::cholesky_t<typename Arg::Float>, Arg::nColor * Arg::nSpin / 2> cholesky(A2);

#pragma unroll
          for (int flavor_ = 0; flavor_ < n_flavor; flavor_++) {
            out_chi[flavor_] = static_cast<real>(0.25) * cholesky.backward(cholesky.forward(A_chi[flavor_]));
          }
        } else {
          HMat A2inv = arg.A2inv(coord.x_cb, parity, chirality);
#pragma unroll
          for (int flavor_ = 0; flavor_ < n_flavor; flavor_++) {
            out_chi[flavor_] = static_cast<real>(2.0) * (A2inv * A_chi[flavor_]);
          }
        }

        swizzle(out_chi, chirality, BACKWARDS); // undo the flavor-chirality swizzle
        Vector tmp = out_chi[0].chiral_reconstruct(0) + out_chi[1].chiral_reconstruct(1);
        tmp.toNonRel(); // switch back to non-chiral basis

        if (xpay) {
          Vector x = arg.x(my_flavor_idx, my_spinor_parity);
          out = x + arg.a * tmp;
        } else {
          // multiplication with a needed here?
          out = arg.a * tmp;
        }
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(my_flavor_idx, my_spinor_parity) = out;
    }
  };
} // namespace quda
