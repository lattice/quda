#pragma once

#include <kernels/dslash_wilson_clover_preconditioned.cuh>
#include <clover_field_order.h>
#include <linalg.cuh>

namespace quda
{

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_>
  struct TwistedCloverArg : WilsonArg<Float, nColor, nDim, reconstruct_> {
    using WilsonArg<Float, nColor, nDim, reconstruct_>::nSpin;
    static constexpr int length = (nSpin / (nSpin / 2)) * 2 * nColor * nColor * (nSpin / 2) * (nSpin / 2) / 2;
    static constexpr bool dynamic_clover = clover::dynamic_inverse();

    typedef typename mapper<Float>::type real;
    typedef typename clover_mapper<Float, length>::type C;
    const C A;
    const C A2inv; // A^{-2}
    real a;        // this is the scaling factor
    real b;        // this is the twist factor
    real b2;

    TwistedCloverArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
                     double a, double b, bool xpay, const ColorSpinorField &x, int parity, bool dagger,
                     const int *comm_override) :
      WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, U, xpay ? 1.0 : 0.0, x, parity, dagger, comm_override),
      A(A, false),
      A2inv(A, dynamic_clover ? false : true), // if dynamic clover we don't want the inverse field
      a(a),
      b(dagger ? -0.5 * b : 0.5 * b), // factor of 0.5 comes from basis transform
      b2(0.25 * b * b)
    {
      checkPrecision(U, A);
      checkLocation(U, A);
    }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct twistedCloverPreconditioned : dslash_default {

    const Arg &arg;
    constexpr twistedCloverPreconditioned(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    /**
       @brief Apply the preconditioned twisted-clover dslash
       - no xpay: out(x) = M*in = A(x)^{-1}D * in(x-mu)
       - with xpay:  out(x) = M*in = (1 + a*A(x)^{-1}D) * in(x-mu)
    */
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int, int parity)
    {
      using namespace linalg; // for Cholesky
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;
      typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
      typedef HMatrix<real, Arg::nColor * Arg::nSpin / 2> Mat;

      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, 0, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;

      Vector out;

      // defined in dslash_wilson.cuh
      applyWilson<nParity, dagger, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active);

      if (mykernel_type != INTERIOR_KERNEL && active) {
        // if we're not the interior kernel, then we must sum the partial
        Vector x = arg.out(coord.x_cb, my_spinor_parity);
        out += x;
      }

      if (isComplete<mykernel_type>(arg, coord) && active) {
        out.toRel(); // switch to chiral basis

        Vector tmp;

#pragma unroll
        for (int chirality = 0; chirality < 2; chirality++) {

          const complex<real> b(0.0, chirality == 0 ? arg.b : -arg.b);
          Mat A = arg.A(coord.x_cb, parity, chirality);
          HalfVector chi = out.chiral_project(chirality);
          chi = A * chi + b * chi;

          if (arg.dynamic_clover) {
            Mat A2 = A.square();
            A2 += arg.b2;
            Cholesky<HMatrix, clover::cholesky_t<typename Arg::Float>, Arg::nColor * Arg::nSpin / 2> cholesky(A2);
            chi = static_cast<real>(0.25) * cholesky.solve(chi);
          } else {
            Mat A2inv = arg.A2inv(coord.x_cb, parity, chirality);
            chi = static_cast<real>(2.0) * (A2inv * chi);
          }
          tmp += chi.chiral_reconstruct(chirality);
        }

        tmp.toNonRel(); // switch back to non-chiral basis

        if (xpay) {
          Vector x = arg.x(coord.x_cb, my_spinor_parity);
          out = x + arg.a * tmp;
        } else {
          out = arg.a * tmp;
        }
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(coord.x_cb, my_spinor_parity) = out;
    }
  };

} // namespace quda
