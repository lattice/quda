#pragma once

#include <kernels/dslash_wilson.cuh>
#include <clover_field_order.h>
#include <linalg.cuh>

namespace quda
{

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_, bool clov_inv_>
  struct WilsonCloverHasenbuschTwistPCArg : WilsonArg<Float, nColor, nDim, reconstruct_> {
    using WilsonArg<Float, nColor, nDim, reconstruct_>::nSpin;
    static constexpr int length = (nSpin / (nSpin / 2)) * 2 * nColor * nColor * (nSpin / 2) * (nSpin / 2) / 2;
    static constexpr bool dynamic_clover = clover::dynamic_inverse();
    static constexpr bool clov_inv = clov_inv_;

    typedef typename clover_mapper<Float, length>::type C;
    typedef typename mapper<Float>::type real;

    const C A;
    const C A_inv;
    real b;

    WilsonCloverHasenbuschTwistPCArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                     const CloverField &A_, double a_, double b_, const ColorSpinorField &x, int parity,
                                     bool dagger, const int *comm_override) :
      WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, U, a_, x, parity, dagger, comm_override),
      A(A_, false),
      A_inv(A_, dynamic_clover ? false : true),
      b(dagger ? -0.5 * b_ : 0.5 * b_) // if dynamic clover we don't want the inverse field
    {
      checkPrecision(U, A_);
      checkLocation(U, A_);
    }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct cloverHasenbuschPreconditioned : dslash_default {

    const Arg &arg;
    constexpr cloverHasenbuschPreconditioned(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    /**
       @brief Apply the clover preconditioned Wilson dslash
       - no xpay: out(x) = M*in = A(x)^{-1}D * in(x-mu)
       - with xpay:  out(x) = M*in = (1 - a*A(x)^{-1}D) * in(x-mu)
    */
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int, int parity)
    {
      using namespace linalg; // for Cholesky
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;
      typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;

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

        if (!Arg::clov_inv) {
          Vector x = arg.x(coord.x_cb, my_spinor_parity);
          out = x + arg.a * out;
        } else {
          out.toRel(); // switch to chiral basis

          Vector tmp;

#pragma unroll
          for (int chirality = 0; chirality < 2; chirality++) {

            HMatrix<real, Arg::nColor * Arg::nSpin / 2> A_inv = arg.A_inv(coord.x_cb, parity, chirality);
            HalfVector chi = out.chiral_project(chirality);

            if (arg.dynamic_clover) {
              Cholesky<HMatrix, clover::cholesky_t<typename Arg::Float>, Arg::nColor * Arg::nSpin / 2> cholesky(A_inv);
              chi = static_cast<real>(0.25) * cholesky.solve(chi);
            } else {
              chi = A_inv * chi;
            }

            tmp += chi.chiral_reconstruct(chirality);
          }

          tmp.toNonRel(); // switch back to non-chiral basis
          Vector x = arg.x(coord.x_cb, my_spinor_parity);
          out = x + arg.a * tmp;
        }

        // At this point: out = x + k A^{-1} D in or out = x + k D in
        //
        // now we must add on i g_5 b A x
        {
          Vector x = arg.x(coord.x_cb, my_spinor_parity);
          x.toRel();
          Vector tmp;
#pragma unroll
          for (int chirality = 0; chirality < 2; chirality++) {

            HMatrix<real, Arg::nColor *Arg::nSpin / 2> A = arg.A(coord.x_cb, parity, chirality);
            HalfVector x_chi = x.chiral_project(chirality);
            HalfVector Ax_chi = A * x_chi;
            const complex<real> b(0.0, chirality == 0 ? static_cast<real>(arg.b) : -static_cast<real>(arg.b));
            x_chi = b * Ax_chi;
            tmp += x_chi.chiral_reconstruct(chirality);
          }
          tmp.toNonRel(); // switch back to non-chiral basis
          out += tmp;
        }
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(coord.x_cb, my_spinor_parity) = out;
    }
  };

} // namespace quda
