#pragma once

#include <kernels/dslash_wilson.cuh>
#include <clover_field_order.h>
#include <linalg.cuh>

namespace quda
{

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_, bool distance_pc_ = false>
  struct WilsonCloverArg : WilsonArg<Float, nColor, nDim, reconstruct_, distance_pc_> {
    using WilsonArg<Float, nColor, nDim, reconstruct_, distance_pc_>::nSpin;
    static constexpr int length = (nSpin / (nSpin / 2)) * 2 * nColor * nColor * (nSpin / 2) * (nSpin / 2) / 2;
    static constexpr bool dynamic_clover = clover::dynamic_inverse();

    typedef typename clover_mapper<Float, length>::type C;
    typedef typename mapper<Float>::type real;

    const C A;    /** the clover field */
    const real a; /** xpay scale factor */

    WilsonCloverArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                    const ColorSpinorField &halo, const GaugeField &U, const CloverField &A, double a,
                    cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override,
                    double alpha0 = 0.0, int t0 = -1) :
      WilsonArg<Float, nColor, nDim, reconstruct_, distance_pc_>(out, in, halo, U, a, x, parity, dagger, comm_override,
                                                                 alpha0, t0),
      A(A, dynamic_clover ? false : true), // if dynamic clover we don't want the inverse field
      a(a)
    {
      checkPrecision(U, A);
      checkLocation(U, A);
    }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct wilsonCloverPreconditioned : dslash_default {

    const Arg &arg;
    constexpr wilsonCloverPreconditioned(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    /**
       @brief Apply the clover preconditioned Wilson dslash
       - no xpay: out(x) = M*in = A(x)^{-1}D * in(x-mu)
       - with xpay:  out(x) = M*in = (1 - kappa*A(x)^{-1}D) * in(x-mu)
    */
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_idx, int parity)
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
      applyWilson<nParity, dagger, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active, src_idx);

      if (mykernel_type != INTERIOR_KERNEL && active) {
        // if we're not the interior kernel, then we must sum the partial
        Vector x = arg.out[src_idx](coord.x_cb, my_spinor_parity);
        out += x;
      }

      if (isComplete<mykernel_type>(arg, coord) && active) {
        out.toRel(); // switch to chiral basis

        Vector tmp;

#pragma unroll
        for (int chirality = 0; chirality < 2; chirality++) {

          HMatrix<real, Arg::nColor * Arg::nSpin / 2> A = arg.A(coord.x_cb, parity, chirality);
          HalfVector chi = out.chiral_project(chirality);

          if (arg.dynamic_clover) {
            Cholesky<HMatrix, clover::cholesky_t<typename Arg::Float>, Arg::nColor * Arg::nSpin / 2> cholesky(A);
            chi = static_cast<real>(0.25) * cholesky.solve(chi);
          } else {
            chi = A * chi;
          }

          tmp += chi.chiral_reconstruct(chirality);
        }

        tmp.toNonRel(); // switch back to non-chiral basis

        if (xpay) {
          Vector x = arg.x[src_idx](coord.x_cb, my_spinor_parity);
          out = x + arg.a * tmp;
        } else {
          out = tmp;
        }
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](coord.x_cb, my_spinor_parity) = out;
    }
  };

} // namespace quda
