#pragma once

#include <kernels/dslash_wilson.cuh>
#include <clover_field_order.h>
#include <linalg.cuh>

namespace quda
{

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_, bool twist_ = false, bool distance_pc_ = false>
  struct WilsonCloverArg : WilsonArg<Float, nColor, nDim, reconstruct_, distance_pc_> {
    using WilsonArg<Float, nColor, nDim, reconstruct_, distance_pc_>::nSpin;
    static constexpr int length = (nSpin / (nSpin / 2)) * 2 * nColor * nColor * (nSpin / 2) * (nSpin / 2) / 2;
    static constexpr bool twist = twist_;

    typedef typename clover_mapper<Float, length, true>::type C;
    typedef typename mapper<Float>::type real;

    const C A;    /** the clover field */
    const real a; /** xpay scale factor */
    const real b; /** chiral twist factor (twisted-clover only) */

    WilsonCloverArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                    const ColorSpinorField &halo, const GaugeField &U, const CloverField &A, double a, double b,
                    cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override,
                    double alpha0 = 0.0, int t0 = -1) :
      WilsonArg<Float, nColor, nDim, reconstruct_, distance_pc_>(out, in, halo, U, a, x, parity, dagger, comm_override,
                                                                 alpha0, t0),
      A(A, false),
      a(a),
      b(dagger ? -0.5 * b : 0.5 * b) // factor of 1/2 comes from clover normalization we need to correct for
    {
      checkPrecision(U, A);
      checkLocation(U, A);
    }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct wilsonClover : dslash_default {

    const Arg &arg;
    constexpr wilsonClover(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    /**
       @brief Apply the Wilson-clover dslash
       out(x) = M*in = A(x)*x(x) + D * in(x-mu)
       Note this routine only exists in xpay form.
    */
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_idx, int parity)
    {
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

      if (mykernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x[src_idx](coord.x_cb, my_spinor_parity);
        x.toRel(); // switch to chiral basis

        Vector tmp;

#pragma unroll
        for (int chirality = 0; chirality < 2; chirality++) {
          constexpr int n = Arg::nColor * Arg::nSpin / 2;
          HMatrix<real, n> A = arg.A(coord.x_cb, parity, chirality);
          HalfVector x_chi = x.chiral_project(chirality);
          HalfVector Ax_chi = A * x_chi;
          if (arg.twist) {
            const complex<real> b(0.0, chirality == 0 ? static_cast<real>(arg.b) : -static_cast<real>(arg.b));
            Ax_chi += b * x_chi;
          }
          tmp += Ax_chi.chiral_reconstruct(chirality);
        }

        tmp.toNonRel(); // switch back to non-chiral basis

        out = tmp + arg.a * out;
      } else if (active) {
        Vector x = arg.out[src_idx](coord.x_cb, my_spinor_parity);
        out = x + arg.a * out;
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](coord.x_cb, my_spinor_parity) = out;
    }
  };

} // namespace quda
