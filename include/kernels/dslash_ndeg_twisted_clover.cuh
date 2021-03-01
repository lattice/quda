#pragma once

#include <kernels/dslash_wilson.cuh>
#include <clover_field_order.h>

namespace quda
{

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_>
    struct NdegTwistedCloverArg : WilsonArg<Float, nColor, nDim, reconstruct_> {

    typedef typename clover_mapper<Float, length, true>::type C;
    typedef typename mapper<Float>::type real;

    const C A; /** the clover field */
    real a; /** this is the Wilson-dslash scale factor */
    real b; /** this is the chiral twist factor */
    real c; /** this is the flavor twist factor */
    
  NdegTwistedCloverArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                       const CloverField &A, double a, double b,
                       double c, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
    WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, U, C, a, x, parity, dagger, comm_override),
      A(A, false),
      a(a),
      // if dagger flip the chiral twist
      b(dagger ? -0.5 * b : 0.5 * b) // factor of 1/2 comes from clover normalization we need to correct for, do we?
      c(0.5*c)
      {
        checkPrecision(U, A);
        checkLocation(U, A);
      }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct nDegTwistedClover : dslash_default {

    Arg &arg;
    constexpr nDegTwistedClover(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    /**
       @brief Apply the non-degenerate twisted-clover dslash
       out(x) = M*in = A(x)*in(x) + a * D * in + (1 + i*b*gamma_5*tau_3 + c*tau_1)*in
       Note this routine only exists in xpay form.
    */
    __device__ __host__ inline void operator()(int idx, int flavor, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;
      typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;

      bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, kernel_type>(arg, idx, flavor, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;

      // defined in dslash_wilson.cuh
      applyWilson<nParity, dagger, kernel_type>(out, arg, coord, parity, idx, thread_dim, active);

      int my_flavor_idx = coord.x_cb + flavor * arg.dc.volume_4d_cb;

      if (kernel_type == INTERIOR_KERNEL) {
        // apply the chiral and flavor twists
        // use consistent load order across s to ensure better cache locality
        Vector x0 = arg.x(coord.x_cb + 0 * arg.dc.volume_4d_cb, my_spinor_parity);
        Vector x1 = arg.x(coord.x_cb + 1 * arg.dc.volume_4d_cb, my_spinor_parity);

        x0.toRel(); // switch to chiral basis
        x1.toRel(); // switch to chiral basis

        Vector tmp;
        if(flavor == 0) {
#pragma unroll
          for (int chirality = 0; chirality < 2; chirality++) {
            constexpr int n = Arg::nColor * Arg::nSpin / 2;
            HMatrix<real, n> A = arg.A(coord.x_cb, parity, chirality);
            HalfVector x0_chi = x0.chiral_project(chirality);
            HalfVector Ax0_chi = A * x0_chi;
            // i * mu * gamma_5 * tau_3
            const complex<real> b(0.0, chirality == 0 ? static_cast<real>(arg.b) : -static_cast<real>(arg.b));
            Ax0_chi += b * x0_chi;
            Ax0_chi += arg.c * x1_chi;
            tmp += Ax0_chi.chiral_reconstruct(chirality);
          }
        } else {
#pragma unroll
          for (int chirality = 0; chirality < 2; chirality++) {
            constexpr int n = Arg::nColor * Arg::nSpin / 2;
            HMatrix<real, n> A = arg.A(coord.x_cb, parity, chirality);
            HalfVector x1_chi = x1.chiral_project(chirality);
            HalfVector Ax1_chi = A * x1_chi;
            // i * mu * gamma_5 * tau_3
            const complex<real> b(0.0, chirality == 0 ? -static_cast<real>(arg.b) : static_cast<real>(arg.b));
            Ax1_chi += b * x1_chi;
            Ax1_chi += arg.c * x0_chi;
            tmp += Ax1_chi.chiral_reconstruct(chirality);
          }
        }
        tmp.toNonRel(); // switch back to non-chiral basis
        out = tmp + arg.a * out;

      } else if (active) {
        Vector x = arg.out(my_flavor_idx, my_spinor_parity);
        out = x + arg.a * out;
      }

      if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(my_flavor_idx, my_spinor_parity) = out;
    }
  };

} // namespace quda
