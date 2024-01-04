#pragma once

#include <kernels/dslash_wilson.cuh>

namespace quda
{

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_>
  struct NdegTwistedMassArg : WilsonArg<Float, nColor, nDim, reconstruct_> {
    typedef typename mapper<Float>::type real;
    real a; /** this is the Wilson-dslash scale factor */
    real b; /** this is the chiral twist factor */
    real c; /** this is the flavor twist factor */

    NdegTwistedMassArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double b,
                       double c, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
      WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, U, a, x, parity, dagger, comm_override),
      a(a),
      b(dagger ? -b : b), // if dagger flip the chiral twist
      c(c)
    {
    }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct nDegTwistedMass : dslash_default {

    const Arg &arg;
    template <typename Ftor> constexpr nDegTwistedMass(const Ftor &ftor) : arg(ftor.arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    /**
       @brief Apply the twisted-mass dslash
       out(x) = M*in = a * D * in + (1 + i*b*gamma_5*tau_3 + c*tau_1)*x
       Note this routine only exists in xpay form.
    */
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int flavor, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, flavor, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;

      // defined in dslash_wilson.cuh
      applyWilson<nParity, dagger, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active);

      int my_flavor_idx = coord.x_cb + flavor * arg.dc.volume_4d_cb;

      if (mykernel_type == INTERIOR_KERNEL) {
        // apply the chiral and flavor twists
        // use consistent load order across s to ensure better cache locality
        Vector x0 = arg.x(coord.x_cb + 0 * arg.dc.volume_4d_cb, my_spinor_parity);
        Vector x1 = arg.x(coord.x_cb + 1 * arg.dc.volume_4d_cb, my_spinor_parity);

        if (flavor == 0) {
          out = x0 + arg.a * out;
          out += arg.b * x0.igamma(4);
          out += arg.c * x1;
        } else {
          out = x1 + arg.a * out;
          out += -arg.b * x1.igamma(4);
          out += arg.c * x0;
        }

      } else if (active) {
        Vector x = arg.out(my_flavor_idx, my_spinor_parity);
        out = x + arg.a * out;
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(my_flavor_idx, my_spinor_parity) = out;
    }
  };

} // namespace quda
