#pragma once

#include <kernels/dslash_wilson.cuh>

namespace quda
{

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_>
  struct TwistedMassArg : WilsonArg<Float, nColor, nDim, reconstruct_> {
    typedef typename mapper<Float>::type real;
    real a; /** xpay scale facotor */
    real b; /** this is the twist factor */

    TwistedMassArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                   const ColorSpinorField &halo, const GaugeField &U, double a, double b,
                   cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override) :
      WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, halo, U, a, x, parity, dagger, comm_override),
      a(a),
      b(dagger ? -b : b) // if dagger flip the twist
    {
    }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct twistedMass : dslash_default {

    const Arg &arg;
    template <typename Ftor> constexpr twistedMass(const Ftor &ftor) : arg(ftor.arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    /**
       @brief Apply the twisted-mass dslash
       out(x) = M*in = a * D * in + (1 + i*b*gamma_5)*x
       Note this routine only exists in xpay form.
    */
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_idx, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

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
        x += arg.b * x.igamma(4);
        out = x + arg.a * out;
      } else if (active) {
        Vector x = arg.out[src_idx](coord.x_cb, my_spinor_parity);
        out = x + arg.a * out;
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](coord.x_cb, my_spinor_parity) = out;
    }
  };

} // namespace quda
