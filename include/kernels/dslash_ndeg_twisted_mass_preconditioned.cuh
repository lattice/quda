#pragma once

#include <kernels/dslash_wilson.cuh>
#include <kernels/dslash_twisted_mass_preconditioned.cuh>
#include <shared_memory_cache_helper.h>

namespace quda
{

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_, bool asymmetric_>
  struct NdegTwistedMassArg : WilsonArg<Float, nColor, nDim, reconstruct_> {
    typedef typename mapper<Float>::type real;
    static constexpr bool asymmetric = asymmetric_; /** whether we are applying the asymetric operator or not */
    real a;          /** this is the Wilson-dslash scale factor */
    real b;          /** this is the chiral twist factor */
    real c;          /** this is the flavor twist factor */
    real a_inv;      /** inverse scaling factor - used to allow early xpay inclusion */
    real b_inv;      /** inverse chiral twist factor - used to allow early xpay inclusion */
    real c_inv;      /** inverse flavor twist factor - used to allow early xpay inclusion */

    NdegTwistedMassArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                       const ColorSpinorField &halo, const GaugeField &U, double a, double b,
                       double c, bool xpay, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override) :
      WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, halo, U, xpay ? 1.0 : 0.0, x, parity, dagger, comm_override),
      a(a),
      b(dagger ? -b : b), // if dagger flip the chiral twist
      c(c),
      a_inv(1.0 / (a * (1.0 + b * b - c * c))),
      b_inv(dagger ? b : -b),
      c_inv(-c)
    {
      // set parameters for twisting in the packing kernel
      if (dagger && !asymmetric) {
        DslashArg<Float, nDim>::twist_a = this->a;
        DslashArg<Float, nDim>::twist_b = this->b;
        DslashArg<Float, nDim>::twist_c = this->c;
      }
    }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct nDegTwistedMassPreconditioned : dslash_default {

    const Arg &arg;
    constexpr nDegTwistedMassPreconditioned(const Arg &arg) : arg(arg) {}
    constexpr int twist_pack() const { return (!Arg::asymmetric && dagger) ? 2 : 0; }
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    /**
       @brief Apply the preconditioned non-degenerate twisted-mass dslash
       out(x) = M*in = a * D * in + (1 + i*b*gamma_5*tau_3 + c*tau_1)*x
       - no xpay: out(x) = M*in = a*(1+i*b*gamma_5*tau_3 + c*tau_1)D * in
       - with xpay:  out(x) = M*in = x + a*(1+i*b*gamma_5 + c*tau_1)D * in
    */

    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_flavor, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      int src_idx = src_flavor / 2;
      int flavor = src_flavor % 2;

      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, flavor, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;

      if (!dagger || Arg::asymmetric) // defined in dslash_wilson.cuh
        applyWilson<nParity, dagger, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active, src_idx);
      else // defined in dslash_twisted_mass_preconditioned
        applyWilsonTM<nParity, dagger, 2, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active, src_idx);

      int my_flavor_idx = coord.x_cb + flavor * arg.dc.volume_4d_cb;

      if (xpay && mykernel_type == INTERIOR_KERNEL) {

        if (!dagger || Arg::asymmetric) { // apply inverse twist which is undone below
          // use consistent load order across s to ensure better cache locality
          Vector x0 = arg.x[src_idx](coord.x_cb + 0 * arg.dc.volume_4d_cb, my_spinor_parity);
          Vector x1 = arg.x[src_idx](coord.x_cb + 1 * arg.dc.volume_4d_cb, my_spinor_parity);
          if (flavor == 0)
            out += arg.a_inv * (x0 + arg.b_inv * x0.igamma(4) + arg.c_inv * x1);
          else
            out += arg.a_inv * (x1 - arg.b_inv * x1.igamma(4) + arg.c_inv * x0);
        } else {
          Vector x = arg.x[src_idx](my_flavor_idx, my_spinor_parity);
          out += x; // just directly add since twist already applied in the dslash
        }

      } else if (mykernel_type != INTERIOR_KERNEL && active) {
        // if we're not the interior kernel, then we must sum the partial
        Vector x = arg.out[src_idx](my_flavor_idx, my_spinor_parity);
        out += x;
      }
      
      if (!dagger || Arg::asymmetric) { // apply A^{-1} to D*in
        SharedMemoryCache<Vector> cache;
        if (isComplete<mykernel_type>(arg, coord) && active) {
          // to apply the preconditioner we need to put "out" in shared memory so the other flavor can access it
          cache.save(out);
        }

        cache.sync(); // safe to sync in here since other threads will exit
        if (isComplete<mykernel_type>(arg, coord) && active) {
          if (flavor == 0)
            out = arg.a * (out + arg.b * out.igamma(4) + arg.c * cache.load_y(target::thread_idx().y + 1));
          else
            out = arg.a * (out - arg.b * out.igamma(4) + arg.c * cache.load_y(target::thread_idx().y - 1));
        }
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](my_flavor_idx, my_spinor_parity) = out;
    }

  };

} // namespace quda
