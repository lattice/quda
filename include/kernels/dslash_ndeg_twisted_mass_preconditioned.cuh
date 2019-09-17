#pragma once

#include <kernels/dslash_wilson.cuh>
#include <kernels/dslash_twisted_mass_preconditioned.cuh>
#include <shared_memory_cache_helper.cuh>

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

    NdegTwistedMassArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double b,
                       double c, bool xpay, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
      WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, U, xpay ? 1.0 : 0.0, x, parity, dagger, comm_override),
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

    Arg &arg;
    constexpr nDegTwistedMassPreconditioned(Arg &arg) : arg(arg) {}
    constexpr int twist_pack() const { return (!Arg::asymmetric && dagger) ? 2 : 0; }
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    /**
       @brief Apply the preconditioned non-degenerate twisted-mass dslash
       out(x) = M*in = a * D * in + (1 + i*b*gamma_5*tau_3 + c*tau_1)*x
       - no xpay: out(x) = M*in = a*(1+i*b*gamma_5*tau_3 + c*tau_1)D * in
       - with xpay:  out(x) = M*in = x + a*(1+i*b*gamma_5 + c*tau_1)D * in
    */
    __device__ __host__ inline void operator()(int idx, int flavor, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;
      typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;

      bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      int coord[Arg::nDim];
      int x_cb = getCoords<QUDA_4D_PC, kernel_type>(coord, arg, idx, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;

      if (!dagger || Arg::asymmetric) // defined in dslash_wilson.cuh
        applyWilson<nParity, dagger, kernel_type>(out, arg, coord, x_cb, flavor, parity, idx, thread_dim, active);
      else // defined in dslash_twisted_mass_preconditioned
        applyWilsonTM<nParity, dagger, 2, kernel_type>(out, arg, coord, x_cb, flavor, parity, idx, thread_dim, active);

      int my_flavor_idx = x_cb + flavor * arg.dc.volume_4d_cb;

      if (xpay && kernel_type == INTERIOR_KERNEL) {

        if (!dagger || Arg::asymmetric) { // apply inverse twist which is undone below
          // use consistent load order across s to ensure better cache locality
          Vector x0 = arg.x(x_cb + 0 * arg.dc.volume_4d_cb, my_spinor_parity);
          Vector x1 = arg.x(x_cb + 1 * arg.dc.volume_4d_cb, my_spinor_parity);
          if (flavor == 0)
            out += arg.a_inv * (x0 + arg.b_inv * x0.igamma(4) + arg.c_inv * x1);
          else
            out += arg.a_inv * (x1 - arg.b_inv * x1.igamma(4) + arg.c_inv * x0);
        } else {
          Vector x = arg.x(my_flavor_idx, my_spinor_parity);
          out += x; // just directly add since twist already applied in the dslash
        }

      } else if (kernel_type != INTERIOR_KERNEL && active) {
        // if we're not the interior kernel, then we must sum the partial
        Vector x = arg.out(my_flavor_idx, my_spinor_parity);
        out += x;
      }

      if (isComplete<kernel_type>(arg, coord) && active) {
        if (!dagger || Arg::asymmetric) { // apply A^{-1} to D*in
          VectorCache<real, Vector> cache;
          // to apply the preconditioner we need to put "out" in shared memory so the other flavor can access it
          cache.save(out);
          cache.sync(); // safe to sync in here since other threads will exit

          if (flavor == 0)
            out = arg.a * (out + arg.b * out.igamma(4) + arg.c * cache.load(threadIdx.x, 1, threadIdx.z));
          else
            out = arg.a * (out - arg.b * out.igamma(4) + arg.c * cache.load(threadIdx.x, 0, threadIdx.z));
        }
      }

      if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(my_flavor_idx, my_spinor_parity) = out;
    }

  };

} // namespace quda
