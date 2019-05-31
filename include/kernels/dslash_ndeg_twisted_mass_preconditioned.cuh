#pragma once

#include <kernels/dslash_wilson.cuh>
#include <kernels/dslash_twisted_mass_preconditioned.cuh>
#include <shared_memory_cache_helper.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType reconstruct_>
  struct NdegTwistedMassArg : WilsonArg<Float, nColor, reconstruct_> {
    typedef typename mapper<Float>::type real;
    real a;          /** this is the Wilson-dslash scale factor */
    real b;          /** this is the chiral twist factor */
    real c;          /** this is the flavor twist factor */
    real a_inv;      /** inverse scaling factor - used to allow early xpay inclusion */
    real b_inv;      /** inverse chiral twist factor - used to allow early xpay inclusion */
    real c_inv;      /** inverse flavor twist factor - used to allow early xpay inclusion */
    bool asymmetric; /** whether we are applying the asymetric operator or not */

    NdegTwistedMassArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double b,
        double c, bool xpay, const ColorSpinorField &x, int parity, bool dagger, bool asymmetric,
        const int *comm_override) :
        WilsonArg<Float, nColor, reconstruct_>(out, in, U, xpay ? 1.0 : 0.0, x, parity, dagger, comm_override),
        a(a),
        b(dagger ? -b : b), // if dagger flip the chiral twist
        c(c),
        a_inv(1.0 / (a * (1.0 + b * b - c * c))),
        b_inv(dagger ? b : -b),
        c_inv(-c),
        asymmetric(asymmetric)
    {
      // set parameters for twisting in the packing kernel
      if (dagger && !asymmetric) {
        DslashArg<Float>::twist_a = this->a;
        DslashArg<Float>::twist_b = this->b;
        DslashArg<Float>::twist_c = this->c;
      }
    }
  };

  /**
     @brief Apply the twisted-mass dslash
       out(x) = M*in = a * D * in + (1 + i*b*gamma_5*tau_3 + c*tau_1)*x
     Note this routine only exists in xpay form.
  */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool asymmetric, bool xpay,
      KernelType kernel_type, typename Arg>
  __device__ __host__ inline void ndegTwistedMass(Arg &arg, int idx, int flavor, int parity)
  {
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real, nColor, 4> Vector;
    typedef ColorSpinor<real, nColor, 2> HalfVector;

    bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
    int thread_dim;                                          // which dimension is thread working on (fused kernel only)
    int coord[nDim];
    int x_cb = getCoords<nDim, QUDA_4D_PC, kernel_type>(coord, arg, idx, parity, thread_dim);

    const int my_spinor_parity = nParity == 2 ? parity : 0;
    Vector out;

    if (!dagger || asymmetric) // defined in dslash_wilson.cuh
      applyWilson<Float, nDim, nColor, nParity, dagger, kernel_type>(
          out, arg, coord, x_cb, flavor, parity, idx, thread_dim, active);
    else // defined in dslash_twisted_mass_preconditioned
      applyWilsonTM<Float, nDim, nColor, nParity, dagger, 2, kernel_type>(
          out, arg, coord, x_cb, flavor, parity, idx, thread_dim, active);

    int my_flavor_idx = x_cb + flavor * arg.dc.volume_4d_cb;

    if (xpay && kernel_type == INTERIOR_KERNEL) {

      if (!dagger || asymmetric) { // apply inverse twist which is undone below
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
      if (!dagger || asymmetric) { // apply A^{-1} to D*in
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

  // CPU kernel for applying the non-degenerate twisted-mass operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  void ndegTwistedMassPreconditionedCPU(Arg arg)
  {
    if (arg.asymmetric) {
      for (int parity = 0; parity < nParity; parity++) {
        // for full fields then set parity from loop else use arg setting
        parity = nParity == 2 ? parity : arg.parity;

        for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
          for (int flavor = 0; flavor < 2; flavor++) {
            ndegTwistedMass<Float, nDim, nColor, nParity, dagger, true, xpay, kernel_type>(arg, x_cb, flavor, parity);
          }
        } // 4-d volumeCB
      }   // parity
    } else {
      for (int parity = 0; parity < nParity; parity++) {
        // for full fields then set parity from loop else use arg setting
        parity = nParity == 2 ? parity : arg.parity;

        for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
          for (int flavor = 0; flavor < 2; flavor++) {
            ndegTwistedMass<Float, nDim, nColor, nParity, dagger, false, xpay, kernel_type>(arg, x_cb, flavor, parity);
          }
        } // 4-d volumeCB
      }   // parity
    }
  }

  // GPU Kernel for applying the non-degenerate twisted-mass operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void ndegTwistedMassPreconditionedGPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for this operator flavor can be spread onto different blocks
    int flavor = blockIdx.y * blockDim.y + threadIdx.y;

    // for full fields set parity from z thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;

    if (arg.asymmetric) {
      // constrain template instantiation for compilation (asymmetric implies dagger and !xpay)
      switch (parity) {
      case 0:
        ndegTwistedMass<Float, nDim, nColor, nParity, true, true, false, kernel_type>(arg, x_cb, flavor, 0);
        break;
      case 1:
        ndegTwistedMass<Float, nDim, nColor, nParity, true, true, false, kernel_type>(arg, x_cb, flavor, 1);
        break;
      }
    } else {
      switch (parity) {
      case 0:
        ndegTwistedMass<Float, nDim, nColor, nParity, dagger, false, xpay, kernel_type>(arg, x_cb, flavor, 0);
        break;
      case 1:
        ndegTwistedMass<Float, nDim, nColor, nParity, dagger, false, xpay, kernel_type>(arg, x_cb, flavor, 1);
        break;
      }
    }
  }

} // namespace quda
