#pragma once

#include <clover_field_order.h>
#include <kernels/dslash_wilson.cuh>
#include <shared_memory_cache_helper.cuh>
#include <linalg.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType reconstruct_, bool dynamic_clover_>
  struct NdegTwistedCloverArg : WilsonArg<Float, nColor, reconstruct_> {
    using WilsonArg<Float, nColor, reconstruct_>::nSpin;
    static constexpr int length = (nSpin / (nSpin / 2)) * 2 * nColor * nColor * (nSpin / 2) * (nSpin / 2) / 2;
    static constexpr bool dynamic_clover = dynamic_clover_;

    typedef typename mapper<Float>::type real;
    typedef typename clover_mapper<Float, length>::type C;
    const C A;
    const C A2inv; // A^{-2}
    real a;          /** this is the Wilson-dslash scale factor */
    real b;          /** this is the chiral twist factor */
    real c;          /** this is the flavor twist factor */

    NdegTwistedCloverArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
                         double a, double b, double c, bool xpay, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override) :
        WilsonArg<Float, nColor, reconstruct_>(out, in, U, xpay ? 1.0 : 0.0, x, parity, dagger, comm_override),
        A(A, false),
        A2inv(A, dynamic_clover ? false : true), // if dynamic clover we don't want the inverse field
        a(a),
        b(dagger ? -0.5 * b : 0.5 * b), // if dagger flip the chiral twist
        c(0.5*c)
    {
    }
  };

  /**
     @brief Apply the twisted-clover dslash
       out(x) = M*in = a * D * in + (1 + i*b*gamma_5*tau_3 + c*tau_1)*x
     Note this routine only exists in xpay form.
  */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void ndegTwistedClover(Arg &arg, int idx, int flavor, int parity)
  {
    using namespace linalg; // for Cholesky
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real, nColor, 4> Vector;
    typedef ColorSpinor<real, nColor, 2> HalfVector;
    constexpr int n = nColor * Arg::nSpin / 2;
    typedef HMatrix<real, n> HMat;

    bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
    int thread_dim;                                          // which dimension is thread working on (fused kernel only)
    int coord[nDim];
    int x_cb = getCoords<nDim, QUDA_4D_PC, kernel_type>(coord, arg, idx, parity, thread_dim);

    const int my_spinor_parity = nParity == 2 ? parity : 0;
    Vector out;

    applyWilson<Float, nDim, nColor, nParity, dagger, kernel_type>(out, arg, coord, x_cb, flavor, parity, idx, thread_dim, active);

    int my_flavor_idx = x_cb + flavor * arg.dc.volume_4d_cb;

    if (kernel_type != INTERIOR_KERNEL && active) {
      // if we're not the interior kernel, then we must sum the partial
      Vector x = arg.out(my_flavor_idx, my_spinor_parity);
      out += x;
    }

    if (isComplete<kernel_type>(arg, coord) && active) {

      VectorCache<real, HalfVector> cache; // used for flavor twist
      Vector tmp;

#pragma unroll
      for (int chirality = 0; chirality < 2; chirality++) {
        HMat A = arg.A(x_cb, parity, chirality);

        HalfVector chi = out.chiral_project(chirality);
        cache.save(chi); // put "chi" in shared memory so the other flavor can access it

        HalfVector A_chi = A * chi;
        const complex<real> b(0.0, chirality == 0 ? arg.b : -arg.b);
        A_chi += b * chi;

        cache.sync(); // safe to sync in here since other threads will exit
        A_chi += arg.c * cache.load(threadIdx.x, 1-flavor, threadIdx.z);

        if (arg.dynamic_clover) {
          HMat A2 = A.square();
          A2 += b.imag() * b.imag();
          A2 += (-arg.c * arg.c);
          Cholesky<HMatrix, real, nColor * Arg::nSpin / 2> cholesky(A2);
          chi = cholesky.backward(cholesky.forward(A_chi));
          tmp += static_cast<real>(0.25) * chi.chiral_reconstruct(chirality);
        } else {
          HMat A2inv = arg.A2inv(x_cb, parity, chirality);
          chi = A2inv * A_chi;
          tmp += static_cast<real>(2.0) * chi.chiral_reconstruct(chirality);
        }
      }

      tmp.toNonRel(); // switch back to non-chiral basis

      if (xpay) {
        Vector x = arg.x(x_cb, my_spinor_parity);
        out = x + arg.a * tmp;
      } else {
        out = arg.a * tmp;
      }
    }

    if (x_cb == 0) {
      out.print();
    }

    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(my_flavor_idx, my_spinor_parity) = out;
  }

  // CPU kernel for applying the non-degenerate twisted-mass operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  void ndegTwistedCloverPreconditionedCPU(Arg arg)
  {
    for (int parity = 0; parity < nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = nParity == 2 ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
        for (int flavor = 0; flavor < 2; flavor++) {
          ndegTwistedClover<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x_cb, flavor, parity);
        }
      } // 4-d volumeCB
    }   // parity
  }

  // GPU Kernel for applying the non-degenerate twisted-mass operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void ndegTwistedCloverPreconditionedGPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for this operator flavor can be spread onto different blocks
    int flavor = blockIdx.y * blockDim.y + threadIdx.y;

    // for full fields set parity from z thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;

    switch (parity) {
    case 0:
      ndegTwistedClover<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x_cb, flavor, 0);
      break;
    case 1:
      ndegTwistedClover<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x_cb, flavor, 1);
      break;
    }
  }

} // namespace quda
