#pragma once

#include <kernels/dslash_wilson_clover_preconditioned.cuh>
#include <clover_field_order.h>
#include <linalg.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType reconstruct_, bool dynamic_clover_>
  struct TwistedCloverArg : WilsonArg<Float, nColor, reconstruct_> {
    using WilsonArg<Float, nColor, reconstruct_>::nSpin;
    static constexpr int length = (nSpin / (nSpin / 2)) * 2 * nColor * nColor * (nSpin / 2) * (nSpin / 2) / 2;
    static constexpr bool dynamic_clover = dynamic_clover_;

    typedef typename mapper<Float>::type real;
    typedef typename clover_mapper<Float, length>::type C;
    const C A;
    const C A2inv; // A^{-2}
    real a;        // this is the scaling factor
    real b;        // this is the twist factor

    TwistedCloverArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
        double a, double b, bool xpay, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
        WilsonArg<Float, nColor, reconstruct_>(out, in, U, xpay ? 1.0 : 0.0, x, parity, dagger, comm_override),
        A(A, false),
        A2inv(A, dynamic_clover ? false : true), // if dynamic clover we don't want the inverse field
        a(a),
        b(dagger ? -0.5 * b : 0.5 * b) // factor of 0.5 comes from basis transform
    {
    }
  };

  /**
     @brief Apply the preconditioned twisted-clover dslash
     - no xpay: out(x) = M*in = A(x)^{-1}D * in(x-mu)
     - with xpay:  out(x) = M*in = (1 + a*A(x)^{-1}D) * in(x-mu)
  */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void twistedClover(Arg &arg, int idx, int parity)
  {
    using namespace linalg; // for Cholesky
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real, nColor, 4> Vector;
    typedef ColorSpinor<real, nColor, 2> HalfVector;
    typedef HMatrix<real, nColor * Arg::nSpin / 2> Mat;

    bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
    int thread_dim;                                          // which dimension is thread working on (fused kernel only)
    int coord[nDim];
    int x_cb = getCoords<nDim, QUDA_4D_PC, kernel_type>(coord, arg, idx, parity, thread_dim);

    const int my_spinor_parity = nParity == 2 ? parity : 0;

    Vector out;

    // defined in dslash_wilson.cuh
    applyWilson<Float, nDim, nColor, nParity, dagger, kernel_type>(
        out, arg, coord, x_cb, 0, parity, idx, thread_dim, active);

    if (kernel_type != INTERIOR_KERNEL && active) {
      // if we're not the interior kernel, then we must sum the partial
      Vector x = arg.out(x_cb, my_spinor_parity);
      out += x;
    }

    if (isComplete<kernel_type>(arg, coord) && active) {
      out.toRel(); // switch to chiral basis

      Vector tmp;

#pragma unroll
      for (int chirality = 0; chirality < 2; chirality++) {

        const complex<real> b(0.0, chirality == 0 ? static_cast<real>(arg.b) : -static_cast<real>(arg.b));
        Mat A = arg.A(x_cb, parity, chirality);
        HalfVector chi = out.chiral_project(chirality);
        chi = A * chi + b * chi;

        if (arg.dynamic_clover) {
          Mat A2 = A.square();
          A2 += b.imag() * b.imag();
          Cholesky<HMatrix, real, nColor * Arg::nSpin / 2> cholesky(A2);
          chi = cholesky.backward(cholesky.forward(chi));
          tmp += static_cast<real>(0.25) * chi.chiral_reconstruct(chirality);
        } else {
          Mat A2inv = arg.A2inv(x_cb, parity, chirality);
          chi = A2inv * chi;
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

    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(x_cb, my_spinor_parity) = out;
  }

  // CPU kernel for applying the Wilson operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  void twistedCloverPreconditionedCPU(Arg arg)
  {

    for (int parity = 0; parity < nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = nParity == 2 ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
        twistedClover<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x_cb, parity);
      } // 4-d volumeCB
    }   // parity
  }

  // GPU Kernel for applying the Wilson operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void twistedCloverPreconditionedGPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for full fields set parity from z thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;

    switch (parity) {
    case 0: twistedClover<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x_cb, 0); break;
    case 1: twistedClover<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x_cb, 1); break;
    }
  }

} // namespace quda
