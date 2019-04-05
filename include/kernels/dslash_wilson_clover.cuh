#pragma once

#include <kernels/dslash_wilson.cuh>
#include <clover_field_order.h>
#include <linalg.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType reconstruct_, bool twist_ = false>
  struct WilsonCloverArg : WilsonArg<Float, nColor, reconstruct_> {
    using WilsonArg<Float, nColor, reconstruct_>::nSpin;
    static constexpr int length = (nSpin / (nSpin / 2)) * 2 * nColor * nColor * (nSpin / 2) * (nSpin / 2) / 2;
    static constexpr bool twist = twist_;

    typedef typename clover_mapper<Float, length>::type C;
    typedef typename mapper<Float>::type real;
    const C A;    /** the clover field */
    const real a; /** xpay scale factor */
    const real b; /** chiral twist factor (twisted-clover only) */

    WilsonCloverArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
        double a, double b, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
        WilsonArg<Float, nColor, reconstruct_>(out, in, U, a, x, parity, dagger, comm_override),
        A(A, false),
        a(a),
        b(dagger ? -0.5 * b : 0.5 * b) // factor of 1/2 comes from clover normalization we need to correct for
    {
    }
  };

  /**
     @brief Apply the Wilson-clover dslash
       out(x) = M*in = A(x)*x(x) + D * in(x-mu)
     Note this routine only exists in xpay form.
  */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void wilsonClover(Arg &arg, int idx, int parity)
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

    // defined in dslash_wilson.cuh
    applyWilson<Float, nDim, nColor, nParity, dagger, kernel_type>(
        out, arg, coord, x_cb, 0, parity, idx, thread_dim, active);

    if (kernel_type == INTERIOR_KERNEL) {
      Vector x = arg.x(x_cb, my_spinor_parity);
      x.toRel(); // switch to chiral basis

      Vector tmp;

#pragma unroll
      for (int chirality = 0; chirality < 2; chirality++) {
        constexpr int n = nColor * Arg::nSpin / 2;
        HMatrix<real, n> A = arg.A(x_cb, parity, chirality);
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
      Vector x = arg.out(x_cb, my_spinor_parity);
      out = x + arg.a * out;
    }

    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(x_cb, my_spinor_parity) = out;
  }

  // CPU kernel for applying the Wilson operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  void wilsonCloverCPU(Arg arg)
  {
    for (int parity = 0; parity < nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = nParity == 2 ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
        wilsonClover<Float, nDim, nColor, nParity, dagger, kernel_type>(arg, x_cb, parity);
      } // 4-d volumeCB
    }   // parity
  }

  // GPU Kernel for applying the Wilson operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void wilsonCloverGPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for full fields set parity from z thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;

    switch (parity) {
    case 0: wilsonClover<Float, nDim, nColor, nParity, dagger, kernel_type>(arg, x_cb, 0); break;
    case 1: wilsonClover<Float, nDim, nColor, nParity, dagger, kernel_type>(arg, x_cb, 1); break;
    }
  }

} // namespace quda
