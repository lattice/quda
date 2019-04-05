#pragma once

#include <kernels/dslash_wilson.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType reconstruct_>
  struct TwistedMassArg : WilsonArg<Float, nColor, reconstruct_> {
    typedef typename mapper<Float>::type real;
    real a; /** xpay scale facotor */
    real b; /** this is the twist factor */

    TwistedMassArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double b,
        const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
        WilsonArg<Float, nColor, reconstruct_>(out, in, U, a, x, parity, dagger, comm_override),
        a(a),
        b(dagger ? -b : b) // if dagger flip the twist
    {
    }
  };

  /**
     @brief Apply the twisted-mass dslash
       out(x) = M*in = a * D * in + (1 + i*b*gamma_5)*x
     Note this routine only exists in xpay form.
  */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void twistedMass(Arg &arg, int idx, int parity)
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
      x += arg.b * x.igamma(4);
      out = x + arg.a * out;
    } else if (active) {
      Vector x = arg.out(x_cb, my_spinor_parity);
      out = x + arg.a * out;
    }

    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(x_cb, my_spinor_parity) = out;
  }

  // CPU kernel for applying the twisted-mass operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, KernelType kernel_type, typename Arg>
  void twistedMassCPU(Arg arg)
  {
    for (int parity = 0; parity < nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = nParity == 2 ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
        twistedMass<Float, nDim, nColor, nParity, dagger, kernel_type>(arg, x_cb, parity);
      } // 4-d volumeCB
    }   // parity
  }

  // GPU Kernel for applying the twisted-mass operator to a vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void twistedMassGPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for full fields set parity from z thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;

    switch (parity) {
    case 0: twistedMass<Float, nDim, nColor, nParity, dagger, kernel_type>(arg, x_cb, 0); break;
    case 1: twistedMass<Float, nDim, nColor, nParity, dagger, kernel_type>(arg, x_cb, 1); break;
    }
  }

} // namespace quda
