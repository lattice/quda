#pragma once

#include <kernels/dslash_wilson.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType reconstruct_>
  struct DomainWall4DArg : WilsonArg<Float,nColor,reconstruct_> {
    typedef typename mapper<Float>::type real;
    int Ls;                             /** fifth dimension length */
    complex<real> a_5[QUDA_MAX_DWF_LS]; /** xpay scale factor for each 4-d subvolume */

    DomainWall4DArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                    double a, double m_5, const Complex *b_5, const Complex *c_5, bool xpay, const ColorSpinorField &x,
                    int parity, bool dagger, const int *comm_override) :
      WilsonArg<Float,nColor,reconstruct_>(out, in, U, xpay ? a : 0.0, x, parity, dagger, comm_override),
      Ls(in.X(4))
    {
      if (b_5 == nullptr || c_5 == nullptr) for (int s=0; s<Ls; s++) a_5[s] = a; // 4-d Shamir
      else for (int s=0; s<Ls; s++) a_5[s] = 0.5 * a / (b_5[s]*(m_5+4.0) + 1.0); // 4-d Mobius
    }
  };

  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void domainWall4D(Arg &arg, int idx, int s, int parity)
  {
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real,nColor,4> Vector;

    bool active = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
    int thread_dim; // which dimension is thread working on (fused kernel only)
    int coord[nDim];
    int x_cb = getCoords<nDim,QUDA_4D_PC,kernel_type>(coord, arg, idx, parity, thread_dim);

    const int my_spinor_parity = nParity == 2 ? parity : 0;
    Vector out;
    applyWilson<Float,nDim,nColor,nParity,dagger,kernel_type>(out, arg, coord, x_cb, s, parity, idx, thread_dim, active);

    int xs = x_cb + s*arg.dc.volume_4d_cb;
    if (xpay && kernel_type == INTERIOR_KERNEL) {
      Vector x = arg.x(xs, my_spinor_parity);
      out = x + arg.a_5[s] * out;
    } else if (kernel_type != INTERIOR_KERNEL && active) {
      Vector x = arg.out(xs, my_spinor_parity);
      out = x + (xpay ? arg.a_5[s] * out : out);
    }

    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(xs, my_spinor_parity) = out;
  }

  // CPU Kernel for applying 4-d Wilson operator to a 5-d vector (replicated along fifth dimension)
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  void domainWall4DCPU(Arg &arg)
  {
    for (int parity= 0; parity < nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = nParity == 2 ? parity : arg.parity;

      for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
        for (int s=0; s<arg.Ls; s++) {
          domainWall4D<Float,nDim,nColor,nParity,dagger,xpay,kernel_type>(arg, x_cb, s, parity);
        }
      } // 4-d volumeCB
    } // parity
  }

  // GPU Kernel for applying 4-d Wilson operator to a 5-d vector (replicated along fifth dimension)
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void domainWall4DGPU(Arg arg)
  {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for this operator Ls is mapped to the y thread dimension
    int s = blockIdx.y*blockDim.y + threadIdx.y;
    if (s >= arg.Ls) return;

    // for full fields set parity from y thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z*blockIdx.z + threadIdx.z : arg.parity;

    switch(parity) {
    case 0: domainWall4D<Float,nDim,nColor,nParity,dagger,xpay,kernel_type>(arg, x_cb, s, 0); break;
    case 1: domainWall4D<Float,nDim,nColor,nParity,dagger,xpay,kernel_type>(arg, x_cb, s, 1); break;
    }
  }

} // namespace quda
