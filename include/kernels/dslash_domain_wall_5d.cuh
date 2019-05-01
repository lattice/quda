#pragma once

#include <kernels/dslash_wilson.cuh>

namespace quda
{

  // fixme: fused kernel (thread dim mappers set after construction?) and xpay

  template <typename Float, int nColor, QudaReconstructType reconstruct_>
  struct DomainWall5DArg : WilsonArg<Float, nColor, reconstruct_> {
    typedef typename mapper<Float>::type real;
    int Ls;   /** fifth dimension length */
    real a;   /** xpay scale factor */
    real m_f; /** fermion mass parameter */

    DomainWall5DArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_f,
        bool xpay, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override) :
        WilsonArg<Float, nColor, reconstruct_>(out, in, U, xpay ? a : 0.0, x, parity, dagger, comm_override),
        Ls(in.X(4)),
        a(a),
        m_f(m_f)
    {
    }
  };

  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __device__ __host__ inline void domainWall5D(Arg &arg, int idx, int parity)
  {
    typedef typename mapper<Float>::type real;
    typedef ColorSpinor<real, nColor, 4> Vector;

    bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
    int thread_dim;                                          // which dimension is thread working on (fused kernel only)
    int coord[nDim];
    int x_cb = getCoords<nDim, QUDA_5D_PC, kernel_type>(coord, arg, idx, parity, thread_dim);

    const int my_spinor_parity = nParity == 2 ? parity : 0;
    Vector out;

    // we pass s=0, since x_cb is a 5-d index that includes s
    applyWilson<Float, nDim, nColor, nParity, dagger, kernel_type>(
        out, arg, coord, x_cb, 0, parity, idx, thread_dim, active);

    if (kernel_type == INTERIOR_KERNEL) { // 5th dimension derivative always local
      constexpr int d = 4;
      const int s = coord[4];
      const int their_spinor_parity = nParity == 2 ? 1 - parity : 0;
      {
        const int fwd_idx = getNeighborIndexCB<nDim>(coord, d, +1, arg.dc);
        constexpr int proj_dir = dagger ? +1 : -1;
        Vector in = arg.in(fwd_idx, their_spinor_parity);
        if (s == arg.Ls - 1) {
          out += (-arg.m_f * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        } else {
          out += in.project(d, proj_dir).reconstruct(d, proj_dir);
        }
      }

      {
        const int back_idx = getNeighborIndexCB<nDim>(coord, d, -1, arg.dc);
        constexpr int proj_dir = dagger ? -1 : +1;
        Vector in = arg.in(back_idx, their_spinor_parity);
        if (s == 0) {
          out += (-arg.m_f * in.project(d, proj_dir)).reconstruct(d, proj_dir);
        } else {
          out += in.project(d, proj_dir).reconstruct(d, proj_dir);
        }
      }
    }

    if (xpay && kernel_type == INTERIOR_KERNEL) {
      Vector x = arg.x(x_cb, my_spinor_parity);
      out = x + arg.a * out;
    } else if (kernel_type != INTERIOR_KERNEL && active) {
      Vector x = arg.out(x_cb, my_spinor_parity);
      out = x + (xpay ? arg.a * out : out);
    }

    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(x_cb, my_spinor_parity) = out;
  }

  // CPU Kernel for applying 5-d Wilson operator to a 5-d vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  void domainWall5DCPU(Arg &arg)
  {
    for (int parity = 0; parity < nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = nParity == 2 ? parity : arg.parity;

      for (int x5_cb = 0; x5_cb < arg.threads * arg.Ls; x5_cb++) { // 5-d volume
        domainWall5D<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x5_cb, parity);
      } // 5-d volumeCB
    }   // parity
  }

  // GPU Kernel for applying 5-d Wilson operator to a 5-d vector
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  __global__ void domainWall5DGPU(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.threads) return;

    // for this operator Ls is mapped to the y thread dimension
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (s >= arg.Ls) return;

    int x5_cb = s * arg.threads + x_cb; // 5-d checkerboard index

    // for full fields set parity from z thread index else use arg setting
    int parity = nParity == 2 ? blockDim.z * blockIdx.z + threadIdx.z : arg.parity;

    switch (parity) {
    case 0: domainWall5D<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x5_cb, 0); break;
    case 1: domainWall5D<Float, nDim, nColor, nParity, dagger, xpay, kernel_type>(arg, x5_cb, 1); break;
    }
  }

} // namespace quda
