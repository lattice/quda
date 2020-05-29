#pragma once

#include <kernels/dslash_wilson.cuh>

namespace quda
{

  constexpr int size = 4096;
  __constant__ char mobius_d[size]; // constant buffer used for Mobius coefficients for GPU kernel

  template <typename Float, int nColor, int nDim, QudaReconstructType reconstruct_>
  struct DomainWall4DArg : WilsonArg<Float, nColor, nDim, reconstruct_> {
    typedef typename mapper<Float>::type real;
    int Ls;                             /** fifth dimension length */
    complex<real> a_5[QUDA_MAX_DWF_LS]; /** xpay scale factor for each 4-d subvolume */

    /**
       @brief Helper function for grabbing the constant struct, whether
       we are on the GPU or CPU.
    */
    inline __device__ __host__ complex<real> a5(int s)
    {
#ifdef __CUDA_ARCH__
      return reinterpret_cast<const complex<real> *>(mobius_d)[s];
#else
      return a_5[s];
#endif
    }

    DomainWall4DArg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_5,
                    const Complex *b_5, const Complex *c_5, bool xpay, const ColorSpinorField &x, int parity,
                    bool dagger, const int *comm_override) :
      WilsonArg<Float, nColor, nDim, reconstruct_>(out, in, U, xpay ? a : 0.0, x, parity, dagger, comm_override),
      Ls(in.X(4))
    {
      if (b_5 == nullptr || c_5 == nullptr)
        for (int s = 0; s < Ls; s++) a_5[s] = a; // 4-d Shamir
      else
        for (int s = 0; s < Ls; s++) a_5[s] = 0.5 * a / (b_5[s] * (m_5 + 4.0) + 1.0); // 4-d Mobius
    }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct domainWall4D : dslash_default {

    Arg &arg;
    constexpr domainWall4D(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    __device__ __host__ inline void operator()(int idx, int s, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      bool active
        = kernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, kernel_type>(arg, idx, s, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector out;
      applyWilson<nParity, dagger, kernel_type>(out, arg, coord, parity, idx, thread_dim, active);

      int xs = coord.x_cb + s * arg.dc.volume_4d_cb;
      if (xpay && kernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x(xs, my_spinor_parity);
        out = x + arg.a5(s) * out;
      } else if (kernel_type != INTERIOR_KERNEL && active) {
        Vector x = arg.out(xs, my_spinor_parity);
        out = x + (xpay ? arg.a5(s) * out : out);
      }

      if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(xs, my_spinor_parity) = out;
    }
  };

} // namespace quda
