#pragma once

#include <kernels/dslash_domain_wall_4d.cuh>
#include <kernels/dslash_domain_wall_m5.cuh>

namespace quda
{

  template <typename Float, int nColor_, int nDim, QudaReconstructType reconstruct_, Dslash5Type dslash5_type_>
  struct DomainWall4DFusedM5Arg
    : DomainWall4DArg<Float, nColor_, nDim, reconstruct_>,
      Dslash5Arg<Float, nColor_, false, false, dslash5_type_> {
    // ^^^ Note that for Dslash5Arg we have xpay == dagger == false. This is because the xpay and dagger are determined
    // by fused kernel, not the dslash5, so the false, false here are simply dummy instantiations.

    static constexpr int nColor = nColor_;

    using DomainWall4DArg = DomainWall4DArg<Float, nColor, nDim, reconstruct_>;
    using DomainWall4DArg::a_5;
    using DomainWall4DArg::dagger;
    using DomainWall4DArg::in;
    using DomainWall4DArg::nParity;
    using DomainWall4DArg::out;
    using DomainWall4DArg::threads;
    using DomainWall4DArg::x;
    using DomainWall4DArg::xpay;

    static constexpr Dslash5Type dslash5_type = dslash5_type_;

    using Dslash5Arg
      = Dslash5Arg<Float, nColor, false, false, dslash5_type>;
    using Dslash5Arg::Ls;

    using real = typename mapper<Float>::type;
    complex<real> alpha;
    complex<real> beta;

    bool fuse_m5inv_m5pre;

    DomainWall4DFusedM5Arg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_5,
                           const Complex *b_5, const Complex *c_5, bool xpay, const ColorSpinorField &x, int parity,
                           bool dagger, const int *comm_override, double m_f) :
      DomainWall4DArg(out, in, U, a, m_5, b_5, c_5, xpay, x, parity, dagger, comm_override),
      Dslash5Arg(out, in, x, m_f, m_5, b_5, c_5, a)
    {
      for (int s = 0; s < Ls; s++) {
        auto kappa_b_s = 0.5 / (b_5[s] * (m_5 + 4.0) + 1.0);
        auto kappa_c_s = 0.5 / (c_5[s] * (m_5 + 4.0) - 1.0);
        auto kappa_s = kappa_b_s / kappa_c_s;
        a_5[s] = a * kappa_b_s * kappa_b_s;
#if 0
        alpha = b_5[s] - c_5[s] / kappa_s;
        beta = c_5[s] / kappa_s;
#endif
      }; // 4-d Mobius
    }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct domainWall4DFusedM5 : dslash_default {

    static constexpr Dslash5Type dslash5_type = Arg::type;

    Arg &arg;
    constexpr domainWall4DFusedM5(Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int s, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim; // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, s, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector stencil_out;
      applyWilson<nParity, dagger, mykernel_type>(stencil_out, arg, coord, parity, idx, thread_dim, active);

      Vector out;
      // In the following `x_cb` are all passed as `x_cb = 0`, since it will not be used if `shared = true`, and `shared = true`
      if (Arg::dslash5_type == M5_INV_MOBIUS) {
        // Apply the m5inv.
        out = variableInv<dagger, Vector, typename Arg::Dslash5Arg>(
          arg, stencil_out, my_spinor_parity, 0, s);
      }

      if (Arg::dslash5_type == DSLASH5_MOBIUS_PRE) {
        // TODO: there doesn't need to have a sync before caching.
        out = d5<dagger, true, Vector, typename Arg::Dslash5Arg>(arg, stencil_out, my_spinor_parity, 0, s);
      }

      if (Arg::dslash5_type == M5_INV_MOBIUS_M5_PRE) {
        // Apply the m5inv.
        out = variableInv<dagger, Vector, typename Arg::Dslash5Arg>(
          arg, stencil_out, my_spinor_parity, 0, s);
#if 0
        out = arg.alpha * out + arg.beta * stencil_out;
#else
        out = d5<dagger, true, Vector, typename Arg::Dslash5Arg>(arg, out, my_spinor_parity, 0, s);
#endif
      }

      int xs = coord.x_cb + s * arg.dc.volume_4d_cb;
      if (xpay && mykernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x(xs, my_spinor_parity);
        out = x + arg.a_5[s] * out;
      } else if (mykernel_type != INTERIOR_KERNEL && active) {
        Vector x = arg.out(xs, my_spinor_parity);
        out = x + (xpay ? arg.a_5[s] * out : out);
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(xs, my_spinor_parity) = out;
    }
  };

} // namespace quda
