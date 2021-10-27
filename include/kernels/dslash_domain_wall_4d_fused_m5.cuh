#pragma once

#include <constant_kernel_arg.h>
#include <kernels/dslash_domain_wall_4d.cuh>
#include <kernels/dslash_domain_wall_m5.cuh>

namespace quda
{

  template <typename Float, int nColor_, int nDim, QudaReconstructType reconstruct_, Dslash5Type dslash5_type_>
  struct DomainWall4DFusedM5Arg : DomainWall4DArg<Float, nColor_, nDim, reconstruct_>,
                                  Dslash5Arg<Float, nColor_, false, false, dslash5_type_> {
    // ^^^ Note that for Dslash5Arg we have xpay == dagger == false. This is because the xpay and dagger are determined
    // by fused kernel, not the dslash5, so the `false, false` here are simply dummy instantiations.

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

    using F = typename DomainWall4DArg::F;

    F y; // The additional output field accessor

    static constexpr Dslash5Type dslash5_type = dslash5_type_;

    using Dslash5Arg = Dslash5Arg<Float, nColor, false, false, dslash5_type>;
    using Dslash5Arg::Ls;

    using real = typename mapper<Float>::type;
    complex<real> alpha;
    complex<real> beta;

    bool fuse_m5inv_m5pre;

    DomainWall4DFusedM5Arg(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_5,
                           const Complex *b_5, const Complex *c_5, bool xpay, const ColorSpinorField &x,
                           ColorSpinorField &y, int parity, bool dagger, const int *comm_override, double m_f) :
      DomainWall4DArg(out, in, U, a, m_5, b_5, c_5, xpay, x, parity, dagger, comm_override),
      Dslash5Arg(out, in, x, m_f, m_5, b_5, c_5, a),
      y(y)
    {
      for (int s = 0; s < Ls; s++) {
        auto kappa_b_s = 0.5 / (b_5[s] * (m_5 + 4.0) + 1.0);
        a_5[s] = a * kappa_b_s * kappa_b_s;
      }; // 4-d Mobius
    }
  };

  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct domainWall4DFusedM5 : dslash_default {

    static constexpr Dslash5Type dslash5_type = Arg::type;

    const Arg &arg;
    constexpr domainWall4DFusedM5(const Arg &arg) : arg(arg) { }
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

      constexpr bool shared = true; // Use shared memory

      // In the following `x_cb` are all passed as `x_cb = 0`, since it will not be used if `shared = true`, and `shared = true`

      if (active) {

        /******
         *  Apply M5pre
         */
        if (Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS_PRE) {
          constexpr bool sync = false;
          out = d5<sync, dagger, shared, Vector, typename Arg::Dslash5Arg>(arg, stencil_out, my_spinor_parity, 0, s);
        }
      }

      int xs = coord.x_cb + s * arg.dc.volume_4d_cb;
      if (Arg::dslash5_type == Dslash5Type::M5_INV_MOBIUS_M5_INV_DAG) {

        /******
         *  Apply the two M5inv's:
         *    this is actually   y = 1 * x - kappa_b^2 * m5inv * D4 * in
         *                     out = m5inv-dagger * y
         */
        if (active) {
          constexpr bool sync = false;
          out = variableInv<sync, dagger, shared, Vector, typename Arg::Dslash5Arg>(arg, stencil_out, my_spinor_parity,
                                                                                    0, s);
        }

        Vector aggregate_external;
        if (xpay && mykernel_type == INTERIOR_KERNEL) {
          Vector x = arg.x(xs, my_spinor_parity);
          out = x + arg.a_5[s] * out;
        } else if (mykernel_type != INTERIOR_KERNEL && active) {
          Vector y = arg.y(xs, my_spinor_parity);
          aggregate_external = xpay ? arg.a_5[s] * out : out;
          out = y + aggregate_external;
        }

        if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.y(xs, my_spinor_parity) = out;

        if (mykernel_type != INTERIOR_KERNEL && active) {
          Vector x = arg.out(xs, my_spinor_parity);
          out = x + aggregate_external;
        }

        bool complete = isComplete<mykernel_type>(arg, coord);

        if (complete && active) {
          constexpr bool sync = true;
          constexpr bool this_dagger = true;
          // Then we apply the second m5inv-dag
          out
            = variableInv<sync, this_dagger, shared, Vector, typename Arg::Dslash5Arg>(arg, out, my_spinor_parity, 0, s);
        }

      } else if (Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS
                 || Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS_PRE_M5_MOB) {

        /******
         *  Apply M5mob:
         *    this is actually out = m5mob * x - kappa_b^2 * D4 * in (Dslash5Type::DSLASH5_MOBIUS)
         *    or               out = m5mob * x - kappa_b^2 * m5pre *D4 * in (Dslash5Type::DSLASH5_PRE_MOBIUS_M5_MOBIUS)
         */

        if (active) {
          if (Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS) { out = stencil_out; }

          if (Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS_PRE_M5_MOB) {
            constexpr bool sync = false;
            out = d5<sync, dagger, shared, Vector, typename Arg::Dslash5Arg, Dslash5Type::DSLASH5_MOBIUS_PRE>(
              arg, stencil_out, my_spinor_parity, 0, s);
          }
        }

        if (xpay && mykernel_type == INTERIOR_KERNEL) {
          Vector x = arg.x(xs, my_spinor_parity);
          constexpr bool sync_m5mob = Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS ? false : true;
          x = d5<sync_m5mob, dagger, shared, Vector, typename Arg::Dslash5Arg, Dslash5Type::DSLASH5_MOBIUS>(
            arg, x, my_spinor_parity, 0, s);
          out = x + arg.a_5[s] * out;
        } else if (mykernel_type != INTERIOR_KERNEL && active) {
          Vector x = arg.out(xs, my_spinor_parity);
          out = x + (xpay ? arg.a_5[s] * out : out);
        }

      } else {

        if ((Arg::dslash5_type == Dslash5Type::M5_INV_MOBIUS_M5_PRE
             || Arg::dslash5_type == Dslash5Type::M5_PRE_MOBIUS_M5_INV)
            && active) {
          out = stencil_out;
        }

        if (Arg::dslash5_type == Dslash5Type::M5_INV_MOBIUS) {
          // Apply the m5inv.
          constexpr bool sync = false;
          out = variableInv<sync, dagger, shared, Vector, typename Arg::Dslash5Arg>(arg, stencil_out, my_spinor_parity,
                                                                                    0, s);
        }

        if (xpay && mykernel_type == INTERIOR_KERNEL) {
          Vector x = arg.x(xs, my_spinor_parity);
          out = x + arg.a_5[s] * out;
        } else if (mykernel_type != INTERIOR_KERNEL && active) {
          Vector x = arg.out(xs, my_spinor_parity);
          out = x + (xpay ? arg.a_5[s] * out : out);
        }

        bool complete = isComplete<mykernel_type>(arg, coord);
        if (complete && active) {

          /******
           *  First apply M5inv, and then M5pre
           */
          if (Arg::dslash5_type == Dslash5Type::M5_INV_MOBIUS_M5_PRE) {
            // Apply the m5inv.
            constexpr bool sync_m5inv = false;
            out = variableInv<sync_m5inv, dagger, shared, Vector, typename Arg::Dslash5Arg>(arg, out, my_spinor_parity,
                                                                                            0, s);
            // Apply the m5pre.
            constexpr bool sync_m5pre = true;
            out = d5<sync_m5pre, dagger, shared, Vector, typename Arg::Dslash5Arg>(arg, out, my_spinor_parity, 0, s);
          }

          /******
           *  First apply M5pre, and then M5inv
           */
          if (Arg::dslash5_type == Dslash5Type::M5_PRE_MOBIUS_M5_INV) {
            // Apply the m5pre.
            constexpr bool sync_m5pre = false;
            out = d5<sync_m5pre, dagger, shared, Vector, typename Arg::Dslash5Arg>(arg, out, my_spinor_parity, 0, s);
            // Apply the m5inv.
            constexpr bool sync_m5inv = true;
            out = variableInv<sync_m5inv, dagger, shared, Vector, typename Arg::Dslash5Arg>(arg, out, my_spinor_parity,
                                                                                            0, s);
          }
        }
      }
      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(xs, my_spinor_parity) = out;
    }
  };

} // namespace quda
