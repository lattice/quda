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

    F y[MAX_MULTI_RHS]; // The additional output field accessor

    static constexpr Dslash5Type dslash5_type = dslash5_type_;

    using Dslash5Arg = Dslash5Arg<Float, nColor, false, false, dslash5_type>;
    using Dslash5Arg::Ls;

    using real = typename mapper<Float>::type;
    complex<real> alpha;
    complex<real> beta;

    bool fuse_m5inv_m5pre;

    DomainWall4DFusedM5Arg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                           const ColorSpinorField &halo, const GaugeField &U, double a, double m_5, const Complex *b_5,
                           const Complex *c_5, bool xpay, cvector_ref<const ColorSpinorField> &x,
                           cvector_ref<ColorSpinorField> &y, int parity, bool dagger, const int *comm_override,
                           double m_f) :
      DomainWall4DArg(out, in, halo, U, a, m_5, b_5, c_5, xpay, x, parity, dagger, comm_override),
      Dslash5Arg(out, in, x, m_f, m_5, b_5, c_5, a)
    {
      for (auto i = 0u; i < y.size(); i++) this->y[i] = y[i];
      for (int s = 0; s < Ls; s++) {
        auto kappa_b_s = 0.5 / (b_5[s] * (m_5 + 4.0) + 1.0);
        a_5[s] = a * kappa_b_s * kappa_b_s;
      }; // 4-d Mobius
    }
  };

  constexpr bool domainWall4DFusedM5shared = true; // Use shared memory
  template <int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg_>
  struct domainWall4DFusedM5 : dslash_default, d5Params<Arg_,domainWall4DFusedM5shared>::Ops  {
    using Arg = Arg_;

    static constexpr Dslash5Type dslash5_type = Arg::type;
    static constexpr bool shared = domainWall4DFusedM5shared;

    const Arg &arg;
    using typename d5Params<Arg_,shared>::Ops::KernelOpsT;
    //constexpr domainWall4DFusedM5(const Arg &arg) : arg(arg) { }
    template <typename Ftor> constexpr domainWall4DFusedM5(const Ftor &ftor) : KernelOpsT(ftor), arg(ftor.arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    template <KernelType mykernel_type = kernel_type, bool allthreads = false>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_s, int parity, bool active = true)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      int src_idx = src_s / arg.Ls;
      int s = src_s % arg.Ls;

      int thread_dim; // which dimension is thread working on (fused kernel only)
      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, s, parity, thread_dim);

      const int my_spinor_parity = nParity == 2 ? parity : 0;
      Vector stencil_out;

      if (!allthreads || active) {
	active &= mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
	applyWilson<nParity, dagger, mykernel_type>(stencil_out, arg, coord, parity, idx, thread_dim, active, src_idx);
      }

      Vector out;

      // In the following `x_cb` are all passed as `x_cb = 0`, since it will not be used if `shared = true`, and `shared = true`

      //if (active) {

      /******
       *  Apply M5pre
       */
      if (Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS_PRE) {
	constexpr bool sync = false;
	//out = d5<allthreads, sync, dagger, shared>(*this, stencil_out, my_spinor_parity, 0, s, active);
	out = d5<true, sync, dagger, shared, Vector, decltype(*this), typename Arg::Dslash5Arg>(*this, stencil_out, my_spinor_parity, 0, s, src_idx, active);
      }
      //}

      int xs = coord.x_cb + s * arg.dc.volume_4d_cb;
      if (Arg::dslash5_type == Dslash5Type::M5_INV_MOBIUS_M5_INV_DAG) {

        /******
         *  Apply the two M5inv's:
         *    this is actually   y = 1 * x - kappa_b^2 * m5inv * D4 * in
         *                     out = m5inv-dagger * y
         */
        //if (active) {
	constexpr bool sync = false;
	//out = variableInv<allthreads, sync, dagger, shared>(*this, stencil_out, my_spinor_parity, 0, s, active);
	out = variableInv<true, sync, dagger, shared, Vector, decltype(*this), typename Arg::Dslash5Arg>(*this, stencil_out, my_spinor_parity, 0, s, src_idx, active);
        //}

        Vector aggregate_external;
        if (active) {
	  if (xpay && mykernel_type == INTERIOR_KERNEL) {
	    Vector x = arg.x[src_idx](xs, my_spinor_parity);
	    out = x + arg.a_5[s] * out;
	  } else if (mykernel_type != INTERIOR_KERNEL) {
	    Vector y = arg.y[src_idx](xs, my_spinor_parity);
	    aggregate_external = xpay ? arg.a_5[s] * out : out;
	    out = y + aggregate_external;
	  }

	  arg.y[src_idx](xs, my_spinor_parity) = out;

	  if (mykernel_type != INTERIOR_KERNEL) {
	    Vector x = arg.out[src_idx](xs, my_spinor_parity);
	    out = x + aggregate_external;
	  }
	}

        bool complete = isComplete<mykernel_type>(arg, coord);

        //if (complete) {
	{
	  bool act = active && complete;
          constexpr bool sync = true;
          constexpr bool this_dagger = true;
          // Then we apply the second m5inv-dag
          //out = variableInv<allthreads, sync, this_dagger, shared>(*this, out, my_spinor_parity, 0, s, active);
          //out = variableInv<true, sync, this_dagger, shared>(*this, out, my_spinor_parity, 0, s, act);
          auto tmp = variableInv<true, sync, this_dagger, shared, Vector, decltype(*this), typename Arg::Dslash5Arg>(*this, out, my_spinor_parity, 0, s, src_idx, act);
	  if (complete) out = tmp;
        }

      } else if (Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS
                 || Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS_PRE_M5_MOB) {

        /******
         *  Apply M5mob:
         *    this is actually out = m5mob * x - kappa_b^2 * D4 * in (Dslash5Type::DSLASH5_MOBIUS)
         *    or               out = m5mob * x - kappa_b^2 * m5pre *D4 * in (Dslash5Type::DSLASH5_PRE_MOBIUS_M5_MOBIUS)
         */

        //if (active) {
	if (Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS) { out = stencil_out; }

	if (Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS_PRE_M5_MOB) {
	  constexpr bool sync = false;
	  //out = d5<sync, dagger, shared, Vector, typename Arg::Dslash5Arg, Dslash5Type::DSLASH5_MOBIUS_PRE>(
	  //arg, stencil_out, my_spinor_parity, 0, s);
	  //out = d5<allthreads, sync, dagger, shared, Vector, std::remove_pointer_t<decltype(this)>, Dslash5Type::DSLASH5_MOBIUS_PRE>
	  //(*this, stencil_out, my_spinor_parity, 0, s, active);
	  out = d5<true, sync, dagger, shared, Vector, decltype(*this), typename Arg::Dslash5Arg, Dslash5Type::DSLASH5_MOBIUS_PRE>
	    (*this, stencil_out, my_spinor_parity, 0, s, src_idx, active);
	}
	//}

        if (xpay && mykernel_type == INTERIOR_KERNEL) {
          Vector x;
	  if(active) x = arg.x[src_idx](xs, my_spinor_parity);
          constexpr bool sync_m5mob = Arg::dslash5_type == Dslash5Type::DSLASH5_MOBIUS ? false : true;
          //x = d5<sync_m5mob, dagger, shared, Vector, typename Arg::Dslash5Arg, Dslash5Type::DSLASH5_MOBIUS>(
	  //arg, x, my_spinor_parity, 0, s);
          x = d5<true, sync_m5mob, dagger, shared, Vector, decltype(*this), typename Arg::Dslash5Arg, Dslash5Type::DSLASH5_MOBIUS>
	    (*this, x, my_spinor_parity, 0, s, src_idx, active);
          out = x + arg.a_5[s] * out;
        } else if (mykernel_type != INTERIOR_KERNEL && active) {
          Vector x = arg.out[src_idx](xs, my_spinor_parity);
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
          out = variableInv<true, sync, dagger, shared>(*this, stencil_out, my_spinor_parity, 0, s, src_idx, active);
        }

	if (active) {
	  if (xpay && mykernel_type == INTERIOR_KERNEL) {
	    Vector x = arg.x[src_idx](xs, my_spinor_parity);
	    out = x + arg.a_5[s] * out;
	  } else if (mykernel_type != INTERIOR_KERNEL) {
	    Vector x = arg.out[src_idx](xs, my_spinor_parity);
	    out = x + (xpay ? arg.a_5[s] * out : out);
	  }
	}

        bool complete = isComplete<mykernel_type>(arg, coord);
        //if (complete) {
	{
	  bool act = active && complete;

          /******
           *  First apply M5inv, and then M5pre
           */
          if (Arg::dslash5_type == Dslash5Type::M5_INV_MOBIUS_M5_PRE) {
            // Apply the m5inv.
            constexpr bool sync_m5inv = false;
            //out = variableInv<true, sync_m5inv, dagger, shared>(*this, out, my_spinor_parity, 0, s, act);
	    auto tmp = variableInv<true, sync_m5inv, dagger, shared>(*this, out, my_spinor_parity, 0, s, src_idx, act);
            // Apply the m5pre.
            constexpr bool sync_m5pre = true;
            //out = d5<true, sync_m5pre, dagger, shared>(*this, out, my_spinor_parity, 0, s, act);
            tmp = d5<true, sync_m5pre, dagger, shared>(*this, tmp, my_spinor_parity, 0, s, src_idx, act);
	    if (complete) out = tmp;
          }

          /******
           *  First apply M5pre, and then M5inv
           */
          if (Arg::dslash5_type == Dslash5Type::M5_PRE_MOBIUS_M5_INV) {
            // Apply the m5pre.
            constexpr bool sync_m5pre = false;
            //out = d5<true, sync_m5pre, dagger, shared>(*this, out, my_spinor_parity, 0, s, act);
	    auto tmp = d5<true, sync_m5pre, dagger, shared>(*this, out, my_spinor_parity, 0, s, src_idx, act);
            // Apply the m5inv.
            constexpr bool sync_m5inv = true;
            //out = variableInv<true, sync_m5inv, dagger, shared>(*this, out, my_spinor_parity, 0, s, act);
            tmp = variableInv<true, sync_m5inv, dagger, shared>(*this, tmp, my_spinor_parity, 0, s, src_idx, act);
	    if (complete) out = tmp;
          }
        }
      }
      if (active) arg.out[src_idx](xs, my_spinor_parity) = out;
    }
  };

} // namespace quda
