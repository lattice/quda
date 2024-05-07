#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <instantiate.h>
#include <tunable_nd.h>
#include <instantiate_dslash.h>
#include <kernels/dslash_mobius_eofa.cuh>

namespace quda
{
  namespace mobius_eofa
  {
    template <typename store_t, int nColor> class Dslash5 : public TunableKernel3D
    {
      cvector_ref<ColorSpinorField> &out;
      cvector_ref<const ColorSpinorField> &in;
      cvector_ref<const ColorSpinorField> &x;
      double m_f;
      double m_5;
      const Complex *b_5;
      const Complex *c_5;
      double a;
      bool eofa_pm;
      double inv;
      double kappa;
      const double *eofa_u;
      const double *eofa_x;
      const double *eofa_y;
      bool dagger;
      bool xpay;
      Dslash5Type type;
      double sherman_morrison;

      static constexpr bool shared = true; // whether to use shared memory cache blocking for M5inv

      long long flops() const
      {
        // FIXME: Fix the flop count
        long long Ls = in[0].X(4);
        long long bulk = (Ls - 2) * (in.Volume() / Ls);
        long long wall = 2 * in.Volume() / Ls;
        long long n = in.Ncolor() * in.Nspin();

        long long flops_ = 0;
        switch (type) {
        case Dslash5Type::M5_EOFA:
        case Dslash5Type::M5INV_EOFA: flops_ = n * (8ll * bulk + 10ll * wall + (xpay ? 4ll * in.Volume() : 0)); break;
        default: errorQuda("Unknown Dslash5Type %d for EOFA", static_cast<int>(type));
        }

        return in.size() * flops_;
      }

      long long bytes() const
      {
        size_t bytes = 0;
        switch (type) {
        case Dslash5Type::M5_EOFA:
        case Dslash5Type::M5INV_EOFA:
          bytes = in.size() * (out[0].Bytes() + 2 * in[0].Bytes() + (xpay ? x[0].Bytes() : 0));
          break;
        default: errorQuda("Unknown Dslash5Type %d for EOFA", static_cast<int>(type));
        }
        return in.size() * bytes;
        ;
      }

      unsigned int minThreads() const { return in[0].VolumeCB() / in[0].X(4); }
      int blockStep() const { return 4; }
      int blockMin() const { return 4; }
      unsigned int sharedBytesPerThread() const
      {
        // spin components in shared depend on inversion algorithm
        int nSpin = in.Nspin();
        return 2 * nSpin * nColor * sizeof(typename mapper<store_t>::type);
      }

      // overloaded to return max dynamic shared memory if doing shared-memory
      // inverse
      unsigned int maxSharedBytesPerBlock() const
      {
        if (shared && (type == Dslash5Type::M5_EOFA || type == Dslash5Type::M5INV_EOFA)) {
          return maxDynamicSharedBytesPerBlock();
        } else {
          return TunableKernel3D::maxSharedBytesPerBlock();
        }
      }

    public:
      Dslash5(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
              cvector_ref<const ColorSpinorField> &x, const double m_f, const double m_5, const Complex *b_5,
              const Complex *c_5, double a, bool eofa_pm, double inv, double kappa, const double *eofa_u,
              const double *eofa_x, const double *eofa_y, double sherman_morrison, bool dagger, Dslash5Type type) :
        TunableKernel3D(in[0], in.size() * in[0].X(4), in.SiteSubset()),
        out(out),
        in(in),
        x(x),
        m_f(m_f),
        m_5(m_5),
        b_5(b_5),
        c_5(c_5),
        a(a),
        eofa_pm(eofa_pm),
        inv(inv),
        kappa(kappa),
        eofa_u(eofa_u),
        eofa_x(eofa_x),
        eofa_y(eofa_y),
        dagger(dagger),
        xpay(a == 0.0 ? false : true),
        type(type),
        sherman_morrison(sherman_morrison)
      {
        // Ls must be contained in the block and different fields on different blocks
        resizeStep(in[0].X(4), 1);
        tune_block_y = false;

        if (dagger) strcat(aux, ",Dagger");
        if (xpay) strcat(aux, ",xpay");
        if (eofa_pm) {
          strcat(aux, ",eofa_plus");
        } else {
          strcat(aux, ",eofa_minus");
        }
        switch (type) {
        case Dslash5Type::M5_EOFA: strcat(aux, ",mobius_Dslash5Type::M5_EOFA"); break;
        case Dslash5Type::M5INV_EOFA: strcat(aux, ",mobius_Dslash5Type::M5INV_EOFA"); break;
        default: errorQuda("Unknown Dslash5Type %d", static_cast<int>(type));
        }
        setRHSstring(aux, in.size());

        apply(device::get_default_stream());
      }

      template <bool pm, bool dagger, bool xpay, Dslash5Type type> using Arg = Dslash5Arg<store_t, nColor, pm, dagger, xpay, type>;

      template <Dslash5Type type, template <typename> class F>
      void Launch(TuneParam &tp, const qudaStream_t &stream)
      {
        if (eofa_pm)
          if (xpay)
            dagger ?
              launch<F>(tp, stream, Arg<true, true, true, type>(out, in, x, m_f, m_5, b_5, c_5, a, inv, kappa, eofa_u, eofa_x, eofa_y, sherman_morrison)) :
              launch<F>(tp, stream, Arg<true, false, true, type>(out, in, x, m_f, m_5, b_5, c_5, a, inv, kappa, eofa_u, eofa_x, eofa_y, sherman_morrison));
          else
            dagger ?
              launch<F>(tp, stream, Arg<true, true, false, type>(out, in, x, m_f, m_5, b_5, c_5, a, inv, kappa, eofa_u, eofa_x, eofa_y, sherman_morrison)) :
              launch<F>(tp, stream, Arg<true, false, false, type>(out, in, x, m_f, m_5, b_5, c_5, a, inv, kappa, eofa_u, eofa_x, eofa_y, sherman_morrison));
        else
          if (xpay)
            dagger ?
              launch<F>(tp, stream, Arg<false, true, true, type>(out, in, x, m_f, m_5, b_5, c_5, a, inv, kappa, eofa_u, eofa_x, eofa_y, sherman_morrison)) :
              launch<F>(tp, stream, Arg<false, false, true, type>(out, in, x, m_f, m_5, b_5, c_5, a, inv, kappa, eofa_u, eofa_x, eofa_y, sherman_morrison));
          else
            dagger ?
              launch<F>(tp, stream, Arg<false, true, false, type>(out, in, x, m_f, m_5, b_5, c_5, a, inv, kappa, eofa_u, eofa_x, eofa_y, sherman_morrison)) :
              launch<F>(tp, stream, Arg<false, false, false, type>(out, in, x, m_f, m_5, b_5, c_5, a, inv, kappa, eofa_u, eofa_x, eofa_y, sherman_morrison));
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

        if (shared && (type == Dslash5Type::M5_EOFA || type == Dslash5Type::M5INV_EOFA)) {
          // if inverse kernel uses shared memory then maximize total shared memory
          tp.set_max_shared_bytes = true;
        }

        switch (type) {
        case Dslash5Type::M5_EOFA: Launch<Dslash5Type::M5_EOFA, eofa_dslash5>(tp, stream); break;
        case Dslash5Type::M5INV_EOFA: Launch<Dslash5Type::M5INV_EOFA, eofa_dslash5inv>(tp, stream); break;
        default: errorQuda("Unknown Dslash5Type %d", static_cast<int>(type));
        }
      }
    };

    // Apply the 5th dimension dslash operator to a colorspinor field
    // out = Dslash5*in
    void apply_dslash5(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                       cvector_ref<const ColorSpinorField> &x, double m_f, double m_5, const Complex *b_5,
                       const Complex *c_5, double a, int eofa_pm, double inv, double kappa, const double *eofa_u,
                       const double *eofa_x, const double *eofa_y, double sherman_morrison, bool dagger, Dslash5Type type)
    {
      if constexpr (is_enabled<QUDA_MOBIUS_DWF_EOFA_DSLASH>()) {
        checkLocation(out, in, x); // check all locations match
        instantiate<Dslash5>(out, in, x, m_f, m_5, b_5, c_5, a, eofa_pm, inv, kappa, eofa_u, eofa_x, eofa_y,
                             sherman_morrison, dagger, type);
      } else {
        errorQuda("Mobius EOFA operator has not been built");
      }
    }

  } // namespace mobius_eofa
} // namespace quda
