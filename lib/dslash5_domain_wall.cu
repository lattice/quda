#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/dslash_domain_wall_m5.cuh>

namespace quda
{

  template <typename Float, int nColor> class Dslash5 : public TunableKernel3D
  {
    cvector_ref<ColorSpinorField> &out;
    cvector_ref<const ColorSpinorField> &in;
    cvector_ref<const ColorSpinorField> &x;
    double m_f;
    double m_5;
    const Complex *b_5;
    const Complex *c_5;
    double a;
    bool dagger;
    bool xpay;
    Dslash5Type type;

    long long flops() const
    {
      long long Ls = in.X(4);
      long long bulk = (Ls - 2) * (in.Volume() / Ls);
      long long wall = 2 * in.Volume() / Ls;
      long long n = in.Ncolor() * in.Nspin();

      long long flops_ = 0;
      switch (type) {
      case Dslash5Type::DSLASH5_DWF: flops_ = n * (8ll * bulk + 10ll * wall + (xpay ? 4ll * in.Volume() : 0)); break;
      case Dslash5Type::DSLASH5_MOBIUS_PRE:
        flops_ = n * (8ll * bulk + 10ll * wall + 14ll * in.Volume() + (xpay ? 8ll * in.Volume() : 0));
        break;
      case Dslash5Type::DSLASH5_MOBIUS:
        flops_ = n * (8ll * bulk + 10ll * wall + 8ll * in.Volume() + (xpay ? 8ll * in.Volume() : 0));
        break;
      case Dslash5Type::M5_INV_DWF:
      case Dslash5Type::M5_INV_MOBIUS: // FIXME flops
        flops_ = ((2 + 8 * n) * Ls + (xpay ? 4ll : 0)) * in.Volume();
        break;
      case Dslash5Type::M5_INV_ZMOBIUS: flops_ = ((12 + 16 * n) * Ls + (xpay ? 8ll : 0)) * in.Volume(); break;
      default: errorQuda("Unexpected Dslash5Type %d", static_cast<int>(type));
      }

      return in.size() * flops_;
    }

    long long bytes() const
    {
      long long Ls = in.X(4);
      size_t bytes = 0u;
      switch (type) {
      case Dslash5Type::DSLASH5_DWF: bytes = out.Bytes() + 2 * in.Bytes() + (xpay ? x.Bytes() : 0); break;
      case Dslash5Type::DSLASH5_MOBIUS_PRE: bytes = out.Bytes() + 3 * in.Bytes() + (xpay ? x.Bytes() : 0); break;
      case Dslash5Type::DSLASH5_MOBIUS: bytes = out.Bytes() + 3 * in.Bytes() + (xpay ? x.Bytes() : 0); break;
      case Dslash5Type::M5_INV_DWF: bytes = out.Bytes() + Ls * in.Bytes() + (xpay ? x.Bytes() : 0); break;
      case Dslash5Type::M5_INV_MOBIUS: bytes = out.Bytes() + Ls * in.Bytes() + (xpay ? x.Bytes() : 0); break;
      case Dslash5Type::M5_INV_ZMOBIUS: bytes = out.Bytes() + Ls * in.Bytes() + (xpay ? x.Bytes() : 0); break;
      default: errorQuda("Unexpected Dslash5Type %d", static_cast<int>(type));
      }
      return bytes;
    }

    unsigned int minThreads() const { return in.VolumeCB() / in.X(4); }
    int blockStep() const { return 4; }
    int blockMin() const { return 4; }
    unsigned int sharedBytesPerThread() const
    {
      if (mobius_m5::shared()
          && (type == Dslash5Type::M5_INV_DWF || type == Dslash5Type::M5_INV_MOBIUS
              || type == Dslash5Type::M5_INV_ZMOBIUS)) {
        // spin components in shared depend on inversion algorithm
        bool isInv = type == Dslash5Type::M5_INV_DWF || type == Dslash5Type::M5_INV_MOBIUS
          || type == Dslash5Type::M5_INV_ZMOBIUS;
        int nSpin = (!isInv || mobius_m5::var_inverse()) ? mobius_m5::use_half_vector() ? in.Nspin() / 2 : in.Nspin() :
                                                           in.Nspin();
        return 2 * nSpin * nColor * sizeof(typename mapper<Float>::type);
      } else {
        return 0;
      }
    }

    // overloaded to return max dynamic shared memory if doing shared-memory inverse
    unsigned int maxSharedBytesPerBlock() const
    {
      if (mobius_m5::shared()
          && (type == Dslash5Type::M5_INV_DWF || type == Dslash5Type::M5_INV_MOBIUS
              || type == Dslash5Type::M5_INV_ZMOBIUS)) {
        return maxDynamicSharedBytesPerBlock();
      } else {
        return TunableKernel3D::maxSharedBytesPerBlock();
      }
    }

  public:
    Dslash5(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
            cvector_ref<const ColorSpinorField> &x, double m_f, double m_5, const Complex *b_5, const Complex *c_5,
            double a, bool dagger, Dslash5Type type) :
      TunableKernel3D(in[0], in.size() * in.X(4), in.SiteSubset()),
      out(out),
      in(in),
      x(x),
      m_f(m_f),
      m_5(m_5),
      b_5(b_5),
      c_5(c_5),
      a(a),
      dagger(dagger),
      xpay(a == 0.0 ? false : true),
      type(type)
    {
      if (mobius_m5::shared()
          && (type == Dslash5Type::M5_INV_DWF || type == Dslash5Type::M5_INV_MOBIUS
              || type == Dslash5Type::M5_INV_ZMOBIUS)) {
        // Ls must be contained in the block and different fields on different blocks
        resizeStep(in.X(4), 1);
        tune_block_y = false;
      }

      if (dagger) strcat(aux, ",Dagger");
      if (xpay) strcat(aux, ",xpay");
      switch (type) {
      case Dslash5Type::DSLASH5_DWF: strcat(aux, ",Dslash5Type::DSLASH5_DWF"); break;
      case Dslash5Type::DSLASH5_MOBIUS_PRE: strcat(aux, ",Dslash5Type::DSLASH5_MOBIUS_PRE"); break;
      case Dslash5Type::DSLASH5_MOBIUS: strcat(aux, ",Dslash5Type::DSLASH5_MOBIUS"); break;
      case Dslash5Type::M5_INV_DWF: strcat(aux, ",Dslash5Type::M5_INV_DWF"); break;
      case Dslash5Type::M5_INV_MOBIUS: strcat(aux, ",Dslash5Type::M5_INV_MOBIUS"); break;
      case Dslash5Type::M5_INV_ZMOBIUS: strcat(aux, ",Dslash5Type::M5_INV_ZMOBIUS"); break;
      default: errorQuda("Unexpected Dslash5Type %d", static_cast<int>(type));
      }
      setRHSstring(aux, in.size());

      apply(device::get_default_stream());
    }

    template <bool dagger, bool xpay, Dslash5Type type> using Arg = Dslash5Arg<Float, nColor, dagger, xpay, type>;

    template <Dslash5Type type, template <typename> class F>
    void Launch(TuneParam &tp, const qudaStream_t &stream)
    {
      if (xpay)
        dagger ?
          launch<F>(tp, stream, Arg<true, true, type>(out, in, x, m_f, m_5, b_5, c_5, a)) :
          launch<F>(tp, stream, Arg<false, true, type>(out, in, x, m_f, m_5, b_5, c_5, a));
      else
        dagger ?
          launch<F>(tp, stream, Arg<true, false, type>(out, in, x, m_f, m_5, b_5, c_5, a)) :
          launch<F>(tp, stream, Arg<false, false, type>(out, in, x, m_f, m_5, b_5, c_5, a));
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (mobius_m5::shared()
          && (type == Dslash5Type::M5_INV_DWF || type == Dslash5Type::M5_INV_MOBIUS
              || type == Dslash5Type::M5_INV_ZMOBIUS)) {
        tp.set_max_shared_bytes = true; // if inverse kernel uses shared memory then maximize total shared memory pool
      }

      switch (type) {
      case Dslash5Type::DSLASH5_DWF: Launch<Dslash5Type::DSLASH5_DWF, dslash5>(tp, stream); break;
      case Dslash5Type::DSLASH5_MOBIUS_PRE: Launch<Dslash5Type::DSLASH5_MOBIUS_PRE, dslash5>(tp, stream); break;
      case Dslash5Type::DSLASH5_MOBIUS: Launch<Dslash5Type::DSLASH5_MOBIUS, dslash5>(tp, stream); break;
      case Dslash5Type::M5_INV_DWF: Launch<Dslash5Type::M5_INV_DWF, dslash5inv>(tp, stream); break;
      case Dslash5Type::M5_INV_MOBIUS: Launch<Dslash5Type::M5_INV_MOBIUS, dslash5inv>(tp, stream); break;
      case Dslash5Type::M5_INV_ZMOBIUS: Launch<Dslash5Type::M5_INV_ZMOBIUS, dslash5inv>(tp, stream); break;
      default: errorQuda("Unexpected Dslash5Type %d", static_cast<int>(type));
      }
    }
  };

  // Apply the 5th dimension dslash operator to a colorspinor field
  // out = Dslash5*in
  void ApplyDslash5(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                    cvector_ref<const ColorSpinorField> &x, double m_f, double m_5, const Complex *b_5,
                    const Complex *c_5, double a, bool dagger, Dslash5Type type)
  {
    if (is_enabled<QUDA_DOMAIN_WALL_4D_DSLASH>()) {
      if (in.PCType() != QUDA_4D_PC) errorQuda("Only 4-d preconditioned fields are supported");
      checkLocation(out, in, x); // check all locations match
      instantiate_recurse3<Dslash5>(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type);
    } else {
      errorQuda("Domain wall operator has not been built");
    }
  }

} // namespace quda
