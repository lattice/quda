#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <dslash_quda.h>
#include <index_helper.cuh>

#include <kernels/dslash_domain_wall_m5.cuh>

namespace quda
{

  template <typename Float, int nColor, typename Arg> class Dslash5 : public TunableVectorYZ
  {

protected:
    Arg &arg;
    const ColorSpinorField &meta;
    static constexpr bool shared = true; // whether to use shared memory cache blocking for M5inv

    /** Whether to use variable or fixed coefficient algorithm.  Must be true if using ZMOBIUS */
    static constexpr bool var_inverse = true;

    long long flops() const
    {
      long long Ls = meta.X(4);
      long long bulk = (Ls - 2) * (meta.Volume() / Ls);
      long long wall = 2 * meta.Volume() / Ls;
      long long n = meta.Ncolor() * meta.Nspin();

      long long flops_ = 0;
      switch (arg.type) {
      case DSLASH5_DWF: flops_ = n * (8ll * bulk + 10ll * wall + (arg.xpay ? 4ll * meta.Volume() : 0)); break;
      case DSLASH5_MOBIUS_PRE:
        flops_ = n * (8ll * bulk + 10ll * wall + 14ll * meta.Volume() + (arg.xpay ? 8ll * meta.Volume() : 0));
        break;
      case DSLASH5_MOBIUS:
        flops_ = n * (8ll * bulk + 10ll * wall + 8ll * meta.Volume() + (arg.xpay ? 8ll * meta.Volume() : 0));
        break;
      case M5_INV_DWF:
      case M5_INV_MOBIUS: // FIXME flops
        flops_ = ((2 + 8 * n) * Ls + (arg.xpay ? 4ll : 0)) * meta.Volume();
        break;
      case M5_INV_ZMOBIUS: flops_ = ((12 + 16 * n) * Ls + (arg.xpay ? 8ll : 0)) * meta.Volume(); break;
      default: errorQuda("Unknown Dslash5Type %d", arg.type);
      }

      return flops_;
    }

    long long bytes() const
    {
      long long Ls = meta.X(4);
      switch (arg.type) {
      case DSLASH5_DWF: return arg.out.Bytes() + 2 * arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      case DSLASH5_MOBIUS_PRE: return arg.out.Bytes() + 3 * arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      case DSLASH5_MOBIUS: return arg.out.Bytes() + 3 * arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      case M5_INV_DWF: return arg.out.Bytes() + Ls * arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      case M5_INV_MOBIUS: return arg.out.Bytes() + Ls * arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      case M5_INV_ZMOBIUS: return arg.out.Bytes() + Ls * arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
      default: errorQuda("Unknown Dslash5Type %d", arg.type);
      }
      return 0ll;
    }

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volume_4d_cb; }
    int blockStep() const { return 4; }
    int blockMin() const { return 4; }
    unsigned int sharedBytesPerThread() const
    {
      if (shared && (arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS || arg.type == M5_INV_ZMOBIUS)) {
        // spin components in shared depend on inversion algorithm
        int nSpin = var_inverse ? meta.Nspin() / 2 : meta.Nspin();
        return 2 * nSpin * nColor * sizeof(typename mapper<Float>::type);
      } else {
        return 0;
      }
    }

    // overloaded to return max dynamic shared memory if doing shared-memory inverse
    unsigned int maxSharedBytesPerBlock() const
    {
      if (shared && (arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS || arg.type == M5_INV_ZMOBIUS)) {
        return maxDynamicSharedBytesPerBlock();
      } else {
        return TunableVectorYZ::maxSharedBytesPerBlock();
      }
    }

public:
    Dslash5(Arg &arg, const ColorSpinorField &meta) : TunableVectorYZ(arg.Ls, arg.nParity), arg(arg), meta(meta)
    {
      strcpy(aux, meta.AuxString());
      if (arg.dagger) strcat(aux, ",Dagger");
      if (arg.xpay) strcat(aux, ",xpay");
      switch (arg.type) {
      case DSLASH5_DWF: strcat(aux, ",DSLASH5_DWF"); break;
      case DSLASH5_MOBIUS_PRE: strcat(aux, ",DSLASH5_MOBIUS_PRE"); break;
      case DSLASH5_MOBIUS: strcat(aux, ",DSLASH5_MOBIUS"); break;
      case M5_INV_DWF: strcat(aux, ",M5_INV_DWF"); break;
      case M5_INV_MOBIUS: strcat(aux, ",M5_INV_MOBIUS"); break;
      case M5_INV_ZMOBIUS: strcat(aux, ",M5_INV_ZMOBIUS"); break;
      default: errorQuda("Unknown Dslash5Type %d", arg.type);
      }
    }
    virtual ~Dslash5() {}

    template <typename T> inline void launch(T *f, const TuneParam &tp, Arg &arg, const qudaStream_t &stream)
    {
      if (shared && (arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS || arg.type == M5_INV_ZMOBIUS)) {
        // if inverse kernel uses shared memory then maximize total shared memory pool
        setMaxDynamicSharedBytesPerBlock(f);
      }
      void *args[] = {&arg};
      qudaLaunchKernel((const void *)f, tp.grid, tp.block, args, tp.shared_bytes, stream);
    }

    void apply(const qudaStream_t &stream)
    {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("CPU variant not instantiated");
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (arg.type == DSLASH5_DWF) {
          if (arg.xpay)
            arg.dagger ? launch(dslash5GPU<Float, nColor, true, true, DSLASH5_DWF, Arg>, tp, arg, stream) :
                         launch(dslash5GPU<Float, nColor, false, true, DSLASH5_DWF, Arg>, tp, arg, stream);
          else
            arg.dagger ? launch(dslash5GPU<Float, nColor, true, false, DSLASH5_DWF, Arg>, tp, arg, stream) :
                         launch(dslash5GPU<Float, nColor, false, false, DSLASH5_DWF, Arg>, tp, arg, stream);
        } else if (arg.type == DSLASH5_MOBIUS_PRE) {
          if (arg.xpay)
            arg.dagger ? launch(dslash5GPU<Float, nColor, true, true, DSLASH5_MOBIUS_PRE, Arg>, tp, arg, stream) :
                         launch(dslash5GPU<Float, nColor, false, true, DSLASH5_MOBIUS_PRE, Arg>, tp, arg, stream);
          else
            arg.dagger ? launch(dslash5GPU<Float, nColor, true, false, DSLASH5_MOBIUS_PRE, Arg>, tp, arg, stream) :
                         launch(dslash5GPU<Float, nColor, false, false, DSLASH5_MOBIUS_PRE, Arg>, tp, arg, stream);
        } else if (arg.type == DSLASH5_MOBIUS) {
          if (arg.xpay)
            arg.dagger ? launch(dslash5GPU<Float, nColor, true, true, DSLASH5_MOBIUS, Arg>, tp, arg, stream) :
                         launch(dslash5GPU<Float, nColor, false, true, DSLASH5_MOBIUS, Arg>, tp, arg, stream);
          else
            arg.dagger ? launch(dslash5GPU<Float, nColor, true, false, DSLASH5_MOBIUS, Arg>, tp, arg, stream) :
                         launch(dslash5GPU<Float, nColor, false, false, DSLASH5_MOBIUS, Arg>, tp, arg, stream);
        } else if (arg.type == M5_INV_DWF) {
          if (arg.xpay)
            arg.dagger ?
                launch(dslash5invGPU<Float, nColor, true, true, M5_INV_DWF, shared, var_inverse, Arg>, tp, arg, stream) :
                launch(dslash5invGPU<Float, nColor, false, true, M5_INV_DWF, shared, var_inverse, Arg>, tp, arg, stream);
          else
            arg.dagger ?
                launch(dslash5invGPU<Float, nColor, true, false, M5_INV_DWF, shared, var_inverse, Arg>, tp, arg, stream) :
                launch(dslash5invGPU<Float, nColor, false, false, M5_INV_DWF, shared, var_inverse, Arg>, tp, arg, stream);
        } else if (arg.type == M5_INV_MOBIUS) {
          if (arg.xpay)
            arg.dagger ? launch(
                dslash5invGPU<Float, nColor, true, true, M5_INV_MOBIUS, shared, var_inverse, Arg>, tp, arg, stream) :
                         launch(dslash5invGPU<Float, nColor, false, true, M5_INV_MOBIUS, shared, var_inverse, Arg>, tp,
                             arg, stream);
          else
            arg.dagger ? launch(
                dslash5invGPU<Float, nColor, true, false, M5_INV_MOBIUS, shared, var_inverse, Arg>, tp, arg, stream) :
                         launch(dslash5invGPU<Float, nColor, false, false, M5_INV_MOBIUS, shared, var_inverse, Arg>, tp,
                             arg, stream);
        } else if (arg.type == M5_INV_ZMOBIUS) {
          if (arg.xpay)
            arg.dagger ? launch(
                dslash5invGPU<Float, nColor, true, true, M5_INV_ZMOBIUS, shared, var_inverse, Arg>, tp, arg, stream) :
                         launch(dslash5invGPU<Float, nColor, false, true, M5_INV_ZMOBIUS, shared, var_inverse, Arg>, tp,
                             arg, stream);
          else
            arg.dagger ? launch(
                dslash5invGPU<Float, nColor, true, false, M5_INV_ZMOBIUS, shared, var_inverse, Arg>, tp, arg, stream) :
                         launch(dslash5invGPU<Float, nColor, false, false, M5_INV_ZMOBIUS, shared, var_inverse, Arg>,
                             tp, arg, stream);
        }
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::initTuneParam(param);
      if (shared && (arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS || arg.type == M5_INV_ZMOBIUS)) {
        param.block.y = arg.Ls; // Ls must be contained in the block
        param.grid.y = 1;
        param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
      }
    }

    void defaultTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::defaultTuneParam(param);
      if (shared && (arg.type == M5_INV_DWF || arg.type == M5_INV_MOBIUS || arg.type == M5_INV_ZMOBIUS)) {
        param.block.y = arg.Ls; // Ls must be contained in the block
        param.grid.y = 1;
        param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };

  template <typename Float, int nColor>
  void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
      double m_5, const Complex *b_5, const Complex *c_5, double a, bool dagger, Dslash5Type type)
  {
    Dslash5Arg<Float, nColor> arg(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type);
    Dslash5<Float, nColor, Dslash5Arg<Float, nColor>> dslash(arg, in);
    dslash.apply(streams[Nstream - 1]);
  }

  // template on the number of colors
  template <typename Float>
  void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
      double m_5, const Complex *b_5, const Complex *c_5, double a, bool dagger, Dslash5Type type)
  {
    switch (in.Ncolor()) {
    case 3: ApplyDslash5<Float, 3>(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    default: errorQuda("Unsupported number of colors %d\n", in.Ncolor());
    }
  }

  // Apply the 5th dimension dslash operator to a colorspinor field
  // out = Dslash5*in
  void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
      double m_5, const Complex *b_5, const Complex *c_5, double a, bool dagger, Dslash5Type type)
  {
#ifdef GPU_DOMAIN_WALL_DIRAC
    if (in.PCType() != QUDA_4D_PC) errorQuda("Only 4-d preconditioned fields are supported");
    checkLocation(out, in); // check all locations match

    switch (checkPrecision(out, in)) {
    case QUDA_DOUBLE_PRECISION: ApplyDslash5<double>(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    case QUDA_SINGLE_PRECISION: ApplyDslash5<float>(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    case QUDA_HALF_PRECISION: ApplyDslash5<short>(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    case QUDA_QUARTER_PRECISION: ApplyDslash5<char>(out, in, x, m_f, m_5, b_5, c_5, a, dagger, type); break;
    default: errorQuda("Unsupported precision %d\n", in.Precision());
    }
#else
    errorQuda("Domain wall dslash has not been built");
#endif
  }

} // namespace quda
