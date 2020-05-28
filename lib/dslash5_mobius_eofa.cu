#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <dslash_quda.h>
#include <index_helper.cuh>

#include <kernels/dslash_mobius_eofa.cuh>

namespace quda
{
  namespace mobius_eofa
  {
    template <typename storage_type, int nColor, typename Arg> class Dslash5 : public TunableVectorYZ
    {
    protected:
      Arg &arg;
      const ColorSpinorField &meta;
      static constexpr bool shared = true; // whether to use shared memory cache blocking for M5inv

      long long flops() const
      {
        // FIXME: Fix the flop count
        long long Ls = meta.X(4);
        long long bulk = (Ls - 2) * (meta.Volume() / Ls);
        long long wall = 2 * meta.Volume() / Ls;
        long long n = meta.Ncolor() * meta.Nspin();

        long long flops_ = 0;
        switch (arg.type) {
        case M5_EOFA:
        case M5INV_EOFA: flops_ = n * (8ll * bulk + 10ll * wall + (arg.xpay ? 4ll * meta.Volume() : 0)); break;
        default: errorQuda("Unknown Dslash5Type %d for EOFA", arg.type);
        }

        return flops_;
      }

      long long bytes() const
      {
        long long Ls = meta.X(4);
        switch (arg.type) {
        case M5_EOFA:
        case M5INV_EOFA: return arg.out.Bytes() + 2 * arg.in.Bytes() + (arg.xpay ? arg.x.Bytes() : 0);
        default: errorQuda("Unknown Dslash5Type %d for EOFA", arg.type);
        }
        return 0ll;
      }

      bool tuneGridDim() const { return false; }
      unsigned int minThreads() const { return arg.volume_4d_cb; }
      int blockStep() const { return 4; }
      int blockMin() const { return 4; }
      unsigned int sharedBytesPerThread() const
      {
        // spin components in shared depend on inversion algorithm
        int nSpin = meta.Nspin();
        return 2 * nSpin * nColor * sizeof(typename mapper<storage_type>::type);
      }

      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      // overloaded to return max dynamic shared memory if doing shared-memory
      // inverse
      unsigned int maxSharedBytesPerBlock() const
      {
        if (shared && (arg.type == M5_EOFA || arg.type == M5INV_EOFA)) {
          return maxDynamicSharedBytesPerBlock();
        } else {
          return TunableVectorYZ::maxSharedBytesPerBlock();
        }
      }

    public:
      Dslash5(Arg &arg, const ColorSpinorField &meta) : TunableVectorYZ(arg.Ls, arg.nParity), arg(arg), meta(meta)
      {
        TunableVectorY::resizeStep(arg.Ls);
        strcpy(aux, meta.AuxString());
        if (arg.dagger) strcat(aux, ",Dagger");
        if (arg.xpay) strcat(aux, ",xpay");
        if (arg.eofa_pm) {
          strcat(aux, ",eofa_plus");
        } else {
          strcat(aux, ",eofa_minus");
        }
        switch (arg.type) {
        case M5_EOFA: strcat(aux, ",mobius_M5_EOFA"); break;
        case M5INV_EOFA: strcat(aux, ",mobius_M5INV_EOFA"); break;
        default: errorQuda("Unknown Dslash5Type %d", arg.type);
        }
      }
      virtual ~Dslash5() { }

      template <typename T> inline void launch(T *f, const TuneParam &tp, Arg &arg, const qudaStream_t &stream)
      {
        if (shared && (arg.type == M5_EOFA || arg.type == M5INV_EOFA)) {
          // if inverse kernel uses shared memory then maximize total shared memory
          setMaxDynamicSharedBytesPerBlock(f);
        }
        void *args[] = {&arg};
        qudaLaunchKernel((const void *)f, tp.grid, tp.block, args, tp.shared_bytes, stream);
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (arg.type == M5_EOFA) {
          if (arg.eofa_pm) {
            if (arg.xpay) {
              arg.dagger ? launch(dslash5GPU<storage_type, nColor, true, true, true, M5_EOFA, Arg>, tp, arg, stream) :
                           launch(dslash5GPU<storage_type, nColor, false, true, true, M5_EOFA, Arg>, tp, arg, stream);
            } else {
              arg.dagger ? launch(dslash5GPU<storage_type, nColor, true, true, false, M5_EOFA, Arg>, tp, arg, stream) :
                           launch(dslash5GPU<storage_type, nColor, false, true, false, M5_EOFA, Arg>, tp, arg, stream);
            }
          } else {
            if (arg.xpay) {
              arg.dagger ? launch(dslash5GPU<storage_type, nColor, true, false, true, M5_EOFA, Arg>, tp, arg, stream) :
                           launch(dslash5GPU<storage_type, nColor, false, false, true, M5_EOFA, Arg>, tp, arg, stream);
            } else {
              arg.dagger ? launch(dslash5GPU<storage_type, nColor, true, false, false, M5_EOFA, Arg>, tp, arg, stream) :
                           launch(dslash5GPU<storage_type, nColor, false, false, false, M5_EOFA, Arg>, tp, arg, stream);
            }
          }
        } else if (arg.type == M5INV_EOFA) {
          if (arg.eofa_pm) {
            if (arg.xpay) {
              arg.dagger ? launch(dslash5GPU<storage_type, nColor, true, true, true, M5INV_EOFA, Arg>, tp, arg, stream) :
                           launch(dslash5GPU<storage_type, nColor, false, true, true, M5INV_EOFA, Arg>, tp, arg, stream);
            } else {
              arg.dagger ?
                launch(dslash5GPU<storage_type, nColor, true, true, false, M5INV_EOFA, Arg>, tp, arg, stream) :
                launch(dslash5GPU<storage_type, nColor, false, true, false, M5INV_EOFA, Arg>, tp, arg, stream);
            }
          } else {
            if (arg.xpay) {
              arg.dagger ?
                launch(dslash5GPU<storage_type, nColor, true, false, true, M5INV_EOFA, Arg>, tp, arg, stream) :
                launch(dslash5GPU<storage_type, nColor, false, false, true, M5INV_EOFA, Arg>, tp, arg, stream);
            } else {
              arg.dagger ?
                launch(dslash5GPU<storage_type, nColor, true, false, false, M5INV_EOFA, Arg>, tp, arg, stream) :
                launch(dslash5GPU<storage_type, nColor, false, false, false, M5INV_EOFA, Arg>, tp, arg, stream);
            }
          }
        } else {
          errorQuda("Unknown Dslash5Type %d", arg.type);
        }
      }

      void initTuneParam(TuneParam &param) const
      {
        TunableVectorYZ::initTuneParam(param);
        param.block.y = arg.Ls; // Ls must be contained in the block
        param.grid.y = 1;
        param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
      }

      void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

      TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    };

    template <typename storage_type, int nColor>
    void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
                      double m_5, const Complex *b_5, const Complex *c_5, double a, int eofa_pm, double inv,
                      double kappa, const double *eofa_u, const double *eofa_x, const double *eofa_y,
                      double sherman_morrison, bool dagger, Dslash5Type type)
    {
      Dslash5Arg<storage_type, nColor> arg(out, in, x, m_f, m_5, b_5, c_5, a, eofa_pm, inv, kappa, eofa_u, eofa_x,
                                           eofa_y, sherman_morrison, dagger, type);
      Dslash5<storage_type, nColor, Dslash5Arg<storage_type, nColor>> dslash(arg, in);
      dslash.apply(streams[Nstream - 1]);
    }

    // template on the number of colors
    template <typename storage_type>
    void ApplyDslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
                      double m_5, const Complex *b_5, const Complex *c_5, double a, int eofa_pm, double inv,
                      double kappa, const double *eofa_u, const double *eofa_x, const double *eofa_y,
                      double sherman_morrison, bool dagger, Dslash5Type type)
    {
      switch (in.Ncolor()) {
      case 3:
        ApplyDslash5<storage_type, 3>(out, in, x, m_f, m_5, b_5, c_5, a, eofa_pm, inv, kappa, eofa_u, eofa_x, eofa_y,
                                      sherman_morrison, dagger, type);
        break;
      default: errorQuda("Unsupported number of colors %d\n", in.Ncolor());
      }
    }

    // Apply the 5th dimension dslash operator to a colorspinor field
    // out = Dslash5*in
    void apply_dslash5(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &x, double m_f,
                       double m_5, const Complex *b_5, const Complex *c_5, double a, int eofa_pm, double inv,
                       double kappa, const double *eofa_u, const double *eofa_x, const double *eofa_y,
                       double sherman_morrison, bool dagger, Dslash5Type type)
    {
#ifdef GPU_DOMAIN_WALL_DIRAC
      checkLocation(out, in); // check all locations match

      switch (checkPrecision(out, in)) {
      case QUDA_DOUBLE_PRECISION:
        ApplyDslash5<double>(out, in, x, m_f, m_5, b_5, c_5, a, eofa_pm, inv, kappa, eofa_u, eofa_x, eofa_y,
                             sherman_morrison, dagger, type);
        break;
      case QUDA_SINGLE_PRECISION:
        ApplyDslash5<float>(out, in, x, m_f, m_5, b_5, c_5, a, eofa_pm, inv, kappa, eofa_u, eofa_x, eofa_y,
                            sherman_morrison, dagger, type);
        break;
      case QUDA_HALF_PRECISION:
        ApplyDslash5<short>(out, in, x, m_f, m_5, b_5, c_5, a, eofa_pm, inv, kappa, eofa_u, eofa_x, eofa_y,
                            sherman_morrison, dagger, type);
        break;
      default: errorQuda("Unsupported precision %d\n", in.Precision());
      }
#else
      errorQuda("Mobius EOFA dslash has not been built");
#endif
    }
  } // namespace mobius_eofa
} // namespace quda
