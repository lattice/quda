#include <color_spinor_field.h>
#include <device_vector.h>
#include <tunable_nd.h>
#include <madwf_transfer.h>
#include <kernels/madwf_transfer.cuh>

namespace quda
{
  namespace madwf_ml
  {

    template <class storage_t, class matrix_t> class transfer_5D_wrapper : public TunableKernel3D
    {
      ColorSpinorField &out;
      const ColorSpinorField &in;
      const matrix_t *wm_p;
      bool dagger;

    private:
      unsigned int sharedBytesPerThread() const { return 0; }

      unsigned int sharedBytesPerBlock(const TuneParam &) const { return out.X(4) * in.X(4) * sizeof(matrix_t); }

      unsigned int minThreads() const { return out.VolumeCB() / out.X(4); }

    public:
      transfer_5D_wrapper(ColorSpinorField &out, const ColorSpinorField &in, const matrix_t *wm_p, bool dagger) :
        TunableKernel3D(out, out.X(4), out.SiteSubset()), out(out), in(in), wm_p(wm_p), dagger(dagger)
      {
        TunableKernel2D_base<false>::resizeStep(out.X(4)); // Ls must be contained in the block

        char tmp[512];
        sprintf(tmp, ",%02d->%02d", in.X(4), out.X(4));
        strcat(aux, tmp);
        strcat(aux, ",transfer_5D");
        if (dagger) strcat(aux, ",Dagger");

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (dagger) {
          launch<Transfer5D>(tp, stream, Transfer5DArg<storage_t, matrix_t, true>(out, in, wm_p));
        } else {
          launch<Transfer5D>(tp, stream, Transfer5DArg<storage_t, matrix_t, false>(out, in, wm_p));
        }
      }

      long long flops() const { return 8ll * out.X(4) * 4ll * in.VolumeCB(); }
      long long bytes() const { return in.Bytes() + out.Bytes(); }
    };

#ifdef GPU_DOMAIN_WALL_DIRAC
    template <class transfer_float, transfer_5D_t transfer_t>
    void transfer_5d_hh(ColorSpinorField &out, const ColorSpinorField &in, const device_vector<float> &tp, bool dagger)
    {
      using matrix_t = typename transfer_5D_mapper<transfer_float, transfer_t>::type;

      checkLocation(out, in); // check all locations match

      if (in.SiteSubset() != QUDA_PARITY_SITE_SUBSET || out.SiteSubset() != QUDA_PARITY_SITE_SUBSET) {
        errorQuda("ColorSpinorFields are not single parity: in = %d, out = %d",
            in.SiteSubset(), out.SiteSubset());
      }

      size_t m_size = in.X(4) * out.X(4) * sizeof(matrix_t);
      if (tp.size() * sizeof(float) != m_size) {
        errorQuda("Training Parameter size mismatch %lu neq %lu.\n", tp.size() * sizeof(float), m_size);
      }

      switch (checkPrecision(out, in)) {
      case QUDA_HALF_PRECISION: {
        transfer_5D_wrapper<short, matrix_t> w(out, in, reinterpret_cast<const matrix_t *>(tp.data()), dagger);
      } break;
      case QUDA_QUARTER_PRECISION: {
        transfer_5D_wrapper<int8_t, matrix_t> w(out, in, reinterpret_cast<const matrix_t *>(tp.data()), dagger);
      } break;

      default: errorQuda("Unsupported precision %d\n", in.Precision());
      }
    }
#else
    template <class transfer_float, transfer_5D_t transfer_t>
    void transfer_5d_hh(ColorSpinorField &, const ColorSpinorField &, const device_vector<float> &, bool)
    {
      errorQuda("Mobius dslash has not been built");
    }
#endif

    template void transfer_5d_hh<float, transfer_5D_t::Spin>(ColorSpinorField &, const ColorSpinorField &,
                                                                const device_vector<float> &, bool);

  } // namespace madwf_ml
} // namespace quda
