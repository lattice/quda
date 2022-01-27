#include <color_spinor_field.h>
#include <device_vector.h>
#include <tunable_nd.h>
#include <madwf_transfer.h>
#include <kernels/madwf_transfer.cuh>
#include <madwf_ml.h>
#include <uint_to_char.h>

namespace quda
{
  namespace madwf_ml
  {

    template <class storage_t> class transfer_5D_wrapper : public TunableKernel3D
    {
      ColorSpinorField &out;
      const ColorSpinorField &in;
      const MadwfAcc::transfer_float *wm_p;
      bool dagger;
      using matrix_t = typename transfer_5D_mapper<MadwfAcc::transfer_float, 4, 3, MadwfAcc::transfer_t>::type;

      unsigned int sharedBytesPerThread() const { return 0; }

      unsigned int sharedBytesPerBlock(const TuneParam &) const { return out.X(4) * in.X(4) * sizeof(matrix_t); }

      unsigned int minThreads() const { return out.VolumeCB() / out.X(4); }

    public:
      transfer_5D_wrapper(ColorSpinorField &out, const ColorSpinorField &in, const MadwfAcc::transfer_float *wm_p,
                          bool dagger) :
        TunableKernel3D(out, out.X(4), out.SiteSubset()), out(out), in(in), wm_p(wm_p), dagger(dagger)
      {
        TunableKernel2D_base<false>::resizeStep(out.X(4)); // Ls must be contained in the block

        char in_str[16];
        char out_str[16];
        strcat(aux, ",");
        i32toa(in_str, in.X(4));
        strcat(aux, in_str);
        strcat(aux, ",");
        i32toa(out_str, out.X(4));
        strcat(aux, out_str);
        strcat(aux, ",transfer_5D");
        if (dagger) strcat(aux, ",Dagger");

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (dagger) {
          launch<Transfer5D>(tp, stream, Transfer5DArg<storage_t, 4, 3, true>(out, in, wm_p));
        } else {
          launch<Transfer5D>(tp, stream, Transfer5DArg<storage_t, 4, 3, false>(out, in, wm_p));
        }
      }

      long long flops() const { return 8ll * out.X(4) * 4ll * 12ll * in.VolumeCB(); }
      long long bytes() const { return in.Bytes() + out.Bytes(); }
    };

#ifdef GPU_DOMAIN_WALL_DIRAC
    void transfer_5d_hh(ColorSpinorField &out, const ColorSpinorField &in,
                        const device_vector<MadwfAcc::transfer_float> &tp, bool dagger)
    {
      checkLocation(out, in); // check all locations match

      if (in.SiteSubset() != QUDA_PARITY_SITE_SUBSET || out.SiteSubset() != QUDA_PARITY_SITE_SUBSET) {
        errorQuda("ColorSpinorFields are not single parity: in = %d, out = %d", in.SiteSubset(), out.SiteSubset());
      }

      if (out.Ndim() != 5 || in.Ndim() != 5) { errorQuda("we need a 5 dimensional field for this."); }
      if (out.Nspin() != 4 || in.Nspin() != 4)
        errorQuda("out.nSpin = %d, in.nSpin = %d not supported", out.Nspin(), in.Nspin());
      if (out.Ncolor() != 3 || in.Ncolor() != 3)
        errorQuda("out.nColor = %d, in.nColor = %d, not supported", out.Ncolor(), in.Ncolor());

      using matrix_t = typename transfer_5D_mapper<MadwfAcc::transfer_float, 4, 3, MadwfAcc::transfer_t>::type;
      size_t m_size = in.X(4) * out.X(4) * sizeof(matrix_t);
      if (tp.size() * sizeof(float) != m_size) {
        errorQuda("Training Parameter size mismatch %lu neq %lu.\n", tp.size() * sizeof(float), m_size);
      }

      instantiate_madwf<transfer_5D_wrapper>(out, in, tp.data(), dagger);
    }
#else
    void transfer_5d_hh(ColorSpinorField &, const ColorSpinorField &, const device_vector<MadwfAcc::transfer_float> &,
                        bool)
    {
      errorQuda("Mobius dslash has not been built");
    }
#endif

  } // namespace madwf_ml
} // namespace quda
