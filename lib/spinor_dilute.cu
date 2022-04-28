#include <color_spinor_field.h>
#include <kernels/spinor_dilute.cuh>
#include <tunable_nd.h>
#include <instantiate.h>

namespace quda {

  template <typename real, int Ns, int Nc>
  class SpinorDilute : TunableKernel2D {
    std::vector<ColorSpinorField> &v;
    const ColorSpinorField &src;
    QudaDilutionType type;
    unsigned int minThreads() const { return src.VolumeCB(); }

  public:
    SpinorDilute(const ColorSpinorField &src, std::vector<ColorSpinorField> &v, QudaDilutionType type) :
      TunableKernel2D(src, src.SiteSubset()),
      v(v),
      src(src),
      type(type)
    {
      switch (type) {
      case QUDA_DILUTION_SPIN: strcat(aux, ",spin_dilution"); break;
      case QUDA_DILUTION_COLOR: strcat(aux, ",color_dilution"); break;
      case QUDA_DILUTION_SPIN_COLOR: strcat(aux, ",spin_color_dilution"); break;
      case QUDA_DILUTION_SPIN_COLOR_EVEN_ODD: strcat(aux, ",spin_color_even_odd_dilution"); break;
      default: errorQuda("Unsupported dilution type %d", type);
      }
      if (v.size() != static_cast<unsigned int>(get_size<Ns, Nc>(type)))
        errorQuda("Input container size %lu does not match expected size %d for dilution type", v.size(), get_size<Ns, Nc>(type));
      apply(device::get_default_stream());
    }

    template <QudaDilutionType type> using Arg = SpinorDiluteArg<real, Ns, Nc, type>;

    template <QudaDilutionType type>
    auto constexpr sequence() { return std::make_index_sequence<get_size<Ns, Nc>(type)>(); }

    template <QudaDilutionType type>
    void apply(TuneParam &tp, const qudaStream_t &stream) { launch<DiluteSpinor>(tp, stream, Arg<type>(v, src, sequence<type>())); }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch (type) {
      case QUDA_DILUTION_SPIN: apply<QUDA_DILUTION_SPIN>(tp, stream); break;
      case QUDA_DILUTION_COLOR: apply<QUDA_DILUTION_COLOR>(tp, stream); break;
      case QUDA_DILUTION_SPIN_COLOR: apply<QUDA_DILUTION_SPIN_COLOR>(tp, stream); break;
      case QUDA_DILUTION_SPIN_COLOR_EVEN_ODD: apply<QUDA_DILUTION_SPIN_COLOR_EVEN_ODD>(tp, stream); break;
      default: errorQuda("Dilution type %d not supported", type);
      }
    }

    long long bytes() const { return v.size() * v[0].Bytes() + src.Bytes(); }
  };

  void spinorDilute(std::vector<ColorSpinorField> &v, const ColorSpinorField &src, QudaDilutionType type)
  {
    instantiateSpinor<SpinorDilute>(src, v, type);
  }

} // namespace quda
