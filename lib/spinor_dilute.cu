#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/spinor_dilute.cuh>
#include <instantiate.h>
#include <blas_quda.h>

namespace quda {

  template <typename real, int Ns, int Nc, QudaFieldOrder order>
  class SpinorDilute : TunableKernel2D {
    std::vector<ColorSpinorField> &v;
    const ColorSpinorField &src;
    QudaDilutionType type;
    unsigned int minThreads() const { return src.VolumeCB(); }

  public:
    SpinorDilute(std::vector<ColorSpinorField> &v, const ColorSpinorField &src, QudaDilutionType type) :
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

    template <QudaDilutionType type> using Arg = SpinorDiluteArg<real, Ns, Nc, order, type>;

    template <QudaDilutionType type>
    auto constexpr sequence() { return std::make_index_sequence<get_size<Ns, Nc>(type)>(); }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch (type) {
      case QUDA_DILUTION_SPIN:
        launch<DiluteSpinor>(tp, stream, Arg<QUDA_DILUTION_SPIN>(v, src, sequence<QUDA_DILUTION_SPIN>()));
        break;
      case QUDA_DILUTION_COLOR:
        launch<DiluteSpinor>(tp, stream, Arg<QUDA_DILUTION_COLOR>(v, src, sequence<QUDA_DILUTION_COLOR>()));
        break;
      case QUDA_DILUTION_SPIN_COLOR:
        launch<DiluteSpinor>(tp, stream, Arg<QUDA_DILUTION_SPIN_COLOR>(v, src, sequence<QUDA_DILUTION_SPIN_COLOR>()));
        break;
      case QUDA_DILUTION_SPIN_COLOR_EVEN_ODD:
        launch<DiluteSpinor>(tp, stream, Arg<QUDA_DILUTION_SPIN_COLOR_EVEN_ODD>(v, src, sequence<QUDA_DILUTION_SPIN_COLOR_EVEN_ODD>()));
        break;
      default: errorQuda("Dilution type %d not supported", type);
      }
    }

    long long flops() const { return 0; }
    long long bytes() const { return v.size() * v[0].Bytes() + src.Bytes(); }
  };

  /** Decide on the input order*/
  template <typename real, int Ns, int Nc>
  void spinorDilute(std::vector<ColorSpinorField> &v, const ColorSpinorField &src, QudaDilutionType type)
  {
    if (src.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
      SpinorDilute<real, Ns, Nc, QUDA_FLOAT2_FIELD_ORDER>(v, src, type);
    } else if (src.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
      SpinorDilute<real, Ns, Nc, QUDA_FLOAT4_FIELD_ORDER>(v, src, type);
    } else if (src.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      SpinorDilute<real, Ns, Nc, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(v, src, type);
    } else {
      errorQuda("Order %d not defined (Ns=%d, Nc=%d)", src.FieldOrder(), Ns, Nc);
    }
  }

  template <typename real, int Ns>
  void spinorDilute(std::vector<ColorSpinorField> &v, const ColorSpinorField &src, QudaDilutionType type)
  {
    if (src.Ncolor() == 3) {
      spinorDilute<real, Ns, 3>(v, src, type);
    } else {
      errorQuda("nColor = %d not implemented", src.Ncolor());
    }
  }

  template <typename real>
  void spinorDilute(std::vector<ColorSpinorField> &v, const ColorSpinorField &src, QudaDilutionType type)
  {
    if (src.Nspin() == 4) {
      if constexpr (is_enabled_spin<4>()) spinorDilute<real, 4>(v, src, type);
      else errorQuda("spinorDilute has not been built for nSpin=%d fields", src.Nspin());
    } else if (src.Nspin() == 2) {
      if constexpr (is_enabled_spin<2>()) spinorDilute<real, 2>(v, src, type);
      else errorQuda("spinorDilute has not been built for nSpin=%d fields", src.Nspin());
    } else if (src.Nspin() == 1) {
      if constexpr (is_enabled_spin<1>()) spinorDilute<real, 1>(v, src, type);
      else errorQuda("spinorDilute has not been built for nSpin=%d fields", src.Nspin());
    } else {
      errorQuda("Nspin = %d not implemented", src.Nspin());
    }
  }

  void spinorDilute(std::vector<ColorSpinorField> &v, const ColorSpinorField &src, QudaDilutionType type)
  {
    switch (src.Precision()) {
    case QUDA_DOUBLE_PRECISION: spinorDilute<double>(v, src, type); break;
    case QUDA_SINGLE_PRECISION: spinorDilute<float>(v, src, type); break;
    default: errorQuda("Precision %d not implemented", src.Precision());
    }
  }

} // namespace quda
