#include <tuple>
#include <memory>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <blas_quda.h>
#include <instantiate.h>
#include <domain_decomposition_helper.cuh>
#include <tunable_nd.h>
#include <kernels/color_spinor_project_domain_decomp.cuh>

namespace quda
{

  template <typename Float, typename DDArg, int nSpin, int nColor, typename Order>
  class ProjectDD : public TunableKernel2D
  {
    using Arg = ProjectDDArg<Float, DDArg, nSpin, nColor, Order>;
    ColorSpinorField &out;

    bool tuneSharedBytes() const { return false; }
    unsigned int minThreads() const { return out.VolumeCB(); }

  public:
    ProjectDD(ColorSpinorField &out) : TunableKernel2D(out, out.SiteSubset()), out(out)
    {
      strcat(aux, out.AuxString().c_str());
      switch (out.DD().type) {
      case QUDA_DD_NO: strcat(aux, ",DDNo"); break;
      case QUDA_DD_RED_BLACK: strcat(aux, ",DDRedBlack"); break;
      default: errorQuda("DD type %d not implemented", out.DD().type);
      }

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      constexpr bool enable_host = true;
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<ProjectDD_, enable_host>(tp, stream, Arg(out));
    }

    long long bytes() const { return out.Bytes(); }
  };

  template <typename Float, typename DDArg, int nSpin, int nColor, typename Order>
  void genericProjectDD(ColorSpinorField &a)
  {
    ProjectDD<Float, DDArg, nSpin, nColor, Order> A(a);
  }

  /** Decide on the field order*/
  template <typename Float, typename DDArg, int nSpin, int nColor> void genericProjectDD(ColorSpinorField &a)
  {
    if (a.isNative()) {
      using Order = typename colorspinor_mapper<Float, nSpin, nColor>::type;
      genericProjectDD<Float, DDArg, nSpin, nColor, Order>(a);
    } else if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      using Order = SpaceSpinorColorOrder<Float, nSpin, nColor>;
      genericProjectDD<Float, DDArg, nSpin, nColor, Order>(a);
    } else if (a.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      using Order = SpaceColorSpinorOrder<Float, nSpin, nColor>;
      genericProjectDD<Float, DDArg, nSpin, nColor, Order>(a);
    } else if (a.FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
      using Order = PaddedSpaceSpinorColorOrder<Float, nSpin, nColor>;
      if constexpr (is_enabled<QUDA_TIFR_GAUGE_ORDER>())
        genericProjectDD<Float, DDArg, nSpin, nColor, Order>(a);
      else
        errorQuda("TIFR interface has not been built");
    } else if (a.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER) {
      using Order = QDPJITDiracOrder<Float, nSpin, nColor>;
      if constexpr (is_enabled<QUDA_QDPJIT_GAUGE_ORDER>())
        genericProjectDD<Float, DDArg, nSpin, nColor, Order>(a);
      else
        errorQuda("QDPJIT interface has not been built");
    } else {
      errorQuda("Order %d not defined (Ns=%d, Nc=%d, precision = %d)", a.FieldOrder(), nSpin, nColor, a.Precision());
    }
  }

  template <typename Float, typename DDArg> void genericProjectDD(ColorSpinorField &a)
  {
    switch (a.Nspin()) {
    case (1):
      if constexpr (is_enabled_spin(1)) genericProjectDD<Float, DDArg, 1, 3>(a);
      break;
    case (2):
      if constexpr (is_enabled_spin(2)) genericProjectDD<Float, DDArg, 2, 3>(a);
      break;
    case (4):
      if constexpr (is_enabled_spin(4)) genericProjectDD<Float, DDArg, 4, 3>(a);
      break;
    default: errorQuda("Nspin %d not implemented", a.Nspin());
    }
  }

  template <typename Float> void genericProjectDD(ColorSpinorField &a)
  {
    switch (a.DD().type) {
    case QUDA_DD_NO: genericProjectDD<Float, DDNo>(a); break;
    case QUDA_DD_RED_BLACK: genericProjectDD<Float, DDRedBlack>(a); break;
    default: errorQuda("DD type %d not implemented", a.DD().type);
    }
  }

  void genericProjectDD(ColorSpinorField &a)
  {
    switch (a.Precision()) {
    case QUDA_DOUBLE_PRECISION: genericProjectDD<double>(a); break;
    case QUDA_SINGLE_PRECISION: genericProjectDD<float>(a); break;
    case QUDA_HALF_PRECISION: genericProjectDD<short>(a); break;
    case QUDA_QUARTER_PRECISION: genericProjectDD<int8_t>(a); break;
    default: errorQuda("Precision %d not implemented", a.Precision());
    }
  }
} // namespace quda
