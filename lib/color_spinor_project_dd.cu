#include <tuple>
#include <memory>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <blas_quda.h>
#include <instantiate.h>
#include <domain_decomposition_helper.cuh>
#include <tunable_nd.h>
#include <kernels/color_spinor_project_dd.cuh>

namespace quda
{

  template <typename Float, typename DDArg, int nSpin, int nColor, typename Order>
  class ProjectDD : public TunableKernel2D
  {
    using Arg = ProjectDDArg<Float, DDArg, nSpin, nColor, Order>;
    ColorSpinorField &out;

    bool advanceSharedBytes(TuneParam &) const { return false; } // Don't tune shared mem
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

    long long flops() const { return 0; }
    long long bytes() const { return out.Bytes(); }
  };

  template <typename P, typename DDArg> void projectDD(P &p, DDArg &dd, const ColorSpinorField &meta)
  {
    Coord<4> coord;
    int X[4] = {meta.full_dim(0), meta.full_dim(1), meta.full_dim(2), meta.full_dim(3)};
    int commCoord[4] = {comm_coord(0) * X[0], comm_coord(1) * X[1], comm_coord(2) * X[2], comm_coord(3) * X[3]};

    for (int parity = 0; parity < p.Nparity(); parity++) {
      for (int x_cb = 0; x_cb < p.VolumeCB(); x_cb++) {
        getCoords(coord, x_cb, X, parity);
        for (int i = 0; i < coord.size(); i++) { coord.gx[i] = commCoord[i] + coord.x[i]; }

        if (dd.isZero(coord)) {
          for (int s = 0; s < p.Nspin(); s++)
            for (int c = 0; c < p.Ncolor(); c++) p(parity, x_cb, s, c) = 0;
        }
      }
    }
  }

  template <typename Float, typename DDArg, int nSpin, int nColor, typename Order>
  void genericProjectDD(ColorSpinorField &a)
  {
    /* Reference CPU implementation
    if (a.Location() == QUDA_CPU_FIELD_LOCATION and a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      FieldOrderCB<Float, nSpin, nColor, 1, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> A(a);
      DDArg dd(a);
      return projectDD(A, dd, a);
    } */

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
    default: errorQuda("Precision %d not implemented", a.Precision());
    }
  }
} // namespace quda
