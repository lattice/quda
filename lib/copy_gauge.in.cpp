#include <gauge_field.h>
#include <multigrid.h>
#include <int_list.hpp>

namespace quda
{

#define COPY_GAUGE_DECL(IN, OUT)                                                                                       \
  void copyGenericGauge_##IN##_##OUT(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale,  \
                                     void *Out, void *In, void **ghostOut, void **ghostIn, int type)

  COPY_GAUGE_DECL(double, double);
#if QUDA_PRECISION & 4
  COPY_GAUGE_DECL(double, single);
  COPY_GAUGE_DECL(single, double);
  COPY_GAUGE_DECL(single, single);
#endif
#if QUDA_PRECISION & 2
  COPY_GAUGE_DECL(double, half);
  COPY_GAUGE_DECL(half, double);
  COPY_GAUGE_DECL(half, half);
#endif
#if QUDA_PRECISION & 1
  COPY_GAUGE_DECL(double, quarter);
  COPY_GAUGE_DECL(quarter, double);
  COPY_GAUGE_DECL(quarter, quarter);
#endif
#if (QUDA_PRECISION & 4) && (QUDA_PRECISION & 2)
  COPY_GAUGE_DECL(single, half);
  COPY_GAUGE_DECL(half, single);
#endif
#if (QUDA_PRECISION & 4) && (QUDA_PRECISION & 1)
  COPY_GAUGE_DECL(single, quarter);
  COPY_GAUGE_DECL(quarter, single);
#endif
#if (QUDA_PRECISION & 2) && (QUDA_PRECISION & 1)
  COPY_GAUGE_DECL(half, quarter);
  COPY_GAUGE_DECL(quarter, half);
#endif

#undef COPY_GAUGE_DECL

#define COPY_GAUGE_MG_DECL(IN, OUT)                                                                                    \
  template <int nColor>                                                                                                \
  void copyGenericGaugeMG_##IN##_##OUT(GaugeField &out, const GaugeField &in, QudaFieldLocation location,              \
                                       double scale, void *Out, void *In, void **ghostOut, void **ghostIn, int type)

  COPY_GAUGE_MG_DECL(double, double);
#if QUDA_PRECISION & 4
  COPY_GAUGE_MG_DECL(double, single);
  COPY_GAUGE_MG_DECL(single, double);
  COPY_GAUGE_MG_DECL(single, single);
#endif
#if QUDA_PRECISION & 2
  COPY_GAUGE_MG_DECL(double, half);
  COPY_GAUGE_MG_DECL(half, double);
  COPY_GAUGE_MG_DECL(half, half);
#endif
#if (QUDA_PRECISION & 4) && (QUDA_PRECISION & 2)
  COPY_GAUGE_MG_DECL(single, half);
  COPY_GAUGE_MG_DECL(half, single);
#endif

#undef COPY_GAUGE_MG_DECL

  template <int nColor>
  void copyGenericGaugeMG(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, void *Out,
                          void *In, void **ghostOut, void **ghostIn, int type)
  {
    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        copyGenericGaugeMG_double_double<nColor>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#if QUDA_PRECISION & 4
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        copyGenericGaugeMG_double_single<nColor>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
#if QUDA_PRECISION & 2
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        copyGenericGaugeMG_double_half<nColor>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
#if QUDA_PRECISION & 4
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        copyGenericGaugeMG_single_double<nColor>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        copyGenericGaugeMG_single_single<nColor>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#if QUDA_PRECISION & 2
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        copyGenericGaugeMG_single_half<nColor>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
#endif
#if QUDA_PRECISION & 2
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        copyGenericGaugeMG_half_double<nColor>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#if QUDA_PRECISION & 4
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        copyGenericGaugeMG_half_single<nColor>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        copyGenericGaugeMG_half_half<nColor>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
#endif
    } else {
      errorQuda("Unsupported input precision %d", in.Precision());
    }
  }

  template <int Nc, int... N>
  void copyGenericGaugeMG(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, void *Out,
                          void *In, void **ghostOut, void **ghostIn, int type, IntList<Nc, N...>)
  {
    if (in.Ncolor() / 2 == Nc) {
      copyGenericGaugeMG<Nc>(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
    } else {
      if constexpr (sizeof...(N) > 0) {
        copyGenericGaugeMG(out, in, location, scale, Out, In, ghostOut, ghostIn, type, IntList<N...>());
      } else {
        errorQuda("Nc = %d has not been instantiated", in.Ncolor() / 2);
      }
    }
  }

  void checkMomOrder(const GaugeField &u)
  {
    if (u.Order() == QUDA_TIFR_GAUGE_ORDER || u.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
      if (u.Reconstruct() != QUDA_RECONSTRUCT_NO)
        errorQuda("Unsuported order %d and reconstruct %d combination", u.Order(), u.Reconstruct());
    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {
      if (u.Reconstruct() != QUDA_RECONSTRUCT_10)
        errorQuda("Unsuported order %d and reconstruct %d combination", u.Order(), u.Reconstruct());
    } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      if (u.Reconstruct() != QUDA_RECONSTRUCT_10)
        errorQuda("Unsuported order %d and reconstruct %d combination", u.Order(), u.Reconstruct());
    } else if (u.Order() == QUDA_MILC_SITE_GAUGE_ORDER) {
      if (u.Reconstruct() != QUDA_RECONSTRUCT_10)
        errorQuda("Unsuported order %d and reconstruct %d combination", u.Order(), u.Reconstruct());
    } else if (u.Order() != QUDA_NATIVE_GAUGE_ORDER) {
      errorQuda("Unsupported gauge field order %d", u.Order());
    }
  }

  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, void *Out,
                        void *In, void **ghostOut, void **ghostIn, int type)
  {
    // do not copy the ghost zone if it does not exist
    if (type == 0 && (in.GhostExchange() != QUDA_GHOST_EXCHANGE_PAD || out.GhostExchange() != QUDA_GHOST_EXCHANGE_PAD))
      type = 2;

    if (in.Ncolor() != out.Ncolor()) errorQuda("Colors (%d,%d) do not match", out.Ncolor(), in.Ncolor());

    if (out.Geometry() != in.Geometry())
      errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());

    if (in.Ncolor() != 3) {
      if constexpr (is_enabled_multigrid()) {
        // clang-format off
        copyGenericGaugeMG(out, in, location, scale, Out, In, ghostOut, ghostIn, type, IntList<@QUDA_MULTIGRID_NVEC_LIST@>());
        // clang-format on
      } else {
        errorQuda("Multigrid has not been built");
      }
    } else if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        copyGenericGauge_double_double(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#if QUDA_PRECISION & 4
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        copyGenericGauge_double_single(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
#if QUDA_PRECISION & 2
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        copyGenericGauge_double_half(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
#if QUDA_PRECISION & 1
      } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
        copyGenericGauge_double_quarter(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
#if QUDA_PRECISION & 4
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        copyGenericGauge_single_double(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        copyGenericGauge_single_single(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#if QUDA_PRECISION & 2
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        copyGenericGauge_single_half(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
#if QUDA_PRECISION & 1
      } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
        copyGenericGauge_single_quarter(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
#endif
#if QUDA_PRECISION & 2
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        copyGenericGauge_half_double(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#if QUDA_PRECISION & 4
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        copyGenericGauge_half_single(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        copyGenericGauge_half_half(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#if QUDA_PRECISION & 1
      } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
        copyGenericGauge_half_quarter(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
#endif
#if QUDA_PRECISION & 1
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        copyGenericGauge_quarter_double(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#if QUDA_PRECISION & 4
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        copyGenericGauge_quarter_single(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
#if QUDA_PRECISION & 2
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        copyGenericGauge_quarter_half(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
#endif
      } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
        copyGenericGauge_quarter_quarter(out, in, location, scale, Out, In, ghostOut, ghostIn, type);
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
#endif
    } else {
      errorQuda("Unsupported input precision %d", in.Precision());
    }
  }

} // namespace quda
