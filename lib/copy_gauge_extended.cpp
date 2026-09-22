#include <gauge_field.h>

namespace quda
{

#define COPY_GAUGE_EX_DECL(IN, OUT)                                                                                    \
  void copyExtendedGauge_##IN##_##OUT(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, \
                                      void *Out, void *In)

  COPY_GAUGE_EX_DECL(double, double);
#if QUDA_PRECISION & 4
  COPY_GAUGE_EX_DECL(double, single);
  COPY_GAUGE_EX_DECL(single, double);
  COPY_GAUGE_EX_DECL(single, single);
#endif
#if QUDA_PRECISION & 2
  COPY_GAUGE_EX_DECL(double, half);
  COPY_GAUGE_EX_DECL(half, double);
  COPY_GAUGE_EX_DECL(half, half);
#endif
#if QUDA_PRECISION & 1
  COPY_GAUGE_EX_DECL(double, quarter);
  COPY_GAUGE_EX_DECL(quarter, double);
  COPY_GAUGE_EX_DECL(quarter, quarter);
#endif
#if (QUDA_PRECISION & 4) && (QUDA_PRECISION & 2)
  COPY_GAUGE_EX_DECL(single, half);
  COPY_GAUGE_EX_DECL(half, single);
#endif
#if (QUDA_PRECISION & 4) && (QUDA_PRECISION & 1)
  COPY_GAUGE_EX_DECL(single, quarter);
  COPY_GAUGE_EX_DECL(quarter, single);
#endif
#if (QUDA_PRECISION & 2) && (QUDA_PRECISION & 1)
  COPY_GAUGE_EX_DECL(half, quarter);
  COPY_GAUGE_EX_DECL(quarter, half);
#endif

#undef COPY_GAUGE_EX_DECL

  void copyExtendedGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, void *Out,
                         void *In)
  {
    for (int d = 0; d < in.Ndim(); d++) {
      if ((out.X()[d] - in.X()[d]) % 2 != 0) errorQuda("Cannot copy into an asymmetrically extended gauge field");
    }

    if (in.Precision() == QUDA_DOUBLE_PRECISION) {
      if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        copyExtendedGauge_double_double(out, in, location, scale, Out, In);
#if QUDA_PRECISION & 4
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        copyExtendedGauge_double_single(out, in, location, scale, Out, In);
#endif
#if QUDA_PRECISION & 2
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        copyExtendedGauge_double_half(out, in, location, scale, Out, In);
#endif
#if QUDA_PRECISION & 1
      } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
        copyExtendedGauge_double_quarter(out, in, location, scale, Out, In);
#endif
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
#if QUDA_PRECISION & 4
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        copyExtendedGauge_single_double(out, in, location, scale, Out, In);
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        copyExtendedGauge_single_single(out, in, location, scale, Out, In);
#if QUDA_PRECISION & 2
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        copyExtendedGauge_single_half(out, in, location, scale, Out, In);
#endif
#if QUDA_PRECISION & 1
      } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
        copyExtendedGauge_single_quarter(out, in, location, scale, Out, In);
#endif
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
#endif
#if QUDA_PRECISION & 2
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        copyExtendedGauge_half_double(out, in, location, scale, Out, In);
#if QUDA_PRECISION & 4
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        copyExtendedGauge_half_single(out, in, location, scale, Out, In);
#endif
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        copyExtendedGauge_half_half(out, in, location, scale, Out, In);
#if QUDA_PRECISION & 1
      } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
        copyExtendedGauge_half_quarter(out, in, location, scale, Out, In);
#endif
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
#endif
#if QUDA_PRECISION & 1
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      if (out.Precision() == QUDA_DOUBLE_PRECISION) {
        copyExtendedGauge_quarter_double(out, in, location, scale, Out, In);
#if QUDA_PRECISION & 4
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        copyExtendedGauge_quarter_single(out, in, location, scale, Out, In);
#endif
#if QUDA_PRECISION & 2
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        copyExtendedGauge_quarter_half(out, in, location, scale, Out, In);
#endif
      } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
        copyExtendedGauge_quarter_quarter(out, in, location, scale, Out, In);
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
#endif
    } else {
      errorQuda("Unsupported input precision %d", in.Precision());
    }
  }

} // namespace quda
