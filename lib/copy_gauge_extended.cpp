#include <gauge_field.h>
#include <instantiate.h>

namespace quda
{

#define COPY_GAUGE_EX_DECL(IN, OUT)                                                                                    \
  void copyExtendedGauge_##IN##_##OUT(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, \
                                      void *Out, void *In)

  COPY_GAUGE_EX_DECL(double, double);
  COPY_GAUGE_EX_DECL(double, single);
  COPY_GAUGE_EX_DECL(double, half);
  COPY_GAUGE_EX_DECL(double, quarter);
  COPY_GAUGE_EX_DECL(single, double);
  COPY_GAUGE_EX_DECL(single, single);
  COPY_GAUGE_EX_DECL(single, half);
  COPY_GAUGE_EX_DECL(single, quarter);
  COPY_GAUGE_EX_DECL(half, double);
  COPY_GAUGE_EX_DECL(half, single);
  COPY_GAUGE_EX_DECL(half, half);
  COPY_GAUGE_EX_DECL(half, quarter);
  COPY_GAUGE_EX_DECL(quarter, double);
  COPY_GAUGE_EX_DECL(quarter, single);
  COPY_GAUGE_EX_DECL(quarter, half);
  COPY_GAUGE_EX_DECL(quarter, quarter);

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
      } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
          copyExtendedGauge_double_single(out, in, location, scale, Out, In);
        else
          errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
      } else if (out.Precision() == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION))
          copyExtendedGauge_double_half(out, in, location, scale, Out, In);
        else
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
        if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
          copyExtendedGauge_double_quarter(out, in, location, scale, Out, In);
        else
          errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
      } else {
        errorQuda("Unsupported output precision %d", out.Precision());
      }
    } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled(QUDA_SINGLE_PRECISION)) {
        if (out.Precision() == QUDA_DOUBLE_PRECISION) {
          copyExtendedGauge_single_double(out, in, location, scale, Out, In);
        } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
          copyExtendedGauge_single_single(out, in, location, scale, Out, In);
        } else if (out.Precision() == QUDA_HALF_PRECISION) {
          if constexpr (is_enabled(QUDA_HALF_PRECISION))
            copyExtendedGauge_single_half(out, in, location, scale, Out, In);
          else
            errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
        } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
          if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
            copyExtendedGauge_single_quarter(out, in, location, scale, Out, In);
          else
            errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
        } else {
          errorQuda("Unsupported output precision %d", out.Precision());
        }
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
      }
    } else if (in.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
        if (out.Precision() == QUDA_DOUBLE_PRECISION) {
          copyExtendedGauge_half_double(out, in, location, scale, Out, In);
        } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
          if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
            copyExtendedGauge_half_single(out, in, location, scale, Out, In);
          else
            errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
        } else if (out.Precision() == QUDA_HALF_PRECISION) {
          copyExtendedGauge_half_half(out, in, location, scale, Out, In);
        } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
          if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
            copyExtendedGauge_half_quarter(out, in, location, scale, Out, In);
          else
            errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
        } else {
          errorQuda("Unsupported output precision %d", out.Precision());
        }
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      }
    } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled(QUDA_QUARTER_PRECISION)) {
        if (out.Precision() == QUDA_DOUBLE_PRECISION) {
          copyExtendedGauge_quarter_double(out, in, location, scale, Out, In);
        } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
          if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
            copyExtendedGauge_quarter_single(out, in, location, scale, Out, In);
          else
            errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
        } else if (out.Precision() == QUDA_HALF_PRECISION) {
          if constexpr (is_enabled(QUDA_HALF_PRECISION))
            copyExtendedGauge_quarter_half(out, in, location, scale, Out, In);
          else
            errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
        } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
          copyExtendedGauge_quarter_quarter(out, in, location, scale, Out, In);
        } else {
          errorQuda("Unsupported output precision %d", out.Precision());
        }
      } else {
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
      }
    } else {
      errorQuda("Unsupported input precision %d", in.Precision());
    }
  }

} // namespace quda
