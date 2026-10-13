#include <tuple>
#include <color_spinor_field.h>
#include <instantiate.h>
#include <multigrid.h>

namespace quda
{

  using copy_pack = std::tuple<ColorSpinorField &, const ColorSpinorField &, QudaFieldLocation, void *, const void *>;

#define COPY_COLOR_SPINOR_DECL(IN, OUT) void copyGenericColorSpinor_##IN##_##OUT(const copy_pack &pack)

  COPY_COLOR_SPINOR_DECL(double, double);
  COPY_COLOR_SPINOR_DECL(double, single);
  COPY_COLOR_SPINOR_DECL(double, half);
  COPY_COLOR_SPINOR_DECL(double, quarter);
  COPY_COLOR_SPINOR_DECL(single, double);
  COPY_COLOR_SPINOR_DECL(single, single);
  COPY_COLOR_SPINOR_DECL(single, half);
  COPY_COLOR_SPINOR_DECL(single, quarter);
  COPY_COLOR_SPINOR_DECL(half, double);
  COPY_COLOR_SPINOR_DECL(half, single);
  COPY_COLOR_SPINOR_DECL(half, half);
  COPY_COLOR_SPINOR_DECL(half, quarter);
  COPY_COLOR_SPINOR_DECL(quarter, double);
  COPY_COLOR_SPINOR_DECL(quarter, single);
  COPY_COLOR_SPINOR_DECL(quarter, half);
  COPY_COLOR_SPINOR_DECL(quarter, quarter);

#undef COPY_COLOR_SPINOR_DECL

#define COPY_COLOR_SPINOR_MG_DECL(IN, OUT) void copyGenericColorSpinorMG_##IN##_##OUT(const copy_pack &pack)

  COPY_COLOR_SPINOR_MG_DECL(double, double);
  COPY_COLOR_SPINOR_MG_DECL(single, double);
  COPY_COLOR_SPINOR_MG_DECL(double, single);
  COPY_COLOR_SPINOR_MG_DECL(single, single);
  COPY_COLOR_SPINOR_MG_DECL(half, single);
  COPY_COLOR_SPINOR_MG_DECL(quarter, single);
  COPY_COLOR_SPINOR_MG_DECL(single, half);
  COPY_COLOR_SPINOR_MG_DECL(half, half);
  COPY_COLOR_SPINOR_MG_DECL(quarter, half);
  COPY_COLOR_SPINOR_MG_DECL(single, quarter);
  COPY_COLOR_SPINOR_MG_DECL(half, quarter);
  COPY_COLOR_SPINOR_MG_DECL(quarter, quarter);

#undef COPY_COLOR_SPINOR_MG_DECL

  void copyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, QudaFieldLocation location, void *Dst,
                              const void *Src)
  {
    if (dst.SiteSubset() != src.SiteSubset())
      errorQuda("Destination %d and source %d site subsets not equal", dst.SiteSubset(), src.SiteSubset());

    if (dst.Ncolor() != src.Ncolor())
      errorQuda("Destination %d and source %d colors not equal", dst.Ncolor(), src.Ncolor());

    copy_pack pack(dst, src, location, Dst, Src);
    if (dst.Ncolor() == 3) {
      if (src.Precision() == QUDA_DOUBLE_PRECISION) {
        if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinor_double_double(pack);
        } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
          if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
            copyGenericColorSpinor_double_single(pack);
          else
            errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
        } else if (dst.Precision() == QUDA_HALF_PRECISION) {
          if constexpr (is_enabled(QUDA_HALF_PRECISION))
            copyGenericColorSpinor_double_half(pack);
          else
            errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
        } else if (dst.Precision() == QUDA_QUARTER_PRECISION) {
          if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
            copyGenericColorSpinor_double_quarter(pack);
          else
            errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION)) {
          if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
            copyGenericColorSpinor_single_double(pack);
          } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
            copyGenericColorSpinor_single_single(pack);
          } else if (dst.Precision() == QUDA_HALF_PRECISION) {
            if constexpr (is_enabled(QUDA_HALF_PRECISION))
              copyGenericColorSpinor_single_half(pack);
            else
              errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
          } else if (dst.Precision() == QUDA_QUARTER_PRECISION) {
            if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
              copyGenericColorSpinor_single_quarter(pack);
            else
              errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
          } else {
            errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
          }
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
        }
      } else if (src.Precision() == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
          if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
            copyGenericColorSpinor_half_double(pack);
          } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
            if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
              copyGenericColorSpinor_half_single(pack);
            else
              errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
          } else if (dst.Precision() == QUDA_HALF_PRECISION) {
            copyGenericColorSpinor_half_half(pack);
          } else if (dst.Precision() == QUDA_QUARTER_PRECISION) {
            if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
              copyGenericColorSpinor_half_quarter(pack);
            else
              errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
          } else {
            errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
          }
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
        }
      } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
        if constexpr (is_enabled(QUDA_QUARTER_PRECISION)) {
          if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
            copyGenericColorSpinor_quarter_double(pack);
          } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
            if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
              copyGenericColorSpinor_quarter_single(pack);
            else
              errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
          } else if (dst.Precision() == QUDA_HALF_PRECISION) {
            if constexpr (is_enabled(QUDA_HALF_PRECISION))
              copyGenericColorSpinor_quarter_half(pack);
            else
              errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
          } else if (dst.Precision() == QUDA_QUARTER_PRECISION) {
            copyGenericColorSpinor_quarter_quarter(pack);
          } else {
            errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
          }
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
        }
      } else {
        errorQuda("Unsupported Source Precision %d", src.Precision());
      }
    } else {
      if constexpr (is_enabled_multigrid()) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
            copyGenericColorSpinorMG_double_double(pack);
          } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
            if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
              copyGenericColorSpinorMG_double_single(pack);
            else
              errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
          } else {
            errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
          }
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          if constexpr (is_enabled(QUDA_SINGLE_PRECISION)) {
            if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
              copyGenericColorSpinorMG_single_double(pack);
            } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
              copyGenericColorSpinorMG_single_single(pack);
            } else if (dst.Precision() == QUDA_HALF_PRECISION) {
              if constexpr (is_enabled(QUDA_HALF_PRECISION))
                copyGenericColorSpinorMG_single_half(pack);
              else
                errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
            } else if (dst.Precision() == QUDA_QUARTER_PRECISION) {
              if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
                copyGenericColorSpinorMG_single_quarter(pack);
              else
                errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
            } else {
              errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(),
                        src.Precision());
            }
          } else {
            errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
          }
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
            if (dst.Precision() == QUDA_SINGLE_PRECISION) {
              if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
                copyGenericColorSpinorMG_half_single(pack);
              else
                errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
            } else if (dst.Precision() == QUDA_HALF_PRECISION) {
              copyGenericColorSpinorMG_half_half(pack);
            } else if (dst.Precision() == QUDA_QUARTER_PRECISION) {
              if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
                copyGenericColorSpinorMG_half_quarter(pack);
              else
                errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
            } else {
              errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(),
                        src.Precision());
            }
          } else {
            errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
          }
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          if constexpr (is_enabled(QUDA_QUARTER_PRECISION)) {
            if (dst.Precision() == QUDA_SINGLE_PRECISION) {
              if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
                copyGenericColorSpinorMG_quarter_single(pack);
              else
                errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
            } else if (dst.Precision() == QUDA_HALF_PRECISION) {
              if constexpr (is_enabled(QUDA_HALF_PRECISION))
                copyGenericColorSpinorMG_quarter_half(pack);
              else
                errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
            } else if (dst.Precision() == QUDA_QUARTER_PRECISION) {
              copyGenericColorSpinorMG_quarter_quarter(pack);
            } else {
              errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(),
                        src.Precision());
            }
          } else {
            errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
          }
        } else {
          errorQuda("Unsupported Source Precision %d", src.Precision());
        }
      } else {
        errorQuda("Multigrid has not been built");
      }
    }
  }

} // namespace quda
