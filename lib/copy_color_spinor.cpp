#include <tuple>
#include <color_spinor_field.h>

namespace quda
{

  using copy_pack = std::tuple<ColorSpinorField &, const ColorSpinorField &, QudaFieldLocation, void *, const void *>;
  void copyGenericColorSpinorDD(const copy_pack &pack);
  void copyGenericColorSpinorDS(const copy_pack &pack);
  void copyGenericColorSpinorDH(const copy_pack &pack);
  void copyGenericColorSpinorDQ(const copy_pack &pack);

  void copyGenericColorSpinorSD(const copy_pack &pack);
  void copyGenericColorSpinorSS(const copy_pack &pack);
  void copyGenericColorSpinorSH(const copy_pack &pack);
  void copyGenericColorSpinorSQ(const copy_pack &pack);

  void copyGenericColorSpinorHD(const copy_pack &pack);
  void copyGenericColorSpinorHS(const copy_pack &pack);
  void copyGenericColorSpinorHH(const copy_pack &pack);
  void copyGenericColorSpinorHQ(const copy_pack &pack);

  void copyGenericColorSpinorQD(const copy_pack &pack);
  void copyGenericColorSpinorQS(const copy_pack &pack);
  void copyGenericColorSpinorQH(const copy_pack &pack);
  void copyGenericColorSpinorQQ(const copy_pack &pack);

  // multigrid copying routines
  void copyGenericColorSpinorMGDD(const copy_pack &pack);
  void copyGenericColorSpinorMGDS(const copy_pack &pack);
  void copyGenericColorSpinorMGSD(const copy_pack &pack);
  void copyGenericColorSpinorMGSS(const copy_pack &pack);
  void copyGenericColorSpinorMGSH(const copy_pack &pack);
  void copyGenericColorSpinorMGSQ(const copy_pack &pack);
  void copyGenericColorSpinorMGHS(const copy_pack &pack);
  void copyGenericColorSpinorMGHH(const copy_pack &pack);
  void copyGenericColorSpinorMGHQ(const copy_pack &pack);
  void copyGenericColorSpinorMGQS(const copy_pack &pack);
  void copyGenericColorSpinorMGQH(const copy_pack &pack);
  void copyGenericColorSpinorMGQQ(const copy_pack &pack);

  void copyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, QudaFieldLocation location, void *Dst,
                              const void *Src)
  {
    if (dst.SiteSubset() != src.SiteSubset())
      errorQuda("Destination %d and source %d site subsets not equal", dst.SiteSubset(), src.SiteSubset());

    if (dst.Ncolor() != src.Ncolor())
      errorQuda("Destination %d and source %d colors not equal", dst.Ncolor(), src.Ncolor());

    copy_pack pack(dst, src, location, Dst, Src);
    if (dst.Ncolor() == 3) {
      if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorDD(pack);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorDS(pack);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorDH(pack);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorDQ(pack);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorSD(pack);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorSS(pack);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorSH(pack);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorSQ(pack);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_HALF_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorHD(pack);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorHS(pack);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorHH(pack);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorHQ(pack);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_QUARTER_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorQD(pack);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorQS(pack);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorQH(pack);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorQQ(pack);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else {
        errorQuda("Unsupported Destination Precision %d", dst.Precision());
      }
    } else {
      if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorMGDD(pack);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorMGDS(pack);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorMGSD(pack);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorMGSS(pack);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorMGSH(pack);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorMGSQ(pack);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_HALF_PRECISION) {
        if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorMGHS(pack);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorMGHH(pack);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorMGHQ(pack);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_QUARTER_PRECISION) {
        if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorMGQS(pack);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorMGQH(pack);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorMGQQ(pack);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else {
        errorQuda("Unsupported Destination Precision %d", dst.Precision());
      }
    }
  }

} // namespace quda
