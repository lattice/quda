#include <color_spinor_field.h>

namespace quda {

  void copyGenericColorSpinorDD(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorDS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorDH(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorDQ(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  
  void copyGenericColorSpinorSD(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorSS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorSH(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorSQ(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);

  void copyGenericColorSpinorHD(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorHS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorHH(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorHQ(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);

  void copyGenericColorSpinorQD(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorQS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorQH(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorQQ(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);

  // multigrid copying routines
  void copyGenericColorSpinorMGDD(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGDS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGSD(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGSS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGSH(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGSQ(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGHS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGHH(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGHQ(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGQS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGQH(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGQQ(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  

  void copyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, 
			      QudaFieldLocation location, void *Dst, void *Src, 
			      void *dstNorm, void *srcNorm) {

    if (dst.SiteSubset() != src.SiteSubset())
      errorQuda("Destination %d and source %d site subsets not equal", dst.SiteSubset(), src.SiteSubset());

    if (dst.Ncolor() != src.Ncolor()) 
      errorQuda("Destination %d and source %d colors not equal", dst.Ncolor(), src.Ncolor());

    if (dst.Ncolor() == 3) {
      if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorDD(dst, src, location, (double*)Dst, (double*)Src);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorDS(dst, src, location, (double*)Dst, (float*)Src);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorDH(dst, src, location, (double*)Dst, (short*)Src, 0, (float*)srcNorm);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorDQ(dst, src, location, (double*)Dst, (char*)Src, 0, (float*)srcNorm);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorSD(dst, src, location, (float*)Dst, (double*)Src);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorSS(dst, src, location, (float*)Dst, (float*)Src);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorSH(dst, src, location, (float*)Dst, (short*)Src, 0, (float*)srcNorm);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorSQ(dst, src, location, (float*)Dst, (char*)Src, 0, (float*)srcNorm);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_HALF_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorHD(dst, src, location, (short*)Dst, (double*)Src, (float*)dstNorm, 0);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorHS(dst, src, location, (short*)Dst, (float*)Src, (float*)dstNorm, 0);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorHH(dst, src, location, (short*)Dst, (short*)Src, (float*)dstNorm, (float*)srcNorm);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorHQ(dst, src, location, (short*)Dst, (char*)Src, (float*)dstNorm, (float*)srcNorm);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_QUARTER_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorQD(dst, src, location, (char*)Dst, (double*)Src, (float*)dstNorm, 0);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorQS(dst, src, location, (char*)Dst, (float*)Src, (float*)dstNorm, 0);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorQH(dst, src, location, (char*)Dst, (short*)Src, (float*)dstNorm, (float*)srcNorm);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorQQ(dst, src, location, (char*)Dst, (char*)Src, (float*)dstNorm, (float*)srcNorm);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else {
        errorQuda("Unsupported Destination Precision %d", dst.Precision());
      }
    } else {
      if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorMGDD(dst, src, location, Dst, Src);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorMGDS(dst, src, location, (double*)Dst, (float*)Src);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
        if (src.Precision() == QUDA_DOUBLE_PRECISION) {
          copyGenericColorSpinorMGSD(dst, src, location, (float*)Dst, (double*)Src);
        } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorMGSS(dst, src, location, (float*)Dst, (float*)Src);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorMGSH(dst, src, location, (float*)Dst, (short*)Src);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorMGSQ(dst, src, location, (float*)Dst, (char*)Src);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_HALF_PRECISION) {
        if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorMGHS(dst, src, location, (short*)Dst, (float*)Src);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorMGHH(dst, src, location, (short*)Dst, (short*)Src);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorMGHQ(dst, src, location, (short*)Dst, (char*)Src);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else if (dst.Precision() == QUDA_QUARTER_PRECISION) {
        if (src.Precision() == QUDA_SINGLE_PRECISION) {
          copyGenericColorSpinorMGQS(dst, src, location, (char*)Dst, (float*)Src);
        } else if (src.Precision() == QUDA_HALF_PRECISION) {
          copyGenericColorSpinorMGQH(dst, src, location, (char*)Dst, (short*)Src);
        } else if (src.Precision() == QUDA_QUARTER_PRECISION) {
          copyGenericColorSpinorMGQQ(dst, src, location, (char*)Dst, (char*)Src);
        } else {
          errorQuda("Unsupported Destination Precision %d with Source Precision %d", dst.Precision(), src.Precision());
        }
      } else {
        errorQuda("Unsupported Destination Precision %d", dst.Precision());
      }
    }
  }  

} // namespace quda
