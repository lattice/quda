#include <color_spinor_field.h>

namespace quda {

  void copyGenericColorSpinorDD(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorDS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorDH(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  
  void copyGenericColorSpinorSD(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorSS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorSH(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);

  void copyGenericColorSpinorHD(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorHS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorHH(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);

  // multigrid copying routines
  void copyGenericColorSpinorMGDD(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGDS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGSD(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);
  void copyGenericColorSpinorMGSS(ColorSpinorField &, const ColorSpinorField&, QudaFieldLocation, void*, void*, void*a=0, void *b=0);

  void copyGenericColorSpinor(ColorSpinorField &dst, const ColorSpinorField &src, 
			      QudaFieldLocation location, void *Dst, void *Src, 
			      void *dstNorm, void *srcNorm) {

    if (dst.Ncolor() != src.Ncolor()) 
      errorQuda("Destination %d and source %d colors not equal", dst.Ncolor(), src.Ncolor());

    if (dst.Ncolor() == 3) {
      if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
	if (src.Precision() == QUDA_DOUBLE_PRECISION) {
	  copyGenericColorSpinorDD(dst, src, location, Dst, Src);
	} else if (src.Precision() == QUDA_SINGLE_PRECISION) {
	  copyGenericColorSpinorDS(dst, src, location, (double*)Dst, (float*)Src);
	} else if (src.Precision() == QUDA_HALF_PRECISION) {
	  copyGenericColorSpinorDH(dst, src, location, (double*)Dst, (short*)Src, 0, (float*)srcNorm);
	} else {
	  errorQuda("Unsupported Precision %d", src.Precision());
	}
      } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
	if (src.Precision() == QUDA_DOUBLE_PRECISION) {
	  copyGenericColorSpinorSD(dst, src, location, (float*)Dst, (double*)Src);
	} else if (src.Precision() == QUDA_SINGLE_PRECISION) {
	  copyGenericColorSpinorSS(dst, src, location, (float*)Dst, (float*)Src);
	} else if (src.Precision() == QUDA_HALF_PRECISION) {
	  copyGenericColorSpinorSH(dst, src, location, (float*)Dst, (short*)Src, 0, (float*)srcNorm);
	} else {
	  errorQuda("Unsupported Precision %d", src.Precision());
	}
      } else if (dst.Precision() == QUDA_HALF_PRECISION) {
	if (src.Precision() == QUDA_DOUBLE_PRECISION) {
	  copyGenericColorSpinorHD(dst, src, location, (short*)Dst, (double*)Src, (float*)dstNorm, 0);
	} else if (src.Precision() == QUDA_SINGLE_PRECISION) {
	  copyGenericColorSpinorHS(dst, src, location, (short*)Dst, (float*)Src, (float*)dstNorm, 0);
	} else if (src.Precision() == QUDA_HALF_PRECISION) {
	  copyGenericColorSpinorHH(dst, src, location, (short*)Dst, (short*)Src, (float*)dstNorm, (float*)srcNorm);
	} else {
	  errorQuda("Unsupported Precision %d", src.Precision());
	}
      } else {
	errorQuda("Unsupported Precision %d", dst.Precision());
      }
    } else {
      if (dst.Precision() == QUDA_DOUBLE_PRECISION) {
	if (src.Precision() == QUDA_DOUBLE_PRECISION) {
	  copyGenericColorSpinorMGDD(dst, src, location, Dst, Src);
	} else if (src.Precision() == QUDA_SINGLE_PRECISION) {
	  copyGenericColorSpinorMGDS(dst, src, location, (double*)Dst, (float*)Src);
	} else {
	  errorQuda("Unsupported Precision %d", src.Precision());
	}
      } else if (dst.Precision() == QUDA_SINGLE_PRECISION) {
	if (src.Precision() == QUDA_DOUBLE_PRECISION) {
	  copyGenericColorSpinorMGSD(dst, src, location, (float*)Dst, (double*)Src);
	} else if (src.Precision() == QUDA_SINGLE_PRECISION) {
	  copyGenericColorSpinorMGSS(dst, src, location, (float*)Dst, (float*)Src);
	} else {
	  errorQuda("Unsupported Precision %d", src.Precision());
	}
      } else {
	errorQuda("Unsupported Precision %d", dst.Precision());
      }
    }
  }  

} // namespace quda
