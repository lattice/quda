#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorDS(ColorSpinorField &dst, const ColorSpinorField &src, 
				QudaFieldLocation location, void *Dst, void *Src, 
				void *dstNorm, void *srcNorm) {
#if QUDA_PRECISION & 4
    CopyGenericColorSpinor<3>(dst, src, location, (double*)Dst, (float*)Src);
#else
    errorQuda("QUDA_PRECISION=%d does not enable precision combination %d %d", QUDA_PRECISION, dst.Precision(), src.Precision());
#endif
  }  

} // namespace quda
