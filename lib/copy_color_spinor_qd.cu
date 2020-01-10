#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorQD(ColorSpinorField &dst, const ColorSpinorField &src, 
				QudaFieldLocation location, void *Dst, void *Src, 
				void *dstNorm, void *srcNorm) {
    CopyGenericColorSpinor<3>(dst, src, location, (char*)Dst, (double*)Src, (float*)dstNorm, 0);
  }  

} // namespace quda
