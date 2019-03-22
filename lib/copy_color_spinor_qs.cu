#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorQS(ColorSpinorField &dst, const ColorSpinorField &src, 
				QudaFieldLocation location, void *Dst, void *Src, 
				void *dstNorm, void *srcNorm) {
    CopyGenericColorSpinor<3>(dst, src, location, (char*)Dst, (float*)Src, (float*)dstNorm, (float*)srcNorm);
  }  

} // namespace quda
