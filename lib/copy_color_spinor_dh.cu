#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorDH(ColorSpinorField &dst, const ColorSpinorField &src, 
				QudaFieldLocation location, void *Dst, void *Src, 
				void *dstNorm, void *srcNorm) {
    CopyGenericColorSpinor<3>(dst, src, location, (double*)Dst, (short*)Src, 0, (float*)srcNorm);
  }  

} // namespace quda
