#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorDQ(ColorSpinorField &dst, const ColorSpinorField &src, 
				QudaFieldLocation location, void *Dst, void *Src, 
				void *dstNorm, void *srcNorm) {
    printfQuda("In copyGenericColorSpinorDQ, entering CopyGenericColorSpinor<3> from %s to %s\n", src.Location() == QUDA_CPU_FIELD_LOCATION ? "CPU" : "GPU", dst.Location() == QUDA_CPU_FIELD_LOCATION ? "CPU" : "GPU");
    CopyGenericColorSpinor<3>(dst, src, location, (double*)Dst, (char*)Src, 0, (float*)srcNorm);
  }  

} // namespace quda
