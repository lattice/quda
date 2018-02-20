#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorQD(ColorSpinorField &dst, const ColorSpinorField &src, 
				QudaFieldLocation location, void *Dst, void *Src, 
				void *dstNorm, void *srcNorm) {
    printfQuda("In copyGenericColorSpinorQD, entering CopyGenericColorSpinor<3> from %s to %s\n", src.Location() == QUDA_CPU_FIELD_LOCATION ? "CPU" : "GPU", dst.Location() == QUDA_CPU_FIELD_LOCATION ? "CPU" : "GPU");
    CopyGenericColorSpinor<3>(dst, src, location, (char*)Dst, (double*)Src, (float*)dstNorm, 0);
  }  

} // namespace quda
