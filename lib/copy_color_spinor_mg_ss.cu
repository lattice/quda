#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGSS(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

#ifdef GPU_MULTIGRID
    float *dst_ptr = static_cast<float*>(Dst);
    float *src_ptr = static_cast<float*>(Src);

    INSTANTIATE_COLOR;
#else
    errorQuda("Multigrid has not been enabled");
#endif
  }

} // namespace quda
