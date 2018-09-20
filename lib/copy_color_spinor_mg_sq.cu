#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGSQ(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

#if defined(GPU_MULTIGRID)
    float *dst_ptr = static_cast<float*>(Dst);
    char *src_ptr = static_cast<char*>(Src);

    INSTANTIATE_COLOR;
#else
    errorQuda("Double precision multigrid has not been enabled");
#endif

  }

} // namespace quda
