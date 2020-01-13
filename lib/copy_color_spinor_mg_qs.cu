#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGQS(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

#if defined(GPU_MULTIGRID) && (QUDA_PRECISION & 1)
    char *dst_ptr = static_cast<char*>(Dst);
    float *src_ptr = static_cast<float*>(Src);

    INSTANTIATE_COLOR;
#else
    errorQuda("Quarter precision has not been enabled");
#endif

  }

} // namespace quda
