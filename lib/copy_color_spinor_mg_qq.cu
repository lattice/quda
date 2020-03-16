#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGQQ(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

#if defined(GPU_MULTIGRID) && (QUDA_PRECISION & 1)
    char *dst_ptr = static_cast<char*>(Dst);
    char *src_ptr = static_cast<char*>(Src);

    INSTANTIATE_COLOR;
#else
    errorQuda("Quarter precision has not been enabled");
#endif

  }

} // namespace quda
