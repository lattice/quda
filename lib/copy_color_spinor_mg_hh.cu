#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGHH(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

#if defined(GPU_MULTIGRID)
    short *dst_ptr = static_cast<short*>(Dst);
    short *src_ptr = static_cast<short*>(Src);

    INSTANTIATE_COLOR;
#else
    errorQuda("Double precision multigrid has not been enabled");
#endif

  }

} // namespace quda
