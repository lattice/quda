#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGHH(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

#if defined(GPU_MULTIGRID) && (QUDA_PRECISION & 2)
    auto *dst_ptr = static_cast<short*>(Dst);
    auto *src_ptr = static_cast<short*>(Src);

    INSTANTIATE_COLOR;
#else
    errorQuda("Half precision has not been enabled");
#endif

  }

} // namespace quda
