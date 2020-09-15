#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGHQ(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

#if defined(GPU_MULTIGRID) && (QUDA_PRECISION & 2) && (QUDA_PRECISION & 1)
    auto *dst_ptr = static_cast<short*>(Dst);
    auto *src_ptr = static_cast<int8_t*>(Src);

    INSTANTIATE_COLOR;
#else
    errorQuda("Half and quarter precision have not been enabled");
#endif

  }

} // namespace quda
