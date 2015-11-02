#define BUILD_LIMITED_COPY
#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGSS(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

    float *dst_ptr = static_cast<float*>(Dst);
    float *src_ptr = static_cast<float*>(Src);

    switch(src.Ncolor()) {
#ifdef GPU_MULTIGRID
    case 1:
      CopyGenericColorSpinor<1>(dst, src, location, dst_ptr, src_ptr);
      break;
    case 2:
      CopyGenericColorSpinor<2>(dst, src, location, dst_ptr, src_ptr);
      break;
    case 4:
      CopyGenericColorSpinor<4>(dst, src, location, dst_ptr, src_ptr);
      break;
    case 6:
      CopyGenericColorSpinor<6>(dst, src, location, dst_ptr, src_ptr);
      break;
    case 9:
      CopyGenericColorSpinor<9>(dst, src, location, dst_ptr, src_ptr);
      break;
    case 24:
      CopyGenericColorSpinor<24>(dst, src, location, dst_ptr, src_ptr);
      break;
    case 72:
      CopyGenericColorSpinor<72>(dst, src, location, dst_ptr, src_ptr);
      break;
#endif
    default:
      errorQuda("Ncolors=%d not supported", src.Ncolor());
    }

  }

} // namespace quda
