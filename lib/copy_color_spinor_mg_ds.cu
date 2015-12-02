#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGDS(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

    double *dst_ptr = static_cast<double*>(Dst);
    float *src_ptr = static_cast<float*>(Src);

    INSTANTIATE_COLOR;

  }

} // namespace quda
