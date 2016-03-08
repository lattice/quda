#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGSD(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

    float *dst_ptr = static_cast<float*>(Dst);
    double *src_ptr = static_cast<double*>(Src);

    INSTANTIATE_COLOR;

  }

} // namespace quda
