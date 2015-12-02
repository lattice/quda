#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGDD(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

    double *dst_ptr = static_cast<double*>(Dst);
    double *src_ptr = static_cast<double*>(Src);

    INSTANTIATE_COLOR;

  }

} // namespace quda
