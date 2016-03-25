#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGSD(ColorSpinorField &dst, const ColorSpinorField &src, 
				  QudaFieldLocation location, void *Dst, void *Src, 
				  void *dstNorm, void *srcNorm) {

#ifdef GPU_MULTIGRID_DOUBLE
    float *dst_ptr = static_cast<float*>(Dst);
    double *src_ptr = static_cast<double*>(Src);

    INSTANTIATE_COLOR;
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif

  }

} // namespace quda
