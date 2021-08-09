#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorHD(const copy_pack_t &pack)
  {
#if QUDA_PRECISION & 2
    CopyGenericColorSpinor<3, short, double>(pack);
#else
    errorQuda("QUDA_PRECISION=%d does not enable precision combination %d %d", QUDA_PRECISION, std::get<0>(pack).Precision(), std::get<1>(pack).Precision());
#endif
  }  

} // namespace quda
