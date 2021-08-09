#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorSS(const copy_pack_t &pack)
  {
#if QUDA_PRECISION & 4
    CopyGenericColorSpinor<3, float, float>(pack);
#else
    errorQuda("QUDA_PRECISION=%d does not enable precision combination %d %d", QUDA_PRECISION, std::get<0>(pack).Precision(), std::get<1>(pack).Precision());
#endif
  }  

} // namespace quda
