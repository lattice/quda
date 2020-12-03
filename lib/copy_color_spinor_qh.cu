#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorQH(const copy_pack_t &pack)
  {
#if (QUDA_PRECISION & 2) && (QUDA_PRECISION & 1)
    CopyGenericColorSpinor<3, int8_t, short>(pack);
#else
    errorQuda("QUDA_PRECISION=%d does not enable precision combination %d %d", QUDA_PRECISION, std::get<0>(pack).Precision(), std::get<1>(pack).Precision());
#endif
  }  

} // namespace quda
