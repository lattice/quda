#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorSQ(const copy_pack_t &pack)
  {
#if (QUDA_PRECISION & 4) && (QUDA_PRECISION & 1)
    CopyGenericColorSpinor<3, float, int8_t>(pack);
#else
    errorQuda("QUDA_PRECISION=%d does not enable precision combination %d %d", QUDA_PRECISION, std::get<0>(pack).Precision(), std::get<1>(pack).Precision());
#endif
  }

} // namespace quda
