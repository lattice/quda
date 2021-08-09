#include <copy_color_spinor_mg.cuh>

namespace quda {
  
  void copyGenericColorSpinorMGDD(const copy_pack_t &pack)
  {
#if defined(GPU_MULTIGRID)
    instantiateColor<double, double>(std::get<0>(pack), pack);
#else
    errorQuda("Multigrid has not been enabled (precision = %d %d)", std::get<0>(pack).Precision(), std::get<1>(pack).Precision());
#endif
  }

} // namespace quda
