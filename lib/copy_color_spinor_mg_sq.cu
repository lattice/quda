#include <copy_color_spinor_mg.hpp>

namespace quda {
  
  void copyGenericColorSpinorMGSQ(const copy_pack_t &pack)
  {
    if constexpr (is_enabled_multigrid()) {
      instantiateColor<float, int8_t>(std::get<0>(pack), pack);
    } else {
      errorQuda("Multigrid has not been enabled (precision = %d %d)", std::get<0>(pack).Precision(), std::get<1>(pack).Precision());
    }
  }

} // namespace quda
