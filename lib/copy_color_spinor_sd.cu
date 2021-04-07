#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorSD(const copy_pack_t &pack)
  {
    CopyGenericColorSpinor<3, float, double>(pack);
  }  

} // namespace quda
