#include <copy_color_spinor.cuh>

namespace quda {
  
  void copyGenericColorSpinorDD(const copy_pack_t &pack)
  {
    CopyGenericColorSpinor<3, double, double>(pack);
  }  

} // namespace quda
