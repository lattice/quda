#ifndef _RESIZE_QUDA_H
#define _RESIZE_QUDA_H

//struct DecompParam;
//class cudaColorSpinorField;

#include <domain_decomposition.h>
#include <color_spinor_field.h>

namespace quda {
  
  void cropCuda(cudaColorSpinorField& dst, const cudaColorSpinorField &src, const DecompParam& params);
  void extendCuda(cudaColorSpinorField& dst, cudaColorSpinorField &src, const DecompParam& params, const int* const domain_overlap);

} // namespace quda




#endif // _RESIZE_QUDA_H
