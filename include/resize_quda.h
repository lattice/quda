#ifndef _RESIZE_QUDA_H
#define _RESIZE_QUDA_H

struct DecompParams;
struct cudaColorSpinorField;

namespace quda {
  
  void cropCuda(cudaColorSpinorField& dst, const cudaColorSpinorField &src, const DecompParams& params);
  void extendCuda(cudaColorSpinorField& dst, cudaColorSpinorField &src, const DecompParams& params);

} // namespace quda




#endif // _RESIZE_QUDA_H
