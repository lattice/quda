#include <domain_decomposition.h>

#include <cassert>

namespace quda {

  void initDecompParams(DecompParams* const params, const int X[], const int Y[])
  {

    for(int dir=0; dir<4; ++dir){
      const int border = Y[dir]-X[dir];
      assert((border >= 0) && !(border & 1)); // !(border & 1) => border is even;
    }
  

    params->X1 = X[0];
    params->X2 = X[1];
    params->X3 = X[2];
    params->X4 = X[3];
    return;
  }


} // namespace quda
