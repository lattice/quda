#ifndef _HW_QUDA_H
#define _HW_QUDA_H

#include <enum_quda.h>
#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif

  FullHw createHwQuda(int* X, QudaPrecision precision);
  void loadHwToGPU(FullHw ret, void* hw, QudaPrecision cpu_prec);  
  void freeHwQuda(FullHw hw);
    
#ifdef __cplusplus
}
#endif

#endif // _HW_QUDA_H
