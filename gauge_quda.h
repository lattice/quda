#ifndef _QUDA_GAUGE_H
#define _QUDA_GAUGE_H

#include <enum_quda.h>
#include <dslash_quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  void allocateGaugeField(FullGauge *gauge, ReconstructType reconstruct, Precision precision);
  void freeGaugeField();

  void createGaugeField(void *gauge);
  
#ifdef __cplusplus
}
#endif

#endif // _QUDA_GAUGE_H
