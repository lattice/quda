#ifndef _QUDA_GAUGE_H
#define _QUDA_GAUGE_H

#include <enum_quda.h>
#include <dslash_quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, ReconstructType reconstruct, 
			Precision precision, int *X, double anisotropy);
  void freeGaugeField(FullGauge *cudaCauge);
  
#ifdef __cplusplus
}
#endif

#endif // _QUDA_GAUGE_H
