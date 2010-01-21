#ifndef _GAUGE_QUDA_H
#define _GAUGE_QUDA_H

#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif

  void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, QudaPrecision cuda_prec, QudaPrecision cpu_prec,
			GaugeFieldOrder gauge_order, ReconstructType reconstruct, GaugeFixed gauge_fixed,
			Tboundary t_boundary, int *XX, double anisotropy, int pad);

  void restoreGaugeField(void *cpuGauge, FullGauge *cudaGauge, QudaPrecision cpu_prec, GaugeFieldOrder gauge_order);

  void freeGaugeField(FullGauge *cudaGauge);
  
#ifdef __cplusplus
}
#endif

#endif // _GAUGE_QUDA_H
