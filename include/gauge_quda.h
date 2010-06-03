#ifndef _GAUGE_QUDA_H
#define _GAUGE_QUDA_H

#include <quda_internal.h>
#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, QudaPrecision cuda_prec, QudaPrecision cpu_prec,
			GaugeFieldOrder gauge_order, ReconstructType reconstruct, GaugeFixed gauge_fixed,
			Tboundary t_boundary, int *XX, double anisotropy, int pad);

  void restoreGaugeField(void *cpuGauge, FullGauge *cudaGauge, QudaPrecision cpu_prec, GaugeFieldOrder gauge_order);

  void freeGaugeField(FullGauge *cudaGauge);
  
  void loadLinkToGPU(FullGauge cudaGauge, void *cpuGauge, QudaGaugeParam* param);
  void storeLinkToCPU(void* cpuGauge, FullGauge *cudaGauge, QudaGaugeParam* param);
  void createLinkQuda(FullGauge* cudaGauge, QudaGaugeParam* param);
  void createStapleQuda(FullStaple* cudaStaple, QudaGaugeParam* param);
  void freeStapleQuda(FullStaple* cudaStaple);
  void createMomQuda(FullMom* cudaMom, QudaGaugeParam* param);
  void freeMomQuda(FullMom *cudaMom);
  void storeMomToCPU(void* mom, FullMom cudaMom, QudaGaugeParam* param);
  void loadMomToGPU(FullMom cudaMom, void* mom, QudaGaugeParam* param);

#define freeLinkQuda freeGaugeField

#define momSiteSize    10 //real numbers per momentum
#define gaugeSiteSize 18 // real numbers per gauge field
  
#ifdef __cplusplus
}
#endif

#endif // _GAUGE_QUDA_H
