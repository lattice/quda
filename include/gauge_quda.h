#ifndef _GAUGE_QUDA_H
#define _GAUGE_QUDA_H

#include <quda_internal.h>
#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, QudaPrecision cuda_prec, QudaPrecision cpu_prec,
			GaugeFieldOrder gauge_order, ReconstructType reconstruct, GaugeFixed gauge_fixed,
			Tboundary t_boundary, int *XX, double anisotropy, double tadpole_coeff, int pad, 
			QudaLinkType type);
  void createGaugeField_mg(FullGauge *cudaGauge, void *cpuGauge, void* ghost_gauge, QudaPrecision cuda_prec, QudaPrecision cpu_prec,
			   GaugeFieldOrder gauge_order, ReconstructType reconstruct, GaugeFixed gauge_fixed,
			   Tboundary t_boundary, int *XX, double anisotropy, double tadpole_coeff, int pad, int num_faces, QudaLinkType flag);

  void restoreGaugeField(void *cpuGauge, FullGauge *cudaGauge, QudaPrecision cpu_prec, GaugeFieldOrder gauge_order);

  void freeGaugeField(FullGauge *cudaGauge);
  
  void loadLinkToGPU(FullGauge cudaGauge, void **cpuGauge, void** ghost_cpuGauge,
		     void** ghost_cpuGuage_diag, QudaGaugeParam* param);
  void storeLinkToCPU(void* cpuGauge, FullGauge *cudaGauge, QudaGaugeParam* param);
  void createLinkQuda(FullGauge* cudaGauge, QudaGaugeParam* param);
  void createStapleQuda(FullStaple* cudaStaple, QudaGaugeParam* param);
  void freeStapleQuda(FullStaple* cudaStaple);
  void createMomQuda(FullMom* cudaMom, QudaGaugeParam* param);
  void freeMomQuda(FullMom *cudaMom);
  void storeMomToCPU(void* mom, FullMom cudaMom, QudaGaugeParam* param);
  void loadMomToGPU(FullMom cudaMom, void* mom, QudaGaugeParam* param);
  void packGhostStaple(FullStaple* cudaStaple, void** fwd_nbr_buf, void** back_nbr_buf, 
		       void* f_norm_buf, void* b_norm_buf, cudaStream_t* stream);
  void  unpackGhostStaple(FullStaple* cudaStaple, void** fwd_nbr_buf, void** back_nbr_buf, 
			  void* f_norm_buf, void* b_norm_buf, cudaStream_t* stream);
  void pack_ghost_all_staples_cpu(void *staple, void **cpuGhostStapleBack, void** cpuGhostStapleFwd, int nFace, QudaPrecision precision);
  void pack_ghost_all_links(void **cpuLink, void **cpuGhostBack, void** cpuGhostFwd, int nFace, QudaPrecision precision);
  void pack_gauge_diag(void* buf, int* X, void** sitelink, int nu, int mu, int dir1, int dir2, QudaPrecision prec);
#define freeLinkQuda freeGaugeField

#define momSiteSize   10 // real numbers per momentum
#define gaugeSiteSize 18 // real numbers per gauge field
  
#ifdef __cplusplus
}
#endif

#endif // _GAUGE_QUDA_H
