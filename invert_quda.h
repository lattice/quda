#ifndef _INVERT_QUDA_H
#define _INVERT_QUDA_H

#include <enum_quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct QudaGaugeParam_s {

    int X;
    int Y;
    int Z;
    int T;

    float anisotropy;

    QudaGaugeFieldOrder gauge_order;

    QudaPrecision cpu_prec;
    QudaPrecision cuda_prec;

    QudaReconstructType reconstruct;
    QudaGaugeFixed gauge_fix;

    QudaTboundary t_boundary;

    int packed_size;
    float gaugeGiB;

  } QudaGaugeParam;

  typedef struct QudaInvertParam_s {
    
    float kappa;  
    QudaMassNormalization mass_normalization;

    QudaInverterType inv_type;
    float tol;
    int iter;
    int maxiter;

    QudaMatPCType matpc_type;
    QudaSolutionType solution_type;

    QudaPreserveSource preserve_source;

    QudaPrecision cpu_prec;
    QudaPrecision cuda_prec;
    QudaDiracFieldOrder dirac_order;

    float spinorGiB;
    float gflops;
    float secs;

  } QudaInvertParam;

  // Interface functions
  void initQuda(int dev);
  void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param);
  void invertQuda(void *h_x, void *h_b, QudaInvertParam *param);
  void endQuda(void);


#ifdef __cplusplus
}
#endif

#endif // _INVERT_CUDA_H
