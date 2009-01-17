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

    GaugeFieldOrder gauge_order;

    Precision cpu_prec;
    Precision cuda_prec;

    ReconstructType reconstruct;
    GaugeFixed gauge_fix;

    Tboundary t_boundary;

    int packed_size;
    float gaugeGiB;

  } QudaGaugeParam;

  typedef struct QudaInvertParam_s {
    
    float kappa;  
    MassNormalization mass_normalization;

    InverterType inv_type;
    float tol;
    int iter;
    int maxiter;

    MatPCType matpc_type;
    SolutionType solution_type;

    PreserveSource preserve_source;

    Precision cpu_prec;
    Precision cuda_prec;
    DiracFieldOrder dirac_order;

    float spinorGiB;
    float gflops;
    float secs;

  } QudaInvertParam;

  void initQuda(int dev);
  void loadQuda(void *h_gauge, QudaGaugeParam *param);
  void endQuda(void);

  void invertQuda(void *h_x, void *h_b, QudaInvertParam *param);

#ifdef __cplusplus
}
#endif

#endif // _INVERT_CUDA_H
