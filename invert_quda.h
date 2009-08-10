#ifndef _INVERT_QUDA_H
#define _INVERT_QUDA_H

#include <enum_quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct QudaGaugeParam_s {

    int X[4];

    double anisotropy;

    QudaGaugeFieldOrder gauge_order;

    QudaPrecision cpu_prec;

    QudaPrecision cuda_prec;
    QudaReconstructType reconstruct;

    QudaPrecision cuda_prec_sloppy;
    QudaReconstructType reconstruct_sloppy;

    QudaGaugeFixed gauge_fix;

    QudaTboundary t_boundary;

    int packed_size;
    double gaugeGiB;

    int blockDim; // number of threads in a block
    int blockDim_sloppy;
  } QudaGaugeParam;

  typedef struct QudaInvertParam_s {
    
    double kappa;  
    QudaMassNormalization mass_normalization;

    QudaDslashType dslash_type;
    QudaInverterType inv_type;
    double tol;
    int iter;
    int maxiter;
    double reliable_delta; // reliable update tolerance

    QudaMatPCType matpc_type;
    QudaSolutionType solution_type;

    QudaPreserveSource preserve_source;

    QudaPrecision cpu_prec;
    QudaPrecision cuda_prec;
    QudaPrecision cuda_prec_sloppy;

    QudaDiracFieldOrder dirac_order;

    double spinorGiB;
    double gflops;
    double secs;

  } QudaInvertParam;

  // Interface functions
  void initQuda(int dev);
  void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param);
  void invertQuda(void *h_x, void *h_b, QudaInvertParam *param);

  void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int parity, int dagger);
  void MatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int dagger);
  void MatPCDagMatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param);

  void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int dagger);

  void endQuda(void);

  void printGaugeParam(QudaGaugeParam *);
  void printInvertParam(QudaInvertParam *);

#ifdef __cplusplus
}
#endif

#endif // _INVERT_CUDA_H
