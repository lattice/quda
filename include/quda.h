#ifndef _QUDA_H
#define _QUDA_H

#include <enum_quda.h>

#define QUDA_VERSION 000206 // version 0.2.6

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct QudaGaugeParam_s {

    int X[4];

    double anisotropy;

    QudaGaugeFieldOrder gauge_order;

    QudaTboundary t_boundary;

    QudaPrecision cpu_prec;

    QudaPrecision cuda_prec;
    QudaReconstructType reconstruct;

    QudaPrecision cuda_prec_sloppy;
    QudaReconstructType reconstruct_sloppy;

    QudaGaugeFixed gauge_fix;

    int blockDim; // number of threads in a block
    int blockDim_sloppy;

    int ga_pad;

    int packed_size;
    double gaugeGiB;

  } QudaGaugeParam;

  typedef struct QudaInvertParam_s {
    
    QudaDslashType dslash_type;
    QudaInverterType inv_type;

    double kappa;  
    double tol;
    int maxiter;
    double reliable_delta; // reliable update tolerance

    QudaMatPCType matpc_type;
    QudaSolutionType solution_type;
    QudaMassNormalization mass_normalization;

    QudaPreserveSource preserve_source;

    QudaPrecision cpu_prec;
    QudaPrecision cuda_prec;
    QudaPrecision cuda_prec_sloppy;

    QudaDiracFieldOrder dirac_order;

    QudaPrecision clover_cpu_prec;
    QudaPrecision clover_cuda_prec;
    QudaPrecision clover_cuda_prec_sloppy;

    QudaCloverFieldOrder clover_order;

    QudaVerbosity verbosity;

    int sp_pad;
    int cl_pad;

    int iter;
    double spinorGiB;
    double cloverGiB;
    double gflops;
    double secs;

  } QudaInvertParam;

  // Interface functions, found in interface_quda.cpp

  void initQuda(int dev); 
  void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param);
  void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param);
  void loadCloverQuda(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param);

  void invertQuda(void *h_x, void *h_b, QudaInvertParam *param);

  void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int parity, int dagger);

  void dslash3DQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int parity, int dagger);

  void MatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int dagger);
  void MatPCDagMatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param);
  void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int dagger);

  void endQuda(void);

  QudaGaugeParam newQudaGaugeParam(void);
  QudaInvertParam newQudaInvertParam(void);

  void printQudaGaugeParam(QudaGaugeParam *param);
  void printQudaInvertParam(QudaInvertParam *param);

#ifdef __cplusplus
}
#endif

#endif // _QUDA_H
