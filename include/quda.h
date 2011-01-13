#ifndef _QUDA_H
#define _QUDA_H

#include <enum_quda.h>

#define QUDA_VERSION 000302 // version 0.3.2

#ifdef __cplusplus
extern "C" {
#endif

  // When adding new members to QudaGaugeParam and QudaInvertParam,
  // be sure to update lib/check_params.h

  typedef struct QudaGaugeParam_s {

    int X[4];

    double anisotropy;    // used for Wilson and Wilson-clover
    double tadpole_coeff; // used for staggered only

    QudaLinkType type;
    QudaGaugeFieldOrder gauge_order;

    QudaTboundary t_boundary;

    QudaPrecision cpu_prec;

    QudaPrecision cuda_prec;
    QudaReconstructType reconstruct;

    QudaPrecision cuda_prec_sloppy;
    QudaReconstructType reconstruct_sloppy;

    QudaGaugeFixed gauge_fix;

    int ga_pad;

    int packed_size;
    double gaugeGiB;

  } QudaGaugeParam;

  typedef struct QudaInvertParam_s {

    QudaDslashType dslash_type;
    QudaInverterType inv_type;

    double mass;  // used for staggered only
    double kappa; // used for Wilson and Wilson-clover

    double m5; // domain wall shift parameter
    int Ls; // domain wall 5th dimension

    double mu; // twisted mass parameter
    QudaTwistFlavorType twist_flavor; // twisted mass flavor

    double tol;
    int maxiter;
    double reliable_delta; // reliable update tolerance

    QudaSolutionType solution_type; // type of system to solve
    QudaSolveType solve_type; // how to solve it
    QudaMatPCType matpc_type;
    QudaDagType dagger;
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

    QudaTune dirac_tune; // tune the Dirac operator when it is first created?
    QudaPreserveDirac preserve_dirac; // free the Dirac operator or keep it resident?

  } QudaInvertParam;


  // Interface functions, found in interface_quda.cpp

  void initQuda(int dev);
  void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param);
  void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param);
  void loadCloverQuda(void *h_clover, void *h_clovinv,
		      QudaInvertParam *inv_param);

  void invertQuda(void *h_x, void *h_b, QudaInvertParam *param);
  void invertMultiShiftQuda(void **_hp_x, void *_hp_b, QudaInvertParam *param,
			    double* offsets, int num_offsets,
			    double* residue_sq);

  void endInvertQuda(); // frees the Dirac operator
  
  void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param,
		  QudaParity parity);
  void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param);
  void MatDagMatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param);

  void endQuda(void);

  QudaGaugeParam newQudaGaugeParam(void);
  QudaInvertParam newQudaInvertParam(void);

  void printQudaGaugeParam(QudaGaugeParam *param);
  void printQudaInvertParam(QudaInvertParam *param);

#ifdef __cplusplus
}
#endif

#endif // _QUDA_H
