

#ifndef _QUDA_H
#define _QUDA_H

#include <enum_quda.h>

#define QUDA_VERSION_MAJOR       0
#define QUDA_VERSION_MINOR       3
#define QUDA_VERSION_SUBMINOR   3 
#define QUDA_VERSION ((QUDA_VERSION_MAJOR<<16) | (QUDA_VERSION_MINOR<<8) | QUDA_VERSION_SUBMINOR)

#ifdef __cplusplus
extern "C" {
#endif

  // When adding new members to QudaGaugeParam and QudaInvertParam, 
  // be sure to update lib/check_params.h
#define QUDA_MAX_DIM 6
#define QUDA_MAX_MULTI_SHIFT 32 // the maximum number of shifts for the multi-shift solver

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

    QudaPrecision cuda_prec_precondition;
    QudaReconstructType reconstruct_precondition;

    QudaGaugeFixed gauge_fix;

    int ga_pad;
    int site_ga_pad;  //used in link fattening
    int staple_pad;   //used in link fattening
    int llfat_ga_pad; //used in link fattening
    int packed_size;
    double gaugeGiB;

    int flag;  //generic flag. In fatlink computation, it is used to indicate 
	       //whether to preserve gauge field or not
    
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

    int num_offset; // number of offsets
    double offset[QUDA_MAX_MULTI_SHIFT]; // shift offsets for multi-shift solver
    double tol_offset[QUDA_MAX_MULTI_SHIFT]; // solver tolerance for each offset

    QudaSolutionType solution_type; // type of system to solve
    QudaSolveType solve_type; // how to solve it
    QudaMatPCType matpc_type;
    QudaDagType dagger;
    QudaMassNormalization mass_normalization;

    QudaPreserveSource preserve_source;

    QudaPrecision cpu_prec;
    QudaPrecision cuda_prec;
    QudaPrecision cuda_prec_sloppy;
    QudaPrecision cuda_prec_precondition;

    QudaDiracFieldOrder dirac_order;
    QudaGammaBasis gamma_basis; // the gamma basis of the input and output cpu fields 

    QudaPrecision clover_cpu_prec;
    QudaPrecision clover_cuda_prec;
    QudaPrecision clover_cuda_prec_sloppy;
    QudaPrecision clover_cuda_prec_precondition;

    QudaCloverFieldOrder clover_order;
    QudaUseInitGuess use_init_guess;

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

    int gcrNkrylov;  // maximum size of Krylov space used by solver

    // these parameters are used for domain decomposition
    QudaVerbosity verbosity_precondition; // verbosity of inner Krylov solver
    QudaInverterType inv_type_precondition; // the inner Krylov solver used by preconditioner
    double tol_precondition; // tolerance used by inner solver
    int maxiter_precondition; // max number of iterations used by inner solver    
    QudaPrecision prec_precondition; // the precision for the preconditioned solver

    double omega; // the relaxation parameter that is used in GCR-DD (default = 1.0)

    //int commDim[QUDA_MAX_DIM];
    //int commDimSloppy[QUDA_MAX_DIM];
    //int ghostDim[QUDA_MAX_DIM];
  } QudaInvertParam;


  // Interface functions, found in interface_quda.cpp
  void initQuda(int dev);
  void disableNumaAffinityQuda(void);
  void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param);
  void freeGaugeQuda(void);

  void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param);
  void loadCloverQuda(void *h_clover, void *h_clovinv,
		      QudaInvertParam *inv_param);
  void freeCloverQuda(void);

  void invertQuda(void *h_x, void *h_b, QudaInvertParam *param);
  void invertMultiShiftQuda(void **_hp_x, void *_hp_b, QudaInvertParam *param,
			    double* offsets, int num_offsets,
			    double* residue_sq);
  void invertMultiShiftQudaMixed(void **_hp_x, void *_hp_b, QudaInvertParam *param,
				 double* offsets, int num_offsets, double* residue_sq);
    

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


  void  record_gauge(int* X, void *_fatlink, int _fatlink_pad, 
		     void* _longlink, int _longlink_pad, 
		     QudaReconstructType _longlink_recon,QudaReconstructType _longlink_recon_sloppy,
		     QudaGaugeParam *_param);

  // these are temporary additions until we objectify the gauge field
  void set_dim(int *);
  void pack_ghost(void **cpuLink, void **cpuGhost, int nFace, QudaPrecision precision);
 

  void setFatLinkPadding(QudaComputeFatMethod method, QudaGaugeParam* param);
 
  int computeFatLinkQuda(void* fatlink, void** sitelink, double* act_path_coeff, 
			 QudaGaugeParam* param, 
			 QudaComputeFatMethod method);
  
  
#ifdef HOST_DEBUG
#define CUERR  do{ cudaError_t cuda_err;                                \
    if ((cuda_err = cudaGetLastError()) != cudaSuccess) {               \
      fprintf(stderr, "ERROR: CUDA error: %s, line %d, function %s, file %s\n", \
              cudaGetErrorString(cuda_err),  __LINE__, __FUNCTION__, __FILE__); \
      exit(cuda_err);}}while(0)
#else
#define CUERR
#endif

extern int verbose;
  
#ifdef MULTI_GPU
#define PRINTF(fmt,...) do{						\
    if (verbose){							\
      printf("[%d]"fmt, comm_rank(), ##__VA_ARGS__);			\
    }else{								\
      if (comm_rank()==0){						\
	printf("[%d]"fmt, comm_rank(), ##__VA_ARGS__);			\
      }									\
    }									\
  }while(0)	
#else
#define PRINTF printf
#endif

  // Initializes a communications world
  void initCommsQuda(int argc, char **argv, const int *X, const int nDim);
  // Ends a communications world
  void endCommsQuda();

#ifdef __cplusplus
}
#endif

#endif // _QUDA_H
