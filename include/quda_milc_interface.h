#ifndef _QUDA_MILC_INTERFACE_H
#define _QUDA_MILC_INTERFACE_H

#include <enum_quda.h>
#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct {
    int max_iter;
    QudaParity evenodd; // options are QUDA_EVEN_PARITY, QUDA_ODD_PARITY, QUDA_INVALID_PARITY
    int mixed_precision;
    double boundary_phase[4];
  } QudaInvertArgs_t;


  typedef struct {
    const int* latsize;
    const int* machsize; // grid size
    int device; // device  number 
  } QudaLayout_t; 


  typedef struct {
    QudaVerbosity verbosity;
    QudaLayout_t layout;
  } QudaInitArgs_t; // passed to the initialization struct


  typedef struct {
    int reunit_allow_svd;
    int reunit_svd_only;
    double reunit_svd_abs_error;
    double reunit_svd_rel_error;
    double force_filter;  
  } QudaHisqParams_t;


  typedef struct {
    int su3_source;     // is the incoming gauge field su3?
    int use_pinned_memory;  // use page-locked memory in Quda?
  } QudaFatLinkArgs_t;


  void qudaInit(QudaInitArgs_t input);

  void qudaSetLayout(QudaLayout_t layout);

  void qudaFinalize();


  void qudaHisqParamsInit(QudaHisqParams_t hisq_params);


  void qudaLoadKSLink(int precision, QudaFatLinkArgs_t fatlink_args, const double act_path_coeff[6], void* inlink, void* fatlink, void* longlink);


  void qudaLoadUnitarizedLink(int precision, QudaFatLinkArgs_t fatlink_args, const double path_coeff[6], void* inlink, void* fatlink, void* ulink);


  void qudaInvert(int external_precision,
      int quda_precision,
      double mass,
      QudaInvertArgs_t inv_args,
      double target_resid,
      double target_relresid,
      const void* const milc_fatlink,
      const void* const milc_longlink,
      const double tadpole,
      void* source,
      void* solution,
      double* const final_resid,
      double* const final_rel_resid,
      int* num_iters); 


  void qudaDDInvert(int external_precision,
      int quda_precision,
      double mass,
      QudaInvertArgs_t inv_args,
      double target_residual,
      double target_fermilab_residual,
      const int * const domain_overlap,
      const void* const fatlink,
      const void* const longlink,
      void* source,
      void* solution,
      double* const final_residual,
      double* const final_fermilab_residual,
      int* num_iters);


  void qudaMultishiftInvert(
      int external_precision,    
      int precision, 
      int num_offsets,
      double* const offset,
      QudaInvertArgs_t inv_args,
      const double* target_residual,
      const double* target_relative_residual,
      const void* const milc_fatlink,
      const void* const milc_longlink,
      const double tadpole,
      void* source,
      void** solutionArray, 
      double* const final_residual,
      double* const final_relative_residual,
      int* num_iters);


  void qudaCloverInvert(int external_precision, 
      int quda_precision,
      double kappa,
      double clover_coeff,
      QudaInvertArgs_t inv_args,
      double target_residual,
      double target_fermilab_residual,
      const void* milc_link,
      void* milc_clover, 
      void* milc_clover_inv,
      void* source,
      void* solution,
      double* const final_residual, 
      double* const final_fermilab_residual,
      int* num_iters
      );

  void qudaLoadGaugeField(int external_precision, 
      int quda_precision,
      QudaInvertArgs_t inv_args,
      const void* milc_link) ;

  void qudaFreeGaugeField();

  void qudaLoadCloverField(int external_precision, 
      int quda_precision,
      QudaInvertArgs_t inv_args,
      void* milc_clover, 
      void* milc_clover_inv,
      QudaSolutionType solution_type,
      QudaSolveType solve_type,
      double clover_coeff,
      int compute_trlog,
      double *trlog) ;

  void qudaFreeCloverField();

  void qudaCloverMultishiftInvert(int external_precision, 
      int quda_precision,
      int num_offsets,
      double* const offset,
      double kappa,
      double clover_coeff,
      QudaInvertArgs_t inv_args,
      const double* target_residual,
      const void* milc_link,
      void* milc_clover, 
      void* milc_clover_inv,
      void* source,
      void** solutionArray,
      double* const final_residual, 
      int* num_iters
      );

  void qudaCloverMultishiftMDInvert(int external_precision, 
      int quda_precision,
      int num_offsets,
      double* const offset,
      double kappa,
      double clover_coeff,
      QudaInvertArgs_t inv_args,
      const double* target_residual,
      const void* milc_link,
      void* milc_clover, 
      void* milc_clover_inv,
      void* source,
      void** psiEven,
      void** psiOdd,
      void** pEven,
      void** pOdd,
      double* const final_residual, 
      int* num_iters
      );

  void qudaHisqForce(
      int precision,
      const double level2_coeff[6],
      const double fat7_coeff[6],
      const void* const staple_src[4],
      const void* const one_link_src[4],
      const void* const naik_src[4],
      const void* const w_link,
      const void* const v_link,
      const void* const u_link,
      void* const milc_momentum);


  void qudaAsqtadForce(
      int precision,
      const double act_path_coeff[6],
      const void* const one_link_src[4],
      const void* const naik_src[4],
      const void* const link,
      void* const milc_momentum);


  void qudaGaugeForce(int precision,
      int num_loop_types,
      double milc_loop_coeff[3],
      double eb3,
      void* milc_sitelink,
      void* milc_momentum);


  void qudaComputeOprod(int precision,
      int num_terms,
      double** coeff,
      void** quark_field,
      void* oprod[2]);


  void qudaUpdateU(int precision, 
      double eps,
      void* momentum, 
      void* link);

  void qudaCloverTrace(void* out, void* clover, int mu, int nu);


  void qudaCloverDerivative(void* out, void* gauge, void* oprod, 
      int mu, int nu, double coeff, int precision, int parity, int conjugate);


  void* qudaCreateExtendedGaugeField(void* gauge, int geometry, int precision);

  void* qudaCreateGaugeField(void* gauge, int geometry, int precision);




  /**
   * @brief Gauge fixing with overrelaxation with support for single and multi GPU.
   * @param[in] precision, 1 for single precision else for double precision
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] relax_boost, gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
   * @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when iteration reachs the maximum number of steps defined by Nsteps
   * @param[in] reunit_interval, reunitarize gauge field when iteration count is a multiple of this
   * @param[in] stopWtheta, 0 for MILC criterium and 1 to use the theta value
   * @param[in,out] milc_sitelink, MILC gauge field to be fixed
   */
  void qudaGaugeFixingOVR( const int precision,
    const unsigned int gauge_dir, 
    const int Nsteps,
    const int verbose_interval,
    const double relax_boost,
    const double tolerance,
    const unsigned int reunit_interval,
    const unsigned int stopWtheta,
    void* milc_sitelink
    );


  /**
   * @brief Gauge fixing with Steepest descent method with FFTs with support for single GPU only.
   * @param[in] precision, 1 for single precision else for double precision
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] alpha, gauge fixing parameter of the method, most common value is 0.08
   * @param[in] autotune, 1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value 
   * @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when iteration reachs the maximum number of steps defined by Nsteps
   * @param[in] stopWtheta, 0 for MILC criterium and 1 to use the theta value
   * @param[in,out] milc_sitelink, MILC gauge field to be fixed
   */
  void qudaGaugeFixingFFT( int precision,
    unsigned int gauge_dir, 
    int Nsteps,
    int verbose_interval,
    double alpha,
    unsigned int autotune,
    double tolerance,
    unsigned int stopWtheta,
    void* milc_sitelink
    );


  void qudaSaveGaugeField(void* gauge, void* inGauge);

  void qudaDestroyGaugeField(void* gauge);


#ifdef __cplusplus
}
#endif


#endif // _QUDA_MILC_INTERFACE_H
