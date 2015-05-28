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


  /**
   * Evolve the gauge field by step size dt, using the momentum field
   * I.e., Evalulate U(t+dt) = e(dt pi) U(t).  All fields are CPU fields in MILC order.
   *
   * @param precision Precision of the field (2 - double, 1 - single)
   * @param dt The integration step size step
   * @param momentum The momentum field
   * @param The gauge field to be updated 
   */
  void qudaUpdateU(int precision, 
		   double eps,
		   void* momentum, 
		   void* link);
  
  /**
   * Compute the sigma trace field (part of clover force computation).
   * All the pointers here are for QUDA native device objects.  The
   * precisions of all fields must match.  This function requires that
   * there is a persistent clover field.
   * 
   * @param out Sigma trace field  (QUDA device field, geometry = 1)
   * @param dummy (not used)
   * @param mu mu direction
   * @param nu nu direction
   */
  void qudaCloverTrace(void* out,
		       void* dummy,
		       int mu,
		       int nu);


  /**
   * Compute the derivative of the clover term (part of clover force
   * computation).  All the pointers here are for QUDA native device
   * objects.  The precisions of all fields must match.
   * 
   * @param out Clover derivative field (QUDA device field, geometry = 1)
   * @param gauge Gauge field (extended QUDA device field, gemoetry = 4)
   * @param oprod Matrix field (outer product) which is multiplied by the derivative
   * @param mu mu direction
   * @param nu nu direction
   * @param coeff Coefficient of the clover derviative (including stepsize and clover coefficient)
   * @param precision Precision of the fields (2 = double, 1 = single)
   * @param parity Parity for which we are computing
   * @param conjugate Whether to make the oprod field anti-hermitian prior to multiplication
   */
  void qudaCloverDerivative(void* out,
			    void* gauge,
			    void* oprod, 
			    int mu,
			    int nu,
			    double coeff,
			    int precision,
			    int parity,
			    int conjugate);


  /**
   * Take a gauge field on the host, load it onto the device and extend it.
   * Return a pointer to the extended gauge field object.
   *
   * @param gauge The CPU gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scaler, 4 - vector, 6 - tensor)
   * @param precision The precision of the fields (2 - double, 1 - single)
   * @return Pointer to the gauge field (cast as a void*)
   */
  void* qudaCreateExtendedGaugeField(void* gauge,
				     int geometry,
				     int precision);

  /**
   * Take the QUDA resident gauge field and extend it.
   * Return a pointer to the extended gauge field object.
   *
   * @param gauge The CPU gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scaler, 4 - vector, 6 - tensor)
   * @param precision The precision of the fields (2 - double, 1 - single)
   * @return Pointer to the gauge field (cast as a void*)
   */
  void* qudaResidentExtendedGaugeField(void* gauge,
				       int geometry,
				       int precision);

  /**
   * Allocate a gauge (matrix) field on the device and optionally download a host gauge field.
   *
   * @param gauge The host gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scaler, 4 - vector, 6 - tensor)
   * @param precision The precision of the field to be created (2 - double, 1 - single)
   * @return Pointer to the gauge field (cast as a void*)
   */
  void* qudaCreateGaugeField(void* gauge,
			     int geometry,
			     int precision);

  /**
   * Copy the QUDA gauge (matrix) field on the device to the CPU
   *
   * @param outGauge Pointer to the host gauge field
   * @param inGauge Pointer to the device gauge field (QUDA device field)
   */
  void qudaSaveGaugeField(void* gauge,
			  void* inGauge);

  /**
   * Reinterpret gauge as a pointer to cudaGaugeField and call destructor.
   *
   * @param gauge Gauge field to be freed
   */
  void qudaDestroyGaugeField(void* gauge);

  
#ifdef __cplusplus
}
#endif


#endif // _QUDA_MILC_INTERFACE_H
