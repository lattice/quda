#ifndef _QUDA_FORTRAN_H
#define _QUDA_FORTRAN_H

/**
 * @file  quda_fortran.h
 * @brief Fortran interface functions
 *
 * The following are Fortran interface functions to QUDA that mirror
 * the C-equivalents.  This essentially just means making all calls by
 * reference, using all the lower-case characters and adding a trailing
 * underscore.
 */

#include <quda_constants.h>

#ifdef __cplusplus
extern "C" {
#endif

  /**
   * Initialize the library.  This is a low-level interface that is
   * called by initQuda.  Calling initQudaDevice requires that the
   * user also call initQudaMemory before using QUDA.
   *
   * @param device CUDA device number to use.  In a multi-GPU build,
   *               this parameter may either be set explicitly on a
   *               per-process basis or set to -1 to enable a default
   *               allocation of devices to processes.  
   */
  void init_quda_device_(int *device);

  /**
   * Initialize the library persistant memory allocations (both host
   * and device).  This is a low-level interface that is called by
   * initQuda.  Calling initQudaMemory requires that the user has
   * previously called initQudaDevice.
   */
  void init_quda_memory_();

  /**
   * Initialize the library.  Under the interface this just calls
   * initQudaMemory and initQudaDevice.
   *
   * @param device  CUDA device number to use.  In a multi-GPU build,
   *                this parameter may be either set explicitly on a
   *                per-process basis or set to -1 to enable a default
   *                allocation of devices to processes.
   */
  void init_quda_(int *device);

  /**
   * Finalize the library.
   */
  void end_quda_(void);

  /**
   * Setter method for the comm grid size to all us to reuse BQCD's
   * MPI topology.  This is considered a temporary hack that will be
   * fixed when an interface for setting the logical topology is
   * created (issue 31 on github).
   */
  void comm_set_gridsize_(int *grid);

  /**
   * Initializes the QudaGaugeParam with default entries.
   * @param The QudaGaugeParam to be initialized
   */
  void new_quda_gauge_param_(QudaGaugeParam *param);

  /**
   * Initializes the QudaInvertParam with default entries.
   * @param The QudaInvertParam to be initialized
   */
  void new_quda_invert_param_(QudaInvertParam *param);

  /**
   * Load the gauge field from the host.
   * @param h_gauge Base pointer to host gauge field (regardless of dimensionality)
   * @param param   Contains all metadata regarding host and device storage
   */
  void load_gauge_quda_(void *h_gauge, QudaGaugeParam *param);

  /**
   * Free QUDA's internal copy of the gauge field.
   */
  void free_gauge_quda_(void);

  /**
   * Free QUDA's internal copy of the gauge field.
   */
  void free_sloppy_gauge_quda_(void);

  /**
   * Load the clover term and/or the clover inverse from the host.
   * Either h_clover or h_clovinv may be set to NULL.
   * @param h_clover    Base pointer to host clover field
   * @param h_cloverinv Base pointer to host clover inverse field
   * @param inv_param   Contains all metadata regarding host and device storage
   */
  void load_clover_quda_(void *h_clover, void *h_clovinv,
			 QudaInvertParam *inv_param);

  /**
   * Free QUDA's internal copy of the clover term and/or clover inverse.
   */
  void free_clover_quda_(void);

  /**
   * Apply the Dslash operator (D_{eo} or D_{oe}).
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage
   * @param parity The destination parity of the field
   */
  void dslash_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param,
		    QudaParity *parity);

  /**
   * Apply the clover operator or its inverse.
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage
   * @param parity The source and destination parity of the field
   * @param inverse Whether to apply the inverse of the clover term
   */
  void clover_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param,
		    QudaParity *parity, int *inverse);

  /**
   * Apply the full Dslash matrix, possibly even/odd preconditioned.
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage
   */
  void mat_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param);

  /**
   * Apply M^{\dag}M, possibly even/odd preconditioned.
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage
   */
  void mat_dag_mat_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param);

  /**
   * Perform the solve, according to the parameters set in param.  It
   * is assumed that the gauge field has already been loaded via
   * loadGaugeQuda().
   * @param h_x    Solution spinor field
   * @param h_b    Source spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage and solver parameters
   */
  void invert_quda_(void *h_x, void *h_b, QudaInvertParam *param);

  /**
   * Solve for multiple shifts (e.g., masses).
   * @param _hp_x    Array of solution spinor fields (large contiguous allocation)
   * @param _hp_b    Array of source spinor fields
   * @param param  Contains all metadata regarding host and device
   *               storage and solver parameters
   */
  void invert_multishift_quda_(void *_hp_x, void *_hp_b, QudaInvertParam *param);


  /**
   * Evolve the gauge field by step size dt, using the momentum field
   * I.e., Evalulate U(t+dt) = e(dt pi) U(t) 
   *
   * @param gauge The gauge field to be updated 
   * @param momentum The momentum field
   * @param dt The integration step size step
   * @param conj_mom Whether to conjugate the momentum matrix
   * @param exact Whether to use an exact exponential or Taylor expand
   * @param param The parameters of the external fields and the computation settings
   */
  void update_gauge_field_quda_(void* gauge, void* momentum, double *dt, 
				bool *conj_mom, bool *exact, QudaGaugeParam* param);

  /**
   * Compute the naive staggered force.  All fields must be in the same precision.
   *
   * @param mom Momentum field
   * @param dt Integrating step size
   * @param delta Additional scale factor when updating momentum (mom += delta * [force]_TA
   * @param gauge Gauge field (at present only supports resident gauge field)
   * @param x Array of single-parity solution vectors (at present only supports resident solutions)
   * @param gauge_param Gauge field meta data
   * @param invert_param Dirac and solver meta data
   */
  void compute_staggered_force_quda_(void *mom, double *dt, double *delta, void *gauge, void *x,
				     QudaGaugeParam *gauge_param, QudaInvertParam *invert_param);

  /**
   * @brief Compute the gauge force and update the mometum field
   *
   * @param[in,out] mom The momentum field to be updated
   * @param[in] gauge The gauge field from which we compute the force
   * @param[in] num_loop_types Number of loop types in the gauge action
   * @param[in] loop_coeff Coefficients of the different loops in the Symanzik action
   * @param[in] dt The integration step size
   * @param[in] param The parameters of the external fields and the computation settings
   */
  void compute_gauge_force_quda_(void *mom, void *gauge, int *num_loop_types, double *coeff, double *dt,
				 QudaGaugeParam *param);

  /**
   * @brief Apply the staggered phase factors to the resident gauge field
   */
  void apply_staggered_phase_quda_();

  /**
   * @brief Remove the staggered phase factors to the resident gauge field
   */
  void remove_staggered_phase_quda_();

  /**
   * @brief Evaluate the kinetic (momentum) contribution to classical
   * Hamiltonian for Hybrid Monte Carlo.
   *
   * @param kin Kinetic energy
   * @param momentum The momentum field
   * @param param The parameters of the external fields and the computation settings
   * @return momentum action
   */
  void kinetic_quda_(double *kin, void* momentum, QudaGaugeParam* param);

  /**
   * @param Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.
   * @param Array for storing the averages (total, spatial, temporal)
   */
  void plaq_quda_(double plaq[3]);

  /**
   * @brief fTemporary function exposed for TIFR benchmarking
   */
  void set_kernel_pack_t_(int *pack);

  /**
   * @brief Flush the chronological history for the given index
   * @param[in] index Index for which we are flushing
   */
  void flush_chrono_quda_(int *index);

  /**
   * @brief Pinned a pre-existing memory allocation
   * @param[in] ptr Pointer to buffer to be pinned
   * @param[in] size Size of allocation
   */
  void register_pinned_quda_(void *ptr, size_t *bytes);

  /**
   * @brief Pinned a pre-existing memory allocation
   * @param[in] ptr Pointer to buffer to be unpinned
   */
  void unregister_pinned_quda_(void *ptr);

#ifdef __cplusplus
}
#endif 

#endif /* _QUDA_FORTRAN_H */
