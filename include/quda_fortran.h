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
   * Temporary function exposed for TIFR benchmarking
   */
  void set_kernel_pack_t_(int *pack);

#ifdef __cplusplus
}
#endif 

#endif /* _QUDA_FORTRAN_H */
