namespace quda {
  /**
     Compute the plaquette of the gauge field

     @param U The gauge field upon which to compute the plaquette
     @param location The locaiton where to do the computation 
     @return double3 variable returning (plaquette, spatial plaquette,
     temporal plaquette) site averages normalized such that each
     plaquette is in the range [0,1]
   */
  double3 plaquette(const GaugeField& U,
		    QudaFieldLocation location);

  
  /**
     Apply APE smearing to the gauge field

     @param dataDs Output smeared field
     @param dataOr Input gauge field
     @param alpha smearing parameter
     @param location Location of the computation
  */
  void APEStep (GaugeField &dataDs,
		const GaugeField& dataOr,
		double alpha,
		QudaFieldLocation location);


  /**
   * @brief Gauge fixing with overrelaxation with support for single and multi GPU.
   * @param[in,out] data, quda gauge field
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] relax_boost, gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
   * @param[in] tolerance, torelance value to stop the method, if this
   * value is zero then the method stops when iteration reachs the
   * maximum number of steps defined by Nsteps
   * @param[in] reunit_interval, reunitarize gauge field when iteration count is a multiple of this
   * @param[in] stopWtheta, 0 for MILC criterium and 1 to use the theta value
   */
  void gaugefixingOVR( cudaGaugeField& data,
		       const unsigned int gauge_dir,
                       const unsigned int Nsteps,
		       const unsigned int verbose_interval,
		       const double relax_boost,
                       const double tolerance,
		       const unsigned int reunit_interval,
		       const unsigned int stopWtheta);


  /**
   * @brief Gauge fixing with Steepest descent method with FFTs with support for single GPU only.
   * @param[in,out] data, quda gauge field
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] alpha, gauge fixing parameter of the method, most common value is 0.08
   * @param[in] autotune, 1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value 
   * @param[in] tolerance, torelance value to stop the method, if this
   * value is zero then the method stops when iteration reachs the
   * maximum number of steps defined by Nsteps
   * @param[in] stopWtheta, 0 for MILC criterium and 1 to use the theta value
   */
  void gaugefixingFFT( cudaGaugeField& data, const unsigned int gauge_dir,
		       const unsigned int Nsteps,
		       const unsigned int verbose_interval,
		       const double alpha,
		       const unsigned int autotune,
                       const double tolerance,
		       const unsigned int stopWtheta);
  /**
     Compute the Fmunu tensor
     @param Fmunu The Fmunu tensor
     @param gauge The gauge field upon which to compute the Fmnu tensor
     @param location The location of where to do the computation
   */
  void computeFmunu(GaugeField &Fmunu, 
		    const GaugeField& gauge, 
		    QudaFieldLocation location);

  /**
     Compute the topological charge
     @param Fmunu The Fmunu tensor, usually calculated from a smeared configuration
     @param location The location of where to do the computation, currently supports only the GPU
   */

  double computeQCharge(GaugeField& Fmunu, QudaFieldLocation location);
}
