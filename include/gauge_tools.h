#include <random_quda.h>

namespace quda
{

  /**
     Compute the plaquette of the gauge field

     @param U The gauge field upon which to compute the plaquette
     @return double3 variable returning (plaquette, spatial plaquette,
     temporal plaquette) site averages normalized such that each
     plaquette is in the range [0,1]
   */
  double3 plaquette(const GaugeField &U);

  /**
     Generate Gaussian distributed su(N) or SU(N) fields.  If U is a
     momentum field, then we generate random Gaussian distributed
     field in the Lie algebra using the anti-Hermitation convention.
     If U is in the group then we create a Gaussian distributed su(n)
     field and exponentiate it, e.g., U = exp(epsilon H), where H is
     the distributed su(n) field

     @param U The output gauge field
     @param rngstate random states
     @param epsilon Rotation from unit gauge
  */
  void gaugeGauss(GaugeField &U, RNG &rngstate, double epsilon);

  /**
     Generate Gaussian distributed su(N) or SU(N) fields.  If U is a
     momentum field, then we generate random Gaussian distributed
     field in the Lie algebra using the anti-Hermitation convention.
     If U is in the group then we create a Gaussian distributed su(n)
     field and expoentiate it, e.g., U = exp(epsilon H), where H is
     the distributed su(n) field

     @param U The GaugeField
     @param seed The seed used for the RNG
     @param epsilon Rotation from unit gauge
  */
  void gaugeGauss(GaugeField &U, long seed, double epsilon);

  /**
     Apply APE smearing to the gauge field

     @param dataDs Output smeared field
     @param dataOr Input gauge field
     @param alpha smearing parameter
  */
  void APEStep (GaugeField &dataDs,
		const GaugeField& dataOr,
		double alpha);

  /**
     Apply STOUT smearing to the gauge field

     @param dataDs Output smeared field
     @param dataOr Input gauge field
     @param rho smearing parameter
  */
  void STOUTStep (GaugeField &dataDs,
		  const GaugeField& dataOr,
		  double rho);

  /**
     Apply Over Improved STOUT smearing to the gauge field

     @param dataDs Output smeared field
     @param dataOr Input gauge field
     @param rho smearing parameter
     @param epsilon smearing parameter
  */
  void OvrImpSTOUTStep (GaugeField &dataDs,
			const GaugeField& dataOr,
			double rho, double epsilon);


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
		       const int gauge_dir,
                       const int Nsteps,
		       const int verbose_interval,
		       const double relax_boost,
                       const double tolerance,
		       const int reunit_interval,
		       const int stopWtheta);


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
  void gaugefixingFFT( cudaGaugeField& data, const int gauge_dir,
		       const int Nsteps,
		       const int verbose_interval,
		       const double alpha,
		       const int autotune,
                       const double tolerance,
		       const int stopWtheta);
  /**
     Compute the Fmunu tensor
     @param Fmunu The Fmunu tensor
     @param gauge The gauge field upon which to compute the Fmnu tensor
   */
  void computeFmunu(GaugeField &Fmunu, const GaugeField &gauge);

  /**
     Compute the topological charge
     @param Fmunu The Fmunu tensor, usually calculated from a smeared configuration
     @return double The total topological charge
   */
  double computeQCharge(const GaugeField &Fmunu);

  /**
   Compute the topological charge density per lattice site
   @param[in] Fmunu The Fmunu tensor, usually calculated from a smeared configuration
   @param[out] qDensity The topological charge at each lattice site
   @return double The total topological charge
  */
  double computeQChargeDensity(const GaugeField &Fmunu, void *result);
}
