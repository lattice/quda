#include <random_quda.h>

namespace quda
{

  /**
     @brief Compute the plaquette of the gauge field

     @param[in] U The gauge field upon which to compute the plaquette
     @return double3 variable returning (plaquette, spatial plaquette,
     temporal plaquette) site averages normalized such that each
     plaquette is in the range [0,1]
   */
  double3 plaquette(const GaugeField &U);

  /**
     @brief Generate Gaussian distributed su(N) or SU(N) fields.  If U
     is a momentum field, then we generate random Gaussian distributed
     field in the Lie algebra using the anti-Hermitation convention.
     If U is in the group then we create a Gaussian distributed su(n)
     field and exponentiate it, e.g., U = exp(sigma * H), where H is
     the distributed su(n) field and sigma is the width of the
     distribution (sigma = 0 results in a free field, and sigma = 1 has
     maximum disorder).

     @param[out] U The output gauge field
     @param[in] rngstate random states
     @param[in] sigma Width of Gaussian distrubution
  */

  void gaugeGauss(GaugeField &U, RNG &rngstate, double epsilon);

  /**
     @brief Generate Gaussian distributed su(N) or SU(N) fields.  If U
     is a momentum field, then we generate random Gaussian distributed
     field in the Lie algebra using the anti-Hermitation convention.
     If U is in the group then we create a Gaussian distributed su(n)
     field and exponentiate it, e.g., U = exp(sigma * H), where H is
     the distributed su(n) field and sigma is the width of the
     distribution (sigma = 0 results in a free field, and sigma = 1 has
     maximum disorder).

     @param[out] U The GaugeField
     @param[in] seed The seed used for the RNG
     @param[in] sigma Wdith of the Gaussian distribution
  */

  void gaugeGauss(GaugeField &U, unsigned long long seed, double epsilon);

  /**
     @brief Apply APE smearing to the gauge field

     @param[out] dataDs Output smeared field
     @param[in] dataOr Input gauge field
     @param[in] alpha smearing parameter
  */
  void APEStep(GaugeField &dataDs, const GaugeField &dataOr, double alpha);

  /**
     @brief Apply STOUT smearing to the gauge field

     @param[out] dataDs Output smeared field
     @param[in] dataOr Input gauge field
     @param[in] rho smearing parameter
  */
  void STOUTStep(GaugeField &dataDs, const GaugeField &dataOr, double rho);

  /**
     @brief Apply Over Improved STOUT smearing to the gauge field

     @param[out] dataDs Output smeared field
     @param[in] dataOr Input gauge field
     @param[in] rho smearing parameter
     @param[in] epsilon smearing parameter
  */
  void OvrImpSTOUTStep(GaugeField &dataDs, const GaugeField &dataOr, double rho, double epsilon);

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
     @brief Compute the Fmunu tensor
     @param[out] Fmunu The Fmunu tensor
     @param[in] gauge The gauge field upon which to compute the Fmnu tensor
   */
  void computeFmunu(GaugeField &Fmunu, const GaugeField &gauge);

  /**
     @brief Compute the topological charge
     @param[in] Fmunu The Fmunu tensor, usually calculated from a smeared configuration
     @return double The total topological charge
   */
  double computeQCharge(const GaugeField &Fmunu);

  /**
     @brief Compute the topological charge density per lattice site
     @param[in] Fmunu The Fmunu tensor, usually calculated from a smeared configuration
     @param[out] qDensity The topological charge at each lattice site
     @return double The total topological charge
  */
  double computeQChargeDensity(const GaugeField &Fmunu, void *result);

} // namespace quda
