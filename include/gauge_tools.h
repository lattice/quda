#include <random_quda.h>
#include <timer.h>

namespace quda
{

  /**
   * @brief Calculates a variety of gauge-field observables.
   * @param[in] Gauge field upon which we are measuring.
   * @param[in,out] param Parameter struct that defines which
   * observables we are making and the resulting observables.
   * @param[in] profile TimeProfile instance used for profiling.
   */
  void gaugeObservables(GaugeField &u, QudaGaugeObservableParam &param, TimeProfile &profile);

  /**
   * @brief Project the input gauge field onto the SU(3) group.  This
   * is a destructive operation.  The number of link failures is
   * reported so appropriate action can be taken.
   *
   * @param U Gauge field that we are projecting onto SU(3)
   * @param tol Tolerance to which the iterative algorithm works
   * @param fails Number of link failures (device pointer)
   */
  void projectSU3(GaugeField &U, double tol, int *fails);

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
  void APEStep(GaugeField &dataDs, GaugeField &dataOr, double alpha);

  /**
     @brief Apply STOUT smearing to the gauge field

     @param[out] dataDs Output smeared field
     @param[in] dataOr Input gauge field
     @param[in] rho smearing parameter
  */
  void STOUTStep(GaugeField &dataDs, GaugeField &dataOr, double rho);

  /**
     @brief Apply Over Improved STOUT smearing to the gauge field

     @param[out] dataDs Output smeared field
     @param[in] dataOr Input gauge field
     @param[in] rho smearing parameter
     @param[in] epsilon smearing parameter
  */
  void OvrImpSTOUTStep(GaugeField &dataDs, GaugeField &dataOr, double rho, double epsilon);

  /**
     @brief Apply Wilson Flow steps W1, W2, Vt to the gauge field.
     This routine assumes that the input and output fields are
     extended, with the input field being exchanged prior to calling
     this function.  On exit from this routine, the output field will
     have been exchanged.
     @param[out] dataDs Output smeared field
     @param[in] dataTemp Temp space
     @param[in] dataOr Input gauge field
     @param[in] epsilon Step size
     @param[in] wflow_type Wilson (1x1) or Symanzik improved (2x1) staples
  */
  void WFlowStep(GaugeField &out, GaugeField &temp, GaugeField &in, double epsilon, QudaWFlowType wflow_type);

  /**
   * @brief Gauge fixing with overrelaxation with support for single and multi GPU.
   * @param[in,out] data, quda gauge field
   * @param[in] fix_param Parameter struct that defines the gauge fixing
   */
  void gaugeFixingOVR(GaugeField &data, QudaGaugeFixParam &fix_param);
  
  /**
   * @brief Gauge fixing with Steepest descent method with FFTs with support for single GPU only.
   * @param[in,out] data, quda gauge field
   * @param[in] fix_param Parameter struct that defines the gauge fixing
   */
  void gaugeFixingFFT(GaugeField &data, QudaGaugeFixParam &fix_param);

  /**
     @brief Compute the Fmunu tensor
     @param[out] Fmunu The Fmunu tensor
     @param[in] gauge The gauge field upon which to compute the Fmnu tensor
   */
  void computeFmunu(GaugeField &Fmunu, const GaugeField &gauge);

  /**
     @brief Compute the topological charge and field energy
     @param[out] energy The total, spatial, and temporal field energy
     @param[out] qcharge The total topological charge
     @param[in] Fmunu The Fmunu tensor, usually calculated from a
     smeared configuration
   */
  void computeQCharge(double energy[3], double &qcharge, const GaugeField &Fmunu);

  /**
     @brief Compute the topological charge, field energy and the
     topological charge density per lattice site
     @param[out] energy The total, spatial, and temporal field energy
     @param[out] qcharge The total topological charge
     @param[out] qdensity The topological charge at each lattice site
     @param[in] Fmunu The Fmunu tensor, usually calculated from a
     smeared configuration
  */
  void computeQChargeDensity(double energy[3], double &qcharge, void *qdensity, const GaugeField &Fmunu);

} // namespace quda
