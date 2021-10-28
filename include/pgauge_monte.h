#pragma once

#include <gauge_field.h>
#include <random_quda.h>

namespace quda {

  /**
   * @brief Perform heatbath and overrelaxation. Performs nhb heatbath steps followed by nover overrelaxation steps.
   *
   * @param[in,out] data Gauge field
   * @param[in,out] rngstate state of the CURAND random number generator
   * @param[in] Beta inverse of the gauge coupling, beta = 2 Nc / g_0^2
   * @param[in] nhb number of heatbath steps
   * @param[in] nover number of overrelaxation steps
   */
  void Monte(GaugeField &data, RNG &rngstate, double Beta, int nhb, int nover);

  /**
   * @brief Perform a cold start to the gauge field, identity SU(3)
   * matrix, also fills the ghost links in multi-GPU case (no need to
   * exchange data)
   *
   * @param[in,out] data Gauge field
   */
  void InitGaugeField(GaugeField &data);

  /**
   * @brief Perform a hot start to the gauge field, random SU(3)
   * matrix, followed by reunitarization, also exchange borders links
   * in multi-GPU case.
   *
   * @param[in,out] data Gauge field
   * @param[in,out] rngstate state of the CURAND random number generator
   */
  void InitGaugeField(GaugeField &data, RNG &rngstate);

  /**
   * @brief Exchange "borders" between nodes. Although the radius
   * border is 2, it only updates the interior radius border, i.e., at
   * 1 and X[d-2] where X[d] already includes the Radius border, and
   * don't update at 0 and X[d-1] faces.
   *
   * @param[in,out] data Gauge field
   * @param[in] n_dim Number of dimensions to exchange
   * @param[in] parity Field parity
   */
  void PGaugeExchange(GaugeField &data, const int n_dim, const int parity);

  /**
   * @brief Release all allocated memory used to exchange data between nodes
   */
  void PGaugeExchangeFree();

  /**
   * @brief Calculate the Determinant
   *
   * @param[in] data Gauge field
   * @returns double2 complex Determinant value
   */
  double2 getLinkDeterminant(GaugeField &data);

  /**
   * @brief Calculate the Trace
   *
   * @param[in] data Gauge field
   * @returns double2 complex trace value
   */
  double2 getLinkTrace(GaugeField &data);
}
