#pragma once

#include <random>

/**
 * @brief Generate a uniform random number on [lower, upper)
 *
 * @tparam real_t Floating point type
 * @param[in] i Checkboard lattice site
 * @param[in] parity Parity of site
 * @param[in] lower Lower bound of range, inclusive
 * @param[in] upper Upper bound of range, exclusive
 * @return Random number under requested uniform distribution
 */
template <typename real_t = double> real_t random_uniform_host(int i, int parity, real_t lower = 0, real_t upper = 1)
{
  // generates in [lower, upper)
  std::uniform_real_distribution<real_t> dist {lower, upper};
  return dist(host_rand[parity * Vh + i]);
}

/**
 * @brief Generate a Gaussian-distributed random number
 *
 * @tparam real_t Floating point type
 * @param[in] i Checkboard lattice site
 * @param[in] parity Parity of site
 * @param[in] mean Center of the distribution
 * @param[in] stddev Standard deviation of the distribution
 * @return Random number under requested Gaussian distribution
 */
template <typename real_t = double> real_t random_gaussian_host(int i, int parity, real_t mean = 0, real_t stddev = 1)
{
  std::normal_distribution<real_t> dist {mean, stddev};
  return dist(host_rand[parity * Vh + i]);
}
