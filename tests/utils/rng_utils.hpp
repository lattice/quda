#pragma once

#define HYPERCUBIC_RNG

/**
 * @brief Constexpr of whether or not to use the host hypercubic RNG
 *
 * @return Whether or not to use the host hypercubic RNG
 */
constexpr bool use_hypercubic_host_rng()
{
#ifdef HYPERCUBIC_RNG
  return true;
#else
  return false;
#endif
}

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
  if constexpr (use_hypercubic_host_rng()) {
    // generates in [lower, upper)
    std::uniform_real_distribution<real_t> dist {lower, upper};
    return dist(host_rand[parity * Vh + i]);
  } else {
    return ((upper - lower) * static_cast<real_t>(rand()) / RAND_MAX) + lower;
  }
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
  if constexpr (use_hypercubic_host_rng()) {
    std::normal_distribution<real_t> dist {mean, stddev};
    return dist(host_rand[parity * Vh + i]);
  } else {
    // Box-Muller generates two random numbers at a time, so cache
    // a previous one if appropriate
    static bool number_waiting = false;
    static double backup_number = 0;

    if (number_waiting) {
      number_waiting = false;
      return static_cast<real_t>(backup_number);
    } else {
      // uniform numbers on (0, 1)
      int u1 = 0, u2 = 0;
      while (u1 == 0) { u1 = rand(); }
      while (u2 == 0) { u2 = rand(); }

      auto lnu1 = stddev * sqrt(-2. * log((double)u1 / RAND_MAX));
      double sn, cs;
      sincos(2. * M_PI * (double)u2 / RAND_MAX, &sn, &cs);

      backup_number = lnu1 * sn + mean;
      number_waiting = true;

      return static_cast<real_t>(lnu1 * cs + mean);
    }
  }
}