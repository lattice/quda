#pragma once

#include <random>
// #include <curand_kernel.h>

namespace quda {
/*
#if defined(XORWOW)
  using rng_state_t = curandStateXORWOW;
#elif defined(MRG32k3a)
  using rng_state_t = curandStateMRG32k3a;
#else
  using rng_state_t = curandStateMRG32k3a;
#endif
*/
  using rng_state_t = std::mt19937_64;

  struct RNGState {
    rng_state_t state;
  };

  /**
   * \brief random init
   * @param [in] seed -- The RNG seed
   * @param [in] sequence -- The sequence
   * @param [in] offset -- the offset
   * @param [in,out] state - the RNG State
   */
  __device__ inline void random_init(unsigned long long seed, unsigned long long sequence, unsigned long long offset, RNGState &state)
  {
    ompwip();
    std::seed_seq ss{seed, sequence, offset};
    state.state.seed(ss);
    // curand_init(seed, sequence, offset, &state.state);
  }

  template<class Real>
    struct uniform {

    /**
     * \brief Return a uniform deviate between 0 and 1
     * @param [in,out] the RNG State
     */
    __device__ static inline Real rand(RNGState &state)
    {
      ompwip();
      std::uniform_real_distribution<Real> u;
      return u(state.state);
    }

    /**
     * \brief return a uniform deviate between a and b
     * @param [in,out] the RNG state
     * @param [in] a (the lower end of the range)
     * @param [in] b (the upper end of the range)
     */
    __device__ static inline Real rand(RNGState &state, Real a, Real b)
    {
      ompwip();
      std::uniform_real_distribution<Real> u(a,b);
      return u(state.state);
    }

  };

  template<class Real>
    struct normal {    /**
     * \brief return a gaussian normal deviate with mean of 0
     * @param [in,out] state
     */
    __device__ static inline Real rand(RNGState &state)
    {
      ompwip();
      std::normal_distribution<Real> n;
      return n(state.state);
    }
  };

}
