#pragma once

#include <hiprand_kernel.h>

namespace quda
{

#if defined(XORWOW)
  using rng_state_t = hiprandStateXORWOW;
#elif defined(MRG32K3a)
  using rng_state_t = hiprandStateMRG32k3a;
#else
  using rng_state_t = hiprandStateMRG32k3a;
#endif

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
  __device__ inline void random_init(unsigned long long seed, unsigned long long sequence, unsigned long long offset,
                                     RNGState &state)
  {
    hiprand_init(seed, sequence, offset, &state.state);
  }

  template <class Real> struct uniform {
  };
  template <> struct uniform<float> {

    /**
     * \brief Return a uniform deviate between 0 and 1
     * @param [in,out] the RNG State
     */
    __device__ static inline float rand(RNGState &state) { return hiprand_uniform(&state.state); }

    /**
     * \brief return a uniform deviate between a and b
     * @param [in,out] the RNG state
     * @param [in] a (the lower end of the range)
     * @param [in] b (the upper end of the range)
     */
    __device__ static inline float rand(RNGState &state, float a, float b)
    {
      return a + (b - a) * hiprand_uniform(&state.state);
    }
  };

  template <> struct uniform<double> {
    /**
     * \brief Return a uniform deviate between 0 and 1
     * @param [in,out] the RNG State
     */
    __device__ static inline double rand(RNGState &state) { return hiprand_uniform_double(&state.state); }

    /**
     * \brief Return a uniform deviate between a and b
     * @param [in,out] the RNG State
     * @param [in] a -- the lower end of the range
     * @param [in] b -- the high end of the range
     */
    __device__ static inline double rand(RNGState &state, double a, double b)
    {
      return a + (b - a) * hiprand_uniform_double(&state.state);
    }
  };

  template <class Real> struct normal {
  };

  template <> struct normal<float> {
    /**
     * \brief return a gaussian normal deviate with mean of 0
     * @param [in,out] state
     */
    __device__ static inline float rand(RNGState &state) { return hiprand_normal(&state.state); }
  };

  template <> struct normal<double> {
    /**
     * \brief return a gaussian (normal) deviate with a mean of 0
     * @param [in,out] state
     */
    __device__ static inline double rand(RNGState &state) { return hiprand_normal_double(&state.state); }
  };

} // namespace quda
