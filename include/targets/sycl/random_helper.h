#pragma once

#include <random_quda.h>
//#include <curand_kernel.h>

namespace quda {

#if 0
#if defined(XORWOW)
  using rng_state_t = curandStateXORWOW;
#elif defined(MRG32k3a)
  using rng_state_t = curandStateMRG32k3a;
#else
  using rng_state_t = curandStateMRG32k3a;
#endif
  using rng_state_t = int;
#endif

  struct RNGState {
    //rng_state_t state;
    unsigned long long seed;
    unsigned long long sequence;
    unsigned long long offset;
  };

  /**
   * \brief random init
   * @param [in] seed -- The RNG seed
   * @param [in] sequence -- The sequence
   * @param [in] offset -- the offset
   * @param [in,out] state - the RNG State
   */
  inline void random_init(unsigned long long seed, unsigned long long sequence, unsigned long long offset, RNGState &state)
  {
    //curand_init(seed, sequence, offset, &state.state);
    state.seed = seed;
    state.sequence = sequence;
    state.offset = offset;
  }

  template<class Real>
    struct uniform { };
  template<>
    struct uniform<float> {

    /**
     * \brief Return a uniform deviate between 0 and 1
     * @param [in,out] the RNG State
     */
    static inline float rand(RNGState &state)
    {
      //return 0.0; //curand_uniform(&state.state);
      state.sequence += state.offset;
      state.seed += state.sequence;
      return ((float)(state.seed & 0xffffffff))*(1.0f/((float)(0x100000000)));
    }

    /**
     * \brief return a uniform deviate between a and b
     * @param [in,out] the RNG state
     * @param [in] a (the lower end of the range)
     * @param [in] b (the upper end of the range)
     */
    static inline float rand(RNGState &state, float a, float b)
    {
      return a + (b - a) * rand(state);
    }

  };

  template<>
    struct uniform<double> {
    /**
     * \brief Return a uniform deviate between 0 and 1
     * @param [in,out] the RNG State
     */
    static inline double rand(RNGState &state)
    {
      //return 0.0; //curand_uniform_double(&state.state);
      state.sequence += state.offset;
      state.seed += state.sequence;
      return ((double)(state.seed & 0xffffffff))*(1.0f/((double)(0x100000000)));
    }

    /**
     * \brief Return a uniform deviate between a and b
     * @param [in,out] the RNG State
     * @param [in] a -- the lower end of the range
     * @param [in] b -- the high end of the range
     */
    static inline double rand(RNGState &state, double a, double b)
    {
      return a + (b - a) * rand(state);
    }
  };

  template<class Real>
    struct normal { };

  template<>
    struct normal<float> {
    /**
     * \brief return a gaussian normal deviate with mean of 0
     * @param [in,out] state
     */
    static inline float rand(RNGState &state)
    {
      //return 0.0; //curand_normal(&state.state);
      return uniform<float>::rand(state, -1.0f, 1.0f);
    }
  };

  template<>
    struct normal<double> {
    /**
     * \brief return a gaussian (normal) deviate with a mean of 0
     * @param [in,out] state
     */
    static inline double rand(RNGState &state)
    {
      //return 0.0; //curand_normal_double(&state.state);
      return uniform<double>::rand(state, -1.0, 1.0);
    }
  };

}
