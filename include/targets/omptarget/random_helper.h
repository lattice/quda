#pragma once

#include <random_quda.h>

// We don't use the internal Box-Muller state.
#define ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE
#include <externals/rocrand/rocrand_mrg32k3a.h>

namespace quda {

  struct RNGState {
    rocrand_state_mrg32k3a state;
  };

  /**
   * \brief random init
   * @param [in] seed -- The RNG seed
   * @param [in] sequence -- The sequence
   * @param [in] offset -- the offset
   * @param [in,out] state - the RNG State
   */
  inline void random_init(unsigned long long seed, unsigned long long sequence,
			  unsigned long long offset, RNGState &state)
  {
    //curand_init(seed, sequence, offset, &state.state);
    state.state.seed(seed, sequence, offset);
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
      //curand_uniform(&state.state);
      return ROCRAND_2POW32_INV + (rocrand(&state.state) * ROCRAND_2POW32_INV);
    }

    /**
     * \brief return a uniform deviate between a and b
     * @param [in,out] the RNG state
     * @param [in] a (the lower end of the range)
     * @param [in] b (the upper end of the range)
     */
    static inline float rand(RNGState &state, float a, float b)
    {
      //return a + (b - a) * rand(state);
      return a + (b - a) * (ROCRAND_2POW32_INV + (rocrand(&state.state) * ROCRAND_2POW32_INV));
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
      //curand_uniform_double(&state.state);
      return ROCRAND_2POW32_INV_DOUBLE + (rocrand(&state.state) * ROCRAND_2POW32_INV_DOUBLE);
    }

    /**
     * \brief Return a uniform deviate between a and b
     * @param [in,out] the RNG State
     * @param [in] a -- the lower end of the range
     * @param [in] b -- the high end of the range
     */
    static inline double rand(RNGState &state, double a, double b)
    {
      //return a + (b - a) * rand(state);
      return a + (b - a) * (ROCRAND_2POW32_INV_DOUBLE + (rocrand(&state.state) * ROCRAND_2POW32_INV_DOUBLE));
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
      //curand_normal(&state.state);
      float u = ROCRAND_2POW32_INV + (rocrand(&state.state) * ROCRAND_2POW32_INV);
      float v = (ROCRAND_2POW32_INV + (rocrand(&state.state) * ROCRAND_2POW32_INV)) * ROCRAND_2PI;
      float s = sqrtf(-2.0f * logf(u));
      return sinf(v) * s;
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
      //curand_normal_double(&state.state);
      double u = ROCRAND_2POW32_INV_DOUBLE + (rocrand(&state.state) * ROCRAND_2POW32_INV_DOUBLE);
      double v = (ROCRAND_2POW32_INV_DOUBLE + (rocrand(&state.state) * ROCRAND_2POW32_INV_DOUBLE)) * ROCRAND_PI_DOUBLE * 2.;
      double s = sqrt(-2. * log(u));
      return sin(v) * s;
    }
  };

}
