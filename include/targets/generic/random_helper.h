#pragma once

#include <random_quda.h>
#include <mrg32k3a.h>

namespace quda
{

  struct RNGState {
    target::rng::MRG32k3a state;
    bool has_extf, has_extd;
    float extf;
    double extd;
  };

  /**
   * \brief random init
   * @param [in] seed -- The RNG seed
   * @param [in] sequence -- The sequence
   * @param [in] offset -- the offset
   * @param [in,out] state - the RNG State
   */
  constexpr void random_init(unsigned long long seed, unsigned long long sequence, unsigned long long offset,
                             RNGState &state)
  {
    target::rng::seed(state.state, seed, sequence);
    target::rng::skip(state.state, offset);
    state.has_extf = 0;
    state.has_extd = 0;
    state.extf = 0.0f;
    state.extd = 0.0;
  }

  template <class Real> struct uniform {
  };

  template <> struct uniform<float> {

    /**
     * \brief Return a uniform deviate between 0 and 1
     * @param [in,out] the RNG State
     */
    static constexpr float rand(RNGState &state) { return (float)target::rng::uniform(state.state); }

    /**
     * \brief return a uniform deviate between a and b
     * @param [in,out] the RNG state
     * @param [in] a (the lower end of the range)
     * @param [in] b (the upper end of the range)
     */
    static constexpr float rand(RNGState &state, float a, float b)
    {
      return a + (b - a) * (float)target::rng::uniform(state.state);
    }
  };

  template <> struct uniform<double> {
    /**
     * \brief Return a uniform deviate between 0 and 1
     * @param [in,out] the RNG State
     */
    static constexpr double rand(RNGState &state) { return target::rng::uniform(state.state); }

    /**
     * \brief Return a uniform deviate between a and b
     * @param [in,out] the RNG State
     * @param [in] a -- the lower end of the range
     * @param [in] b -- the high end of the range
     */
    static constexpr double rand(RNGState &state, double a, double b)
    {
      return a + (b - a) * target::rng::uniform(state.state);
    }
  };

  template <class Real> struct normal {
  };

  template <> struct normal<float> {
    /**
     * \brief return a gaussian normal deviate with mean of 0
     * @param [in,out] state
     */
    static inline float rand(RNGState &state)
    {
      if (state.has_extf) {
        state.has_extf = 0;
        return state.extf;
      } else {
        float x, y;
        target::rng::gaussian(state.state, x, y);
        state.has_extf = 1;
        state.extf = y;
        return x;
      }
    }
  };

  template <> struct normal<double> {
    /**
     * \brief return a gaussian (normal) deviate with a mean of 0
     * @param [in,out] state
     */
    static inline double rand(RNGState &state)
    {
      if (state.has_extd) {
        state.has_extd = 0;
        return state.extd;
      } else {
        double x, y;
        target::rng::gaussian(state.state, x, y);
        state.has_extd = 1;
        state.extd = y;
        return x;
      }
    }
  };

} // namespace quda
