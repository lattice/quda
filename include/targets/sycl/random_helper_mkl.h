#pragma once

#include <random_quda.h>
#include <oneapi/mkl/rng/device.hpp>

namespace drng = oneapi::mkl::rng::device;

namespace quda {

  struct RNGState {
    drng::mrg32k3a<1> state;
    //double next_gauss;
    //bool next_valid;
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
    std::array<std::uint32_t, 6> init;
    for(int i=0; i<6; i++) { init[i] = 12345; }
    if(seed != 0) {
      double d1 = 12345.0 * (((unsigned int)seed) ^ 0x55555555);
      double d2 = 12345.0 * ((seed >> 32) ^ 0xaaaaaaaa);
      init[0] = fmod(d1, 4294967087.0);
      init[1] = fmod(d2, 4294967087.0);
      init[2] = fmod(d1, 4294967087.0);
      init[3] = fmod(d2, 4294944443.0);
      init[4] = fmod(d1, 4294944443.0);
      init[5] = fmod(d2, 4294944443.0);
    }
    auto seed_list = {init[0],init[1],init[2],init[3],init[4],init[5]};
    std::initializer_list<std::uint64_t> num_to_skip = {offset, (1<<12)*sequence};
    state.state = drng::mrg32k3a<1>(seed_list, num_to_skip);
    //state.next_valid = false;
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
      drng::uniform<float> distr;
      return generate(distr, state.state);
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
      drng::uniform<float> distr(a,b);
      return generate(distr, state.state);
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
      drng::uniform<double> distr;
      return generate(distr, state.state);
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
      drng::uniform<double> distr(a,b);
      return generate(distr, state.state);
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
#if 0
      float x = 0;
      if(state.next_valid) {
	x = state.next_gauss;
	state.next_valid = false;
      } else {
	drng::gaussian<float> distr;
	x = generate(distr, state.state);
	state.next_gauss = generate(distr, state.state);
	state.next_valid = true;
      }
      return x;
#else
      drng::gaussian<float> distr;
      return generate(distr, state.state);
#endif
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
#if 0
      double x = 0;
      if(state.next_valid) {
	x = state.next_gauss;
	state.next_valid = false;
      } else {
	drng::gaussian<double> distr;
	x = generate(distr, state.state);
	state.next_gauss = generate(distr, state.state);
	//state.next_valid = true;
      }
      return x;
#else
      drng::gaussian<double> distr;
      return generate(distr, state.state);
#endif
    }
  };

}
