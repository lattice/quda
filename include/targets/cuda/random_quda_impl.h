#pragma once

#include <curand_kernel.h>

namespace quda {

#if defined(XORWOW)
  using cuRNGState =  curandStateXORWOW;
#elif defined(MRG32k3a)
  using cuRNGState =  curandStateMRG32k3a;
#else
  using cuRNGState = curandStateMRG32k3a;
#endif

  /** 
   * \brief random init
   * @param [in] seed -- The RNG seed
   * @param [in] sequence -- The sequence
   * @param [in] offset -- the offset
   * @param [in,out] state - the RNG State 
   */
inline
  __device__ void random_init(unsigned long long seed, unsigned long long sequence, unsigned long long offset, cuRNGState *state) 
{
	curand_init(seed,sequence,offset,state);
}

template<class Real>
struct uniform { };
template<>
struct uniform<float> {

	/** 
	 * \brief Return a uniform deviate between 0 and 1
	 * @param [in,out] the RNG State
	 */
    __device__
        static inline float rand(cuRNGState &state) {
        return curand_uniform(&state);
    }

    /** 
     * \brief return a uniform deviate between a and b
     * @param [in,out] the RNG state
     * @param [in] a (the lower end of the range)
     * @param [in] b (the upper end of the range)
     */
    __device__ 
        static inline float rand(cuRNGState &state, float a, float b){
        return a + (b - a) * curand_uniform(&state);
    }

};
template<>
struct uniform<double> {
      /**
       * \brief Return a uniform deviate between 0 and 1
       * @param [in,out] the RNG State
       */
    __device__
        static inline double rand(cuRNGState &state) {
        return curand_uniform_double(&state);
    }
   
    /**
     * \brief Return a uniform deviate between a and b 
     * @param [in,out] the RNG State
     * @param [in] a -- the lower end of the range
     * @param [in] b -- the high end of the range
     */ 
    __device__
        static inline double rand(cuRNGState &state, double a, double b){
        return a + (b - a) * curand_uniform_double(&state);
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
    __device__
        static inline float rand(cuRNGState &state) {
        return curand_normal(&state);
    }
};
template<>
struct normal<double> {
     /**
      * \brief return a gaussian (normal) deviate with a mean of 0
      * @param [in,out] state
      */
    __device__
        static inline double rand(cuRNGState &state) {
        return curand_normal_double(&state);
    }
};

}
