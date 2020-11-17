#pragma once

#include <hiprand_kernel.h>

namespace quda {

#if defined(XORWOW)
  using cuRNGState =  hiprandStateXORWOW;
#elif defined(MRG32k3a)
  using cuRNGState =  hiprandStateMRG32k3a;
#else
  using cuRNGState = hiprandStateMRG32k3a;
#endif

inline
  __device__ void random_init(unsigned long long seed, unsigned long long sequence, unsigned long long offset, cuRNGState *state) 
{
	hiprand_init(seed,sequence,offset,state);
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
        return hiprand_uniform(&state);
    }

    /** 
     * \brief return a uniform deviate between a and b
     * @param [in,out] the RNG state
     * @param [in] a (the lower end of the range)
     * @param [in] b (the upper end of the range)
     */
    __device__ 
        static inline float rand(cuRNGState &state, float a, float b){
        return a + (b - a) * hiprand_uniform(&state);
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
        return hiprand_uniform_double(&state);
    }
   
    /**
     * \brief Return a uniform deviate between a and b 
     * @param [in,out] the RNG State
     * @param [in] a -- the lower end of the range
     * @param [in] b -- the high end of the range
     */ 
    __device__
        static inline double rand(cuRNGState &state, double a, double b){
        return a + (b - a) * hiprand_uniform_double(&state);
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
        return hiprand_normal(&state);
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
        return hiprand_normal_double(&state);
    }
};

}
