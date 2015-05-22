
#ifndef RANDOM_GPU_H
#define RANDOM_GPU_H

//#include <stdio.h>
//#include <string.h>
//#include <iostream>
#include <curand_kernel.h>

namespace quda {

#if defined(XORWOW)
typedef struct curandStateXORWOW cuRNGState;
#elif defined(MRG32k3a)
typedef struct curandStateMRG32k3a cuRNGState;
#else
typedef struct curandStateMRG32k3a cuRNGState;
#endif

/**
    @brief Class declaration to initialize and hold CURAND RNG states
*/
class RNG {
public:
    RNG(int rng_sizes, int seedin, int XX[4]);
    RNG(int rng_sizes, int seedin);
    /*! free array */
    void Release(); 
    /*! initialize curand rng states with seed */
    void Init();
    /*! @brief return curand rng array size */
    int Size(){ return rng_size;};
    int Node_Offset(){ return node_offset;};
    int Seed(){ return seed;};
    cuRNGState* State(){ return state;};
private:
    /*! array with current curand rng state */
    cuRNGState *state;
    /*! initial rng seed */
    int seed;
    /*! @brief number of curand states */
    int rng_size;  
    /*! @brief offset in the index, in case of multigpus */
    int node_offset;
    int X[4];
    /*! @brief allocate curand rng states array in device memory */
    void AllocateRNG();
    /*! @brief CURAND array states initialization */
    void INITRNG(int rng_sizes, int seedin, int offsetin);
};





/**
   @brief Return a random number between a and b
   @param state curand rng state
   @param a lower range
   @param b upper range
   @return  random number in range a,b
*/
template<class Real>
inline  __device__ Real Random(cuRNGState &state, Real a, Real b){
    Real res;
    return res;
}
 
template<>
inline  __device__ float Random<float>(cuRNGState &state, float a, float b){
    return a + (b - a) * curand_uniform(&state);
}

template<>
inline  __device__ double Random<double>(cuRNGState &state, double a, double b){
    return a + (b - a) * curand_uniform_double(&state);
}

/**
   @brief Return a random number between 0 and 1
   @param state curand rng state
   @return  random number in range 0,1
*/
template<class Real>
inline  __device__ Real Random(cuRNGState &state){
    Real res;
    return res;
}
 
template<>
inline  __device__ float Random<float>(cuRNGState &state){
    return curand_uniform(&state);
}

template<>
inline  __device__ double Random<double>(cuRNGState &state){
    return curand_uniform_double(&state);
}


template<class Real>
struct uniform { };
template<>
struct uniform<float> {
    __device__
        static inline float rand(cuRNGState &state) {
        return curand_uniform(&state);
    }
};
template<>
struct uniform<double> {
    __device__
        static inline double rand(cuRNGState &state) {
        return curand_uniform_double(&state);
    }
};



template<class Real>
struct normal { };
template<>
struct normal<float> {
    __device__
        static inline float rand(cuRNGState &state) {
        return curand_normal(&state);
    }
};
template<>
struct normal<double> {
    __device__
        static inline double rand(cuRNGState &state) {
        return curand_normal_double(&state);
    }
};


}

#endif 
