#pragma once

#include <lattice_field.h>

#ifdef __CUDACC_RTC__
#define RNG int
#else
#include <qudarand_kernel.h>

namespace quda {

  /*
#if defined(XORWOW)
typedef struct qurandStateXORWOW quRNGState;
#elif defined(MRG32k3a)
typedef struct qurandStateMRG32k3a quRNGState;
#else
typedef struct qurandStateMRG32k3a quRNGState;
#endif
  */
/**
   @brief Class declaration to initialize and hold CURAND RNG states
*/
class RNG {

  private:
  quRNGState *state;        /*! array with current qurand rng state */
  quRNGState *backup_state; /*! array for backup of current qurand rng state */
  unsigned long long seed;  /*! initial rng seed */
  int size;                 /*! @brief number of qurand states */
  int size_cb;        /*! @brief number of qurand states checkerboarded (equal to size if we have a single parity) */
  int X[4];           /*! @brief local lattice dimensions */
  void AllocateRNG(); /*! @brief allocate qurand rng states array in device memory */

  public:
  /**
     @brief Constructor that takes its metadata from a field
     @param[in] meta The field whose data we use
     @param[in] seed Seed to initialize the RNG
  */
  RNG(const LatticeField &meta, unsigned long long seedin);

  /**
     @brief Constructor that takes its metadata from a param
     @param[in] param The param whose data we use
     @param[in] seed Seed to initialize the RNG
   */
  RNG(const LatticeFieldParam &param, unsigned long long seedin);

  /*! free array */
  void Release();

  /*! initialize qurand rng states with seed */
  void Init();

  unsigned long long Seed() { return seed; };

  __host__ __device__ __inline__ quRNGState *State() { return state; };

  /*! @brief Restore CURAND array states initialization */
  void restore();

  /*! @brief Backup CURAND array states initialization */
  void backup();
};


/**
   @brief Return a random number between a and b
   @param state qurand rng state
   @param a lower range
   @param b upper range
   @return  random number in range a,b
*/
template<class Real>
inline  __device__ Real Random(quRNGState &state, Real a, Real b){
    Real res;
    return res;
}

template<>
inline  __device__ float Random<float>(quRNGState &state, float a, float b){
  return a + (b - a) * qurand_uniform(state);
}

template<>
inline  __device__ double Random<double>(quRNGState &state, double a, double b){
    return a + (b - a) * qurand_uniform_double(state);
}

/**
   @brief Return a random number between 0 and 1
   @param state qurand rng state
   @return  random number in range 0,1
*/
template<class Real>
inline  __device__ Real Random(quRNGState &state){
    Real res;
    return res;
}

template<>
inline  __device__ float Random<float>(quRNGState &state){
    return qurand_uniform(state);
}

template<>
inline  __device__ double Random<double>(quRNGState &state){
    return qurand_uniform_double(state);
}


template<class Real>
struct uniform { };
template<>
struct uniform<float> {
    __device__
        static inline float rand(quRNGState &state) {
        return qurand_uniform(state);
    }
};
template<>
struct uniform<double> {
    __device__
        static inline double rand(quRNGState &state) {
        return qurand_uniform_double(state);
    }
};



template<class Real>
struct normal { };
template<>
struct normal<float> {
    __device__
        static inline float rand(quRNGState &state) {
        return qurand_normal(state);
    }
};
template<>
struct normal<double> {
    __device__
        static inline double rand(quRNGState &state) {
        return qurand_normal_double(state);
    }
};


}

#endif
