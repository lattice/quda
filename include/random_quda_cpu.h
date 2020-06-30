#pragma once

#include <lattice_field.h>

struct qurandStateXORWOW {
  int dum;
};

struct qurandStateMRG32k3a {
  int dum;
};

template <typename T>
float qurand_uniform(T *state)
{
  return 0.0f;
}

template <typename T>
double qurand_uniform_double(T *state)
{
  return 0.0;
}

template <typename T>
float qurand_normal(T *state)
{
  return 0.0f;
}

template <typename T>
double qurand_normal_double(T *state)
{
  return 0.0;
}

namespace quda {
#if defined(XORWOW)
typedef struct qurandStateXORWOW cuRNGState;
#elif defined(MRG32k3a)
typedef struct qurandStateMRG32k3a cuRNGState;
#else
typedef struct qurandStateMRG32k3a cuRNGState;
#endif

/**
   @brief Class declaration to initialize and hold CURAND RNG states
*/
class RNG {

  private:
  cuRNGState *state;        /*! array with current curand rng state */
  cuRNGState *backup_state; /*! array for backup of current curand rng state */
  unsigned long long seed;  /*! initial rng seed */
  int size;                 /*! @brief number of curand states */
  int size_cb;        /*! @brief number of curand states checkerboarded (equal to size if we have a single parity) */
  int X[4];           /*! @brief local lattice dimensions */
  void AllocateRNG(); /*! @brief allocate curand rng states array in device memory */

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

  /*! initialize curand rng states with seed */
  void Init();

  unsigned long long Seed() { return seed; };

  __host__ __device__ __inline__ cuRNGState *State() { return state; };

  /*! @brief Restore CURAND array states initialization */
  void restore();

  /*! @brief Backup CURAND array states initialization */
  void backup();
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
    return a + (b - a) * qurand_uniform(&state);
}

template<>
inline  __device__ double Random<double>(cuRNGState &state, double a, double b){
    return a + (b - a) * qurand_uniform_double(&state);
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
    return qurand_uniform(&state);
}

template<>
inline  __device__ double Random<double>(cuRNGState &state){
    return qurand_uniform_double(&state);
}


template<class Real>
struct uniform { };
template<>
struct uniform<float> {
    __device__
        static inline float rand(cuRNGState &state) {
        return qurand_uniform(&state);
    }
};
template<>
struct uniform<double> {
    __device__
        static inline double rand(cuRNGState &state) {
        return qurand_uniform_double(&state);
    }
};



template<class Real>
struct normal { };
template<>
struct normal<float> {
    __device__
        static inline float rand(cuRNGState &state) {
        return qurand_normal(&state);
    }
};
template<>
struct normal<double> {
    __device__
        static inline double rand(cuRNGState &state) {
        return qurand_normal_double(&state);
    }
};


}
