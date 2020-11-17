#pragma once

#include <quda_define.h>
#include <lattice_field.h>

/* Define templated methods and cuRNGState */
#if defined(QUDA_TARGET_CUDA)
#include "targets/cuda/random_quda_impl.h"
#elif defined(QUDA_TARGET_HIP) 
#include "targets/hip/random_quda_impl.h"
#endif

namespace quda {


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




}
