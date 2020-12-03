#pragma once

#include <quda_define.h>
#include <lattice_field.h>

namespace quda {

  // The nature of the state is defined in the target-specific implementation
  struct RNGState;

  /**
     @brief Class declaration to initialize and hold RNG states
  */
  class RNG {

    RNGState *state;        /*! array with current curand rng state */
    RNGState *backup_state; /*! array for backup of current curand rng state */
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

    /*! initialize rng states with seed */
    void Init();

    unsigned long long Seed() { return seed; };

    /*! @brief Restore rng array states initialization */
    void restore();

    /*! @brief Backup rng array states initialization */
    void backup();

    constexpr RNGState *State() { return state; };
  };

}
>>>>>>> feature/generic_kernel
