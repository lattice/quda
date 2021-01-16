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

    const LatticeField &meta;
    size_t size;             /*! @brief number of curand states */
    RNGState *state;         /*! array with current curand rng state */
    RNGState *backup_state;  /*! array for backup of current curand rng state */
    unsigned long long seed; /*! initial rng seed */
    bool master;             /*! ideally state would be a shared_ptr and we wouldn't need this */

  public:
    /**
       @brief Allocate and initialize RNG states.  Constructor that
       takes its metadata from pre-existing field
       @param[in] meta The field whose data we use
       @param[in] seed Seed to initialize the RNG
    */
    RNG(const LatticeField &meta, unsigned long long seedin);

    /**
       @brief Copy constructor that does not allocate any new state
       and merely aliases another.
       @param[in] rng Prexisting state we are aliasing
    */
    RNG(const RNG &rng);

    /**
       @brief Release Device memory for CURAND RNG states
    */
    virtual ~RNG();

    unsigned long long Seed() { return seed; };

    /*! @brief Restore rng array states initialization */
    void restore();

    /*! @brief Backup rng array states initialization */
    void backup();

    constexpr RNGState *State() { return state; };
  };

}
