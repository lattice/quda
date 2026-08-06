#pragma once

#include <memory>
#include <quda_define.h>
#include <lattice_field.h>

namespace quda {

  // The nature of the state is defined in the target-specific implementation
  struct RNGState;

  /**
     @brief Class declaration to initialize and hold RNG states
  */
  class RNG
  {

    bool is_initialized = false;     /*! @brief whether or not the RNG is initialized */
    size_t size;                     /*! @brief number of rand states */
    std::shared_ptr<RNGState> state; /*! array with current rand rng state */
    RNGState *backup_state;          /*! array for backup of current rand rng state */
    unsigned long long seed;         /*! initial rng seed */

  public:
    /*! @brief Default constructor */
    RNG() = default;

    /**
       @brief Allocate and initialize RNG states.  Constructor that
       takes its metadata from pre-existing field
       @param[in] meta The field whose data we use
       @param[in] seed Seed to initialize the RNG
    */
    RNG(const LatticeField &meta, unsigned long long seedin);

    unsigned long long Seed() { return seed; };

    /*! @brief Check if the RNG is initialized */
    bool isInitialized() { return is_initialized; };

    /*! @brief Restore rng array states initialization */
    void restore();

    /*! @brief Backup rng array states initialization */
    void backup();

    /**
       @brief Number of bytes needed to hold the RNG state array on the
       host, e.g., for checkpointing to disk
    */
    size_t StateBytes();

    /**
       @brief Copy the RNG state array to a host buffer (for
       checkpointing).  The buffer must hold at least StateBytes() bytes.
       @param[out] buffer Host buffer the state is copied to
    */
    void saveState(void *buffer);

    /**
       @brief Restore the RNG state array from a host buffer written by
       saveState().  The buffer contents must come from an RNG of
       identical geometry, process grid, and target backend.
       @param[in] buffer Host buffer the state is read from
    */
    void loadState(const void *buffer);

    /*! @brief Get pointer to RNGState */
    RNGState *State();
  };
}
