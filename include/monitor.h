#include "device.h"

namespace quda
{

  namespace monitor
  {

    /**
       @brief Initialize device monitoring if supported.  On CUDA this
       uses NVML-based monitoring.
    */
    void init();

    /**
       @brief Tear down any state associated with device monitoring
    */
    void destroy();

    /**
       @brief Serlialize the monitor state history to disk.  If
       QUDA_RESOURCE_PATH is not defined then no action is taken
    */
    void serialize();

    /**
       @brief Get the current size of the monitor state.  Used for
       bookending a period for later analysis.
    */
    size_t size();

    struct state_t {
      double energy = 0.0;
      double power = 0.0;
      double temp = 0.0;
      double clock = 0.0;
    };

    /**
       @brief Get the mean state observables between start and end, where
       start and end are two intervals of history in the state.
    */
    state_t mean(size_t start, size_t end);

  } // namespace monitor

} // namespace quda
