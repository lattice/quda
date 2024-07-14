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

  } // namespace monitor

} // namespace quda
