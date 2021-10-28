namespace quda
{

  namespace device
  {

    /**
       @brief Create the device context.  Called by initQuda when
       initializing the library.
     */
    void init(int dev);

    /**
       @brief Query and print to stdout device properties of all GPUs
    */
    void print_device_properties();

    /**
       @brief Create the streams associated with parallel execution.
    */
    void create_context();

    /**
       @brief Free any persistent context state.  Called by endQuda when
       tearing down the library.
     */
    void destroy();

    /**
     * @brief Returns the maximum dynamic shared memory per block.
     * @return The maximum dynamic shared memory to each block of threads
     */
    size_t max_dynamic_shared_memory();

    namespace profile
    {

      /**
         @brief Start profiling
       */
      void start();

      /**
         @brief Stop profiling
       */
      void stop();
    } // namespace profile

  } // namespace device
} // namespace quda
