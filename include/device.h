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
       @brief Create the streams associated with parallel execution.
     */
    void create_context();

    /**
       @brief Free any persistent context state.  Called by endQuda when
       tearing down the library.
     */
    void destroy();

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
    }

  }
}
