#pragma once

#include <quda_api.h>

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
       @brief Initialize this thread to be able to use the device
       presently initalized for this process.  This will error out if
       init() has not previously been called.
     */
    void init_thread();

    /**
       @brief Get number of devices present on node
    */
    int get_device_count();

    /**
       @brief Get the visible devices string from environmental variable,
       e.g. CUDA_VISIBLE_DEVICES=1,0,2 gives '102'
       @param device_list_string The output string
    */
    void get_visible_devices_string(char device_list_string[128]);

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
       @brief Return the stream with the requested index
       @param i Stream index
       @return Stream requested
    */
    qudaStream_t get_stream(unsigned int i);

    /**
       @brief Return the default stream
       @return Default stream
    */
    qudaStream_t get_default_stream();

    /**
       @brief Return the default stream index
       @return Default stream index
    */
    unsigned int get_default_stream_idx();

    /**
       @brief Report if the target device supports managed memory
       @return Return true if managed memory is supported
     */
    bool managed_memory_supported();

    /**
       @brief Report if the target device supports shared memory atomics
       @return Return true if shared memory atomics are supported
     */
    bool shared_memory_atomic_supported();

    /**
       @brief The maximum default shared memory that a CUDA thread
       block can use.

       @return The maximum shared bytes limit per block
     */
    size_t max_default_shared_memory();

    /**
       @brief Returns the maximum dynamic shared memory per block.
       @return The maximum dynamic shared memory to each block of threads
     */
    size_t max_dynamic_shared_memory();

    /**
       @brief Return the maximum number of threads per block
       @return The maximum number of threads per block
    */
    unsigned int max_threads_per_block();

    /**
       @brief Return the maximum number of threads per processor
       @return The maximum number of threads per processor
    */
    unsigned int max_threads_per_processor();

    /**
       @brief Return the maximum number of threads per block in a given dimension
       @param[in] The dimension we are querying (valid values are 0, 1, 2)
       @return The maximum number of threads per block
    */
    unsigned int max_threads_per_block_dim(int i);

    /**
       @brief Return the maximum grid length in a given dimension
       @param[in] The dimension we are querying (valid values are 0, 1, 2)
       @return The maximum number of blocks per grid
    */
    unsigned int max_grid_size(int i);

    /**
       @brief Return the number of independent processors that a given
       device has.  On CUDA this corresponds to the number of
       streaming multiprocessors.
       @return Number of processors
    */
    unsigned int processor_count();

    /**
     * @brief Returns the maximum number of simultaneously resident
     * blocks per SM.  We can directly query this of CUDA 11, but
     * previously this needed to be hand coded.
     * @return The maximum number of simultaneously resident blocks per SM
     */
    unsigned int max_blocks_per_processor();

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
