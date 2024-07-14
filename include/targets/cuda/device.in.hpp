#pragma once

namespace quda
{

  namespace device
  {

    constexpr int maximum_dynamic_shared_memory_override = @QUDA_MAX_SHARED_MEMORY@;

    /**
      @brief A constexpr function to returns the maximum dyanmic shared memory per block.
        See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#feature-availability
     */
    constexpr int maximum_dynamic_shared_memory()
    {
      if constexpr (maximum_dynamic_shared_memory_override > 0) {
        return maximum_dynamic_shared_memory_override;
      } else {
#if (__COMPUTE_CAPABILITY__ < 700)
        return 48 * 1024;
#elif (__COMPUTE_CAPABILITY__ < 750) // 700, 720
        return 96 * 1024;
#elif (__COMPUTE_CAPABILITY__ == 750)
        return 64 * 1024;
#elif (__COMPUTE_CAPABILITY__ == 800)
        return 164 * 1024;
#elif (__COMPUTE_CAPABILITY__ == 860)
        return 100 * 1024;
#elif (__COMPUTE_CAPABILITY__ == 870)
        return 164 * 1024;
#elif (__COMPUTE_CAPABILITY__ == 890)
        return 100 * 1024;
#elif (__COMPUTE_CAPABILITY__ == 900)
        return 228 * 1024;
#elif (__COMPUTE_CAPABILITY__ == 1000)
        return 228 * 1024;
#else
        return 0;
#endif
      }
    }

    /**
      @brief A constexpr function to return the maximum number of resident threads per SM.
        See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#feature-availability
     */
    constexpr unsigned int maximum_resident_threads()
    {
#if (__COMPUTE_CAPABILITY__ < 750)
      return 2048;
#elif (__COMPUTE_CAPABILITY__ == 750)
      return 1024;
#elif (__COMPUTE_CAPABILITY__ == 800)
      return 2048;
#elif ((__COMPUTE_CAPABILITY__ > 800) && (__COMPUTE_CAPABILITY__ < 900))
      return 1536;
#elif (__COMPUTE_CAPABILITY__ == 900)
      return 2048;
#elif (__COMPUTE_CAPABILITY__ == 1000)
      return 2048;
#else
      return 0;
#endif
    }

  } // namespace device
} // namespace quda
