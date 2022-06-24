#pragma once

#include <util_quda.h>
#include <quda_api.h>
#include <target_device.h>

/**
   @file constant_kernel_arg.h

   This file should be included in the kernel files for which we wish
   to utilize __constant__ memory for the kernel parameter struct.
   This needs to be included before the definition of the kernel,
   e.g., kernel.h in order for the compiler to do the kernel
   instantiation correctly.
 */

namespace quda
{

  namespace device
  {

    /**
       @brief The buffer used for kernel parameters
    */
    extern char buffer[max_constant_size()];

    /**
       @brief Helper function that returns kernel argument from buffer.
     */
    template <typename Arg> constexpr std::enable_if_t<!use_kernel_arg<Arg>(), Arg &> get_arg()
    {
      return reinterpret_cast<Arg &>(buffer);
    }

    /**
       @brief Helper function that returns a pointer to the buffer.
     */
    template <typename Arg> constexpr std::enable_if_t<!use_kernel_arg<Arg>(), void *> get_constant_buffer()
    {
      return omp_get_mapped_ptr(buffer, omp_get_default_device());
    }

  } // namespace device

} // namespace quda
