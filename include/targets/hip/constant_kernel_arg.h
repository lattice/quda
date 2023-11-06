#pragma once

#include <target_device.h>

/**
   @file constant_kernel_arg.h

   This file should be included in the kernel files for which we wish
   to utilize __constant__ memory for the kernel parameter struct.
   This needs to be included before the definition of the kernel,
   e.g., kernel.h in order for the compiler to do the kernel
   instantiation correctly.
 */

// set a preprocessor flag that we have included constant_kernel_arg.h
#define QUDA_USE_CONSTANT_MEMORY

namespace quda
{

  namespace device
  {

    /**
       @brief The __constant__ buffer userd for kernel parameters
    */
    __constant__ char buffer[max_constant_size()];

    /**
       @brief Helper function that returns kernel argument from
       __constant__ memory.
     */
    template <typename Arg> constexpr std::enable_if_t<!use_kernel_arg<Arg>(), Arg &> get_arg()
    {
      return reinterpret_cast<Arg &>(buffer);
    }

    /**
       @brief Helper function that returns a pointer to the
       __constant__ memory buffer.
     */
    template <typename Arg> constexpr std::enable_if_t<!use_kernel_arg<Arg>(), void *> get_constant_buffer()
    {
      return qudaGetSymbolAddress(buffer);
    }

  } // namespace device

} // namespace quda
