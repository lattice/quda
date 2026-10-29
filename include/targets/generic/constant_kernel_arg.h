#include <target_device.h>

/**
   @file constant_kernel_arg.h

   This file should be included in the kernel files for which we wish
   to utilize __constant__ memory for the kernel parameter struct.
   This needs to be included before the definition of the kernel,
   e.g., kernel.h in order for the compiler to do the kernel
   instantiation correctly.
 */

/* nothing needed here */
