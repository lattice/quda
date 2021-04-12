/**
   @file dslash_constant_buffer.cu

   @brief When NVSHMEM is enabled, the dslash kernels use RDC which
   means we have to be careful with the __constant__ buffer
   declarations in dslash kernels that include constant_kernel_arg.h,
   since they will collide.  To avoid this, all files that include
   constant_kernel_arg will mark the buffer as extern, with the
   definition being this file.
*/

#define QUDA_CONSTANT_DEFINE
#include <constant_kernel_arg.h>
