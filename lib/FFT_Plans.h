#include "quda_define.h"

#if defined(QUDA_TARGET_CUDA)
#include "targets/cuda/CUFFT_Plans.h"
#elif defined(QUDA_TARGET_HIP)
#include "targets/hip/HIPFFT_Plans.h"
#else
#error "Neither QUDA_TARGET_CUDA nor QUDA_TARGET_HIP are defined"
#endif
