#include "quda_define.h"

#if defined(QUDA_BUILD_TARGET_CUDA)
#include "targets/cuda/CUFFT_Plans.h"
#elif defined(QUDA_BUILD_TARGET_HIP)
#include "targets/hip/HIPFFT_Plans.h"
#endif
