#include <quda_api.h>

#ifdef QUDA_FAST_COMPILE_REDUCE

// only compile block size with a single warp
#define LAUNCH_KERNEL_LOCAL_PARITY(kernel, tunable, tp, stream, arg, ...) \
  switch (tp.block.x) {							\
  case 32: arg.launch_error = qudaLaunchKernel(kernel<32,__VA_ARGS__>, tp, stream, arg); break; \
  case 64:								\
  case 96:								\
  case 128:								\
  case 160:								\
  case 192:								\
  case 224:								\
  case 256:								\
  case 288:								\
  case 320:								\
  case 352:								\
  case 384:								\
  case 416:								\
  case 448:								\
  case 480:								\
  case 512:								\
    arg.launch_error = QUDA_ERROR;                                      \
    tunable.jitifyError() = CUDA_ERROR_INVALID_VALUE;                   \
    break;                                                              \
  default:								\
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
  }

#else

#define LAUNCH_KERNEL_LOCAL_PARITY(kernel, tunable, tp, stream, arg, ...) \
  switch (tp.block.x) {							\
  case 32: arg.launch_error = qudaLaunchKernel(kernel<32,__VA_ARGS__>, tp, stream, arg); break; \
  case 64: arg.launch_error = qudaLaunchKernel(kernel<64,__VA_ARGS__>, tp, stream, arg); break; \
  case 96: arg.launch_error = qudaLaunchKernel(kernel<96,__VA_ARGS__>, tp, stream, arg); break; \
  case 128: arg.launch_error = qudaLaunchKernel(kernel<128,__VA_ARGS__>, tp, stream, arg); break; \
  case 160: arg.launch_error = qudaLaunchKernel(kernel<160,__VA_ARGS__>, tp, stream, arg); break; \
  case 192: arg.launch_error = qudaLaunchKernel(kernel<192,__VA_ARGS__>, tp, stream, arg); break; \
  case 224: arg.launch_error = qudaLaunchKernel(kernel<224,__VA_ARGS__>, tp, stream, arg); break; \
  case 256: arg.launch_error = qudaLaunchKernel(kernel<256,__VA_ARGS__>, tp, stream, arg); break; \
  case 288: arg.launch_error = qudaLaunchKernel(kernel<288,__VA_ARGS__>, tp, stream, arg); break; \
  case 320: arg.launch_error = qudaLaunchKernel(kernel<320,__VA_ARGS__>, tp, stream, arg); break; \
  case 352: arg.launch_error = qudaLaunchKernel(kernel<352,__VA_ARGS__>, tp, stream, arg); break; \
  case 384: arg.launch_error = qudaLaunchKernel(kernel<384,__VA_ARGS__>, tp, stream, arg); break; \
  case 416: arg.launch_error = qudaLaunchKernel(kernel<416,__VA_ARGS__>, tp, stream, arg); break; \
  case 448: arg.launch_error = qudaLaunchKernel(kernel<448,__VA_ARGS__>, tp, stream, arg); break; \
  case 480: arg.launch_error = qudaLaunchKernel(kernel<480,__VA_ARGS__>, tp, stream, arg); break; \
  case 512: arg.launch_error = qudaLaunchKernel(kernel<512,__VA_ARGS__>, tp, stream, arg); break; \
  default:								\
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
  }

#endif

#define LAUNCH_KERNEL_MG_BLOCK_SIZE(kernel, tp, stream, arg, ...)       \
  switch (tp.block.x) {                                                 \
  case 4: qudaLaunchKernel(kernel<4, __VA_ARGS__>, tp, stream, arg); break; \
  case 8: qudaLaunchKernel(kernel<8, __VA_ARGS__>, tp, stream, arg); break; \
  case 9: qudaLaunchKernel(kernel<9, __VA_ARGS__>, tp, stream, arg); break; \
  case 12: qudaLaunchKernel(kernel<12, __VA_ARGS__>, tp, stream, arg); break; \
  case 16: qudaLaunchKernel(kernel<16, __VA_ARGS__>, tp, stream, arg); break; \
  case 18: qudaLaunchKernel(kernel<18, __VA_ARGS__>, tp, stream, arg); break; \
  case 24: qudaLaunchKernel(kernel<24, __VA_ARGS__>, tp, stream, arg); break; \
  case 27: qudaLaunchKernel(kernel<27, __VA_ARGS__>, tp, stream, arg); break; \
  case 32: qudaLaunchKernel(kernel<32, __VA_ARGS__>, tp, stream, arg); break; \
  case 36: qudaLaunchKernel(kernel<36, __VA_ARGS__>, tp, stream, arg); break; \
  case 48: qudaLaunchKernel(kernel<48, __VA_ARGS__>, tp, stream, arg); break; \
  case 54: qudaLaunchKernel(kernel<54, __VA_ARGS__>, tp, stream, arg); break; \
  case 64: qudaLaunchKernel(kernel<64, __VA_ARGS__>, tp, stream, arg); break; \
  case 72: qudaLaunchKernel(kernel<72, __VA_ARGS__>, tp, stream, arg); break; \
  case 81: qudaLaunchKernel(kernel<81, __VA_ARGS__>, tp, stream, arg); break; \
  case 96: qudaLaunchKernel(kernel<96, __VA_ARGS__>, tp, stream, arg); break; \
  case 100: qudaLaunchKernel(kernel<100, __VA_ARGS__>, tp, stream, arg); break; \
  case 108: qudaLaunchKernel(kernel<108, __VA_ARGS__>, tp, stream, arg); break; \
  case 128: qudaLaunchKernel(kernel<128, __VA_ARGS__>, tp, stream, arg); break; \
  case 144: qudaLaunchKernel(kernel<144, __VA_ARGS__>, tp, stream, arg); break; \
  case 192: qudaLaunchKernel(kernel<192, __VA_ARGS__>, tp, stream, arg); break; \
  case 200: qudaLaunchKernel(kernel<200, __VA_ARGS__>, tp, stream, arg); break; \
  case 250: qudaLaunchKernel(kernel<250, __VA_ARGS__>, tp, stream, arg); break; \
  case 256: qudaLaunchKernel(kernel<256, __VA_ARGS__>, tp, stream, arg); break; \
  case 288: qudaLaunchKernel(kernel<288, __VA_ARGS__>, tp, stream, arg); break; \
  case 432: qudaLaunchKernel(kernel<432, __VA_ARGS__>, tp, stream, arg); break; \
  case 500: qudaLaunchKernel(kernel<500, __VA_ARGS__>, tp, stream, arg); break; \
  case 512: qudaLaunchKernel(kernel<512, __VA_ARGS__>, tp, stream, arg); break; \
  default: errorQuda("%s block size %d not instantiated", #kernel, tp.block.x); \
  }
