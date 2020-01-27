#ifdef QUDA_REDUCE_SINGLE_WARP
// only compile block size with a single warp
#define LAUNCH_KERNEL(kernel, tunable, tp, stream, arg, ...)            \
  switch (tp.block.x) {							\
  case 32:								\
    kernel<32,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
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
  case 544:								\
  case 576:								\
  case 608:								\
  case 640:								\
  case 672:								\
  case 704:								\
  case 736:								\
  case 768:								\
  case 800:								\
  case 832:								\
  case 864:								\
  case 896:								\
  case 928:								\
  case 960:								\
  case 992:								\
  case 1024:								\
    tunable.jitifyError() = CUDA_ERROR_INVALID_VALUE;                   \
    break;                                                              \
default:								\
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
  }

#else

#define LAUNCH_KERNEL(kernel, tunable, tp, stream, arg, ...)            \
  switch (tp.block.x) {							\
  case 32:								\
    kernel<32,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 64:								\
    kernel<64,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 96:								\
    kernel<96,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 128:								\
    kernel<128,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 160:								\
    kernel<160,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 192:								\
    kernel<192,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 224:								\
    kernel<224,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 256:								\
    kernel<256,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 288:								\
    kernel<288,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 320:								\
    kernel<320,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 352:								\
    kernel<352,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 384:								\
    kernel<384,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 416:								\
    kernel<416,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 448:								\
    kernel<448,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 480:								\
    kernel<480,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 512:								\
    kernel<512,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 544:								\
    kernel<544,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 576:								\
    kernel<576,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 608:								\
    kernel<608,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 640:								\
    kernel<640,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 672:								\
    kernel<672,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 704:								\
    kernel<704,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 736:								\
    kernel<736,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 768:								\
    kernel<768,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 800:								\
    kernel<800,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 832:								\
    kernel<832,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 864:								\
    kernel<864,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 896:								\
    kernel<896,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 928:								\
    kernel<928,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 960:								\
    kernel<960,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 992:								\
    kernel<992,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 1024:								\
    kernel<1024,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
      break;								\
  default:								\
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
    }

#endif // REDUCE_SINGLE_WARP

#ifdef QUDA_REDUCE_SINGLE_WARP

// only compile block size with a single warp
#define LAUNCH_KERNEL_LOCAL_PARITY(kernel, tunable, tp, stream, arg, ...) \
  switch (tp.block.x) {							\
  case 32:								\
    kernel<32,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
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
    tunable.jitifyError() = CUDA_ERROR_INVALID_VALUE;                   \
    break;                                                              \
  default:								\
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
    }

#else

#define LAUNCH_KERNEL_LOCAL_PARITY(kernel, tunable, tp, stream, arg, ...) \
  switch (tp.block.x) {							\
  case 32:								\
    kernel<32,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 64:								\
    kernel<64,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 96:								\
    kernel<96,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 128:								\
    kernel<128,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 160:								\
    kernel<160,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 192:								\
    kernel<192,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 224:								\
    kernel<224,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 256:								\
    kernel<256,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 288:								\
    kernel<288,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 320:								\
    kernel<320,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 352:								\
    kernel<352,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 384:								\
    kernel<384,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 416:								\
    kernel<416,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 448:								\
    kernel<448,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 480:								\
    kernel<480,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  case 512:								\
    kernel<512,__VA_ARGS__>						\
      <<< tp.grid, tp.block, tp.shared_bytes, stream >>>(arg);		\
    break;								\
  default:								\
    errorQuda("%s not implemented for %d threads", #kernel, tp.block.x); \
    }

#endif

#define LAUNCH_KERNEL_MG_BLOCK_SIZE(kernel, tp, stream, arg, ...)                                                      \
  switch (tp.block.x) {                                                                                                \
  case 4: kernel<4, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                          \
  case 8: kernel<8, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                          \
  case 12: kernel<12, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 16: kernel<16, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 27: kernel<27, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 32: kernel<32, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 36: kernel<36, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 54: kernel<54, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 64: kernel<64, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 72: kernel<72, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 81: kernel<81, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 96: kernel<96, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 100: kernel<100, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  case 108: kernel<108, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  case 128: kernel<128, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  case 144: kernel<144, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  case 192: kernel<192, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  case 200: kernel<200, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  case 256: kernel<256, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  case 288: kernel<288, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  case 432: kernel<432, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  case 500: kernel<500, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  case 512: kernel<512, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  default: errorQuda("%s block size %d not instantiated", #kernel, tp.block.x);                                        \
  }

#ifdef QUDA_REDUCE_SINGLE_WARP

 // only compile block size with a single warp
#define LAUNCH_KERNEL_REDUCE(kernel, tunable, tp, stream, arg, ...)                                                    \
  switch (tp.block.x) {                                                                                                \
  case 32: kernel<32, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 64:                                                                                                             \
  case 96:                                                                                                             \
  case 128:                                                                                                            \
    tunable.jitifyError() = CUDA_ERROR_INVALID_VALUE;                                                                  \
    break;                                                                                                             \
  default: errorQuda("%s block size %d not instantiated", #kernel, tp.block.x);                                        \
  }

#else

#define LAUNCH_KERNEL_REDUCE(kernel, tunable, tp, stream, arg, ...)                                                    \
  switch (tp.block.x) {                                                                                                \
  case 32: kernel<32, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 64: kernel<64, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 96: kernel<96, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                        \
  case 128: kernel<128, __VA_ARGS__><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg); break;                      \
  default: errorQuda("%s block size %d not instantiated", #kernel, tp.block.x);                                        \
  }

#endif
