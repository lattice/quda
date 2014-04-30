#define LAUNCH_KERNEL(kernel, tp, stream, arg, ...)			\
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
