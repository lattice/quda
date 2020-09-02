#pragma once

namespace quda {

  /**
     Helper struct for dealing with spin coarsening.  This helper
     should work with all types of fermions.
   */
  template <int fineSpin, int coarseSpin>
  struct spin_mapper {
    // fineSpin == 1, coarseSpin == 2 identifies staggered fine -> coarse w/ spin.
    static constexpr int spin_block_size = (fineSpin == 1 && coarseSpin == 2) ? 0 : fineSpin / coarseSpin;

    /**
       Return the coarse spin coordinate from the fine spin coordinate
       @param s Fine spin coordinate
       @param parity fine parity, for staggered
       @return Coarse spin coordinate
     */
    __device__ __host__ constexpr inline int operator()( int s, int parity ) const
    { return (spin_block_size == 0) ? parity : s / spin_block_size; }
  };

}

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
