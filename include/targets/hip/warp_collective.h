namespace quda {

  template <int warp_split, typename T> __device__ __host__ inline T warp_combine(T &x)
  {
#ifdef __HIP_DEVICE_COMPILE__
    constexpr int warp_size = device::warp_size();
    if (warp_split > 1) {
#pragma unroll
      for (int i = 0; i < x.size(); i++) {
        // reduce down to the first group of column-split threads
#pragma unroll
        for (int offset = warp_size / 2; offset >= warp_size / warp_split; offset /= 2) {
          // TODO - add support for non-converged warps
          x[i].real(x[i].real() + __shfl_down( x[i].real(), offset));
          x[i].imag(x[i].imag() + __shfl_down( x[i].imag(), offset));
        }
      }
    }
#endif
    return x;
  }

}
