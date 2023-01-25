namespace quda {

  template <int warp_split, typename T> inline T warp_combine(T &x)
  {
    if (warp_split > 1) {
      constexpr int warp_size = device::warp_size();
      auto sg = sycl::ext::oneapi::experimental::this_sub_group();
      //const int sg_size = sg.get_local_range().size();
#if 0
      // reduce down to the first group of column-split threads
#pragma unroll
      for (int offset = warp_size / 2; offset >= warp_size / warp_split; offset /= 2) {
	auto y = sycl::shift_group_left(sg, x, offset);
#pragma unroll
	for (int i = 0; i < x.size(); i++) {
	  x[i] += y[i];
	}
      }
#else
#pragma unroll
      for (int i = 0; i < x.size(); i++) {
        // reduce down to the first group of column-split threads
#pragma unroll
        for (int offset = warp_size / 2; offset >= warp_size / warp_split; offset /= 2) {
	  x[i].real(x[i].real() + sycl::shift_group_left(sg, x[i].real(), offset));
	  x[i].imag(x[i].imag() + sycl::shift_group_left(sg, x[i].imag(), offset));
	  //x[i] += sycl::shift_group_left(sg, x[i], offset);
	  //x[i].real(x[i].real() + sycl::permute_group_by_xor(sg, x[i].real(), offset));
	  //x[i].imag(x[i].imag() + sycl::permute_group_by_xor(sg, x[i].imag(), offset));
        }
      }
#endif
    }
    return x;
  }

}
