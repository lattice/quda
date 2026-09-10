namespace quda
{

  template <template <typename> class Functor, typename Arg> void BlockKernel2D_host(const Arg &arg)
  {
    if constexpr (needsSharedMem<typename Functor<Arg>::KernelOpsT>) {
      constexpr auto smemsize = sharedMemSize<typename Functor<Arg>::KernelOpsT>(dim3(1, 1, 1));
      char smem[smemsize];
      Functor<Arg> t {arg, &smem[0]};
#pragma omp parallel for
      for (unsigned int y = 0; y < arg.grid_dim.y; y++) {
	for (unsigned int x = 0; x < arg.grid_dim.x; x++) {
	  t(dim3(x, y, 0), dim3(0, 0, 0));
	}
      }
    } else {
      Functor<Arg> t(arg);
#pragma omp parallel for
      for (unsigned int y = 0; y < arg.grid_dim.y; y++) {
	for (unsigned int x = 0; x < arg.grid_dim.x; x++) {
	  t(dim3(x, y, 0), dim3(0, 0, 0));
	}
      }
    }
  }

} // namespace quda
