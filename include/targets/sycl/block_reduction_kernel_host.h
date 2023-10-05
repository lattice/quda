namespace quda
{

  template <template <typename> class Functor, typename Arg> void BlockKernel2D_host(const Arg &arg)
  {
    if constexpr (needsSharedMem<typename Functor<Arg>::SpecialOpsT>) {
      constexpr auto smemsize = sharedMemSize<typename Functor<Arg>::SpecialOpsT>(dim3(1,1,1));
      char smem[smemsize];
      Functor<Arg> t{arg, &smem[0]};
      dim3 block(0, 0, 0);
      for (block.y = 0; block.y < arg.grid_dim.y; block.y++) {
	for (block.x = 0; block.x < arg.grid_dim.x; block.x++) { t(block, dim3(0, 0, 0)); }
      }
    } else {
      Functor<Arg> t(arg);
      dim3 block(0, 0, 0);
      for (block.y = 0; block.y < arg.grid_dim.y; block.y++) {
	for (block.x = 0; block.x < arg.grid_dim.x; block.x++) { t(block, dim3(0, 0, 0)); }
      }
    }
  }

} // namespace quda
