namespace quda
{

  template <template <typename> class Functor, typename Arg> void BlockKernel2D_host(const Arg &arg)
  {
    Functor<Arg> t(arg);
    dim3 block(0, 0, 0);
    for (block.y = 0; block.y < arg.grid_dim.y; block.y++) {
      for (block.x = 0; block.x < arg.grid_dim.x; block.x++) { t(block, dim3(0, 0, 0)); }
    }
  }

} // namespace quda
