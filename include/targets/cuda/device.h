namespace quda {

  /**
     @brief Helper function that returns the warp-size of the
     architecture we are running on.
   */
  constexpr int warp_size() { return 32; }

#ifdef QUDA_FAST_COMPILE_REDUCE
  /**
     @brief Helper function that returns the maximum number of threads
     in a block in the x dimension.  This is the specialized variant
     used when have fast-compilation mode enabled.
   */
  template <int block_size_y = 1, int block_size_z = 1>
  constexpr unsigned int max_block_size()
  {
    return warp_size();
  }
#else
  /**
     @brief Helper function that returns the maximum number of threads
     in a block in the x dimension.
   */
  template <int block_size_y = 1, int block_size_z = 1>
  constexpr unsigned int max_block_size()
  {
    return std::max(warp_size(), 1024 / (block_size_y * block_size_z));
  }
#endif

}
