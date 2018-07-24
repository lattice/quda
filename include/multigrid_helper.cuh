#pragma once

namespace quda {

  /**
     Helper struct for dealing with spin coarsening.  This helper
     should work with all types of fermions.
   */
  template <int fineSpin, int coarseSpin>
  struct spin_mapper {
    static constexpr int spin_block_size = fineSpin / coarseSpin;

    /**
       Return the coarse spin coordinate from the fine spin coordinate
       @param s Fine spin coordinate
       @return Coarse spin coordinate
     */
    __device__ __host__ inline int operator()( int s ) const
    { return s / (spin_block_size > 0 ? spin_block_size : 1); }
  };



}
