#pragma once

namespace quda {

  /**
     A.S.: we don't have spin dofr for top-level staeggered field, so, strictly speaking, the spin mapper is undefined
  */


  /**
     Helper struct for dealing with spin coarsening.  This helper
     should work with all types of fermions.
   */
  template <int fineSpin, int coarseSpin>
  struct spin_mapper {
    static constexpr int spin_block_size = (fineSpin != 1) ? fineSpin / coarseSpin : 1;

    /**
       Return the coarse spin coordinate from the fine spin coordinate (trivial opertion for the top level staggered field)
       @param s Fine spin coordinate
       @return Coarse spin coordinate
     */
    __device__ __host__ constexpr inline int operator()( int s ) const
    { return s / (spin_block_size > 0 ? spin_block_size : 1); }
  };



}
