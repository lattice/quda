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
    constexpr int operator()(int s, int parity) const
    {
      return (spin_block_size == 0) ? parity : s / spin_block_size;
    }
  };

}
