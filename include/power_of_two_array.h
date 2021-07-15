#pragma once

namespace quda {

	// Utility functions go into anonymous namespace
	namespace {
	  /** 
	   * @brief compute number of elements of an array containing
	   *   powers of 2 starting at a minumum up to and including a maximum
	   *
	   */
	   template<unsigned int Min, unsigned int Max>
	   constexpr unsigned int numElements() noexcept {
             unsigned int i = 0;
             for(auto j=Min; j <= Max; j *=2 ) i++;
             return i;
           }
	}

	/**
	 * @brief A struct containing a compile time generated array
	 * containing powers of 2 starting at Min up to and includeing Max
	 * with thanks to StackOverflow:
	 *   https://stackoverflow.com/questions/19019252/create-n-element-constexpr-array-in-c11
	 */
	template<unsigned int Min, unsigned int Max>
        struct PowerOfTwoArray {

#if __cplusplus >= 201703L
	  std::array<unsigned int, numElements<Min,Max>()> data_;
#elif __cplusplus >= 201402L
	  unsigned int data_[ numElements<Min,Max>() ];
#else
#error  "Power of 2 Array needs C++ 14 or newer"
#endif
	
	  // The magic constexpr constructor 
          constexpr PowerOfTwoArray() : data_() {
            static_assert( Min <= Max, "Min has to be <= Max");
            for(unsigned int i=0, j=Min; j <= Max; j*=2, i++) data_[i] = j;
          }

	  /**
	   * @brief returns the size of the array
	   */
          constexpr unsigned int size() const noexcept {
            return numElements<Min,Max>();
          }

	  /** 
	   * @brief read only constant index operator[]
	   * @param i the index to look up
	   */
          constexpr unsigned int operator[](int i) const noexcept {
            return data_[i];
          }
       }; // end struct

#if 0
       template<typename Tag, unsigned int Min, unsigned int Max> 
       struct TaggedBlockMapper {
         // List of block sizes we wish to instantiate.  The required block
         // size is equal to number of fine points per aggregate, rounded
         // up to a whole power of two.  So for example, 2x2x2x2 and
         // 3x3x3x1 aggregation would both use the same block size 32
#ifndef QUDA_FAST_COMPILE_REDUCE
         using array_type = PowerOfTwoArray<Min, Max>;
#else
         using array_type = PowerOfTwoArray<Max, Max>;
#endif

         static constexpr array_type block=array_type(); 
         /**
          *  @brief Return the first power of two block that is larger than the required size
          */
         static unsigned int block_mapper(unsigned int raw_block)
         {
           for (unsigned int b=0; b < block.size();  b++) if (raw_block <= block[b]) return block[b];
           errorQuda("Invalid raw block size %d\n", raw_block);
           return 0;
         }
      };
#endif
      
} // end namespace cuda 
