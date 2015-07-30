#pragma once
#include <cub/cub.cuh>

/**
   @file cub_helper.cuh

   @section Description
 
   Provides helper functors for custom datatypes for cub algorithms.
 */

namespace quda {

  /**
     Helper functor for generic addition reduction.
  */
  template <typename T>
  struct Summ {
    __host__ __device__ __forceinline__ T operator() (const T &a, const T &b){
      return a + b;
    }
  };

  /**
     Helper functor for double2 addition reduction.
  */
  template <>
  struct Summ<double2>{
    __host__ __device__ __forceinline__ double2 operator() (const double2 &a, const double2 &b){
      return make_double2(a.x + b.x, a.y + b.y);
    }
  };

}
