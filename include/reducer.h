#pragma once

#include "complex_quda.h"
#include "quda_constants.h"
#include "quda_api.h"
#include <math_helper.cuh>
#include <float_vector.h>
#include "comm_quda.h"
#include "fast_intdiv.h"

/**
   @file reducer.h

   Implementations of various helper classes used for reductions,
   together with transformers used to aid reduction.  Generally
   speaking the reducers are binary operators, taking two arguments
   and reducing to a single value.  Transformers on the other hand are
   unary operators, though the return type may be different.  The
   transformers are able to work on fixed-point data, as we have
   specializations to handle the rescaling required in these cases.
*/

namespace quda
{

#ifdef QUAD_SUM
  using device_reduce_t = doubledouble;
#else
  using device_reduce_t = double;
#endif

  namespace reducer
  {
    /**
       @brief Inititalizes any persistent buffers required for performing global
       reductions.  If necessary, any previously allocated buffers will be resized.
       @param n_reduce The number of reductions to perform
       @param reduce_size Size in bytes of each value
    */
    void init(int n_reduce, size_t reduce_size);

    /**
       @brief Free any persistent buffers associated with global
       reductions.
    */
    void destroy();

    /**
       @return pointer to device reduction buffer
    */
    void *get_device_buffer();

    /**
       @return pointer to device-mapped host reduction buffer
    */
    void *get_mapped_buffer();

    /**
       @return pointer to host reduction buffer
    */
    void *get_host_buffer();

    /**
       @brief get_count returns the pointer to the counter array used
       for tracking the number of completed thread blocks.  We
       template this function, since the return type is target
       dependent.
       @return pointer to the reduction count array.
     */
    template <typename count_t> count_t *get_count();

    /**
       @return reference to the event used for synchronizing
       reductions with the host
     */
    qudaEvent_t &get_event();
  } // namespace reducer

  /**
     plus reducer, used for conventional sum reductions
   */
  template <typename T> struct plus {
    static constexpr bool do_sum = true;
    using reduce_t = T;
    using reducer_t = plus<T>;
    template <typename U> static inline void comm_reduce(std::vector<U> &a) { comm_allreduce_sum(a); }
    __device__ __host__ static inline T init() { return zero<T>(); }
    __device__ __host__ static inline T apply(T a, T b) { return a + b; }
    __device__ __host__ inline T operator()(T a, T b) const { return apply(a, b); }
  };

  /**
     maximum reducer, used for max reductions
   */
  template <typename T> struct maximum {
    static constexpr bool do_sum = false;
    using reduce_t = T;
    using reducer_t = maximum<T>;
    template <typename U> static inline void comm_reduce(std::vector<U> &a) { comm_allreduce_max(a); }
    __device__ __host__ static inline T init() { return low<T>::value(); }
    __device__ __host__ static inline T apply(T a, T b) { return max(a, b); }
    __device__ __host__ inline T operator()(T a, T b) const { return apply(a, b); }
  };

  /**
     minimum reducer, used for min reductions
   */
  template <typename T> struct minimum {
    static constexpr bool do_sum = false;
    using reduce_t = T;
    using reducer_t = minimum<T>;
    template <typename U> static inline void comm_reduce(std::vector<U> &a) { comm_allreduce_min(a); }
    __device__ __host__ static inline T init() { return high<T>::value(); }
    __device__ __host__ static inline T apply(T a, T b) { return min(a, b); }
    __device__ __host__ inline T operator()(T a, T b) const { return apply(a, b); }
  };

  /**
     square transformer, return the L2 norm squared of the input
   */
  template <typename ReduceType, typename Float> struct square_ {
    square_(ReduceType = 1.0) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<Float> &x) const
    {
      return static_cast<ReduceType>(norm(x));
    }
  };

  /**
     square transformer, return the L2 norm squared of the input
     (int8_t specialization)
   */
  template <typename ReduceType> struct square_<ReduceType, int8_t> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int8_t> &x) const
    {
      return norm(scale * complex<ReduceType>(x.real(), x.imag()));
    }
  };

  /**
     square transformer, return the L2 norm squared of the input
     (short specialization)
   */
  template <typename ReduceType> struct square_<ReduceType, short> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<short> &x) const
    {
      return norm(scale * complex<ReduceType>(x.real(), x.imag()));
    }
  };

  /**
     square transformer, return the L2 norm squared of the input
     (int specialization)
   */
  template <typename ReduceType> struct square_<ReduceType, int> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int> &x) const
    {
      return norm(scale * complex<ReduceType>(x.real(), x.imag()));
    }
  };

  /**
     abs transformer, return the absolute value of the input
   */
  template <typename Float, typename storeFloat> struct abs_ {
    abs_(const Float = 1.0) { }
    __host__ __device__ inline Float operator()(const quda::complex<storeFloat> &x) const { return abs(x); }
  };

  /**
     abs transformer, return the absolute value of the input (int8_t
     specialization)
   */
  template <typename Float> struct abs_<Float, int8_t> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int8_t> &x) const
    {
      return abs(scale * complex<Float>(x.real(), x.imag()));
    }
  };

  /**
     abs transformer, return the absolute value of the input (short
     specialization)
   */
  template <typename Float> struct abs_<Float, short> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<short> &x) const
    {
      return abs(scale * complex<Float>(x.real(), x.imag()));
    }
  };

  /**
     abs transformer, return the absolute value of the input (int
     specialization)
   */
  template <typename Float> struct abs_<Float, int> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int> &x) const
    {
      return abs(scale * complex<Float>(x.real(), x.imag()));
    }
  };

  /**
     abs_max transformer, return the maximum of the absolute value of the real
     and imaginary components
   */
  template <typename Float, typename storeFloat> struct abs_max_ {
    abs_max_(const Float = 1.0) { }
    __host__ __device__ inline Float operator()(const quda::complex<storeFloat> &x) const
    {
      return maximum<Float>::apply(abs(x.real()), abs(x.imag()));
    }
  };

  /**
     abs_max transformer, return the maximum of the absolute value of the real
     and imaginary components (int8_t specialziation)
   */
  template <typename Float> struct abs_max_<Float, int8_t> {
    Float scale;
    abs_max_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int8_t> &x) const
    {
      return maximum<Float>::apply(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

  /**
     abs_max transformer, return the maximum of the absolute value of the real
     and imaginary components (short specialziation)
   */
  template <typename Float> struct abs_max_<Float, short> {
    Float scale;
    abs_max_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<short> &x) const
    {
      return maximum<Float>::apply(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

  /**
     abs_max transformer, return the maximum of the absolute value of the real
     and imaginary components (int specialziation)
   */
  template <typename Float> struct abs_max_<Float, int> {
    Float scale;
    abs_max_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int> &x) const
    {
      return maximum<Float>::apply(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

  /**
     abs_min transformer, return the minimum of the absolute value of the real
     and imaginary components
   */
  template <typename Float, typename storeFloat> struct abs_min_ {
    abs_min_(const Float = 1.0) { }
    __host__ __device__ inline Float operator()(const quda::complex<storeFloat> &x) const
    {
      return minimum<Float>::apply(abs(x.real()), abs(x.imag()));
    }
  };

  /**
     abs_min transformer, return the minimum of the absolute value of the real
     and imaginary components (int8_t specialziation)
   */
  template <typename Float> struct abs_min_<Float, int8_t> {
    Float scale;
    abs_min_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int8_t> &x) const
    {
      return minimum<Float>::apply(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

  /**
     abs_min transformer, return the minimum of the absolute value of the real
     and imaginary components (short specialziation)
   */
  template <typename Float> struct abs_min_<Float, short> {
    Float scale;
    abs_min_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<short> &x) const
    {
      return minimum<Float>::apply(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

  /**
     abs_min transformer, return the minimum of the absolute value of the real
     and imaginary components (int specialziation)
   */
  template <typename Float> struct abs_min_<Float, int> {
    Float scale;
    abs_min_(const Float scale) : scale(scale) { }
    __host__ __device__ inline Float operator()(const quda::complex<int> &x) const
    {
      return minimum<Float>::apply(abs(scale * x.real()), abs(scale * x.imag()));
    }
  };

  /**
     identity transformer, preserves input
   */
  struct identity {
    identity() { }
    template <typename T> constexpr T operator()(T a) const { return a; }
  };

  /**
     milc gauge field mapper, allows us to pick out a given dimension subset
   */
  struct milc_mapper {
    int dim;
    int geometry;
    int_fastdiv width;
    milc_mapper(int dim, int geometry, int width) : dim(dim), geometry(geometry), width(width)
    {
      if (dim < 0 || dim >= geometry) errorQuda("invalid dimension %d", dim);
    }

    template <typename T> constexpr auto operator()(T i) const
    {
      auto inner = i % width;
      auto outer = i / width;
      return (outer * geometry + dim) * width + inner;
    }
  };

} // namespace quda
