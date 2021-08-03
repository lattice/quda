#pragma once

#include "complex_quda.h"

namespace quda {

  template <typename T> struct plus {
    static constexpr bool do_sum = true;
    using reducer_t = plus<T>;
    __device__ __host__ T operator()(T a, T b) const { return a + b; }
    __device__ __host__ static T reduce(T a, T b) { return a + b; }
    __device__ __host__ static T init() { return quda::zero<T>(); }
  };

  template <typename T> struct maximum {
    static constexpr bool do_sum = false;
    using reducer_t = maximum<T>;
    __device__ __host__ T operator()(T a, T b) const { return a > b ? a : b; }
  };

  template <typename T> struct minimum {
    static constexpr bool do_sum = false;
    using reducer_t = minimum<T>;
    __device__ __host__ T operator()(T a, T b) const { return a < b ? a : b; }
  };

  template <typename T> struct identity {
    static constexpr bool do_sum = false;
    __device__ __host__ T operator()(T a) const { return a; }
  };


  template<typename ReduceType, typename Float> struct square_ {
    square_(ReduceType = 1.0) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<Float> &x) const
    { return static_cast<ReduceType>(norm(x)); }
  };

  template <typename ReduceType> struct square_<ReduceType, int8_t> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int8_t> &x) const
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename ReduceType> struct square_<ReduceType,short> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<short> &x) const
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename ReduceType> struct square_<ReduceType,int> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int> &x) const
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename Float, typename storeFloat> struct abs_ {
    abs_(const Float = 1.0) { }
    __host__ __device__ Float operator()(const quda::complex<storeFloat> &x) const
    { return abs(x); }
  };

  template <typename Float> struct abs_<Float, int8_t> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int8_t> &x) const
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };

  template<typename Float> struct abs_<Float,short> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<short> &x) const
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };

  template<typename Float> struct abs_<Float,int> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int> &x) const
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };

  template<typename Float, typename storeFloat> struct abs_max_ {
    abs_max_(const Float = 1.0) { }
    __host__ __device__ Float operator()(const quda::complex<storeFloat> &x) const
    { return maximum<Float>()(abs(x.real()), abs(x.imag())); }
  };

  template <typename Float> struct abs_max_<Float, int8_t> {
    Float scale;
    abs_max_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int8_t> &x) const
    { return maximum<Float>()(abs(scale * x.real()), abs(scale * x.imag())); }
  };

  template<typename Float> struct abs_max_<Float,short> {
    Float scale;
    abs_max_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<short> &x) const
    { return maximum<Float>()(abs(scale * x.real()), abs(scale * x.imag())); }
  };

  template<typename Float> struct abs_max_<Float,int> {
    Float scale;
    abs_max_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int> &x) const
    { return maximum<Float>()(abs(scale * x.real()), abs(scale * x.imag())); }
  };

  template<typename Float, typename storeFloat> struct abs_min_ {
    abs_min_(const Float = 1.0) { }
    __host__ __device__ Float operator()(const quda::complex<storeFloat> &x) const
    { return minimum<Float>()(abs(x.real()), abs(x.imag())); }
  };

  template <typename Float> struct abs_min_<Float, int8_t> {
    Float scale;
    abs_min_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int8_t> &x) const
    { return minimum<Float>()(abs(scale * x.real()), abs(scale * x.imag())); }
  };

  template<typename Float> struct abs_min_<Float,short> {
    Float scale;
    abs_min_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<short> &x) const
    { return minimum<Float>()(abs(scale * x.real()), abs(scale * x.imag())); }
  };

  template<typename Float> struct abs_min_<Float,int> {
    Float scale;
    abs_min_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int> &x) const
    { return minimum<Float>()(abs(scale * x.real()), abs(scale * x.imag())); }
  };

}
