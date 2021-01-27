#pragma once
namespace quda {


  template <typename T> struct plus {
    static constexpr bool do_sum = true;
    __device__ __host__ T operator()(T a, T b) const { return a + b; }
  };

#if 1
  // Dumb tests suggest this is still needed for some reason
  template<>
  struct plus<double2> {
    static constexpr bool do_sum = true;
    using Double2 = double2;

    __device__ __host__ inline Double2 operator()( Double2 a, Double2 b ) {
	    Double2 ret_val{0,0};
	    ret_val.x = a.x + b.x;
	    ret_val.y = a.y + b.y;
	    return ret_val;
    }
  };
#endif

  template <typename T> struct maximum {
    static constexpr bool do_sum = false;
    __device__ __host__ T operator()(T a, T b) const { return a > b ? a : b; }
  };

  template <typename T> struct minimum {
    static constexpr bool do_sum = false;
    __device__ __host__ T operator()(T a, T b) const { return a < b ? a : b; }
  };

  template <typename T> struct identity {
    static constexpr bool do_sum = false;
    __device__ __host__ T operator()(T a) const { return a; }
  };


  template<typename ReduceType, typename Float> struct square_ {
    square_(ReduceType = 1.0) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<Float> &x)
    { return static_cast<ReduceType>(norm(x)); }
  };

  template <typename ReduceType> struct square_<ReduceType, int8_t> {
    const ReduceType scale;
  square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int8_t> &x)
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename ReduceType> struct square_<ReduceType,short> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<short> &x)
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename ReduceType> struct square_<ReduceType,int> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int> &x)
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename Float, typename storeFloat> struct abs_ {
    abs_(const Float = 1.0) { }
    __host__ __device__ Float operator()(const quda::complex<storeFloat> &x) { return abs(x); }
  };

  template <typename Float> struct abs_<Float, int8_t> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int8_t> &x)
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };

  template<typename Float> struct abs_<Float,short> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<short> &x)
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };

  template<typename Float> struct abs_<Float,int> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int> &x)
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };

}
