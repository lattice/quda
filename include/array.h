#pragma once

namespace quda {

  template <typename T> constexpr T zero() { return static_cast<T>(0); }
  template <> constexpr double2 zero() { return double2{0.0, 0.0}; }
  template <> constexpr double3 zero() { return double3{0.0, 0.0, 0.0}; }
  template <> constexpr double4 zero() { return double4{0.0, 0.0, 0.0, 0.0}; }

  template <> constexpr float2 zero() { return float2{0.0f, 0.0f}; }
  template <> constexpr float3 zero() { return float3{0.0f, 0.0f, 0.0f}; }
  template <> constexpr float4 zero() { return float4{0.0f, 0.0f, 0.0f, 0.0f}; }

#ifdef QUAD_SUM
  template <> __device__ __host__ inline doubledouble zero() { return doubledouble(); }
  template <> __device__ __host__ inline doubledouble2 zero() { return doubledouble2(); }
  template <> __device__ __host__ inline doubledouble3 zero() { return doubledouble3(); }
#endif

  /**
     Simple array object which mimics std::array
   */
  template <typename T, int n> struct array {
    using value_type = T;
    T data[n];

    constexpr T &operator[](int i) { return data[i]; }
    constexpr const T &operator[](int i) const { return data[i]; }
    constexpr int size() const { return n; }

    array(const array<T, n> &) = default;
    array(array<T, n> &&) = default;

    __device__ __host__ constexpr array() { for (int i = 0; i < n; i++) data[i] = zero<T>(); }
    template <typename... U>
    __device__ __host__ constexpr array(T first, const U... data) : data {first, data...} { }
    __device__ __host__ constexpr array(T a) { for (auto &e : data) e = a; }

    array<T, n> &operator=(const array<T, n> &) = default;
    array<T, n> &operator=(array<T, n> &&) = default;
  };

  template <typename T, int n> std::ostream &operator<<(std::ostream &output, const array<T, n> &a)
  {
    output << "{ ";
    for (int i = 0; i < n - 1; i++) output << a[i] << ", ";
    output << a[n - 1] << " }";
    return output;
  }

}
