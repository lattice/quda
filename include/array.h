#pragma once

#include <iostream>
#include <quda_arch.h>

namespace quda
{

  /**
     Simple array object which mimics std::array
   */
  template <typename T, int n> struct array {
    using value_type = T;
    static constexpr int N = n;
    T data[n];

    constexpr T &operator[](int i) { return data[i]; }
    constexpr const T &operator[](int i) const { return data[i]; }
    constexpr int size() const { return n; }

    array() = default;
    array(const array<T, n> &) = default;
    array(array<T, n> &&) = default;

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

  /**
   * @brief Element-wise maximum of two arrays
   * @param a first array
   * @param b second array
   */
  template <typename T, int N> __host__ __device__ inline array<T, N> max(const array<T, N> &a, const array<T, N> &b)
  {
    array<T, N> result;
    for (int i = 0; i < N; i++) { result[i] = a[i] > b[i] ? a[i] : b[i]; }
    return result;
  }

  /**
   * @brief Element-wise minimum of two arrays
   * @param a first array
   * @param b second array
   */
  template <typename T, int N> __host__ __device__ inline array<T, N> min(const array<T, N> &a, const array<T, N> &b)
  {
    array<T, N> result;
    for (int i = 0; i < N; i++) { result[i] = a[i] < b[i] ? a[i] : b[i]; }
    return result;
  }

  template <typename T, int m, int n> using array_2d = array<array<T, n>, m>;
  template <typename T, int m, int n, int k> using array_3d = array<array<array<T, k>, n>, m>;

  struct assign_t {
    template <class T> __device__ __host__ inline void operator()(T *out, T in) { *out = in; }
  };

} // namespace quda
