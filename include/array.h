#pragma once

#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>
#include <utility>
#include <quda_arch.h>

namespace quda
{

  /**
     Simple array object which mimics std::array: value-initialized by default, list-like
     construction from 1..n convertible values (remaining entries are value-initialized),
     and assignment from std::initializer_list. Copy/move use the defaulted members.
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

    template <typename U> constexpr array<T, n> &operator=(const array<U, n> &other)
    {
      for (int i = 0; i < n; i++) data[i] = other[i];
      return *this;
    }

    /** Construction from std::initializer_list. */
    constexpr array(std::initializer_list<T> il) noexcept(true)
    {
      std::size_t k = 0;
      for (const T &e : il) {
        if (k >= static_cast<std::size_t>(n)) break;
        data[static_cast<int>(k++)] = e;
      }
      for (; k < static_cast<std::size_t>(n); ++k) data[static_cast<int>(k)] = T {};
    }

    /**
       Construct from one up to n values (extras are value-initialized). Disabled for a
       single argument that is exactly array<T,n> so copy/move construction is unchanged.
     */
    template <typename U0, typename... Urest>
    requires(n > 0 && (1 + sizeof...(Urest)) <= n && std::convertible_to<U0, T> && (std::convertible_to<Urest, T> && ...)
             && !(sizeof...(Urest) == 0
                  && std::is_same_v<std::remove_cvref_t<U0>, array<T, n>>)) constexpr array(U0 &&u0, Urest &&...ur) noexcept
    {
      for (int i = 0; i < n; ++i) data[i] = T {};
      int idx = 0;
      data[idx++] = static_cast<T>(std::forward<U0>(u0));
      ((void)(data[idx++] = static_cast<T>(std::forward<Urest>(ur))), ...);
    }

    /** Assignment from std::initializer_list; pads with T{}. */
    constexpr array<T, n> &operator=(std::initializer_list<T> il) noexcept(true)
    {
      std::size_t k = 0;
      for (const T &e : il) {
        if (k >= static_cast<std::size_t>(n)) break;
        data[static_cast<int>(k++)] = e;
      }
      for (; k < static_cast<std::size_t>(n); ++k) data[static_cast<int>(k)] = T {};
      return *this;
    }
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
