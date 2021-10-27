#pragma once

namespace quda
{

  /**
     Simple array object which mimics std::array
   */
  template <typename T, int n> struct array {
    using value_type = T;
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

} // namespace quda
