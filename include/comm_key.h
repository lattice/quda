#pragma once

#include <array.h>

namespace quda
{

  struct CommKey {

    static constexpr int n_dim = 4;
    array<int, n_dim> key = {0, 0, 0, 0};

    constexpr int product() { return key[0] * key[1] * key[2] * key[3]; }

    constexpr int &operator[](int d) { return key[d]; }

    constexpr const int &operator[](int d) const { return key[d]; }

    constexpr auto data() { return key.data; }

    constexpr auto data() const { return key.data; }

    constexpr bool is_valid() const { return (key[0] > 0) && (key[1] > 0) && (key[2] > 0) && (key[3] > 0); }

    bool operator==(const CommKey &other) const
    {
      bool is_same = true;
      for (auto i = 0; i < n_dim; i++)
        if (key[i] != other.key[i]) is_same = false;
      return is_same;
    }
  };

  constexpr int inline product(const CommKey &input) { return input[0] * input[1] * input[2] * input[3]; }

  constexpr CommKey inline operator+(const CommKey &lhs, const CommKey &rhs)
  {
    CommKey sum;
    for (int d = 0; d < CommKey::n_dim; d++) { sum[d] = lhs[d] + rhs[d]; }
    return sum;
  }

  constexpr CommKey inline operator*(const CommKey &lhs, const CommKey &rhs)
  {
    CommKey product;
    for (int d = 0; d < CommKey::n_dim; d++) { product[d] = lhs[d] * rhs[d]; }
    return product;
  }

  constexpr CommKey inline operator/(const CommKey &lhs, const CommKey &rhs)
  {
    CommKey quotient;
    for (int d = 0; d < CommKey::n_dim; d++) { quotient[d] = lhs[d] / rhs[d]; }
    return quotient;
  }

  constexpr CommKey inline operator%(const CommKey &lhs, const CommKey &rhs)
  {
    CommKey mod;
    for (int d = 0; d < CommKey::n_dim; d++) { mod[d] = lhs[d] % rhs[d]; }
    return mod;
  }

  constexpr bool inline operator<(const CommKey &lhs, const CommKey &rhs)
  {
    for (int d = 0; d < CommKey::n_dim; d++) {
      if (lhs[d] < rhs[d]) { return true; }
    }
    return false;
  }

  constexpr bool inline operator>(const CommKey &lhs, const CommKey &rhs)
  {
    for (int d = 0; d < CommKey::n_dim; d++) {
      if (lhs[d] > rhs[d]) { return true; }
    }
    return false;
  }

  constexpr CommKey inline coordinate_from_index(int index, CommKey dim)
  {
    CommKey coord;
    for (int d = 0; d < CommKey::n_dim; d++) {
      coord[d] = index % dim[d];
      index /= dim[d];
    }
    return coord;
  }

  constexpr int inline index_from_coordinate(CommKey coord, CommKey dim)
  {
    return ((coord[3] * dim[2] + coord[2]) * dim[1] + coord[1]) * dim[0] + coord[0];
  }

} // namespace quda
