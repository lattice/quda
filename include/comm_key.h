#pragma once

namespace quda
{

  struct CommKey {

    static constexpr int n_dim = 4;

    int array[n_dim] = {0, 0, 0, 0};

    constexpr inline int product() { return array[0] * array[1] * array[2] * array[3]; }

    constexpr inline int &operator[](int d) { return array[d]; }

    constexpr inline const int &operator[](int d) const { return array[d]; }

    constexpr inline int *data() { return array; }

    constexpr inline const int *data() const { return array; }

    constexpr inline bool is_valid() const
    {
      return (array[0] > 0) && (array[1] > 0) && (array[2] > 0) && (array[3] > 0);
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
