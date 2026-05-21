#pragma once

#include <cstdint>
#include <type_traits>

namespace quda
{

  /**
     @brief Packed small integer vector in one 32- or 64-bit register word.

     Valid (element type, length) pairs:
     - 8-bit:  N = 4 (32-bit word) or N = 8 (64-bit word)
     - 16-bit: N = 2 (32-bit word) or N = 4 (64-bit word)
  */
  template <class T, int N> class packed_array
  {
    static_assert(std::is_integral<T>::value, "packed_array element type must be integral");
    static_assert(sizeof(T) == 1 || sizeof(T) == 2, "packed_array element size must be 1 or 2 bytes");
    static_assert((sizeof(T) == 1 && (N == 4 || N == 8)) || (sizeof(T) == 2 && (N == 2 || N == 4)),
                  "packed_array: use N=4,8 for 8-bit types or N=2,4 for 16-bit types (32/64-bit total width)");

    using lane_t = typename std::conditional<sizeof(T) == 1, uint8_t, uint16_t>::type;

    static constexpr int element_bits = 8 * static_cast<int>(sizeof(T));
    static constexpr int total_bits = element_bits * N;
    using storage_t = typename std::conditional<total_bits == 32, uint32_t, uint64_t>::type;
    static constexpr storage_t lane_mask = (storage_t(1) << element_bits) - storage_t(1);

    storage_t data = 0;

    static constexpr lane_t to_lane(T value) { return static_cast<lane_t>(value); }

    static constexpr storage_t lane_to_storage(lane_t value, int index)
    {
      return static_cast<storage_t>(value) << (index * element_bits);
    }

  public:
    constexpr packed_array() = default;

    /** Constructor from packed integer */
    constexpr explicit packed_array(storage_t packed) : data(packed) { }

    /** Constructor from individual elements (N == 2) */
    constexpr packed_array(T v0, T v1)
      : data(lane_to_storage(to_lane(v0), 0) | lane_to_storage(to_lane(v1), 1))
    {
      static_assert(N == 2, "packed_array: 2 constructor arguments require N == 2");
    }

    /** Constructor from individual elements (N == 4) */
    constexpr packed_array(T v0, T v1, T v2, T v3)
      : data(lane_to_storage(to_lane(v0), 0) | lane_to_storage(to_lane(v1), 1) | lane_to_storage(to_lane(v2), 2)
             | lane_to_storage(to_lane(v3), 3))
    {
      static_assert(N == 4, "packed_array: 4 constructor arguments require N == 4");
    }

    /** Constructor from individual elements (N == 8) */
    constexpr packed_array(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7)
      : data(lane_to_storage(to_lane(v0), 0) | lane_to_storage(to_lane(v1), 1) | lane_to_storage(to_lane(v2), 2)
             | lane_to_storage(to_lane(v3), 3) | lane_to_storage(to_lane(v4), 4) | lane_to_storage(to_lane(v5), 5)
             | lane_to_storage(to_lane(v6), 6) | lane_to_storage(to_lane(v7), 7))
    {
      static_assert(N == 8, "packed_array: 8 constructor arguments require N == 8");
    }

    /** Get at index */
    constexpr T get(int index) const
    {
      return static_cast<T>((data >> (index * element_bits)) & lane_mask);
    }

    /** Set at index */
    constexpr void set(int index, T value)
    {
      const int shift = index * element_bits;
      data = (data & ~(lane_mask << shift)) | lane_to_storage(to_lane(value), index);
    }

    constexpr T operator[](int index) const { return get(index); }

    class proxy
    {
    private:
      packed_array &arr;
      int idx;

    public:
      constexpr proxy(packed_array &a, int i) : arr(a), idx(i) { }

      constexpr proxy &operator=(T value)
      {
        arr.set(idx, value);
        return *this;
      }

      constexpr operator T() const { return arr.get(idx); }

      constexpr T operator++(int)
      {
        T prev = arr.get(idx);
        arr.set(idx, prev + 1);
        return prev;
      }

      constexpr T operator--(int)
      {
        T prev = arr.get(idx);
        arr.set(idx, prev - 1);
        return prev;
      }
    };

    constexpr proxy operator[](int index) { return proxy(*this, index); }

    constexpr storage_t getPacked() const { return data; }

    constexpr void setPacked(storage_t packed) { data = packed; }

    constexpr void clear() { data = 0; }

    /** Fill all positions with the same value */
    constexpr void fill(T value)
    {
      const lane_t lane = to_lane(value);
      storage_t pattern = lane_to_storage(lane, 0);
      for (int i = 1; i < N; ++i) { pattern |= lane_to_storage(lane, i); }
      data = pattern;
    }

    constexpr bool operator==(const packed_array &other) const { return data == other.data; }
    constexpr bool operator!=(const packed_array &other) const { return data != other.data; }
  };

} // namespace quda
