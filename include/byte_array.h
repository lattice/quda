#pragma once

#include <cstdint>

namespace quda
{

  /**
     @brief This is a class that emulates an array of 4x 8-bit types in
     a single 32-bit word.  This provides an indexable array that can be
     stored in registers without the overhead of branching.

     In the future we could generalize this to a 64-bit array of
     length 8, or use wider elements.
  */
  template <class T = int8_t, int n = 4> class byte_array
  {
    static_assert(sizeof(T) == 1, "element size must be 1 byte");
    static_assert(n == 4, "array length of 4 required");

  private:
    uint32_t data = 0;

  public:
    constexpr byte_array() = default;

    /** Constructor from packed integer */
    constexpr explicit byte_array(uint32_t packed) : data(packed) { }

    /** Constructor from 4 individual 8-bit values */
    constexpr byte_array(T v0, T v1, T v2, T v3)
    {
      data = (static_cast<uint32_t>(static_cast<uint8_t>(v0))) | (static_cast<uint32_t>(static_cast<uint8_t>(v1)) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24);
    }

    /** Get at index (0-3) */
    constexpr T get(int index) const { return static_cast<T>((data >> (index * 8)) & 0xFF); }

    /** Set at index (0-3) */
    constexpr void set(int index, T value)
    {
      int shift = index * 8;
      data = (data & ~(0xFF << shift)) | (static_cast<uint32_t>(static_cast<uint8_t>(value)) << shift);
    }

    constexpr T operator[](int index) const { return get(index); } /** Array accessor */

    class proxy
    {
    private:
      byte_array &arr;
      int idx;

    public:
      constexpr proxy(byte_array &a, int i) : arr(a), idx(i) { }

      constexpr proxy &operator=(T value)
      {
        arr.set(idx, value);
        return *this;
      }

      constexpr operator T() const { return arr.get(idx); } /** Conversion operator for getting value */

      constexpr auto operator++(int)
      {
        arr.set(idx, arr.get(idx) + 1);
        return arr.get(idx);
      }

      constexpr auto operator--(int)
      {
        arr.set(idx, arr.get(idx) - 1);
        return arr.get(idx);
      }
    };

    constexpr proxy operator[](int index) { return proxy(*this, index); }

    constexpr uint32_t getPacked() const { return data; } /** Get packed integer */

    constexpr void setPacked(uint32_t packed) { data = packed; } /** Set from packed integer */

    constexpr void clear() { data = 0; } /** Clear contents */

    /** Fill all positions with the same T value */
    constexpr void fill(T value)
    {
      uint32_t byte = static_cast<uint8_t>(value);
      data = (byte << 24) | (byte << 16) | (byte << 8) | byte;
    }

    /** Comparison operators */
    constexpr bool operator==(const byte_array &other) const { return data == other.data; }
    constexpr bool operator!=(const byte_array &other) const { return data != other.data; }
  };

} // namespace quda
