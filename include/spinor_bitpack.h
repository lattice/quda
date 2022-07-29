#pragma once

#include <limits>
#include <math_helper.cuh>

namespace quda {

  union float_structure {
    float f;
    struct float32 {
      unsigned int mantissa : 23;
      unsigned int exponent : 8;
      unsigned int sign     : 1;
    } s;
  };

  template <unsigned int B, typename T> __host__ __device__ T signextend(const T x)
  {
    static_assert(std::is_signed_v<T>, "Cannot sign extend an unsigned type");
    struct {T x:B;} s;
    s.x = x;
    return s.x;
  }

  /**
     @brief Helper function that returns the required scale factor
     (necessary given lack of constexpr pow function)
     @tparam bitwidth The bitwidth we are seeking to pack to
     @return The required scale factor, which corresponds to the
     maximum value (2^(bitwidth-1) -1)
  */
  template <unsigned int bitwidth> constexpr auto get_scale();
  template<> constexpr auto get_scale<20>() { return 524287; } // 2^19-1
  template<> constexpr auto get_scale<24>() { return 8388607; } // 2^23-1
  template<> constexpr auto get_scale<25>() { return 16777215; } // 2^24-1
  template<> constexpr auto get_scale<30>() { return 536870911; } // 2^29 - 1
  template<> constexpr auto get_scale<31>() { return 1073741823; } // 2^30 - 1

  template <unsigned int> struct spinor_packed;

  /**
     packed 20-bit 3 component complex vector
  */
  template <> struct alignas(16) spinor_packed<20> {
    static constexpr unsigned int bitwidth = 20;
    static constexpr float scale = get_scale<bitwidth>();

    // 32 bits
    unsigned int a_re : 20;
    unsigned int a_im_hi : 12;

    // 32 bits
    unsigned int a_im_lo : 8;
    unsigned int b_re : 20;
    unsigned int b_im_hi : 4;

    // 32 bits
    unsigned int b_im_lo : 16;
    unsigned int c_re_hi : 16;

    // 32 bits
    unsigned int c_re_lo : 4;
    unsigned int c_im : 20;
    unsigned int exponent : 8;

    spinor_packed() = default;
    template <typename spinor> __host__ __device__ spinor_packed(const spinor &in) { pack(in); }

    /**
       @brief Pack a 3 component complex vector into this
       @param[in] in The input vector we which to pack
    */
    template <typename spinor> __host__ __device__ inline void pack(const spinor &in)
    {
      // find the max
      float max[2] = {fabsf(in[0].real()), fabsf(in[0].imag())};
#pragma unroll
      for (int i = 1; i < 3; i++) {
        max[0] = fmaxf(max[0], fabsf(in[i].real()));
        max[1] = fmaxf(max[1], fabsf(in[i].imag()));
      }
      max[0] = fmaxf(max[0], max[1]);

      // ensures correct max covers all values if input vector is higher precision
      if (sizeof(in[0].real()) > sizeof(float)) max[0] += max[0] * std::numeric_limits<float>::epsilon();

      // compute rounded up exponent for rescaling
      float_structure fs;
      fs.f = fdividef(max[0], scale);
      fs.s.exponent++;
      fs.s.mantissa = 0;
      exponent = fs.s.exponent;

      // rescale and convert to integer
      int vs[6];
#pragma unroll
      for (int i = 0; i < 3; i++) {
        vs[2 * i + 0] = lrint(fdividef(in[i].real(), fs.f));
        vs[2 * i + 1] = lrint(fdividef(in[i].imag(), fs.f));
      }

      unsigned int vu[6];
#pragma unroll
      for (int i = 0; i < 6; i++) memcpy(vu + i, vs + i, sizeof(int));

      // split into required bitfields
      a_re = vu[0];
      a_im_hi = vu[1] >> 8;

      a_im_lo = vu[1] & 255;
      b_re = vu[2];
      b_im_hi = vu[3] >> 16;

      b_im_lo = vu[3] & 65535;
      c_re_hi = vu[4] >> 4;

      c_re_lo = vu[4] & 15;
      c_im = vu[5];
    }

    /**
       @brief Unpack into a 3-component complex vector
       @param[out] out The resulting unpacked complex vector
    */
    template <typename spinor> __host__ __device__ inline void unpack(spinor &v)
    {
      // reconstruct 20-bit numbers
      unsigned int vu[6];
      vu[0] = a_re;
      vu[1] = (a_im_hi << 8) + a_im_lo;
      vu[2] = b_re;
      vu[3] = (b_im_hi << 16) + b_im_lo;
      vu[4] = (c_re_hi << 4) + c_re_lo;
      vu[5] = c_im;

      // convert to signed
      int vs[6];
#pragma unroll
      for (int i = 0; i < 6; i++) memcpy(vs + i, vu + i, sizeof(int));

      // signed extend to 32 bits and rescale
      float_structure fs;
      fs.f = 0;
      fs.s.exponent = exponent;

      using real = decltype(v[0].real());
#pragma unroll
      for (int i = 0; i < 3; i++) {
        v[i].real(static_cast<real>(signextend<bitwidth>(vs[2 * i + 0])) * fs.f);
        v[i].imag(static_cast<real>(signextend<bitwidth>(vs[2 * i + 1])) * fs.f);
      }
    }

  };

  /**
     packed 30-bit 3 component complex vector
  */
  template <> struct spinor_packed<30> {
    static constexpr unsigned int bitwidth = 30;
    static constexpr float scale = get_scale<bitwidth>();

    // 32 bits
    unsigned int a_re : bitwidth;
    unsigned int exponent0: 2;

    // 32 bits
    unsigned int a_im : bitwidth;
    unsigned int exponent1: 2;

    // 32 bits
    unsigned int b_re : bitwidth;
    unsigned int exponent2: 2;

    // 32 bits
    unsigned int b_im : bitwidth;
    unsigned int exponent3: 2;

    // 32 bits
    unsigned int c_re : bitwidth;
    unsigned int dummy0: 2;

    // 32 bits
    unsigned int c_im : bitwidth;
    unsigned int dummy1: 2;

    spinor_packed() = default;
    template <typename spinor> __host__ __device__ spinor_packed(const spinor &in) { pack(in); }

    /**
       @brief Pack a 3 component complex vector into this
       @param[in] in The input vector we which to pack
    */
    template <typename spinor>  __host__ __device__ inline void pack(const spinor &in)
    {
      // find the max
      float max[2] = {fabsf(in[0].real()), fabsf(in[0].imag())};
#pragma unroll
      for (int i = 1; i < 3; i++) {
        max[0] = fmaxf(max[0], fabsf(in[i].real()));
        max[1] = fmaxf(max[1], fabsf(in[i].imag()));
      }
      max[0] = fmaxf(max[0], max[1]);

      // ensures correct max covers all values if input vector is higher precision
      if (sizeof(in[0].real()) > sizeof(float)) max[0] += max[0] * std::numeric_limits<float>::epsilon();

      // compute rounded up exponent for rescaling
      float_structure fs;
      fs.f = fdividef(max[0], scale);
      fs.s.exponent++;
      fs.s.mantissa = 0;

      // pack the exponent
      exponent0 = fs.s.exponent >> 0;
      exponent1 = fs.s.exponent >> 2;
      exponent2 = fs.s.exponent >> 4;
      exponent3 = fs.s.exponent >> 6;

      // rescale and convert to integer
      int vs[6];
#pragma unroll
      for (int i = 0; i < 3; i++) {
        vs[2 * i + 0] = lrint(in[i].real() / fs.f); // FIXME - can we avoid the division here?
        vs[2 * i + 1] = lrint(in[i].imag() / fs.f);
      }

      unsigned int vu[6];
#pragma unroll
      for (int i = 0; i < 6; i++) memcpy(vu + i, vs + i, sizeof(int));

      // split into required bitfields
      a_re = vu[0];
      a_im = vu[1];
      b_re = vu[2];
      b_im = vu[3];
      c_re = vu[4];
      c_im = vu[5];
    }

    /**
       @brief Unpack into a 3-component complex vector
       @param[out] out The resulting unpacked complex vector
    */
    template <typename spinor> __host__ __device__ inline void unpack(spinor &v)
    {
      // reconstruct 30-bit numbers
      unsigned int vu[6];
      vu[0] = a_re;
      vu[1] = a_im;
      vu[2] = b_re;
      vu[3] = b_im;
      vu[4] = c_re;
      vu[5] = c_im;

      // convert to signed
      int vs[6];
#pragma unroll
      for (int i = 0; i < 6; i++) memcpy(vs + i, vu + i, sizeof(int));

      // signed extend to 32 bits and rescale
      // FIXME could construct fp64 number here directly to avoid the conversion
      float_structure fs;
      fs.f = 0;
      fs.s.exponent = exponent0 + (exponent1 << 2) + (exponent2 << 4) + (exponent3 << 6);

      using real = decltype(v[0].real());
#pragma unroll
      for (int i = 0; i < 3; i++) {
        v[i].real(static_cast<real>(signextend<bitwidth>(vs[2 * i + 0])) * fs.f);
        v[i].imag(static_cast<real>(signextend<bitwidth>(vs[2 * i + 1])) * fs.f);
      }
    }

  };

  using spinor_20 = spinor_packed<20>;
  static_assert(sizeof(spinor_20) == 16, "spinor_20 must be 16 bytes");

  using spinor_30 = spinor_packed<30>;
  static_assert(sizeof(spinor_30) == 24, "spinor_30 must be 24 bytes");

}
