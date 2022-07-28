#pragma once

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

  template <typename T, unsigned B> __host__ __device__ T signextend(const T x)
  {
    struct {T x:B;} s;
    s.x = x;
    return s.x;
  }

  /**
     packed 20-bit 3 component complex vector
  */
  struct alignas(16) spinor_20 {
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
  };

  /**
     @brief Pack a 3 component complex vector into a spinor_20 type
     @param[out] out The resulting packed spinor object
     @param[in] in The input vector we which to pack
   */
  template <typename spinor>
  __host__ __device__ inline void pack(spinor_20 &out, const spinor &in)
  {
    constexpr float scale = 524287; // 2^19-1

    // find the max
    float max[2] = {fabs((float)in[0].real()), fabs((float)in[0].imag())};
#pragma unroll
    for (int i = 1; i < 3; i++) {
      max[0] = fmaxf(max[0], fabs(in[i].real()));
      max[1] = fmaxf(max[1], fabs(in[i].imag()));
    }
    max[0] = fmaxf(max[0], max[1]);

    // compute rounded up exponent for rescaling
    float_structure fs;
    fs.f = fdividef(max[0], scale);
    fs.s.exponent++;
    fs.s.mantissa = 0;
    out.exponent = fs.s.exponent;

    // rescale and convert to integer
    int vs[6];
#pragma unroll
    for (int i = 0; i < 3; i++) {
      vs[2 * i + 0] = lrintf(fdividef(in[i].real(), fs.f));
      vs[2 * i + 1] = lrintf(fdividef(in[i].imag(), fs.f));
    }

    unsigned int vu[6];
#pragma unroll
    for (int i = 0; i < 6; i++) memcpy(vu + i, vs + i, sizeof(int));

    // split into required bitfields
    out.a_re = vu[0];
    out.a_im_hi = vu[1] >> 8;

    out.a_im_lo = vu[1] & 255;
    out.b_re = vu[2];
    out.b_im_hi = vu[3] >> 16;

    out.b_im_lo = vu[3] & 65535;
    out.c_re_hi = vu[4] >> 4;

    out.c_re_lo = vu[4] & 15;
    out.c_im = vu[5];
  }

  /**
     @brief Unpack a 3 component complex vector from a spinor_20 type
     @param[out] out The resulting unpacked complex vector
     @param[in] in The input spinor_20 we which to unpack
   */
  template <typename spinor>
  __host__ __device__ inline void unpack(spinor &v, const spinor_20 &in)
  {
    // reconstruct 20-bit numbers
    unsigned int vu[6];
    vu[0] = in.a_re;
    vu[1] = (in.a_im_hi << 8) + in.a_im_lo;
    vu[2] = in.b_re;
    vu[3] = (in.b_im_hi << 16) + in.b_im_lo;
    vu[4] = (in.c_re_hi << 4) + in.c_re_lo;
    vu[5] = in.c_im;

    // convert to signed
    int vs[6];
#pragma unroll
    for (int i = 0; i < 6; i++) memcpy(vs + i, vu + i, sizeof(int));

    // signed extend to 32 bits and rescale
    float_structure fs;
    fs.f = 0;
    fs.s.exponent = in.exponent;

#pragma unroll
    for (int i = 0; i < 3; i++) {
      v[i].real((float)signextend<signed int, 20>(vs[2 * i + 0]) * fs.f);
      v[i].imag((float)signextend<signed int, 20>(vs[2 * i + 1]) * fs.f);
    }
  }
 
}
