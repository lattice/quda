#pragma once

#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <dslash_quda.h>
#include <index_helper.cuh>
#include <inline_ptx.h>
#include <math_helper.cuh>
#include <shared_memory_cache_helper.h>

#include <quda_fp16.cuh>

#include <block_reduce_helper.h>
#include <mma_tensor_op/mma_dispatch.cuh>

namespace quda
{

  // matrix a for a generic matrix: column major, M/M_sm(size/padded size) by k
  // (spin,Ls) by (spin,Ls), where left most index is the fastest changing
  // one(spin).
  // x by y
  // For now, assuming it's trivial in spin
  template <int block_dim_x, int Ls, int M_sm, class compute_type>
  __device__ inline void construct_matrix_a_generic(half *sm_a, compute_type *generic)
  {
    int offset_k = threadIdx.y * 4;
    int x = threadIdx.x;

    while (x < Ls) {
      int offset_m = x * 4;
      float value = generic[x * Ls + threadIdx.y]; // Assuming the input matrix is row major

      // exponent = 0 means we are on the diagonal.
      sm_a[(offset_k + 0) * (M_sm) + (offset_m + 0)] = value;
      sm_a[(offset_k + 1) * (M_sm) + (offset_m + 1)] = value;
      sm_a[(offset_k + 2) * (M_sm) + (offset_m + 2)] = value;
      sm_a[(offset_k + 3) * (M_sm) + (offset_m + 3)] = value;

      // sm_a[ (offset_k+0)*(M_sm)+(offset_m+0) ] = factorR + factorL;
      sm_a[(offset_k + 0) * (M_sm) + (offset_m + 1)] = static_cast<half>(0.0f);
      sm_a[(offset_k + 0) * (M_sm) + (offset_m + 2)] = static_cast<half>(0.0f);
      sm_a[(offset_k + 0) * (M_sm) + (offset_m + 3)] = static_cast<half>(0.0f);

      sm_a[(offset_k + 1) * (M_sm) + (offset_m + 0)] = static_cast<half>(0.0f);
      // sm_a[ (offset_k+1)*(M_sm)+(offset_m+1) ] = factorR + factorL;
      sm_a[(offset_k + 1) * (M_sm) + (offset_m + 2)] = static_cast<half>(0.0f);
      sm_a[(offset_k + 1) * (M_sm) + (offset_m + 3)] = static_cast<half>(0.0f);

      sm_a[(offset_k + 2) * (M_sm) + (offset_m + 0)] = static_cast<half>(0.0f);
      sm_a[(offset_k + 2) * (M_sm) + (offset_m + 1)] = static_cast<half>(0.0f);
      // sm_a[ (offset_k+2)*(M_sm)+(offset_m+2) ] = factorR + factorL;
      sm_a[(offset_k + 2) * (M_sm) + (offset_m + 3)] = static_cast<half>(0.0f);

      sm_a[(offset_k + 3) * (M_sm) + (offset_m + 0)] = static_cast<half>(0.0f);
      sm_a[(offset_k + 3) * (M_sm) + (offset_m + 1)] = static_cast<half>(0.0f);
      sm_a[(offset_k + 3) * (M_sm) + (offset_m + 2)] = static_cast<half>(0.0f);
      // sm_a[ (offset_k+3)*(M_sm)+(offset_m+3) ] = factorR + factorL;

      x += block_dim_x;
    }
  }

  // matrix a for m5inv: column major, M/M_sm(size/padded size) by k
  // (spin,Ls) by (spin,Ls), where left most index is the fastest changing
  // one(spin).
  // x by y
  template <int M_sm, bool dagger, class Arg>
  __device__ inline void construct_matrix_a_m5inv(Arg &arg, half *sm_a, const float *mp = nullptr,
                                                  const float *mm = nullptr)
  {
    constexpr int Ls = Arg::Ls;
    const float k = arg.kappa;
    // if we rescale, then the actual matrix is alpha*m5inv+beta.
    // Otherwise a = 1., b = 0.;
    const float b = arg.beta;

    const float inv = arg.alpha * arg.fac_inv;

    auto offset_k = threadIdx.y * 4;
    auto x = threadIdx.x;

    while (x < Ls) {
      int offset_m = x * 2;
      float factorR, factorL;

      if (mp && mm) {
        if (dagger) {
          factorR = mp[x * Ls + threadIdx.y];
          factorL = mm[x * Ls + threadIdx.y];
        } else {
          factorR = mp[threadIdx.y * Ls + x];
          factorL = mm[threadIdx.y * Ls + x];
        }
      } else {
        int exponent;
        if (dagger) {
          exponent = x > threadIdx.y ? Ls - x + threadIdx.y : threadIdx.y - x;
          factorR = inv * powf(k, __int2float_rn(exponent)) * (x > threadIdx.y ? -arg.m_f : 1.f);
        } else {
          exponent = x < threadIdx.y ? Ls - threadIdx.y + x : x - threadIdx.y;
          factorR = inv * powf(k, __int2float_rn(exponent)) * (x < threadIdx.y ? -arg.m_f : 1.f);
        }

        if (dagger) {
          exponent = x < threadIdx.y ? Ls - threadIdx.y + x : x - threadIdx.y;
          factorL = inv * powf(k, __int2float_rn(exponent)) * (x < threadIdx.y ? -arg.m_f : 1.f);
        } else {
          exponent = x > threadIdx.y ? Ls - x + threadIdx.y : threadIdx.y - x;
          factorL = inv * powf(k, __int2float_rn(exponent)) * (x > threadIdx.y ? -arg.m_f : 1.f);
        }
      }

      float RpL = x == threadIdx.y ? factorR + factorL + b : factorR + factorL;
      float RmL = factorR - factorL;

      half2 *A = reinterpret_cast<half2 *>(sm_a);

      A[(offset_k + 0) * (M_sm / 2) + (offset_m + 0)] = __floats2half2_rn(RpL, 0.0f);
      A[(offset_k + 0) * (M_sm / 2) + (offset_m + 1)] = __floats2half2_rn(RmL, 0.0f);

      A[(offset_k + 1) * (M_sm / 2) + (offset_m + 0)] = __floats2half2_rn(0.0f, RpL);
      A[(offset_k + 1) * (M_sm / 2) + (offset_m + 1)] = __floats2half2_rn(0.0f, RmL);

      A[(offset_k + 2) * (M_sm / 2) + (offset_m + 0)] = __floats2half2_rn(RmL, 0.0f);
      A[(offset_k + 2) * (M_sm / 2) + (offset_m + 1)] = __floats2half2_rn(RpL, 0.0f);

      A[(offset_k + 3) * (M_sm / 2) + (offset_m + 0)] = __floats2half2_rn(0.0f, RmL);
      A[(offset_k + 3) * (M_sm / 2) + (offset_m + 1)] = __floats2half2_rn(0.0f, RpL);

      x += Arg::block_dim_x;
    }
  }

  // matrix a for m5pre: column major, M/M_sm(size/padded size) by k
  // (spin,Ls) by (spin,Ls), where left most index is the fastest changing
  // one(spin).
  // x by y
  template <int M_sm, bool dagger, class Arg>
  __device__ inline void construct_matrix_a_d5(Arg &arg, half *sm_a)
  {
    constexpr int Ls = Arg::Ls;
    // if we rescale, then the actual matrix is alpha*m5inv+beta.
    // Otherwise a = 1., b = 0.;
    const float b = arg.beta;

    auto offset_k = threadIdx.y * 4;
    auto x = threadIdx.x;

    while (x < Ls) {
      int offset_m = x * 2;
      int exponent = x - threadIdx.y;
      float factorR, factorL;

      if (dagger) {
        factorR = (exponent == -1 ? 1.f : (exponent == +Ls - 1 ? -arg.m_f : 0.f));
      } else {
        factorR = (exponent == +1 ? 1.f : (exponent == -Ls + 1 ? -arg.m_f : 0.f));
      }

      if (dagger) {
        factorL = (exponent == +1 ? 1.f : (exponent == -Ls + 1 ? -arg.m_f : 0.f));
      } else {
        factorL = (exponent == -1 ? 1.f : (exponent == +Ls - 1 ? -arg.m_f : 0.f));
      }

      // exponent = 0 means we are on the diagonal.
      float RpL = exponent == 0 ? arg.alpha * (factorR + factorL) + b : arg.alpha * (factorR + factorL);
      float RmL = arg.alpha * (factorR - factorL);

      half2 *A = reinterpret_cast<half2 *>(sm_a);

      A[(offset_k + 0) * (M_sm / 2) + (offset_m + 0)] = __floats2half2_rn(RpL, 0.0f);
      A[(offset_k + 0) * (M_sm / 2) + (offset_m + 1)] = __floats2half2_rn(RmL, 0.0f);

      A[(offset_k + 1) * (M_sm / 2) + (offset_m + 0)] = __floats2half2_rn(0.0f, RpL);
      A[(offset_k + 1) * (M_sm / 2) + (offset_m + 1)] = __floats2half2_rn(0.0f, RmL);

      A[(offset_k + 2) * (M_sm / 2) + (offset_m + 0)] = __floats2half2_rn(RmL, 0.0f);
      A[(offset_k + 2) * (M_sm / 2) + (offset_m + 1)] = __floats2half2_rn(RpL, 0.0f);

      A[(offset_k + 3) * (M_sm / 2) + (offset_m + 0)] = __floats2half2_rn(0.0f, RmL);
      A[(offset_k + 3) * (M_sm / 2) + (offset_m + 1)] = __floats2half2_rn(0.0f, RpL);

      x += Arg::block_dim_x;
    }
  }

  template <class integer_vec> __device__ inline integer_vec __2half22integer4_rn(const half2 &a, const half2 &b)
  {
    integer_vec c;
    c.x = __half2short_rn(a.x);
    c.y = __half2short_rn(a.y);
    c.z = __half2short_rn(b.x);
    c.w = __half2short_rn(b.y);
    return c;
  }

  template <class integer_vec>
  __device__ inline integer_vec __4half22integer8_rn(const half2 &a, const half2 &b, const half2 &c, const half2 &d)
  {
    integer_vec e;
    e.x.x = __half2short_rn(a.x);
    e.x.y = __half2short_rn(a.y);
    e.x.z = __half2short_rn(b.x);
    e.x.w = __half2short_rn(b.y);
    e.y.x = __half2short_rn(c.x);
    e.y.y = __half2short_rn(c.y);
    e.y.z = __half2short_rn(d.x);
    e.y.w = __half2short_rn(d.y);
    return e;
  }

  __device__ inline void __half_max_abs_half2__(half &max, const half2 &input)
  {
    half2 lh = habs2(input);
    if (__hgt(lh.x, max)) { max = lh.x; }
    if (__hgt(lh.y, max)) { max = lh.y; }
  }

  __inline__ __device__ void __float_max_abs_floats__(float &max, const float &input)
  {
    constexpr uint32_t maximum_mask = 0x7fffffffu; // 0111 1111 1111 1111 1111 1111 1111 1111
    uint32_t input_masked = *reinterpret_cast<const uint32_t *>(&input) & maximum_mask;
    if (*reinterpret_cast<float *>(&input_masked) > max) { max = *reinterpret_cast<float *>(&input_masked); }
  }

  __device__ inline void warp_wise_reduce_float(float &f)
  {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      float other_f = __shfl_down_sync(device::warp_converged_mask(), f, offset);
      if (other_f > f) { f = other_f; }
    }
  }

  constexpr float target_scale = 2e3;

  template <class Vector>
  __device__ inline float block_wise_reduce_vector(const Vector &v)
  {
    // Find the maximum absolute value in a lane
    float warp_max[2] = {0.0f, 0.0f};
#pragma unroll
    for (int spin = 0; spin < 4; spin++) {
#pragma unroll
      for (int color = 0; color < 3; color++) {
        __float_max_abs_floats__(warp_max[0], v(spin, color).real());
        __float_max_abs_floats__(warp_max[1], v(spin, color).imag());
      }
    }
    warp_max[0] = fmaxf(warp_max[0], warp_max[1]);

    constexpr int block_dim = 2;
    return BlockReduce<float, block_dim>().AllMax(warp_max[0]) / target_scale;
  }

  // Actually does more than the function name suggests.
  // will find the maximum absolute value among the vector, scale that, and store
  // to sm_b
  template <int N_sm_d2, bool accumulate, bool store = true, class Vector>
  __device__ inline void load_matrix_b_vector(Vector &v, half2 *sm_b, float &scale, float m_scale = 1.0f)
  {
    if (accumulate) {
      float previous_scale = scale * m_scale;
#pragma unroll
      for (int spin = 0; spin < 4; spin++) {
#pragma unroll
        for (int color = 0; color < 3; color++) {
          int idx = (threadIdx.y * 4 + spin) * N_sm_d2 + 3 * threadIdx.x + color;
          half2 h = sm_b[idx];
          v(spin, color) += complex<float>(h.x, h.y) * previous_scale;
        }
      }
    }
    if (store) {
      scale = block_wise_reduce_vector(v);
#pragma unroll
      for (int spin = 0; spin < 4; spin++) {
#pragma unroll
        for (int color = 0; color < 3; color++) {
          float real = v(spin, color).real() / scale;
          float imag = v(spin, color).imag() / scale;
          int idx = (threadIdx.y * 4 + spin) * N_sm_d2 + 3 * threadIdx.x + color;
          sm_b[idx] = __floats2half2_rn(real, imag);
        }
      }
    }
  }

  // Store results(scaled short/char values and scale) in shared memroy to global
  // memroy.
  template <class storage_type, int N_sm, class Output>
  __device__ inline void store_matrix_c(Output &output, half2 *sm_b, int sid, const float scale)
  {
    half max_ = 0.0f;
    constexpr int N_sm_d2 = N_sm / 2;
#pragma unroll
    for (int spin = 0; spin < 4; spin++) {
#pragma unroll
      for (int color = 0; color < 3; color++) {
        int idx = (threadIdx.y * 4 + spin) * N_sm_d2 + 3 * threadIdx.x + color;
        __half_max_abs_half2__(max_, sm_b[idx]);
      }
    }

    output.norm[sid] = __half2float(max_) * scale * fixedInvMaxValue<storage_type>::value;

    const half2 max_i_div_max2_ = __half2half2(__hdiv(fixedMaxValue<storage_type>::value, max_));
#if QUDA_ORDER_FP == 8 // use float8/short8
    typedef typename VectorType<storage_type, 8>::type storage_vec;
    storage_vec *out = reinterpret_cast<storage_vec *>(output.field);
    half2 a, b, c, d;

    a = __hmul2(sm_b[(threadIdx.y * 4 + 0) * N_sm_d2 + 3 * threadIdx.x + 0], max_i_div_max2_);
    b = __hmul2(sm_b[(threadIdx.y * 4 + 0) * N_sm_d2 + 3 * threadIdx.x + 1], max_i_div_max2_);
    c = __hmul2(sm_b[(threadIdx.y * 4 + 0) * N_sm_d2 + 3 * threadIdx.x + 2], max_i_div_max2_);
    d = __hmul2(sm_b[(threadIdx.y * 4 + 1) * N_sm_d2 + 3 * threadIdx.x + 0], max_i_div_max2_);
    vector_store(&out[sid + 0 * output.volumeCB], 0, __4half22integer8_rn<storage_vec>(a, b, c, d));

    a = __hmul2(sm_b[(threadIdx.y * 4 + 1) * N_sm_d2 + 3 * threadIdx.x + 1], max_i_div_max2_);
    b = __hmul2(sm_b[(threadIdx.y * 4 + 1) * N_sm_d2 + 3 * threadIdx.x + 2], max_i_div_max2_);
    c = __hmul2(sm_b[(threadIdx.y * 4 + 2) * N_sm_d2 + 3 * threadIdx.x + 0], max_i_div_max2_);
    d = __hmul2(sm_b[(threadIdx.y * 4 + 2) * N_sm_d2 + 3 * threadIdx.x + 1], max_i_div_max2_);
    vector_store(&out[sid + 1 * output.volumeCB], 0, __4half22integer8_rn<storage_vec>(a, b, c, d));

    a = __hmul2(sm_b[(threadIdx.y * 4 + 2) * N_sm_d2 + 3 * threadIdx.x + 2], max_i_div_max2_);
    b = __hmul2(sm_b[(threadIdx.y * 4 + 3) * N_sm_d2 + 3 * threadIdx.x + 0], max_i_div_max2_);
    c = __hmul2(sm_b[(threadIdx.y * 4 + 3) * N_sm_d2 + 3 * threadIdx.x + 1], max_i_div_max2_);
    d = __hmul2(sm_b[(threadIdx.y * 4 + 3) * N_sm_d2 + 3 * threadIdx.x + 2], max_i_div_max2_);
    vector_store(&out[sid + 2 * output.volumeCB], 0, __4half22integer8_rn<storage_vec>(a, b, c, d));
#elif QUDA_ORDER_FP == 4
    typedef typename VectorType<storage_type, 4>::type storage_vec;
    storage_vec *out = reinterpret_cast<storage_vec *>(output.field);
    half2 a, b;

    a = __hmul2(sm_b[(threadIdx.y * 4 + 0) * N_sm_d2 + 3 * threadIdx.x + 0], max_i_div_max2_);
    b = __hmul2(sm_b[(threadIdx.y * 4 + 0) * N_sm_d2 + 3 * threadIdx.x + 1], max_i_div_max2_);
    out[sid + 0 * output.volumeCB] = __2half22integer4_rn<storage_vec>(a, b);

    a = __hmul2(sm_b[(threadIdx.y * 4 + 0) * N_sm_d2 + 3 * threadIdx.x + 2], max_i_div_max2_);
    b = __hmul2(sm_b[(threadIdx.y * 4 + 1) * N_sm_d2 + 3 * threadIdx.x + 0], max_i_div_max2_);
    out[sid + 1 * output.volumeCB] = __2half22integer4_rn<storage_vec>(a, b);

    a = __hmul2(sm_b[(threadIdx.y * 4 + 1) * N_sm_d2 + 3 * threadIdx.x + 1], max_i_div_max2_);
    b = __hmul2(sm_b[(threadIdx.y * 4 + 1) * N_sm_d2 + 3 * threadIdx.x + 2], max_i_div_max2_);
    out[sid + 2 * output.volumeCB] = __2half22integer4_rn<storage_vec>(a, b);

    a = __hmul2(sm_b[(threadIdx.y * 4 + 2) * N_sm_d2 + 3 * threadIdx.x + 0], max_i_div_max2_);
    b = __hmul2(sm_b[(threadIdx.y * 4 + 2) * N_sm_d2 + 3 * threadIdx.x + 1], max_i_div_max2_);
    out[sid + 3 * output.volumeCB] = __2half22integer4_rn<storage_vec>(a, b);

    a = __hmul2(sm_b[(threadIdx.y * 4 + 2) * N_sm_d2 + 3 * threadIdx.x + 2], max_i_div_max2_);
    b = __hmul2(sm_b[(threadIdx.y * 4 + 3) * N_sm_d2 + 3 * threadIdx.x + 0], max_i_div_max2_);
    out[sid + 4 * output.volumeCB] = __2half22integer4_rn<storage_vec>(a, b);

    a = __hmul2(sm_b[(threadIdx.y * 4 + 3) * N_sm_d2 + 3 * threadIdx.x + 1], max_i_div_max2_);
    b = __hmul2(sm_b[(threadIdx.y * 4 + 3) * N_sm_d2 + 3 * threadIdx.x + 2], max_i_div_max2_);
    out[sid + 5 * output.volumeCB] = __2half22integer4_rn<storage_vec>(a, b);
#endif
  }

  template <class mma_t, int BlockDimX, int Ls, int M, int N, int M_PAD, int N_PAD, bool reload, class T>
  __device__ inline void mma_sync_gemm(T op_a[], half *sm_a, half *sm_b, half *sm_c,
                                       const typename mma_t::WarpRegisterMapping &wrm)
  {

    constexpr int tile_row_dim = M / mma_t::MMA_M; // number of tiles in the column dimension
    constexpr int tile_col_dim = N / mma_t::MMA_N; // number of tiles in the row dimension
    constexpr int tile_acc_dim = M / mma_t::MMA_K; // number of tiles in the row dimension

    constexpr int total_warp = BlockDimX * Ls / 32;

    static_assert((tile_row_dim * tile_col_dim) % total_warp == 0,
                  "Total number of tiles should be divisible by the number of warps.");
    static_assert(tile_col_dim % (tile_row_dim * tile_col_dim / total_warp) == 0,
                  "Each warp should only be responsible a single tile row.");

    constexpr int total_tile = tile_row_dim * tile_col_dim;
    constexpr int warp_cycle = total_tile / total_warp;

    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = thread_id >> 5;
    const int warp_row = warp_id * warp_cycle / tile_col_dim;

#pragma unroll
    for (int c = 0; c < warp_cycle; c++) {

      typename mma_t::OperandC op_c;

      // The logical warp assigned to each part of the matrix.
      const int logical_warp_index = warp_id * warp_cycle + c;
      const int warp_col = logical_warp_index - warp_row * tile_col_dim;
      // e.g. for 12 warps:
      // 000|111|222|333
      // 444|555|666|777
      // 888|999|000|111

#pragma unroll
      for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {

        if (reload) { // the data in registers can be resued.
          op_a[0].template load<M_PAD>(sm_a, tile_k, warp_row, wrm);
        }

        typename mma_t::OperandB op_b;
        op_b.template load<N_PAD>(sm_b, tile_k, warp_col, wrm);

        if (reload) {
          mma_t::mma(op_a[0], op_b, op_c);
        } else {
          mma_t::mma(op_a[tile_k], op_b, op_c);
        }
      }

      __syncthreads();

      op_c.template store<N_PAD>(sm_c, warp_row, warp_col, wrm);
    }
  }

} // namespace quda

