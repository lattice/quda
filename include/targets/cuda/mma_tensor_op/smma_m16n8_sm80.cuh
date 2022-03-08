#pragma once

#include "mma_inst.cuh"

template <class T>
static void __device__ inline get_big_small(float &big, float &small, const float &f){
  constexpr unsigned big_mask = std::is_same<T, bfloat16>::value ? 0xffff0000 : 0xffffe000;
  // constexpr unsigned small_mask = std::is_same<T, bfloat16>::value ? 0x8000 : 0x1000;

  const unsigned &u32 = reinterpret_cast<const unsigned &>(f);
  unsigned big_u32 = u32 & big_mask;
  big = reinterpret_cast<float &>(big_u32);

  small = f - big;
  // unsigned &small_u32 = reinterpret_cast<unsigned &>(small);
  // small_u32 += small_mask;
}

template <class dest_t, int input_vn>
struct Shuffle { };

template <>
struct Shuffle <bfloat16, 2>{
  static constexpr int input_vn = 2;

  void __device__ inline operator()(unsigned &big, unsigned &small, float f[input_vn]){
    float f_big[input_vn];
    float f_small[input_vn];

#pragma unroll
    for (int i = 0; i < input_vn; i++) {
      get_big_small<bfloat16>(f_big[i], f_small[i], f[i]);
    }

    bfloat162 &big_b32 = reinterpret_cast<bfloat162 &>(big);
    big_b32 = __floats2bfloat162_rn(f_big[0], f_big[1]);
    bfloat162 &small_b32 = reinterpret_cast<bfloat162 &>(small);
    small_b32 = __floats2bfloat162_rn(f_small[0], f_small[1]);
  }
};

template <>
struct Shuffle <tfloat32, 1>{
  static constexpr int input_vn = 1;

  void __device__ inline operator()(unsigned &big, unsigned &small, float f[input_vn]){
    float f_big[input_vn];
    float f_small[input_vn];

#pragma unroll
    for (int i = 0; i < input_vn; i++) {
      get_big_small<tfloat32>(f_big[i], f_small[i], f[i]);
    }

    float &big_store = reinterpret_cast<float &>(big);
    big_store = f_big[0];
    float &small_store = reinterpret_cast<float &>(small);
    small_store = f_small[0];
  }
};

template <class shuffle_t, int inst_k_, int warp_m_, int warp_n_>
struct Smma {

  static constexpr bool use_intermediate_accumulator() {
    return true;
  };

  static constexpr int warp_m = warp_m_;
  static constexpr int warp_n = warp_n_;

  static constexpr int inst_m = 16;
  static constexpr int inst_n = 8;
  static constexpr int inst_k = inst_k_;

  static constexpr int mma_m = inst_m * warp_m;
  static constexpr int mma_n = inst_n * warp_n;
  static constexpr int mma_k = inst_k;

  using store_t = unsigned;
  using input_t = float;

  static constexpr int input_vn = std::is_same<shuffle_t, bfloat16>::value ? 2 : 1; // input (op A/B) vector length

  static constexpr int t_pad = input_vn % 2 == 0 ? 4 : 8;
  static constexpr int n_pad = input_vn % 2 == 0 ? 8 : 4;

  static constexpr int acc_pad = 8;

  struct WarpRegisterMapping {

    int lane_id;
    int group_id;
    int thread_id_in_group;

    __device__ WarpRegisterMapping(int thread_id) :
      lane_id(thread_id & 31),
      group_id(lane_id >> 2),
      thread_id_in_group(lane_id & 3)
    {
    }
  };

  struct OperandA {

    static constexpr int thread_k = inst_k / (input_vn * 4);
    static constexpr int thread_m = inst_m / 8;
    static constexpr int thread_count = thread_k * thread_m;

    store_t big[warp_m * thread_count];
    store_t small[warp_m * thread_count];

    template <int lda> __device__ inline void load(void *smem, int tile_k, int tile_m, const WarpRegisterMapping &wrm)
    { // Assuming col major smem layout

      input_t *A = reinterpret_cast<input_t *>(smem);

#pragma unroll
      for (int tk = 0; tk < thread_k; tk++) {
#pragma unroll
        for (int tm = 0; tm < thread_m; tm++) {
#pragma unroll
          for (int wm = 0; wm < warp_m; wm++) {
            int k = tile_k * mma_k + (tk * 4 + wrm.thread_id_in_group) * input_vn;
            int m = tile_m * mma_m + wm * inst_m + (tm * 8 + wrm.group_id);
            float f[input_vn];
#pragma unroll
            for (int v = 0; v < input_vn; v++) {
              f[v] = A[(k + v) * lda + m];
            }
            int rc = wm * thread_count + (tk * thread_m + tm);
            Shuffle<shuffle_t, input_vn> s;
            s(big[rc], small[rc], f);
          }
        }
      }
    }
  };

  struct OperandB {

    static constexpr int thread_k = inst_k / (input_vn * 4);
    static constexpr int thread_n = inst_n / 8;
    static constexpr int thread_count = thread_k * thread_n;

    store_t big[warp_n * thread_count];
    store_t small[warp_n * thread_count];

    template <int ldb> __device__ inline void load(void *smem, int tile_k, int tile_n, const WarpRegisterMapping &wrm)
    { // Assuming col major smem layout

      input_t *B = reinterpret_cast<input_t *>(smem);

#pragma unroll
      for (int tk = 0; tk < thread_k; tk++) {
#pragma unroll
        for (int tn = 0; tn < thread_n; tn++) {
#pragma unroll
          for (int wn = 0; wn < warp_n; wn++) {
            int k = tile_k * mma_k + (tk * 4 + wrm.thread_id_in_group) * input_vn;
            int n = tile_n * mma_n + wn * inst_n + (tn * 8 + wrm.group_id);
            float f[input_vn];
#pragma unroll
            for (int v = 0; v < input_vn; v++) {
              f[v] = B[(k + v) * ldb + n];
            }

            int rc = wn * thread_count + (tk * thread_n + tn);
            Shuffle<shuffle_t, input_vn> s;
            s(big[rc], small[rc], f);
          }
        }
      }
    }
  };

  struct OperandC {

    static constexpr int thread_m = inst_m / 8;
    static constexpr int thread_n = inst_n / 4;
    static constexpr int thread_count = thread_m * thread_n;

    using reg_type = float;
    reg_type reg[warp_m * warp_n * thread_count];

    __device__ OperandC()
    {
#pragma unroll
      for (int i = 0; i < warp_m * warp_n * thread_count; i++) { reg[i] = 0; }
    }

    template <int ldc> __device__ void store(void *ptr, int m_offset, int n_offset, const WarpRegisterMapping &wrm)
    {
      reg_type *C = reinterpret_cast<reg_type *>(ptr);
#pragma unroll
      for (int tm = 0; tm < thread_m; tm++) {
#pragma unroll
        for (int tn = 0; tn < thread_n; tn++) {
#pragma unroll
          for (int wn = 0; wn < warp_n; wn++) {
#pragma unroll
            for (int wm = 0; wm < warp_m; wm++) {
              int m = m_offset + wm * inst_m + (wrm.group_id + tm * 8);
              int n = n_offset + wn * inst_n + (wrm.thread_id_in_group * 2 + tn);
              C[m * ldc + n] = reg[(wn * warp_m + wm) * thread_count + (tm * thread_n + tn)];
            }
          }
        }
      }
    }
  };

  static __device__ void mma(OperandC &op_c, const OperandA &op_a, const OperandB &op_b)
  {
#pragma unroll
    for (int wm = 0; wm < warp_m; wm++) {
#pragma unroll
      for (int wn = 0; wn < warp_n; wn++) {

        int a_offset = wm * OperandA::thread_count;
        int b_offset = wn * OperandB::thread_count;
        int c_offset = (wn * warp_m + wm) * OperandC::thread_count;

        MmaInst<inst_m, inst_n, inst_k, shuffle_t, float> mma;

        if (use_intermediate_accumulator()) {
          float acc[OperandC::thread_count];
#pragma unroll
          for (int c = 0; c < OperandC::thread_count; c++) {
            acc[c] = 0;
          }

          mma(acc, &op_a.big[a_offset], &op_b.big[b_offset]);
          mma(acc, &op_a.big[a_offset], &op_b.small[b_offset]);
          mma(acc, &op_a.small[a_offset], &op_b.big[b_offset]);

#pragma unroll
          for (int c = 0; c < OperandC::thread_count; c++) {
            op_c.reg[c_offset + c] += acc[c];
          }
        } else {
          mma(&op_c.reg[c_offset], &op_a.big[a_offset], &op_b.big[b_offset]);
          mma(&op_c.reg[c_offset], &op_a.big[a_offset], &op_b.small[b_offset]);
          mma(&op_c.reg[c_offset], &op_a.small[a_offset], &op_b.big[b_offset]);
        }
      }
    }
  }

};

