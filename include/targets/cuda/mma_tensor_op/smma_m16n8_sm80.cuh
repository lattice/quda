#pragma once

#include <mma_tensor_op/mma_instruction.cuh>
#include <tune_key.h>
#include <uint_to_char.h>

namespace quda
{

  namespace smma
  {

    using bfloat16 = mma::bfloat16;
    using bfloat162 = mma::bfloat162;
    using tfloat32 = mma::tfloat32;
    using half = mma::half;
    using half2 = mma::half2;

    template <class T> static void __device__ inline get_big_small(float &big, float &small, const float &f)
    {
      constexpr unsigned big_mask = std::is_same_v<T, bfloat16> ? 0xffff0000 : 0xffffe000;
      constexpr unsigned small_mask = std::is_same_v<T, bfloat16> ? 0x8000 : 0x1000;

      const unsigned &u32 = reinterpret_cast<const unsigned &>(f);
      unsigned big_u32 = u32 & big_mask;
      big = reinterpret_cast<float &>(big_u32);

      small = f - big;
      unsigned &small_u32 = reinterpret_cast<unsigned &>(small);
      small_u32 += small_mask;
    }

    template <class dest_t, int input_vn> struct Shuffle {
    };

    template <> struct Shuffle<bfloat16, 2> {
      static constexpr int input_vn = 2;

      void __device__ inline operator()(unsigned &big, unsigned &small, float f[input_vn])
      {
        float f_big[input_vn];
        float f_small[input_vn];

#pragma unroll
        for (int i = 0; i < input_vn; i++) { get_big_small<bfloat16>(f_big[i], f_small[i], f[i]); }

        bfloat162 &big_b32 = reinterpret_cast<bfloat162 &>(big);
        big_b32 = __floats2bfloat162_rn(f_big[0], f_big[1]);
        bfloat162 &small_b32 = reinterpret_cast<bfloat162 &>(small);
        small_b32 = __floats2bfloat162_rn(f_small[0], f_small[1]);
      }
    };

    template <> struct Shuffle<half, 2> {
      static constexpr int input_vn = 2;

      void __device__ inline operator()(unsigned &big, unsigned &small, float f[input_vn])
      {
        float f_big[input_vn];
        float f_small[input_vn];

#pragma unroll
        for (int i = 0; i < input_vn; i++) { get_big_small<tfloat32>(f_big[i], f_small[i], f[i]); }

        half2 &big_b32 = reinterpret_cast<half2 &>(big);
        big_b32 = __floats2half2_rn(f_big[0], f_big[1]);
        half2 &small_b32 = reinterpret_cast<half2 &>(small);
        small_b32 = __floats2half2_rn(f_small[0], f_small[1]);
      }
    };

    template <> struct Shuffle<tfloat32, 1> {
      static constexpr int input_vn = 1;

      void __device__ inline operator()(unsigned &big, unsigned &small, float f[input_vn])
      {
        float f_big[input_vn];
        float f_small[input_vn];

#pragma unroll
        for (int i = 0; i < input_vn; i++) { get_big_small<tfloat32>(f_big[i], f_small[i], f[i]); }

        float &big_store = reinterpret_cast<float &>(big);
        big_store = f_big[0];
        float &small_store = reinterpret_cast<float &>(small);
        small_store = f_small[0];
      }
    };

    template <class shuffle_t, int inst_k_, int warp_m_, int warp_n_> struct smma_t {

      static constexpr bool use_intermediate_accumulator() { return true; };

      static constexpr int warp_m = warp_m_;
      static constexpr int warp_n = warp_n_;

      static constexpr int inst_m = 16;
      static constexpr int inst_n = 8;
      static constexpr int inst_k = inst_k_;

      static constexpr int MMA_M = inst_m * warp_m;
      static constexpr int MMA_N = inst_n * warp_n;
      static constexpr int MMA_K = inst_k;

      static constexpr int mma_m = MMA_M;
      static constexpr int mma_n = MMA_N;
      static constexpr int mma_k = MMA_K;

      static std::string get_type_name()
      {
        char s[TuneKey::aux_n];
        if constexpr (std::is_same_v<shuffle_t, tfloat32>) {
          strcpy(s, ",3xtfloat32,m");
        } else if constexpr (std::is_same_v<shuffle_t, bfloat16>) {
          strcpy(s, ",3xbfloat16,m");
        } else if constexpr (std::is_same_v<shuffle_t, half>) {
          strcpy(s, ",3xfp16,m");
        } else {
          strcpy(s, "unknown_mma_type,m");
        }
        i32toa(s + strlen(s), MMA_M);
        strcat(s, "n");
        i32toa(s + strlen(s), MMA_N);
        strcat(s, "k");
        i32toa(s + strlen(s), MMA_K);
        return s;
      }

      static constexpr int warp_size = 32;

      using store_t = unsigned;
      using input_t = float;

      using compute_t = float;
      using load_t = float;

      static constexpr int input_vn = std::is_same_v<shuffle_t, tfloat32> ? 1 : 2; // input (op A/B) vector length

      static constexpr int t_pad = input_vn % 2 == 0 ? 4 : 8;
      static constexpr int n_pad = input_vn % 2 == 0 ? 8 : 4;

      static __device__ __host__ constexpr int inline pad_size(int m)
      {
        return std::is_same_v<shuffle_t, tfloat32> ? (m - 8 + 15) / 16 * 16 + 8 - m : (m - 4 + 7) / 8 * 8 + 4 - m;
      }

      static constexpr int acc_pad = 8;

      struct WarpRegisterMapping {

        int warp_id;
        int lane_id;
        int group_id;
        int thread_id_in_group;

        __device__ WarpRegisterMapping(int thread_id) :
          warp_id(thread_id / 32), lane_id(thread_id & 31), group_id(lane_id >> 2), thread_id_in_group(lane_id & 3)
        {
        }
      };

      struct OperandA {

        static constexpr int thread_k = inst_k / (input_vn * 4);
        static constexpr int thread_m = inst_m / 8;
        static constexpr int thread_count = thread_k * thread_m;

        store_t big[warp_m * thread_count];
        store_t small[warp_m * thread_count];

        __device__ inline void negate()
        {
#pragma unroll
          for (int i = 0; i < warp_m * thread_count; i++) {
            constexpr unsigned flip_bit_sign = std::is_same_v<shuffle_t, tfloat32> ? 0x80000000 : 0x80008000;
            big[i] ^= flip_bit_sign;
            small[i] ^= flip_bit_sign;
          }
        }

        template <int lda>
        __device__ inline void load(void *smem, int tile_k, int tile_m, const WarpRegisterMapping &wrm)
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
                for (int v = 0; v < input_vn; v++) { f[v] = A[(k + v) * lda + m]; }
                int rc = wm * thread_count + (tk * thread_m + tm);
                Shuffle<shuffle_t, input_vn> s;
                s(big[rc], small[rc], f);
              }
            }
          }
        }

        template <class smem_obj_t>
        __device__ inline void load(const smem_obj_t &smem_obj, int tile_k, int tile_m, const WarpRegisterMapping &wrm)
        { // Assuming col major smem layout

          input_t *A = reinterpret_cast<input_t *>(smem_obj.ptr);

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
                for (int v = 0; v < input_vn; v++) { f[v] = A[(k + v) * smem_obj_t::ldn + m]; }
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

        template <int ldb>
        __device__ inline void load(void *smem, int tile_k, int tile_n, const WarpRegisterMapping &wrm)
        { // Assuming row major smem layout

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
                for (int v = 0; v < input_vn; v++) { f[v] = B[(k + v) * ldb + n]; }

                int rc = wn * thread_count + (tk * thread_n + tn);
                Shuffle<shuffle_t, input_vn> s;
                s(big[rc], small[rc], f);
              }
            }
          }
        }

        template <class smem_obj_t>
        __device__ inline void load(const smem_obj_t &smem_obj, int tile_k, int tile_n, const WarpRegisterMapping &wrm)
        { // Assuming row major smem layout

          input_t *B = reinterpret_cast<input_t *>(smem_obj.ptr);

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
                for (int v = 0; v < input_vn; v++) { f[v] = B[(k + v) * smem_obj_t::ldn + n]; }

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

        __device__ inline void zero()
        {
#pragma unroll
          for (int i = 0; i < warp_m * warp_n * thread_count; i++) { reg[i] = 0; }
        }

        __device__ inline OperandC() { zero(); }

        __device__ inline void ax(float alpha)
        {
#pragma unroll
          for (int i = 0; i < warp_m * warp_n * thread_count; i++) { reg[i] *= alpha; }
        }

        template <int ldc> __device__ void store(void *ptr, int warp_row, int warp_col, const WarpRegisterMapping &wrm)
        {
          // This method is only used for the mobius preconditioner where shuffle_t = half.
          static_assert(std::is_same_v<shuffle_t, half> == true,
                        "This method should only be used for mobius preconditioner.");
          static_assert(thread_n == 2, "This method should only be used for mobius preconditioner.");
          half2 *C = reinterpret_cast<half2 *>(ptr);
#pragma unroll
          for (int tm = 0; tm < thread_m; tm++) {
#pragma unroll
            for (int wn = 0; wn < warp_n; wn++) {
#pragma unroll
              for (int wm = 0; wm < warp_m; wm++) {
                int m = warp_row * mma_m + wm * inst_m + (wrm.group_id + tm * 8);
                int n = warp_col * mma_n + wn * inst_n + (wrm.thread_id_in_group * 2);
                C[(m * ldc + n) / 2] = __floats2half2_rn(reg[(wn * warp_m + wm) * thread_count + (tm * thread_n + 0)],
                                                         reg[(wn * warp_m + wm) * thread_count + (tm * thread_n + 1)]);
              }
            }
          }
        }

        template <class F> __device__ inline void abs_max(F &max)
        {
#pragma unroll
          for (int i = 0; i < warp_m * warp_n * thread_count; i++) { max = fmax(max, fabsf(reg[i])); }
        }
      };

      static __device__ void mma(const OperandA &op_a, const OperandB &op_b, OperandC &op_c)
      {
#pragma unroll
        for (int wm = 0; wm < warp_m; wm++) {
#pragma unroll
          for (int wn = 0; wn < warp_n; wn++) {

            int a_offset = wm * OperandA::thread_count;
            int b_offset = wn * OperandB::thread_count;
            int c_offset = (wn * warp_m + wm) * OperandC::thread_count;

            mma::mma_instruction_t<inst_m, inst_n, inst_k, shuffle_t, float> mma_instruction;

            if (use_intermediate_accumulator()) {
              float acc[OperandC::thread_count];
#pragma unroll
              for (int c = 0; c < OperandC::thread_count; c++) { acc[c] = 0; }

              mma_instruction(acc, &op_a.big[a_offset], &op_b.big[b_offset]);
              mma_instruction(acc, &op_a.big[a_offset], &op_b.small[b_offset]);
              mma_instruction(acc, &op_a.small[a_offset], &op_b.big[b_offset]);

#pragma unroll
              for (int c = 0; c < OperandC::thread_count; c++) { op_c.reg[c_offset + c] += acc[c]; }
            } else {
              mma_instruction(&op_c.reg[c_offset], &op_a.big[a_offset], &op_b.big[b_offset]);
              mma_instruction(&op_c.reg[c_offset], &op_a.big[a_offset], &op_b.small[b_offset]);
              mma_instruction(&op_c.reg[c_offset], &op_a.small[a_offset], &op_b.big[b_offset]);
            }
          }
        }
      }

      template <int M, int N, int ldc, bool dagger, class gmem_op_t, class op_t>
      static inline __device__ void store_complex(int m_offset, int n_offset, const WarpRegisterMapping &wrm,
                                                  gmem_op_t &cc, const OperandC &op_c_real, const OperandC &op_c_imag,
                                                  op_t op)
      {
        using store_t = typename gmem_op_t::store_type;
        using complex_t = complex<store_t>;

        auto *C = reinterpret_cast<complex_t *>(cc.data());

        constexpr int thread_m = OperandC::thread_m;
        constexpr int thread_n = OperandC::thread_n;
        constexpr int thread_count = OperandC::thread_count;

        constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));
        if constexpr (dagger) {
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
                  if (!check_bounds || (m < N && n < M)) {
                    int reg_index = (wn * warp_m + wm) * thread_count + tm * thread_n + tn;
                    if constexpr (gmem_op_t::fixed) {
                      auto scale = cc.get_scale();
                      complex_t out = {f2i_round<store_t>(scale * op_c_real.reg[reg_index]),
                                       f2i_round<store_t>(-scale * op_c_imag.reg[reg_index])};
                      op(&C[n * ldc + m], out);
                    } else {
                      complex_t out = {op_c_real.reg[reg_index], -op_c_imag.reg[reg_index]};
                      op(&C[n * ldc + m], out);
                    }
                  }
                }
              }
            }
          }
        } else {
#pragma unroll
          for (int tm = 0; tm < thread_m; tm++) {
#pragma unroll
            for (int wn = 0; wn < warp_n; wn++) {
#pragma unroll
              for (int wm = 0; wm < warp_m; wm++) {
                static_assert(thread_n == 2, "thread_n == 2");
                int m = m_offset + wm * inst_m + (wrm.group_id + tm * 8);
                int n = n_offset + wn * inst_n + (wrm.thread_id_in_group * 2);
                if (!check_bounds || (m < M && n < N)) {
                  using vector_t = typename VectorType<store_t, 4>::type; // array<store_t, 4>;
                  int reg_index = (wn * warp_m + wm) * thread_count + tm * thread_n;
                  if constexpr (gmem_op_t::fixed) {
                    auto scale = cc.get_scale();
                    vector_t out = {f2i_round<store_t>(scale * op_c_real.reg[reg_index + 0]),
                                    f2i_round<store_t>(scale * op_c_imag.reg[reg_index + 0]),
                                    f2i_round<store_t>(scale * op_c_real.reg[reg_index + 1]),
                                    f2i_round<store_t>(scale * op_c_imag.reg[reg_index + 1])};
                    op(reinterpret_cast<vector_t *>(&C[m * ldc + n]), out);
                  } else {
                    vector_t out = {op_c_real.reg[reg_index + 0], op_c_imag.reg[reg_index + 0],
                                    op_c_real.reg[reg_index + 1], op_c_imag.reg[reg_index + 1]};
                    op(reinterpret_cast<vector_t *>(&C[m * ldc + n]), out);
                  }
                }
              }
            }
          }
        }
      }
    };

  } // namespace smma

} // namespace quda
