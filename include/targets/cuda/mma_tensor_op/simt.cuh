#pragma once

#include <tune_key.h>
#include <uint_to_char.h>

namespace quda
{

  namespace simt
  {

    template <class T, int inst_m_, int inst_n_, int warp_m_, int warp_n_> struct simt_t {

      static constexpr int warp_m = warp_m_; // tiling in m
      static constexpr int warp_n = warp_n_; // tiling in n

      static constexpr int inst_m = inst_m_;
      static constexpr int inst_n = inst_n_;
      static constexpr int inst_k = 1;

      static constexpr int MMA_M = inst_m * warp_m;
      static constexpr int MMA_N = inst_n * warp_n;
      static constexpr int MMA_K = inst_k;

      static constexpr int mma_m = MMA_M;
      static constexpr int mma_n = MMA_N;
      static constexpr int mma_k = MMA_K;

      static constexpr int warp_size = 32;

      static_assert(inst_m * inst_n == warp_size, "inst_m * inst_n == warp_size");

      using store_t = T;
      using input_t = T;

      using compute_t = T;
      using load_t = T;

      static std::string get_type_name()
      {
        char s[TuneKey::aux_n] = ",simt,m";
        i32toa(s + strlen(s), MMA_M);
        strcat(s, "n");
        i32toa(s + strlen(s), MMA_N);
        strcat(s, "k");
        i32toa(s + strlen(s), MMA_K);
        return s;
      }

      static __device__ __host__ constexpr int inline pad_size(int) { return 0; }

      struct WarpRegisterMapping {

        int warp_id;
        int lane_id;
        int idx_m;
        int idx_n;

        __device__ WarpRegisterMapping(int thread_id) :
          warp_id(thread_id / 32), lane_id(thread_id & 31), idx_m(lane_id % inst_m), idx_n(lane_id / inst_m)
        {
        }
      };

      struct OperandA {

        store_t reg[warp_m];

        __device__ inline void negate()
        {
#pragma unroll
          for (int i = 0; i < warp_m; i++) { reg[i] = -reg[i]; }
        }

        template <class smem_obj_t>
        __device__ inline void load(const smem_obj_t &smem_obj, int tile_k, int tile_m, const WarpRegisterMapping &wrm)
        { // Assuming col major smem layout

          input_t *A = reinterpret_cast<input_t *>(smem_obj.ptr);

#pragma unroll
          for (int wm = 0; wm < warp_m; wm++) {
            int k = tile_k * mma_k;
            int m = tile_m * mma_m + wrm.idx_m * warp_m + wm;
            reg[wm] = A[k * smem_obj_t::ldn + m];
          }
        }
      };

      struct OperandB {

        store_t reg[warp_n];

        template <class smem_obj_t>
        __device__ inline void load(const smem_obj_t &smem_obj, int tile_k, int tile_n, const WarpRegisterMapping &wrm)
        { // Assuming row major smem layout

          input_t *B = reinterpret_cast<input_t *>(smem_obj.ptr);

#pragma unroll
          for (int wn = 0; wn < warp_n; wn++) {
            int k = tile_k * mma_k;
            int n = tile_n * mma_n + wrm.idx_n * warp_n + wn;
            reg[wn] = B[k * smem_obj_t::ldn + n];
          }
        }
      };

      struct OperandC {

        using reg_type = T;
        reg_type reg[warp_m * warp_n];

        __device__ inline void zero()
        {
#pragma unroll
          for (int i = 0; i < warp_m * warp_n; i++) { reg[i] = 0; }
        }

        __device__ inline OperandC() { zero(); }

        __device__ inline void ax(T alpha)
        {
#pragma unroll
          for (int i = 0; i < warp_m * warp_n; i++) { reg[i] *= alpha; }
        }

        template <int ldc> __device__ void store(void *ptr, int m_offset, int n_offset, const WarpRegisterMapping &wrm)
        {
          reg_type *C = reinterpret_cast<reg_type *>(ptr);
#pragma unroll
          for (int wn = 0; wn < warp_n; wn++) {
#pragma unroll
            for (int wm = 0; wm < warp_m; wm++) {
              int m = m_offset + wrm.idx_m * warp_m + wm;
              int n = n_offset + wrm.idx_n * warp_n + wn;
              C[m * ldc + n] = reg[wn * warp_m + wm];
            }
          }
        }

        template <class F> __device__ inline void abs_max(F &max)
        {
#pragma unroll
          for (int i = 0; i < warp_m * warp_n; i++) { max = fmax(max, fabsf(reg[i])); }
        }
      };

      static __device__ void mma(const OperandA &op_a, const OperandB &op_b, OperandC &op_c)
      {
#pragma unroll
        for (int wm = 0; wm < warp_m; wm++) {
#pragma unroll
          for (int wn = 0; wn < warp_n; wn++) { op_c.reg[wn * warp_m + wm] += op_a.reg[wm] * op_b.reg[wn]; }
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

        constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));
#pragma unroll
        for (int wn = 0; wn < warp_n; wn++) {
#pragma unroll
          for (int wm = 0; wm < warp_m; wm++) {
            int m = m_offset + wrm.idx_m * warp_m + wm;
            int n = n_offset + wrm.idx_n * warp_n + wn;
            if constexpr (dagger) {
              if (!check_bounds || (m < N && n < M)) {
                if constexpr (gmem_op_t::fixed) {
                  auto scale = cc.get_scale();
                  complex_t out = {f2i_round<store_t>(scale * op_c_real.reg[wn * warp_m + wm]),
                                   f2i_round<store_t>(-scale * op_c_imag.reg[wn * warp_m + wm])};
                  op(&C[n * ldc + m], out);
                } else {
                  complex_t out = {op_c_real.reg[wn * warp_m + wm], -op_c_imag.reg[wn * warp_m + wm]};
                  op(&C[n * ldc + m], out);
                }
              }
            } else {
              if (!check_bounds || (m < M && n < N)) {
                if constexpr (gmem_op_t::fixed) {
                  auto scale = cc.get_scale();
                  complex_t out = {f2i_round<store_t>(scale * op_c_real.reg[wn * warp_m + wm]),
                                   f2i_round<store_t>(scale * op_c_imag.reg[wn * warp_m + wm])};
                  op(&C[m * ldc + n], out);
                } else {
                  complex_t out = {op_c_real.reg[wn * warp_m + wm], op_c_imag.reg[wn * warp_m + wm]};
                  op(&C[m * ldc + n], out);
                }
              }
            }
          }
        }
      }
    };

  } // namespace simt

} // namespace quda
