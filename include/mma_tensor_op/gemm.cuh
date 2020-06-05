#pragma once

// XXX: Load different header for different archs.
#include <mma_tensor_op/hmma_m16n16k16_sm70.cuh>

#define USE_GMEM_MMA_PIPELINING

namespace quda
{
  namespace mma
  {
    constexpr int shared_memory_bytes(int bM, int bN, int bK)
    {
      return (bM + pad_size(bM) + bN + pad_size(bN)) * bK * 2 * sizeof(half);
    }

    template <class T, int M_, int N_, int ldm_, int ldn_> struct SharedMemoryObject {

      static constexpr int M = M_;
      static constexpr int N = N_;
      static constexpr int ldm = ldm_;
      static constexpr int ldn = ldn_;

      T *ptr;

      __device__ inline SharedMemoryObject(T *ptr_) : ptr(ptr_) { }

      __device__ inline T &operator()(int i, int j) { return ptr[i * ldm + j * ldn]; }

      __device__ inline const T &operator()(int i, int j) const { return ptr[i * ldm + j * ldn]; }

      template <class VecType> __device__ inline void vector_load(int i, int j, VecType vec)
      {
        VecType *ptr_ = reinterpret_cast<VecType *>(ptr);
        constexpr int vector_length = sizeof(VecType) / sizeof(T);
        ptr_[(i * ldm + j * ldn) / vector_length] = vec;
      }
#if 0
      template <class VecType> __device__ inline auto vector_get(int i, int j)
      {
        VecType *ptr_ = reinterpret_cast<VecType *>(ptr);
        constexpr int vector_length = sizeof(VecType) / sizeof(T);
        return ptr_[(i * ldm + j * ldn) / vector_length];
      }
#endif
    };

    enum class EpilogueType { COMPUTE_MAX_ONLY, VECTOR_STORE, ATOMIC_STORE_DAGGER_NO, ATOMIC_STORE_DAGGER_YES };

    template <int M, int N, int ldm, int ldn, class T> __device__ inline auto make_smem_obj(T *ptr_)
    {
      return SharedMemoryObject<T, M, N, ldm, ldn> {ptr_};
    }

    template <int M, int N, int m_stride, int n_stride, bool transpose, class SmemAccessor> struct GlobalMemoryLoader {

      static constexpr int sM = SmemAccessor::M; // This is the block M (bM)
      static constexpr int sN = SmemAccessor::N;

      static constexpr int m_stride_pack = m_stride * 2;
      static constexpr int m_dim = (sM + m_stride_pack - 1) / m_stride_pack;
      static constexpr int n_dim = (sN + n_stride - 1) / n_stride;

#ifdef USE_GMEM_MMA_PIPELINING
      half2 reg_real[m_dim * n_dim];
      half2 reg_imag[m_dim * n_dim];
#endif

      const int y;
      const int z;

      __device__ inline GlobalMemoryLoader() : y(threadIdx.y), z(threadIdx.z * 2) { }

#ifdef USE_GMEM_MMA_PIPELINING
      template <int ld, bool dagger, class GmemAccessor>
      __device__ inline void g2r(const GmemAccessor &gmem, int m_offset, int n_offset)
      {
        auto p = gmem.data();
        auto scale_inv = gmem.scale_inv;
        constexpr bool fixed = GmemAccessor::fixed;
#pragma unroll
        for (int n = 0; n < n_dim; n++) {
#pragma unroll
          for (int m = 0; m < m_dim; m++) {
            const int n_idx_blk = n * n_stride + y;
            const int m_idx_blk = m * m_stride_pack + z;

            if (m_idx_blk < sM && n_idx_blk < sN) {

              int n_idx = n_idx_blk + n_offset;
              int m_idx = m_idx_blk + m_offset;

              if (n_idx < N && m_idx < M) {

                if (transpose == dagger) {
                  auto xx = p[(m_idx + 0) * ld + n_idx];
                  auto yy = p[(m_idx + 1) * ld + n_idx];

                  if (fixed) {
                    reg_real[m * n_dim + n] = __floats2half2_rn(scale_inv * xx.real(), scale_inv * yy.real());
                    auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
                    reg_imag[m * n_dim + n] = __floats2half2_rn(scale_inv_conj * xx.imag(), scale_inv_conj * yy.imag());
                  } else {
                    reg_real[m * n_dim + n] = __floats2half2_rn(+xx.real(), +yy.real());
                    reg_imag[m * n_dim + n]
                      = __floats2half2_rn(dagger ? -xx.imag() : +xx.imag(), dagger ? -yy.imag() : +yy.imag());
                  }
                } else {
                  using store_type = typename GmemAccessor::store_type;
                  using store_vector_type = typename VectorType<store_type, 4>::type;
                  store_vector_type v = *reinterpret_cast<store_vector_type *>(&p[n_idx * ld + m_idx]);

                  if (fixed) {
                    reg_real[m * n_dim + n] = __floats2half2_rn(scale_inv * v.x, scale_inv * v.z);
                    auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
                    reg_imag[m * n_dim + n] = __floats2half2_rn(scale_inv_conj * v.y, scale_inv_conj * v.w);
                  } else {
                    reg_real[m * n_dim + n] = __floats2half2_rn(+v.x, +v.z);
                    reg_imag[m * n_dim + n] = __floats2half2_rn(dagger ? -v.y : +v.y, dagger ? -v.w : +v.w);
                  }
                }

              } else {

                reg_real[m * n_dim + n] = __half2half2(0);
                reg_imag[m * n_dim + n] = __half2half2(0);
              }
            }
          }
        }
      }

      template <class SmemObj> __device__ inline void r2s(SmemObj &smem_real, SmemObj &smem_imag)
      {
#pragma unroll
        for (int n = 0; n < n_dim; n++) {
#pragma unroll
          for (int m = 0; m < m_dim; m++) {
            const int n_idx = n * n_stride + y;
            const int m_idx = m * m_stride_pack + z;
            if (m_idx < sM && n_idx < sN) {
              smem_real.vector_load(m_idx, n_idx, reg_real[m * n_dim + n]);
              smem_imag.vector_load(m_idx, n_idx, reg_imag[m * n_dim + n]);
            }
          }
        }
      }
#else
      template <int ld, bool dagger, class SmemObj, class GmemAccessor>
      __device__ inline void g2s(SmemObj &smem_real, SmemObj &smem_imag, const GmemAccessor &gmem, int m_offset,
                                 int n_offset)
      {
        auto p = gmem.data();
        auto scale_inv = gmem.scale_inv;
        constexpr bool fixed = GmemAccessor::fixed;
#pragma unroll
        for (int n = 0; n < n_dim; n++) {
#pragma unroll
          for (int m = 0; m < m_dim; m++) {
            const int n_idx_smem = n * n_stride + y;
            const int m_idx_smem = m * m_stride_pack + z;

            if (m_idx_smem < sM && n_idx_smem < sN) {

              const int n_idx = n_idx_smem + n_offset;
              const int m_idx = m_idx_smem + m_offset;

              if (n_idx < N && m_idx < M) {

                if (transpose == dagger) {

                  auto xx = p[(m_idx + 0) * ld + n_idx];
                  auto yy = p[(m_idx + 1) * ld + n_idx];

                  if (fixed) {
                    smem_real.vector_load(m_idx_smem, n_idx_smem,
                                          __floats2half2_rn(scale_inv * xx.real(), scale_inv * yy.real()));
                    auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
                    smem_imag.vector_load(m_idx_smem, n_idx_smem,
                                          __floats2half2_rn(scale_inv_conj * xx.imag(), scale_inv_conj * yy.imag()));
                  } else {
                    smem_real.vector_load(m_idx_smem, n_idx_smem, __floats2half2_rn(+xx.real(), +yy.real()));
                    smem_imag.vector_load(
                      m_idx_smem, n_idx_smem,
                      __floats2half2_rn(dagger ? -xx.imag() : +xx.imag(), dagger ? -yy.imag() : +yy.imag()));
                  }

                } else {

                  using store_type = typename GmemAccessor::store_type;
                  using store_vector_type = typename VectorType<store_type, 4>::type;
                  store_vector_type v = *reinterpret_cast<store_vector_type *>(&p[n_idx * ld + m_idx]);

                  if (fixed) {
                    smem_real.vector_load(m_idx_smem, n_idx_smem, __floats2half2_rn(scale_inv * v.x, scale_inv * v.z));
                    auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
                    smem_imag.vector_load(m_idx_smem, n_idx_smem,
                                          __floats2half2_rn(scale_inv_conj * v.y, scale_inv_conj * v.w));
                  } else {
                    smem_real.vector_load(m_idx_smem, n_idx_smem, __floats2half2_rn(+v.x, +v.z));
                    smem_imag.vector_load(m_idx_smem, n_idx_smem,
                                          __floats2half2_rn(dagger ? -v.y : +v.y, dagger ? -v.w : +v.w));
                  }
                }

              } else {

                smem_real.vector_load(m_idx_smem, n_idx_smem, __half2half2(0));
                smem_imag.vector_load(m_idx_smem, n_idx_smem, __half2half2(0));
              }
            }
          }
        }
      }
#endif
    };

    template <class OperandC, int warp_cycle, int tile_col_dim> struct MmaOp {

      static constexpr int size = warp_cycle;

      OperandC op_c_real[warp_cycle];
      OperandC op_c_imag[warp_cycle];

      WarpRegisterMapping wrm;

      __device__ inline MmaOp(int thread_id) : wrm(thread_id) { }

      __device__ inline void zero()
      {
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {
          op_c_real[c].zero();
          op_c_imag[c].zero();
        }
      }

      __device__ inline void ax(float alpha)
      {
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {
          op_c_real[c].ax(alpha);
          op_c_imag[c].ax(alpha);
        }
      }

      template <int tile_acc_dim, class SmemObjA, class SmemObjB>
      __device__ inline void mma(const SmemObjA &smem_obj_a_real, const SmemObjA &smem_obj_a_imag,
                                 const SmemObjB &smem_obj_b_real, const SmemObjB &smem_obj_b_imag)
      {

#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {

          // The logical warp assigned to each part of the matrix.
          const int logical_warp_index = wrm.warp_id * warp_cycle + c;
          const int warp_row = logical_warp_index / tile_col_dim;
          const int warp_col = logical_warp_index - warp_row * tile_col_dim;

#pragma unroll
          for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {

            MmaOperandA op_a_real;
            op_a_real.load(smem_obj_a_real, tile_k, warp_row, wrm);
            MmaOperandA op_a_imag;
            op_a_imag.load(smem_obj_a_imag, tile_k, warp_row, wrm);

            MmaOperandB op_b_real;
            op_b_real.load(smem_obj_b_real, tile_k, warp_col, wrm);
            MmaOperandB op_b_imag;
            op_b_imag.load(smem_obj_b_imag, tile_k, warp_col, wrm);

            gemm(op_a_real, op_b_real, op_c_real[c]);
            gemm(op_a_imag, op_b_real, op_c_imag[c]);
            gemm(op_a_real, op_b_imag, op_c_imag[c]);
            // negate op_imag
            op_a_imag.negate();
            gemm(op_a_imag, op_b_imag, op_c_real[c]);
          }
        }
      }

      template <int ldc, class C> __device__ inline void store(C &c_accessor, int m_offset, int n_offset)
      {
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {

          const int logical_warp_index = wrm.warp_id * warp_cycle + c;
          const int warp_row = logical_warp_index / tile_col_dim;
          const int warp_col = logical_warp_index - warp_row * tile_col_dim;

          const int warp_m_offset = warp_row * WMMA_M + m_offset;
          const int warp_n_offset = warp_col * WMMA_N + n_offset;

          store_complex<ldc>(warp_m_offset, warp_n_offset, wrm, c_accessor, op_c_real[c], op_c_imag[c]);
        }
      }

      template <int ldc, bool dagger, class C>
      __device__ inline void store_atomic(C &c_accessor, int m_offset, int n_offset)
      {
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {

          const int logical_warp_index = wrm.warp_id * warp_cycle + c;
          const int warp_row = logical_warp_index / tile_col_dim;
          const int warp_col = logical_warp_index - warp_row * tile_col_dim;

          const int warp_m_offset = warp_row * WMMA_M + m_offset;
          const int warp_n_offset = warp_col * WMMA_N + n_offset;

          store_complex_atomic<ldc, dagger>(warp_m_offset, warp_n_offset, wrm, c_accessor, op_c_real[c], op_c_imag[c]);
        }
      }

      __device__ inline void abs_max(float &max)
      {
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {
          op_c_real[c].abs_max(max);
          op_c_imag[c].abs_max(max);
        }
      }
    };

    template <int M_, int N_, int K_, int lda_, int ldb_, int ldc_, int bM_, int bN_, int bK_, int block_y, int block_z>
    struct MmaConfig {

      static constexpr int M = M_;
      static constexpr int N = N_;
      static constexpr int K = K_;
      static constexpr int lda = lda_;
      static constexpr int ldb = ldb_;
      static constexpr int ldc = ldc_;
      static constexpr int bM = bM_;
      static constexpr int bN = bN_;
      static constexpr int bK = bK_;

      static constexpr int tile_row_dim = bM / WMMA_M; // number of tiles in the column dimension
      static constexpr int tile_col_dim = bN / WMMA_N; // number of tiles in the row dimension
      static constexpr int tile_acc_dim = bK / WMMA_K; // number of tiles in the accumulate dimension

      static constexpr int smem_lda = bM + pad_size(bM);
      static constexpr int smem_ldb = bN + pad_size(bN);

      static constexpr int n_row = block_z;
      static constexpr int n_col = block_y;

      static constexpr int total_warp = n_row * n_col / warp_size;

      static constexpr int total_tile = tile_row_dim * tile_col_dim;
      static constexpr int warp_cycle = total_tile / total_warp;

      static constexpr bool a_transpose = false;
      static constexpr bool b_transpose = true;

#ifdef USE_FP16_HMMA_ACCUMULATE
      using accumuate_reg_type = half;
#else
      using accumuate_reg_type = float;
#endif
      static_assert(bM % WMMA_M == 0, "bM must be divisible by WMMA_M.");
      static_assert(bN % WMMA_N == 0, "bN must be divisible by WMMA_N.");
      static_assert(bK % WMMA_K == 0, "bK must be divisible by WMMA_K.");

      static_assert((tile_row_dim * tile_col_dim) % total_warp == 0,
                    "Total number of tiles should be divisible by the number of warps.");

      using SmemObjA = SharedMemoryObject<half, bM, bK, 1, smem_lda>;
      using SmemObjB = SharedMemoryObject<half, bN, bK, 1, smem_ldb>;

      SmemObjA smem_obj_a_real;
      SmemObjA smem_obj_a_imag;
      SmemObjB smem_obj_b_real;
      SmemObjB smem_obj_b_imag;

      MmaOp<MmaOperandC<accumuate_reg_type>, warp_cycle, tile_col_dim> mma_op; // (thread_id);

      GlobalMemoryLoader<M, K, n_row, n_col, a_transpose, SmemObjA> a_loader;

      GlobalMemoryLoader<N, K, n_row, n_col, b_transpose, SmemObjB> b_loader;

      __device__ inline MmaConfig(half *smem_ptr) :
        smem_obj_a_real(smem_ptr),
        smem_obj_a_imag(smem_obj_a_real.ptr + smem_lda * bK),
        smem_obj_b_real(smem_obj_a_imag.ptr + smem_lda * bK),
        smem_obj_b_imag(smem_obj_b_real.ptr + smem_ldb * bK),
        mma_op((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x),
        a_loader(),
        b_loader()
      {
      }

      template <bool a_dagger, bool b_dagger, bool compute_max_only, class A, class B, class C>
      __device__ inline float perform_mma(const A &a, const B &b, C &c, int m_offset, int n_offset)
      {
        float max = 0;

#ifdef USE_GMEM_MMA_PIPELINING
        a_loader.g2r<lda, a_dagger>(a, m_offset, 0);
        a_loader.r2s(smem_obj_a_real, smem_obj_a_imag);

        b_loader.g2r<ldb, b_dagger>(b, n_offset, 0);
        b_loader.r2s(smem_obj_b_real, smem_obj_b_imag);

        __syncthreads();
#endif

#pragma unroll 1
        for (int bk = 0; bk < K; bk += bK) {

#ifdef USE_GMEM_MMA_PIPELINING
          if (bk + bK < K) {
            a_loader.g2r<lda, a_dagger>(a, m_offset, bk + bK);
            b_loader.g2r<ldb, b_dagger>(b, n_offset, bk + bK);
          }
#else
          __syncthreads();
          a_loader.g2s<lda, a_dagger>(smem_obj_a_real, smem_obj_a_imag, a, m_offset, bk);
          b_loader.g2s<ldb, b_dagger>(smem_obj_b_real, smem_obj_b_imag, b, n_offset, bk);
          __syncthreads();
#endif

          mma_op.mma<tile_acc_dim>(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag);

#ifdef USE_GMEM_MMA_PIPELINING
          if (bk + bK < K) {
            __syncthreads();

            a_loader.r2s(smem_obj_a_real, smem_obj_a_imag);
            b_loader.r2s(smem_obj_b_real, smem_obj_b_imag);

            __syncthreads();
          }
#endif
        }

        // wrap up!
        if (compute_max_only) {
          mma_op.abs_max(max);
        } else {
          mma_op.store<ldc>(c, m_offset, n_offset);
        }

        return max;
      }
    };

  } // namespace mma

} // namespace quda
