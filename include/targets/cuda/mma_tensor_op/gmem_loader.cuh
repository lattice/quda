#pragma once

#include <pipeline.cuh>
#include <cub/block/block_reduce.cuh>

namespace quda
{
  namespace mma
  {

    /**
      @brief Defining how many elements/atoms are there in type T ...
     */
    template <class T> struct batch_multiple {
    };

    /**
      @brief ... e.g. there are 2 half's in a half2
     */
    template <> struct batch_multiple<half2> {
      static constexpr int value = 2;
    };

    template <> struct batch_multiple<float> {
      static constexpr int value = 1;
    };

    inline __device__ void zero(float2 &reg_real, float2 &reg_imag)
    {
      reg_real.x = 0;
      reg_real.y = 0;
      reg_imag.x = 0;
      reg_imag.y = 0;
    }

    inline __device__ void zero(half2 &reg_real, half2 &reg_imag)
    {
      reg_real = __half2half2(0);
      reg_imag = __half2half2(0);
    }

    inline __device__ void zero(float &reg_real, float &reg_imag)
    {
      reg_real = 0;
      reg_imag = 0;
    }

    inline __device__ float abs_max(float a, float max) { return fmaxf(fabsf(a), max); }

    inline __device__ float abs_max(float2 a, float max) { return fmaxf(fabsf(a.y), fmaxf(fabsf(a.x), max)); }

    template <class T, int batch> struct batch_load_t {
    };

    template <> struct batch_load_t<complex<float>, 1> {
      static void __device__ load(complex<float> v[1], complex<float> *ptr) { v[0] = *ptr; }
    };

    template <> struct batch_load_t<complex<float>, 2> {
      static void __device__ load(complex<float> v[2], complex<float> *ptr)
      {
        float4 l = *reinterpret_cast<float4 *>(ptr);
        v[0].real(l.x);
        v[0].imag(l.y);
        v[1].real(l.z);
        v[1].imag(l.w);
      }
    };

    template <> struct batch_load_t<complex<short>, 1> {
      static void __device__ load(complex<short> v[1], complex<short> *ptr) { v[0] = *ptr; }
    };

    template <> struct batch_load_t<complex<short>, 2> {
      static void __device__ load(complex<short> v[2], complex<short> *ptr)
      {
        short4 l = *reinterpret_cast<short4 *>(ptr);
        v[0].real(l.x);
        v[0].imag(l.y);
        v[1].real(l.z);
        v[1].imag(l.w);
      }
    };

    template <> struct batch_load_t<complex<short>, 4> {
      static void __device__ load(complex<short> v[4], complex<short> *ptr)
      {
        short8 l = *reinterpret_cast<short8 *>(ptr);
        v[0].real(l.x.x);
        v[0].imag(l.x.y);
        v[1].real(l.x.z);
        v[1].imag(l.x.w);
        v[2].real(l.y.x);
        v[2].imag(l.y.y);
        v[3].real(l.y.z);
        v[3].imag(l.y.w);
      }
    };

    template <class T, int batch> struct make_vector_t {
    };

    template <> struct make_vector_t<float, 1> {
      static auto __device__ get(float v[]) { return v[0]; }
    };

    template <> struct make_vector_t<float, 2> {
      static auto __device__ get(float v[])
      {
        float2 ret_value;
        ret_value.x = v[0];
        ret_value.y = v[1];
        return ret_value;
      }
    };

    template <> struct make_vector_t<float, 4> {
      static auto __device__ get(float v[])
      {
        float4 ret_value;
        ret_value.x = v[0];
        ret_value.y = v[1];
        ret_value.z = v[2];
        ret_value.w = v[3];
        return ret_value;
      }
    };

    template <> struct make_vector_t<half2, 1> {
      static auto __device__ get(half2 v[]) { return v[0]; }
    };

    template <> struct make_vector_t<half2, 2> {
      static auto __device__ get(half2 v[])
      {
        uint2 ret_value;
        ret_value.x = reinterpret_cast<unsigned int *>(v)[0];
        ret_value.y = reinterpret_cast<unsigned int *>(v)[1];
        return ret_value;
      }
    };

    /**
      @brief Load from global memory and store data in registers.
     */
    template <bool x, bool fixed, bool dagger, int ld, int batch, class T>
    inline __device__ void convert_x(float2 reg_real[batch], float2 reg_imag[batch], complex<T> *p, int m_idx,
                                     int n_idx, float scale_inv)
    {
      if constexpr (x) {
        complex<T> vx[batch];
        complex<T> vy[batch];
        batch_load_t<complex<T>, batch>::load(vx, &p[(m_idx + 0) * ld + n_idx]);
        batch_load_t<complex<T>, batch>::load(vy, &p[(m_idx + 1) * ld + n_idx]);

#pragma unroll
        for (int b = 0; b < batch; b++) {
          if constexpr (fixed) {
            reg_real[b].x = scale_inv * vx[b].real();
            reg_real[b].y = scale_inv * vy[b].real();
            auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
            reg_imag[b].x = scale_inv_conj * vx[b].imag();
            reg_imag[b].y = scale_inv_conj * vy[b].imag();
          } else {
            reg_real[b].x = +vx[b].real();
            reg_real[b].y = +vy[b].real();
            reg_imag[b].x = dagger ? -vx[b].imag() : +vx[b].imag();
            reg_imag[b].y = dagger ? -vy[b].imag() : +vy[b].imag();
          }
        }
      } else {
        complex<T> v[batch * 2];
        batch_load_t<complex<T>, batch * 2>::load(v, &p[n_idx * ld + m_idx]);

#pragma unroll
        for (int b = 0; b < batch; b++) {
          if constexpr (fixed) {
            reg_real[b].x = scale_inv * v[b * 2 + 0].real();
            reg_real[b].y = scale_inv * v[b * 2 + 1].real();
            auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
            reg_imag[b].x = scale_inv_conj * v[b * 2 + 0].imag();
            reg_imag[b].y = scale_inv_conj * v[b * 2 + 1].imag();
          } else {
            reg_real[b].x = +v[b * 2 + 0].real();
            reg_real[b].y = +v[b * 2 + 1].real();
            reg_imag[b].x = dagger ? -v[b * 2 + 0].imag() : +v[b * 2 + 0].imag();
            reg_imag[b].y = dagger ? -v[b * 2 + 1].imag() : +v[b * 2 + 1].imag();
          }
        }
      }
    }

    /**
      @brief Load from global memory and store data in registers.
     */
    template <bool x, bool fixed, bool dagger, int ld, int batch, class T>
    inline __device__ void convert_x(half2 reg_real[batch], half2 reg_imag[batch], complex<T> *p, int m_idx, int n_idx,
                                     float scale_inv)
    {
      if constexpr (x) {
        complex<T> vx[batch];
        complex<T> vy[batch];
        batch_load_t<complex<T>, batch>::load(vx, &p[(m_idx + 0) * ld + n_idx]);
        batch_load_t<complex<T>, batch>::load(vy, &p[(m_idx + 1) * ld + n_idx]);

#pragma unroll
        for (int b = 0; b < batch; b++) {
          if constexpr (fixed) {
            reg_real[b] = __floats2half2_rn(scale_inv * vx[b].real(), scale_inv * vy[b].real());
            auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
            reg_imag[b] = __floats2half2_rn(scale_inv_conj * vx[b].imag(), scale_inv_conj * vy[b].imag());
          } else {
            reg_real[b] = __floats2half2_rn(+vx[b].real(), +vy[b].real());
            reg_imag[b]
              = __floats2half2_rn(dagger ? -vx[b].imag() : +vx[b].imag(), dagger ? -vy[b].imag() : +vy[b].imag());
          }
        }
      } else {
        complex<T> v[batch * 2];
        batch_load_t<complex<T>, batch * 2>::load(v, &p[n_idx * ld + m_idx]);

#pragma unroll
        for (int b = 0; b < batch; b++) {
          if constexpr (fixed) {
            reg_real[b] = __floats2half2_rn(scale_inv * v[b * 2].real(), scale_inv * v[b * 2 + 1].real());
            auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
            reg_imag[b] = __floats2half2_rn(scale_inv_conj * v[b * 2].imag(), scale_inv_conj * v[b * 2 + 1].imag());
          } else {
            reg_real[b] = __floats2half2_rn(+v[b * 2].real(), +v[b * 2 + 1].real());
            reg_imag[b] = __floats2half2_rn(dagger ? -v[b * 2].imag() : +v[b * 2].imag(),
                                            dagger ? -v[b * 2 + 1].imag() : +v[b * 2 + 1].imag());
          }
        }
      }
    }

    /**
      @brief Load from global memory and store data in registers.
     */
    template <bool x, bool fixed, bool dagger, int ld, int batch, class T>
    inline __device__ void convert_x_rescale(half2 reg_real[batch], half2 reg_imag[batch], complex<T> *p, int m_idx,
                                             int n_idx, float scale_inv, float rescale)
    {
      if constexpr (x) {
        complex<T> vx[batch];
        complex<T> vy[batch];
        batch_load_t<complex<T>, batch>::load(vx, &p[(m_idx + 0) * ld + n_idx]);
        batch_load_t<complex<T>, batch>::load(vy, &p[(m_idx + 1) * ld + n_idx]);

#pragma unroll
        for (int b = 0; b < batch; b++) {
          if constexpr (fixed) {
            float scale_inv_rescale = scale_inv * rescale;
            reg_real[b] = __floats2half2_rn(scale_inv_rescale * vx[b].real(), scale_inv_rescale * vy[b].real());
            auto scale_inv_conj = dagger ? -scale_inv_rescale : scale_inv_rescale;
            reg_imag[b] = __floats2half2_rn(scale_inv_conj * vx[b].imag(), scale_inv_conj * vy[b].imag());
          } else {
            reg_real[b] = __floats2half2_rn(+vx[b].real() * rescale, +vy[b].real() * rescale);
            reg_imag[b] = __floats2half2_rn((dagger ? -vx[b].imag() : +vx[b].imag()) * rescale,
                                            (dagger ? -vy[b].imag() : +vy[b].imag()) * rescale);
          }
        }
      } else {
        complex<T> v[batch * 2];
        batch_load_t<complex<T>, batch * 2>::load(v, &p[n_idx * ld + m_idx]);

#pragma unroll
        for (int b = 0; b < batch; b++) {
          if constexpr (fixed) {
            float scale_inv_rescale = scale_inv * rescale;
            reg_real[b] = __floats2half2_rn(scale_inv_rescale * v[b * 2].real(), scale_inv_rescale * v[b * 2 + 1].real());
            auto scale_inv_conj = dagger ? -scale_inv_rescale : scale_inv_rescale;
            reg_imag[b] = __floats2half2_rn(scale_inv_conj * v[b * 2].imag(), scale_inv_conj * v[b * 2 + 1].imag());
          } else {
            reg_real[b] = __floats2half2_rn(+v[b * 2].real() * rescale, +v[b * 2 + 1].real() * rescale);
            reg_imag[b] = __floats2half2_rn((dagger ? -v[b * 2].imag() : +v[b * 2].imag()) * rescale,
                                            (dagger ? -v[b * 2 + 1].imag() : +v[b * 2 + 1].imag()) * rescale);
          }
        }
      }
    }

    /**
      @brief Load from global memory and store data in registers.
     */
    template <bool x, bool fixed, bool dagger, int ld, int batch, class T>
    inline __device__ void convert_x(float reg_real[batch], float reg_imag[batch], complex<T> *p, int m_idx, int n_idx,
                                     float scale_inv)
    {
      complex<T> v[batch];
      if constexpr (x) {
        batch_load_t<complex<T>, batch>::load(v, &p[m_idx * ld + n_idx]);
#pragma unroll
        for (int b = 0; b < batch; b++) {
          // auto xx = p[m_idx * ld + n_idx];
          if constexpr (fixed) {
            reg_real[b] = scale_inv * v[b].real();
            auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
            reg_imag[b] = scale_inv_conj * v[b].imag();
          } else {
            reg_real[b] = v[b].real();
            reg_imag[b] = dagger ? -v[b].imag() : v[b].imag();
          }
        }
      } else {
        complex<T> v[batch];
        batch_load_t<complex<T>, batch>::load(v, &p[n_idx * ld + m_idx]);
#pragma unroll
        for (int b = 0; b < batch; b++) {
          // auto xx = p[n_idx * ld + m_idx];
          if constexpr (fixed) {
            reg_real[b] = scale_inv * v[b].real();
            auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
            reg_imag[b] = scale_inv_conj * v[b].imag();
          } else {
            reg_real[b] = v[b].real();
            reg_imag[b] = dagger ? -v[b].imag() : v[b].imag();
          }
        }
      }
    }

    /**
      @brief Load from global memory and store data in registers.
     */
    template <bool x, bool fixed, bool dagger, int ld, int batch, class T>
    inline __device__ void convert_x_rescale(float reg_real[batch], float reg_imag[batch], complex<T> *p, int m_idx, int n_idx,
                                             float scale_inv, float rescale)
    {
      complex<T> v[batch];
      scale_inv *= rescale;
      if constexpr (x) {
        batch_load_t<complex<T>, batch>::load(v, &p[m_idx * ld + n_idx]);
#pragma unroll
        for (int b = 0; b < batch; b++) {
          if constexpr (fixed) {
            reg_real[b] = scale_inv * v[b].real();
            auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
            reg_imag[b] = scale_inv_conj * v[b].imag();
          } else {
            reg_real[b] = v[b].real() * rescale;
            reg_imag[b] = (dagger ? -v[b].imag() : v[b].imag()) * rescale;
          }
        }
      } else {
        complex<T> v[batch];
        batch_load_t<complex<T>, batch>::load(v, &p[n_idx * ld + m_idx]);
#pragma unroll
        for (int b = 0; b < batch; b++) {
          if constexpr (fixed) {
            reg_real[b] = scale_inv * v[b].real();
            auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
            reg_imag[b] = scale_inv_conj * v[b].imag();
          } else {
            reg_real[b] = v[b].real() * rescale;
            reg_imag[b] = (dagger ? -v[b].imag() : v[b].imag()) * rescale;
          }
        }
      }
    }

    /**
      @brief Load from global memory and store data in registers.
     */
    template <bool x, bool fixed, bool dagger, int ld, class T>
    inline __device__ float find_abs_max(half2, complex<T> *p, int m_idx, int n_idx, float scale_inv)
    {
      float this_max = 0.0f;

      if constexpr (x) {
        auto xx = p[m_idx * ld + n_idx];
        auto yy = p[(m_idx + 1) * ld + n_idx];

        if constexpr (fixed) {
          this_max = abs_max(scale_inv * xx.real(), this_max);
          this_max = abs_max(scale_inv * xx.imag(), this_max);
          this_max = abs_max(scale_inv * yy.real(), this_max);
          this_max = abs_max(scale_inv * yy.imag(), this_max);
        } else {
          this_max = abs_max(xx.real(), this_max);
          this_max = abs_max(xx.imag(), this_max);
          this_max = abs_max(yy.real(), this_max);
          this_max = abs_max(yy.imag(), this_max);
        }
      } else {
        auto xx = p[n_idx * ld + m_idx];
        auto yy = p[n_idx * ld + m_idx + 1];

        if constexpr (fixed) {
          this_max = abs_max(scale_inv * xx.real(), this_max);
          this_max = abs_max(scale_inv * xx.imag(), this_max);
          this_max = abs_max(scale_inv * yy.real(), this_max);
          this_max = abs_max(scale_inv * yy.imag(), this_max);
        } else {
          this_max = abs_max(xx.real(), this_max);
          this_max = abs_max(xx.imag(), this_max);
          this_max = abs_max(yy.real(), this_max);
          this_max = abs_max(yy.imag(), this_max);
        }
      }

      return this_max;
    }

    /**
      @brief Load from global memory and store data in registers.
     */
    template <bool x, bool fixed, bool dagger, int ld, class T>
    inline __device__ float find_abs_max(float, complex<T> *p, int m_idx, int n_idx, float scale_inv)
    {
      float this_max = 0.0f;

      if constexpr (x) {
        auto xx = p[m_idx * ld + n_idx];

        if constexpr (fixed) {
          this_max = abs_max(scale_inv * xx.real(), this_max);
          this_max = abs_max(scale_inv * xx.imag(), this_max);
        } else {
          this_max = abs_max(xx.real(), this_max);
          this_max = abs_max(xx.imag(), this_max);
        }
      } else {
        auto xx = p[n_idx * ld + m_idx];

        if constexpr (fixed) {
          this_max = abs_max(scale_inv * xx.real(), this_max);
          this_max = abs_max(scale_inv * xx.imag(), this_max);
        } else {
          this_max = abs_max(xx.real(), this_max);
          this_max = abs_max(xx.imag(), this_max);
        }
      }

      return this_max;
    }

    template <class T> constexpr int get_mn_batch(int internal_batch, int register_dim, int block)
    {
      return ((register_dim % (internal_batch * 4) == 0 && block % (internal_batch * 4) == 0
               && sizeof(T) * internal_batch * 8 <= 16) ?
                4 :
                ((register_dim % (internal_batch * 2) == 0 && block % (internal_batch * 2) == 0
                  && sizeof(T) * internal_batch * 4 <= 16) ?
                   2 :
                   1));
    }

    /**
     * A loader object that loads data from global memory to registers (g2r), and then to shared memory (r2s)
     * M, N: the global memory matrix size, for bound check only
     * bM, bN: the shared memory matrix size
     * block_y, block_z: CTA dimension in the y and z directions
     * transpose: the global memory always assumes a column-major order, transpose = true if the destination
          shared memory is row-major.
     */
    template <class load_t, int M, int N, int bM, int bN, int block_y, int block_z, bool transpose>
    struct GlobalMemoryLoader {

      static constexpr int batch = batch_multiple<load_t>::value;

      static constexpr int m_stride_n = block_y * batch;
      static constexpr int n_stride_n = block_z * 1;
      static constexpr int m_stride_t = block_z * batch;
      static constexpr int n_stride_t = block_y * 1;

      static constexpr int register_count
        = std::max(((bN + n_stride_n - 1) / n_stride_n) * ((bM + m_stride_n - 1) / m_stride_n),
                   ((bN + n_stride_t - 1) / n_stride_t) * ((bM + m_stride_t - 1) / m_stride_t));

      load_t reg_real[register_count];
      load_t reg_imag[register_count];

      template <int ld, bool dagger, class T, class gmem_accessor_t>
      __device__ inline float g2tmp(const gmem_accessor_t &gmem, int m_offset, int n_offset, complex<T> *smem_ptr,
                                    pipeline_t &pipe)
      {
        auto p = gmem.data();

        constexpr bool check_bounds = !(M % bM == 0 && N % bN == 0);

        int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
        constexpr int element_per_thread = 16 / (sizeof(T) * 2);
        while (thread_id * element_per_thread < bM * bN) {
          if (transpose != dagger) {
            int m = element_per_thread * (thread_id % (bM / element_per_thread));
            int n = thread_id / (bM / element_per_thread);
            if (!check_bounds || (n + n_offset < N && m + m_offset < M)) {
              auto dst_ptr = reinterpret_cast<float4 *>(&smem_ptr[n * (bM + 4) + m]);
              auto src_ptr = reinterpret_cast<float4 *>(&p[(n + n_offset) * ld + m + m_offset]);
              memcpy_async(dst_ptr, src_ptr, sizeof(float4), pipe);
            }
          } else {
            int m = thread_id / (bN / element_per_thread);
            int n = element_per_thread * (thread_id % (bN / element_per_thread));
            if (!check_bounds || (n + n_offset < N && m + m_offset < M)) {
              auto dst_ptr = reinterpret_cast<float4 *>(&smem_ptr[m * (bN + 4) + n]);
              auto src_ptr = reinterpret_cast<float4 *>(&p[(m + m_offset) * ld + n + n_offset]);
              memcpy_async(dst_ptr, src_ptr, sizeof(float4), pipe);
            }
          }
          thread_id += blockDim.x * blockDim.y * blockDim.z;
        }
        return gmem.get_scale_inv();
      }

      template <int ld, bool dagger, bool fixed, class T, class smem_accessor_t>
      __device__ inline float tmp2s_rescale(complex<T> *smem_ptr, float scale_inv, smem_accessor_t &smem_real,
                                            smem_accessor_t &smem_imag)
      {
        // for each iteration, each warp loads a tile
        int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
        int warp_id = thread_id / 32;
        int lane_id = thread_id % 32;
        int thread_in_group = lane_id % 4;
        int group_id = lane_id / 4;
        constexpr int w_m = 8 * batch;
        constexpr int w_k = 4;
        static_assert(bM % w_m == 0, "bM %% w_m");
        static_assert(bN % w_k == 0, "bN %% w_k");

        constexpr int tile_dim_m = bM / w_m;
        constexpr int tile_dim_k = bN / w_k;

        constexpr int total_tiles = tile_dim_k * tile_dim_m;
        constexpr int n_warp = block_y * block_z / 32;
        constexpr int warp_cycle = (total_tiles + n_warp - 1) / n_warp;

        float thread_max = 0.0f;

#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {
          int logical_warp_index = c * n_warp + warp_id;
          if (logical_warp_index < total_tiles) {
            int warp_m = (c * n_warp + warp_id) % tile_dim_m;
            int warp_k = (c * n_warp + warp_id) / tile_dim_m;

            int smem_m_offset = warp_m * w_m + group_id * batch;
            int smem_k_offset = warp_k * w_k + thread_in_group;

            int gmem_m_offset = smem_m_offset;
            int gmem_k_offset = smem_k_offset;

            constexpr bool x = (transpose == dagger);
            float this_max
              = find_abs_max<x, fixed, dagger, (x ? bN + 4 : bM + 4)>(load_t{}, smem_ptr, gmem_m_offset, gmem_k_offset, scale_inv);
            thread_max = fmaxf(this_max, thread_max);
          }
        }

        // block all-reduce thread_max
        using block_reduce_t = cub::BlockReduce<float, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_y, block_z>;
        __shared__ typename block_reduce_t::TempStorage temp_storage;
        float block_max = block_reduce_t(temp_storage).Reduce(thread_max, cub::Max());

        __shared__ float block_max_all;
        if (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z) == 0) {
          if (block_max > 0.0f) {
            block_max_all = block_max;
          } else {
            block_max_all = 1.0f;
          }
        }
        __syncthreads();

        float block_rescale_factor = 65504.0f / block_max_all; // 65504 = the maximum FP16 number

#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {
          int logical_warp_index = c * n_warp + warp_id;
          if (logical_warp_index < total_tiles) {
            int warp_m = (c * n_warp + warp_id) % tile_dim_m;
            int warp_k = (c * n_warp + warp_id) / tile_dim_m;

            int smem_m_offset = warp_m * w_m + group_id * batch;
            int smem_k_offset = warp_k * w_k + thread_in_group;

            int gmem_m_offset = smem_m_offset;
            int gmem_k_offset = smem_k_offset;

            load_t real;
            load_t imag;

            constexpr bool x = (transpose == dagger);
            // if constexpr (std::is_same_v<load_t, float>) {
            //   convert_x_rescale<x, fixed, dagger, x ? bN + 4 : bM + 4>(real, imag, smem_ptr, gmem_m_offset, gmem_k_offset,
            //                                                            scale_inv, block_rescale_factor);
            // } else {
              convert_x_rescale<x, fixed, dagger, x ? bN + 4 : bM + 4, 1>(&real, &imag, smem_ptr, gmem_m_offset, gmem_k_offset,
                                                                          scale_inv, block_rescale_factor);
            // }
            smem_real.vector_load(smem_m_offset, smem_k_offset, real);
            smem_imag.vector_load(smem_m_offset, smem_k_offset, imag);
          }
        }

        return 1.0f / block_rescale_factor;
      }

      template <int ld, bool dagger, bool fixed, class T, class smem_accessor_t>
      __device__ inline void tmp2s(complex<T> *smem_ptr, float scale_inv, smem_accessor_t &smem_real,
                                   smem_accessor_t &smem_imag)
      {
        // for each iteration, each warp loads a tile
        int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
        int warp_id = thread_id / 32;
        int lane_id = thread_id % 32;
        int thread_in_group = lane_id % 4;
        int group_id = lane_id / 4;
        constexpr int w_m = 8 * batch;
        constexpr int w_k = 4;
        static_assert(bM % w_m == 0, "bM %% w_m");
        static_assert(bN % w_k == 0, "bN %% w_k");

        constexpr int tile_dim_m = bM / w_m;
        constexpr int tile_dim_k = bN / w_k;

        constexpr int total_tiles = tile_dim_k * tile_dim_m;
        constexpr int n_warp = block_y * block_z / 32;
        constexpr int warp_cycle = (total_tiles + n_warp - 1) / n_warp;
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {
          int logical_warp_index = c * n_warp + warp_id;
          if (logical_warp_index < total_tiles) {
            int warp_m = (c * n_warp + warp_id) % tile_dim_m;
            int warp_k = (c * n_warp + warp_id) / tile_dim_m;

            int smem_m_offset = warp_m * w_m + group_id * batch;
            int smem_k_offset = warp_k * w_k + thread_in_group;

            int gmem_m_offset = smem_m_offset;
            int gmem_k_offset = smem_k_offset;

            load_t real;
            load_t imag;

            constexpr bool x = (transpose == dagger);
            convert_x<x, fixed, dagger, x ? bN + 4 : bM + 4, 1>(&real, &imag, smem_ptr, gmem_m_offset, gmem_k_offset,
                                                                scale_inv);
            smem_real.vector_load(smem_m_offset, smem_k_offset, real);
            smem_imag.vector_load(smem_m_offset, smem_k_offset, imag);
          }
        }
      }

      /**
       * ld: leading dimension of global memory
       * dagger: if we need to store daggered (tranpose and hermision conjugate)
       */
      template <int ld, bool dagger, bool rescale, class GmemAccessor>
      __device__ inline float g2r_rescale(const GmemAccessor &gmem, int m_offset, int n_offset)
      {
        auto p = gmem.data();
        auto scale_inv = gmem.get_scale_inv();
        constexpr bool fixed = GmemAccessor::fixed;

        using store_t = typename GmemAccessor::store_type;

        constexpr bool x = (transpose == dagger);

        constexpr int n_stride = x ? block_y : block_z;
        constexpr int m_stride = x ? block_z * batch : block_y * batch;
        int n_thread_offset = x ? threadIdx.y : threadIdx.z;
        int m_thread_offset = x ? threadIdx.z * batch : threadIdx.y * batch;

        constexpr int n_dim = (bN + n_stride - 1) / n_stride;
        constexpr int m_dim = (bM + m_stride - 1) / m_stride;

        constexpr bool check_global_bound = !(M % bM == 0 && N % bN == 0);
        constexpr bool check_shared_bound = !(bM % m_stride == 0 && bN % n_stride == 0);

        using store_array_t = typename VectorType<float, batch>::type;

        store_array_t f_real[register_count];
        store_array_t f_imag[register_count];

        if constexpr (x) {
          constexpr int n_batch = get_mn_batch<store_t>(1, n_dim, bN);
#pragma unroll
          for (int n = 0; n < n_dim / n_batch; n++) {

#pragma unroll
            for (int m = 0; m < m_dim; m++) {

              int n_idx_blk = (n * n_stride + n_thread_offset) * n_batch;
              int m_idx_blk = m * m_stride + m_thread_offset;

              int n_idx = n_idx_blk + n_offset;
              int m_idx = m_idx_blk + m_offset;

              bool b1 = !check_shared_bound || (m_idx_blk < bM && n_idx_blk < bN);
              bool b2 = !check_global_bound || (n_idx < N && m_idx < M);

              if (b1 && b2) {
                convert_x<x, fixed, dagger, ld, n_batch>(&f_real[m * n_dim + n * n_batch],
                                                         &f_imag[m * n_dim + n * n_batch], p, m_idx, n_idx, scale_inv);
              } else {
#pragma unroll
                for (int b = 0; b < n_batch; b++) {
                  zero(f_real[m * n_dim + n * n_batch + b], f_imag[m * n_dim + n * n_batch + b]);
                }
              }
            }
          }
        } else {
          constexpr int m_batch = get_mn_batch<store_t>(batch, m_dim, bM);
#pragma unroll
          for (int n = 0; n < n_dim; n++) {

#pragma unroll
            for (int m = 0; m < m_dim / m_batch; m++) {

              int n_idx_blk = n * n_stride + n_thread_offset;
              int m_idx_blk = (m * m_stride + m_thread_offset) * m_batch;

              int n_idx = n_idx_blk + n_offset;
              int m_idx = m_idx_blk + m_offset;

              bool b1 = !check_shared_bound || (m_idx_blk < bM && n_idx_blk < bN);
              bool b2 = !check_global_bound || (n_idx < N && m_idx < M);

              if (b1 && b2) {
                store_array_t v_real[m_batch];
                store_array_t v_imag[m_batch];
                convert_x<x, fixed, dagger, ld, m_batch>(v_real, v_imag, p, m_idx, n_idx, scale_inv);
#pragma unroll
                for (int b = 0; b < m_batch; b++) {
                  f_real[(m * m_batch + b) * n_dim + n] = v_real[b];
                  f_imag[(m * m_batch + b) * n_dim + n] = v_imag[b];
                }
              } else {
#pragma unroll
                for (int b = 0; b < m_batch; b++) {
                  zero(f_real[(m * m_batch + b) * n_dim + n], f_imag[(m * m_batch + b) * n_dim + n]);
                }
              }
            }
          }
        }

        float block_rescale_factor = 1.0f;
        if constexpr (rescale) {
          float thread_max = 0;
#pragma unroll
          for (int n = 0; n < n_dim; n++) {
#pragma unroll
            for (int m = 0; m < m_dim; m++) {
              thread_max = abs_max(f_real[m * n_dim + n], thread_max);
              thread_max = abs_max(f_imag[m * n_dim + n], thread_max);
            }
          }

          // block all-reduce thread_max
          using block_reduce_t = cub::BlockReduce<float, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_y, block_z>;
          __shared__ typename block_reduce_t::TempStorage temp_storage;
          float block_max = block_reduce_t(temp_storage).Reduce(thread_max, cub::Max());

          __shared__ float block_max_all;
          if (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z) == 0) {
            if (block_max > 0.0f) {
              block_max_all = block_max;
            } else {
              block_max_all = 1.0f;
            }
          }
          __syncthreads();

          block_rescale_factor = 65504.0f / block_max_all; // 65504 = the maximum FP16 number
        }

        if constexpr (std::is_same_v<load_t, half2>) {
#pragma unroll
          for (int n = 0; n < n_dim; n++) {
#pragma unroll
            for (int m = 0; m < m_dim; m++) {
              reg_real[m * n_dim + n] = __floats2half2_rn(f_real[m * n_dim + n].x * block_rescale_factor,
                                                          f_real[m * n_dim + n].y * block_rescale_factor);
              reg_imag[m * n_dim + n] = __floats2half2_rn(f_imag[m * n_dim + n].x * block_rescale_factor,
                                                          f_imag[m * n_dim + n].y * block_rescale_factor);
            }
          }
        } else {
#pragma unroll
          for (int n = 0; n < n_dim; n++) {
#pragma unroll
            for (int m = 0; m < m_dim; m++) {
              reg_real[m * n_dim + n] = f_real[m * n_dim + n] * block_rescale_factor;
              reg_imag[m * n_dim + n] = f_imag[m * n_dim + n] * block_rescale_factor;
            }
          }
        }

        return 1.0f / block_rescale_factor;
      }

      /**
       * ld: leading dimension of global memory
       * dagger: if we need to store daggered (tranpose and hermision conjugate)
       */
      template <int ld, bool dagger, class GmemAccessor>
      __device__ inline void g2r(const GmemAccessor &gmem, int m_offset, int n_offset)
      {
        constexpr bool rescale = false;
        g2r_rescale<ld, dagger, rescale>(gmem, m_offset, n_offset);
      }

      template <class GmemAccessor, bool dagger, class SmemObj>
      __device__ inline void r2s(SmemObj &smem_real, SmemObj &smem_imag)
      {
        constexpr bool x = (transpose == dagger);

        using store_t = typename GmemAccessor::store_type;

        constexpr int n_stride = transpose == dagger ? block_y : block_z;
        constexpr int m_stride = transpose == dagger ? block_z * batch : block_y * batch;
        int n_thread_offset = transpose == dagger ? threadIdx.y : threadIdx.z;
        int m_thread_offset = transpose == dagger ? threadIdx.z * batch : threadIdx.y * batch;

        constexpr int n_dim = (bN + n_stride - 1) / n_stride;
        constexpr int m_dim = (bM + m_stride - 1) / m_stride;

        if constexpr (x) {
          constexpr int n_batch = get_mn_batch<store_t>(1, n_dim, bN);
#pragma unroll
          for (int n = 0; n < n_dim / n_batch; n++) {
#pragma unroll
            for (int m = 0; m < m_dim; m++) {
              const int n_idx = (n * n_stride + n_thread_offset) * n_batch;
              const int m_idx = m * m_stride + m_thread_offset;
              if (m_idx < bM && n_idx < bN) {
                if constexpr (SmemObj::ldn == 1 && SmemObj::ldm % n_batch == 0) {
                  smem_real.vector_load(m_idx, n_idx,
                                        make_vector_t<load_t, n_batch>::get(&reg_real[m * n_dim + n * n_batch]));
                  smem_imag.vector_load(m_idx, n_idx,
                                        make_vector_t<load_t, n_batch>::get(&reg_imag[m * n_dim + n * n_batch]));
                } else {
#pragma unroll
                  for (int b = 0; b < n_batch; b++) {
                    smem_real.vector_load(m_idx, n_idx + b, reg_real[m * n_dim + n * n_batch + b]);
                    smem_imag.vector_load(m_idx, n_idx + b, reg_imag[m * n_dim + n * n_batch + b]);
                  }
                }
              }
            }
          }
        } else {
          constexpr int m_batch = get_mn_batch<store_t>(batch, m_dim, bM);
#pragma unroll
          for (int n = 0; n < n_dim; n++) {
#pragma unroll
            for (int m = 0; m < m_dim / m_batch; m++) {
              const int n_idx = n * n_stride + n_thread_offset;
              const int m_idx = (m * m_stride + m_thread_offset) * m_batch;
              if (m_idx < bM && n_idx < bN) {
                if constexpr (SmemObj::ldm == 1 && SmemObj::ldn % (batch * m_batch) == 0) {
                  load_t v_real[m_batch];
                  load_t v_imag[m_batch];
#pragma unroll
                  for (int b = 0; b < m_batch; b++) {
                    v_real[b] = reg_real[(m * m_batch + b) * n_dim + n];
                    v_imag[b] = reg_imag[(m * m_batch + b) * n_dim + n];
                  }
                  smem_real.vector_load(m_idx, n_idx, make_vector_t<load_t, m_batch>::get(v_real));
                  smem_imag.vector_load(m_idx, n_idx, make_vector_t<load_t, m_batch>::get(v_imag));
                } else {
#pragma unroll
                  for (int b = 0; b < m_batch; b++) {
                    smem_real.vector_load(m_idx + b * batch, n_idx, reg_real[(m * m_batch + b) * n_dim + n]);
                    smem_imag.vector_load(m_idx + b * batch, n_idx, reg_imag[(m * m_batch + b) * n_dim + n]);
                  }
                }
              }
            }
          }
        }
      }
    };

  } // namespace mma

} // namespace quda
