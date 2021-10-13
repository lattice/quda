#pragma once

#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <shared_memory_cache_helper.cuh>
#include <kernels/madwf_transfer.cuh>

#include <cub_helper.cuh>

namespace quda
{

  namespace madwf_ml
  {
    constexpr int warp_size = 32;

    template <class T> __device__ inline void warp_reduce(T &f)
    {
      int index = target::thread_idx().y * (target::block_dim().x / warp_size) + (target::thread_idx().x / warp_size);
      typedef cub::WarpReduce<T> WarpReduce;
      // Allocate WarpReduce shared memory for 4 warps
      __shared__ typename WarpReduce::TempStorage temp_storage[32];
      // Return the warp-wide sums to each lane0 (threads 0, 32, 64, and 96)
      f = WarpReduce(temp_storage[index]).Sum(f);
    }

    template <class T> __device__ inline void block_reduce_x(T &f)
    {
      int lane_id_x = target::thread_idx().x % warp_size;
      int warp_id_x = target::thread_idx().x / warp_size;
      int warp_ct_x = target::block_dim().x / warp_size;

      __shared__ T smem[32];

      warp_reduce(f);
      // Now lane 0 of each warp holds the reduced value

      if (warp_ct_x > 1) {
        int index = target::thread_idx().y * warp_ct_x + warp_id_x;
        if (lane_id_x == 0) { smem[index] = f; }
        __syncthreads();
        if (warp_id_x == 0) {
          f = (lane_id_x < warp_ct_x) ? smem[index] : 0;
          warp_reduce(f);
        }
      }
      // Now the first thread in the x direction holds the reduction result.
    }

    /**
      @brief Form a 4-by-4 tensor from v and w (color d.o.f. is contracted)
      @param[in] mp The buffer to accumulate on
      @param[in] v, w input vectors
     */
    template <class real>
      __device__ __host__ inline void vector_tensor_matrix(SpinMatrix<real> *mp, int m_index, const WilsonVector<real> &v,
          const WilsonVector<real> &w)
      {
        complex<real> *p = reinterpret_cast<complex<real> *>(mp);

#pragma unroll
        for (int a = 0; a < spin_dim; a++) {
#pragma unroll
          for (int b = 0; b < spin_dim; b++) {
            int cs = a * spin_dim + b;
            complex<real> z = 0;
#pragma unroll
            for (int color = 0; color < color_dim; color++) { z += conj(conj(v(a, color)) * w(b, color)); }
            // Perform a block reduction across the x direction
#ifdef __CUDA_ARCH__
            block_reduce_x(z);
#endif
            if (target::thread_idx().x == 0) { p[(m_index * spin_dim * spin_dim + cs) * target::grid_dim().x + target::block_idx().x] = z; }
          }
        }
      }

    template <class storage_type, class matrix_type_, int block_dim_x_> struct Tensor5DArg: kernel_param<> {

      using F = typename colorspinor_mapper<storage_type, 4, 3>::type;
      using real = typename mapper<storage_type>::type;
      using Vector = ColorSpinor<real, 3, 4>;
      using matrix_type = matrix_type_;

      static constexpr int block_dim_x = block_dim_x_;

      const F out; // output vector field
      const F in;  // input vector field

      const int Ls_out; // length of 5th dimension of the out field 
      const int Ls_in;  // length of 5th dimension of the in field

      const int volume_4d_cb;

      matrix_type *reduce_buffer = nullptr;

      const int nParity;

      matrix_type *result_d;

      int batch;

      Tensor5DArg(const ColorSpinorField &out, const ColorSpinorField &in, int num_x_blocks, matrix_type *result_d):
        kernel_param(dim3(out.VolumeCB() / out.X(4), out.X(4), out.SiteSubset())),
        out(out),
        in(in),
        Ls_out(out.X(4)),
        Ls_in(in.X(4)),
        volume_4d_cb(in.VolumeCB() / in.X(4)),
        nParity(in.SiteSubset()),
        result_d(result_d),
        batch(num_x_blocks)
      {

        if (volume_4d_cb != static_cast<int>(out.VolumeCB() / Ls_out)) {
          errorQuda("Input and Output fields should have the same 4d volume: %d neq %d.\n", volume_4d_cb,
              static_cast<int>(out.VolumeCB() / Ls_out));
        }

        if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
        if (in.Ncolor() != 3) errorQuda("nColor = %d not support", in.Ncolor());
        if (out.Nspin() != 4) errorQuda("nSpin = %d not support", out.Nspin());
        if (out.Ncolor() != 3) errorQuda("nColor = %d not support", out.Ncolor());

        if (!in.isNative() || !out.isNative())
          errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());

        size_t reduce_bytes = num_x_blocks * sizeof(matrix_type) * out.X(4) * in.X(4);
        reduce_buffer = reinterpret_cast<matrix_type *>(device_malloc(reduce_bytes));
      }

      ~Tensor5DArg() {
        if (reduce_buffer) { device_free(reduce_buffer); }
      }

    };

    template <class Arg>
      struct Tensor5D {

        const Arg &arg;
        constexpr Tensor5D(const Arg &arg) : arg(arg) {}
        static constexpr const char *filename() { return KERNEL_FILE; }

        /**
          @brief Form a Ls_in-by-Ls_out tensor product from the two input vectors. The type of reduce_buffer
                  determines if the resulting tensor has spin and/or color d.o.f, or just two chiral d.o.f
          @param[in] parity Parity we are on
          @param[in] x_b Checkerboarded 4-d space-time index
          @param[in] s Ls dimension coordinate
         */
        __device__ __host__ inline void operator()(int x_cb, int s, int parity)
        {
          using Vector = typename Arg::Vector;

          const int Ls_in = arg.Ls_in;
          const int Ls_out = arg.Ls_out;
          const int volume_4d_cb = arg.volume_4d_cb;
          auto reduce_buffer = arg.reduce_buffer;

          if (x_cb >= volume_4d_cb) return;
          if (s >= Ls_out) return;
          if (parity >= arg.nParity) return;

          SharedMemoryCache<Vector> cache;

          int ld = Ls_in * target::block_dim().x;
          int t = s;
          while (t < Ls_in) {
            int index = t * target::block_dim().x + target::thread_idx().x;
            cache.save_idx(index, ld, arg.in(t * volume_4d_cb + x_cb, parity));
            t += target::block_dim().y;
          }
          cache.sync();

          // t -> s_in, s-> s_out
          const Vector v = arg.out(s * volume_4d_cb + x_cb, parity);
          for (t = 0; t < Ls_in; t++) {
            const Vector w = cache.load_idx(t * target::block_dim().x + target::thread_idx().x, ld);
            int wm_index = s * Ls_in + t;
            vector_tensor_matrix(reduce_buffer, wm_index, v, w);
          }
        }

      };

      template <class Arg>
      struct Tensor5DReduce {

        const Arg &arg;
        constexpr Tensor5DReduce(const Arg &arg) : arg(arg) {}
        static constexpr const char *filename() { return KERNEL_FILE; }

        __device__ __host__ inline void operator()(int, int, int)
        {
          using T = complex<typename Arg::real>;

          int thread_idx = target::thread_idx().x;
          int batch = arg.batch;

          T *in = reinterpret_cast<T *>(arg.reduce_buffer);
          T z = 0;
          while (thread_idx < batch) {
            z += in[target::block_idx().x * batch + thread_idx];
            thread_idx += target::block_dim().x;
          }

#ifdef __CUDA_ARCH__
          typedef cub::BlockReduce<T, Arg::block_dim_x> BlockReduce;
          __shared__ typename BlockReduce::TempStorage temp_storage;
          z = BlockReduce(temp_storage).Sum(z);
#endif
          T *out = reinterpret_cast<T *>(arg.result_d);
          if (target::thread_idx().x == 0) { out[target::block_idx().x] = z; }
        }

      };
  }

}
