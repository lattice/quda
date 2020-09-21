#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <dslash_quda.h>
#include <index_helper.cuh>
#include <inline_ptx.h>
#include <math_helper.cuh>
#include <shared_memory_cache_helper.cuh>

#include <cub_helper.cuh>
#include <thrust_helper.cuh>

#include <madwf_transfer.h>

namespace quda
{
  namespace madwf_ml
  {

    template <class real> using WilsonVector = ColorSpinor<real, 3, 4>;

    template <class real, int dim> class Matrix
    {

      static constexpr int size = dim * dim;
      complex<real> data[size];

    public:
      __device__ __host__ inline Matrix<real, dim>()
      {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] = 0; }
      }

      __device__ __host__ inline Matrix<real, dim>(const Matrix<real, dim> &a)
      {
#pragma unroll
        for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
      }

      __device__ __host__ inline Matrix<real, dim> &operator=(const Matrix<real, dim> &a)
      {
        if (this != &a) {
#pragma unroll
          for (int i = 0; i < size; i++) { data[i] = a.data[i]; }
        }
        return *this;
      }

      __device__ __host__ inline complex<real> &operator()(int index) { return data[index]; }

      __device__ __host__ inline const complex<real> &operator()(int index) const { return data[index]; }

      // Wilson Matrix is row major
      __device__ __host__ inline complex<real> &operator()(int row, int column) { return data[row * dim + column]; }

      __device__ __host__ inline const complex<real> &operator()(int row, int column) const
      {
        return data[row * dim + column];
      }
    };

    template <class real> using WilsonMatrix = Matrix<real, color_spin_dim>;

    template <bool dagger, class real>
    __device__ __host__ inline WilsonVector<real> matrix_vector_multiply(const WilsonMatrix<real> &m,
                                                                         const WilsonVector<real> &v)
    {
      WilsonVector<real> out; // out is initialized to zero
#pragma unroll
      for (int column = 0; column < color_spin_dim; column++) {
        auto v_col = v(column);
#pragma unroll
        for (int row = 0; row < color_spin_dim; row++) {
          if (dagger) {
            out(row) += conj(m(column, row)) * v_col;
          } else {
            out(row) += m(row, column) * v_col;
          }
        }
      }
      return out;
    }

    template <class real> using SpinMatrix = Matrix<real, spin_dim>;

    template <bool dagger, class real>
    __device__ __host__ inline WilsonVector<real> matrix_vector_multiply(const SpinMatrix<real> &m,
                                                                         const WilsonVector<real> &v)
    {
      WilsonVector<real> out; // out is initialized to zero
#pragma unroll
      for (int color = 0; color < color_dim; color++) {
#pragma unroll
        for (int column = 0; column < spin_dim; column++) {
          auto v_col = v(column, color);
#pragma unroll
          for (int row = 0; row < spin_dim; row++) {
            if (dagger) {
              out(row, color) += conj(m(column, row)) * v_col;
            } else {
              out(row, color) += m(row, column) * v_col;
            }
          }
        }
      }
      return out;
    }

    template <class real> using ChiralProjector = complex<real>[2];

    template <bool dagger, class real>
    __device__ __host__ inline WilsonVector<real> matrix_vector_multiply(const ChiralProjector<real> &m,
                                                                         const WilsonVector<real> &v)
    {
      if (dagger) {
        return conj(m[0]) * v.project(4, +1).reconstruct(4, +1) + conj(m[1]) * v.project(4, -1).reconstruct(4, -1);
      } else {
        return m[0] * v.project(4, +1).reconstruct(4, +1) + m[1] * v.project(4, -1).reconstruct(4, -1);
      }
    }

    constexpr int warp_size = 32;

    template <class T> __device__ inline void warp_reduce(T &f)
    {
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        T other_f = __shfl_down_sync(0xffffffffu, f, offset);
        f += other_f;
      }
    }

    template <class T> __device__ inline void block_reduce_x(T &f)
    {

      int lane_id_x = threadIdx.x % warp_size;
      int warp_id_x = threadIdx.x / warp_size;
      int block_dim_x = blockDim.x / warp_size;

      __shared__ T smem[32];

      warp_reduce(f);
      // Now lane 0 of each warp holds the reduced value

      if (block_dim_x > 1) {
        int index = threadIdx.y * block_dim_x + warp_id_x;
        if (lane_id_x == 0) { smem[index] = f; }
        __syncthreads();
        if (warp_id_x == 0) {
          f = (lane_id_x < block_dim_x) ? smem[index] : 0;
          warp_reduce(f);
        }
      }
      // Now the first thread in the x direction holds the reduction result.
    }

    template <class real>
    __device__ inline void vector_tensor_matrix(WilsonMatrix<real> *mp, const WilsonVector<real> &v,
                                                const WilsonVector<real> &w)
    {

      real *real_p = reinterpret_cast<real *>(mp);

#pragma unroll
      for (int a = 0; a < color_spin_dim; a++) {
#pragma unroll
        for (int b = 0; b < color_spin_dim; b++) {
          int cs = a * color_spin_dim + b;
          complex<real> z = conj(conj(v(a)) * w(b));
          // Perform a block reduction across the x direction
          block_reduce_x(z);
          if (threadIdx.x == 0) {
            atomicAdd(&real_p[cs * 2 + 0], z.real());
            atomicAdd(&real_p[cs * 2 + 1], z.imag());
          }
        }
      }
    }

    template <class real>
    __device__ inline void vector_tensor_matrix(SpinMatrix<real> *mp, int m_index, const WilsonVector<real> &v,
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
          block_reduce_x(z);
          if (threadIdx.x == 0) { p[(m_index * spin_dim * spin_dim + cs) * gridDim.x + blockIdx.x] = z; }
        }
      }
    }

    template <class real>
    __device__ inline void vector_tensor_matrix(ChiralProjector<real> *mp, int m_index, const WilsonVector<real> &v,
                                                const WilsonVector<real> &w)
    {

      complex<real> *p = reinterpret_cast<complex<real> *>(mp);

#pragma unroll
      for (int pm = 0; pm < 2; pm++) {
        complex<real> z = 0;
        WilsonVector<real> projected_w = w.project(4, 1 - 2 * pm).reconstruct(4, 1 - 2 * pm);
#pragma unroll
        for (int spin = 0; spin < spin_dim; spin++) {
#pragma unroll
          for (int color = 0; color < color_dim; color++) {
            z += conj(conj(v(spin, color)) * projected_w(spin, color));
          }
        }
        // Perform a block reduction across the x direction
        block_reduce_x(z);
        if (threadIdx.x == 0) { p[(m_index * 2 + pm) * gridDim.x + blockIdx.x] = z; }
      }
    }

#ifdef GPU_DOMAIN_WALL_DIRAC

    template <class storage_type, class matrix_type> struct MadwfMlArg {

      typedef typename colorspinor_mapper<storage_type, 4, 3>::type F;
      typedef typename mapper<storage_type>::type real;

      F out;      // output vector field
      const F in; // input vector field

      const int Ls_out; // length of 5th dimension
      const int Ls_in;  // length of 5th dimension

      const int volume_4d_cb;

      matrix_type *tensor_out_p;
      const matrix_type *wm_p;

      typedef matrix_type MatrixType;
      static constexpr int matrix_size = sizeof(matrix_type);

      const bool dagger; // dagger

      const int nParity;

      const bool transfer;

      MadwfMlArg(ColorSpinorField &out, const ColorSpinorField &in, const matrix_type *wm_p, bool dagger) :
        out(out),
        in(in),
        volume_4d_cb(in.VolumeCB() / in.X(4)),
        Ls_in(in.X(4)),
        Ls_out(out.X(4)),
        wm_p(wm_p),
        dagger(dagger),
        transfer(true),
        nParity(in.SiteSubset())
      {

        if (volume_4d_cb != (int)out.VolumeCB() / Ls_out) {
          errorQuda("Input and Output fields should have the same 4d volume: %d neq %d.\n", volume_4d_cb,
                    (int)out.VolumeCB() / Ls_out);
        }

        if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
        if (in.Ncolor() != 3) errorQuda("nColor = %d not support", in.Ncolor());
        if (out.Nspin() != 4) errorQuda("nSpin = %d not support", out.Nspin());
        if (out.Ncolor() != 3) errorQuda("nColor = %d not support", out.Ncolor());

        if (!in.isNative() || !out.isNative())
          errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());
      }

      MadwfMlArg(ColorSpinorField &out, const ColorSpinorField &in, matrix_type *wm_p) :
        out(out),
        in(in),
        volume_4d_cb(in.VolumeCB() / in.X(4)),
        Ls_in(in.X(4)),
        Ls_out(out.X(4)),
        tensor_out_p(wm_p),
        dagger(false),
        transfer(false),
        nParity(in.SiteSubset())
      {

        if (volume_4d_cb != (int)out.VolumeCB() / Ls_out) {
          errorQuda("Input and Output fields should have the same 4d volume: %d neq %d.\n", volume_4d_cb,
                    (int)out.VolumeCB() / Ls_out);
        }

        if (in.Nspin() != 4) errorQuda("nSpin = %d not support", in.Nspin());
        if (in.Ncolor() != 3) errorQuda("nColor = %d not support", in.Ncolor());
        if (out.Nspin() != 4) errorQuda("nSpin = %d not support", out.Nspin());
        if (out.Ncolor() != 3) errorQuda("nColor = %d not support", out.Ncolor());

        if (!in.isNative() || !out.isNative())
          errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());
      }
    };

    template <class T, int block_size> __global__ void tensor_reduce_kernel(T *out, T *in, int batch_size)
    {

      int tid = threadIdx.x;

      T z = 0;
      while (tid < batch_size) {
        z += in[blockIdx.x * batch_size + tid];
        tid += blockDim.x;
      }

      typedef cub::BlockReduce<T, block_size> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      T aggregate = BlockReduce(temp_storage).Sum(z);

      if (threadIdx.x == 0) { out[blockIdx.x] = aggregate; }
    }

    template <class storage_type, class Arg> __global__ void tensor_5d_kernel(Arg arg)
    {

      typedef typename mapper<storage_type>::type real;
      typedef ColorSpinor<real, 3, 4> Vector;

      const int Ls_in = arg.Ls_in;
      const int Ls_out = arg.Ls_out;
      const int volume_4d_cb = arg.volume_4d_cb;
      auto wm_p = arg.tensor_out_p;

      int index_4d_cb = blockIdx.x * blockDim.x + threadIdx.x;
      int s = blockIdx.y * blockDim.y + threadIdx.y;
      int parity = blockIdx.z * blockDim.z + threadIdx.z;

      if (index_4d_cb >= volume_4d_cb) return;
      if (s >= Ls_out) return;
      if (parity >= arg.nParity) return;

      VectorCache<real, Vector> cache;

      int ld = Ls_in * blockDim.x;
      int t = s;
      while (t < Ls_in) {
        int index = t * blockDim.x + threadIdx.x;
        cache.save(index, ld, arg.in(t * volume_4d_cb + index_4d_cb, parity));
        t += blockDim.y;
      }
      cache.sync();

      // t -> s_in, s-> s_out
      const Vector v = arg.out(s * volume_4d_cb + index_4d_cb, parity);
      for (t = 0; t < Ls_in; t++) {
        const Vector w = cache.load(t * blockDim.x + threadIdx.x, ld);
        int wm_index = s * Ls_in + t;
        vector_tensor_matrix(wm_p, wm_index, v, w);
      }
    }

    template <class storage_type, bool dagger, class Arg> __global__ void transfer_5d_kernel(Arg arg)
    {

      typedef typename mapper<storage_type>::type real;
      typedef ColorSpinor<real, 3, 4> Vector;
      typedef typename Arg::MatrixType matrix_type;

      const int Ls_in = arg.Ls_in;
      const int Ls_out = arg.Ls_out;
      const int volume_4d_cb = arg.volume_4d_cb;
      const matrix_type *wm_p = arg.wm_p;

      int index_4d_cb = blockIdx.x * blockDim.x + threadIdx.x;
      int s = blockIdx.y * blockDim.y + threadIdx.y;
      int parity = blockIdx.z * blockDim.z + threadIdx.z;

      int tid = threadIdx.y * blockDim.x + threadIdx.x;
      extern __shared__ real smem[];
      while (tid < Ls_out * Ls_in * sizeof(matrix_type) / sizeof(real)) {
        smem[tid] = reinterpret_cast<const real *>(wm_p)[tid];
        tid += blockDim.y * blockDim.x;
      }
      __syncthreads();

      if (index_4d_cb >= volume_4d_cb) return;
      if (s >= Ls_out) return;
      if (parity >= arg.nParity) return;

      Vector out;
      // t -> s_in, s-> s_out
      for (int t = 0; t < Ls_in; t++) {
        Vector in = arg.in(t * volume_4d_cb + index_4d_cb, parity);
        int wm_index;
        if (dagger) {
          wm_index = t * Ls_out + s;
        } else {
          wm_index = s * Ls_in + t;
        }
        out += matrix_vector_multiply<dagger>(reinterpret_cast<const matrix_type *>(smem)[wm_index], in);
      }
      arg.out(s * volume_4d_cb + index_4d_cb, parity) = out;
    }

    template <class storage_type, class Arg> class Transfer5d : public TunableVectorYZ
    {

      typedef typename mapper<storage_type>::type real;

      Arg &arg;
      const ColorSpinorField &meta; // this reference is for meta data only

    private:
      unsigned int sharedBytesPerThread() const
      {
        if (arg.transfer) {
          return 0;
        } else {
          return (arg.Ls_in) * color_spin_dim * 2 * sizeof(typename mapper<storage_type>::type) / arg.Ls_out;
        }
      }

      unsigned int sharedBytesPerBlock(const TuneParam &param) const
      {
        if (arg.transfer) {
          return arg.Ls_in * arg.Ls_out * Arg::matrix_size;
        } else {
          return 0;
        }
      }
      // bool advanceSharedBytes(TuneParam &param) const { return false; } // Don't tune shared mem
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.volume_4d_cb; }

    public:
      Transfer5d(const ColorSpinorField &meta, Arg &arg) :
        TunableVectorYZ(arg.Ls_out, arg.nParity), arg(arg), meta(meta)
      {
        strcpy(aux, meta.AuxString());
        char tmp[512];
        sprintf(tmp, ",%02d->%02d", arg.Ls_in, arg.Ls_out);
        strcat(aux, tmp);
        strcat(aux, arg.transfer ? ",transfer_5d" : ",tensor_5d");
        if (arg.dagger) strcat(aux, ",Dagger");
      }

      virtual ~Transfer5d() { ; }

      template <class F> inline void launch(F *f, const TuneParam &tp, Arg &arg, const cudaStream_t &stream)
      {
        qudaLaunchKernel(f, tp, stream, arg);
      }

      void apply(const cudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (arg.transfer) {
          auto f = arg.dagger ? transfer_5d_kernel<storage_type, true, Arg> : transfer_5d_kernel<storage_type, false, Arg>;
          qudaLaunchKernel(f, tp, stream, arg);
        } else {
          using Complex = complex<real>;
          using matrix_type = typename Arg::MatrixType;

          // Each block has a Wilson Matrix.
          int num_x_blocks = tp.grid.x;
          int alloc_size = num_x_blocks * Arg::matrix_size * arg.Ls_in * arg.Ls_out;

          Complex *tensor_out = reinterpret_cast<Complex *>(arg.tensor_out_p);
          arg.tensor_out_p = (matrix_type *)device_malloc(alloc_size);
          Complex *tensor_in = reinterpret_cast<Complex *>(arg.tensor_out_p);

          launch(tensor_5d_kernel<storage_type, Arg>, tp, arg, stream);

          constexpr int block_size = 128;
          tensor_reduce_kernel<Complex, block_size>
            <<<arg.Ls_in * arg.Ls_out * sizeof(matrix_type) / sizeof(Complex), block_size, 0, stream>>>(
              tensor_out, tensor_in, num_x_blocks);

          device_free(tensor_in);
        }
      }

      void initTuneParam(TuneParam &param) const
      {
        TunableVectorYZ::initTuneParam(param);
        param.block.y = arg.Ls_out; // Ls must be contained in the block
        param.grid.y = 1;
        param.shared_bytes
          = sharedBytesPerBlock(param) + sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
      }

      void defaultTuneParam(TuneParam &param) const
      {
        TunableVectorYZ::defaultTuneParam(param);
        param.block.y = arg.Ls_out; // Ls must be contained in the block
        param.grid.y = 1;
        param.shared_bytes
          = sharedBytesPerBlock(param) + sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
      }

      TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

      long long flops() const { return 0; }
      long long bytes() const { return arg.in.Bytes() + arg.out.Bytes(); }
    };

#endif

// The following macro choose the structure of the transfer matrix.
#if 1
    using matrix_type = SpinMatrix<float>;
#else
    using matrix_type = ChiralProjector<float>;
#endif

    void transfer_5d_hh(ColorSpinorField &out, const ColorSpinorField &in, const TrainingParameter<float> &tp, bool dagger)
    {
#ifdef GPU_DOMAIN_WALL_DIRAC
      checkLocation(out, in); // check all locations match
      size_t m_size = in.X(4) * out.X(4) * sizeof(matrix_type);
      if (tp.get_size() * sizeof(float) != m_size) {
        errorQuda("Training Parameter size mismatch %lu neq %lu.\n", tp.get_size() * sizeof(float), m_size);
      }
      switch (checkPrecision(out, in)) {
      case QUDA_HALF_PRECISION: {
        using arg_type = MadwfMlArg<short, matrix_type>;
        arg_type arg(out, in, (const matrix_type *)tp.data(), dagger);
        Transfer5d<short, arg_type> dslash(in, arg);
        dslash.apply(streams[Nstream - 1]);
      } break;
      case QUDA_QUARTER_PRECISION: {
        using arg_type = MadwfMlArg<int8_t, matrix_type>;
        arg_type arg(out, in, (const matrix_type *)tp.data(), dagger);
        Transfer5d<int8_t, arg_type> dslash(in, arg);
        dslash.apply(streams[Nstream - 1]);
      } break;

      default: errorQuda("Unsupported precision %d\n", in.Precision());
      }
#else
      errorQuda("Mobius dslash has not been built");
#endif
    }

    void tensor_5d_hh(ColorSpinorField &out, const ColorSpinorField &in, TrainingParameter<float> &tp)
    {
#ifdef GPU_DOMAIN_WALL_DIRAC
      checkLocation(out, in); // check all locations match
      size_t m_size = in.X(4) * out.X(4) * sizeof(matrix_type);
      if (tp.get_size() * sizeof(float) != m_size) {
        errorQuda("Training Parameter size mismatch %lu neq %lu.\n", tp.get_size() * sizeof(float), m_size);
      }
      switch (checkPrecision(out, in)) {
      case QUDA_HALF_PRECISION: {
        using arg_type = MadwfMlArg<short, matrix_type>;
        cudaMemsetAsync(tp.data(), 0, m_size, streams[Nstream - 1]);
        arg_type arg(out, in, (matrix_type *)tp.data());
        Transfer5d<short, arg_type> dslash(in, arg);
        dslash.apply(streams[Nstream - 1]);
      } break;
      case QUDA_QUARTER_PRECISION: {
        using arg_type = MadwfMlArg<int8_t, matrix_type>;
        cudaMemsetAsync(tp.data(), 0, m_size, streams[Nstream - 1]);
        arg_type arg(out, in, (matrix_type *)tp.data());
        Transfer5d<int8_t, arg_type> dslash(in, arg);
        dslash.apply(streams[Nstream - 1]);
      } break;
      default: errorQuda("Unsupported precision %d\n", in.Precision());
      }
#else
      errorQuda("Mobius dslash has not been built");
#endif
    }

    __global__ void axpby_kernel(complex<float> *out_p, int size, complex<float> a, const complex<float> *x_p,
                                 complex<float> b, const complex<float> *y_p)
    {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      while (index < size) {
        out_p[index] += a * x_p[index] + b * y_p[index];
        index += blockDim.x * gridDim.x;
      }
    }

    void axpby(TrainingParameter<float> &out, complex<float> a, const TrainingParameter<float> &x, complex<float> b,
               const TrainingParameter<float> &y)
    {
      int p_size = out.get_size() / 2; // complex
      constexpr int block_size = 256;
      int grid_size = (p_size + block_size - 1) / block_size;
      axpby_kernel<<<grid_size, block_size, 0, streams[Nstream - 1]>>>(
        (complex<float> *)out.data(), p_size, a, (complex<float> *)x.data(), b, (complex<float> *)y.data());
    }

    float inner_product(const TrainingParameter<float> &a, const TrainingParameter<float> &b)
    {
      size_t p_size = a.get_size();
      if (p_size != b.get_size()) { errorQuda("size mismatch between inputs.\n"); }
      return thrust::inner_product(thrust::device, a.data(), a.data() + p_size, b.data(), 0.0f);
    }

  } // namespace madwf_ml
} // namespace quda
