#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <dslash_quda.h>
#include <index_helper.cuh>
#include <inline_ptx.h>
#include <math_helper.cuh>
#include <tunable_nd.h>
#include <kernels/madwf_transfer.cuh>
#include <kernels/madwf_tensor.cuh>

#include <cub_helper.cuh>
#define THRUST_IGNORE_CUB_VERSION_CHECK
#include "thrust/inner_product.h"

#include <madwf_transfer.h>
#include <quda_cuda_api.h>

namespace quda
{
  namespace madwf_ml
  {
#if 0
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
        Ls_out(out.X(4)),
        Ls_in(in.X(4)),
        volume_4d_cb(in.VolumeCB() / in.X(4)),
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
#endif

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

#if 0
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

      VectorCache<Vector> cache;

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

#endif

    template <class storage_type, class matrix_type> class transfer_5D_wrapper: public TunableKernel3D
    {
      ColorSpinorField &out;
      const ColorSpinorField &in;
      const matrix_type *wm_p;
      bool dagger;

    private:
      unsigned int sharedBytesPerThread() const
      {
        return 0;
      }

      unsigned int sharedBytesPerBlock(const TuneParam &param) const
      {
        return out.X(4) * in.X(4) * sizeof(matrix_type);
      }

      unsigned int minThreads() const { return out.VolumeCB() / out.X(4); }

    public:
      transfer_5D_wrapper(ColorSpinorField &out, const ColorSpinorField &in, const matrix_type *wm_p, bool dagger):
        TunableKernel3D(out, out.X(4), out.SiteSubset()), out(out), in(in), wm_p(wm_p), dagger(dagger)
      {
        TunableKernel2D_base<false>::resizeStep(out.X(4)); // Ls must be contained in the block
        // FIXME: the threadblock must only have one parity

        strcpy(aux, out.AuxString());
        char tmp[512];
        sprintf(tmp, ",%02d->%02d", in.X(4), out.X(4));
        strcat(aux, tmp);
        strcat(aux, ",transfer_5D");
        if (dagger) strcat(aux, ",Dagger");

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        if (dagger) {
          launch<Transfer5D>(tp, stream, Transfer5DArg<storage_type, matrix_type, true>(out, in, wm_p));
        } else {
          launch<Transfer5D>(tp, stream, Transfer5DArg<storage_type, matrix_type, false>(out, in, wm_p));
        }
      }

      long long flops() const { return 0; }
      long long bytes() const { return in.Bytes() + out.Bytes(); }
    };

    template <class storage_type, class matrix_type> class tensor_5D_wrapper : public TunableKernel3D
    {
      const ColorSpinorField &out;
      const ColorSpinorField &in;
      matrix_type *wm_p;

    private:
      unsigned int sharedBytesPerThread() const
      {
        return in.X(4) * color_spin_dim * 2 * sizeof(typename mapper<storage_type>::type) / out.X(4);
      }

      unsigned int sharedBytesPerBlock(const TuneParam &param) const
      {
        return 0;
      }
      // bool advanceSharedBytes(TuneParam &param) const { return false; } // Don't tune shared mem
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return out.VolumeCB() / out.X(4); }

    public:
      tensor_5D_wrapper(const ColorSpinorField &out, const ColorSpinorField &in, matrix_type *wm_p) :
        TunableKernel3D(out, out.X(4), out.SiteSubset()), out(out), in(in), wm_p(wm_p)
      {
        TunableKernel2D_base<false>::resizeStep(out.X(4)); // Ls must be contained in the block
        // FIXME: the threadblock must only have one parity

        strcpy(aux, out.AuxString());
        char tmp[512];
        sprintf(tmp, ",%02d->%02d", in.X(4), out.X(4));
        strcat(aux, tmp);
        strcat(aux, ",tensor_5D");

        apply(device::get_default_stream());
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        tp.set_max_shared_bytes = true;
        using Arg = Tensor5DArg<storage_type, matrix_type>;
        using complex_type = complex<typename Arg::real>;

        // Each block has a Wilson Matrix.
        int num_x_blocks = tp.grid.x;
        int alloc_size = num_x_blocks * sizeof(matrix_type) * in.X(4) * out.X(4);

        Arg arg(out, in, reinterpret_cast<matrix_type *>(device_malloc(alloc_size)));

        launch<Tensor5D>(tp, stream, arg);
        // launch(tensor_5d_kernel<storage_type, Arg>, tp, arg, stream);

        constexpr int block_size = 128;
        tensor_reduce_kernel<complex_type, block_size>
          <<<arg.Ls_in * arg.Ls_out * sizeof(matrix_type) / sizeof(complex_type), block_size, 0, target::cuda::get_stream(stream)>>>(
              reinterpret_cast<complex_type *>(wm_p), reinterpret_cast<complex_type *>(arg.reduce_buffer), num_x_blocks);

        device_free(arg.reduce_buffer);
      }

      long long flops() const { return 0; }
      long long bytes() const { return in.Bytes() + out.Bytes(); }
    };

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
        transfer_5D_wrapper<short, matrix_type> w(out, in, reinterpret_cast<const matrix_type *>(tp.data()), dagger);
      } break;
      case QUDA_QUARTER_PRECISION: {
        transfer_5D_wrapper<int8_t, matrix_type> w(out, in, reinterpret_cast<const matrix_type *>(tp.data()), dagger);
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
        tensor_5D_wrapper<short, matrix_type> w(out, in, reinterpret_cast<matrix_type *>(tp.data()));
      } break;
      case QUDA_QUARTER_PRECISION: {
        tensor_5D_wrapper<int8_t, matrix_type> w(out, in, reinterpret_cast<matrix_type *>(tp.data()));
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
      axpby_kernel<<<grid_size, block_size>>>(
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
