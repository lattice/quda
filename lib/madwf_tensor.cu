#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <device_vector.h>
#include <madwf_transfer.h>
#include <kernels/madwf_tensor.cuh>

#include <cub_helper.cuh>
#include <targets/cuda/quda_cuda_api.h>

namespace quda
{
  namespace madwf_ml
  {

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

      unsigned int sharedBytesPerBlock(const TuneParam &) const
      {
        return 0;
      }

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
        constexpr int block_size = 128;
        
        qudaEvent_t event = qudaEventCreate();

        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        tp.set_max_shared_bytes = true;
        using Arg = Tensor5DArg<storage_type, matrix_type, block_size>;
        using complex_type = complex<typename Arg::real>;

        // Each block has a Wilson Matrix.
        int num_x_blocks = tp.grid.x;

        Arg arg(out, in, num_x_blocks, wm_p);

        launch<Tensor5D>(tp, stream, arg);

        tp.grid = {static_cast<unsigned>(arg.Ls_in * arg.Ls_out * sizeof(matrix_type) / sizeof(complex_type)), 1, 1};
        tp.block = {block_size, 1, 1};
        arg.threads = {tp.grid.x * tp.block.x, 1, 1};
        launch<Tensor5DReduce>(tp, stream, arg);

        qudaEventRecord(event, stream);
        while (!qudaEventQuery(event)) {}
        qudaEventDestroy(event);
      }

      long long flops() const { return 0; }
      long long bytes() const { return in.Bytes() + out.Bytes(); }
    };

    template <class transfer_float, transfer_5D_type transfer_type>
    void tensor_5d_hh(ColorSpinorField &out, const ColorSpinorField &in, device_vector<float> &tp)
    {
      using matrix_type = typename transfer_5D_mapper<transfer_float, transfer_type>::type;

#ifdef GPU_DOMAIN_WALL_DIRAC
      checkLocation(out, in); // check all locations match
      size_t m_size = in.X(4) * out.X(4) * sizeof(matrix_type);
      if (tp.size() * sizeof(float) != m_size) {
        errorQuda("Training Parameter size mismatch %lu neq %lu.\n", tp.size() * sizeof(float), m_size);
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

    template
    void tensor_5d_hh<float, transfer_5D_type::Spin>(ColorSpinorField &out, const ColorSpinorField &in, device_vector<float> &tp);

  } // namespace madwf_ml
} // namespace quda
