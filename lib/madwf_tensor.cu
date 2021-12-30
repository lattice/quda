#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <device_vector.h>
#include <madwf_transfer.h>
#include <kernels/madwf_tensor.cuh>
#include <tunable_reduction.h>

namespace quda
{
  namespace madwf_ml
  {

    template <class storage_t, class matrix_t> class tensor_5D_wrapper : public TunableMultiReduction<1>
    {
      const ColorSpinorField &x;
      const ColorSpinorField &y;
      matrix_t *wm_p;

    private:
      unsigned int minThreads() const { return x.VolumeCB() / x.X(4); }

    public:
      tensor_5D_wrapper(const ColorSpinorField &x, const ColorSpinorField &y, matrix_t *wm_p) :
        TunableMultiReduction(x, x.X(4) * y.X(4)), x(x), y(y), wm_p(wm_p)
      {
        char tmp[512];
        sprintf(tmp, ",%02d->%02d", y.X(4), x.X(4));
        strcat(aux, tmp);
        strcat(aux, ",tensor_5D");

        commAsyncReductionSet(true);
        apply(device::get_default_stream());
        commAsyncReductionSet(false);
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        using Arg = Tensor5DReduceArg<storage_t>;
        Arg arg(x, y);
        using reduce_t = typename Arg::reduce_t;
        arg.set_device_output_buffer(reinterpret_cast<reduce_t *>(wm_p));
        std::vector<reduce_t> v; // dummy vector
        launch<Tensor5DReduce, reduce_t, comm_reduce_null<reduce_t>>(v, tp, stream, arg);
      }

      long long flops() const { return 8ll * x.X(4) * 4ll * y.VolumeCB(); }
      long long bytes() const { return y.Bytes() + x.Bytes(); }
    };

#ifdef GPU_DOMAIN_WALL_DIRAC
    template <class transfer_float, transfer_5D_t transfer_t>
    void tensor_5d_hh(const ColorSpinorField &x, const ColorSpinorField &y, device_vector<float> &transfer_parameter)
    {
      using matrix_t = typename transfer_5D_mapper<transfer_float, transfer_t>::type;

      checkLocation(x, y); // check all locations match

      if (y.SiteSubset() != QUDA_PARITY_SITE_SUBSET || x.SiteSubset() != QUDA_PARITY_SITE_SUBSET) {
        errorQuda("ColorSpinorFields are not single parity: y = %d, x = %d", y.SiteSubset(), x.SiteSubset());
      }

      size_t m_size = y.X(4) * x.X(4) * sizeof(matrix_t);
      if (transfer_parameter.size() * sizeof(float) != m_size) {
        errorQuda("Training Parameter size mismatch %lu neq %lu.", transfer_parameter.size() * sizeof(float), m_size);
      }

      switch (checkPrecision(x, y)) {
      case QUDA_HALF_PRECISION: {
        tensor_5D_wrapper<short, matrix_t> w(x, y, reinterpret_cast<matrix_t *>(transfer_parameter.data()));
      } break;
      case QUDA_QUARTER_PRECISION: {
        tensor_5D_wrapper<int8_t, matrix_t> w(x, y, reinterpret_cast<matrix_t *>(transfer_parameter.data()));
      } break;
      default: errorQuda("Unsupported precision %d", y.Precision());
      }
    }
#else
    template <class transfer_float, transfer_5D_t transfer_t>
    void tensor_5d_hh(ColorSpinorField &, const ColorSpinorField &, device_vector<float> &)
    {
      errorQuda("Mobius dslash has not been built");
    }
#endif

    template void tensor_5d_hh<float, transfer_5D_t::Spin>(const ColorSpinorField &, const ColorSpinorField &,
                                                           device_vector<float> &);

  } // namespace madwf_ml
} // namespace quda
