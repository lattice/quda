#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <device_vector.h>
#include <madwf_transfer.h>
#include <kernels/madwf_tensor.cuh>
#include <tunable_reduction.h>
#include <madwf_ml.h>
#include <uint_to_char.h>

namespace quda
{
  namespace madwf_ml
  {

    template <class storage_t> class tensor_5D_wrapper : public TunableMultiReduction
    {
      const ColorSpinorField &x;
      const ColorSpinorField &y;
      using Arg = Tensor5DReduceArg<storage_t, 4, 3>;
      using reduce_t = typename Arg::reduce_t;
      reduce_t *wm_p;

      unsigned int minThreads() const { return x.VolumeCB() / x.X(4); }

    public:
      tensor_5D_wrapper(const ColorSpinorField &x, const ColorSpinorField &y, MadwfAcc::transfer_float *wm_p) :
        TunableMultiReduction(x, 1u, x.X(4) * y.X(4)), x(x), y(y), wm_p(reinterpret_cast<reduce_t *>(wm_p))
      {
        char x_str[16];
        char y_str[16];
        strcat(aux, ",");
        i32toa(x_str, x.X(4));
        strcat(aux, x_str);
        strcat(aux, "x");
        i32toa(y_str, y.X(4));
        strcat(aux, y_str);
        strcat(aux, ",tensor_5D");

        commAsyncReductionSet(true);
        apply(device::get_default_stream());
        commAsyncReductionSet(false);
      }

      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        Arg arg(x, y);
        arg.set_output_async_buffer(reinterpret_cast<reduce_t *>(wm_p));
        std::vector<reduce_t> v; // dummy vector
        launch<Tensor5DReduce>(v, tp, stream, arg);
      }

      long long flops() const { return (18ll + 2ll + 2ll) * x.X(4) * 4ll * y.VolumeCB() * 4ll; }
      long long bytes() const { return y.Bytes() + x.Bytes(); }
    };

#ifdef GPU_DOMAIN_WALL_DIRAC
    void tensor_5d_hh(ColorSpinorField &x, const ColorSpinorField &y,
                      device_vector<MadwfAcc::transfer_float> &transfer_parameter)
    {
      checkLocation(x, y); // check all locations match

      if (y.SiteSubset() != QUDA_PARITY_SITE_SUBSET || x.SiteSubset() != QUDA_PARITY_SITE_SUBSET) {
        errorQuda("ColorSpinorFields are not single parity: y = %d, x = %d", y.SiteSubset(), x.SiteSubset());
      }

      if (x.Ndim() != 5 || y.Ndim() != 5) { errorQuda("we need a 5 dimensional field for this."); }
      if (x.Nspin() != 4 || y.Nspin() != 4) errorQuda("x.nSpin = %d, y.nSpin = %d not supported", x.Nspin(), y.Nspin());
      if (x.Ncolor() != 3 || y.Ncolor() != 3)
        errorQuda("x.nColor = %d, y.nColor = %d, not supported", x.Ncolor(), y.Ncolor());

      using matrix_t = typename transfer_5D_mapper<MadwfAcc::transfer_float, 4, 3, MadwfAcc::transfer_t>::type;

      size_t m_size = y.X(4) * x.X(4) * sizeof(matrix_t);
      if (transfer_parameter.size() * sizeof(float) != m_size) {
        errorQuda("Training Parameter size mismatch %lu neq %lu.", transfer_parameter.size() * sizeof(float), m_size);
      }

      instantiate_madwf<tensor_5D_wrapper>(x, y, transfer_parameter.data());
    }
#else
    void tensor_5d_hh(ColorSpinorField &, const ColorSpinorField &, device_vector<MadwfAcc::transfer_float> &)
    {
      errorQuda("Mobius dslash has not been built");
    }
#endif

  } // namespace madwf_ml
} // namespace quda
