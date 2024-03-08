#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/spinor_reweight.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Float, int Ns, int Nc> class SpinorDistanceReweight : TunableKernel2D
  {
    ColorSpinorField &v;
    Float alpha0;
    int t0;
    unsigned int minThreads() const { return v.VolumeCB(); }

  public:
    SpinorDistanceReweight(ColorSpinorField &v, double alpha0, int t0) :
      TunableKernel2D(v, v.SiteSubset()), v(v), alpha0(alpha0), t0(t0)
    {
      strcat(aux, ",cosh");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<DistanceReweightSpinor>(tp, stream, SpinorDistanceReweightArg<Float, Ns, Nc>(v, alpha0, t0));
    }

    long long bytes() const { return 2 * v.Bytes(); }
    void preTune() { v.backup(); }
    void postTune() { v.restore(); }
  };

  template <typename Float> void spinorDistanceReweight(ColorSpinorField &src, double alpha0, int t0)
  {
    if (src.Ncolor() == 3) {
      if (src.Nspin() == 4) {
        if constexpr (is_enabled_spin(4)) SpinorDistanceReweight<Float, 4, 3>(src, alpha0, t0);
      } else if (src.Nspin() == 2) {
        if constexpr (is_enabled_spin(2)) SpinorDistanceReweight<Float, 2, 3>(src, alpha0, t0);
      } else if (src.Nspin() == 1) {
        if constexpr (is_enabled_spin(1)) SpinorDistanceReweight<Float, 1, 3>(src, alpha0, t0);
      } else {
        errorQuda("Nspin = %d not implemented", src.Nspin());
      }
    } else {
      errorQuda("Ncolor = %d not implemented", src.Ncolor());
    }
  }

  void spinorDistanceReweight(ColorSpinorField &src_, double alpha0, int t0)
  {
    // if src is a CPU field then create GPU field
    ColorSpinorField src;
    ColorSpinorParam param(src_);
    bool copy_back = false;
    if (src_.Location() == QUDA_CPU_FIELD_LOCATION || src_.Precision() < QUDA_SINGLE_PRECISION) {
      QudaPrecision prec = std::max(src_.Precision(), QUDA_SINGLE_PRECISION);
      param.setPrecision(prec, prec, true); // change to native field order
      param.create = QUDA_NULL_FIELD_CREATE;
      param.location = QUDA_CUDA_FIELD_LOCATION;
      src = ColorSpinorField(param);
      copy_back = true;
    } else {
      src = src_.create_alias(param);
    }

    if (src.Precision() == QUDA_DOUBLE_PRECISION) {
      spinorDistanceReweight<double>(src, alpha0, t0);
    } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
      spinorDistanceReweight<float>(src, alpha0, t0);
    } else {
      errorQuda("Precision %d not implemented", src.Precision());
    }

    if (copy_back) src_ = src; // copy back if needed
  }

} // namespace quda
