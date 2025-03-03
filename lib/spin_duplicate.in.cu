#include <color_spinor_field.h>
#include <kernels/spin_duplicate.cuh>
#include <tunable_nd.h>
#include <instantiate.h>

namespace quda
{

  template <typename real, int Ns, int Nc> class SpinDuplicate : TunableKernel2D
  {
    cvector_ref<ColorSpinorField> &v;
    const ColorSpinorField &src;
    unsigned int minThreads() const { return src.VolumeCB(); }

  public:
    SpinDuplicate(cvector_ref<ColorSpinorField> &v, const ColorSpinorField &src) :
      TunableKernel2D(src, src.SiteSubset()), v(v), src(src)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<DuplicateSpinor>(tp, stream, SpinorDuplicateArg<real, Ns, Nc>(v, src));
    }

    long long bytes() const { return v.Bytes() + src.Bytes(); }
  };

  template <int...> struct IntList {
  };

  template <typename real, int Ns, int Nc, int... N>
  void spinDuplicate(cvector_ref<ColorSpinorField> &v, const ColorSpinorField &src, IntList<Nc, N...>)
  {
    if (src.Ncolor() == Nc) {
      SpinDuplicate<real, Ns, Nc>(v, src);
    } else {
      if constexpr (sizeof...(N) > 0)
        spinDuplicate<real, Ns>(v, src, IntList<N...>());
      else
        errorQuda("nColor = %d not implemented", src.Ncolor());
    }
  }

  template <typename real> void spinDilute(cvector_ref<ColorSpinorField> &v, const ColorSpinorField &src)
  {
    checkNative(src);
    if (!is_enabled_spin(src.Nspin())) errorQuda("spinDuplicate has not been built for nSpin=%d fields", src.Nspin());

    if (src.Nspin() == 4) {
      if constexpr (is_enabled_spin(4)) spinDuplicate<real, 4>(v, src, IntList<3>());
    } else if (src.Nspin() == 2) {
      if constexpr (is_enabled_spin(2)) spinDuplicate<real, 2>(v, src, IntList<3, @QUDA_MULTIGRID_NVEC_LIST@>());
    } else {
      errorQuda("Nspin = %d not implemented", src.Nspin());
    }
  }

  void spinDuplicate(cvector_ref<ColorSpinorField> &v, const ColorSpinorField &src)
  {
    if (static_cast<int>(v.size()) != src.Nspin()) errorQuda("v size %lu must equal nSpin %d", v.size(), src.Nspin());
    switch (src.Precision()) {
    case QUDA_DOUBLE_PRECISION: spinDilute<double>(v, src); break;
    case QUDA_SINGLE_PRECISION: spinDilute<float>(v, src); break;
    default: errorQuda("Not instantiated %d", src.Precision());
    }
  }

} // namespace quda
