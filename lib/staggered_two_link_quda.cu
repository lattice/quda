#include <utility>
#include <quda_internal.h>
#include <gauge_field.h>
#include <ks_improved_force.h>
#include <tune_quda.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/staggered_two_link.cuh>

namespace quda
{

  using namespace staggered_quark_smearing;

  template <typename real, int nColor, QudaReconstructType recon>
  class ComputeTwoLink : public TunableKernel3D
  {
    GaugeField &twoLink;
    const GaugeField &link;
    unsigned int minThreads() const { return twoLink.VolumeCB(); }

  public:
    ComputeTwoLink(GaugeField &twoLink, const GaugeField &link) :
      TunableKernel3D(link, 2, 4),
      twoLink(twoLink),
      link(link)
    {
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<TwoLink>(tp, stream, TwoLinkArg<real, nColor, recon>(twoLink, link));
    }

    long long flops() const { return 4 * twoLink.Volume() * nColor * nColor * (8 * nColor - 2); }
    long long bytes() const { return 2 * link.Bytes() + twoLink.Bytes(); }
  };

  void computeTwoLink(GaugeField &twoLink, const GaugeField &link)
  {
    if constexpr (is_enabled<QUDA_STAGGERED_DSLASH>()) {
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
      checkNative(twoLink, link);
      checkLocation(twoLink, link);
      checkPrecision(twoLink, link);
      //FIXME: enable link-12/8 reconstruction
      instantiate<ComputeTwoLink, ReconstructNone>(twoLink, link);
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    } else {
      errorQuda("Two-link computation requires staggered operator to be enabled");
    }
  }

} // namespace quda
