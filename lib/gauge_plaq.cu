#include <gauge_field.h>
#include <instantiate.h>
#include <tunable_reduction.h>
#include <kernels/gauge_plaq.cuh>

namespace quda {

  template<typename Float, int nColor, QudaReconstructType recon>
  class GaugePlaq : public TunableReduction2D {
    const GaugeField &u;
    array<real_t, 2> &plq;

  public:
    GaugePlaq(const GaugeField &u, array<real_t, 2> &plq) :
      TunableReduction2D(u),
      u(u),
      plq(plq)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugePlaqArg<Float, nColor, recon> arg(u);
      launch<Plaquette>(plq, tp, stream, arg);
      for (int i = 0; i < 2; i++) plq[i] /= 9.*2*arg.threads.x*comm_size();
    }

    long long flops() const
    {
      auto Nc = u.Ncolor();
      return 6ll*u.Volume()*(3 * (8 * Nc * Nc * Nc - 2 * Nc * Nc) + Nc);
    }
    long long bytes() const { return u.Bytes(); }
  };

  array<real_t, 3> plaquette(const GaugeField &U)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    array<real_t, 2> plq{0.0, 0.0};
    instantiate<GaugePlaq, ReconstructGauge>(U, plq);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);

    return{(0.5*(plq[0] + plq[1]), plq[0], plq[1])};
  }

} // namespace quda
