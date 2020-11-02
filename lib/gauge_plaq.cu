#include <gauge_field.h>
#include <instantiate.h>
#include <tunable_reduction.h>
#include <kernels/gauge_plaq.cuh>

namespace quda {

  template<typename Float, int nColor, QudaReconstructType recon>
  class GaugePlaq : public TunableReduction2D<> {
    const GaugeField &u;
    double2 &plq;

  public:
    GaugePlaq(const GaugeField &u, double2 &plq) :
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
      if (!activeTuning()) {
        for (int i = 0; i < 2; i++) ((double*)&plq)[i] /= 9.*2*arg.threads.x*comm_size();
      }
    }

    long long flops() const
    {
      auto Nc = u.Ncolor();
      return 6ll*u.Volume()*(3 * (8 * Nc * Nc * Nc - 2 * Nc * Nc) + Nc);
    }
    long long bytes() const { return u.Bytes(); }
  };

  double3 plaquette(const GaugeField &U)
  {
    double2 plq;
    instantiate<GaugePlaq>(U, plq);
    double3 plaq = make_double3(0.5*(plq.x + plq.y), plq.x, plq.y);
    return plaq;
  }

} // namespace quda
