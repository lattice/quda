#include <gauge_field.h>
#include <instantiate.h>
#include <tunable_reduction.h>
#include <kernels/gauge_plaqrect.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> class GaugePlaqRect : public TunableReduction2D
  {
    const GaugeField &u;
    array<double, 4> &plqrct;

  public:
    GaugePlaqRect(const GaugeField &u, array<double, 4> &plqrct) : TunableReduction2D(u), u(u), plqrct(plqrct)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugePlaqRectArg<Float, nColor, recon> arg(u);
      launch<PlaquetteRectangle>(plqrct, tp, stream, arg);
      // Normalize plaquette and rectangle
      for (int i = 0; i < 2; i++) plqrct[i] /= 9. * 2 * arg.threads.x * comm_size();
      for (int i = 2; i < 4; i++) plqrct[i] /= 9. * 4 * arg.threads.x * comm_size();
    }

    long long flops() const
    {
      auto Nc = u.Ncolor();
      // 10 mat-mat multiplies to compute plaquette and 2 rectangles
      // Nc * Nc * (8 * Nc - 2) flops per mat-mat multiply
      // Plus 2 traces ~ 2 * Nc flops
      // All of the above * 6 (number of planes) * volume
      return 6ll * u.Volume() * (10 * Nc * Nc * (8 * Nc - 2) + 2 * Nc);
    }
    long long bytes() const { return u.Bytes(); }
  };

  double4 plaquetteRectangle(const GaugeField &U)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    array<double, 4> plqrct {0.0, 0.0, 0.0, 0.0};
    instantiate<GaugePlaqRect, ReconstructGauge>(U, plqrct);
    double4 plaqrect = {plqrct[0], plqrct[1], plqrct[2], plqrct[3]};
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    return plaqrect;
  }

} // namespace quda
