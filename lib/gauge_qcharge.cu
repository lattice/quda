#include <gauge_field.h>
#include <instantiate.h>
#include <tunable_reduction.h>
#include <kernels/gauge_qcharge.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon>
  class QCharge : TunableReduction2D {
    const GaugeField &Fmunu;
    array<real_t, 3> &energy;
    real_t &qcharge;
    void *qdensity;
    bool density;

  public:
    QCharge(const GaugeField &Fmunu, array<real_t, 3> &energy, real_t &qcharge, void *qdensity, bool density) :
      TunableReduction2D(Fmunu),
      Fmunu(Fmunu),
      energy(energy),
      qcharge(qcharge),
      qdensity(qdensity),
      density(density)
    {
      if (!Fmunu.isNative()) errorQuda("Topological charge only supported on native ordered fields");
      apply(device::get_default_stream());
    }

    template <bool compute_density = false> using Arg = QChargeArg<Float, nColor, recon, compute_density>;

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      array<real_t, 3> result{};
      if (!density) {
        Arg<false> arg(Fmunu, static_cast<Float*>(qdensity));
        launch<qCharge>(result, tp, stream, arg);
      } else {
        Arg<true> arg(Fmunu, static_cast<Float*>(qdensity));
        launch<qCharge>(result, tp, stream, arg);
      }

      for (int i=0; i<2; i++) energy[i+1] = result[i] / (Fmunu.Volume() * comm_size());
      energy[0] = energy[1] + energy[2];
      qcharge = result[2];
    }

    long long flops() const
    {
      auto Nc = Fmunu.Ncolor();
      auto mm_flops = 8 * Nc * Nc * (Nc - 2);
      auto traceless_flops = (Nc * Nc + Nc + 1);
      auto energy_flops = 6 * (mm_flops + traceless_flops + Nc);
      auto q_flops = 3*mm_flops + 2 * Nc + 2;
      return Fmunu.Volume() * (energy_flops + q_flops);
    }

    long long bytes() const { return Fmunu.Bytes() + Fmunu.Volume() * (density * Fmunu.Precision()); }
  }; // QChargeCompute

  real_t computeQCharge(array<real_t, 3> &energy, const GaugeField &Fmunu)
  {
    real_t qcharge;
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<QCharge, ReconstructNone>(Fmunu, energy, qcharge, nullptr, false);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    return qcharge;
  }

  real_t computeQChargeDensity(array<real_t, 3> &energy, void *qdensity, const GaugeField &Fmunu)
  {
    real_t qcharge;
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<QCharge, ReconstructNone>(Fmunu, energy, qcharge, qdensity, true);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    return qcharge;
  }

} // namespace quda
