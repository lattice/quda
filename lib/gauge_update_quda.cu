#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_update.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon_u>
   class UpdateGaugeField : public TunableKernel3D {
    using real = typename mapper<Float>::type;
    static constexpr QudaReconstructType recon_m = QUDA_RECONSTRUCT_10;
    static constexpr int N = 8; // degree of exponential expansion
    GaugeField &out;
    const GaugeField &in;
    const GaugeField &mom;
    const real dt;
    const bool conj_mom;
    const bool exact;
    template <bool conj_mom, bool exact> using Arg =
      UpdateGaugeArg<Float, nColor, recon_u, recon_m, N, conj_mom, exact>;

    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    UpdateGaugeField(GaugeField &out, const GaugeField &in, const GaugeField &mom,
                     double dt, bool conj_mom, bool exact) :
      TunableKernel3D(in, 2, in.Geometry()),
      out(out),
      in(in),
      mom(mom),
      dt(static_cast<real>(dt)),
      conj_mom(conj_mom),
      exact(exact)
    {
      if (conj_mom) strcat(aux, ",conj_mom");
      if (exact) strcat(aux, ",exact");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (conj_mom) {
        if (exact) launch<UpdateGauge>(tp, stream, Arg<true, true>(out, in, mom, dt));
        else       launch<UpdateGauge>(tp, stream, Arg<true, false>(out, in, mom, dt));
      } else {
        if (exact) launch<UpdateGauge>(tp, stream, Arg<false, true>(out, in, mom, dt));
        else       launch<UpdateGauge>(tp, stream, Arg<false, false>(out, in, mom, dt));
      }
    } // apply

    long long flops() const {
      const int Nc = nColor;
      return in.Geometry()*in.Volume()*N*(Nc*Nc*2 +                 // scalar-matrix multiply
                                          (8*Nc*Nc*Nc - 2*Nc*Nc) +  // matrix-matrix multiply
                                          Nc*Nc*2);                 // matrix-matrix addition
    }

    long long bytes() const { return in.Bytes() + out.Bytes() + mom.Bytes(); }
  };

  void updateGaugeField(GaugeField &out, double dt, const GaugeField& in, const GaugeField& mom, bool conj_mom, bool exact)
  {
    checkPrecision(out, in, mom);
    checkLocation(out, in, mom);
    checkReconstruct(out, in);
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10) errorQuda("Reconstruction type %d not supported", mom.Reconstruct());
    instantiate<UpdateGaugeField,ReconstructNo12>(out, in, mom, dt, conj_mom, exact);
  }

} // namespace quda
