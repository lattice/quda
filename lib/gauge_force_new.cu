#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_force_new.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon_in> class ForceGaugeNew : public TunableKernel3D
  {
    const GaugeField &in;
    GaugeField &mom;
    const QudaGaugeActionType action_type;
    const double epsilon;
    void *path_coeff;
    
    unsigned int minThreads() const { return mom.VolumeCB(); }
    
    // (2,3): 2 for parity in the y thread dim, 4 corresponds to the dimesions over
    // which cache reuse will be used.
  public:
    ForceGaugeNew(const GaugeField &in, GaugeField &mom, const QudaGaugeActionType action_type, const double epsilon, void *path_coeff) :
      TunableKernel3D(in, 2, 4),
      in(in),
      mom(mom),
      action_type(action_type),
      epsilon(epsilon),
      path_coeff(path_coeff)
    {
      switch (action_type) {
      case QUDA_GAUGE_ACTION_TYPE_WILSON: strcat(aux, ",wilson"); break;
      case QUDA_GAUGE_ACTION_TYPE_SYMANZIK: strcat(aux, ",symanzik"); break;
      case QUDA_GAUGE_ACTION_TYPE_LUSCHER_WEISZ: strcat(aux, ",luscher-weisz"); break;
      default: errorQuda("Unexpected action type %d", action_type);
      }
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }
    
    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if(action_type == QUDA_GAUGE_ACTION_TYPE_WILSON) {
	GaugeForceNewArg<Float, nColor, recon_in, 1> arg(mom, in, epsilon, (Float*)path_coeff);
	launch<GaugeForceNew>(tp, stream, arg);
      }
      else if(action_type == QUDA_GAUGE_ACTION_TYPE_SYMANZIK) {
	GaugeForceNewArg<Float, nColor, recon_in, 2> arg(mom, in, epsilon, (Float*)path_coeff);
	launch<GaugeForceNew>(tp, stream, arg);
      }
      else if(action_type == QUDA_GAUGE_ACTION_TYPE_LUSCHER_WEISZ) {
	GaugeForceNewArg<Float, nColor, recon_in, 3> arg(mom, in, epsilon, (Float*)path_coeff);
	launch<GaugeForceNew>(tp, stream, arg);
      }
      else errorQuda("Unknown gauge action type %d", action_type);
    }

    void preTune() { mom.backup(); }
    void postTune() { mom.restore(); }
    
    long long flops() const { return (288ll - 48ll + 1ll) * 198ll * 2 * mom.VolumeCB() * 4; }
    long long bytes() const { return ((288ll + 1ll) * in.Bytes() + 2ll*mom.Bytes()) * 2 * mom.VolumeCB() * 4; }
  };
  
  void gaugeForceNew(GaugeField& mom, const GaugeField& in, const QudaGaugeActionType action_type, const double epsilon, void *path_coeff)
  {
    checkPrecision(mom, in);
    checkLocation(mom, in);
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10) errorQuda("Reconstruction type %d not supported", mom.Reconstruct());    
    instantiate<ForceGaugeNew>(in, mom, action_type, epsilon, path_coeff);
  }
} // namespace quda
