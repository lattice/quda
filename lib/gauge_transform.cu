#include <quda_internal.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <kernels/gauge_transform.cuh>
#include <instantiate.h>

namespace quda {
  
  template <typename Float, int nColor, QudaReconstructType recon> class GaugeTransform : TunableKernel3D
  {
    using real = typename mapper<Float>::type;
    const GaugeField &transformation;
    GaugeField &gauge;
    
    bool tuneSharedBytes() const { return false; }
    unsigned int minThreads() const { return gauge.LocalVolumeCB(); }
    unsigned int maxBlockSize(const TuneParam &) const { return 32; }
    int blockStep() const { return 8; }
    int blockMin() const { return 8; }

  public:
    GaugeTransform(GaugeField &gauge, const GaugeField &transformation) :
      TunableKernel3D(gauge, 2, 4),
      transformation(transformation),
      gauge(gauge)
    {
      strcat(aux, comm_dim_partitioned_string());
      strcat(aux,",computeGaugeTransform");
      apply(device::get_default_stream());
    }
    
     void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugeTransformArg<Float, nColor, recon> arg(gauge, transformation);
      launch<Transform>(tp, stream, arg);
      gauge.exchangeExtendedGhost(gauge.R(),false);
    }
    
    void preTune() { gauge.backup(); }
    void postTune() { gauge.restore(); }
    
    long long flops() const
    {
      // only counts number of mat-muls per thread
      long long threads = gauge.LocalVolume() * 4;
      long long mat_flops = nColor * nColor * (8 * nColor - 2);
      long long mat_muls = 1; // 1 comes from Z * conj(U) term
      return mat_muls * mat_flops * threads;
    }
    
    long long bytes() const
    {
      return 0;
    }
  }; // GaugeTransform
  
#ifdef GPU_GAUGE_TOOLS
  void gaugeTransform(GaugeField &gauge, const GaugeField &tranformation)
  {
    checkPrecision(gauge, tranformation);
    checkNative(gauge, tranformation);    
    instantiate<GaugeTransform>(gauge, tranformation);
  }
#else
  void gaugeTransform(GaugeField &, const GaugeField &)
  {
    errorQuda("Gauge tools are not built");
  }
#endif

}
