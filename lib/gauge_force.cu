#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_force.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon_u> class ForceGauge : public TunableKernel3D
  {
    const GaugeField &u;
    GaugeField &mom;
    double epsilon;
    const paths &p;
    unsigned int minThreads() const { return mom.VolumeCB(); }

  public:
    ForceGauge(const GaugeField &u, GaugeField &mom, double epsilon, const paths &p) :
      TunableKernel3D(u, 2, 4),
      u(u),
      mom(mom),
      epsilon(epsilon),
      p(p)
    {
      strcat(aux, ",num_paths=");
      strcat(aux, std::to_string(p.num_paths).c_str());
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<GaugeForce>(tp, stream, GaugeForceArg<Float, nColor, recon_u, QUDA_RECONSTRUCT_10>(mom, u, epsilon, p));
    }

    void preTune() { mom.backup(); }
    void postTune() { mom.restore(); }

    long long flops() const { return (p.count - p.num_paths + 1) * 198ll * mom.Volume() * 4; }
    long long bytes() const { return (p.count + 1ll) * u.Bytes() + 2 * mom.Bytes(); }
  };

#ifdef GPU_GAUGE_FORCE
  void gaugeForce(GaugeField& mom, const GaugeField& u, double epsilon, int ***input_path,
                  int *length_h, double *path_coeff_h, int num_paths, int path_max_length)
  {
    checkPrecision(mom, u);
    checkLocation(mom, u);
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10) errorQuda("Reconstruction type %d not supported", mom.Reconstruct());

    // create path struct in a single allocation
    size_t bytes = 4 * num_paths * path_max_length * sizeof(int) + num_paths*sizeof(int) + num_paths*sizeof(double);
    void *buffer = pool_device_malloc(bytes);
    paths p(buffer, bytes, input_path, length_h, path_coeff_h, num_paths, path_max_length);

    // gauge field must be passed as first argument so we peel off its reconstruct type
    instantiate<ForceGauge,ReconstructNo12>(u, mom, epsilon, p);
    pool_device_free(buffer);
  }
#else
  void gaugeForce(GaugeField&, const GaugeField&, double, int ***, int *, double *, int, int)
  {
    errorQuda("Gauge force has not been built");
  }
#endif

} // namespace quda
