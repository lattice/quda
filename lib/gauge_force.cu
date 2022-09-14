#include <tunable_nd.h>
#include <instantiate.h>
#include <gauge_path_quda.h>
#include <kernels/gauge_force.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon_u, bool compute_force=true> class ForceGauge : public TunableKernel3D
  {
    const GaugeField &u;
    GaugeField &mom;
    double epsilon;
    const paths<4> &p;
    unsigned int minThreads() const { return mom.VolumeCB(); }

  public:
    ForceGauge(const GaugeField &u, GaugeField &mom, double epsilon, const paths<4> &p) :
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
      launch<GaugeForce>(tp, stream, GaugeForceArg<Float, nColor, recon_u,
                         compute_force ? QUDA_RECONSTRUCT_10 : QUDA_RECONSTRUCT_NO, compute_force>(mom, u, epsilon, p));
    }

    void preTune() { mom.backup(); }
    void postTune() { mom.restore(); }

    long long flops() const { return (p.count - p.num_paths + 1) * 198ll * mom.Volume() * 4; }
    long long bytes() const { return (p.count + 1ll) * u.Bytes() + 2 * mom.Bytes(); }
  };

  template<typename Float, int nColor, QudaReconstructType recon_u> using GaugeForce_ = ForceGauge<Float,nColor,recon_u,true>;

  template<typename Float, int nColor, QudaReconstructType recon_u> using GaugePath = ForceGauge<Float,nColor,recon_u,false>;

  void gaugeForce(GaugeField& mom, const GaugeField& u, double epsilon, std::vector<int**>& input_path,
                  std::vector<int>& length, std::vector<double>& path_coeff, int num_paths, int path_max_length)
  {
    checkPrecision(mom, u);
    checkLocation(mom, u);
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10) errorQuda("Reconstruction type %d not supported", mom.Reconstruct());

    paths<4> p(input_path, length, path_coeff, num_paths, path_max_length);

    // gauge field must be passed as first argument so we peel off its reconstruct type
    instantiate<GaugeForce_>(u, mom, epsilon, p);
    p.free();
  }
  
  void gaugePath(GaugeField& out, const GaugeField& u, double coeff, std::vector<int**>& input_path,
		 std::vector<int>& length, std::vector<double>& path_coeff, int num_paths, int path_max_length)
  {
    checkPrecision(out, u);
    checkLocation(out, u);
    if (out.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Reconstruction type %d not supported", out.Reconstruct());

    paths<4> p(input_path, length, path_coeff, num_paths, path_max_length);

    // gauge field must be passed as first argument so we peel off its reconstruct type
    instantiate<GaugePath>(u, out, coeff, p);
    p.free();
  }

} // namespace quda
