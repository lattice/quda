#include <tunable_nd.h>
#include <gauge_field.h>
#include <kernels/field_strength_tensor.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> class Fmunu : TunableKernel3D
  {
    GaugeField &f;
    const GaugeField &u;
    unsigned int minThreads() const { return f.VolumeCB(); }
    unsigned int sharedBytesPerThread() const { return 4 * sizeof(int); } // for thread_array

  public:
    Fmunu(const GaugeField &u, GaugeField &f) :
      TunableKernel3D(u, 2, 6),
      f(f),
      u(u)
    {
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<ComputeFmunu>(tp, stream, FmunuArg<Float, nColor, recon>(f, u));
    }

    long long flops() const { return (2430 + 36) * 6 * f.Volume(); }
    long long bytes() const { return ((16 * u.Reconstruct() + f.Reconstruct()) * 6 * f.Volume() * f.Precision()); }
  };

  void computeFmunu(GaugeField &f, const GaugeField &u)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    checkPrecision(f, u);
    instantiate2<Fmunu,ReconstructWilson>(u, f); // u must be first here for correct template instantiation
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
