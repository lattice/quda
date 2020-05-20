#include <tune_quda.h>
#include <gauge_field.h>
#include <jitify_helper.cuh>
#include <kernels/field_strength_tensor.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> class Fmunu : TunableVectorYZ
  {
    FmunuArg<Float, nColor, recon> arg;
    const GaugeField &meta;

    unsigned int minThreads() const { return arg.threads; }
    bool tuneGridDim() const { return false; }

public:
    Fmunu(const GaugeField &u, GaugeField &f) :
      TunableVectorYZ(2, 6),
      arg(f, u),
      meta(u)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        create_jitify_program("kernels/field_strength_tensor.cuh");
#endif
      }
      apply(0);
      qudaDeviceSynchronize();
      checkCudaError();
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
      using namespace jitify::reflection;
      jitify_error = program->kernel("quda::computeFmunuKernel").instantiate(Type<decltype(arg)>())
        .configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      computeFmunuKernel<<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return (2430 + 36) * 6 * 2 * (long long)arg.threads; }
    long long bytes() const
    {
      return ((16 * arg.u.Bytes() + arg.f.Bytes()) * 6 * 2 * arg.threads);
    } //  Ignores link reconstruction

  }; // Fmunu

  void computeFmunu(GaugeField &f, const GaugeField &u)
  {
#ifdef GPU_GAUGE_TOOLS
    checkPrecision(f, u);
    instantiate<Fmunu,ReconstructWilson>(u, f); // u must be first here for correct template instantiation
#else
    errorQuda("Gauge tools are not built");
#endif // GPU_GAUGE_TOOLS
  }
} // namespace quda
