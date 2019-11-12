#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>

#include <jitify_helper.cuh>
#include <kernels/gauge_ape.cuh>
#include <instantiate.h>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeAPE : TunableVectorYZ
  {
    GaugeAPEArg<Float,nColor,recon> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeAPE(GaugeField &out, const GaugeField &in, double alpha) :
      TunableVectorYZ(2, 3),
      arg(out, in, alpha),
      meta(in)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
#ifdef JITIFY
      create_jitify_program("kernels/gauge_ape.cuh");
#endif
      apply(0);
      qudaDeviceSynchronize();
    }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
      using namespace jitify::reflection;
      jitify_error = program->kernel("quda::computeAPEStep").instantiate(Type<Arg>())
        .configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      computeAPEStep<<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    void preTune() { arg.out.save(); } // defensive measure in case they alias
    void postTune() { arg.out.load(); }

    long long flops() const { return 3 * (2 + 2 * 4) * 198ll * arg.threads; } // just counts matrix multiplication
    long long bytes() const { return 3 * ((1 + 2 * 6) * arg.in.Bytes() + arg.out.Bytes()) * arg.threads; }
  }; // GaugeAPE

  void APEStep(GaugeField &out, const GaugeField& in, double alpha) {
#ifdef GPU_GAUGE_TOOLS
    checkPrecision(out, in);
    checkReconstruct(out, in);

    if (!out.isNative()) errorQuda("Order %d with %d reconstruct not supported", in.Order(), in.Reconstruct());
    if (!in.isNative()) errorQuda("Order %d with %d reconstruct not supported", out.Order(), out.Reconstruct());

    instantiate<GaugeAPE>(out, in, alpha);
#else
    errorQuda("Gauge tools are not built");
#endif
  }
}
