#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>

#include <jitify_helper.cuh>
#include <kernels/gauge_stout.cuh>
#include <instantiate.h>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeSTOUT : TunableVectorYZ
  {
    static constexpr int stoutDim = 3; // apply stouting in space only
    GaugeSTOUTArg<Float, nColor, recon, stoutDim> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeSTOUT(GaugeField &out, const GaugeField &in, double rho) :
      TunableVectorYZ(2, stoutDim),
      arg(out, in, rho),
      meta(in)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
#ifdef JITIFY
      create_jitify_program("kernels/gauge_stout.cuh");
#endif
      apply(0);
      qudaDeviceSynchronize();
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
      using namespace jitify::reflection;
      jitify_error = program->kernel("quda::computeSTOUTStep").instantiate(Type<decltype(arg)>())
        .configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      computeSTOUTStep<<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    void preTune() { arg.out.save(); } // defensive measure in case they alias
    void postTune() { arg.out.load(); }

    long long flops() const { return 3 * (2 + 2 * 4) * 198ll * arg.threads; } // just counts matrix multiplication
    long long bytes() const { return ((1 + stoutDim * 6) * arg.in.Bytes() + arg.out.Bytes()) * arg.threads; } // 6 links per dim, 1 in, 1 out.
  }; // GaugeSTOUT
  
  void STOUTStep(GaugeField &out, GaugeField &in, double rho)
  {
#ifdef GPU_GAUGE_TOOLS
    checkPrecision(out, in);
    checkReconstruct(out, in);

    if (!out.isNative()) errorQuda("Order %d with %d reconstruct not supported", in.Order(), in.Reconstruct());
    if (!in.isNative()) errorQuda("Order %d with %d reconstruct not supported", out.Order(), out.Reconstruct());

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    instantiate<GaugeSTOUT>(out, in, rho);
    out.exchangeExtendedGhost(out.R(), false);    
#else
    errorQuda("Gauge tools are not built");
#endif
  }

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeOvrImpSTOUT : TunableVectorYZ
  {
    static constexpr int stoutDim = 4; // apply stouting in all dims
    GaugeSTOUTArg<Float, nColor, recon, stoutDim> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    GaugeOvrImpSTOUT(GaugeField &out, const GaugeField &in, double rho, double epsilon) :
      TunableVectorYZ(2, stoutDim),
      arg(out, in, rho, epsilon),
      meta(in)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
#ifdef JITIFY
      create_jitify_program("kernels/gauge_stout.cuh");
#endif
      apply(0);
      qudaDeviceSynchronize();
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
      using namespace jitify::reflection;
      jitify_error = program->kernel("quda::computeOvrImpSTOUTStep").instantiate(Type<decltype(arg)>())
        .configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      computeOvrImpSTOUTStep<<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    void preTune() { arg.out.save(); } // defensive measure in case they alias
    void postTune() { arg.out.load(); }

    long long flops() const { return 4*(18+2+2*4)*198ll*arg.threads; } // just counts matrix multiplication
    long long bytes() const { return ((1 + stoutDim * 24) * arg.in.Bytes() + arg.out.Bytes()) * arg.threads; } //24 links per dim, 1 in, 1 out
  }; // GaugeOvrImpSTOUT

  void OvrImpSTOUTStep(GaugeField &out, GaugeField& in, double rho, double epsilon)
  {
#ifdef GPU_GAUGE_TOOLS
    checkPrecision(out, in);
    checkReconstruct(out, in);

    if (!out.isNative()) errorQuda("Order %d with %d reconstruct not supported", in.Order(), in.Reconstruct());
    if (!in.isNative()) errorQuda("Order %d with %d reconstruct not supported", out.Order(), out.Reconstruct());

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    instantiate<GaugeOvrImpSTOUT>(out, in, rho, epsilon);
    out.exchangeExtendedGhost(out.R(), false);
    
#else
    errorQuda("Gauge tools are not built");
#endif
  }
}
