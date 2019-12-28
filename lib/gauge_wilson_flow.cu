#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>

#include <jitify_helper.cuh>
#include <kernels/gauge_wilson_flow.cuh>
#include <instantiate.h>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeWFlowW1 : TunableVectorYZ
  {
    static constexpr int wFlowDim = 4; // apply Wilson Flow in all dims
    GaugeWFlowArg<Float, nColor, recon, wFlowDim> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeWFlowW1(GaugeField &out, GaugeField &temp, GaugeField &in, double epsilon) :
      TunableVectorYZ(2, wFlowDim),
      arg(out, temp, in, epsilon),
      meta(in)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
#ifdef JITIFY
      create_jitify_program("kernels/gauge_wilson_flow.cuh");
#endif
      apply(0);
      qudaDeviceSynchronize();
    }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
      using namespace jitify::reflection;
      jitify_error = program->kernel("quda::computeWFlowStepW1").instantiate(Type<decltype(arg)>())
        .configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      computeWFlowStepW1<<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    void preTune() { arg.out.save(); } // defensive measure in case they alias
    void postTune() { arg.out.load(); }

    //DMH: FIXME Re-evaluate these
    long long flops() const { return 3 * (2 + 2 * 4) * 198ll * arg.threads; } // just counts matrix multiplication
    long long bytes() const { return 3 * ((1 + 2 * 6) * arg.in.Bytes() + arg.out.Bytes()) * arg.threads; }
  }; // GaugeWFlowW1

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeWFlowW2 : TunableVectorYZ
  {
    static constexpr int wFlowDim = 4; // apply Wilson Flow in all dims
    GaugeWFlowArg<Float, nColor, recon, wFlowDim> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeWFlowW2(GaugeField &out, GaugeField &temp, GaugeField &in, double epsilon) :
      TunableVectorYZ(2, wFlowDim),
      arg(out, temp, in, epsilon),
      meta(in)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
#ifdef JITIFY
      create_jitify_program("kernels/gauge_wilson_flow.cuh");
#endif
      apply(0);
      qudaDeviceSynchronize();
    }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
      using namespace jitify::reflection;
      jitify_error = program->kernel("quda::computeWFlowStepW2").instantiate(Type<decltype(arg)>())
        .configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      computeWFlowStepW2<<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    void preTune() { arg.out.save(); } // defensive measure in case they alias
    void postTune() { arg.out.load(); }

    //DMH: FIXME Re-evaluate these
    long long flops() const { return 3 * (2 + 2 * 4) * 198ll * arg.threads; } // just counts matrix multiplication
    long long bytes() const { return 3 * ((1 + 2 * 6) * arg.in.Bytes() + arg.out.Bytes()) * arg.threads; }
  }; // GaugeWFlowW2

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeWFlowVt : TunableVectorYZ
  {
    static constexpr int wFlowDim = 4; // apply Wilson Flow in all dims
    GaugeWFlowArg<Float, nColor, recon, wFlowDim> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeWFlowVt(GaugeField &out, GaugeField &temp, GaugeField &in, double epsilon) :
      TunableVectorYZ(2, wFlowDim),
      arg(out, temp, in, epsilon),
      meta(in)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
#ifdef JITIFY
      create_jitify_program("kernels/gauge_wilson_flow.cuh");
#endif
      apply(0);
      qudaDeviceSynchronize();
    }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
      using namespace jitify::reflection;
      jitify_error = program->kernel("quda::computeWFlowStepVt").instantiate(Type<decltype(arg)>())
        .configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      computeWFlowStepVt<<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    void preTune() { arg.out.save(); } // defensive measure in case they alias
    void postTune() { arg.out.load(); }

    //DMH: FIXME Re-evaluate these
    long long flops() const { return 3 * (2 + 2 * 4) * 198ll * arg.threads; } // just counts matrix multiplication
    long long bytes() const { return 3 * ((1 + 2 * 6) * arg.in.Bytes() + arg.out.Bytes()) * arg.threads; }
  }; // GaugeWFlow

  void WFlowStep(GaugeField &out, GaugeField &temp, GaugeField &in, double epsilon)
  {
#ifdef GPU_GAUGE_TOOLS
    checkPrecision(out, in);
    checkReconstruct(out, in);

    if (!out.isNative()) errorQuda("Order %d with %d reconstruct not supported", in.Order(), in.Reconstruct());
    if (!in.isNative()) errorQuda("Order %d with %d reconstruct not supported", out.Order(), out.Reconstruct());
    
    instantiate<GaugeWFlowW1>(out, temp, in, epsilon);
    instantiate<GaugeWFlowW2>(in, temp, out, epsilon);
    instantiate<GaugeWFlowVt>(out, temp, in, epsilon);
#else
    errorQuda("Gauge tools are not built");
#endif
  }  
}
