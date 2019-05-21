#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>

#include <launch_kernel.cuh>
#include <jitify_helper.cuh>
#include <kernels/gauge_qcharge.cuh>

namespace quda
{

#ifdef GPU_GAUGE_TOOLS

  template <typename Float, typename Arg> class QChargeCompute : TunableLocalParity
  {
    Arg &arg;
    const GaugeField &meta;

  private:
    bool tuneGridDim() const { return true; }
    unsigned int minThreads() const { return arg.threads; }
    
  public:
    QChargeCompute(Arg &arg, const GaugeField &meta) : arg(arg), meta(meta)
    {
#ifdef JITIFY
      create_jitify_program("kernels/gauge_qcharge.cuh");
#endif
    }
    virtual ~QChargeCompute() {}
    
    void apply(const cudaStream_t &stream)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
        arg.result_h[0] = 0.;
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::qChargeComputeKernel")
	  .instantiate((int)tp.block.x, Type<Float>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
	LAUNCH_KERNEL(qChargeComputeKernel, tp, stream, arg, Float);
#endif
        qudaDeviceSynchronize();
      } else { // run the CPU code
        errorQuda("qChargeComputeKernel not supported on CPU");
      }
    }
    
    TuneKey tuneKey() const
    {
      std::stringstream aux;
      aux << "threads=" << arg.threads << ",prec=" << sizeof(Float);
      return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
    }

    long long flops() const { return 2 * arg.threads * (3 * 198 + 9); }
    long long bytes() const { return 2 * arg.threads * (6 * 18) * sizeof(Float); }
  }; // QChargeCompute

  template <typename Float, typename Gauge> void computeQCharge(const Gauge data, const GaugeField &Fmunu, Float &qChg)
  {
    QChargeArg<Float, Gauge> arg(data, Fmunu);
    QChargeCompute<Float, QChargeArg<Float, Gauge>> qChargeCompute(arg, Fmunu);
    qChargeCompute.apply(0);
    checkCudaError();
    comm_allreduce((double *)arg.result_h);
    qChg = arg.result_h[0];
  }

  template <typename Float> Float computeQCharge(const GaugeField &Fmunu)
  {

    Float qChg = 0.0;

    if (!Fmunu.isNative()) errorQuda("Topological charge computation only supported on native ordered fields");

    if (Fmunu.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type Gauge;
      computeQCharge<Float>(Gauge(Fmunu), Fmunu, qChg);
    } else if (Fmunu.Reconstruct() == QUDA_RECONSTRUCT_12) {
      typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_12>::type Gauge;
      computeQCharge<Float>(Gauge(Fmunu), Fmunu, qChg);
    } else if (Fmunu.Reconstruct() == QUDA_RECONSTRUCT_8) {
      typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_8>::type Gauge;
      computeQCharge<Float>(Gauge(Fmunu), Fmunu, qChg);
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", Fmunu.Reconstruct());
    }    
    return qChg;
  }
#endif

  double computeQCharge(const GaugeField &Fmunu)
  {    
    double qChg = 0.0;
#ifdef GPU_GAUGE_TOOLS
    
    if (!Fmunu.isNative()) errorQuda("Order %d with %d reconstruct not supported", Fmunu.Order(), Fmunu.Reconstruct());
    
    if (Fmunu.Precision() == QUDA_SINGLE_PRECISION) {
      qChg = computeQCharge<float>(Fmunu);
    } else if (Fmunu.Precision() == QUDA_DOUBLE_PRECISION) {
      qChg = computeQCharge<double>(Fmunu);
    } else {
      errorQuda("Precision %d not supported", Fmunu.Precision());
    }
#else
    errorQuda("Gauge tools are not built");
#endif // GPU_GAUGE_TOOLS
    return qChg;
  }    
  
#ifdef GPU_GAUGE_TOOLS
  
  template <typename Float, typename Arg> class QChargeDensityCompute : TunableLocalParity
  {
    Arg &arg;
    const GaugeField &meta;
    Float *qDensity;
    
  private:
    bool tuneGridDim() const { return true; }
    unsigned int minThreads() const { return arg.threads; }
    
  public:
    QChargeDensityCompute(Arg &arg, const GaugeField &meta, Float *qDensity) : arg(arg), meta(meta), qDensity(qDensity)
    {
#ifdef JITIFY
      create_jitify_program("kernels/gauge_qcharge.cuh");
#endif
    }
    virtual ~QChargeDensityCompute() {}
    
    void apply(const cudaStream_t &stream)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
        arg.result_h[0] = 0.;
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::qChargeDensityComputeKernel")
	  .instantiate((int)tp.block.x, Type<Float>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
	LAUNCH_KERNEL(qChargeDensityComputeKernel, tp, stream, arg, Float);
#endif
        qudaDeviceSynchronize();
      } else { // run the CPU code
        errorQuda("qChargeDensityComputeKernel not supported on CPU");
      }
    }
    
    TuneKey tuneKey() const
    {
      std::stringstream aux;
      aux << "threads=" << arg.threads << ",prec=" << sizeof(Float);
      return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
    }

    long long flops() const { return 2 * arg.threads * (3 * 198 + 9); }
    long long bytes() const { return 2 * arg.threads * (6 * 18) * sizeof(Float); }
  }; // QChargeDensityCompute
  
  template <typename Float, typename Gauge> void computeQChargeDensity(const Gauge data, const GaugeField &Fmunu, Float *qDensity, Float &qChg)
  {
    QChargeDensityArg<Float, Gauge> arg(data, Fmunu, qDensity);
    QChargeDensityCompute<Float, QChargeDensityArg<Float, Gauge>> qChargeDensityCompute(arg, Fmunu, qDensity);
    qChargeDensityCompute.apply(0);
    checkCudaError();
    comm_allreduce((double *)arg.result_h);
    qChg = arg.result_h[0];
  }

  template <typename Float> Float computeQChargeDensity(const GaugeField &Fmunu, Float *qDensity)
  {    
    Float qChg = 0.0;

    if (!Fmunu.isNative()) errorQuda("Topological charge computation only supported on native ordered fields");

    if (Fmunu.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type Gauge;
      computeQChargeDensity<Float>(Gauge(Fmunu), Fmunu, qDensity, qChg);
    } else if (Fmunu.Reconstruct() == QUDA_RECONSTRUCT_12) {
      typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_12>::type Gauge;
      computeQChargeDensity<Float>(Gauge(Fmunu), Fmunu, qDensity, qChg);
    } else if (Fmunu.Reconstruct() == QUDA_RECONSTRUCT_8) {
      typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_8>::type Gauge;
      computeQChargeDensity<Float>(Gauge(Fmunu), Fmunu, qDensity, qChg);
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", Fmunu.Reconstruct());
    }

    return qChg;
  }
#endif // GPU_GAUGE_TOOLS

  
  double computeQChargeDensity(const GaugeField &Fmunu, void *qDensity)
  {
    
    double qChg = 0.0;
#ifdef GPU_GAUGE_TOOLS
    
    if (!Fmunu.isNative()) errorQuda("Order %d with %d reconstruct not supported", Fmunu.Order(), Fmunu.Reconstruct());
    
    if (Fmunu.Precision() == QUDA_SINGLE_PRECISION) {
      qChg = computeQChargeDensity<float>(Fmunu, (float*)qDensity);
    } else if (Fmunu.Precision() == QUDA_DOUBLE_PRECISION) {
      qChg = computeQChargeDensity<double>(Fmunu, (double*)qDensity);
    } else {
      errorQuda("Precision %d not supported", Fmunu.Precision());
    }
#else
    errorQuda("Gauge tools are not built");
#endif // GPU_GAUGE_TOOLS
    return qChg;
  }
} // namespace quda
