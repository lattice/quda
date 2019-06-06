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
    long long bytes() const { return 2 * arg.threads * ((6 * 18) + Arg::density) * sizeof(Float); }
  }; // QChargeCompute

  template <typename Float, typename Gauge, bool density>
  void computeQCharge(const Gauge data, const GaugeField &Fmunu, Float *qDensity, Float &qChg)
  {
    QChargeArg<Float, Gauge, density> arg(data, Fmunu, qDensity);
    QChargeCompute<Float, decltype(arg)> qChargeCompute(arg, Fmunu);
    qChargeCompute.apply(0);
    checkCudaError();
    comm_allreduce((double *)arg.result_h);
    qChg = arg.result_h[0];
  }

  template <typename Float, bool density> Float computeQCharge(const GaugeField &Fmunu, Float *qDensity = nullptr)
  {
    Float qChg = 0.0;

    if (!Fmunu.isNative()) errorQuda("Topological charge computation only supported on native ordered fields");

    if (Fmunu.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type Gauge;
      computeQCharge<Float, Gauge, density>(Gauge(Fmunu), Fmunu, qDensity, qChg);
    } else if (Fmunu.Reconstruct() == QUDA_RECONSTRUCT_12) {
      typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_12>::type Gauge;
      computeQCharge<Float, Gauge, density>(Gauge(Fmunu), Fmunu, qDensity, qChg);
    } else if (Fmunu.Reconstruct() == QUDA_RECONSTRUCT_8) {
      typedef typename gauge_mapper<Float, QUDA_RECONSTRUCT_8>::type Gauge;
      computeQCharge<Float, Gauge, density>(Gauge(Fmunu), Fmunu, qDensity, qChg);
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", Fmunu.Reconstruct());
    }

    return qChg;
  }
#endif // GPU_GAUGE_TOOLS

  double computeQCharge(const GaugeField &Fmunu)
  {
    double qChg = 0.0;
#ifdef GPU_GAUGE_TOOLS
    if (!Fmunu.isNative()) errorQuda("Order %d with %d reconstruct not supported", Fmunu.Order(), Fmunu.Reconstruct());

    if (Fmunu.Precision() == QUDA_SINGLE_PRECISION) {
      qChg = computeQCharge<float, false>(Fmunu);
    } else if (Fmunu.Precision() == QUDA_DOUBLE_PRECISION) {
      qChg = computeQCharge<double, false>(Fmunu);
    } else {
      errorQuda("Precision %d not supported", Fmunu.Precision());
    }
#else
    errorQuda("Gauge tools are not built");
#endif // GPU_GAUGE_TOOLS
    return qChg;
  }

  double computeQChargeDensity(const GaugeField &Fmunu, void *qDensity)
  {
    double qChg = 0.0;
#ifdef GPU_GAUGE_TOOLS
    if (!Fmunu.isNative()) errorQuda("Order %d with %d reconstruct not supported", Fmunu.Order(), Fmunu.Reconstruct());

    if (Fmunu.Precision() == QUDA_SINGLE_PRECISION) {
      qChg = computeQCharge<float, true>(Fmunu, (float *)qDensity);
    } else if (Fmunu.Precision() == QUDA_DOUBLE_PRECISION) {
      qChg = computeQCharge<double, true>(Fmunu, (double *)qDensity);
    } else {
      errorQuda("Precision %d not supported", Fmunu.Precision());
    }
#else
    errorQuda("Gauge tools are not built");
#endif // GPU_GAUGE_TOOLS
    return qChg;
  }
} // namespace quda
