#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>

#include <launch_kernel.cuh>
#include <jitify_helper.cuh>
#include <kernels/gauge_qcharge.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Arg> class QChargeCompute : TunableLocalParity
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

    void apply(const cudaStream_t &stream)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
        arg.result_h[0] = 0.;
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::qChargeComputeKernel")
                         .instantiate((int)tp.block.x, Type<Arg>())
                         .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                         .launch(arg);
#else
	LAUNCH_KERNEL(qChargeComputeKernel, (*this), tp, stream, arg, Arg);
#endif
      } else { // run the CPU code
        errorQuda("qChargeComputeKernel not supported on CPU");
      }
    }

    TuneKey tuneKey() const
    {
      return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString());
    }

    long long flops() const { return 2 * arg.threads * (3 * 198 + 9); }
    long long bytes() const { return 2 * arg.threads * ((6 * 18) + Arg::density) * sizeof(typename Arg::Float); }
  }; // QChargeCompute

  template <typename Float, int nColor, QudaReconstructType recon> struct QCharge {
    QCharge(const GaugeField &Fmunu, double &charge, void *qDensity, bool density)
    {
      if (!Fmunu.isNative()) errorQuda("Topological charge computation only supported on native ordered fields");

      if (density) {
        QChargeArg<Float, nColor, recon, true> arg(Fmunu, (Float*)qDensity);
        QChargeCompute<decltype(arg)> qChargeCompute(arg, Fmunu);
        qChargeCompute.apply(0);
        qudaDeviceSynchronize();

        checkCudaError();
        comm_allreduce((double *)arg.result_h);
        charge = arg.result_h[0];
      } else {
        QChargeArg<Float, nColor, recon, false> arg(Fmunu, (Float*)qDensity);
        QChargeCompute<decltype(arg)> qChargeCompute(arg, Fmunu);
        qChargeCompute.apply(0);
        qudaDeviceSynchronize();

        checkCudaError();
        comm_allreduce((double *)arg.result_h);
        charge = arg.result_h[0];
      }
    }
  };

  double computeQCharge(const GaugeField &Fmunu)
  {
    double charge = 0.0;
#ifdef GPU_GAUGE_TOOLS
    instantiate<QCharge,ReconstructNone>(Fmunu, charge, nullptr, false);
#else
    errorQuda("Gauge tools are not built");
#endif // GPU_GAUGE_TOOLS
    return charge;
  }

  double computeQChargeDensity(const GaugeField &Fmunu, void *qDensity)
  {
    double charge = 0.0;
#ifdef GPU_GAUGE_TOOLS
    instantiate<QCharge,ReconstructNone>(Fmunu, charge, qDensity, true);
#else
    errorQuda("Gauge tools are not built");
#endif // GPU_GAUGE_TOOLS
    return charge;
  }
} // namespace quda
