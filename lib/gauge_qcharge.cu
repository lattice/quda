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
    bool tuneSharedBytes() const { return false; }
    bool tuneGridDim() const { return true; }
    unsigned int minThreads() const { return arg.threads; }

  public:
    QChargeCompute(Arg &arg, const GaugeField &meta) :
      TunableLocalParity(),
      arg(arg),
      meta(meta)
    {
#ifdef JITIFY
      create_jitify_program("kernels/gauge_qcharge.cuh");
#endif
    }

    void apply(const qudaStream_t &stream)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
	for (int i=0; i<4; i++) ((double*)arg.result_h)[i] = 0.0;
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::qChargeComputeKernel")
                         .instantiate((int)tp.block.x, Type<Arg>())
                         .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                         .launch(arg);
#else
	LAUNCH_KERNEL_LOCAL_PARITY(qChargeComputeKernel, (*this), tp, stream, arg, Arg);
#endif
      } else { // run the CPU code
        errorQuda("qChargeComputeKernel not supported on CPU");
      }
    }

    TuneKey tuneKey() const
    {
      return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString());
    }

    long long flops() const
    {
      auto mm_flops = 8 * Arg::nColor * Arg::nColor * (Arg::nColor - 2);
      auto traceless_flops = (Arg::nColor * Arg::nColor + Arg::nColor + 1);
      auto energy_flops = 6 * (mm_flops + traceless_flops + Arg::nColor);
      auto q_flops = 3*mm_flops + 2*Arg::nColor + 2;
      return 2ll * arg.threads * (energy_flops + q_flops);
    }

    long long bytes() const { return 2 * arg.threads * (6 * arg.f.Bytes() + Arg::density * sizeof(typename Arg::Float)); }
  }; // QChargeCompute

  template <typename Float, int nColor, QudaReconstructType recon> struct QCharge {
    QCharge(const GaugeField &Fmunu, double energy[3], double &qcharge, void *qdensity, bool density)
    {
      if (!Fmunu.isNative()) errorQuda("Topological charge computation only supported on native ordered fields");

      if (density) {
        QChargeArg<Float, nColor, recon, true> arg(Fmunu, (Float*)qdensity);
        QChargeCompute<decltype(arg)> qChargeCompute(arg, Fmunu);
        qChargeCompute.apply(0);
        qudaDeviceSynchronize();

        comm_allreduce_array((double *)arg.result_h, 3);
        for (int i=0; i<2; i++) energy[i+1] = ((double*)arg.result_h)[i] / (2.0*arg.threads*comm_size());
        qcharge = ((double*)arg.result_h)[2];
      } else {
        QChargeArg<Float, nColor, recon, false> arg(Fmunu, (Float*)qdensity);
        QChargeCompute<decltype(arg)> qChargeCompute(arg, Fmunu);
        qChargeCompute.apply(0);
        qudaDeviceSynchronize();

        comm_allreduce_array((double *)arg.result_h, 3);
        for (int i=0; i<2; i++) energy[i+1] = ((double*)arg.result_h)[i] / (2.0*arg.threads*comm_size());
        qcharge = ((double*)arg.result_h)[2];
      }

      energy[0] = energy[1] + energy[2];
    }
  };

  void computeQCharge(double energy[3], double &qcharge, const GaugeField &Fmunu)
  {
#ifdef GPU_GAUGE_TOOLS
    instantiate<QCharge,ReconstructNone>(Fmunu, energy, qcharge, nullptr, false);
#else
    errorQuda("Gauge tools are not built");
#endif // GPU_GAUGE_TOOLS
  }

  void computeQChargeDensity(double energy[3], double &qcharge, void *qdensity, const GaugeField &Fmunu)
  {
#ifdef GPU_GAUGE_TOOLS
    instantiate<QCharge,ReconstructNone>(Fmunu, energy, qcharge, qdensity, true);
#else
    errorQuda("Gauge tools are not built");
#endif // GPU_GAUGE_TOOLS
  }
} // namespace quda
