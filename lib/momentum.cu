#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field_order.h>
#include <reduce_helper.h>
#include <instantiate.h>
#include <fstream>

#include <tunable_reduction.h>
#include <kernels/momentum.cuh>

namespace quda {

  bool forceMonitor() {
    static bool init = false;
    static bool monitor = false;
    if (!init) {
      char *path = getenv("QUDA_RESOURCE_PATH");
      char *enable_force_monitor = getenv("QUDA_ENABLE_FORCE_MONITOR");
      if (path && enable_force_monitor && strcmp(enable_force_monitor, "1") == 0) monitor = true;
      init = true;
    }
    return monitor;
  }

  static std::stringstream force_stream;
  static long long force_count = 0;
  static long long force_flush = 1000; // how many force samples we accumulate before flushing

  void flushForceMonitor() {
    if (!forceMonitor() || comm_rank() != 0) return;

    static std::string path = std::string(getenv("QUDA_RESOURCE_PATH"));
    static char *profile_fname = getenv("QUDA_PROFILE_OUTPUT_BASE");

    std::ofstream force_file;
    static long long count = 0;
    if (count == 0) {
      path += (profile_fname ? std::string("/") + profile_fname + "_force.tsv" : std::string("/force.tsv"));
      force_file.open(path.c_str());
      force_file << "Force\tL1\tL2\tdt" << std::endl;
    } else {
      force_file.open(path.c_str(), std::ios_base::app);
    }
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Flushing force monitor data to %s\n", path.c_str());
    force_file << force_stream.str();

    force_file.flush();
    force_file.close();

    // empty the stream buffer
    force_stream.clear();
    force_stream.str(std::string());

    count++;
  }

  void forceRecord(double2 &force, double dt, const char *fname) {
    qudaDeviceSynchronize();
    comm_allreduce_max_array((double*)&force, 2);

    if (comm_rank()==0) {
      force_stream << fname << "\t" << std::setprecision(5) << force.x << "\t"
                   << std::setprecision(5) << force.y << "\t"
                   << std::setprecision(5) << dt << std::endl;
      if (++force_count % force_flush == 0) flushForceMonitor();
    }
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class ActionMom : TunableReduction2D<MomAction> {
    MomActionArg<Float, nColor, recon> arg;
    const GaugeField &meta;

  public:
    ActionMom(const GaugeField &mom, double &action) :
      TunableReduction2D(mom),
      arg(mom),
      meta(mom)
    {
      apply(0);
      arg.complete(action);
      comm_allreduce(&action);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch(tp, stream, arg);
    }

    long long flops() const { return 4*2*arg.threads*23; }
    long long bytes() const { return 4*2*arg.threads*arg.mom.Bytes(); }
  };

  double computeMomAction(const GaugeField& mom) {
    if (!mom.isNative()) errorQuda("Unsupported output ordering: %d\n", mom.Order());
    double action = 0.0;
#ifdef GPU_GAUGE_TOOLS
    instantiate<ActionMom, Reconstruct10>(mom, action);
#else
    errorQuda("%s not build", __func__);
#endif
    return action;
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class UpdateMom : TunableReduction2D<MomUpdate> {
    UpdateMomArg<Float, nColor, recon> arg;
    const GaugeField &meta;

  public:
    UpdateMom(GaugeField &force, GaugeField &mom, double coeff, const char *fname) :
      TunableReduction2D(mom),
      arg(mom, coeff, force),
      meta(force)
    {
      double2 force_max;
      apply(0);
      if (forceMonitor()) {
        arg.complete(force_max);
        forceRecord(force_max, arg.coeff, fname);
      }
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<max_reducer2>(tp, stream, arg);
    }

    void preTune() { arg.mom.save();}
    void postTune() { arg.mom.load();}
    long long flops() const { return 4*2*arg.threads*(36+42); }
    long long bytes() const { return 4*2*arg.threads*(2*arg.mom.Bytes()+arg.force.Bytes()); }
  };

  void updateMomentum(GaugeField &mom, double coeff, GaugeField &force, const char *fname)
  {
#ifdef GPU_GAUGE_TOOLS
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10)
      errorQuda("Momentum field with reconstruct %d not supported", mom.Reconstruct());

    checkPrecision(mom, force);
    instantiate<UpdateMom, ReconstructMom>(force, mom, coeff, fname);
    checkCudaError();
#else
    errorQuda("%s not built", __func__);
#endif // GPU_GAUGE_TOOLS
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  struct ApplyUArg : BaseArg<Float, nColor, recon>
  {
    typedef typename gauge_mapper<Float,recon>::type G;
    typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type F;
    F force;
    const G U;
    int X[4]; // grid dimensions
    ApplyUArg(GaugeField &force, const GaugeField &U) :
      BaseArg<Float, nColor, recon>(U),
      force(force),
      U(U)
    {
      for (int dir=0; dir<4; ++dir) X[dir] = U.X()[dir];
    }
  };

  template <typename Arg>
  __global__ void ApplyUKernel(Arg arg)
  {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if (x >= arg.threads) return;
    int parity = threadIdx.y + blockIdx.y * blockDim.y;
    using mat = Matrix<complex<typename Arg::Float>,Arg::nColor>;

    for (int d=0; d<4; d++) {
      mat f = arg.force(d, x, parity);
      mat u = arg.U(d, x, parity);

      f = u * f;

      arg.force(d, x, parity) = f;
    }
  } // ApplyU

  template <typename Float, int nColor, QudaReconstructType recon>
  class ApplyU : TunableVectorY {
    ApplyUArg<Float, nColor, recon> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

  public:
    ApplyU(const GaugeField &U, GaugeField &force) :
      TunableVectorY(2),
      arg(force, U),
      meta(U)
    {
      apply(0);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      qudaLaunchKernel(ApplyUKernel<decltype(arg)>, tp, stream, arg);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }
    void preTune() { arg.force.save();}
    void postTune() { arg.force.load();}
    long long flops() const { return 4*2*arg.threads*198; }
    long long bytes() const { return 4*2*arg.threads*(2*arg.force.Bytes()+arg.U.Bytes()); }
  };

  void applyU(GaugeField &force, GaugeField &U)
  {
#ifdef GPU_GAUGE_TOOLS
    if (!force.isNative()) errorQuda("Unsupported output ordering: %d\n", force.Order());
    checkPrecision(force, U);
    instantiate<ApplyU, ReconstructNo12>(U, force);
    checkCudaError();
#else
    errorQuda("%s not built", __func__);
#endif // GPU_GAUGE_TOOLS
  }

} // namespace quda
