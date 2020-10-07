#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field_order.h>
#include <reduce_helper.h>
#include <instantiate.h>
#include <fstream>

#include <tunable_reduction.h>
#include <tunable_nd.h>
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
  class ActionMom : TunableReduction2D<> {
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
      launch<MomAction>(tp, stream, arg);
    }

    long long flops() const { return 4*2*arg.threads.x*23; }
    long long bytes() const { return 4*2*arg.threads.x*arg.mom.Bytes(); }
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
  class UpdateMom : TunableReduction2D<> {
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
      launch<MomUpdate, max_reducer2>(tp, stream, arg);
    }

    void preTune() { arg.mom.save();}
    void postTune() { arg.mom.load();}
    long long flops() const { return 4*2*arg.threads.x*(36+42); }
    long long bytes() const { return 4*2*arg.threads.x*(2*arg.mom.Bytes()+arg.force.Bytes()); }
  };

  void updateMomentum(GaugeField &mom, double coeff, GaugeField &force, const char *fname)
  {
#ifdef GPU_GAUGE_TOOLS
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10)
      errorQuda("Momentum field with reconstruct %d not supported", mom.Reconstruct());

    checkPrecision(mom, force);
    instantiate<UpdateMom, ReconstructMom>(force, mom, coeff, fname);
#else
    errorQuda("%s not built", __func__);
#endif // GPU_GAUGE_TOOLS
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class UApply : TunableKernel2D {
    const GaugeField &U;
    GaugeField &force;
    unsigned int minThreads() const { return U.LocalVolumeCB(); }

  public:
    UApply(const GaugeField &U, GaugeField &force) :
      TunableKernel2D(U, 2),
      U(U),
      force(force)
    {
      apply(0);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ApplyUArg<Float, nColor, recon> arg(force, U);
      launch<ApplyU>(tp, stream, arg);
    }

    void preTune() { force.backup(); }
    void postTune() { force.restore(); }
    long long flops() const
    {
      int Nc = U.Ncolor();
      return 4 * U.Volume() * (8 * Nc * Nc * Nc - 2 * Nc * Nc);
    }
    long long bytes() const { return 2 * force.Bytes() + U.Bytes(); }
  };

  void applyU(GaugeField &force, GaugeField &U)
  {
#ifdef GPU_GAUGE_TOOLS
    if (!force.isNative()) errorQuda("Unsupported output ordering: %d\n", force.Order());
    checkPrecision(force, U);
    instantiate<UApply, ReconstructNo12>(U, force);
#else
    errorQuda("%s not built", __func__);
#endif // GPU_GAUGE_TOOLS
  }

} // namespace quda
