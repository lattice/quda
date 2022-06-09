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

  void forceRecord(array<double, 2> &force, double dt, const char *fname)
  {
    if (comm_rank()==0) {
      force_stream << fname << "\t" << std::setprecision(5) << force[0] << "\t"
                   << std::setprecision(5) << force[1] << "\t"
                   << std::setprecision(5) << dt << std::endl;
      if (++force_count % force_flush == 0) flushForceMonitor();
    }
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class ActionMom : TunableReduction2D {
    const GaugeField &mom;
    double &action;

  public:
    ActionMom(const GaugeField &mom, double &action) :
      TunableReduction2D(mom),
      mom(mom),
      action(action)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      MomActionArg<Float, nColor, recon> arg(mom);
      launch<MomAction>(action, tp, stream, arg);
    }

    long long flops() const { return mom.Geometry()*mom.Volume()*23; }
    long long bytes() const { return mom.Bytes(); }
  };

  double computeMomAction(const GaugeField& mom) {
    if (!mom.isNative()) errorQuda("Unsupported output ordering: %d\n", mom.Order());
    double action = 0.0;
    instantiate<ActionMom, Reconstruct10>(mom, action);
    return action;
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class UpdateMom : TunableReduction2D {
    using Arg = UpdateMomArg<Float, nColor, recon>;
    const GaugeField &force;
    GaugeField &mom;
    double coeff;
    typename Arg::reduce_t force_max;

  public:
    UpdateMom(const GaugeField &force, GaugeField &mom, double coeff, const char *fname) :
      TunableReduction2D(mom),
      force(force),
      mom(mom),
      coeff(coeff)
    {
      apply(device::get_default_stream());
      if (forceMonitor()) forceRecord(force_max, coeff, fname);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Arg arg(mom, coeff, force);
      launch<MomUpdate>(force_max, tp, stream, arg);
    }

    void preTune() { mom.backup();}
    void postTune() { mom.restore();}
    long long flops() const { return 4 * mom.Volume() * (36+42); }
    long long bytes() const { return 2 * mom.Bytes() + force.Bytes(); }
  };

  void updateMomentum(GaugeField &mom, double coeff, GaugeField &force, const char *fname)
  {
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10)
      errorQuda("Momentum field with reconstruct %d not supported", mom.Reconstruct());

    checkPrecision(mom, force);
    instantiate<UpdateMom, ReconstructMom>(force, mom, coeff, fname);
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
      apply(device::get_default_stream());
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
    if (!force.isNative()) errorQuda("Unsupported output ordering: %d\n", force.Order());
    checkPrecision(force, U);
    instantiate<UApply, ReconstructNo12>(U, force);
  }

} // namespace quda
