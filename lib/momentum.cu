#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <cub_helper.cuh>
#include <instantiate.h>
#include <fstream>

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

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct BaseArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    int threads; // number of active threads required
    BaseArg(const GaugeField &meta) :
      threads(meta.VolumeCB()) {}
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  struct MomActionArg : ReduceArg<double>, BaseArg<Float, nColor, recon> {
    typedef typename gauge_mapper<Float, recon>::type Mom;
    const Mom mom;

    MomActionArg(const GaugeField &mom) :
      BaseArg<Float, nColor, recon>(mom),
      mom(mom) {}
  };

  // calculate the momentum contribution to the action.  This uses the
  // MILC convention where we subtract 4.0 from each matrix norm in
  // order to increase stability
  template <int blockSize, typename Arg>
  __global__ void computeMomAction(Arg arg){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y;
    double action = 0.0;
    using matrix = Matrix<complex<typename Arg::Float>, Arg::nColor>;

    while (x < arg.threads) {
      // loop over direction
      for (int mu=0; mu<4; mu++) {
	const matrix mom = arg.mom(mu, x, parity);

        double local_sum = 0.0;
        local_sum  = 0.5 * mom(0,0).imag() * mom(0,0).imag();
        local_sum += 0.5 * mom(1,1).imag() * mom(1,1).imag();
        local_sum += 0.5 * mom(2,2).imag() * mom(2,2).imag();
        local_sum += mom(0,1).real() * mom(0,1).real();
        local_sum += mom(0,1).imag() * mom(0,1).imag();
        local_sum += mom(0,2).real() * mom(0,2).real();
        local_sum += mom(0,2).imag() * mom(0,2).imag();
        local_sum += mom(1,2).real() * mom(1,2).real();
        local_sum += mom(1,2).imag() * mom(1,2).imag();
	local_sum -= 4.0;

	action += local_sum;
      }

      x += blockDim.x*gridDim.x;
    }

    // perform final inter-block reduction and write out result
    reduce2d<blockSize,2>(arg, action);
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class MomAction : TunableLocalParity {
    MomActionArg<Float, nColor, recon> arg;
    const GaugeField &meta;
    bool tuneGridDim() const { return true; }

  public:
    MomAction(const GaugeField &mom, double &action) :
      arg(mom),
      meta(mom)
    {
      apply(0);
      qudaDeviceSynchronize();
      comm_allreduce((double*)arg.result_h);
      action = arg.result_h[0];
    }

    void apply(const qudaStream_t &stream)
    {
      arg.result_h[0] = 0.0;
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      LAUNCH_KERNEL_LOCAL_PARITY(computeMomAction, (*this), tp, stream, arg, decltype(arg));
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }
    long long flops() const { return 4*2*arg.threads*23; }
    long long bytes() const { return 4*2*arg.threads*arg.mom.Bytes(); }
  };

  double computeMomAction(const GaugeField& mom) {
    double action = 0.0;
#ifdef GPU_GAUGE_TOOLS
    instantiate<MomAction, Reconstruct10>(mom, action);
#else
    errorQuda("%s not build", __func__);
#endif
    return action;
  }

  template<typename Float, int nColor, QudaReconstructType recon>
  struct UpdateMomArg : ReduceArg<double2>, BaseArg<Float, nColor, recon>
  {
    typename gauge_mapper<Float, QUDA_RECONSTRUCT_10>::type mom;
    typename gauge_mapper<Float, recon>::type force;
    Float coeff;
    int X[4]; // grid dimensions on mom
    int E[4]; // grid dimensions on force (possibly extended)
    int border[4]; //
    UpdateMomArg(GaugeField &mom, const Float &coeff, GaugeField &force) :
      BaseArg<Float, nColor, recon>(mom),
      mom(mom),
      coeff(coeff),
      force(force) {
      for (int dir=0; dir<4; ++dir) {
        X[dir] = mom.X()[dir];
        E[dir] = force.X()[dir];
        border[dir] = force.R()[dir];
      }
    }
  };

  /**
     @brief Functor for finding the maximum over a double2 field.
     Each lane of the double2 is evaluated separately.  This functor
     is passed to the reduce helper.
   */
  struct max_reducer2 {
    __device__ __host__ inline  double2 operator()(const double2 &a, const double2 &b) {
      return make_double2(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y);
    }
  };

  template <int blockSize, typename Float, typename Arg>
  __global__ void UpdateMomKernel(Arg arg) {
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = threadIdx.y;
    double2 norm2 = make_double2(0.0,0.0);
    max_reducer2 r;

    while (x_cb<arg.threads) {
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      for (int d=0; d<4; d++) x[d] += arg.border[d];
      int e_cb = linkIndex(x,arg.E);

#pragma unroll
      for (int d=0; d<4; d++) {
	Matrix<complex<Float>,3> m = arg.mom(d, x_cb, parity);
        Matrix<complex<Float>,3> f = arg.force(d, e_cb, parity);

        // project to traceless anti-hermitian prior to taking norm
	makeAntiHerm(f);

        // compute force norms
        norm2 = r(make_double2(f.L1(), f.L2()), norm2);

        m = m + arg.coeff * f;

        // strictly speaking this shouldn't be needed since the
        // momentum should already be traceless anti-hermitian but at
        // present the unit test will fail without this
	makeAntiHerm(m);
	arg.mom(d, x_cb, parity) = m;
      }

      x_cb += gridDim.x*blockDim.x;
    }

    // perform final inter-block reduction and write out result
    reduce2d<blockSize,2,double2,false,max_reducer2>(arg, norm2, 0);
  } // UpdateMom

  template <typename Float, int nColor, QudaReconstructType recon>
  class UpdateMom : TunableLocalParity {
    UpdateMomArg<Float, nColor, recon> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return true; }

  public:
    UpdateMom(GaugeField &force, GaugeField &mom, double coeff, const char *fname) :
      arg(mom, coeff, force),
      meta(force)
    {
      apply(0);
      if (forceMonitor()) forceRecord(*((double2*)arg.result_h), arg.coeff, fname);
    }

    void apply(const qudaStream_t &stream)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	LAUNCH_KERNEL_LOCAL_PARITY(UpdateMomKernel, (*this), tp, stream, arg, Float);
      } else {
	errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString()); }
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
    ApplyUArg(GaugeField  &force, const GaugeField &U) :
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
    int parity = threadIdx.y;
    using mat = Matrix<complex<typename Arg::Float>,Arg::nColor>;

    while (x<arg.threads) {
      for (int d=0; d<4; d++) {
	mat f = arg.force(d, x, parity);
	mat u = arg.U(d, x, parity);

	f = u * f;

	arg.force(d, x, parity) = f;
      }

      x += gridDim.x*blockDim.x;
    }
  } // ApplyU

  template <typename Float, int nColor, QudaReconstructType recon>
  class ApplyU : TunableLocalParity {
    ApplyUArg<Float, nColor, recon> arg;
    const GaugeField &meta;

  public:
    ApplyU(const GaugeField &U, GaugeField &force) :
      arg(force, U),
      meta(U)
    {
      apply(0);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ApplyUKernel<<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
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
