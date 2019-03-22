#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <cub_helper.cuh>
#include <fstream>

namespace quda {

using namespace gauge;

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

#ifdef GPU_GAUGE_TOOLS

  template <typename Mom>
  struct MomActionArg : public ReduceArg<double> {
    int threads; // number of active threads required
    Mom mom;
    int X[4]; // grid dimensions
    
    MomActionArg(const Mom &mom, const GaugeField &meta)
      : ReduceArg<double>(), mom(mom) {
      threads = meta.VolumeCB();
      for(int dir=0; dir<4; ++dir) X[dir] = meta.X()[dir];
    }
  };

  template<int blockSize, typename Float, typename Mom>
  __global__ void computeMomAction(MomActionArg<Mom> arg){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y;
    double action = 0.0;
    
    while (x < arg.threads) {
      // loop over direction
      for (int mu=0; mu<4; mu++) {
	Float v[10];
	arg.mom.load(v, x, mu, parity);

	double local_sum = 0.0;
	for (int j=0; j<6; j++) local_sum += v[j]*v[j];
	for (int j=6; j<9; j++) local_sum += 0.5*v[j]*v[j];
	local_sum -= 4.0;
	action += local_sum;
      }

      x += blockDim.x*gridDim.x;
    }
    
    // perform final inter-block reduction and write out result
    reduce2d<blockSize,2>(arg, action);
  }

  template<typename Float, typename Mom>
  class MomAction : TunableLocalParity {
    MomActionArg<Mom> &arg;
    const GaugeField &meta;

  private:
    bool tuneGridDim() const { return true; }

  public:
    MomAction(MomActionArg<Mom> &arg, const GaugeField &meta) : arg(arg), meta(meta) {}
    virtual ~MomAction () { }

    void apply(const cudaStream_t &stream){
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION){
	arg.result_h[0] = 0.0;
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	LAUNCH_KERNEL_LOCAL_PARITY(computeMomAction, tp, stream, arg, Float, Mom);
      } else {
	errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const {
      std::stringstream aux;
      aux << "threads=" << arg.threads << ",prec="  << sizeof(Float);
      return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
    }

    long long flops() const { return 4*2*arg.threads*23; }
    long long bytes() const { return 4*2*arg.threads*arg.mom.Bytes(); }
  };

  template<typename Float, typename Mom>
  void momAction(const Mom mom, const GaugeField& meta, double &action) {
    MomActionArg<Mom> arg(mom, meta);
    MomAction<Float,Mom> momAction(arg, meta);

    momAction.apply(0);
    qudaDeviceSynchronize();

    comm_allreduce((double*)arg.result_h);
    action = arg.result_h[0];
  }
  
  template<typename Float>
  double momAction(const GaugeField& mom) {
    double action = 0.0;
    
    if (mom.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (mom.Reconstruct() == QUDA_RECONSTRUCT_10) {
	momAction<Float>(FloatNOrder<Float,10,2,10>(mom), mom, action);
      } else {
	errorQuda("Reconstruction type %d not supported", mom.Reconstruct());
      }
    } else {
      errorQuda("Gauge Field order %d not supported", mom.Order());
    }
    
    return action;
  }
#endif
  
  double computeMomAction(const GaugeField& mom) {
    double action = 0.0;
#ifdef GPU_GAUGE_TOOLS
    if (mom.Precision() == QUDA_DOUBLE_PRECISION) {
      action = momAction<double>(mom);
    } else if(mom.Precision() == QUDA_SINGLE_PRECISION) {
      action = momAction<float>(mom);
    } else {
      errorQuda("Precision %d not supported", mom.Precision());
    }
#else
    errorQuda("%s not build", __func__);
#endif
    return action;
  }


#ifdef GPU_GAUGE_TOOLS
  template<typename Float, QudaReconstructType reconstruct_>
  struct UpdateMomArg : public ReduceArg<double2> {
    int threads;
    static constexpr int force_recon = (reconstruct_ == QUDA_RECONSTRUCT_10 ? 11 : 18);
    FloatNOrder<Float,18,2,11> mom;
    FloatNOrder<Float,18,2,force_recon> force;
    Float coeff;
    int X[4]; // grid dimensions on mom
    int E[4]; // grid dimensions on force (possibly extended)
    int border[4]; //
    UpdateMomArg(GaugeField &mom, const Float &coeff, GaugeField &force)
      : threads(mom.VolumeCB()), mom(mom), coeff(coeff), force(force) {
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
    __device__ __host__ inline double2 operator()(const double2 &a, const double2 &b) {
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

  
  template<typename Float, typename Arg>
  class UpdateMom : TunableLocalParity {
    Arg &arg;
    const GaugeField &meta;

  private:
    bool tuneGridDim() const { return true; }

  public:
    UpdateMom(Arg &arg, const GaugeField &meta) : arg(arg), meta(meta) {}
    virtual ~UpdateMom () { }

    void apply(const cudaStream_t &stream){
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	LAUNCH_KERNEL_LOCAL_PARITY(UpdateMomKernel, tp, stream, arg, Float);
      } else {
	errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const {
      std::stringstream aux;
      aux << "threads=" << arg.threads << ",prec="  << sizeof(Float);
      return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
    }

    void preTune() { arg.mom.save();}
    void postTune() { arg.mom.load();}
    long long flops() const { return 4*2*arg.threads*(36+42); }
    long long bytes() const { return 4*2*arg.threads*(2*arg.mom.Bytes()+arg.force.Bytes()); }
  };

  template<typename Float, QudaReconstructType reconstruct>
  void updateMomentum(GaugeField &mom, Float coeff, GaugeField &force, const char *fname) {
    UpdateMomArg<Float,reconstruct> arg(mom, coeff, force);
    UpdateMom<Float,decltype(arg)> update(arg, force);
    update.apply(0);

    if (forceMonitor()) forceRecord(*((double2*)arg.result_h), arg.coeff, fname);
  }
  
  template <typename Float>
  void updateMomentum(GaugeField &mom, double coeff, GaugeField &force, const char *fname) {
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10)
      errorQuda("Momentum field with reconstruct %d not supported", mom.Reconstruct());
    if (force.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Force field with order %d not supported", force.Order());

    if (force.Reconstruct() == QUDA_RECONSTRUCT_10) {
      updateMomentum<Float,QUDA_RECONSTRUCT_10>(mom, coeff, force, fname);
    } else if (force.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      updateMomentum<Float,QUDA_RECONSTRUCT_NO>(mom, coeff, force, fname);
    } else {
      errorQuda("Unsupported force reconstruction: %d", force.Reconstruct());
    }
    
  }
#endif // GPU_GAUGE_TOOLS

  void updateMomentum(GaugeField &mom, double coeff, GaugeField &force, const char *fname) {
#ifdef GPU_GAUGE_TOOLS
    if(mom.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Unsupported output ordering: %d\n", mom.Order());

    if (mom.Precision() != force.Precision()) 
      errorQuda("Mixed precision not supported: %d %d\n", mom.Precision(), force.Precision());

    if (mom.Precision() == QUDA_DOUBLE_PRECISION) {
      updateMomentum<double>(mom, coeff, force, fname);
    } else if (mom.Precision() == QUDA_SINGLE_PRECISION) {
      updateMomentum<float>(mom, coeff, force, fname);
    } else {
      errorQuda("Unsupported precision: %d", mom.Precision());
    }      

    checkCudaError();
#else 
    errorQuda("%s not built", __func__);
#endif // GPU_GAUGE_TOOLS

    return;
  }


#ifdef GPU_GAUGE_TOOLS

  template<typename Float, typename Force, typename Gauge>
  struct ApplyUArg {
    int threads;
    Force force;
    Gauge U;
    int X[4]; // grid dimensions
    ApplyUArg(Force &force, Gauge &U, GaugeField &meta)
      : threads(meta.VolumeCB()), force(force), U(U) {
      for (int dir=0; dir<4; ++dir) X[dir] = meta.X()[dir];
    }
  };

  template<typename Float, typename Force, typename Gauge>
  __global__ void ApplyUKernel(ApplyUArg<Float,Force,Gauge> arg) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = threadIdx.y;
    Matrix<complex<Float>,3> f, u;

    while (x<arg.threads) {
      for (int d=0; d<4; d++) {
	arg.force.load(reinterpret_cast<Float*>(f.data), x, d, parity);
	arg.U.load(reinterpret_cast<Float*>(u.data), x, d, parity);

	f = u * f;

	arg.force.save(reinterpret_cast<Float*>(f.data), x, d, parity);
      }

      x += gridDim.x*blockDim.x;
    }

    return;
  } // ApplyU


  template<typename Float, typename Force, typename Gauge>
  class ApplyU : TunableLocalParity {
    ApplyUArg<Float, Force, Gauge> &arg;
    const GaugeField &meta;

  private:
    unsigned int minThreads() const { return arg.threads; }

  public:
    ApplyU(ApplyUArg<Float,Force,Gauge> &arg, const GaugeField &meta) : arg(arg), meta(meta) {}
    virtual ~ApplyU () { }

    void apply(const cudaStream_t &stream){
      if(meta.Location() == QUDA_CUDA_FIELD_LOCATION){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	ApplyUKernel<Float,Force,Gauge><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      } else {
	errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const {
      std::stringstream aux;
      aux << "threads=" << arg.threads << ",prec="  << sizeof(Float);
      return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
    }

    void preTune() { arg.force.save();}
    void postTune() { arg.force.load();}
    long long flops() const { return 4*2*arg.threads*198; }
    long long bytes() const { return 4*2*arg.threads*(2*arg.force.Bytes()+arg.U.Bytes()); }
  };

  template<typename Float, typename Force, typename Gauge>
  void applyU(Force force, Gauge U, GaugeField &meta) {
    ApplyUArg<Float,Force,Gauge> arg(force, U, meta);
    ApplyU<Float,Force,Gauge> applyU(arg, meta);
    applyU.apply(0);
    qudaDeviceSynchronize();
  }
  template <typename Float>
  void applyU(GaugeField &force, GaugeField &U) {
    if (force.Reconstruct() != QUDA_RECONSTRUCT_NO)
      errorQuda("Force field with reconstruct %d not supported", force.Reconstruct());

    if (U.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      applyU<Float>(FloatNOrder<Float, 18, 2, 18>(force), FloatNOrder<Float, 18, 2, 18>(U), force);
    } else if (U.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      applyU<Float>(FloatNOrder<Float, 18, 2, 18>(force), FloatNOrder<Float, 18, 2, 12>(U), force);
    } else {
      errorQuda("Unsupported gauge reconstruction: %d", U.Reconstruct());
    }

  }
#endif // GPU_GAUGE_TOOLS

  void applyU(GaugeField &force, GaugeField &U) {
#ifdef GPU_GAUGE_TOOLS
    if(force.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Unsupported output ordering: %d\n", force.Order());

    if (force.Precision() != U.Precision())
      errorQuda("Mixed precision not supported: %d %d\n", force.Precision(), U.Precision());

    if (force.Precision() == QUDA_DOUBLE_PRECISION) {
      applyU<double>(force, U);
    } else {
      errorQuda("Unsupported precision: %d", force.Precision());
    }

    checkCudaError();
#else
    errorQuda("%s not built", __func__);
#endif // GPU_GAUGE_TOOLS

    return;
  }

} // namespace quda
