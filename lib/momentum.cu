#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <cub_helper.cuh>

namespace quda {

using namespace gauge;

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
    
    if(x < arg.threads) {  
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
    }
    
    // perform final inter-block reduction and write out result
    reduce2d<blockSize,2>(arg, action);
  }

  template<typename Float, typename Mom>
  class MomAction : TunableLocalParity {
    MomActionArg<Mom> &arg;
    const GaugeField &meta;

  private:
    unsigned int minThreads() const { return arg.threads; }

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
    cudaDeviceSynchronize();

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
  template<typename Float, typename Mom, typename Force>
  struct UpdateMomArg {
    int threads;
    Mom mom;
    Float coeff;
    Force force;
    int X[4]; // grid dimensions
    UpdateMomArg(Mom &mom, const Float &coeff, Force &force, GaugeField &meta)
      : threads(meta.VolumeCB()), mom(mom), coeff(coeff), force(force) {
      for (int dir=0; dir<4; ++dir) X[dir] = meta.X()[dir];
    }
  };

  template<typename Float, typename Mom, typename Force>
  __global__ void UpdateMomKernel(UpdateMomArg<Float, Mom, Force> arg) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = threadIdx.y;
    Matrix<complex<Float>,3> m, f;
    while(x<arg.threads){
      for (int d=0; d<4; d++) {
	arg.mom.load(reinterpret_cast<Float*>(m.data), x, d, parity);
	arg.force.load(reinterpret_cast<Float*>(f.data), x, d, parity);

	m = m + arg.coeff * f;
	makeAntiHerm(m);

	arg.mom.save(reinterpret_cast<Float*>(m.data), x, d, parity); 
      }
      
      x += gridDim.x*blockDim.x;
    }
    return;
  } // UpdateMom

  
  template<typename Float, typename Mom, typename Force>
  class UpdateMom : TunableLocalParity {
    UpdateMomArg<Float, Mom, Force> &arg;
    const GaugeField &meta;

  private:
    unsigned int minThreads() const { return arg.threads; }

  public:
    UpdateMom(UpdateMomArg<Float,Mom,Force> &arg, const GaugeField &meta) : arg(arg), meta(meta) {}
    virtual ~UpdateMom () { }

    void apply(const cudaStream_t &stream){
      if(meta.Location() == QUDA_CUDA_FIELD_LOCATION){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	UpdateMomKernel<Float,Mom,Force><<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
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

  template<typename Float, typename Mom, typename Force>
  void updateMomentum(Mom mom, Float coeff, Force force, GaugeField &meta) {
    UpdateMomArg<Float,Mom,Force> arg(mom, coeff, force, meta);
    UpdateMom<Float,Mom,Force> update(arg, meta);
    update.apply(0);
  }
  
  template <typename Float>
  void updateMomentum(GaugeField &mom, double coeff, GaugeField &force) {
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10)
      errorQuda("Momentum field with reconstruct %d not supported", mom.Reconstruct());

    if (force.Reconstruct() == QUDA_RECONSTRUCT_10) {
      updateMomentum<Float>(FloatNOrder<Float, 18, 2, 11>(mom), static_cast<Float>(coeff),
			      FloatNOrder<Float, 18, 2, 11>(force), force);
    } else if (force.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      updateMomentum<Float>(FloatNOrder<Float, 18, 2, 11>(mom), static_cast<Float>(coeff),
			      FloatNOrder<Float, 18, 2, 18>(force), force);
    } else {
      errorQuda("Unsupported force reconstruction: %d", force.Reconstruct());
    }
    
  }
#endif // GPU_GAUGE_TOOLS

  void updateMomentum(GaugeField &mom, double coeff, GaugeField &force) {
#ifdef GPU_GAUGE_TOOLS
    if(mom.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Unsupported output ordering: %d\n", mom.Order());

    if (mom.Precision() != force.Precision()) 
      errorQuda("Mixed precision not supported: %d %d\n", mom.Precision(), force.Precision());

    if (mom.Precision() == QUDA_DOUBLE_PRECISION) {
      updateMomentum<double>(mom, coeff, force);
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
    cudaDeviceSynchronize();
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
