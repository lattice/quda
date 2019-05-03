#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>

#define  DOUBLE_TOL	1e-15
#define  SINGLE_TOL	2e-6

#include <jitify_helper.cuh>
#include <kernels/gauge_stout.cuh>

namespace quda {

#ifdef GPU_GAUGE_TOOLS

  template <typename Float, typename Arg> class GaugeSTOUT : TunableVectorYZ
  {
    Arg &arg;
    const GaugeField &meta;

private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeSTOUT(Arg &arg, const GaugeField &meta) : TunableVectorYZ(2, 3), arg(arg), meta(meta)
    {
#ifdef JITIFY
      create_jitify_program("kernels/gauge_stout.cuh");
#endif
    }
    virtual ~GaugeSTOUT() {}

    void apply(const cudaStream_t &stream)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::computeSTOUTStep")
                           .instantiate(Type<Float>(), Type<Arg>())
                           .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                           .launch(arg);
#else
        computeSTOUTStep<Float><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
      } else {
        errorQuda("CPU not supported yet\n");
        // computeSTOUTStepCPU(arg);
      }
    }

    TuneKey tuneKey() const
    {
      std::stringstream aux;
      aux << "threads=" << arg.threads << ",prec=" << sizeof(Float);
      return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
    }

    void preTune() { arg.dest.save(); } // defensive measure in case they alias
    void postTune() { arg.dest.load(); }

    long long flops() const { return 3 * (2 + 2 * 4) * 198ll * arg.threads; } // just counts matrix multiplication
    long long bytes() const { return 3 * ((1 + 2 * 6) * arg.origin.Bytes() + arg.dest.Bytes()) * arg.threads; }
  }; // GaugeSTOUT

  template<typename Float,typename GaugeOr, typename GaugeDs>
  void STOUTStep(GaugeOr origin, GaugeDs dest, const GaugeField& dataOr, Float rho) {
    GaugeSTOUTArg<Float,GaugeOr,GaugeDs> arg(origin, dest, dataOr, rho, dataOr.Precision() == QUDA_DOUBLE_PRECISION ? DOUBLE_TOL : SINGLE_TOL);
    GaugeSTOUT<Float, GaugeSTOUTArg<Float, GaugeOr, GaugeDs>> gaugeSTOUT(arg, dataOr);
    gaugeSTOUT.apply(0);
    qudaDeviceSynchronize();
  }

  template<typename Float>
  void STOUTStep(GaugeField &dataDs, const GaugeField& dataOr, Float rho) {

    if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GDs;

      if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho);
      }else{
	errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
      }
    } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_12){
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GDs;
      if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho);
      }else{
	errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
      }
    } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_8){
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GDs;
      if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GOr;
	STOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho);
      }else{
	errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
    } else {
      errorQuda("Reconstruction type %d of destination gauge field not supported", dataDs.Reconstruct());
    }

  }

#endif

  void STOUTStep(GaugeField &dataDs, const GaugeField& dataOr, double rho) {

#ifdef GPU_GAUGE_TOOLS

    if(dataOr.Precision() != dataDs.Precision()) {
      errorQuda("Origin and destination fields must have the same precision\n");
    }

    if(dataDs.Precision() == QUDA_HALF_PRECISION){
      errorQuda("Half precision not supported\n");
    }

    if (!dataOr.isNative())
      errorQuda("Order %d with %d reconstruct not supported", dataOr.Order(), dataOr.Reconstruct());

    if (!dataDs.isNative())
      errorQuda("Order %d with %d reconstruct not supported", dataDs.Order(), dataDs.Reconstruct());

    if (dataDs.Precision() == QUDA_SINGLE_PRECISION){
      STOUTStep<float>(dataDs, dataOr, (float) rho);
    } else if(dataDs.Precision() == QUDA_DOUBLE_PRECISION) {
      STOUTStep<double>(dataDs, dataOr, rho);
    } else {
      errorQuda("Precision %d not supported", dataDs.Precision());
    }
    return;
#else
  errorQuda("Gauge tools are not build");
#endif
  }

  template <typename Float, typename Arg> class GaugeOvrImpSTOUT : TunableVectorYZ
  {
    Arg &arg;
    const GaugeField &meta;

private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeOvrImpSTOUT(Arg &arg, const GaugeField &meta) : TunableVectorYZ(2, 3), arg(arg), meta(meta) {}
    virtual ~GaugeOvrImpSTOUT() {}

    void apply(const cudaStream_t &stream)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::computeOvrImpSTOUTStep")
                           .instantiate(Type<Float>(), Type<Arg>())
                           .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                           .launch(arg);
#else
        computeOvrImpSTOUTStep<Float><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
      } else {
        errorQuda("CPU not supported yet\n");
        // computeOvrImpSTOUTStepCPU(arg);
      }
    }

    TuneKey tuneKey() const
    {
      std::stringstream aux;
      aux << "threads=" << arg.threads << ",prec=" << sizeof(Float);
      return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
    }

    void preTune() { arg.dest.save(); } // defensive measure in case they alias
    void postTune() { arg.dest.load(); }

    long long flops() const { return 4*(18+2+2*4)*198ll*arg.threads; } // just counts matrix multiplication
    long long bytes() const { return 4*((1+2*12)*arg.origin.Bytes()+arg.dest.Bytes())*arg.threads; }
  }; // GaugeSTOUT

  template<typename Float,typename GaugeOr, typename GaugeDs>
  void OvrImpSTOUTStep(GaugeOr origin, GaugeDs dest, const GaugeField& dataOr, Float rho, Float epsilon) {
    GaugeOvrImpSTOUTArg<Float, GaugeOr, GaugeDs> arg(
        origin, dest, dataOr, rho, epsilon, dataOr.Precision() == QUDA_DOUBLE_PRECISION ? DOUBLE_TOL : SINGLE_TOL);
    GaugeOvrImpSTOUT<Float, GaugeOvrImpSTOUTArg<Float, GaugeOr, GaugeDs>> gaugeOvrImpSTOUT(arg, dataOr);
    gaugeOvrImpSTOUT.apply(0);
    qudaDeviceSynchronize();
  }

  template<typename Float>
  void OvrImpSTOUTStep(GaugeField &dataDs, const GaugeField& dataOr, Float rho, Float epsilon) {

    if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GDs;

      if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GOr;
	OvrImpSTOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, epsilon);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GOr;
	OvrImpSTOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, epsilon);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GOr;
	OvrImpSTOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, epsilon);
      }else{
	errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
      }
    } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_12){
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GDs;
      if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GOr;
	OvrImpSTOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, epsilon);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GOr;
	OvrImpSTOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, epsilon);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GOr;
	OvrImpSTOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, epsilon);
      }else{
	errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
      }
    } else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_8){
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GDs;
      if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_NO){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type GOr;
	OvrImpSTOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, epsilon);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_12){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type GOr;
	OvrImpSTOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, epsilon);
      }else if(dataOr.Reconstruct() == QUDA_RECONSTRUCT_8){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type GOr;
	OvrImpSTOUTStep(GOr(dataOr), GDs(dataDs), dataOr, rho, epsilon);
      }else{
	errorQuda("Reconstruction type %d of origin gauge field not supported", dataOr.Reconstruct());
            }
    } else {
      errorQuda("Reconstruction type %d of destination gauge field not supported", dataDs.Reconstruct());
    }

  }


  void OvrImpSTOUTStep(GaugeField &dataDs, const GaugeField& dataOr, double rho, double epsilon) {

#ifdef GPU_GAUGE_TOOLS

    if(dataOr.Precision() != dataDs.Precision()) {
      errorQuda("Origin and destination fields must have the same precision\n");
    }

    if(dataDs.Precision() == QUDA_HALF_PRECISION){
      errorQuda("Half precision not supported\n");
    }

    if (!dataOr.isNative())
      errorQuda("Order %d with %d reconstruct not supported", dataOr.Order(), dataOr.Reconstruct());

    if (!dataDs.isNative())
      errorQuda("Order %d with %d reconstruct not supported", dataDs.Order(), dataDs.Reconstruct());

    if (dataDs.Precision() == QUDA_SINGLE_PRECISION){
      OvrImpSTOUTStep<float>(dataDs, dataOr, (float) rho, epsilon);
    } else if(dataDs.Precision() == QUDA_DOUBLE_PRECISION) {
      OvrImpSTOUTStep<double>(dataDs, dataOr, rho, epsilon);
    } else {
      errorQuda("Precision %d not supported", dataDs.Precision());
    }
    return;
#else
  errorQuda("Gauge tools are not build");
#endif
  }
}
