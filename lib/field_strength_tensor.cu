#include <tune_quda.h>
#include <gauge_field.h>

#include <jitify_helper.cuh>
#include <kernels/field_strength_tensor.cuh>

namespace quda {

#ifdef GPU_GAUGE_TOOLS

  template<typename Float, typename Arg>
    class FmunuCompute : TunableVectorYZ {
      Arg &arg;
      const GaugeField &meta;
      const QudaFieldLocation location;

    private:
      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; }

    public:
      FmunuCompute(Arg &arg, const GaugeField &meta, QudaFieldLocation location)
        : TunableVectorYZ(2,6), arg(arg), meta(meta), location(location) {
	writeAuxString("threads=%d,stride=%d,prec=%lu",arg.threads,meta.Stride(),sizeof(Float));
        if (location == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
          create_jitify_program("kernels/field_strength_tensor.cuh");
#endif
        }
      }
      virtual ~FmunuCompute() {}

      void apply(const cudaStream_t &stream){
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
          using namespace jitify::reflection;
          jitify_error = program->kernel("quda::computeFmunuKernel")
                             .instantiate(Type<Float>(), Type<Arg>())
                             .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                             .launch(arg);
#else
          computeFmunuKernel<Float><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif
        } else {
          computeFmunuCPU<Float>(arg);
        }
      }

      TuneKey tuneKey() const {
	return TuneKey(meta.VolString(), typeid(*this).name(), aux);
      }

      long long flops() const { return (2430 + 36)*6*2*(long long)arg.threads; }
      long long bytes() const { return ((16*arg.gauge.Bytes() + arg.f.Bytes())*6*2*arg.threads); } //  Ignores link reconstruction

    }; // FmunuCompute


  template<typename Float, typename Fmunu, typename Gauge>
  void computeFmunu(Fmunu f_munu, Gauge gauge, const GaugeField &meta, const GaugeField &meta_ex, QudaFieldLocation location) {
    FmunuArg<Float,Fmunu,Gauge> arg(f_munu, gauge, meta, meta_ex);
    FmunuCompute<Float,FmunuArg<Float,Fmunu,Gauge> > fmunuCompute(arg, meta, location);
    fmunuCompute.apply(0);
    qudaDeviceSynchronize();
    checkCudaError();
  }

  template<typename Float>
  void computeFmunu(GaugeField &Fmunu, const GaugeField &gauge, QudaFieldLocation location) {
    if (Fmunu.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (gauge.isNative()) {
	typedef gauge::FloatNOrder<Float, 18, 2, 18> F;

	if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	  computeFmunu<Float>(F(Fmunu), G(gauge), Fmunu, gauge, location);
	} else if(gauge.Reconstruct() == QUDA_RECONSTRUCT_12) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	  computeFmunu<Float>(F(Fmunu), G(gauge), Fmunu, gauge, location);
	} else if(gauge.Reconstruct() == QUDA_RECONSTRUCT_8) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type G;
	  computeFmunu<Float>(F(Fmunu), G(gauge), Fmunu, gauge, location);
	} else {
	  errorQuda("Reconstruction type %d not supported", gauge.Reconstruct());
	}
      } else {
	errorQuda("Gauge field order %d not supported", gauge.Order());
      }
    } else {
      errorQuda("Fmunu field order %d not supported", Fmunu.Order());
    }
    
  }

#endif // GPU_GAUGE_TOOLS

  void computeFmunu(GaugeField &Fmunu, const GaugeField& gauge, QudaFieldLocation location){

#ifdef GPU_GAUGE_TOOLS
    if (Fmunu.Precision() != gauge.Precision()) {
      errorQuda("Fmunu precision %d must match gauge precision %d", Fmunu.Precision(), gauge.Precision());
    }
    
    if (gauge.Precision() == QUDA_DOUBLE_PRECISION){
      computeFmunu<double>(Fmunu, gauge, location);
    } else if(gauge.Precision() == QUDA_SINGLE_PRECISION) {
      computeFmunu<float>(Fmunu, gauge, location);
    } else {
      errorQuda("Precision %d not supported", gauge.Precision());
    }
    return;
#else
    errorQuda("Fmunu has not been built");
#endif // GPU_GAUGE_TOOLS

  }

} // namespace quda

