#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>

#ifdef JITIFY
// display debugging info
#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE        1
#define JITIFY_PRINT_LOG           1
#define JITIFY_PRINT_PTX           1
#define JITIFY_PRINT_LAUNCH        1
#include <jitify.hpp>
#endif // JITIFY

#include <gauge_plaq.cuh>

namespace quda {

#ifdef JITIFY
  using namespace jitify;
  using namespace jitify::reflection;
  static KernelCache kernel_cache;
  static Program program = kernel_cache.program("/home/kate/github/quda-multireduce/lib/gauge_plaq.cuh", 0, {"-std=c++11"});
#endif

  template<typename Float, typename Gauge>
  class GaugePlaq : TunableLocalParity {

    GaugePlaqArg<Gauge> arg;
    const GaugeField &meta;

  private:
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }

  public:
    GaugePlaq(GaugePlaqArg<Gauge> &arg, const GaugeField &meta)
      : TunableLocalParity(), arg(arg), meta(meta) { }

    ~GaugePlaq () { }

    void apply(const cudaStream_t &stream){
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION){
	for (int i=0; i<2; i++) ((double*)arg.result_h)[i] = 0.0;
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY

#if 1
	jitify_error = (tp.block.x*tp.block.y*tp.block.z > deviceProp.maxThreadsPerBlock) ?
	  CUDA_ERROR_LAUNCH_FAILED  : program.kernel("quda::computePlaq")
	  .instantiate((int)tp.block.x,Type<Float>(),Type<Gauge>())
	  .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
	// use TunableVectorY for this
	// experiment with splitting spatial / temporal components across z blocks
	tp.block.z = 1; tp.grid.z = 2;
	jitify_error = (tp.block.x*tp.block.y*tp.block.z > deviceProp.maxThreadsPerBlock) ?
	  CUDA_ERROR_LAUNCH_FAILED  : program.kernel("quda::computePlaq2")
	  .instantiate((int)tp.block.x,(int)tp.block.y,Type<Float>(),Type<Gauge>())
	  .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#endif

#else
	LAUNCH_KERNEL_LOCAL_PARITY(computePlaq, tp, stream, arg, Float, Gauge);
#endif
      } else {
	errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), "_"); }
    long long flops() const { return 6ll*2*arg.threads*(3*198+3); }
    long long bytes() const { return 6ll*2*arg.threads*4*arg.dataOr.Bytes(); }
  };

  template<typename Float, typename Gauge>
  void plaquette(const Gauge dataOr, const GaugeField& data, double2 &plq, QudaFieldLocation location) {
    GaugePlaqArg<Gauge> arg(dataOr, data);
    GaugePlaq<Float,Gauge> gaugePlaq(arg, data);
    gaugePlaq.apply(0);
    cudaDeviceSynchronize();
    comm_allreduce_array((double*)arg.result_h, 2);
    for (int i=0; i<2; i++) ((double*)&plq)[i] = ((double*)arg.result_h)[i] / (9.*2*arg.threads*comm_size());
  }

  template<typename Float>
  void plaquette(const GaugeField& data, double2 &plq, QudaFieldLocation location) {
    INSTANTIATE_RECONSTRUCT(plaquette<Float>, data, plq, location);
  }

  double3 plaquette(const GaugeField& data, QudaFieldLocation location) {

    double2 plq;
    INSTANTIATE_PRECISION(plaquette, data, plq, location);
    double3 plaq = make_double3(0.5*(plq.x + plq.y), plq.x, plq.y);
    return plaq;

  }

} // namespace quda
