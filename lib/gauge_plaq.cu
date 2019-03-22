#include <tune_quda.h>
#include <gauge_field.h>
#include <jitify_helper.cuh>
#include <kernels/gauge_plaq.cuh>

namespace quda {

  template<typename Float, typename Gauge>
  class GaugePlaq : TunableLocalParity {

    GaugePlaqArg<Gauge> arg;
    const GaugeField &meta;

  private:
    bool tuneGridDim() const { return true; }

  public:
    GaugePlaq(GaugePlaqArg<Gauge> &arg, const GaugeField &meta)
      : TunableLocalParity(), arg(arg), meta(meta) {
#ifdef JITIFY
      create_jitify_program("kernels/gauge_plaq.cuh");
#endif
      strcpy(aux,compile_type_str(meta));
    }

    ~GaugePlaq () { }

    void apply(const cudaStream_t &stream){
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION){
	for (int i=0; i<2; i++) ((double*)arg.result_h)[i] = 0.0;
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::computePlaq")
          .instantiate((int)tp.block.x,Type<Float>(),Type<Gauge>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
	LAUNCH_KERNEL_LOCAL_PARITY(computePlaq, tp, stream, arg, Float, Gauge);
#endif
      } else {
	errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    long long flops() const { return 6ll*2*arg.threads*(3*198+3); }
    long long bytes() const { return 6ll*2*arg.threads*4*arg.dataOr.Bytes(); }
  };

  template<typename Float, typename Gauge>
  void plaquette(const Gauge dataOr, const GaugeField& data, double2 &plq, QudaFieldLocation location) {
    GaugePlaqArg<Gauge> arg(dataOr, data);
    GaugePlaq<Float,Gauge> gaugePlaq(arg, data);
    gaugePlaq.apply(0);
    qudaDeviceSynchronize();
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
