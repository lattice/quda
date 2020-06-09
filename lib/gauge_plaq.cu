#include <tune_quda.h>
#include <gauge_field.h>
#include <jitify_helper.cuh>
#include <kernels/gauge_plaq.cuh>
#include <instantiate.h>

namespace quda {

  template<typename Arg>
  class GaugePlaq : TunableLocalParity {

    Arg &arg;
    const GaugeField &meta;
    bool tuneGridDim() const { return true; }
    unsigned int minGridSize() const { return maxGridSize() / 8; }
    int gridStep() const { return minGridSize(); }

  public:
    GaugePlaq(Arg &arg, const GaugeField &meta) :
      TunableLocalParity(),
      arg(arg),
      meta(meta)
    {
#ifdef JITIFY
      create_jitify_program("kernels/gauge_plaq.cuh");
#endif
      strcpy(aux,compile_type_str(meta));
    }

    void apply(const qudaStream_t &stream){
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION){
	for (int i=0; i<2; i++) ((double*)arg.result_h)[i] = 0.0;
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::computePlaq")
          .instantiate((int)tp.block.x,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
	LAUNCH_KERNEL_LOCAL_PARITY(computePlaq, (*this), tp, stream, arg, Arg);
#endif
      } else {
	errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    long long flops() const { return 6ll*2*arg.threads*(3*198+3); }
    long long bytes() const { return 6ll*2*arg.threads*4*arg.U.Bytes(); }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct Plaquette {
    Plaquette(const GaugeField &U, double2 &plq)
    {
      GaugePlaqArg<Float, nColor, recon> arg(U);
      GaugePlaq<decltype(arg)> gaugePlaq(arg, U);
      gaugePlaq.apply(0);
      qudaDeviceSynchronize();
      comm_allreduce_array((double*)arg.result_h, 2);
      for (int i=0; i<2; i++) ((double*)&plq)[i] = ((double*)arg.result_h)[i] / (9.*2*arg.threads*comm_size());
    }
  };

  double3 plaquette(const GaugeField &U)
  {
    double2 plq;
    instantiate<Plaquette>(U, plq);
    double3 plaq = make_double3(0.5*(plq.x + plq.y), plq.x, plq.y);
    return plaq;
  }

} // namespace quda
