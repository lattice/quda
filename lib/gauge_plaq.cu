#include <tune_quda.h>
#include <gauge_field.h>
#include <jitify_helper.cuh>
#include <kernels/gauge_plaq.cuh>
#include <instantiate.h>

namespace quda {

  template<typename Float, int nColor, QudaReconstructType recon>
  class GaugePlaq : TunableLocalParityReduction {
    const GaugeField &u;
    double2 &plq;

  public:
    GaugePlaq(const GaugeField &u, double2 &plq) :
      u(u),
      plq(plq)
    {
#ifdef JITIFY
      create_jitify_program("kernels/gauge_plaq.cuh");
#endif
      strcpy(aux, compile_type_str(u));
      apply(0);
    }

    void apply(const qudaStream_t &stream){
      if (u.Location() == QUDA_CUDA_FIELD_LOCATION) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        GaugePlaqArg<Float, nColor, recon> arg(u);
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::computePlaq")
          .instantiate((int)tp.block.x,type_of(arg))
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
        arg.launch_error = jitify_error == CUDA_SUCCESS ? QUDA_SUCCESS : QUDA_ERROR;
#else
	LAUNCH_KERNEL_LOCAL_PARITY(computePlaq, (*this), tp, stream, arg, decltype(arg));
#endif
        arg.complete(plq);
        if (!activeTuning()) {
          comm_allreduce_array((double*)&plq, 2);
          for (int i = 0; i < 2; i++) ((double*)&plq)[i] /= 9.*2*arg.threads*comm_size();
        }
      } else {
	errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const { return TuneKey(u.VolString(), typeid(*this).name(), aux); }
    long long flops() const
    {
      auto Nc = u.Ncolor();
      return 6ll*u.Volume()*(3 * (8 * Nc * Nc * Nc - 2 * Nc * Nc) + Nc);
    }
    long long bytes() const { return u.Bytes(); }
  };

  double3 plaquette(const GaugeField &U)
  {
    double2 plq;
    instantiate<GaugePlaq>(U, plq);
    double3 plaq = make_double3(0.5*(plq.x + plq.y), plq.x, plq.y);
    return plaq;
  }

} // namespace quda
