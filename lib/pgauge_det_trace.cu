#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <comm_quda.h>
#include <instantiate.h>

#include <tunable_reduction.h>
#include <kernels/gauge_det_trace.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon>
  class CalcFunc : TunableReduction2D<DetTrace> {
    double2 &result;
    const GaugeField &u;
    const int type;

  public:
    CalcFunc(const GaugeField &u, double2 &result, int type) :
      TunableReduction2D(u),
      result(result),
      u(u),
      type(type)
    {
      strcat(aux, type == 0 ? ",det" : ",trace");
      apply(0);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (type == 0) {
        KernelArg<Float, nColor, recon, 0> arg(u);
        launch(tp, stream, arg);
        arg.complete(result, stream);
      } else {
        KernelArg<Float, nColor, recon, 1> arg(u);
        launch(tp, stream, arg);
        arg.complete(result, stream);
      }

      if (!activeTuning()) {
        comm_allreduce_array((double*)&result, 2);
        result.x /= (double)(4*u.LocalVolume()*comm_size());
        result.y /= (double)(4*u.LocalVolume()*comm_size());
      }
    }

    long long flops() const {
      if (u.Ncolor()==3 && type == 0) return 264LL*u.LocalVolume();
      else if (type == 1) return 2*u.Geometry()*u.Ncolor()*u.LocalVolume();
      else return 0;
    }

    long long bytes() const { return u.Bytes(); }
  };

  double2 getLinkDeterminant(GaugeField& data)
  {
    double2 det = make_double2(0.0,0.0);
#ifdef GPU_GAUGE_ALG
    instantiate<CalcFunc>(data, det, 0);
#else
    errorQuda("Pure gauge code has not been built");
#endif // GPU_GAUGE_ALG
    return det;
  }

  double2 getLinkTrace(GaugeField& data)
  {
    double2 det = make_double2(0.0,0.0);
#ifdef GPU_GAUGE_ALG
    instantiate<CalcFunc>(data, det, 1);
#else
    errorQuda("Pure gauge code has not been built");
#endif // GPU_GAUGE_ALG
    return det;
  }

} // namespace quda
