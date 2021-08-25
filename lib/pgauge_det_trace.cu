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
  class CalcFunc : TunableReduction2D<> {
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
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (type == 0) {
        KernelArg<Float, nColor, recon, 0> arg(u);
        launch<DetTrace>(result, tp, stream, arg);
      } else {
        KernelArg<Float, nColor, recon, 1> arg(u);
        launch<DetTrace>(result, tp, stream, arg);
      }

      result.x /= (double)(4*u.LocalVolume()*comm_size());
      result.y /= (double)(4*u.LocalVolume()*comm_size());
    }

    long long flops() const {
      if (u.Ncolor()==3 && type == 0) return 264LL*u.LocalVolume();
      else if (type == 1) return 2*u.Geometry()*u.Ncolor()*u.LocalVolume();
      else return 0;
    }

    long long bytes() const { return u.Bytes(); }
  };

#ifdef GPU_GAUGE_ALG
  double2 getLinkDeterminant(GaugeField& data)
  {
    double2 det = make_double2(0.0,0.0);
    instantiate<CalcFunc>(data, det, 0);
    return det;
  }

  double2 getLinkTrace(GaugeField& data)
  {
    double2 det = make_double2(0.0,0.0);
    instantiate<CalcFunc>(data, det, 1);
    return det;
  }
#else
  double2 getLinkDeterminant(GaugeField&)
  {
    errorQuda("Pure gauge code has not been built");
    return make_double2(0.0,0.0);
  }

  double2 getLinkTrace(GaugeField&)
  {
    errorQuda("Pure gauge code has not been built");
    return make_double2(0.0,0.0);
  }
#endif

} // namespace quda
