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
    const GaugeField &u;
    std::vector<double> &result;
    const compute_type type;

  public:
    CalcFunc(const GaugeField &u, std::vector<double> &result, compute_type type) :
      TunableReduction2D(u),
      u(u),
      result(result),
      type(type)
    {
      strcat(aux, type == compute_type::determinant ? ",det" : ",trace");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (type == compute_type::determinant) {
        KernelArg<Float, nColor, recon, compute_type::determinant> arg(u);
        launch<DetTrace>(result, tp, stream, arg);
      } else {
        KernelArg<Float, nColor, recon, compute_type::trace> arg(u);
        launch<DetTrace>(result, tp, stream, arg);
      }

      for (int i = 0; i < 2; i++) result[i] /= (double)(4*u.LocalVolume()*comm_size());
    }

    long long flops() const {
      if (u.Ncolor() == 3 && type == compute_type::determinant) return 264LL*u.LocalVolume();
      else if (type == compute_type::trace) return 2*u.Geometry()*u.Ncolor()*u.LocalVolume();
      else return 0;
    }

    long long bytes() const { return u.Bytes(); }
  };

#ifdef GPU_GAUGE_ALG
  double2 getLinkDeterminant(GaugeField& data)
  {
    std::vector<double> det{0.0, 0.0};
    instantiate<CalcFunc>(data, det, compute_type::determinant);
    return make_double2(det[0], det[1]);
  }

  double2 getLinkTrace(GaugeField& data)
  {
    std::vector<double> tr{0.0, 0.0};
    instantiate<CalcFunc>(data, tr, compute_type::trace);
    return make_double2(tr[0], tr[1]);
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
