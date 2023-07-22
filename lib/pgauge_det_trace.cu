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
  class CalcFunc : TunableReduction2D {
    const GaugeField &u;
    complex_t &result;
    const compute_type type;

  public:
    CalcFunc(const GaugeField &u, complex_t &result, compute_type type) :
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
      array<real_t, 2> det_trace = {};
      if (type == compute_type::determinant) {
        KernelArg<Float, nColor, recon, compute_type::determinant> arg(u);
        launch<DetTrace>(det_trace, tp, stream, arg);
      } else {
        KernelArg<Float, nColor, recon, compute_type::trace> arg(u);
        launch<DetTrace>(det_trace, tp, stream, arg);
      }

      result = complex_t(det_trace[0], det_trace[1]) / (real_t)(4*u.LocalVolume()*comm_size());
    }

    long long flops() const {
      if (u.Ncolor() == 3 && type == compute_type::determinant) return 264LL*u.LocalVolume();
      else if (type == compute_type::trace) return 2*u.Geometry()*u.Ncolor()*u.LocalVolume();
      else return 0;
    }

    long long bytes() const { return u.Bytes(); }
  };

  complex_t getLinkDeterminant(GaugeField& data)
  {
    complex_t det{0.0, 0.0};
    instantiate<CalcFunc>(data, det, compute_type::determinant);
    return det;
  }

  complex_t getLinkTrace(GaugeField& data)
  {
    complex_t tr{0.0, 0.0};
    instantiate<CalcFunc>(data, tr, compute_type::trace);
    return tr;
  }

} // namespace quda
