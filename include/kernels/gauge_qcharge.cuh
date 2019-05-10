#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>

#include <cub/cub.cuh>
#include <cub_helper.cuh>

#ifndef Pi2
#define Pi2 6.2831853071795864769252867665590
#endif

namespace quda
{

  template <typename Float, typename Gauge> struct QChargeArg : public ReduceArg<double> {
    int threads; // number of active threads required
    Gauge data;

    QChargeArg(const Gauge &data, const GaugeField &Fmunu) : ReduceArg<double>(), data(data), threads(Fmunu.VolumeCB())
    {
    }
  };

  // Core routine for computing the topological charge from the field strength
  template <int blockSize, typename Float, typename Arg> __global__ void qChargeComputeKernel(Arg arg)
  {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    double Q = 0.0;

    while (idx < arg.threads) {
      // Load the field-strength tensor from global memory
      Matrix<complex<Float>, 3> F[6];
      for (int i = 0; i < 6; ++i) F[i] = arg.data(i, idx, parity);

      double Q1 = getTrace(F[0] * F[5]).real();
      double Q2 = getTrace(F[1] * F[4]).real();
      double Q3 = getTrace(F[3] * F[2]).real();
      Q += (Q1 + Q3 - Q2);

      idx += blockDim.x * gridDim.x;
    }
    Q /= (Pi2 * Pi2);

    reduce2d<blockSize, 2>(arg, Q);
  }
} // namespace quda
