#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <cub_helper.cuh>

#ifndef Pi2
#define Pi2 6.2831853071795864769252867665590
#endif

namespace quda
{
  
  template <typename Float, typename Gauge> struct QChargeArg : public ReduceArg<double> {
    int threads; // number of active threads required
    Gauge data;
    
    QChargeArg(const Gauge &data, const GaugeField &Fmunu) :
      ReduceArg<double>(),
      data(data),
      threads(Fmunu.VolumeCB())
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
      Matrix<complex<Float>, 3> F[] = {arg.data(0, idx, parity), arg.data(1, idx, parity), arg.data(2, idx, parity),
                                       arg.data(3, idx, parity), arg.data(4, idx, parity), arg.data(5, idx, parity)};

      double Q1 = getTrace(F[0] * F[5]).real();
      double Q2 = getTrace(F[1] * F[4]).real();
      double Q3 = getTrace(F[3] * F[2]).real();
      Q += (Q1 + Q3 - Q2);

      idx += blockDim.x * gridDim.x;
    }
    Q /= (Pi2 * Pi2);
    
    reduce2d<blockSize, 2>(arg, Q);
  }


  template <typename Float, typename Gauge> struct QChargeDensityArg : public ReduceArg<double> {
    int threads; // number of active threads required
    Gauge data;
    Float *qDensity;
    
    QChargeDensityArg(const Gauge &data, const GaugeField &Fmunu, Float *qDensity) :
      ReduceArg<double>(),
      data(data),
      threads(Fmunu.VolumeCB()),
      qDensity(qDensity)
    {
    }
  };
  
  // Core routine for computing the topological charge from the field strength
  template <int blockSize, typename Float, typename Arg> __global__ void qChargeDensityComputeKernel(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    double Q = 0.0;
    double Q_idx = 0.0;
    int idx;
    
    while (x_cb < arg.threads) {
      // Load the field-strength tensor from global memory
      Matrix<complex<Float>, 3> F[] = {arg.data(0, x_cb, parity), arg.data(1, x_cb, parity), arg.data(2, x_cb, parity),
                                       arg.data(3, x_cb, parity), arg.data(4, x_cb, parity), arg.data(5, x_cb, parity)};
      
      double Q1 = getTrace(F[0] * F[5]).real();
      double Q2 = getTrace(F[1] * F[4]).real();
      double Q3 = getTrace(F[3] * F[2]).real();
      Q_idx = (Q1 + Q3 - Q2);
      Q += Q_idx;
      
      idx = x_cb + parity*arg.threads;
      reinterpret_cast<Float*>(arg.qDensity)[idx] = Q_idx/(Pi2 * Pi2);
      x_cb += blockDim.x * gridDim.x;
    }
    Q /= (Pi2 * Pi2);
    
    reduce2d<blockSize, 2>(arg, Q);
  }  
} // namespace quda
