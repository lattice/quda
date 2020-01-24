#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <cub_helper.cuh>

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_, bool density_ = false> struct QChargeArg :
    public ReduceArg<double>
  {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr bool density = density_;
    typedef typename gauge_mapper<Float,recon>::type F;
    static constexpr Float norm = -1.0 / (4*M_PI*M_PI);
    
    int threads; // number of active threads required
    F f;
    Float *qDensity;

    
    QChargeArg(const GaugeField &Fmunu, Float *qDensity = nullptr) :
      ReduceArg<double>(),
      f(Fmunu),
      threads(Fmunu.VolumeCB()),
      qDensity(qDensity)
    {
    }
  };

  // Core routine for computing the topological charge from the field strength
  template <int blockSize, typename Arg> __global__ void qChargeComputeKernel(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    double Q = 0.0;
    using real = typename Arg::Float;
    typedef Matrix<complex<real>, Arg::nColor> Link;

    while (x_cb < arg.threads) {
      // Load the field-strength tensor from global memory
      Link F[] = {arg.f(0, x_cb, parity), arg.f(1, x_cb, parity), arg.f(2, x_cb, parity),
		  arg.f(3, x_cb, parity), arg.f(4, x_cb, parity), arg.f(5, x_cb, parity)};

      //F1 = F[Y,X], F2 = F[Z,X], F3 = F[Z,Y], 
      //F4 = F[T,X], F5 = F[T,Y], F6 = F[T,Z] 
      double Q1 = getTrace(F[0] * F[5]).real(); 
      double Q2 = getTrace(conj(F[1]) * F[4]).real();
      double Q3 = getTrace(F[3] * F[2]).real();
      double Q_idx = (Q1 + Q3 + Q2);
      Q += Q_idx * Arg::norm;
      if (Arg::density) arg.qDensity[x_cb + parity * arg.threads] = Q_idx * Arg::norm;
      
      x_cb += blockDim.x * gridDim.x;
    }
    
    reduce2d<blockSize, 2>(arg, Q);
  }

} // namespace quda
