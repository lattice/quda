#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <cub_helper.cuh>
#include <quda_matrix.h>

#ifndef oneOnThree
#define oneOnThree 0.333333333333333333333333333333
#endif

namespace quda
{

  template <typename Float_, int nColor_, QudaReconstructType recon_> struct EnergyArg :
    public ReduceArg<double>
  {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type F;
    
    int threads; // number of active threads required
    F f;

    EnergyArg(const GaugeField &Fmunu) :
      ReduceArg<double>(),
      f(Fmunu),
      threads(Fmunu.VolumeCB())
    {
    }
  };

  // Core routine for computing the field energy from the field strength
  template <int blockSize, typename Arg> __global__ void energyComputeKernel(Arg arg)
  {
    int x_cb = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    using real = typename Arg::Float;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    
    double E = 0.0;
    complex<real> trace;
    while (x_cb < arg.threads) {
      // Load the field-strength tensor from global memory
      Matrix<complex<typename Arg::Float>, Arg::nColor> F[] = {arg.f(0, x_cb, parity), arg.f(1, x_cb, parity), arg.f(2, x_cb, parity),
                                                               arg.f(3, x_cb, parity), arg.f(4, x_cb, parity), arg.f(5, x_cb, parity)};

      Link temp1, temp2;
      for(int i=0; i<6; i++) {
	// Rescale from 1/8
	temp1 = 8 * F[i];
	trace = oneOnThree * getTrace(temp1);
	setIdentity(&temp2);
	temp2 = trace * temp2;
	temp1 = 0.5 * (temp1 - temp2);
	E += getTrace(temp1 * conj(temp1)).real();
      }
      x_cb += blockDim.x * gridDim.x;
    }
    reduce2d<blockSize, 2>(arg, E);
  }  
} // namespace quda
