#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <cub_helper.cuh>
#include <quda_matrix.h>


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
    
    Link temp;
    
    while (x_cb < arg.threads) {
      // Load the field-strength tensor from global memory
      Matrix<complex<typename Arg::Float>, Arg::nColor> F[] = {arg.f(0, x_cb, parity), arg.f(1, x_cb, parity), arg.f(2, x_cb, parity),
                                                               arg.f(3, x_cb, parity), arg.f(4, x_cb, parity), arg.f(5, x_cb, parity)};

      for(int i=0; i<6; i++) {
	anti_herm_part(F[i], &temp);
	temp *= temp;
	E += getTrace(temp).real();
      }
      x_cb += blockDim.x * gridDim.x;
    }    
    reduce2d<blockSize, 2>(arg, E);
  }  
} // namespace quda
