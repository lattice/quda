#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <index_helper.cuh>
#include <cub_helper.cuh>

namespace quda {

  template <typename Gauge>
  struct GaugePlaqArg : public ReduceArg<double2> {
    int threads; // number of active threads required
    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4]; 
    Gauge dataOr;
    
    GaugePlaqArg(const Gauge &dataOr, const GaugeField &data)
      : ReduceArg<double2>(), dataOr(dataOr)
    {
      int R = 0;
      for (int dir=0; dir<4; ++dir){
	border[dir] = data.R()[dir];
	E[dir] = data.X()[dir];
	X[dir] = data.X()[dir] - border[dir]*2;
	R += border[dir];
      }
      threads = X[0]*X[1]*X[2]*X[3]/2;
    }
  };

  template<typename Float, typename Arg>
  __device__ inline double plaquette(Arg &arg, int x[], int parity, int mu, int nu) {
    typedef Matrix<complex<Float>,3> Link;

    int dx[4] = {0, 0, 0, 0};
    Link U1 = arg.dataOr(mu, linkIndexShift(x,dx,arg.E), parity);
    dx[mu]++;
    Link U2 = arg.dataOr(nu, linkIndexShift(x,dx,arg.E), 1-parity);
    dx[mu]--;
    dx[nu]++;
    Link U3 = arg.dataOr(mu, linkIndexShift(x,dx,arg.E), 1-parity);
    dx[nu]--;
    Link U4 = arg.dataOr(nu, linkIndexShift(x,dx,arg.E), parity);

    return getTrace( U1 * U2 * conj(U3) * conj(U4) ).x;
  }

  template<int blockSize, typename Float, typename Gauge>
  __global__ void computePlaq(GaugePlaqArg<Gauge> arg){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y;

    double2 plaq = make_double2(0.0,0.0);

    if(idx < arg.threads) {
      int x[4];
      getCoords(x, idx, arg.X, parity);
      for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

      for (int mu = 0; mu < 3; mu++) {
	for (int nu = (mu+1); nu < 3; nu++) {
	  plaq.x += plaquette<Float>(arg, x, parity, mu, nu);
	}

	plaq.y += plaquette<Float>(arg, x, parity, mu, 3);
      }
    }
    // perform final inter-block reduction and write out result
    reduce2d<blockSize,2>(arg, plaq);
  }

  // alternative implementation that templates over x and y block
  // dimensions and splits the calculation over more threads
  // x - checker-boarded spacetime
  // y - parity
  // z - spatial / temporal plaquettes
  template<int block_x, int block_y, typename Float, typename Gauge>
  __global__ void computePlaq2(GaugePlaqArg<Gauge> arg){
    int idx    = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y + blockIdx.y*blockDim.y;
    int z      = blockIdx.z;

    double plaq = 0.0;

    if (idx < arg.threads) {
      int x[4];
      getCoords(x, idx, arg.X, parity);
      for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

      switch(z) {
      case 0:
	plaq += plaquette<Float>(arg, x, parity, 0, 1);
	plaq += plaquette<Float>(arg, x, parity, 0, 2);
	plaq += plaquette<Float>(arg, x, parity, 1, 2);
	break;
      case 1:
	plaq += plaquette<Float>(arg, x, parity, 0, 3);
	plaq += plaquette<Float>(arg, x, parity, 1, 3);
	plaq += plaquette<Float>(arg, x, parity, 2, 3);
	break;
      }
    }

    // perform final inter-block reduction and write out result
    reduce2d<block_x,block_y,double,true>(arg, plaq, blockIdx.z);
  }

} // namespace quda
