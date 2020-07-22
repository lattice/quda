#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <index_helper.cuh>
#include <reduce_helper.cuh>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct GaugePlaqArg : public ReduceArg<double2> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    int threads; // number of active threads required
    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4]; 
    Gauge U;
    
    GaugePlaqArg(const GaugeField &U_) :
      ReduceArg<double2>(),
      U(U_)
    {
      int R = 0;
      for (int dir=0; dir<4; ++dir){
	border[dir] = U_.R()[dir];
	E[dir] = U_.X()[dir];
	X[dir] = U_.X()[dir] - border[dir]*2;
	R += border[dir];
      }
      threads = X[0]*X[1]*X[2]*X[3]/2;
    }
  };

  template<typename Arg>
  __device__ inline double plaquette(Arg &arg, int x[], int parity, int mu, int nu) {
    typedef Matrix<complex<typename Arg::Float>,3> Link;

#if 1
    int dx[4] = {0, 0, 0, 0};
    Link U1 = arg.U(mu, linkIndexShift(x,dx,arg.E), parity);
    dx[mu]++;
    Link U2 = arg.U(nu, linkIndexShift(x,dx,arg.E), 1-parity);
    dx[mu]--;
    dx[nu]++;
    Link U3 = arg.U(mu, linkIndexShift(x,dx,arg.E), 1-parity);
    dx[nu]--;
    Link U4 = arg.U(nu, linkIndexShift(x,dx,arg.E), parity);

    //return getTrace( U1 * U2 * conj(U3) * conj(U4) ).real();
    auto t = getTrace( U1 * U2 * conj(U3) * conj(U4) ).real();
    //printf("U: %g\t%g\t%g\t%g\n", U1(0,0).real(), U2(0,0).real(), U3(0,0).real(), U4(0,0).real());
    //printf("t: %g\n", t);
    return t;
#else
    int dx[4] = {0, 0, 0, 0};
    Link U1 = arg.U(mu, linkIndexShift(x,dx,arg.E), parity);

    dx[mu]++;
    Link U2 = arg.U(nu, linkIndexShift(x,dx,arg.E), 1-parity);
    dx[mu]--;

    Link U12 = U1 * U2;

    dx[nu]++;
    Link U3 = arg.U(mu, linkIndexShift(x,dx,arg.E), 1-parity);
    dx[nu]--;

    Link U123 = U12 * conj(U3);
    Link U4 = arg.U(nu, linkIndexShift(x,dx,arg.E), parity);

    return getTrace( U123 * conj(U4) ).real();
#endif
  }

  template<typename Arg>
  __device__ inline double plaquette2(Arg &arg, int idx, int parity, int mu, int nu) {
    typedef Matrix<complex<typename Arg::Float>,3> Link;

    //int x[4];
    //getCoords(x, idx, arg.X, parity);
#pragma unroll
    //for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates
    //int dx[4] = {0, 0, 0, 0};
    //dx[mu]++;
    //int idxmu = linkIndexShift(x,dx,arg.E);
    //dx[mu]--;
    //dx[nu]++;
    //int idxnu = linkIndexShift(x,dx,arg.E);

    Link U1 = arg.U(mu, idx, parity);
    Link U2 = arg.U(nu, idx, 1-parity);
    Link U3 = arg.U(mu, idx, 1-parity);
    Link U4 = arg.U(nu, idx, parity);

    return getTrace( U1 * U2 * conj(U3) * conj(U4) ).real();
  }

  template<int blockSize, typename Arg>
  __global__ void computePlaq(Arg arg){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int parity = threadIdx.y;

    double2 plaq = make_double2(0.0,0.0);

    while (idx < arg.threads) {
      int x[4];
      getCoords(x, idx, arg.X, parity);
#pragma unroll
      for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

//#pragma unroll
      for (int mu = 0; mu < 3; mu++) {
//#pragma unroll
	for (int nu = 0; nu < 3; nu++) {
	  if (nu >= mu + 1) plaq.x += plaquette(arg, x, parity, mu, nu);
	  //if (nu >= mu + 1) plaq.x += plaquette2(arg, idx, parity, mu, nu);
	}

	plaq.y += plaquette(arg, x, parity, mu, 3);
	//plaq.y += plaquette2(arg, idx, parity, mu, 3);
      }

      idx += blockDim.x*gridDim.x;
    }

#if 0
    {
      typedef Matrix<complex<typename Arg::Float>,3> Link;
      int x[4];
      getCoords(x, 0, arg.X, parity);
      int dx[4] = {0, 0, 0, 0};
      Link U0 = arg.U(0, linkIndexShift(x,dx,arg.E), parity);
      Link U1 = arg.U(1, linkIndexShift(x,dx,arg.E), parity);
      Link U2 = arg.U(2, linkIndexShift(x,dx,arg.E), parity);
      Link U3 = arg.U(3, linkIndexShift(x,dx,arg.E), parity);
      printf("link0: %g\n", U0(0,0).real());
      printf("link1: %g\n", U1(0,0).real());
      printf("link2: %g\n", U2(0,0).real());
      printf("link3: %g\n", U3(0,0).real());
      printf("plaq: %g\n", plaq.x);
    }
#endif
    // perform final inter-block reduction and write out result
    reduce2d<blockSize,2>(arg, plaq);
  }

} // namespace quda
