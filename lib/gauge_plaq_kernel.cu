//#include <cub_helper.cuh>
#include <index_helper.cuh>
#include <quda_matrix.h>

template <typename G>
struct GaugePlaqArg { //: public ReduceArg<double2> {
  int threads; // number of active threads required
  int X[4]; // grid dimensions
#ifdef MULTI_GPU
  int border[4]; 
#endif
  G dataOr;
  
#ifndef __CUDACC_RTC__
  GaugePlaqArg(const G &dataOr, const GaugeField &data)
    : /*ReduceArg<double2>(),*/ dataOr(dataOr)
  {
    
#ifdef MULTI_GPU
    for(int dir=0; dir<4; ++dir){
      border[dir] = 2;
      X[dir] = data.X()[dir] - border[dir]*2;
    }
#else
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
#endif
    threads = X[0]*X[1]*X[2]*X[3]/2;
  }
#endif

};

#ifndef __CUDACC_RTC__
  template<int blockSize, typename Real, typename Gauge>
#endif
  __global__ void computePlaq(GaugePlaqArg<Gauge> arg){

      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      int parity = threadIdx.y;

      printf("threads = %d\n", arg.threads);

      /*
      double2 plaq = make_double2(0.0,0.0);

      if(idx < arg.threads) {
        typedef typename ComplexTypeId<Real>::Type Cmplx;
        int X[4]; 
        for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

        int x[4];
        getCoords(x, idx, X, parity);
#ifdef MULTI_GPU
        for(int dr=0; dr<4; ++dr) {
          x[dr] += arg.border[dr];
          X[dr] += 2*arg.border[dr];
        }
#endif

        int dx[4] = {0, 0, 0, 0};
        for (int mu = 0; mu < 3; mu++) {
          for (int nu = (mu+1); nu < 3; nu++) {
            Matrix<Cmplx,3> U1, U2, U3, U4, tmpM;

            arg.dataOr.load((Real*)(U1.data),linkIndexShift(x,dx,X), mu, parity);
	    dx[mu]++;
            arg.dataOr.load((Real*)(U2.data),linkIndexShift(x,dx,X), nu, 1-parity);
            dx[mu]--;
            dx[nu]++;
            arg.dataOr.load((Real*)(U3.data),linkIndexShift(x,dx,X), mu, 1-parity);
	    dx[nu]--;
            arg.dataOr.load((Real*)(U4.data),linkIndexShift(x,dx,X), nu, parity);

	    tmpM = U1 * U2;
	    tmpM = tmpM * conj(U3);
	    tmpM = tmpM * conj(U4);

	    plaq.x += getTrace(tmpM).x;
          }

          Matrix<Cmplx,3> U1, U2, U3, U4, tmpM;

          arg.dataOr.load((Real*)(U1.data),linkIndexShift(x,dx,X), mu, parity);
          dx[mu]++;
          arg.dataOr.load((Real*)(U2.data),linkIndexShift(x,dx,X), 3, 1-parity);
          dx[mu]--;
          dx[3]++;
          arg.dataOr.load((Real*)(U3.data),linkIndexShift(x,dx,X), mu, 1-parity);
          dx[3]--;
          arg.dataOr.load((Real*)(U4.data),linkIndexShift(x,dx,X), 3, parity);

          tmpM = U1 * U2;
          tmpM = tmpM * conj(U3);
          tmpM = tmpM * conj(U4);

          plaq.y += getTrace(tmpM).x;
        }
      }

      // perform final inter-block reduction and write out result
      reduce2d<blockSize,2>(arg, plaq);*/
  }
