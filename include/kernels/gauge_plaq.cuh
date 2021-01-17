#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <reduction_kernel.h>

namespace quda {

  template <typename Float_, int nColor_, QudaReconstructType recon_>
  struct GaugePlaqArg : public ReduceArg<double2> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == N_COLORS, "GaugePlaqArg instantiated incorrectly");
    static constexpr QudaReconstructType recon = recon_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    dim3 threads; // number of active threads required
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
      threads.x = X[0]*X[1]*X[2]*X[3]/2;
    }

    __device__ __host__ double2 init() const { return zero<double2>(); }
  };

  template<typename Arg>
  __device__ inline double plaquette(Arg &arg, int x[], int parity, int mu, int nu)
  {
    using Link = Matrix<complex<typename Arg::Float>,Arg::nColor>;

    int dx[4] = {0, 0, 0, 0};
    Link U1 = arg.U(mu, linkIndexShift(x,dx,arg.E), parity);
    dx[mu]++;
    Link U2 = arg.U(nu, linkIndexShift(x,dx,arg.E), 1-parity);
    dx[mu]--;
    dx[nu]++;
    Link U3 = arg.U(mu, linkIndexShift(x,dx,arg.E), 1-parity);
    dx[nu]--;
    Link U4 = arg.U(nu, linkIndexShift(x,dx,arg.E), parity);

    if(N_COLORS == 1) {
      //complex<typename Arg::Float> tr(0.0,0.0);
      Link test1 = U1;
      Link test2 = conj(U1);
      //for(int i=0; i<N_COLORS; i++) tr += Prod(i,i);
      
      for(int i=0; i<N_COLORS; i++) {
	for(int j=0; j<N_COLORS; j++) {
	  printf("test1 %d %d %d %d , (%d,%d %d) : %d %d (%.6f,%.6f) \n", x[0], x[1], x[2], x[3], mu, nu, parity, i, j, (test1)(i,j).x, (test1)(i,j).y);
	}
      }
      
      for(int i=0; i<N_COLORS; i++) {
	for(int j=0; j<N_COLORS; j++) {
	  printf("test2 %d %d %d %d , (%d,%d %d) : %d %d (%.6f,%.6f) \n", x[0], x[1], x[2], x[3], mu, nu, parity, j, i, (test2)(j,i).x, (test2)(j,i).y);
	}
      }
    }
    //printf("%d %d %d %d , (%d,%d %d) : Trace %.6f\n",  x[0], x[1], x[2], x[3], mu, nu, parity, tr.x);
    return getTrace( U1 * U2 * conj(U3) * conj(U4) ).real();
  }
  
  template <typename Arg> struct Plaquette {

    using reduce_t = double2;
    Arg &arg;
    constexpr Plaquette(Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    // return the plaquette at site (x_cb, parity)
    template <typename Reducer>
    __device__ __host__ inline reduce_t operator()(reduce_t &value, Reducer &r, int x_cb, int parity)
    {
      reduce_t plaq = zero<reduce_t>();

      int x[4];
      getCoords(x, x_cb, arg.X, parity);
#pragma unroll
      for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

#pragma unroll
      for (int mu = 0; mu < 3; mu++) {
#pragma unroll
        for (int nu = 0; nu < 3; nu++) {
          if (nu >= mu + 1) plaq.x += plaquette(arg, x, parity, mu, nu);
        }

        plaq.y += plaquette(arg, x, parity, mu, 3);
      }

      return r(plaq, value);
    }

  };

} // namespace quda
