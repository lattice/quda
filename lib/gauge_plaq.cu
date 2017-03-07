#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <atomic.cuh>
#include <cub_helper.cuh>
#include <index_helper.cuh>

namespace quda {

#ifdef GPU_GAUGE_TOOLS

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

  template<int blockSize, typename Float, typename Gauge>
  __global__ void computePlaq(GaugePlaqArg<Gauge> arg){
    typedef Matrix<complex<Float>,3> Link;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y;

    double2 plaq = make_double2(0.0,0.0);

    if(idx < arg.threads) {
      int x[4];
      getCoords(x, idx, arg.X, parity);
      for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

      int dx[4] = {0, 0, 0, 0};
      for (int mu = 0; mu < 3; mu++) {
	for (int nu = (mu+1); nu < 3; nu++) {

	  Link U1 = arg.dataOr(mu, linkIndexShift(x,dx,arg.E), parity);
	  dx[mu]++;
	  Link U2 = arg.dataOr(nu, linkIndexShift(x,dx,arg.E), 1-parity);
	  dx[mu]--;
	  dx[nu]++;
	  Link U3 = arg.dataOr(mu, linkIndexShift(x,dx,arg.E), 1-parity);
	  dx[nu]--;
	  Link U4 = arg.dataOr(nu, linkIndexShift(x,dx,arg.E), parity);

	  plaq.x += getTrace( U1 * U2 * conj(U3) * conj(U4) ).x;
	}

	Link U1 = arg.dataOr(mu, linkIndexShift(x,dx,arg.E), parity);
	dx[mu]++;
	Link U2 = arg.dataOr(3, linkIndexShift(x,dx,arg.E), 1-parity);
	dx[mu]--;
	dx[3]++;
	Link U3 = arg.dataOr(mu,linkIndexShift(x,dx,arg.E), 1-parity);
	dx[3]--;
	Link U4 = arg.dataOr(3, linkIndexShift(x,dx,arg.E), parity);

	plaq.y += getTrace( U1 * U2 * conj(U3) * conj(U4) ).x;
      }
    }

    // perform final inter-block reduction and write out result
    reduce2d<blockSize,2>(arg, plaq);
  }

  template<typename Float, typename Gauge>
    class GaugePlaq : TunableLocalParity {
      GaugePlaqArg<Gauge> arg;
      const QudaFieldLocation location;

      private:
      unsigned int minThreads() const { return arg.threads; }

      public:
      GaugePlaq(GaugePlaqArg<Gauge> &arg, QudaFieldLocation location)
        : arg(arg), location(location) {}
      ~GaugePlaq () { }

      void apply(const cudaStream_t &stream){
        if(location == QUDA_CUDA_FIELD_LOCATION){
          arg.result_h[0] = make_double2(0.,0.);
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

	  LAUNCH_KERNEL_LOCAL_PARITY(computePlaq, tp, stream, arg, Float, Gauge);
	  cudaDeviceSynchronize();
        } else {
          errorQuda("CPU not supported yet\n");
        }
      }

      TuneKey tuneKey() const {
        std::stringstream vol, aux;
        vol << arg.X[0] << "x" << arg.X[1] << "x" << arg.X[2] << "x" << arg.X[3];
	aux << "threads=" << arg.threads << ",prec="  << sizeof(Float);
        return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
      }

      long long flops() const { return 6ll*2*arg.threads*(3*198+3); }
      long long bytes() const { return 6ll*4*2*arg.threads*arg.dataOr.Bytes(); } 
    }; 

  template<typename Float, typename Gauge>
  void plaquette(const Gauge dataOr, const GaugeField& data, double2 &plq, QudaFieldLocation location) {
      GaugePlaqArg<Gauge> arg(dataOr, data);
      GaugePlaq<Float,Gauge> gaugePlaq(arg, location);
      gaugePlaq.apply(0);

      comm_allreduce_array((double*) arg.result_h, 2);
      arg.result_h[0].x /= 9.*(2*arg.threads*comm_size());
      arg.result_h[0].y /= 9.*(2*arg.threads*comm_size());
      plq.x = arg.result_h[0].x;
      plq.y = arg.result_h[0].y;
    }

  template<typename Float>
  void plaquette(const GaugeField& data, double2 &plq, QudaFieldLocation location) {
    INSTANTIATE_RECONSTRUCT(plaquette<Float>, data, plq, location);
  }
#endif

  double3 plaquette(const GaugeField& data, QudaFieldLocation location) {

#ifdef GPU_GAUGE_TOOLS
    double2 plq;
    INSTANTIATE_PRECISION(plaquette, data, plq, location);
    double3 plaq = make_double3(0.5*(plq.x + plq.y), plq.x, plq.y);
#else
    errorQuda("Gauge tools are not build");
    double3 plaq = make_double3(0., 0., 0.);
#endif
    return plaq;
  }

} // namespace quda
