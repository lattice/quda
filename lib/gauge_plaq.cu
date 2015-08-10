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
  struct GaugePlaqArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
#ifdef MULTI_GPU
    int border[4]; 
#endif
    Gauge dataOr;
    
    double2 *partial;
    double2 *plaq;
    double2 *plaq_h;

    GaugePlaqArg(const Gauge &dataOr, const GaugeField &data)
      : dataOr(dataOr), 
	partial(static_cast<double2*>(getDeviceReduceBuffer())),
	plaq(static_cast<double2*>(getMappedHostReduceBuffer())),
	plaq_h(static_cast<double2*>(getHostReduceBuffer())) 
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
  };

  template<int blockSize, typename Float, typename Gauge>
    __global__ void computePlaq(GaugePlaqArg<Gauge> arg){
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      int parity = threadIdx.y;

      double2 plaq = make_double2(0.0,0.0);

      if(idx < arg.threads) {
        typedef typename ComplexTypeId<Float>::Type Cmplx;
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

            arg.dataOr.load((Float*)(U1.data),linkIndexShift(x,dx,X), mu, parity);
	    dx[mu]++;
            arg.dataOr.load((Float*)(U2.data),linkIndexShift(x,dx,X), nu, 1-parity);
            dx[mu]--;
            dx[nu]++;
            arg.dataOr.load((Float*)(U3.data),linkIndexShift(x,dx,X), mu, 1-parity);
	    dx[nu]--;
            arg.dataOr.load((Float*)(U4.data),linkIndexShift(x,dx,X), nu, parity);

	    tmpM = U1 * U2;
	    tmpM = tmpM * conj(U3);
	    tmpM = tmpM * conj(U4);

	    plaq.x += getTrace(tmpM).x;
          }

          Matrix<Cmplx,3> U1, U2, U3, U4, tmpM;

          arg.dataOr.load((Float*)(U1.data),linkIndexShift(x,dx,X), mu, parity);
          dx[mu]++;
          arg.dataOr.load((Float*)(U2.data),linkIndexShift(x,dx,X), 3, 1-parity);
          dx[mu]--;
          dx[3]++;
          arg.dataOr.load((Float*)(U3.data),linkIndexShift(x,dx,X), mu, 1-parity);
          dx[3]--;
          arg.dataOr.load((Float*)(U4.data),linkIndexShift(x,dx,X), 3, parity);

          tmpM = U1 * U2;
          tmpM = tmpM * conj(U3);
          tmpM = tmpM * conj(U4);

          plaq.y += getTrace(tmpM).x;
        }
      }

      // perform final inter-block reduction and write out result
      reduce2d<blockSize,2>(arg.plaq, arg.partial, plaq);
  }

  template<typename Float, typename Gauge>
    class GaugePlaq : Tunable {
      GaugePlaqArg<Gauge> arg;
      const QudaFieldLocation location;

      private:
      unsigned int sharedBytesPerThread() const { return 0; }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      GaugePlaq(GaugePlaqArg<Gauge> &arg, QudaFieldLocation location)
        : arg(arg), location(location) {}
      ~GaugePlaq () { }

      bool advanceBlockDim(TuneParam &param) const {
      	bool rtn = Tunable::advanceBlockDim(param);
	param.block.y = 2;
	return rtn;
      }

      void initTuneParam(TuneParam &param) const {
	Tunable::initTuneParam(param);
	param.block.y = 2;
      }

      void apply(const cudaStream_t &stream){
        if(location == QUDA_CUDA_FIELD_LOCATION){
          arg.plaq_h[0] = make_double2(0.,0.);
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

	  LAUNCH_KERNEL(computePlaq, tp, stream, arg, Float, Gauge);
	  cudaDeviceSynchronize();
        } else {
          errorQuda("CPU not supported yet\n");
        }
      }

      TuneKey tuneKey() const {
        std::stringstream vol, aux;
        vol << arg.X[0] << "x";
        vol << arg.X[1] << "x";
        vol << arg.X[2] << "x";
	vol << arg.X[3];
	aux << "threads=" << arg.threads << ",prec="  << sizeof(Float);
        return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
      }

      std::string paramString(const TuneParam &param) const {
        std::stringstream ps;
        ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
        ps << "shared=" << param.shared_bytes;
        return ps.str();
      }

      void preTune(){}
      void postTune(){}
      long long flops() const { return 6ll*2*arg.threads*(3*198+3); }
      long long bytes() const { return 6ll*4*2*arg.threads*arg.dataOr.Bytes(); } 

    }; 

  template<typename Float, typename Gauge>
    void plaquette(const Gauge dataOr, const GaugeField& data, QudaFieldLocation location, double2 &plq) {
      GaugePlaqArg<Gauge> arg(dataOr, data);
      GaugePlaq<Float,Gauge> gaugePlaq(arg, location);
      gaugePlaq.apply(0);

      comm_allreduce_array((double*) arg.plaq_h, 2);
      arg.plaq_h[0].x /= 9.*(2*arg.threads*comm_size());
      arg.plaq_h[0].y /= 9.*(2*arg.threads*comm_size());

      plq.x = arg.plaq_h[0].x;
      plq.y = arg.plaq_h[0].y;
    }

  template<typename Float>
    double2 plaquette(const GaugeField& data, QudaFieldLocation location) {
      double2 res;
      if (!data.isNative()) errorQuda("Plaquette computation only supported on native ordered fields");

      if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Gauge;
	plaquette<Float>(Gauge(data), data, location, res);
      } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Gauge;
	plaquette<Float>(Gauge(data), data, location, res);
      } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Gauge;
	plaquette<Float>(Gauge(data), data, location, res);
      } else {
	errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
      }

      return res;
    }
#endif

  double3 plaquette(const GaugeField& data, QudaFieldLocation location) {

#ifdef GPU_GAUGE_TOOLS
    double2 plq;
    if(data.Precision() == QUDA_HALF_PRECISION) {
      errorQuda("Half precision not supported\n");
    }
    if (data.Precision() == QUDA_SINGLE_PRECISION) {
      plq = plaquette<float> (data, location);
    } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
      plq = plaquette<double>(data, location);
    } else {
      errorQuda("Precision %d not supported", data.Precision());
    }
    double3 plaq = make_double3(0.5*(plq.x + plq.y), plq.x, plq.y);
#else
    errorQuda("Gauge tools are not build");
    double3 plaq = make_double3(0., 0., 0.);
#endif
    return plaq;
  }
}
