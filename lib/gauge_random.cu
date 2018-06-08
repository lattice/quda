#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <launch_kernel.cuh>
#include <atomic.cuh>
#include <cub_helper.cuh>
#include <index_helper.cuh>
#include <random_quda.h>

namespace quda {

#ifdef GPU_GAUGE_TOOLS

  template <typename Gauge>
  struct GaugeGaussArg {
    int threads; // number of active threads required
    int E[4]; // extended grid dimensions
    int X[4]; // true grid dimensions
    int border[4]; 
    Gauge dataDs;
    RNG rngstate;
    
    GaugeGaussArg(const Gauge &dataDs, const GaugeField &data, RNG &rngstate)
      : dataDs(dataDs), rngstate(rngstate)
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


  template<typename Float>
  __device__ __host__  Matrix<complex<Float>,3> genGaussSU3(cuRNGState &localState){
       Matrix<complex<Float>, 3> ret;
	       //ret(i,j) = 0.0;
	       //ret(i,j) = complex<Float>( (Float)(Random<Float>(localState) - 0.5), (Float)(Random<Float>(localState) - 0.5) );

       Float  rand1[4], rand2[4], phi[4], radius[4], temp1[4], temp2[4];

       for (int i=0; i<4; ++i)
       {
	   rand1[i]= Random<Float>(localState);
	   rand2[i]= Random<Float>(localState);
       }

       for (int i=0; i<4; ++i)
       {
	   phi[i]=2.0*M_PI*rand1[i];
	   rand2[i] = rand2[i];
	   radius[i]=sqrt( -log(rand2[i]) );

	   temp1[i] = radius[i]*cos(phi[i]);
	   temp2[i] = radius[i]*sin(phi[i]);
       }

       ret(0,0) = complex<Float>( temp1[2] + 1./sqrt(3.0)*temp2[3], 0.0);
       ret(0,1) = complex<Float>( temp1[0], -temp1[1]);
       ret(0,2) = complex<Float>( temp1[3], -temp2[0]);
       ret(1,0) = complex<Float>( temp1[0], temp1[1] );
       ret(1,1) = complex<Float>( -temp1[2] + 1./sqrt(3.0) * temp2[3], 0.0 );
       ret(1,2) = complex<Float>( temp2[1], -temp2[2] );
       ret(2,0) = complex<Float>( temp1[3], temp2[0] );
       ret(2,1) = complex<Float>( temp2[1], temp2[2] );
       ret(2,2) = complex<Float>( - 2./sqrt(3.0) * temp2[3], 0.0 );

       return ret;
  }


  template<typename Float, typename Gauge>
  __global__ void computeGenGauss(GaugeGaussArg<Gauge> arg){
    typedef Matrix<complex<Float>,3> Link;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int parity = threadIdx.y + blockIdx.y*blockDim.y;

    typedef Matrix<complex<Float>,3> Link;


    if(idx < arg.threads) {
	int x[4];
	getCoords(x, idx, arg.X, parity);
	for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

	int dx[4] = {0, 0, 0, 0};
	for(int mu = 0; mu < 4; mu++){
	    cuRNGState localState = arg.rngstate.State()[idx + parity*arg.threads];

	    Link U = genGaussSU3<Float>(localState);

	    arg.rngstate.State()[ idx + parity*arg.threads ] = localState;
	    arg.dataDs(mu, linkIndexShift(x,dx,arg.X), parity) = U;
	}

    }
  }

  template<typename Float, typename Gauge>
    class GaugeGauss : TunableVectorY {
      GaugeGaussArg<Gauge> arg;
      GaugeField &gf;

      private:
      unsigned int minThreads() const { return arg.threads; }
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.

      public:
      GaugeGauss(GaugeGaussArg<Gauge> &arg, GaugeField &gf)
        : TunableVectorY(2), arg(arg), gf(gf){}
      ~GaugeGauss () { }

      void apply(const cudaStream_t &stream){
        if(gf.Location() == QUDA_CUDA_FIELD_LOCATION){
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

          computeGenGauss<Float><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
	  qudaDeviceSynchronize();
        } else {
          errorQuda("Randomize GaugeFields on CPU not supported yet\n");
        }
      }

      TuneKey tuneKey() const {
        std::stringstream vol, aux;
        vol << arg.X[0] << "x" << arg.X[1] << "x" << arg.X[2] << "x" << arg.X[3];
	aux << "threads=" << arg.threads << ",prec="  << sizeof(Float);
        return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
      }

      long long flops() const { return 0; }
      long long bytes() const { return 0; } 


      void preTune(){
	  //gf.backup();
	  arg.rngstate.backup();
      }
      void postTune(){
	  //gf.restore();
	  arg.rngstate.restore();
      }

    }; 

  template<typename Float, typename Gauge>
  void genGauss(const Gauge dataDs, GaugeField& data, RNG &rngstate) {
      GaugeGaussArg<Gauge> arg(dataDs, data, rngstate);
      GaugeGauss<Float,Gauge> gaugeGauss(arg, data);
      gaugeGauss.apply(0);

    }



  template<typename Float>
  void gaugeGauss(GaugeField &dataDs, RNG &rngstate) {

      if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Gauge;
	  genGauss<Float>(Gauge(dataDs), dataDs, rngstate);
      }else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_12){
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Gauge;
	  genGauss<Float>(Gauge(dataDs), dataDs, rngstate);
      }else if(dataDs.Reconstruct() == QUDA_RECONSTRUCT_8){
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Gauge;
	  genGauss<Float>(Gauge(dataDs), dataDs, rngstate);
      }else{
	  errorQuda("Reconstruction type %d of origin gauge field not supported", dataDs.Reconstruct());
      }

  }

#endif

  void gaugeGauss(GaugeField &dataDs, RNG &rngstate) {

#ifdef GPU_GAUGE_TOOLS

      if(dataDs.Precision() == QUDA_HALF_PRECISION){
	  errorQuda("Half precision not supported\n");
      }

      if (!dataDs.isNative())
	  errorQuda("Order %d with %d reconstruct not supported", dataDs.Order(), dataDs.Reconstruct());

      if (dataDs.Precision() == QUDA_SINGLE_PRECISION){
	  gaugeGauss<float>(dataDs, rngstate);
      } else if(dataDs.Precision() == QUDA_DOUBLE_PRECISION) {
	  gaugeGauss<double>(dataDs, rngstate);
      } else {
	  errorQuda("Precision %d not supported", dataDs.Precision());
      }
      return;
#else
      errorQuda("Gauge tools are not build");
#endif
  }

}
