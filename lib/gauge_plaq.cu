#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <cub/cub.cuh> 
#include <launch_kernel.cuh>

namespace quda {

#ifdef GPU_GAUGE_TOOLS

//  template <typename Float, typename Gauge>
  template <typename Gauge>
  struct GaugePlaqArg {
    int threads; // number of active threads required
    int X[4]; // grid dimensions
#ifdef MULTI_GPU
    int border[4]; 
#endif
    Gauge dataOr;
    double *plaq;
    double *plaq_h;

    GaugePlaqArg(const Gauge &dataOr, const GaugeField &data)
      : dataOr(dataOr), plaq_h(static_cast<double*>(pinned_malloc(sizeof(double)))) {
#ifdef MULTI_GPU
        for(int dir=0; dir<4; ++dir){
          border[dir] = 2;
        }

        for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir] - border[dir]*2;
#else
        for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
#endif
	threads = X[0]*X[1]*X[2]*X[3];
/*	if ((cudaMallocHost(&plaq_h, sizeof(double))) == cudaErrorMemoryAllocation)
	   errorQuda	("Error allocating memory for plaquette.\n");
	if ((cudaMalloc(&plaq, sizeof(double))) == cudaErrorMemoryAllocation)
	   errorQuda	("Error allocating memory for plaquette.\n");
*/
	cudaHostGetDevicePointer(&plaq, plaq_h, 0);

    }
  };

  static __inline__ __device__ double atomicAdd(double *addr, double val)
  {
    double old=*addr, assumed;
    
    do {
      assumed = old;
      old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
					    __double_as_longlong(assumed),
					    __double_as_longlong(val+assumed)));
    } while( __double_as_longlong(assumed)!=__double_as_longlong(old) );
    
    return old;
  }

  __device__ __host__ inline int linkIndex3(int x[], int dx[], const int X[4]) {
    int y[4];
    for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
    int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
    return idx;
  }


  __device__ __host__ inline void getCoords3(int x[4], int cb_index, const int X[4], int parity) 
  {
    x[3] = cb_index/(X[2]*X[1]*X[0]/2);
    x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    x[1] = (cb_index/(X[0]/2)) % X[1];
    x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    return;
  }

  template<int blockSize, typename Float, typename Gauge>
    __global__ void computePlaq(GaugePlaqArg<Gauge> arg){
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      if(idx >= arg.threads) return;
      typedef typename ComplexTypeId<Float>::Type Cmplx;
      int parity = 0;
      if(idx >= arg.threads/2) {
        parity = 1;
        idx -= arg.threads/2;
      }

      int X[4]; 
      for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

      int x[4];
      getCoords3(x, idx, X, parity);
#ifdef MULTI_GPU
      for(int dr=0; dr<4; ++dr) {
           x[dr] += arg.border[dr];
           X[dr] += 2*arg.border[dr];
      }
#endif
      double plaq = 0.;

      int dx[4] = {0, 0, 0, 0};
      for (int mu = 0; mu < 3; mu++) {
        for (int nu = (mu+1); nu < 4; nu++) {
          Matrix<Cmplx,3> U1, U2, U3, U4, tmpM;

          arg.dataOr.load((Float*)(U1.data),linkIndex3(x,dx,X), mu, parity);
	  dx[mu]++;
          arg.dataOr.load((Float*)(U2.data),linkIndex3(x,dx,X), nu, 1-parity);
	  dx[mu]--;
	  dx[nu]++;
          arg.dataOr.load((Float*)(U3.data),linkIndex3(x,dx,X), mu, 1-parity);
	  dx[nu]--;
          arg.dataOr.load((Float*)(U4.data),linkIndex3(x,dx,X), nu, parity);

	  tmpM	= U1 * U2;
	  tmpM  = tmpM * conj(U3);
	  tmpM  = tmpM * conj(U4);

	  plaq += getTrace(tmpM).x;
        }
      }

      typedef cub::BlockReduce<double, blockSize> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      double aggregate = BlockReduce(temp_storage).Sum(plaq);

      if (threadIdx.x == 0) atomicAdd((double *) arg.plaq, aggregate);
  }

  template<typename Float, typename Gauge>
    class GaugePlaq : Tunable {
      GaugePlaqArg<Gauge> arg;
      const QudaFieldLocation location;

      private:
      unsigned int sharedBytesPerThread() const { return sizeof(Float); }
      unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

      bool tuneSharedBytes() const { return false; } // Don't tune shared memory
      bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
      unsigned int minThreads() const { return arg.threads; }

      public:
      GaugePlaq(GaugePlaqArg<Gauge> &arg, QudaFieldLocation location)
        : arg(arg), location(location) {}
      ~GaugePlaq () { host_free(arg.plaq_h); }

      void apply(const cudaStream_t &stream){
        if(location == QUDA_CUDA_FIELD_LOCATION){
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

	  LAUNCH_KERNEL(computePlaq, tp, stream, arg, Float, Gauge);

//	  cudaMemcpy(arg.plaq_h, arg.plaq, sizeof(double), cudaMemcpyDeviceToHost); 
	  cudaDeviceSynchronize();

	  #ifdef MULTI_GPU
	    comm_allreduce((double*) arg.plaq_h);
	    const int nNodes = comm_dim(0)*comm_dim(1)*comm_dim(2)*comm_dim(3);
            ((double *) arg.plaq_h)[0]	/= 18.*(arg.threads*nNodes);
	  #else
            ((double *) arg.plaq_h)[0]	/= 18.*arg.threads;
	  #endif
        } else {
          errorQuda("CPU not supported yet\n");
          //computePlaqCPU(arg);
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
      long long flops() const { return (1)*6*arg.threads; }
      long long bytes() const { return (1)*6*arg.threads*sizeof(Float); } // Only correct if there is no link reconstruction

    }; 

  template<typename Float, typename Gauge>
    void plaquette(const Gauge dataOr, const GaugeField& data, QudaFieldLocation location, Float &plq) {
      GaugePlaqArg<Gauge> arg(dataOr, data);
      GaugePlaq<Float,Gauge> gaugePlaq(arg, location);
      gaugePlaq.apply(0);
      cudaDeviceSynchronize();
      plq = ((double *) arg.plaq_h)[0];
    }

  template<typename Float>
    Float plaquette(const GaugeField& data, QudaFieldLocation location) {

      // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
      // Need to fix this!!

      Float res;

      if(data.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
        if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          plaquette(FloatNOrder<Float, 18, 2, 18>(data), data, location, res);
        } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
          plaquette(FloatNOrder<Float, 18, 2, 12>(data), data, location, res);
        } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
          plaquette(FloatNOrder<Float, 18, 2,  8>(data), data, location, res);
        } else {
          errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
        }
      } else if(data.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
        if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
          plaquette(FloatNOrder<Float, 18, 4, 18>(data), data, location, res);
        } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
          plaquette(FloatNOrder<Float, 18, 4, 12>(data), data, location, res);
        } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
          plaquette(FloatNOrder<Float, 18, 4,  8>(data), data, location, res);
        } else {
          errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
        }
      } else {
        errorQuda("Invalid Gauge Order\n");
      }

      return res;
    }
#endif

  double plaquette(const GaugeField& data, QudaFieldLocation location) {

#ifdef GPU_GAUGE_TOOLS
    if(data.Precision() == QUDA_HALF_PRECISION) {
      errorQuda("Half precision not supported\n");
    }
    if (data.Precision() == QUDA_SINGLE_PRECISION) {
      return plaquette<float> (data, location);
    } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
      return plaquette<double>(data, location);
    } else {
      errorQuda("Precision %d not supported", data.Precision());
    }
#else
  errorQuda("Gauge tools are not build");
#endif

  }
}
