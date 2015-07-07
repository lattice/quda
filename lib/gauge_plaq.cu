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
    double2 *plaq;
    double2 *plaq_h;

    GaugePlaqArg(const Gauge &dataOr, const GaugeField &data)
      : dataOr(dataOr), plaq_h(static_cast<double2*>(pinned_malloc(sizeof(double2)))) {
#ifdef MULTI_GPU
        for(int dir=0; dir<4; ++dir){
          border[dir] = 2;
        }

        for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir] - border[dir]*2;
#else
        for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
#endif
	threads = X[0]*X[1]*X[2]*X[3];
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

  static  __inline__ __device__ double2 atomicAdd(double2 *addr, double2 val){
    double2 old=*addr;
    old.x = atomicAdd((double*)addr, val.x);
    old.y = atomicAdd((double*)addr+1, val.y);
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

  template <typename T>
  struct Summ {
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b){
        return a + b;
    }
  };

  template <>
  struct Summ<double2>{
    __host__ __device__ __forceinline__ double2 operator()(const double2 &a, const double2 &b){
        return make_double2(a.x+b.x, a.y+b.y);
    }
  };


  template<int blockSize, typename Float, typename Gauge>
    __global__ void computePlaq(GaugePlaqArg<Gauge> arg){
      int idx = threadIdx.x + blockIdx.x*blockDim.x;

      double2 plaq;

      plaq.x = 0.;
      plaq.y = 0.;

      if(idx < arg.threads) {
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

        int dx[4] = {0, 0, 0, 0};
        for (int mu = 0; mu < 3; mu++) {
          for (int nu = (mu+1); nu < 3; nu++) {
            Matrix<Cmplx,3> U1, U2, U3, U4, tmpM;

            arg.dataOr.load((Float*)(U1.data),linkIndex3(x,dx,X), mu, parity);
	    dx[mu]++;
            arg.dataOr.load((Float*)(U2.data),linkIndex3(x,dx,X), nu, 1-parity);
            dx[mu]--;
            dx[nu]++;
            arg.dataOr.load((Float*)(U3.data),linkIndex3(x,dx,X), mu, 1-parity);
	    dx[nu]--;
            arg.dataOr.load((Float*)(U4.data),linkIndex3(x,dx,X), nu, parity);

	    tmpM = U1 * U2;
	    tmpM = tmpM * conj(U3);
	    tmpM = tmpM * conj(U4);

	    plaq.x += getTrace(tmpM).x;
          }

          Matrix<Cmplx,3> U1, U2, U3, U4, tmpM;

          arg.dataOr.load((Float*)(U1.data),linkIndex3(x,dx,X), mu, parity);
          dx[mu]++;
          arg.dataOr.load((Float*)(U2.data),linkIndex3(x,dx,X), 3, 1-parity);
          dx[mu]--;
          dx[3]++;
          arg.dataOr.load((Float*)(U3.data),linkIndex3(x,dx,X), mu, 1-parity);
          dx[3]--;
          arg.dataOr.load((Float*)(U4.data),linkIndex3(x,dx,X), 3, parity);

          tmpM = U1 * U2;
          tmpM = tmpM * conj(U3);
          tmpM = tmpM * conj(U4);

          plaq.y += getTrace(tmpM).x;
        }
      }

      typedef cub::BlockReduce<double2, blockSize> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
//      double2 aggregate = BlockReduce(temp_storage).Sum(plaq);
      double2 aggregate = BlockReduce(temp_storage).Reduce(plaq, Summ<double2>());

      if (threadIdx.x == 0) atomicAdd(arg.plaq, aggregate);
  }

  template<typename Float, typename Gauge>
    class GaugePlaq : Tunable {
      GaugePlaqArg<Gauge> arg;
      const QudaFieldLocation location;

      private:
      unsigned int sharedBytesPerThread() const { return 0; }
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
          arg.plaq_h[0] = make_double2(0.,0.);
          TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

	  LAUNCH_KERNEL(computePlaq, tp, stream, arg, Float, Gauge);

	  cudaDeviceSynchronize();

	  comm_allreduce_array((double*) arg.plaq_h, 2);
	  const int nNodes = comm_dim(0)*comm_dim(1)*comm_dim(2)*comm_dim(3);
	  arg.plaq_h[0].x /= 9.*(arg.threads*nNodes);
	  arg.plaq_h[0].y /= 9.*(arg.threads*nNodes);
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
      long long flops() const { return 6*arg.threads*(3*198+3); }
      long long bytes() const { return 6*arg.threads*arg.dataOr.Bytes(); } 

    }; 

  template<typename Float, typename Gauge>
    void plaquette(const Gauge dataOr, const GaugeField& data, QudaFieldLocation location, double2 &plq) {
      GaugePlaqArg<Gauge> arg(dataOr, data);
      GaugePlaq<Float,Gauge> gaugePlaq(arg, location);
      gaugePlaq.apply(0);
      cudaDeviceSynchronize();

      plq.x = arg.plaq_h[0].x;
      plq.y = arg.plaq_h[0].y;
    }


  // Use traits to reduce the template explosion
  template<typename,QudaReconstructType> struct gauge_mapper { };
  template<> struct gauge_mapper<double,QUDA_RECONSTRUCT_NO> { typedef FloatNOrder<double, 18, 2, 18> type; };
  template<> struct gauge_mapper<double,QUDA_RECONSTRUCT_12> { typedef FloatNOrder<double, 18, 2, 12> type; };
  template<> struct gauge_mapper<double,QUDA_RECONSTRUCT_8> { typedef FloatNOrder<double, 18, 2, 8> type; };

  template<> struct gauge_mapper<float,QUDA_RECONSTRUCT_NO> { typedef FloatNOrder<float, 18, 2, 18> type; };
  template<> struct gauge_mapper<float,QUDA_RECONSTRUCT_12> { typedef FloatNOrder<float, 18, 4, 12> type; };
  template<> struct gauge_mapper<float,QUDA_RECONSTRUCT_8> { typedef FloatNOrder<float, 18, 4, 8> type; };


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
#else
    errorQuda("Gauge tools are not build");
#endif
    
    double3 plaq = make_double3(0.5*(plq.x + plq.y), plq.x, plq.y);
    return plaq;
  }
}
