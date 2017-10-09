#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <generics/ldg.h>

namespace quda {

#ifdef GPU_GAUGE_FORCE

  template <typename Mom, typename Gauge>
  struct GaugeForceArg {
    Mom mom;
    const Gauge u;

    int threads;
    int X[4]; // the regular volume parameters
    int E[4]; // the extended volume parameters
    int border[4]; // radius of border

    int num_paths;
    int path_max_length;

    double coeff;

    const int *input_path_d[4];
    const int *length_d;
    const double *path_coeff_d;

    int count; // equal to sum of all path lengths.  Used a convenience for computing perf

    GaugeForceArg(Mom &mom, const Gauge &u, int num_paths, int path_max_length, double coeff,
                  int **input_path_d, const int *length_d, const double* path_coeff_d, int count,
		  const GaugeField &meta_mom, const GaugeField &meta_u)
      : mom(mom), u(u), threads(meta_mom.VolumeCB()), num_paths(num_paths),
	path_max_length(path_max_length), coeff(coeff),
	input_path_d{ input_path_d[0], input_path_d[1], input_path_d[2], input_path_d[3] },
	length_d(length_d), path_coeff_d(path_coeff_d), count(count)
    {
      for(int i=0; i<4; i++) {
	X[i] = meta_mom.X()[i];
	E[i] = meta_u.X()[i];
	border[i] = (E[i] - X[i])/2;
      }
    }

    virtual ~GaugeForceArg() { }
  };

  __device__ __host__ inline static int flipDir(int dir) { return (7-dir); }
  __device__ __host__ inline static bool isForwards(int dir) { return (dir <= 3); }

  // this ensures that array elements are held in cache
  template <typename T>
  __device__ __host__ inline static T cache(const T *ptr, int idx) {
#ifdef __CUDA_ARCH__
    return __ldg(ptr+idx);
#else
    return ptr[idx];
#endif
  }

  template<typename Float, typename Arg, int dir>
  __device__ __host__ inline void GaugeForceKernel(Arg &arg, int idx, int parity)
  {
    typedef Matrix<complex<Float>,3> Link;

    int x[4] = {0, 0, 0, 0};
    getCoords(x, idx, arg.X, parity);
    for (int dr=0; dr<4; ++dr) x[dr] += arg.border[dr]; // extended grid coordinates

    //linkA: current matrix
    //linkB: the loaded matrix in this round
    Link linkA, linkB, staple;

#ifdef __CUDA_ARCH__
    extern __shared__ int s[];
    int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
    s[tid] = 0;
    signed char *dx = (signed char*)&s[tid];
#else
    int dx[4] = {0, 0, 0, 0};
#endif

    for (int i=0; i<arg.num_paths; i++) {
      Float coeff = cache(arg.path_coeff_d,i);
      if (coeff == 0) continue;

      const int* path = arg.input_path_d[dir] + i*arg.path_max_length;

      // start from end of link in direction dir
      int nbr_oddbit = (parity^1);
      dx[dir]++;

      int path0 = cache(path,0);
      int lnkdir = isForwards(path0) ? path0 : flipDir(path0);

      if (isForwards(path0)) {
        linkB = arg.u(lnkdir, linkIndexShift(x,dx,arg.E), nbr_oddbit);
        linkA = linkB;
        dx[lnkdir]++; // now have to update location
	nbr_oddbit = nbr_oddbit^1;
      } else {
        dx[lnkdir]--; // if we are going backwards the link is on the adjacent site
        nbr_oddbit = nbr_oddbit^1;
	linkB = arg.u(lnkdir, linkIndexShift(x,dx,arg.E), nbr_oddbit);
        linkA = conj(linkB);
      }
	
      for (int j=1; j<cache(arg.length_d,i); j++) {

        int pathj = cache(path,j);
        int lnkdir = isForwards(pathj) ? pathj : flipDir(pathj);

        if (isForwards(pathj)) {
          linkB = arg.u(lnkdir, linkIndexShift(x,dx,arg.E), nbr_oddbit);
          linkA = linkA * linkB;
          dx[lnkdir]++; // now have to update to new location
          nbr_oddbit = nbr_oddbit^1;	
        } else {
          dx[lnkdir]--; // if we are going backwards the link is on the adjacent site
	  nbr_oddbit = nbr_oddbit^1;
          linkB = arg.u(lnkdir, linkIndexShift(x,dx,arg.E), nbr_oddbit);
          linkA = linkA * conj(linkB);
        }
      } //j
      staple = staple + coeff*linkA;
    } //i

    // multiply by U(x)
    linkA = arg.u(dir, linkIndex(x,arg.E), parity);
    linkA = linkA * staple;

    // update mom(x)
    Link mom = arg.mom(dir, idx, parity);
    mom = mom - arg.coeff * linkA;
    makeAntiHerm(mom);
    arg.mom(dir, idx, parity) = mom;
    return;
  }

  template <typename Float, typename Arg>
  void GaugeForceCPU(Arg &arg) {
    for (int dir=0; dir<4; dir++) {
      for (int parity=0; parity<2; parity++) {
        for (int idx=0; idx<arg.threads; idx++) {
	  switch(dir) {
	  case 0:
	    GaugeForceKernel<Float,Arg,0>(arg, idx, parity);
	    break;
	  case 1:
	    GaugeForceKernel<Float,Arg,1>(arg, idx, parity);
	    break;
	  case 2:
	    GaugeForceKernel<Float,Arg,2>(arg, idx, parity);
	    break;
	  case 3:
	    GaugeForceKernel<Float,Arg,3>(arg, idx, parity);
	    break;
	  }
        }
      }
    }
    return;
  }

  template <typename Float, typename Arg>
  __global__ void GaugeForceGPU(Arg arg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= arg.threads) return;
    int parity = blockIdx.y * blockDim.y + threadIdx.y;
    int dir = blockIdx.z * blockDim.z + threadIdx.z;
    switch(dir) {
    case 0:
      GaugeForceKernel<Float,Arg,0>(arg, idx, parity);
      break;
    case 1:
      GaugeForceKernel<Float,Arg,1>(arg, idx, parity);
      break;
    case 2:
      GaugeForceKernel<Float,Arg,2>(arg, idx, parity);
      break;
    case 3:
      GaugeForceKernel<Float,Arg,3>(arg, idx, parity);
      break;
    }
    return;
  }

  template <typename Float, typename Arg>
  class GaugeForce : public TunableVectorY {

  private:
    Arg &arg;
    QudaFieldLocation location;
    const char *vol_str;
    unsigned int sharedBytesPerThread() const { return 4; } // for dynamic indexing array
    unsigned int minThreads() const { return arg.threads; }
    bool tuneGridDim() const { return false; } // don't tune the grid dimension

  public:
    GaugeForce(Arg &arg, const GaugeField &meta_mom, const GaugeField &meta_u)
      : TunableVectorY(2), arg(arg), location(meta_mom.Location()), vol_str(meta_mom.VolString()) { }
    virtual ~GaugeForce() { }

    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	GaugeForceGPU<Float,Arg><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      } else {
	GaugeForceCPU<Float,Arg>(arg);
      }
    }
  
    void preTune() { arg.mom.save(); }
    void postTune() { arg.mom.load(); } 
  
    long long flops() const { return (arg.count - arg.num_paths + 1) * 198ll * 2 * arg.mom.volumeCB * 4; }
    long long bytes() const { return ((arg.count + 1ll) * arg.u.Bytes() + 2ll*arg.mom.Bytes()) * 2 * arg.mom.volumeCB * 4; }

    TuneKey tuneKey() const {
      std::stringstream aux;
      char comm[5];
      comm[0] = (commDimPartitioned(0) ? '1' : '0');
      comm[1] = (commDimPartitioned(1) ? '1' : '0');
      comm[2] = (commDimPartitioned(2) ? '1' : '0');
      comm[3] = (commDimPartitioned(3) ? '1' : '0');
      comm[4] = '\0';
      aux << "comm=" << comm << ",threads=" << arg.threads << ",num_paths=" << arg.num_paths;
      return TuneKey(vol_str, typeid(*this).name(), aux.str().c_str());
    }  

    bool advanceBlockDim(TuneParam &param) const {
      dim3 block = param.block;
      dim3 grid = param.grid;
      bool rtn = TunableVectorY::advanceBlockDim(param);
      param.block.z = block.z;
      param.grid.z = grid.z;

      if (!rtn) {
	if (param.block.z < 4) {
	  param.block.z *= 2;
	  param.grid.z = 4 / param.block.z;
	  rtn = true;
	} else {
	  param.block.z = 1;
	  param.grid.z = 4;
	  rtn = false;
	}
      }
      return rtn;
    }
    
    void initTuneParam(TuneParam &param) const {
      TunableVectorY::initTuneParam(param);
      param.block.z = 1;
      param.grid.z = 4;
    }

    void defaultTuneParam(TuneParam &param) const {
      TunableVectorY::defaultTuneParam(param);
      param.block.z = 1;
      param.grid.z = 4;
    }
  };
  
  template <typename Float, typename Mom, typename Gauge>
  void gaugeForce(Mom mom, const Gauge &u, GaugeField& meta_mom, const GaugeField& meta_u, const double coeff,
		  int ***input_path, const int* length_h, const double* path_coeff_h, const int num_paths, const int path_max_length)
  {
    size_t bytes = num_paths*path_max_length*sizeof(int);
    int *input_path_d[4];

    int count = 0;
    for (int dir=0; dir<4; dir++) {
      input_path_d[dir] = (int*)device_malloc(bytes);
      cudaMemset(input_path_d[dir], 0, bytes);

      int* input_path_h = (int*)safe_malloc(bytes);
      memset(input_path_h, 0, bytes);
      
      // flatten the input_path array for copying to the device
      for (int i=0; i < num_paths; i++) {
	for (int j=0; j < length_h[i]; j++) {
	  input_path_h[i*path_max_length + j] = input_path[dir][i][j];
          if (dir==0) count++;
	}
      }
      qudaMemcpy(input_path_d[dir], input_path_h, bytes, cudaMemcpyHostToDevice);

      host_free(input_path_h);
    }
      
    //length
    int* length_d = (int*)device_malloc(num_paths*sizeof(int));
    qudaMemcpy(length_d, length_h, num_paths*sizeof(int), cudaMemcpyHostToDevice);

    //path_coeff
    double* path_coeff_d = (double*)device_malloc(num_paths*sizeof(double));
    qudaMemcpy(path_coeff_d, path_coeff_h, num_paths*sizeof(double), cudaMemcpyHostToDevice);

    GaugeForceArg<Mom,Gauge> arg(mom, u, num_paths, path_max_length, coeff, input_path_d,
				 length_d, path_coeff_d, count, meta_mom, meta_u);
    GaugeForce<Float,GaugeForceArg<Mom,Gauge> > gauge_force(arg, meta_mom, meta_u);
    gauge_force.apply(0);
    checkCudaError();

    device_free(length_d);
    device_free(path_coeff_d);
    for (int dir=0; dir<4; dir++) device_free(input_path_d[dir]);
  }

  template <typename Float>
  void gaugeForce(GaugeField& mom, const GaugeField& u, const double coeff, int ***input_path,
		  const int* length, const double* path_coeff, const int num_paths, const int max_length)
  {
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10)
      errorQuda("Reconstruction type %d not supported", mom.Reconstruct());

    if (mom.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      typedef typename gauge::FloatNOrder<Float,18,2,11> M;
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	gaugeForce<Float,M,G>(M(mom), G(u), mom, u, coeff, input_path, length, path_coeff, num_paths, max_length);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	gaugeForce<Float,M,G>(M(mom), G(u), mom, u, coeff, input_path, length, path_coeff, num_paths, max_length);	
      } else {
	errorQuda("Reconstruction type %d not supported", u.Reconstruct());
      }
    } else {
      errorQuda("Gauge Field order %d not supported", mom.Order());
    }

  }
#endif // GPU_GAUGE_FORCE


  void gaugeForce(GaugeField& mom, const GaugeField& u, double coeff, int ***input_path, 
		  int *length, double *path_coeff, int num_paths, int max_length)
  {
#ifdef GPU_GAUGE_FORCE
    if (mom.Precision() != u.Precision()) errorQuda("Mixed precision not supported");
    if (mom.Location() != u.Location()) errorQuda("Mixed field locations not supported");

    switch(mom.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      gaugeForce<double>(mom, u, coeff, input_path, length, path_coeff, num_paths, max_length);
      break;
    case QUDA_SINGLE_PRECISION:
      gaugeForce<float>(mom, u, coeff, input_path, length, path_coeff, num_paths, max_length);
      break;
    default:
      errorQuda("Unsupported precision %d", mom.Precision());
    }
#else
    errorQuda("Gauge force has not been built");
#endif // GPU_GAUGE_FORCE
  }

} // namespace quda


