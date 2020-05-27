#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <generics/ldg.h>
#include <tune_quda.h>
#include <instantiate.h>

namespace quda {

  struct paths {
    const int num_paths;
    const int max_length;
    int *input_path[4];
    const int *length;
    const double *path_coeff;
    int count;

    paths(void *buffer, size_t bytes, int ***input_path, int *length_h, double *path_coeff_h, int num_paths, int max_length) :
      num_paths(num_paths),
      max_length(max_length),
      count(0)
    {
      void *path_h = safe_malloc(bytes);
      memset(path_h, 0, bytes);

      int *input_path_h = (int*)path_h;
      for (int dir=0; dir<4; dir++) {
        // flatten the input_path array for copying to the device
        for (int i=0; i < num_paths; i++) {
          for (int j=0; j < length_h[i]; j++) {
            input_path_h[dir*num_paths*max_length + i*max_length + j] = input_path[dir][i][j];
            if (dir==0) count++;
          }
        }
      }

      // length array
      memcpy((char*)path_h + 4 * num_paths * max_length * sizeof(int), length_h, num_paths*sizeof(int));

      // path_coeff array
      memcpy((char*)path_h + 4 * num_paths * max_length * sizeof(int) + num_paths*sizeof(int), path_coeff_h, num_paths*sizeof(double));

      qudaMemcpy(buffer, path_h, bytes, cudaMemcpyHostToDevice);
      host_free(path_h);

      // finally set the pointers to the correct offsets in the buffer
      for (int d=0; d < 4; d++) this->input_path[d] = (int*)((char*)buffer + d*num_paths*max_length*sizeof(int));
      length = (int*)((char*)buffer + 4*num_paths*max_length*sizeof(int));
      path_coeff = (double*)((char*)buffer + 4 * num_paths * max_length * sizeof(int) + num_paths*sizeof(int));
    }
  };

  template <typename Float_, int nColor_, QudaReconstructType recon_u, QudaReconstructType recon_m>
  struct GaugeForceArg {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    typedef typename gauge_mapper<Float,recon_u>::type Gauge;
    typedef typename gauge_mapper<Float,recon_m>::type Mom;

    Mom mom;
    const Gauge u;

    int threads;
    int X[4]; // the regular volume parameters
    int E[4]; // the extended volume parameters
    int border[4]; // radius of border

    Float epsilon; // stepsize and any other overall scaling factor
    const paths p;

    GaugeForceArg(GaugeField &mom, const GaugeField &u, double epsilon, const paths &p)
      : mom(mom), u(u),
        threads(mom.VolumeCB()),
	epsilon(epsilon),
        p(p)
    {
      for (int i=0; i<4; i++) {
	X[i] = mom.X()[i];
	E[i] = u.X()[i];
	border[i] = (E[i] - X[i])/2;
      }
    }
  };

  constexpr int flipDir(int dir) { return (7-dir); }
  constexpr bool isForwards(int dir) { return (dir <= 3); }

  // this ensures that array elements are held in cache
  template <typename T> constexpr T cache(const T *ptr, int idx) {
#ifdef __CUDA_ARCH__
    return __ldg(ptr+idx);
#else
    return ptr[idx];
#endif
  }

  template <typename Arg, int dir>
  __device__ __host__ inline void GaugeForceKernel(Arg &arg, int idx, int parity)
  {
    using real = typename Arg::Float;
    typedef Matrix<complex<real>,Arg::nColor> Link;

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

    for (int i=0; i<arg.p.num_paths; i++) {
      real coeff = cache(arg.p.path_coeff, i);
      if (coeff == 0) continue;

      const int* path = arg.p.input_path[dir] + i*arg.p.max_length;

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

      for (int j=1; j<cache(arg.p.length,i); j++) {

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
    mom = mom - arg.epsilon * linkA;
    makeAntiHerm(mom);
    arg.mom(dir, idx, parity) = mom;
  }

  template <typename Arg>
  __global__ void GaugeForceKernel(Arg arg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= arg.threads) return;
    int parity = blockIdx.y * blockDim.y + threadIdx.y;
    int dir = blockIdx.z * blockDim.z + threadIdx.z;
    if (dir >= 4) return;

    switch(dir) {
    case 0: GaugeForceKernel<Arg,0>(arg, idx, parity); break;
    case 1: GaugeForceKernel<Arg,1>(arg, idx, parity); break;
    case 2: GaugeForceKernel<Arg,2>(arg, idx, parity); break;
    case 3: GaugeForceKernel<Arg,3>(arg, idx, parity); break;
    }
  }

  template <typename Float, int nColor, QudaReconstructType recon_u> class GaugeForce : public TunableVectorYZ {

    GaugeForceArg<Float, nColor, recon_u, QUDA_RECONSTRUCT_10> arg;
    const GaugeField &meta;

    unsigned int sharedBytesPerThread() const { return 4; } // for dynamic indexing array
    unsigned int minThreads() const { return arg.threads; }
    bool tuneGridDim() const { return false; } // don't tune the grid dimension

  public:
    GaugeForce(const GaugeField &u, GaugeField &mom, double epsilon, const paths &p) :
      TunableVectorYZ(2,4),
      arg(mom, u, epsilon, p),
      meta(u)
    {
      apply(0);
      qudaDeviceSynchronize();
      checkCudaError();
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugeForceKernel<decltype(arg)><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
    }

    void preTune() { arg.mom.save(); }
    void postTune() { arg.mom.load(); }

    long long flops() const { return (arg.p.count - arg.p.num_paths + 1) * 198ll * 2 * arg.mom.volumeCB * 4; }
    long long bytes() const { return ((arg.p.count + 1ll) * arg.u.Bytes() + 2ll*arg.mom.Bytes()) * 2 * arg.mom.volumeCB * 4; }

    TuneKey tuneKey() const {
      std::stringstream aux;
      aux << meta.AuxString() << ",num_paths=" << arg.p.num_paths << comm_dim_partitioned_string();
      return TuneKey(meta.VolString(), typeid(*this).name(), aux.str().c_str());
    }
  };

  void gaugeForce(GaugeField& mom, const GaugeField& u, double epsilon, int ***input_path,
                  int *length_h, double *path_coeff_h, int num_paths, int path_max_length)
  {
    checkPrecision(mom, u);
    checkLocation(mom, u);
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10) errorQuda("Reconstruction type %d not supported", mom.Reconstruct());

    // create path struct in a single allocation
    size_t bytes = 4 * num_paths * path_max_length * sizeof(int) + num_paths*sizeof(int) + num_paths*sizeof(double);
    void *buffer = pool_device_malloc(bytes);
    paths p(buffer, bytes, input_path, length_h, path_coeff_h, num_paths, path_max_length);

#ifdef GPU_GAUGE_FORCE
    // gauge field must be passed as first argument so we peel off its reconstruct type
    instantiate<GaugeForce,ReconstructNo12>(u, mom, epsilon, p);
#else
    errorQuda("Gauge force has not been built");
#endif // GPU_GAUGE_FORCE
    pool_device_free(buffer);
  }

} // namespace quda
