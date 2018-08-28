#include <color_spinor_field.h>
#include <tune_quda.h>
#include <multigrid_helper.cuh>
#include <fast_intdiv.h>
#include <cub_helper.cuh>
#include <uint_to_char.h>
#include <typeinfo>
#include <vector>
#include <assert.h>

// this removes ghost accessor reducing the parameter space needed
#define DISABLE_GHOST true // do not rename this (it is both a template parameter and a macro)

#include <color_spinor_field_order.h>


namespace quda {

#ifdef GPU_MULTIGRID

  using namespace quda::colorspinor;

// enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
//#define SWIZZLE

  // to avoid overflowing the parameter space we put the B array into a separate constant memory buffer
#define MAX_MATRIX_SIZE 4096
  static __constant__ signed char B_array_d[MAX_MATRIX_SIZE];
  static signed char B_array_h[MAX_MATRIX_SIZE];

  /**
      Kernel argument struct
  */
  template <typename Rotator, typename Vector, int fineSpin, int spinBlockSize, int coarseSpin, int nVec>
  struct BlockOrthoArg {
    Rotator V;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field
    int coarseVolume;
    int fineVolumeCB;
    int geoBlockSizeCB; // number of geometric elements in each checkerboarded block
    int_fastdiv swizzle; // swizzle factor for transposing blockIdx.x mapping to coarse grid coordinate
    const Vector *B;
    template <typename... T>
    BlockOrthoArg(ColorSpinorField &V,
                  const int *fine_to_coarse, const int *coarse_to_fine,
		  int parity, const int *geo_bs, const ColorSpinorField &meta,
		  T... B) :
      V(V), fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
      spin_map(), parity(parity), nParity(meta.SiteSubset()),
      B( V.Location() == QUDA_CPU_FIELD_LOCATION ? reinterpret_cast<Vector*>(B_array_h) : nullptr)
    {
      const Vector Btmp[nVec]{*B...};
      if (sizeof(Btmp) > MAX_MATRIX_SIZE) errorQuda("B array size (%lu) is larger than maximum allowed (%d)\n", sizeof(Btmp), MAX_MATRIX_SIZE);
      if (V.Location() == QUDA_CUDA_FIELD_LOCATION) cudaMemcpyToSymbolAsync(B_array_d, Btmp, sizeof(Btmp), 0, cudaMemcpyHostToDevice,0);
      else memcpy(B_array_h, Btmp, sizeof(Btmp));

      int geoBlockSize = 1;
      for (int d = 0; d < V.Ndim(); d++) geoBlockSize *= geo_bs[d];
      geoBlockSizeCB = geoBlockSize/2;
      coarseVolume = meta.Volume() / geoBlockSize;
      fineVolumeCB = meta.VolumeCB();
      if (nParity != 2) errorQuda("BlockOrtho only presently supports full fields");
    }
  };


  template<typename sumType, typename real, int nColor, typename Arg>
  inline __device__ __host__ complex<sumType> colorInnerProduct(int i, complex<real> v[nColor], int parity, int x_cb, int s, const Arg &arg) {
    complex<sumType> dot = 0.0;
#pragma unroll
    for (int c=0; c<nColor; c++) dot += static_cast<complex<real> >(conj(arg.V(parity,x_cb,s,c,i)) * v[c]);
    return dot;
  }

  template<typename sumType, typename real, int nColor, typename Arg>
  inline __device__ __host__ sumType colorNorm(complex<real> v[nColor], int parity, int x_cb, int s, const Arg &arg) {
    sumType  nrm(0.0);
#pragma unroll
    for (int c=0; c<nColor; c++) nrm += norm(v[c]);
    return nrm;
  }

  template<typename real, int nColor, typename Arg>
  inline __device__ __host__ void colorScaleSubtract(complex<real> v[nColor], complex<real> a, int i, int parity, int x_cb, int s, const Arg &arg) {
#pragma unroll
    for (int c=0; c<nColor; c++) v[c] -= a * arg.V(parity,x_cb,s,c,i);
  }

  template<typename real, int nColor, typename Arg>
  inline __device__ __host__ void colorScale(complex<real> v[nColor], real a, int parity, int x_cb, int s, const Arg &arg) {
#pragma unroll
    for (int c=0; c<nColor; c++) v[c] *= a;
  }


  template <typename sumFloat, typename Float, int nSpin, int spinBlockSize, int nColor, int coarseSpin, int nVec, typename Arg>
  void blockOrthoCPU(Arg &arg) {

    // loop over geometric blocks
#pragma omp parallel for
    for (int x_coarse=0; x_coarse<arg.coarseVolume; x_coarse++) {

      for (int j=0; j<nVec; j++) {

	for (int i=0; i<j; i++) {

	  // compute (j,i) block inner products
	  complex<sumFloat> dot[coarseSpin];
	  for (int s=0; s<coarseSpin; s++) dot[s] = 0.0;
	  for (int parity=0; parity<arg.nParity; parity++) {
	    parity = (arg.nParity == 2) ? parity : arg.parity;

	    for (int b=0; b<arg.geoBlockSizeCB; b++) {

	      int x = arg.coarse_to_fine[ (x_coarse*2 + parity) * arg.geoBlockSizeCB + b];
	      int x_cb = x - parity*arg.fineVolumeCB;

	      complex<Float> v[nSpin][nColor];
	      for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) v[s][c] = arg.B[j](parity, x_cb, s, c);

	      for (int s=0; s<nSpin; s++) {
		dot[arg.spin_map(s,parity)] += colorInnerProduct<sumFloat,Float,nColor,Arg>(i, v[s], parity, x_cb, s, arg);
	      }
	    }
	  }

	  // subtract the i blocks to orthogonalise
	  for (int parity=0; parity<arg.nParity; parity++) {
	    parity = (arg.nParity == 2) ? parity : arg.parity;

	    for (int b=0; b<arg.geoBlockSizeCB; b++) {

	      int x = arg.coarse_to_fine[ (x_coarse*2 + parity) * arg.geoBlockSizeCB + b];
	      int x_cb = x - parity*arg.fineVolumeCB;

	      complex<Float> v[nSpin][nColor];
	      if (i==0) for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) v[s][c] = arg.B[j](parity, x_cb, s, c);
	      else for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);

	      for (int s=0; s<nSpin; s++) {
		colorScaleSubtract<Float,nColor,Arg>(v[s], static_cast<complex<Float> >(dot[arg.spin_map(s,parity)]), i, parity, x_cb, s, arg);
	      }

	      for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) arg.V(parity, x_cb, s, c, j) = v[s][c];
	    }
	  }

	} // i

	sumFloat nrm[coarseSpin] = { };
	for (int parity=0; parity<arg.nParity; parity++) {
	  parity = (arg.nParity == 2) ? parity : arg.parity;

	  for (int b=0; b<arg.geoBlockSizeCB; b++) {

	    int x = arg.coarse_to_fine[ (x_coarse*2 + parity) * arg.geoBlockSizeCB + b];
	    int x_cb = x - parity*arg.fineVolumeCB;

	    complex<Float> v[nSpin][nColor];
	    if (j==0) for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) v[s][c] = arg.B[j](parity, x_cb, s, c);
	    else for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);

	    for (int s=0; s<nSpin; s++) {
	      nrm[arg.spin_map(s,parity)] += colorNorm<sumFloat,Float,nColor,Arg>(v[s], parity, x_cb, s, arg);
	    }
	  }
	}

	for (int s=0; s<coarseSpin; s++) nrm[s] = nrm[s] > 0.0 ? rsqrt(nrm[s]) : 0.0;

	for (int parity=0; parity<arg.nParity; parity++) {
	  parity = (arg.nParity == 2) ? parity : arg.parity;

	  for (int b=0; b<arg.geoBlockSizeCB; b++) {

	    int x = arg.coarse_to_fine[ (x_coarse*2 + parity) * arg.geoBlockSizeCB + b];
	    int x_cb = x - parity*arg.fineVolumeCB;

	    complex<Float> v[nSpin][nColor];
	    if (j==0) for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) v[s][c] = arg.B[j](parity, x_cb, s, c);
	    else for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);

	    for (int s=0; s<nSpin; s++) {
	      colorScale<Float,nColor,Arg>(v[s], nrm[arg.spin_map(s,parity)], parity, x_cb, s, arg);
	    }

	    for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) arg.V(parity, x_cb, s, c, j) = v[s][c];
	  }

	}

      } // j

    } // x_coarse
  }

  template <typename sumFloat, typename Float, int nSpin, int spinBlockSize, int nColor, int coarseSpin, int nVec,
	    typename Arg, int block_size>
  __launch_bounds__(2*block_size)
  __global__ void blockOrthoGPU(Arg arg) {

    int x_coarse = blockIdx.x;
#ifdef SWIZZLE
    // the portion of the grid that is exactly divisible by the number of SMs
    const int gridp = gridDim.x - gridDim.x % arg.swizzle;

    if (blockIdx.x < gridp) {
      // this is the portion of the block that we are going to transpose
      const int i = blockIdx.x % arg.swizzle;
      const int j = blockIdx.x / arg.swizzle;

      // tranpose the coordinates
      x_coarse = i * (gridp / arg.swizzle) + j;
    }
#endif
    int parity = (arg.nParity == 2) ? threadIdx.y + blockIdx.y*blockDim.y : arg.parity;
    int x = arg.coarse_to_fine[ (x_coarse*2 + parity) * blockDim.x + threadIdx.x];
    int x_cb = x - parity*arg.fineVolumeCB;
    if (x_cb >= arg.fineVolumeCB) return;

    typedef vector_type<complex<sumFloat>,coarseSpin> cvector;
    typedef vector_type<sumFloat,coarseSpin> rvector;
    typedef cub::BlockReduce<cvector, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 2> dotReduce;
    typedef cub::BlockReduce<rvector, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 2> normReduce;

    __shared__ typename dotReduce::TempStorage dot_storage;
    typename normReduce::TempStorage *norm_storage = (typename normReduce::TempStorage*)&dot_storage;
    cvector *dot_ = (cvector*)&dot_storage;
    rvector *nrm_ = (rvector*)&dot_storage;

    // cast the constant memory buffer to a Vector array
    typedef typename std::remove_reference<decltype(*arg.B)>::type Vector;
    const Vector *B = reinterpret_cast<const Vector*>(B_array_d);

    for (int j=0; j<nVec; j++) {

      complex<Float> v[nSpin][nColor];
#pragma unroll
      for (int s=0; s<nSpin; s++)
#pragma unroll
	for (int c=0; c<nColor; c++) v[s][c] = B[j](parity, x_cb, s, c);

      for (int i=0; i<j; i++) {

	cvector dot;
#pragma unroll
	for (int s=0; s<coarseSpin; s++) dot[s] = 0.0;

	// compute (j,i) block inner products
#pragma unroll
	for (int s=0; s<nSpin; s++) {
	  dot[arg.spin_map(s,parity)] += colorInnerProduct<sumFloat,Float,nColor,Arg>(i, v[s], parity, x_cb, s, arg);
	}

	__syncthreads();
	dot = dotReduce(dot_storage).Sum(dot);
	if (threadIdx.x==0 && threadIdx.y==0) *dot_ = dot;
	__syncthreads();
	dot = *dot_;

	// subtract the blocks to orthogonalise
#pragma unroll
	for (int s=0; s<nSpin; s++) {
	  colorScaleSubtract<Float,nColor,Arg>(v[s], static_cast<complex<Float> >(dot[arg.spin_map(s,parity)]), i, parity, x_cb, s, arg);
	}

      } // i

      // normalize the block
      rvector nrm;
#pragma unroll
      for (int s=0; s<coarseSpin; s++) nrm[s] = static_cast<sumFloat>(0.0);

#pragma unroll
      for (int s=0; s<nSpin; s++) {
	nrm[arg.spin_map(s,parity)] += colorNorm<sumFloat,Float,nColor,Arg>(v[s], parity, x_cb, s, arg);
      }

      __syncthreads();
      nrm = normReduce(*norm_storage).Sum(nrm);
      if (threadIdx.x==0 && threadIdx.y==0) *nrm_ = nrm;
      __syncthreads();
      nrm = *nrm_;

#pragma unroll
      for (int s=0; s<coarseSpin; s++) nrm[s] = nrm[s] > 0.0 ? rsqrt(nrm[s]) : 0.0;

#pragma unroll
      for (int s=0; s<nSpin; s++) {
	colorScale<Float,nColor,Arg>(v[s], nrm[arg.spin_map(s,parity)], parity, x_cb, s, arg);
      }

#pragma unroll
      for (int s=0; s<nSpin; s++)
#pragma unroll
	for (int c=0; c<nColor; c++) arg.V(parity, x_cb, s, c, j) = v[s][c];

    } // j

  }


  template <typename sumType, typename vFloat, typename bFloat, int nSpin, int spinBlockSize, int nColor_, int coarseSpin, int nVec>
  class BlockOrtho : public Tunable {

    // we only support block-format on fine grid where Ncolor=3
    static constexpr int nColor = isFixed<bFloat>::value ? 3 : nColor_;

    typedef typename mapper<vFloat>::type RegType;
    ColorSpinorField &V;
    const std::vector<ColorSpinorField*> B;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const int *geo_bs;
    int geoBlockSize;
    int nBlock;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    unsigned int minThreads() const { return V.VolumeCB(); } // fine parity is the block y dimension

  public:
    BlockOrtho(ColorSpinorField &V, const std::vector<ColorSpinorField*> B,
	       const int *fine_to_coarse, const int *coarse_to_fine, const int *geo_bs)
      : V(V), B(B), fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine), geo_bs(geo_bs)
    {
      if (nColor_ != nColor) errorQuda("Number of colors %d not supported with this precision %lu\n", nColor_, sizeof(bFloat));
      strcpy(aux, V.AuxString());
      strcat(aux, V.Location() == QUDA_CPU_FIELD_LOCATION ? ",CPU,block_size=" : ",GPU,block_size=");
      char size[8];
      geoBlockSize = 1;
      for (int d = 0; d < V.Ndim(); d++) geoBlockSize *= geo_bs[d];
      i32toa(size, geoBlockSize);
      strcat(aux,size);
      if (V.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());

      int chiralBlocks = (nSpin==1) ? 2 : V.Nspin() / spinBlockSize; //always 2 for staggered.
      nBlock = (V.Volume()/geoBlockSize) * chiralBlocks;
    }

    virtual ~BlockOrtho() { }

    /**
       @brief Helper function for expanding the std::vector into a
       parameter pack that we can use to instantiate the const arrays
       in BlockOrthoArg and then call the CPU variant of the block
       orthogonalization.
     */
    template <typename Rotator, typename Vector, std::size_t... S>
    void CPU(const std::vector<ColorSpinorField*> &B, std::index_sequence<S...>) {
      typedef BlockOrthoArg<Rotator,Vector,nSpin,spinBlockSize,coarseSpin,nVec> Arg;
      Arg arg(V, fine_to_coarse, coarse_to_fine, QUDA_INVALID_PARITY, geo_bs, V, B[S]...);
      blockOrthoCPU<sumType,RegType,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg>(arg);
    }

    /**
       @brief Helper function for expanding the std::vector into a
       parameter pack that we can use to instantiate the const arrays
       in BlockOrthoArg and then call the GPU variant of the block
       orthogonalization.
     */
    template <typename Rotator, typename Vector, int block_size, std::size_t... S>
    void GPU(const TuneParam &tp, const cudaStream_t &stream, const std::vector<ColorSpinorField*> &B, std::index_sequence<S...>) {
      typedef typename mapper<vFloat>::type RegType; // need to redeclare typedef (WAR for CUDA 7 and 8)
      typedef BlockOrthoArg<Rotator,Vector,nSpin,spinBlockSize,coarseSpin,nVec> Arg;
      Arg arg(V, fine_to_coarse, coarse_to_fine, QUDA_INVALID_PARITY, geo_bs, V, B[S]...);
      arg.swizzle = tp.aux.x;
      blockOrthoGPU<sumType,RegType,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,block_size>
	<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (V.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (V.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER && B[0]->FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	  typedef FieldOrderCB<RegType,nSpin,nColor,nVec,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,vFloat,vFloat,DISABLE_GHOST> Rotator;
	  typedef FieldOrderCB<RegType,nSpin,nColor,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,bFloat,bFloat,DISABLE_GHOST> Vector;
	  CPU<Rotator,Vector>(B, std::make_index_sequence<nVec>());
	} else {
	  errorQuda("Unsupported field order %d\n", V.FieldOrder());
	}
      } else {
#if __COMPUTE_CAPABILITY__ >= 300
	if (V.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && B[0]->FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
	  typedef FieldOrderCB<RegType,nSpin,nColor,nVec,QUDA_FLOAT2_FIELD_ORDER,vFloat,vFloat,DISABLE_GHOST> Rotator;
	  typedef FieldOrderCB<RegType,nSpin,nColor,1,QUDA_FLOAT2_FIELD_ORDER,bFloat,bFloat,DISABLE_GHOST,isFixed<bFloat>::value> Vector;

	  switch (geoBlockSize/2) {
	  case   4: GPU<Rotator,Vector,  4>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 2x2x2x1 aggregates
	  case   8: GPU<Rotator,Vector,  8>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 2x2x2x2 aggregates
	  case  12: GPU<Rotator,Vector, 12>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 2x2x2x3 aggregates
	  case  16: GPU<Rotator,Vector, 16>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 4x2x2x1 aggregates
	  case  27: GPU<Rotator,Vector, 27>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 3x3x3x1 aggregates
	  case  32: GPU<Rotator,Vector, 32>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 4x4x2x2 aggregates
	  case  36: GPU<Rotator,Vector, 36>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 3x3x2x4 aggregates
	  case  54: GPU<Rotator,Vector, 54>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 3x3x3x4 aggregates
	  case  64: GPU<Rotator,Vector, 64>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 2x4x4x4 aggregates
	  case  81: GPU<Rotator,Vector, 81>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 3x3x3x6 aggregates
	  case  96: GPU<Rotator,Vector, 96>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 4x4x4x3 aggregates
	  case 100: GPU<Rotator,Vector,100>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 5x5x2x4 aggregates
	  case 108: GPU<Rotator,Vector,108>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 3x3x3x8 aggregates
	  case 128: GPU<Rotator,Vector,128>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 4x4x4x4 aggregates
	  case 144: GPU<Rotator,Vector,144>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 4x4x3x6 aggregates
	  case 192: GPU<Rotator,Vector,192>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 4x4x3x8 aggregates
	  case 200: GPU<Rotator,Vector,200>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 5x5x5x4 aggregates
	  case 256: GPU<Rotator,Vector,256>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 4x4x4x8 aggregates
	  case 432: GPU<Rotator,Vector,432>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 6x6x6x4 aggregates
	  case 500: GPU<Rotator,Vector,500>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 5x5x5x8 aggregates
	  case 512: GPU<Rotator,Vector,512>(tp,stream,B,std::make_index_sequence<nVec>()); break; // for 4x4x8x8 aggregates
	  default: errorQuda("Block size %d not instantiated", geoBlockSize/2);
	  }
	} else {
	  errorQuda("Unsupported field order V=%d B=%d\n", V.FieldOrder(), B[0]->FieldOrder());
	}
#else
	errorQuda("GPU block orthogonalization not supported on this GPU architecture");
#endif
      }
    }

    bool advanceAux(TuneParam &param) const
    {
#ifdef SWIZZLE
      if (param.aux.x < 2*deviceProp.multiProcessorCount) {
        param.aux.x++;
	return true;
      } else {
        param.aux.x = 1;
	return false;
      }
#else
      return false;
#endif
    }

    bool advanceTuneParam(TuneParam &param) const {
      if (V.Location() == QUDA_CUDA_FIELD_LOCATION) {
	return advanceSharedBytes(param) || advanceAux(param);
      } else {
	return false;
      }
    }

    TuneKey tuneKey() const { return TuneKey(V.VolString(), typeid(*this).name(), aux); }

    void initTuneParam(TuneParam &param) const { defaultTuneParam(param); }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const {
      param.block = dim3(geoBlockSize/2, V.SiteSubset(), 1);
      param.grid = dim3( (minThreads()+param.block.x-1) / param.block.x, 1, 1);
      param.shared_bytes = 0;
      param.aux.x = 1; // swizzle factor
    }

    long long flops() const { return nBlock * (geoBlockSize/2) * (spinBlockSize == 0 ? 1 : 2*spinBlockSize) / 2 * nColor * (nVec * ((nVec-1) * (8l + 8l)) + 6l); }
    long long bytes() const { return (((nVec+1)*nVec)/2) * (V.Bytes()/nVec) + V.Bytes(); } // load + store

    char *saveOut, *saveOutNorm;

    void preTune() { V.backup(); }
    void postTune() { V.restore(); }

  };

  template<typename vFloat, typename bFloat, int nSpin, int spinBlockSize, int nColor, int nVec>
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B,
			  const int *fine_to_coarse, const int *coarse_to_fine,
			  const int *geo_bs) {

    int geo_blocksize = 1;
    for (int d = 0; d < V.Ndim(); d++) geo_blocksize *= geo_bs[d];

    int blocksize = geo_blocksize * V.Ncolor();
    if (spinBlockSize == 0) { blocksize /= 2; } else { blocksize *= spinBlockSize; }
    int chiralBlocks = (spinBlockSize == 0) ? 2 : V.Nspin() / spinBlockSize; //always 2 for staggered.
    int numblocks = (V.Volume()/geo_blocksize) * chiralBlocks;
    constexpr int coarseSpin = (nSpin == 4 || nSpin == 2 || spinBlockSize == 0) ? 2 : 1;

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Block Orthogonalizing %d blocks of %d length and width %d\n", numblocks, blocksize, nVec);

    V.Scale(1.0); // by definition this is true
    BlockOrtho<double,vFloat,bFloat,nSpin,spinBlockSize,nColor,coarseSpin,nVec> ortho(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
    ortho.apply(0);
    checkCudaError();
  }

  template<typename vFloat, typename bFloat, int nSpin, int spinBlockSize>
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B,
			  const int *fine_to_coarse, const int *coarse_to_fine, const int *geo_bs) {

    const int Nvec = B.size();
    if (V.Ncolor()/Nvec == 3) {

      constexpr int nColor = 3;
      if (Nvec == 6) { // for Wilson free field
	BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,6>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
      } else if (Nvec == 24) {
	BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,24>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
      } else if (Nvec == 32) {
	BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,32>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
      } else if (Nvec == 48) {
	BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,48>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
      } else {
	errorQuda("Unsupported nVec %d\n", Nvec);
      }

    } else if (V.Ncolor()/Nvec == 6) {

      constexpr int nColor = 6;
      if (Nvec == 6) {
	BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,6>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
      } else {
	errorQuda("Unsupported nVec %d\n", Nvec);
      }
    
    } else if (V.Ncolor()/Nvec == 24) {

      constexpr int nColor = 24;
      if (Nvec == 24) {
	BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,24>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
      } else if (Nvec == 32) {
	BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,32>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
      } else {
	errorQuda("Unsupported nVec %d\n", Nvec);
      }

    } else if (V.Ncolor()/Nvec == 32) {

      constexpr int nColor = 32;
      if (Nvec == 32) {
	BlockOrthogonalize<vFloat,bFloat,nSpin,spinBlockSize,nColor,32>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
      } else {
	errorQuda("Unsupported nVec %d\n", Nvec);
      }

    } else {
      errorQuda("Unsupported nColor %d\n", V.Ncolor()/Nvec);
    }
  }

  template<typename vFloat, typename bFloat>
  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B,
			  const int *fine_to_coarse, const int *coarse_to_fine,
			  const int *geo_bs, const int spin_bs) {
    if(V.Nspin() ==2 && spin_bs == 1) { //coarsening coarse fermions w/ chirality.
      BlockOrthogonalize<vFloat,bFloat,2,1>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
#ifdef GPU_WILSON_DIRAC
    } else if (V.Nspin() == 4 && spin_bs == 2) { // coarsening Wilson-like fermions.
      BlockOrthogonalize<vFloat,bFloat,4,2>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
#endif
#ifdef GPU_STAGGERED_DIRAC
    } else if (V.Nspin() == 1 && spin_bs == 0) { // coarsening staggered fermions.
      BlockOrthogonalize<vFloat,bFloat,1,0>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
    } else if (V.Nspin() == 1 && spin_bs == 1) { // coarsening Laplace-like operators.
      BlockOrthogonalize<vFloat,bFloat,1,1>(V, B, fine_to_coarse, coarse_to_fine, geo_bs);
#endif
    } else {
      errorQuda("Unsupported nSpin %d and spinBlockSize %d combination.\n", V.Nspin(), spin_bs);
    }
  }

#endif // GPU_MULTIGRID

  void BlockOrthogonalize(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B,
			  const int *fine_to_coarse, const int *coarse_to_fine,
			  const int *geo_bs, const int spin_bs) {
#ifdef GPU_MULTIGRID
    if (V.Precision() == QUDA_DOUBLE_PRECISION && B[0]->Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      BlockOrthogonalize<double>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (V.Precision() == QUDA_SINGLE_PRECISION && B[0]->Precision() == QUDA_SINGLE_PRECISION) {
      BlockOrthogonalize<float,float>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs);
    } else if (V.Precision() == QUDA_HALF_PRECISION && B[0]->Precision() == QUDA_SINGLE_PRECISION) {
      BlockOrthogonalize<short,float>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs);
    } else if (V.Precision() == QUDA_HALF_PRECISION && B[0]->Precision() == QUDA_HALF_PRECISION) {
      BlockOrthogonalize<short,short>(V, B, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs);
    } else {
      errorQuda("Unsupported precision combination V=%d B=%d\n", V.Precision(), B[0]->Precision());
    }
#else
    errorQuda("Multigrid has not been built");
#endif // GPU_MULTIGRID
  }

} // namespace quda
