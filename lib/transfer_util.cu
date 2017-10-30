#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <multigrid_helper.cuh>
#include <fast_intdiv.h>
#include <cub_helper.cuh>
#include <uint_to_char.h>
#include <typeinfo>
#include <vector>
#include <assert.h>

namespace quda {

  using namespace quda::colorspinor;

  template<typename real, int nSpin, int nColor, int nVec, QudaFieldOrder order>
  struct FillVArg {

    FieldOrderCB<real,nSpin,nColor,nVec,order> V;
    FieldOrderCB<real,nSpin,nColor,1,order> B;
    const int v;

    FillVArg(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int v)
      : V(V), B(*(B[v])), v(v) { }

  };

  // CPU routine to copy the null-space vectors into the V-field
  template <typename Float, int nSpin, int nColor, int nVec, typename Arg>
  void FillVCPU(Arg &arg, int v) {

    for (int parity=0; parity<arg.V.Nparity(); parity++) {
      for (int x_cb=0; x_cb<arg.V.VolumeCB(); x_cb++) {
	for (int s=0; s<nSpin; s++) {
	  for (int c=0; c<nColor; c++) {
	    arg.V(parity, x_cb, s, c, arg.v) = arg.B(parity, x_cb, s, c);
	  }
	}
      }
    }

  }

  // GPU kernel to copy the null-space vectors into the V-field
  template <typename Float, int nSpin, int nColor, int nVec, typename Arg>
  __global__ void FillVGPU(Arg arg, int v) {

    int x_cb = threadIdx.x + blockDim.x*blockIdx.x;
    int parity = threadIdx.y + blockDim.y*blockIdx.y;
    if (x_cb >= arg.V.VolumeCB()) return;

    for (int s=0; s<nSpin; s++) {
      for (int c=0; c<nColor; c++) {
	arg.V(parity, x_cb, s, c, arg.v) = arg.B(parity, x_cb, s, c);
      }
    }

  }

  template <typename real, int nSpin, int nColor, int nVec>
  class FillVLaunch : public TunableVectorY {

    ColorSpinorField &V;
    const std::vector<ColorSpinorField*> &B;
    const int v;
    unsigned int minThreads() const { return V.VolumeCB(); }
    bool tuneGridDim() const { return false; }

  public:
    FillVLaunch(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, const int v)
      : TunableVectorY(2), V(V), B(B), v(v) {
      (V.Location() == QUDA_CPU_FIELD_LOCATION) ? strcpy(aux,"CPU") : strcpy(aux,"GPU");
    }
    virtual ~FillVLaunch() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (V.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (V.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	  FillVArg<real,nSpin,nColor,nVec,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> arg(V,B,v);
	  FillVCPU<real,nSpin,nColor,nVec>(arg,v);
	} else {
	  errorQuda("Field order not implemented %d", V.FieldOrder());
	}
      } else {
	if (V.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
	  FillVArg<real,nSpin,nColor,nVec,QUDA_FLOAT2_FIELD_ORDER> arg(V,B,v);
	  FillVGPU<real,nSpin,nColor,nVec> <<<tp.grid,tp.block,tp.shared_bytes>>>(arg,v);
	} else {
	  errorQuda("Field order not implemented %d", V.FieldOrder());
	}
      }
    }

    bool advanceTuneParam(TuneParam &param) const {
      if (V.Location() == QUDA_CUDA_FIELD_LOCATION) {
	return advanceSharedBytes(param) || advanceBlockDim(param);
      } else {
	return false;
      }
    }

    TuneKey tuneKey() const { return TuneKey(V.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return 0; }
    long long bytes() const { return 2ll*B[0]->Bytes(); }
  };


  template <typename real, int nSpin, int nColor, int nVec>
  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B) {
    for (int v=0; v<nVec; v++) {
      FillVLaunch<real,nSpin,nColor,nVec> f(V,B,v);
      f.apply(0);
    }
  }

  // For staggered this does not include factor 2 due to parity decomposition!
  template <typename Float, int nSpin, int nColor>
  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec) {
    if (Nvec == 2) {
      FillV<Float,nSpin,nColor,2>(V,B);
    } else if (Nvec == 4) {
      FillV<Float,nSpin,nColor,4>(V,B);
    } else if (Nvec == 8) {
      FillV<Float,nSpin,nColor,8>(V,B);
    } else if (Nvec == 12) {
      FillV<Float,nSpin,nColor,12>(V,B);
    } else if (Nvec == 16) {
      FillV<Float,nSpin,nColor,16>(V,B);
    } else if (Nvec == 20) {
      FillV<Float,nSpin,nColor,20>(V,B);
    } else if (Nvec == 24) {
      FillV<Float,nSpin,nColor,24>(V,B);
    } else if (Nvec == 32) {
      FillV<Float,nSpin,nColor,32>(V,B);
    } else if (Nvec == 48) {
      FillV<Float,nSpin,nColor,48>(V,B);
    } else {
      errorQuda("Unsupported Nvec %d", Nvec);
    }
  }

  template <typename Float, int nSpin>
  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec) {
    if (B[0]->Ncolor()*Nvec != V.Ncolor()) errorQuda("Something wrong here");

    if (B[0]->Ncolor() == 2) {
      FillV<Float,nSpin,2>(V,B,Nvec);
    } else if(B[0]->Ncolor() == 3) {
      FillV<Float,nSpin,3>(V,B,Nvec);
    } else if(B[0]->Ncolor() == 8) {
      FillV<Float,nSpin,8>(V,B,Nvec);
    } else if(B[0]->Ncolor() == 16) {
      FillV<Float,nSpin,16>(V,B,Nvec);
    } else if(B[0]->Ncolor() == 24) {
      FillV<Float,nSpin,24>(V,B,Nvec);
    } else if(B[0]->Ncolor() == 32) {
      FillV<Float,nSpin,32>(V,B,Nvec);
    } else {
      errorQuda("Unsupported nColor %d", B[0]->Ncolor());
    }
  }

  template <typename Float>
  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec) {
    if (V.Nspin() == 4) {
      FillV<Float,4>(V,B,Nvec);
    } else if (V.Nspin() == 2) {
      FillV<Float,2>(V,B,Nvec);
#ifdef GPU_STAGGERED_DIRAC
    } else if (V.Nspin() == 1) {
      FillV<Float,1>(V,B,Nvec);
#endif
    } else {
      errorQuda("Unsupported nSpin %d", V.Nspin());
    }
  }

  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec) {
    if (V.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      FillV<double>(V,B,Nvec);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (V.Precision() == QUDA_SINGLE_PRECISION) {
      FillV<float>(V,B,Nvec);
    } else {
      errorQuda("Unsupported precision %d", V.Precision());
    }
  }

  using namespace quda::colorspinor;

// enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
//#define SWIZZLE

  /**
      Kernel argument struct
  */
  template <typename Rotator, int fineSpin, int spinBlockSize, int coarseSpin>
  struct BlockOrthoArg {
    Rotator V;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field
    int coarseVolume;
    int geoBlockSize; // number of geometric elements in each block
    int geoBlockSizeCB; // number of geometric elements in each checkerboarded block
    int nBlock; // number of blocks we are orthogonalizing
    int_fastdiv swizzle; // swizzle factor for transposing blockIdx.x mapping to coarse grid coordinate

    BlockOrthoArg(Rotator &V, const int *fine_to_coarse, const int *coarse_to_fine,
		  int parity, const int *geo_bs, const ColorSpinorField &meta) :
      V(V), fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
      spin_map(), parity(parity), nParity(meta.SiteSubset()), swizzle(1)
    {
      geoBlockSize = 1;
      for (int d = 0; d < V.Ndim(); d++) geoBlockSize *= geo_bs[d];
      geoBlockSizeCB = geoBlockSize/2;
      int chiralBlocks = (fineSpin==1) ? 2 : V.Nspin() / spinBlockSize; //always 2 for staggered.
      nBlock = (meta.Volume()/geoBlockSize) * chiralBlocks;
      coarseVolume = meta.Volume() / geoBlockSize;
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
	      int x_cb = x - parity*arg.V.VolumeCB();

	      complex<Float> v[nSpin][nColor];
	      for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);

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
	      int x_cb = x - parity*arg.V.VolumeCB();

	      complex<Float> v[nSpin][nColor];
	      for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);

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
	    int x_cb = x - parity*arg.V.VolumeCB();

	    complex<Float> v[nSpin][nColor];
	    for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);

	    for (int s=0; s<nSpin; s++) {
	      nrm[arg.spin_map(s,parity)] += colorNorm<sumFloat,Float,nColor,Arg>(v[s], parity, x_cb, s, arg);
	    }
	  }
	}

	for (int s=0; s<coarseSpin; s++) nrm[s] = nrm[s] > 0.0 ? 1.0/(sqrt(nrm[s])) : 0.0;

	for (int parity=0; parity<arg.nParity; parity++) {
	  parity = (arg.nParity == 2) ? parity : arg.parity;

	  for (int b=0; b<arg.geoBlockSizeCB; b++) {

	    int x = arg.coarse_to_fine[ (x_coarse*2 + parity) * arg.geoBlockSizeCB + b];
	    int x_cb = x - parity*arg.V.VolumeCB();

	    complex<Float> v[nSpin][nColor];
	    for (int s=0; s<nSpin; s++) for (int c=0; c<nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);

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
    int x_cb = x - parity*arg.V.VolumeCB();
    if (x_cb >= arg.V.VolumeCB()) return;

    typedef vector_type<complex<sumFloat>,coarseSpin> cvector;
    typedef vector_type<sumFloat,coarseSpin> rvector;
    typedef cub::BlockReduce<cvector, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 2> dotReduce;
    typedef cub::BlockReduce<rvector, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 2> normReduce;

    __shared__ typename dotReduce::TempStorage dot_storage;
    typename normReduce::TempStorage *norm_storage = (typename normReduce::TempStorage*)&dot_storage;
    cvector *dot_ = (cvector*)&dot_storage;
    rvector *nrm_ = (rvector*)&dot_storage;

    for (int j=0; j<nVec; j++) {

      complex<Float> v[nSpin][nColor];
#pragma unroll
      for (int s=0; s<nSpin; s++)
#pragma unroll
	for (int c=0; c<nColor; c++) v[s][c] = arg.V(parity, x_cb, s, c, j);

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
      for (int s=0; s<coarseSpin; s++) nrm[s] = nrm[s] > 0.0 ? 1.0/sqrt(nrm[s]) : 0.0;

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


  template <typename sumType, typename real, int nSpin, int spinBlockSize, int nColor, int coarseSpin, int nVec>
  class BlockOrtho : public Tunable {

    ColorSpinorField &V;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const int *geo_bs;
    int geoBlockSize;
    int nBlock;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    unsigned int minThreads() const { return V.VolumeCB(); } // fine parity is the block y dimension

  public:
    BlockOrtho(ColorSpinorField &V, const int *fine_to_coarse, const int *coarse_to_fine,
	       const int *geo_bs)
      : V(V), fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine), geo_bs(geo_bs)
    {
      strcpy(aux, V.AuxString());
      strcat(aux, V.Location() == QUDA_CPU_FIELD_LOCATION ? ",CPU,block_size=" : ",GPU,block_size=");
      char size[8];
      geoBlockSize = 1;
      for (int d = 0; d < V.Ndim(); d++) geoBlockSize *= geo_bs[d];
      i32toa(size, geoBlockSize);
      strcat(aux,size);

      int chiralBlocks = (nSpin==1) ? 2 : V.Nspin() / spinBlockSize; //always 2 for staggered.
      nBlock = (V.Volume()/geoBlockSize) * chiralBlocks;
    }

    virtual ~BlockOrtho() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (V.Location() == QUDA_CPU_FIELD_LOCATION) {
	if (V.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	  typedef FieldOrderCB<real,nSpin,nColor,nVec,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> VectorField;
	  VectorField vOrder(const_cast<ColorSpinorField&>(V));
	  typedef BlockOrthoArg<VectorField,nSpin,spinBlockSize,coarseSpin> Arg;
	  Arg arg(vOrder, fine_to_coarse, coarse_to_fine, QUDA_INVALID_PARITY, geo_bs, V);
	  blockOrthoCPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg>(arg);
	} else {
	  errorQuda("Unsupported field order %d\n", V.FieldOrder());
	}
      } else {
#if __COMPUTE_CAPABILITY__ >= 300
	if (V.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
	  typedef FieldOrderCB<real,nSpin,nColor,nVec,QUDA_FLOAT2_FIELD_ORDER> VectorField;
	  VectorField vOrder(const_cast<ColorSpinorField&>(V));
	  typedef BlockOrthoArg<VectorField,nSpin,spinBlockSize,coarseSpin> Arg;
	  Arg arg(vOrder, fine_to_coarse, coarse_to_fine, QUDA_INVALID_PARITY, geo_bs, V);
	  arg.swizzle = tp.aux.x;

	  if (arg.geoBlockSizeCB == 4) {          // for 2x2x2x1 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,4>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 8) {   // for 2x2x2x2 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,8>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 16) {  // for 4x2x2x2 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,16>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 27) {  // for 3x3x3x2 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,27>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 32) {  // for 4x4x2x2 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,32>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 36) {  // for 3x3x2x4 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,36>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 54) {  // for 3x3x3x4 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,54>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 64) {  // for 2x4x4x4 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,64>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 100) {  // for 5x5x2x4 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,100>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 108) {  // for 3x3x3x8 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,108>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 128) { // for 4x4x4x4 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,128>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 200) { // for 5x5x2x8  aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,200>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 256) { // for 4x4x4x8  aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,256>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 432) { // for 6x6x6x4 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,432>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 500) { // 5x5x5x8 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,500>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (arg.geoBlockSizeCB == 512) { // 4x4x8x8 aggregates
	    blockOrthoGPU<sumType,real,nSpin,spinBlockSize,nColor,coarseSpin,nVec,Arg,512>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else {
	    errorQuda("Block size %d not instantiated", arg.geoBlockSizeCB);
	  }
	} else {
	  errorQuda("Unsupported field order %d\n", V.FieldOrder());
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

    void preTune() {
      saveOut = new char[V.Bytes()];
      cudaMemcpy(saveOut, V.V(), V.Bytes(), cudaMemcpyDeviceToHost);
      if (V.Precision() == QUDA_HALF_PRECISION && V.NormBytes()) {
	saveOutNorm = new char[V.NormBytes()];
	cudaMemcpy(saveOutNorm, V.Norm(), V.NormBytes(), cudaMemcpyDeviceToHost);
      }
    }

    void postTune() {
      cudaMemcpy((void*)V.V(), saveOut, V.Bytes(), cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (V.Precision() == QUDA_HALF_PRECISION && V.NormBytes()) {
	cudaMemcpy((void*)V.Norm(), saveOutNorm, V.NormBytes(), cudaMemcpyHostToDevice);
	delete[] saveOutNorm;
      }
    }

  };

  template<typename Float, int nSpin, int spinBlockSize, int nColor, int nVec>
  void BlockOrthogonalize(ColorSpinorField &V, const int *fine_to_coarse, const int *coarse_to_fine,
			  const int *geo_bs) {

    int geo_blocksize = 1;
    for (int d = 0; d < V.Ndim(); d++) geo_blocksize *= geo_bs[d];

    int blocksize = geo_blocksize * V.Ncolor();
    if (spinBlockSize == 0) { blocksize /= 2; } else { blocksize *= spinBlockSize; }
    int chiralBlocks = (spinBlockSize == 0) ? 2 : V.Nspin() / spinBlockSize; //always 2 for staggered.
    int numblocks = (V.Volume()/geo_blocksize) * chiralBlocks;
    constexpr int coarseSpin = (nSpin == 4 || nSpin == 2 || spinBlockSize == 0) ? 2 : 1;

    printfQuda("Block Orthogonalizing %d blocks of %d length and width %d\n", numblocks, blocksize, nVec);

    BlockOrtho<double,Float,nSpin,spinBlockSize,nColor,coarseSpin,nVec> ortho(V, fine_to_coarse, coarse_to_fine, geo_bs);
    ortho.apply(0);
  }

  template<typename Float, int nSpin, int spinBlockSize>
  void BlockOrthogonalize(ColorSpinorField &V, int Nvec,
			  const int *fine_to_coarse, const int *coarse_to_fine, const int *geo_bs) {

    if (V.Ncolor()/Nvec == 3) {

      constexpr int nColor = 3;
      if (Nvec == 2) {
	BlockOrthogonalize<Float,nSpin,spinBlockSize,nColor,2>(V, fine_to_coarse, coarse_to_fine, geo_bs);
      } else if (Nvec == 24) {
	BlockOrthogonalize<Float,nSpin,spinBlockSize,nColor,24>(V, fine_to_coarse, coarse_to_fine, geo_bs);
      } else if (Nvec == 32) {
	BlockOrthogonalize<Float,nSpin,spinBlockSize,nColor,32>(V, fine_to_coarse, coarse_to_fine, geo_bs);
      } else {
	errorQuda("Unsupported nVec %d\n", Nvec);
      }

    } else if (V.Ncolor()/Nvec == 2) {

      constexpr int nColor = 2;
      if (Nvec == 2) {
	BlockOrthogonalize<Float,nSpin,spinBlockSize,nColor,2>(V, fine_to_coarse, coarse_to_fine, geo_bs);
      } else {
	errorQuda("Unsupported nVec %d\n", Nvec);
      }

    } else if (V.Ncolor()/Nvec == 24) {

      constexpr int nColor = 24;
      if (Nvec == 24) {
	BlockOrthogonalize<Float,nSpin,spinBlockSize,nColor,24>(V, fine_to_coarse, coarse_to_fine, geo_bs);
      } else if (Nvec == 32) {
	BlockOrthogonalize<Float,nSpin,spinBlockSize,nColor,32>(V, fine_to_coarse, coarse_to_fine, geo_bs);
      } else {
	errorQuda("Unsupported nVec %d\n", Nvec);
      }

    } else if (V.Ncolor()/Nvec == 32) {

      constexpr int nColor = 32;
      if (Nvec == 32) {
	BlockOrthogonalize<Float,nSpin,spinBlockSize,nColor,32>(V, fine_to_coarse, coarse_to_fine, geo_bs);
      } else {
	errorQuda("Unsupported nVec %d\n", Nvec);
      }

    } else {
      errorQuda("Unsupported nColor %d\n", V.Ncolor()/Nvec);
    }
  }

  template<typename Float>
  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, 
			  const int *fine_to_coarse, const int *coarse_to_fine, const int *geo_bs, const int spin_bs) {
    if(V.Nspin() ==2 && spin_bs == 1) { //coarsening coarse fermions w/ chirality.
      BlockOrthogonalize<Float,2,1>(V, Nvec, fine_to_coarse, coarse_to_fine, geo_bs);
#ifdef GPU_WILSON_DIRAC
    } else if (V.Nspin() == 4 && spin_bs == 2) { // coarsening Wilson-like fermions.
      BlockOrthogonalize<Float,4,2>(V, Nvec, fine_to_coarse, coarse_to_fine, geo_bs); 
#endif
#ifdef GPU_STAGGERED_DIRAC
    } else if (V.Nspin() == 1 && spin_bs == 0) { // coarsening staggered fermions.
      BlockOrthogonalize<Float,1,0>(V, Nvec, fine_to_coarse, coarse_to_fine, geo_bs);
    } else if (V.Nspin() == 1 && spin_bs == 1) { // coarsening Laplace-like operators.
      BlockOrthogonalize<Float,1,1>(V, Nvec, fine_to_coarse, coarse_to_fine, geo_bs);
#endif
    } else {
      errorQuda("Unsupported nSpin %d and spinBlockSize %d combination.\n", V.Nspin(), spin_bs);
    }
  }

  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, 
			  const int *fine_to_coarse, const int *coarse_to_fine, const int *geo_bs, const int spin_bs) {
    if (V.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      BlockOrthogonalize<double>(V, Nvec, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (V.Precision() == QUDA_SINGLE_PRECISION) {
      BlockOrthogonalize<float>(V, Nvec, fine_to_coarse, coarse_to_fine, geo_bs, spin_bs);
    } else {
      errorQuda("Unsupported precision %d\n", V.Precision());
    }
  }

} // namespace quda
