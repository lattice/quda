#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <typeinfo>
#include <vector>
#include <assert.h>

namespace quda {

#ifdef GPU_MULTIGRID

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

    bool advanceTuneParam(TuneParam &param) const { return false; }

    TuneKey tuneKey() const { return TuneKey(V.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return 0; }
    long long bytes() const { return 2*V.Bytes(); }
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

#endif // GPU_MULTIGRID

  void FillV(ColorSpinorField &V, const std::vector<ColorSpinorField*> &B, int Nvec) {

#ifdef GPU_MULTIGRID
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
#else
    errorQuda("Multigrid has not been built");
#endif
  }

#ifdef GPU_MULTIGRID

  // Creates a block-ordered version of a ColorSpinorField
  // N.B.: Only works for the V field, as we need to block spin.
  template <bool toBlock, int nVec, class Complex, class FieldOrder>
  void blockOrderV(Complex *out, FieldOrder &in,
		   const int *geo_map, const int *geo_bs, int spin_bs,
		   const cpuColorSpinorField &V) {
    //printfQuda("in.Ncolor = %d\n", in.Ncolor());
    int nSpin_coarse = in.Nspin() / spin_bs; // this is number of chiral blocks

    //Compute the size of each block
    int geoBlockSize = 1;
    for (int d=0; d<in.Ndim(); d++) geoBlockSize *= geo_bs[d];
    int blockSize = geoBlockSize * in.Ncolor() * spin_bs; // blockSize includes internal dof

    int x[QUDA_MAX_DIM]; // global coordinates
    int y[QUDA_MAX_DIM]; // local coordinates within a block (full site ordering)

    int checkLength = in.Nparity() * in.VolumeCB() * in.Ncolor() * in.Nspin() * in.Nvec();
    int *check = new int[checkLength];
    int count = 0;

    // Run through the fine grid and do the block ordering
    for (int parity = 0; parity<in.Nparity(); parity++) {
      for (int x_cb=0; x_cb<in.VolumeCB(); x_cb++) {
	int i = parity*in.VolumeCB() + x_cb;

	// Get fine grid coordinates
	V.LatticeIndex(x, i);
	
	//Compute the geometric offset within a block 
	// (x fastest direction, t is slowest direction, non-parity ordered)
	int blockOffset = 0;
	for (int d=in.Ndim()-1; d>=0; d--) {
	  y[d] = x[d]%geo_bs[d];
	  blockOffset *= geo_bs[d];
	  blockOffset += y[d];
	}
	
	//Take the block-ordered offset from the coarse grid offset (geo_map) 
	int offset = geo_map[i]*nSpin_coarse*nVec*geoBlockSize*in.Ncolor()*spin_bs;
	
	for (int v=0; v<in.Nvec(); v++) {
	  for (int s=0; s<in.Nspin(); s++) {
	    for (int c=0; c<in.Ncolor(); c++) {
	      
	      int chirality = s / spin_bs; // chirality is the coarse spin
	      int blockSpin = s % spin_bs; // the remaining spin dof left in each block
	      
	      int index = offset +                                              // geo block
		chirality * nVec * geoBlockSize * spin_bs * in.Ncolor() + // chiral block
	                       v * geoBlockSize * spin_bs * in.Ncolor() + // vector
	                            blockOffset * spin_bs * in.Ncolor() + // local geometry
	                                          blockSpin*in.Ncolor() + // block spin
	                                                                   c;   // color

	      if (toBlock) out[index] = in(parity, x_cb, s, c, v); // going to block order
	      else in(parity, x_cb, s, c, v) = out[index]; // coming from block order
	    
	      check[count++] = index;
	    }
	  }
	}
      }

      //printf("blockOrderV done %d / %d\n", i, in.Volume());
    }
    
    if (count != checkLength) {
      errorQuda("Number of elements packed %d does not match expected value %d nvec=%d nspin=%d ncolor=%d", 
		count, checkLength, in.Nvec(), in.Nspin(), in.Ncolor());
    }

    /*
    // need non-quadratic check
    for (int i=0; i<checkLength; i++) {
      for (int j=0; j<i; j++) {
      if (check[i] == check[j]) errorQuda("Collision detected in block ordering\n");
      }
    }
    */
    delete []check;
  }


  // Creates a block-ordered version of a ColorSpinorField, with parity blocking (for staggered fields)
  // N.B.: same as above but parity are separated.
  template <bool toBlock, int nVec, class Complex, class FieldOrder>
  void blockCBOrderV(Complex *out, FieldOrder &in,
		     const int *geo_map, const int *geo_bs, int spin_bs,
		     const cpuColorSpinorField &V) {
    //Compute the size of each block
    int geoBlockSize = 1;
    for (int d=0; d<in.Ndim(); d++) geoBlockSize *= geo_bs[d];
    int blockSize = geoBlockSize * in.Ncolor(); // blockSize includes internal dof

    int x[QUDA_MAX_DIM]; // global coordinates
    int y[QUDA_MAX_DIM]; // local coordinates within a block (full site ordering)

    int checkLength = in.Nparity() * in.VolumeCB() * in.Ncolor() * in.Nvec();
    int *check = new int[checkLength];
    int count = 0;

    // Run through the fine grid and do the block ordering
    for (int parity = 0; parity<in.Nparity(); parity++) {
      for (int x_cb=0; x_cb<in.VolumeCB(); x_cb++) {
	int i = parity*in.VolumeCB() + x_cb;

	// Get fine grid coordinates
	V.LatticeIndex(x, i);

	//Compute the geometric offset within a block 
	// (x fastest direction, t is slowest direction, non-parity ordered)
	int blockOffset = 0;
	for (int d=in.Ndim()-1; d>=0; d--) {
	  y[d] = x[d]%geo_bs[d];
	  blockOffset *= geo_bs[d];
	  blockOffset += y[d];
	}

	//Take the block-ordered offset from the coarse grid offset (geo_map) 
	//A.S.: geo_map introduced for the full site ordering, so ok to use it for the offset
	int offset = geo_map[i]*nVec*geoBlockSize*in.Ncolor();

	const int s = 0;

	for (int v=0; v<in.Nvec(); v++) {
	  for (int c=0; c<in.Ncolor(); c++) {

	    int chirality = (x[0]+x[1]+x[2]+x[3])%2; // chirality is the fine-grid parity flag

	    int index = offset +                                // geo block
	      chirality * nVec * geoBlockSize * in.Ncolor() + // chiral block
	                     v * geoBlockSize * in.Ncolor() + // vector
	                          blockOffset * in.Ncolor() + // local geometry
	                                                       c;   // color

	    if (toBlock) out[index] = in(parity, x_cb, s, c, v); // going to block order
	    else in(parity, x_cb, s, c, v) = out[index]; // coming from block order

	    check[count++] = index;
	  }
	}

	//printf("blockOrderV done %d / %d\n", i, in.Volume());
      } // x_cb
    } // parity

    if (count != checkLength) {
      errorQuda("Number of elements packed %d does not match expected value %d nvec=%d ncolor=%d", 
		count, checkLength, in.Nvec(), in.Ncolor());
    }

    delete []check;
  }




  // Orthogonalise the nc vectors v[] of length n
  // this assumes the ordering v[(b * Nvec + v) * blocksize + i]

  template <typename sumFloat, typename Float, int N>
  void blockGramSchmidt(complex<Float> *v, int nBlocks, int blockSize) {
    
    for (int b=0; b<nBlocks; b++) {
      for (int jc=0; jc<N; jc++) {
      
	for (int ic=0; ic<jc; ic++) {
	  // Calculate dot product.
	  complex<Float> dot = 0.0;
	  for (int i=0; i<blockSize; i++) 
	    dot += conj(v[(b*N+ic)*blockSize+i]) * v[(b*N+jc)*blockSize+i];
	  
	  // Subtract the blocks to orthogonalise
	  for (int i=0; i<blockSize; i++) 
	    v[(b*N+jc)*blockSize+i] -= dot * v[(b*N+ic)*blockSize+i];
	}
	
	// Normalize the block
	// nrm2 is pure real, but need to use Complex because of template.
        sumFloat nrm2 = 0.0;
	for (int i=0; i<blockSize; i++) nrm2 += norm(v[(b*N+jc)*blockSize+i]);
	sumFloat scale = nrm2 > 0.0 ? 1.0/sqrt(nrm2) : 0.0;
	for (int i=0; i<blockSize; i++) v[(b*N+jc)*blockSize+i] *= scale;
      }

      /*      
      for (int jc=0; jc<N; jc++) {
        complex<sumFloat> nrm2 = 0.0;
        for(int i=0; i<blockSize; i++) nrm2 += norm(v[(b*N+jc)*blockSize+i]);
	//printfQuda("block = %d jc = %d nrm2 = %f\n", b, jc, nrm2.real());
      }
      */

      //printf("blockGramSchmidt done %d / %d\n", b, nBlocks);
    }

  }

  template <typename sumType, typename real, int N>
  class BlockGramSchmidt : public Tunable {

    complex<real> *v;
    int nBlock;
    int blockSize;
    const ColorSpinorField &meta;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  public:
    BlockGramSchmidt(complex<real> *v, int nBlock, int blockSize, const ColorSpinorField &meta)
      : v(v), nBlock(nBlock), blockSize(blockSize), meta(meta) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) sprintf(aux, "nBlock=%d,blockSize=%d,CPU", nBlock, blockSize);
      else sprintf(aux, "nBlock=%d,blockSize=%d,GPU", nBlock, blockSize);
    }

    virtual ~BlockGramSchmidt() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	blockGramSchmidt<sumType, real, N>(v, nBlock, blockSize);
      } else {
	errorQuda("Not implemented for GPU");
      }
    }

    bool advanceTuneParam(TuneParam &param) const { return false; }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return nBlock * N * ((N-1) * (8l + 8l) + 2l) * blockSize; }
    long long bytes() const { return 2*meta.Bytes(); }
  };

  template <bool toBlock, int N, typename real, typename Order>
  class BlockOrderV : public Tunable {

    complex<real> *vBlock;
    Order &vOrder;
    const int *geo_map;
    const int *geo_bs;
    int spin_bs;
    const ColorSpinorField &V;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

  public:
    BlockOrderV(complex<real> *vBlock, Order &vOrder, const int *geo_map, const int *geo_bs, int spin_bs, const ColorSpinorField &V)
      : vBlock(vBlock), vOrder(vOrder), geo_map(geo_map), geo_bs(geo_bs), spin_bs(spin_bs), V(V) {
      (V.Location() == QUDA_CPU_FIELD_LOCATION) ? strcpy(aux, "CPU") : strcpy(aux,"GPU");
    }

    virtual ~BlockOrderV() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (V.Location() == QUDA_CPU_FIELD_LOCATION) {
	blockOrderV<toBlock,N,complex<real>,Order>(vBlock,vOrder,geo_map,geo_bs,spin_bs,V);
      } else {
	errorQuda("Not implemented for GPU");
      }
    }

    bool advanceTuneParam(TuneParam &param) const { return false; }

    TuneKey tuneKey() const { return TuneKey(V.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return 0; }
    long long bytes() const { return 2*V.Bytes(); }
  };

#if 0
  using namespace quda::colorspinor;

  /**
      Kernel argument struct
  */
  template <typename Out, typename In, typename Rotator, int fineSpin, int coarseSpin>
  struct BlockOrthoArg {
    const Rotator V;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field
    int swizzle; // swizzle factor for transposing blockIdx.x mapping to coarse grid coordinate

    BlockOrthoArg(Rotator &V, const int *fine_to_coarse, const int *coarse_to_fine,
		  int parity, const ColorSpinorField &meta) :
      out(out), in(in), V(V), fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
      spin_map(), parity(parity), nParity(meta.SiteSubset()), swizzle(1)
    { }

    BlockOrthoArg(const BlockOrthoArg<Out,In,Rotator,fineSpin,coarseSpin> &arg) :
      out(arg.out), in(arg.in), V(arg.V),
      fine_to_coarse(arg.fine_to_coarse), coarse_to_fine(arg.coarse_to_fine), spin_map(),
      parity(arg.parity), nParity(arg.nParity), swizzle(arg.swizzle)
    { }
  };

  template <typename Float, int nVec, int fineSpin, int coarseSpin, typename Arg>
  void BlockOrtho(Arg &arg) {

    constexpr spinBlocks = fineSpin / coarseSpin;

    for (int b=0; b<nBlocks; b++) {
      for (int s=0; s<spinBlocks; s++) {

	for (int k=0; k<nVec; k++) {

	  for (int l=0; l<k; l++) {
	    complex<Float> dot = 0.0;

	    for (int i=0; i<blockSize; i++) {

	      dot += conj(v(parity, x_cb, s, c, l)) * v(parity, x_cb, s, c, k);

	    }

	}

      }
    }

    for (int parity_coarse=0; parity_coarse<2; parity_coarse++)
      for (int x_coarse_cb=0; x_coarse_cb<arg.out.VolumeCB(); x_coarse_cb++)
	for (int s=0; s<coarseSpin; s++)
	  for (int c=0; c<coarseColor; c++)
	    arg.out(parity_coarse, x_coarse_cb, s, c) = 0.0;

    // loop over fine degrees of freedom
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb=0; x_cb<arg.in.VolumeCB(); x_cb++) {

	int x = parity*arg.in.VolumeCB() + x_cb;
	int x_coarse = arg.fine_to_coarse[x];
	int parity_coarse = (x_coarse >= arg.out.VolumeCB()) ? 1 : 0;
	int x_coarse_cb = x_coarse - parity_coarse*arg.out.VolumeCB();

	for (int coarse_color_block=0; coarse_color_block<coarseColor; coarse_color_block+=coarse_colors_per_thread) {
	  complex<Float> tmp[fineSpin*coarse_colors_per_thread];
	  rotateCoarseColor<Float,fineSpin,fineColor,coarseColor,coarse_colors_per_thread>
	    (tmp, arg.in, arg.V, parity, arg.nParity, x_cb, coarse_color_block);

	  for (int s=0; s<fineSpin; s++) {
	    for (int coarse_color_local=0; coarse_color_local<coarse_colors_per_thread; coarse_color_local++) {
	      int c = coarse_color_block + coarse_color_local;
	      arg.out(parity_coarse,x_coarse_cb,arg.spin_map(s),c) += tmp[s*coarse_colors_per_thread+coarse_color_local];
	    }
	  }

	}
      }
    }

  }
#endif

    template<typename Float, int nSpin, int nColor, int nVec>
  void BlockOrthogonalize(ColorSpinorField &V, const int *geo_bs, const int *geo_map, int spin_bs) {
    complex<Float> *Vblock = new complex<Float>[V.Volume()*V.Nspin()*V.Ncolor()];

    if (V.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      constexpr QudaFieldOrder order = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

      typedef FieldOrderCB<Float,nSpin,nColor,nVec,order> VectorField;
      VectorField vOrder(const_cast<ColorSpinorField&>(V));

      int geo_blocksize = 1;
      for (int d = 0; d < V.Ndim(); d++) geo_blocksize *= geo_bs[d];

      int blocksize = geo_blocksize * vOrder.Ncolor() * spin_bs;
      int chiralBlocks = (V.Nspin() == 1) ? 2 : vOrder.Nspin() / spin_bs; //always 2 for staggered.
      int numblocks = (V.Volume()/geo_blocksize) * chiralBlocks;
      if (V.Nspin() == 1) blocksize /= chiralBlocks; //for staggered chiral block size is a parity block size
    
      printfQuda("Block Orthogonalizing %d blocks of %d length and width %d\n", numblocks, blocksize, nVec);

#if 0
      BlockOrthoArg<> arg(V);
      BlockOrtho ortho();
      otho.apply(0);
#endif

      BlockOrderV<true,nVec,Float,VectorField> reorder(Vblock, vOrder, geo_map, geo_bs, spin_bs, V);
      reorder.apply(0);

      BlockGramSchmidt<double,Float,nVec> ortho(Vblock, numblocks, blocksize, V);
      ortho.apply(0);

      BlockOrderV<false,nVec,Float,VectorField> reset(Vblock, vOrder, geo_map, geo_bs, spin_bs, V);
      reset.apply(0);

      delete []Vblock;

    } else {
      errorQuda("Unsupported field order %d\n", V.FieldOrder());
    }

  }

  template<typename Float, int nSpin, int nColor>
  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, const int *geo_bs, const int *geo_map, int spin_bs) {
    if (Nvec == 2) {
      BlockOrthogonalize<Float,nSpin,nColor,2>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 4) {
      BlockOrthogonalize<Float,nSpin,nColor,4>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 8) {
      BlockOrthogonalize<Float,nSpin,nColor,8>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 12) {
      BlockOrthogonalize<Float,nSpin,nColor,12>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 16) {
      BlockOrthogonalize<Float,nSpin,nColor,16>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 20) {
      BlockOrthogonalize<Float,nSpin,nColor,20>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 24) {
      BlockOrthogonalize<Float,nSpin,nColor,24>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 32) {
      BlockOrthogonalize<Float,nSpin,nColor,32>(V, geo_bs, geo_map, spin_bs);
    } else if (Nvec == 48) {
      BlockOrthogonalize<Float,nSpin,nColor,48>(V, geo_bs, geo_map, spin_bs);
    } else {
      errorQuda("Unsupported nVec %d\n", Nvec);
    }
  }

  template<typename Float, int nSpin>
  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, 
			  const int *geo_bs, const int *geo_map, int spin_bs) {
    if (V.Ncolor()/Nvec == 3) {
      BlockOrthogonalize<Float,nSpin,3>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else if (V.Ncolor()/Nvec == 2) {
      BlockOrthogonalize<Float,nSpin,2>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else if (V.Ncolor()/Nvec == 8) {
      BlockOrthogonalize<Float,nSpin,8>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else if (V.Ncolor()/Nvec == 16) {
      BlockOrthogonalize<Float,nSpin,16>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else if (V.Ncolor()/Nvec == 24) {
      BlockOrthogonalize<Float,nSpin,24>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else if (V.Ncolor()/Nvec == 32) {
      BlockOrthogonalize<Float,nSpin,32>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else if (V.Ncolor()/Nvec == 48) {
      BlockOrthogonalize<Float,nSpin,48>(V, Nvec, geo_bs, geo_map, spin_bs); //for staggered, even-odd blocking presumed
    }  
    else {
      errorQuda("Unsupported nColor %d\n", V.Ncolor()/Nvec);
    }
  }

  template<typename Float>
  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, 
			  const int *geo_bs, const int *geo_map, int spin_bs) {
    if (V.Nspin() == 4) {
      BlockOrthogonalize<Float,4>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else if(V.Nspin() ==2) {
      BlockOrthogonalize<Float,2>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else if (V.Nspin() == 1) {
      BlockOrthogonalize<Float,1>(V, Nvec, geo_bs, geo_map, 1);
    }
    else {
      errorQuda("Unsupported nSpin %d\n", V.Nspin());
    }
  }

#endif // GPU_MULTIGRID

  void BlockOrthogonalize(ColorSpinorField &V, int Nvec, 
			  const int *geo_bs, const int *geo_map, int spin_bs) {
#ifdef GPU_MULTIGRID
    if (V.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      BlockOrthogonalize<double>(V, Nvec, geo_bs, geo_map, spin_bs);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (V.Precision() == QUDA_SINGLE_PRECISION) {
      BlockOrthogonalize<float>(V, Nvec, geo_bs, geo_map, spin_bs);
    } else {
      errorQuda("Unsupported precision %d\n", V.Precision());
    }
#else
    errorQuda("Multigrid has not been built");
#endif // GPU_MULTIGRID
  }

} // namespace quda
