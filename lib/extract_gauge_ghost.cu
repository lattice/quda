#include <gauge_field_order.h>

namespace quda {
  template <typename Order, int nDim>
  struct ExtractGhostArg {
    Order order;
    const int nFace;
    int X[nDim];
    int surfaceCB[nDim];
    int A[nDim];
    int B[nDim];
    int C[nDim];
    int f[nDim][nDim];
    int localParity[nDim];
    ExtractGhostArg(const Order &order, int nFace, const int *X_, const int *surfaceCB_, const int *A_,
		    const int *B_, const int *C_, const int f_[nDim][nDim], const int *localParity_) 
  : order(order), nFace(nFace) { 
      for (int d=0; d<nDim; d++) {
	X[d] = X_[d];
	surfaceCB[d] = surfaceCB_[d];
	A[d] = A_[d];
	B[d] = B_[d];
	C[d] = C_[d];
	for (int e=0; e<nDim; e++) f[d][e] = f_[d][e];
	localParity[d] = localParity_[d]; 
      }
    }
  };

  /**
     Generic CPU gauge ghost extraction and packing
     NB This routines is specialized to four dimensions
  */
  template <typename Float, int length, int nDim, typename Order>
  void extractGhost(ExtractGhostArg<Order,nDim> arg) {  
    typedef typename mapper<Float>::type RegType;

    for (int parity=0; parity<2; parity++) {

      for (int dim=0; dim<nDim; dim++) {

	// linear index used for writing into ghost buffer
	int indexDst = 0;
	// the following 4-way loop means this is specialized for 4 dimensions 

	// FIXME redefine a, b, c, d such that we always optimize for locality
	for (int d=arg.X[dim]-arg.nFace; d<arg.X[dim]; d++) { // loop over last nFace faces in this dimension
	  for (int a=0; a<arg.A[dim]; a++) { // loop over the surface elements of this face
	    for (int b=0; b<arg.B[dim]; b++) { // loop over the surface elements of this face
	      for (int c=0; c<arg.C[dim]; c++) { // loop over the surface elements of this face
		// index is a checkboarded spacetime coordinate
		int indexCB = (a*arg.f[dim][0] + b*arg.f[dim][1] + c*arg.f[dim][2] + d*arg.f[dim][3]) >> 1;
		// we only do the extraction for parity we are currently working on
		int oddness = (a+b+c+d) & 1;
		if (oddness == parity) {
		  RegType u[length];
		  arg.order.load(u, indexCB, dim, parity); // load the ghost element from the bulk
		  arg.order.saveGhost(u, indexDst, dim, (parity+arg.localParity[dim])&1);
		  indexDst++;
		} // oddness == parity
	      } // c
	    } // b
	  } // a
	} // d

	assert(indexDst == arg.nFace*arg.surfaceCB[dim]);
      } // dim

    } // parity

  }

  /**
     Generic GPU gauge ghost extraction and packing
     NB This routines is specialized to four dimensions
     FIXME this implementation will have two-way warp divergence
  */
  template <typename Float, int length, int nDim, typename Order>
  __global__ void extractGhostKernel(ExtractGhostArg<Order,nDim> arg) {  
    typedef typename mapper<Float>::type RegType;

    for (int parity=0; parity<2; parity++) {
      for (int dim=0; dim<nDim; dim++) {

	// linear index used for writing into ghost buffer
	int X = blockIdx.x * blockDim.x + threadIdx.x; 	
	if (X >= 2*arg.nFace*arg.surfaceCB[dim]) continue;
	// X = ((d * A + a)*B + b)*C + c
	int dab = X/arg.C[dim];
	int c = X - dab*arg.C[dim];
	int da = dab/arg.B[dim];
	int b = dab - da*arg.B[dim];
	int d = da / arg.A[dim];
	int a = da - d * arg.A[dim];
	d += arg.X[dim]-arg.nFace;

	// index is a checkboarded spacetime coordinate
	int indexCB = (a*arg.f[dim][0] + b*arg.f[dim][1] + c*arg.f[dim][2] + d*arg.f[dim][3]) >> 1;
	// we only do the extraction for parity we are currently working on
	int oddness = (a+b+c+d)&1;
	if (oddness == parity) {
	  RegType u[length];
	  arg.order.load(u, indexCB, dim, parity); // load the ghost element from the bulk
	  arg.order.saveGhost(u, X>>1, dim, (parity+arg.localParity[dim])&1);
	} // oddness == parity

      } // dim

    } // parity

  }

  template <typename Float, int length, int nDim, typename Order>
  class ExtractGhost : Tunable {
    ExtractGhostArg<Order,nDim> arg;
    int size;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return size; }

  public:
    ExtractGhost(ExtractGhostArg<Order,nDim> &arg) : arg(arg) { 
      int faceMax = 0;
      for (int d=0; d<nDim; d++) 
	faceMax = (arg.surfaceCB[d] > faceMax ) ? arg.surfaceCB[d] : faceMax;
      size = 2 * arg.nFace * faceMax; // factor 2 of comes from parity
    }
    virtual ~ExtractGhost() { ; }
  
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      extractGhostKernel<Float, length, nDim, Order> 
	<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << arg.order.volumeCB; 
      aux << "stride=" << arg.order.stride;
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }

    std::string paramString(const TuneParam &param) const { // Don't bother printing the grid dim.
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 0; } 
    long long bytes() const { 
      int sites = 0;
      for (int d=0; d<nDim; d++) sites += arg.nFace*arg.surfaceCB[d];
      return 2 * sites * 2 * arg.order.Bytes(); // parity * sites * i/o * vec size
    } 
  };


  /**
     Generic CPU gauge ghost extraction and packing
     NB This routines is specialized to four dimensions
  */
  template <typename Float, int length, typename Order>
  void extractGhost(Order order, int nFace, const int *surfaceCB, const int *X, QudaFieldLocation location) {  
    const int nDim = 4;
    //loop variables: a, b, c with a the most signifcant and c the least significant
    //A, B, C the maximum value
    //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
    int A[nDim], B[nDim], C[nDim];
    A[0] = X[3]; B[0] = X[2]; C[0] = X[1]; // X dimension face
    A[1] = X[3]; B[1] = X[2]; C[1] = X[0]; // Y dimension face
    A[2] = X[3]; B[2] = X[1]; C[2] = X[0]; // Z dimension face
    A[3] = X[2]; B[3] = X[1]; C[3] = X[0]; // T dimension face    

    //multiplication factor to compute index in original cpu memory
    int f[nDim][nDim]={
      {X[0]*X[1]*X[2],  X[0]*X[1], X[0],               1},
      {X[0]*X[1]*X[2],  X[0]*X[1],    1,            X[0]},
      {X[0]*X[1]*X[2],       X[0],    1,       X[0]*X[1]},
      {     X[0]*X[1],       X[0],    1,  X[0]*X[1]*X[2]}
    };

    //set the local processor parity 
    //switching odd and even ghost gauge when that dimension size is odd
    //only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
    // FIXME - I don't understand this, shouldn't it be commDim(dim) == 0 ?
    int localParity[nDim];
    for (int dim=0; dim<nDim; dim++) 
      //localParity[dim] = (X[dim]%2==0 || commDim(dim)) ? 0 : 1;
      localParity[dim] = ((X[dim] % 2 ==1) && (commDim(dim) > 1)) ? 1 : 0;

    ExtractGhostArg<Order, nDim> arg(order, nFace, X, surfaceCB, A, B, C, f, localParity);

    if (location==QUDA_CPU_FIELD_LOCATION) {
      extractGhost<Float,length,nDim,Order>(arg);
    } else {
      ExtractGhost<Float,length,nDim,Order> extract(arg);
      extract.apply(0);
    }

  }

  /** This is the template driver for extractGhost */
  template <typename Float>
    void extractGhost(const GaugeField &u, Float **Ghost) {

    const int length = 18;

    QudaFieldLocation location = 
      (typeid(u)==typeid(cudaGaugeField)) ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION;

    if (u.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(Float)==typeid(short) && u.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  extractGhost<Float,length>(FloatNOrder<Float,length,2,19>(u, 0, Ghost), 
				     u.Nface(), u.SurfaceCB(), u.X(), location);
	} else {
	  extractGhost<Float,length>(FloatNOrder<Float,length,2,18>(u, 0, Ghost),
				     u.Nface(), u.SurfaceCB(), u.X(), location);
	}
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
	extractGhost<Float,length>(FloatNOrder<Float,length,2,12>(u, 0, Ghost),
				   u.Nface(), u.SurfaceCB(), u.X(), location);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_8) {
	extractGhost<Float,length>(FloatNOrder<Float,length,2,8>(u, 0, Ghost), 
				   u.Nface(), u.SurfaceCB(), u.X(), location);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_13) {
	extractGhost<Float,length>(FloatNOrder<Float,length,2,13>(u, 0, Ghost),
				   u.Nface(), u.SurfaceCB(), u.X(), location);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_9) {
	extractGhost<Float,length>(FloatNOrder<Float,length,2,9>(u, 0, Ghost),
				   u.Nface(), u.SurfaceCB(), u.X(), location);
      }
    } else if (u.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
      if (u.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(Float)==typeid(short) && u.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  extractGhost<Float,length>(FloatNOrder<Float,length,1,19>(u, 0, Ghost),
				     u.Nface(), u.SurfaceCB(), u.X(), location);
	} else {
	  extractGhost<Float,length>(FloatNOrder<Float,length,1,18>(u, 0, Ghost),
				     u.Nface(), u.SurfaceCB(), u.X(), location);
	}
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_12) {
	extractGhost<Float,length>(FloatNOrder<Float,length,4,12>(u, 0, Ghost),
				   u.Nface(), u.SurfaceCB(), u.X(), location);
      } else if (u.Reconstruct() == QUDA_RECONSTRUCT_8) { 
	extractGhost<Float,length>(FloatNOrder<Float,length,4,8>(u, 0, Ghost),
				   u.Nface(), u.SurfaceCB(), u.X(), location);
      } else if(u.Reconstruct() == QUDA_RECONSTRUCT_13){
	extractGhost<Float,length>(FloatNOrder<Float,length,4,13>(u, 0, Ghost),
				   u.Nface(), u.SurfaceCB(), u.X(), location);
      } else if(u.Reconstruct() == QUDA_RECONSTRUCT_9){
	extractGhost<Float,length>(FloatNOrder<Float,length,4,9>(u, 0, Ghost),
				   u.Nface(), u.SurfaceCB(), u.X(), location);
      }
    } else if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      
#ifdef BUILD_QDP_INTERFACE
      extractGhost<Float,length>(QDPOrder<Float,length>(u, 0, Ghost),
				 u.Nface(), u.SurfaceCB(), u.X(), location);
#else
      errorQuda("QDP interface has not been built\n");
#endif
      
    } else if (u.Order() == QUDA_QDPJIT_GAUGE_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      extractGhost<Float,length>(QDPJITOrder<Float,length>(u, 0, Ghost),
				 u.Nface(), u.SurfaceCB(), u.X(), location);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {

#ifdef BUILD_CPS_INTERFACE
      extractGhost<Float,length>(CPSOrder<Float,length>(u, 0, Ghost),
				 u.Nface(), u.SurfaceCB(), u.X(), location);
#else
      errorQuda("CPS interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      extractGhost<Float,length>(MILCOrder<Float,length>(u, 0, Ghost),
				 u.Nface(), u.SurfaceCB(), u.X(), location);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_BQCD_GAUGE_ORDER) {

#ifdef BUILD_BQCD_INTERFACE
      extractGhost<Float,length>(BQCDOrder<Float,length>(u, 0, Ghost),
				 u.Nface(), u.SurfaceCB(), u.X(), location);
#else
      errorQuda("BQCD interface has not been built\n");
#endif

    } else if (u.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      extractGhost<Float,length>(TIFROrder<Float,length>(u, 0, Ghost),
				 u.Nface(), u.SurfaceCB(), u.X(), location);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }

  }

  void extractGaugeGhost(const GaugeField &u, void **ghost) {
    if (u.Precision() == QUDA_DOUBLE_PRECISION) {
      extractGhost(u, (double**)ghost);
    } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
      extractGhost(u, (float**)ghost);
    } else if (u.Precision() == QUDA_HALF_PRECISION) {
      extractGhost(u, (short**)ghost);      
    } else {
      errorQuda("Unknown precision type %d", u.Precision());
    }
  }

} // namespace quda
