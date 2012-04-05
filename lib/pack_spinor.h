/*
  MAC:

  On my mark, unleash template hell

  Here we are templating on the following
  - input precision
  - output precision
  - number of colors
  - number of spins
  - short vector lengh (float, float2, float4 etc.)
  
  This is still quite a mess.  Options to reduce to the amount of code
  bloat here include:

  1. Using functors to define arbitrary ordering
  2. Use class inheritance to the same effect
  3. Abuse the C preprocessor to define arbitrary mappings
  4. Something else

  Solution is to use class inheritance to defined different mappings,
  and combine it with templating on the return type.  Initial attempt
  at this is in the cpuColorSpinorField, but this will eventually roll
  out here too.

*/


/*

  Packing routines

*/

#define PRESERVE_SPINOR_NORM

#ifdef PRESERVE_SPINOR_NORM // Preserve the norm regardless of basis
#define kP (1.0/sqrt(2.0))
#define kU (1.0/sqrt(2.0))
#else // More numerically accurate not to preserve the norm between basis
#define kP (0.5)
#define kU (1.0)
#endif

template <typename Float, int N, int Ns, int Nc>
struct FloatNOrder {
  Float *field;
  int volume;
  int stride;
  FloatNOrder(Float *field, int volume, int stride)
    : field(field), volume(volume), stride(stride) { ; }
  virtual ~FloatNOrder() { ; }

  __device__ __host__ const Float& operator()(int x, int s, int c, int z) const {
    int internal_idx = (s*Nc + c)*2 + z;
    int pad_idx = internal_idx / N;
    return field[(pad_idx * stride + x)*N + internal_idx % N]; 
  }
  
  __device__ __host__ Float& operator()(int x, int s, int c, int z) { 
    int internal_idx = (s*Nc + c)*2 + z;
    int pad_idx = internal_idx / N;
    return field[(pad_idx * stride + x)*N + internal_idx % N]; 
  };
};

template <typename Float, int Ns, int Nc>
struct SpaceColorSpinorOrder {
  Float *field;
  int volume;
  int stride;
  SpaceColorSpinorOrder(Float *field, int volume, int stride) 
    : field(field), volume(volume), stride(stride) 
  { if (volume != stride) errorQuda("Stride must equal volume for this field order"); }
  virtual ~SpaceColorSpinorOrder() { ; }

  __device__ __host__ const Float& operator()(int x, int s, int c, int z) const 
  { return field[((x*Nc + c)*Ns + s)*2 + z]; }
  
  __device__ __host__ Float& operator()(int x, int s, int c, int z) 
  { return field[((x*Nc + c)*Ns + s)*2 + z]; }
};

template <typename Float, int Ns, int Nc>
struct SpaceSpinorColorOrder {
  Float *field;
  int volume;
  int stride;
  SpaceSpinorColorOrder(Float *field, int volume, int stride) 
   : field(field), volume(volume), stride(stride)
  { if (volume != stride) errorQuda("Stride must equal volume for this field order"); }
  virtual ~SpaceSpinorColorOrder() { ; }

  __device__ __host__ const Float& operator()(int x, int s, int c, int z) const 
  { return field[((x*Ns + s)*Nc + c)*2 + z]; }
  
  __device__ __host__ Float& operator()(int x, int s, int c, int z)
  { return field[((x*Ns + s)*Nc + c)*2 + z]; }
};

/** Straight copy with no basis change */
template <typename Output, typename Input, int Ns, int Nc>
class PreserveBasis {
 public:
  __device__ __host__ void operator()(Output &out, const Input &in, int x) {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  out(x, s, c, z) = in(x, s, c, z);
	}
      }
    }
  }
};

/** Transform from relativistic into non-relavisitic basis */
template <typename Output, typename Input, typename Float, int Ns, int Nc>
struct NonRelBasis {
  __device__ __host__ void operator()(Output &out, const Input &in, int x) {
    int s1[4] = {1, 2, 3, 0};
    int s2[4] = {3, 0, 1, 2};
    Float K1[4] = {kP, -kP, -kP, -kP};
    Float K2[4] = {kP, -kP, kP, kP};

    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  out(x, s, c, z) = K1[s]*in(x, s1[s], c, z) + K2[s]*in(x, s2[s], c, z);
	}
      }
    }
  }
};

/** Transform from non-relativistic into relavisitic basis */
template <typename Output, typename Input, typename Float, int Ns, int Nc>
struct RelBasis {
  __device__ __host__ void operator()(Output &out, const Input &in, int x) {
    int s1[4] = {1, 2, 3, 0};
    int s2[4] = {3, 0, 1, 2};
    Float K1[4] = {-kU, kU,  kU,  kU};
    Float K2[4] = {-kU, kU, -kU, -kU};

    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  out(x, s, c, z) = K1[s]*in(x, s1[s], c, z) + K2[s]*in(x, s2[s], c, z);
	}
      }
    }
  }
};

template <typename OutOrder, typename InOrder, typename Basis>
void packSpinor(OutOrder &out, const InOrder &in, Basis basis, int volume) {  
  for (int x=0; x<volume; x++) basis(out, in, x);
}

template <typename OutOrder, typename InOrder, typename Basis>
__global__ void packSpinorKernel(OutOrder out, const InOrder in, Basis basis, int volume) {  
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid < volume) return;
  basis(out, in, sid);
}

template <typename OutOrder, typename InOrder, typename Basis>
class PackSpinor : Tunable {
  const InOrder &in;
  OutOrder &out;
  Basis &basis;
  int volume;

 private:
  int sharedBytesPerThread() const { return 0; }
  int sharedBytesPerBlock() const { return 0; }
  bool advanceGridDim(TuneParam &param) const { return false; } // Don't tune the grid dimensions.
  bool advanceBlockDim(TuneParam &param) const {
    bool advance = Tunable::advanceBlockDim(param);
    if (advance) param.grid = dim3( (volume+param.block.x-1) / param.block.x, 1, 1);
    return advance;
  }

 public:
 PackSpinor(OutOrder &out, const InOrder &in, Basis &basis, int volume) 
   : out(out), in(in), basis(basis), volume(volume) { ; }
  virtual ~PackSpinor() { ; }
  
  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, QUDA_TUNE_YES, QUDA_VERBOSE);
    packSpinorKernel<OutOrder, InOrder, Basis> <<<tp.grid, tp.block, tp.shared_bytes, stream>>> 
      (out, in, basis, volume);
  }

  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << in.volume; 
    aux << "out stride=" << out.stride << " in stride=" << in.stride;
    return TuneKey(vol.str(), typeid(*this).name(), aux.str());
  }

  std::string paramString(const TuneParam &param) const { // Don't bother printing the grid dim.
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }

  virtual void initTuneParam(TuneParam &param) const {
    Tunable::initTuneParam(param);
    param.grid = dim3( (volume+param.block.x-1) / param.block.x, 1, 1);
  }

  /** sets default values for when tuning is disabled */
  virtual void defaultTuneParam(TuneParam &param) const {
    Tunable::defaultTuneParam(param);
    param.grid = dim3( (volume+param.block.x-1) / param.block.x, 1, 1);
  }
};


/** Decide whether we are changing basis or not */
template <int Nc, int Ns, int N, typename OutOrder, typename InOrder, typename Float, typename FloatN>
void packParitySpinor(FloatN *dest, Float *src, OutOrder &outOrder, const InOrder &inOrder, int Vh, int pad, 
		      QudaGammaBasis destBasis, QudaGammaBasis srcBasis, QudaFieldLocation location) {
  if (destBasis==srcBasis) {
    PreserveBasis<OutOrder, InOrder, Ns, Nc> basis;
    if (location == QUDA_CPU_FIELD_LOCATION) {
      packSpinor(outOrder, inOrder, basis, Vh);
    } else {
      PackSpinor<OutOrder, InOrder, PreserveBasis<OutOrder, InOrder, Ns, Nc> > pack(outOrder, inOrder, basis, Vh);
      pack.apply(0);
    }
  } else if (destBasis == QUDA_UKQCD_GAMMA_BASIS && srcBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
    if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
    NonRelBasis<OutOrder, InOrder, Float, Ns, Nc> basis;
    if (location == QUDA_CPU_FIELD_LOCATION) {
      packSpinor(outOrder, inOrder, basis, Vh);
    } else {
      PackSpinor<OutOrder, InOrder, NonRelBasis<OutOrder, InOrder, Float, Ns, Nc> > pack(outOrder, inOrder, basis, Vh);
      pack.apply(0);
    }
  } else if (srcBasis == QUDA_UKQCD_GAMMA_BASIS && destBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
    if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
    RelBasis<OutOrder, InOrder, Float, Ns, Nc> basis;
    if (location == QUDA_CPU_FIELD_LOCATION) {
      packSpinor(outOrder, inOrder, basis, Vh);    
    } else {
      PackSpinor<OutOrder, InOrder, RelBasis<OutOrder, InOrder, Float, Ns, Nc> > pack(outOrder, inOrder, basis, Vh);
      pack.apply(0);
    } 
  } else {
    errorQuda("Basis change not supported");
  }
}


template <int Nc, int Ns, int N, typename Float, typename FloatN>
void packSpinor(FloatN *dest, Float *src, int V, int pad, const int x[], int destLength, 
		int srcLength, QudaSiteSubset srcSubset, QudaSiteOrder siteOrder, 
		QudaGammaBasis destBasis, QudaGammaBasis srcBasis, QudaFieldOrder srcOrder, QudaFieldLocation location) {

  //  printf("%d %d %d %d %d %d %d %d %d %d %d\n", Nc, Ns, N, V, pad, length, srcSubset, subsetOrder, destBasis, srcBasis, srcOrder);

  if (srcSubset == QUDA_FULL_SITE_SUBSET) {
    if (siteOrder == QUDA_LEXICOGRAPHIC_SITE_ORDER) {
      errorQuda("Copying to full fields with lexicographical ordering is not currently supported");
      /*// We are copying from a full spinor field that is not parity ordered
      if (srcOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	packFullSpinor<Nc,Ns,N>(dest, src, V, pad, x, destLength, destBasis, srcBasis);
      } else if (srcOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
	packQLAFullSpinor<Nc,Ns,N>(dest, src, V, pad, x, destLength, destBasis, srcBasis);
      } else {
	errorQuda("Source field order not supported");
	}*/
    } else {
      // We are copying a parity ordered field
      
      // check what src parity ordering is
      unsigned int evenOff, oddOff;
      if (siteOrder == QUDA_EVEN_ODD_SITE_ORDER) {
	evenOff = 0;
	oddOff = srcLength/2;
      } else {
	oddOff = 0;
	evenOff = srcLength/2;
      }

      int Vh = V;
      if (srcOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	SpaceSpinorColorOrder<Float, Ns, Nc> inOrder(src, Vh, Vh+pad);
	FloatNOrder<FloatN, N, Ns, Nc> outOrder(dest, Vh, Vh+pad);
	packParitySpinor<Nc,Ns,N>(dest, src+evenOff, outOrder, inOrder, Vh, pad, destBasis, srcBasis, location);
	packParitySpinor<Nc,Ns,N>(dest + destLength/2, src+oddOff, outOrder, inOrder, Vh, pad, destBasis, srcBasis, location);
      } else if (srcOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
	SpaceColorSpinorOrder<Float, Ns, Nc> inOrder(src, Vh, Vh+pad);
	FloatNOrder<FloatN, N, Ns, Nc> outOrder(dest, Vh, Vh+pad);
	packParitySpinor<Nc,Ns,N>(dest, src+evenOff, outOrder, inOrder, Vh, pad, destBasis, srcBasis, location);
	packParitySpinor<Nc,Ns,N>(dest + destLength/2, src+oddOff, outOrder, inOrder, Vh, pad, destBasis, srcBasis, location);
      } else {
	errorQuda("Source field order not supported");
	}
    }
  } else {
    // src is defined on a single parity only
    if (srcOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      SpaceSpinorColorOrder<Float, Ns, Nc> inOrder(src, V, V+pad);
      FloatNOrder<FloatN, N, Ns, Nc> outOrder(dest, V, V+pad);
      packParitySpinor<Nc,Ns,N>(dest, src, outOrder, inOrder, V, pad, destBasis, srcBasis, location);
    } else if (srcOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      SpaceColorSpinorOrder<Float, Ns, Nc> inOrder(src, V, V+pad);
      FloatNOrder<FloatN, N, Ns, Nc> outOrder(dest, V, V+pad);
      packParitySpinor<Nc,Ns,N>(dest, src, outOrder, inOrder, V, pad, destBasis, srcBasis, location);
    } else {
      errorQuda("Source field order not supported");
    }
  }

}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void unpackSpinor(Float *dest, FloatN *src, int V, int pad, const int x[], int destLength, 
		  int srcLength, QudaSiteSubset destSubset, QudaSiteOrder siteOrder,  
		  QudaGammaBasis destBasis, QudaGammaBasis srcBasis, QudaFieldOrder destOrder, QudaFieldLocation location) {

  if (destSubset == QUDA_FULL_SITE_SUBSET) {
    if (siteOrder == QUDA_LEXICOGRAPHIC_SITE_ORDER) {
      errorQuda("Copying to full fields with lexicographical ordering is not currently supported");
      // We are copying from a full spinor field that is not parity ordered
      /*if (destOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	unpackFullSpinor<Nc,Ns,N>(dest, src, V, pad, x, srcLength, destBasis, srcBasis);
      } else if (destOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
	unpackQLAFullSpinor<Nc,Ns,N>(dest, src, V, pad, x, srcLength, destBasis, srcBasis);
      } else {
	errorQuda("Source field order not supported");
	}*/
    } else {
      // We are copying a parity ordered field
      
      // check what src parity ordering is
      unsigned int evenOff, oddOff;
      if (siteOrder == QUDA_EVEN_ODD_SITE_ORDER) {
	evenOff = 0;
	oddOff = srcLength/2;
      } else {
	oddOff = 0;
	evenOff = srcLength/2;
      }

      int Vh = V/2;
      if (destOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	FloatNOrder<FloatN, N, Ns, Nc> inOrder(src, Vh, Vh+pad);
	SpaceSpinorColorOrder<Float, Ns, Nc> outOrder(dest, Vh, Vh+pad);
	packParitySpinor<Nc,Ns,N>(dest, src+evenOff, outOrder, inOrder, Vh, pad, destBasis, srcBasis, location);
	packParitySpinor<Nc,Ns,N>(dest + destLength/2, src+oddOff, outOrder, inOrder, Vh, pad, destBasis, srcBasis, location);
      } else if (destOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
	FloatNOrder<FloatN, N, Ns, Nc> inOrder(src, Vh, Vh+pad);
	SpaceColorSpinorOrder<Float, Ns, Nc> outOrder(dest, Vh, Vh+pad);
	packParitySpinor<Nc,Ns,N>(dest, src+evenOff, outOrder, inOrder, Vh, pad, destBasis, srcBasis, location);
	packParitySpinor<Nc,Ns,N>(dest + destLength/2, src+oddOff, outOrder, inOrder, Vh, pad, destBasis, srcBasis, location);
      } else {
	errorQuda("Source field order not supported");
      }
    }
  } else {
    // dest is defined on a single parity only
    if (destOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      FloatNOrder<FloatN, N, Ns, Nc> inOrder(src, V, V+pad);
      SpaceSpinorColorOrder<Float, Ns, Nc> outOrder(dest, V, V+pad);
      packParitySpinor<Nc,Ns,N>(dest, src, outOrder, inOrder, V, pad, destBasis, srcBasis, location);
    } else if (destOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
      FloatNOrder<FloatN, N, Ns, Nc> inOrder(src, V, V+pad);
      SpaceColorSpinorOrder<Float, Ns, Nc> outOrder(dest, V, V+pad);
      packParitySpinor<Nc,Ns,N>(dest, src, outOrder, inOrder, V, pad, destBasis, srcBasis, location);
    } else {
      errorQuda("Destination field order not supported");
    }
  }

}



/*
template <int Nc, int Ns, int N, typename Float, typename FloatN>
void packFullSpinor(FloatN *dest, Float *src, int V, int pad, const int x[], int destLength,
		    QudaGammaBasis destBasis, QudaGammaBasis srcBasis) {
  
  int Vh = V/2;
  if (destBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packSpinorField<Nc,Ns,N>(dest+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packSpinorField<Nc,Ns,N>(dest+destLength/2+N*i, src+ 2*Nc*Ns*k, Vh+pad);
      }
    }
  } else if (destBasis == QUDA_UKQCD_GAMMA_BASIS && srcBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
    if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packNonRelSpinorField<Nc,N>(dest+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packNonRelSpinorField<Nc,N>(dest+destLength/2+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
}


template <int Nc, int Ns, int N, typename Float, typename FloatN>
void unpackFullSpinor(Float *dest, FloatN *src, int V, int pad, const int x[], 
		      int srcLength, QudaGammaBasis destBasis, QudaGammaBasis srcBasis) {
  
  int Vh = V/2;
  if (destBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackSpinorField<Nc,Ns,N>(dest+2*Ns*Nc*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackSpinorField<Nc,Ns,N>(dest+2*Ns*Nc*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else if (srcBasis == QUDA_UKQCD_GAMMA_BASIS && destBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
    if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackNonRelSpinorField<Nc,N>(dest+2*Ns*Nc*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackNonRelSpinorField<Nc,N>(dest+2*Ns*Nc*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void unpackQLAFullSpinor(Float *dest, FloatN *src, int V, int pad, const int x[], 
			 int srcLength, QudaGammaBasis destBasis, QudaGammaBasis srcBasis) {
  
  int Vh = V/2;
  if (destBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackQLASpinorField<Nc,Ns,N>(dest+2*Nc*Ns*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackQLASpinorField<Nc,Ns,N>(dest+2*Nc*Ns*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else if (srcBasis == QUDA_UKQCD_GAMMA_BASIS && destBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackNonRelQLASpinorField<Nc,N>(dest+2*Ns*Nc*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackNonRelQLASpinorField<Nc,N>(dest+2*Ns*Nc*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
  }*/
