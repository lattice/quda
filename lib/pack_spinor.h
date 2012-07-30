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

template <typename Float, int Ns, int Nc, int N>
struct FloatNOrder {
  Float *field;
  int volume;
  int stride;
  FloatNOrder(Float *field, int volume, int stride)
    : field(field), volume(volume), stride(stride) { ; }
  virtual ~FloatNOrder() { ; }

  __device__ __host__ inline void load(Float v[Ns*Nc*2], int x, int volume) const {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  int internal_idx = (s*Nc + c)*2 + z;
	  int pad_idx = internal_idx / N;
	  v[(s*Nc+c)*2+z] = field[(pad_idx * stride + x)*N + internal_idx % N];
	}
      }
    }
  }

  __device__ __host__ inline void save(const Float v[Ns*Nc*2], int x, int volume) {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  int internal_idx = (s*Nc + c)*2 + z;
	  int pad_idx = internal_idx / N;
	  field[(pad_idx * stride + x)*N + internal_idx % N] = v[(s*Nc+c)*2+z];
	}
      }
    }
  }

  size_t Bytes() const { return volume * Nc * Ns * 2 * sizeof(Float); }
};

/**! float4 load specialization to obtain full coalescing. */
template<> __device__ inline void FloatNOrder<float, 4, 3, 4>::load(float v[24], int x, int volume) const {
  //read4<4,3>(v, field, stride, x);
#pragma unroll
  for (int i=0; i<4*3*2; i+=4) {
    float4 tmp = ((float4*)field)[i/4 * stride + x];
    v[i] = tmp.x; v[i+1] = tmp.y; v[i+2] = tmp.z; v[i+3] = tmp.w;
  }
}

/**! float4 save specialization to obtain full coalescing. */
template<> __device__ inline void FloatNOrder<float, 4, 3, 4>::save(const float v[24], int x, int volume) {
#pragma unroll
  for (int i=0; i<4*3*2; i+=4) {
    float4 tmp = make_float4(v[i], v[i+1], v[i+2], v[i+3]);
    ((float4*)field)[i/4 * stride + x] = tmp;
  }
}

template <typename Float, int Ns, int Nc>
struct SpaceColorSpinorOrder {
  Float *field;
  int volume;
  int stride;
  SpaceColorSpinorOrder(Float *field, int volume, int stride) 
    : field(field), volume(volume), stride(stride) 
  { if (volume != stride) errorQuda("Stride must equal volume for this field order"); }
  virtual ~SpaceColorSpinorOrder() { ; }

  __device__ __host__ inline void load(Float v[Ns*Nc*2], int x, int volume) const {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  v[(s*Nc+c)*2+z] = field[((x*Nc + c)*Ns + s)*2 + z]; 
	}
      }
    }
  }

  __device__ __host__ inline void save(const Float v[Ns*Nc*2], int x, int volume) {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  field[((x*Nc + c)*Ns + s)*2 + z] = v[(s*Nc+c)*2+z];
	}
      }
    }
  }

  size_t Bytes() const { return volume * Nc * Ns * 2 * sizeof(Float); }
};

template <typename Float, int Ns, int Nc>
  __device__ inline void load_shared(Float v[Ns*Nc*2], Float *field, int x, int volume) {
  const int tid = threadIdx.x;
  const int vec_length = Ns*Nc*2;

  // the length of the block on the last grid site might not extend to all threads
  const int block_dim = (blockIdx.x == gridDim.x-1) ? 
    volume - (gridDim.x-1)*blockDim.x : blockDim.x;

  extern __shared__ Float s_data[];

  int x0 = x-tid;
  int i=tid;
  while (i<vec_length*block_dim) {
    int space_idx = i / vec_length;
    int internal_idx = i - space_idx*vec_length;
    int sh_idx = internal_idx*(blockDim.x+1) + space_idx;
    s_data[sh_idx] = field[x0*vec_length + i];
    i += block_dim;
  }

  __syncthreads();

#pragma unroll
  for (int s=0; s<Ns; s++)
#pragma unroll
    for (int c=0; c<Nc; c++) 
#pragma unroll
      for (int z=0; z<2; z++) { // block+1 to avoid bank conflicts
	int sh_idx = ((c*Ns+s)*2+z)*(blockDim.x+1) + tid;
	v[(s*Nc + c)*2 + z] = s_data[sh_idx];
      }

} 

template <typename Float, int Ns, int Nc>
  __device__ inline void save_shared(Float *field, const Float v[Ns*Nc*2], int x, int volume) {
  const int tid = threadIdx.x;
  const int vec_length = Ns*Nc*2;

  // the length of the block on the last grid site might not extend to all threads
  const int block_dim = (blockIdx.x == gridDim.x-1) ? 
    volume - (gridDim.x-1)*blockDim.x : blockDim.x;

  extern __shared__ Float s_data[];

#pragma unroll
  for (int s=0; s<Ns; s++)
#pragma unroll
    for (int c=0; c<Nc; c++) 
#pragma unroll
      for (int z=0; z<2; z++) { // block+1 to avoid bank conflicts
	int sh_idx = ((c*Ns+s)*2+z)*(blockDim.x+1) + tid;
	s_data[sh_idx] = v[(s*Nc + c)*2 + z];
      }

  __syncthreads();

  int x0 = x-tid;
  int i=tid;
  while (i<vec_length*block_dim) {
    int space_idx = i / vec_length;
    int internal_idx = i - space_idx*vec_length;
    int sh_idx = internal_idx*(blockDim.x+1) + space_idx;
    field[x0*vec_length + i] = s_data[sh_idx];
    i += block_dim;
  }

} 

/**! float load specialization to obtain full coalescing. */
template<> __host__ __device__ inline void SpaceColorSpinorOrder<float, 4, 3>::load(float v[24], int x, int volume) const {
#ifdef __CUDA_ARCH__
  load_shared<float, 4, 3>(v, field, x, volume);
#else
  const int Ns=4;
  const int Nc=3;
  for (int s=0; s<Ns; s++) {
    for (int c=0; c<Nc; c++) {
      for (int z=0; z<2; z++) {
	v[(s*Nc+c)*2+z] = field[((x*Nc + c)*Ns + s)*2 + z]; 
      }
    }
  }
#endif
}

/**! float save specialization to obtain full coalescing. */
template<> __host__ __device__ inline void SpaceColorSpinorOrder<float, 4, 3>::save(const float v[24], int x, int volume) {
#ifdef __CUDA_ARCH__
  save_shared<float, 4, 3>(field, v, x, volume);
#else
  const int Ns=4;
  const int Nc=3;
  for (int s=0; s<Ns; s++) {
    for (int c=0; c<Nc; c++) {
      for (int z=0; z<2; z++) {
	field[((x*Nc + c)*Ns + s)*2 + z] = v[(s*Nc+c)*2+z];
      }
    }
  }
#endif
}

template <typename Float, int Ns, int Nc>
struct SpaceSpinorColorOrder {
  Float *field;
  int volume;
  int stride;
  SpaceSpinorColorOrder(Float *field, int volume, int stride) 
   : field(field), volume(volume), stride(stride)
  { if (volume != stride) errorQuda("Stride must equal volume for this field order"); }
  virtual ~SpaceSpinorColorOrder() { ; }

  __device__ __host__ inline void load(Float v[Ns*Nc*2], int x, int volume) const {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  v[(s*Nc+c)*2+z] = field[((x*Ns + s)*Nc + c)*2 + z];
	}
      }
    }
  }

  __device__ __host__ inline void save(const Float v[Ns*Nc*2], int x, int volume) {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  field[((x*Ns + s)*Nc + c)*2 + z] = v[(s*Nc+c)*2+z];
	}
      }
    }
  }

  size_t Bytes() const { return volume * Nc * Ns * 2 * sizeof(Float); }
};

/** Straight copy with no basis change */
template <typename FloatOut, typename FloatIn, int Ns, int Nc>
class PreserveBasis {
 public:
  __device__ __host__ inline void operator()(FloatOut out[Ns*Nc*2], const FloatIn in[Ns*Nc*2]) {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  out[(s*Nc+c)*2+z] = in[(s*Nc+c)*2+z];
	}
      }
    }
  }
};

/** Transform from relativistic into non-relavisitic basis */
template <typename FloatOut, typename FloatIn, int Ns, int Nc>
struct NonRelBasis {
  __device__ __host__ inline void operator()(FloatOut out[Ns*Nc*2], const FloatIn in[Ns*Nc*2]) {
    int s1[4] = {1, 2, 3, 0};
    int s2[4] = {3, 0, 1, 2};
    FloatOut K1[4] = {kP, -kP, -kP, -kP};
    FloatOut K2[4] = {kP, -kP, kP, kP};
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  out[(s*Nc+c)*2+z] = K1[s]*in[(s1[s]*Nc+c)*2+z] + K2[s]*in[(s2[s]*Nc+c)*2+z];
	}
      }
    }
  }
};

/** Transform from non-relativistic into relavisitic basis */
template <typename FloatOut, typename FloatIn, int Ns, int Nc>
struct RelBasis {
  __device__ __host__ inline void operator()(FloatOut out[Ns*Nc*2], const FloatIn in[Ns*Nc*2]) {
    int s1[4] = {1, 2, 3, 0};
    int s2[4] = {3, 0, 1, 2};
    FloatOut K1[4] = {-kU, kU,  kU,  kU};
    FloatOut K2[4] = {-kU, kU, -kU, -kU};
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
	for (int z=0; z<2; z++) {
	  out[(s*Nc+c)*2+z] = K1[s]*in[(s1[s]*Nc+c)*2+z] + K2[s]*in[(s2[s]*Nc+c)*2+z];
	}
      }
    }
  }
};

/** CPU function to reorder spinor fields.  */
template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis>
void packSpinor(OutOrder &outOrder, const InOrder &inOrder, Basis basis, int volume) {  
  for (int x=0; x<volume; x++) {
    FloatIn in[Ns*Nc*2];
    FloatOut out[Ns*Nc*2];
    inOrder.load(in, x, volume);
    basis(out, in);
    outOrder.save(out, x, volume);
  }
}

/** CUDA kernel to reorder spinor fields.  Adopts a similar form as the CPU version, using the same inlined functions. */
template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis>
__global__ void packSpinorKernel(OutOrder outOrder, const InOrder inOrder, Basis basis, int volume) {  
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  FloatIn in[Ns*Nc*2];
  FloatOut out[Ns*Nc*2];
  inOrder.load(in, x, volume);

  if (x >= volume) return;
  basis(out, in);
  outOrder.save(out, x, volume);
}

template <typename FloatOut, typename FloatIn, int Ns, int Nc, typename OutOrder, typename InOrder, typename Basis>
class PackSpinor : Tunable {
  const InOrder &in;
  OutOrder &out;
  Basis &basis;
  int volume;

 private:
  int sharedBytesPerThread() const { 
    size_t regSize = sizeof(FloatOut) > sizeof(FloatIn) ? sizeof(FloatOut) : sizeof(FloatIn);
    return Ns*Nc*2*regSize;
  }

  // the minimum shared memory per block is (block+1) because we pad to avoid bank conflicts
  int sharedBytesPerBlock(const TuneParam &param) const { return (param.block.x+1)*sharedBytesPerThread(); }
  bool advanceSharedBytes(TuneParam &param) const { return false; } // Don't tune shared mem
  bool advanceGridDim(TuneParam &param) const { return false; } // Don't tune the grid dimensions.
  bool advanceBlockDim(TuneParam &param) const {
    bool advance = Tunable::advanceBlockDim(param);
    if (advance) param.grid = dim3( (volume+param.block.x-1) / param.block.x, 1, 1);
    param.shared_bytes = sharedBytesPerThread() * (param.block.x+1); // FIXME: use sharedBytesPerBlock
    return advance;
  }

 public:
  PackSpinor(OutOrder &out, const InOrder &in, Basis &basis, int volume) 
   : out(out), in(in), basis(basis), volume(volume) { ; }
  virtual ~PackSpinor() { ; }
  
  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, QUDA_TUNE_YES, QUDA_DEBUG_VERBOSE);
    packSpinorKernel<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, Basis> 
      <<<tp.grid, tp.block, tp.shared_bytes, stream>>> 
      (out, in, basis, volume);
  }

  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << in.volume; 
    aux << "out_stride=" << out.stride << ",in_stride=" << in.stride;
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

  long long flops() const { return 0; } 
  long long bytes() const { return in.Bytes() + out.Bytes(); } 
};


/** Decide whether we are changing basis or not */
template <int Ns, int Nc, typename OutOrder, typename InOrder, typename FloatOut, typename FloatIn>
void packParitySpinor(FloatOut *dst, FloatIn *src, OutOrder &outOrder, const InOrder &inOrder, int Vh, int pad, 
		      QudaGammaBasis dstBasis, QudaGammaBasis srcBasis, QudaFieldLocation location) {
  if (dstBasis==srcBasis) {
    PreserveBasis<FloatOut, FloatIn, Ns, Nc> basis;
    if (location == QUDA_CPU_FIELD_LOCATION) {
      packSpinor<FloatOut, FloatIn, Ns, Nc>(outOrder, inOrder, basis, Vh);
    } else {
      PackSpinor<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, PreserveBasis<FloatOut, FloatIn, Ns, Nc> > pack(outOrder, inOrder, basis, Vh);
      pack.apply(0);
    }
  } else if (dstBasis == QUDA_UKQCD_GAMMA_BASIS && srcBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
    if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
    NonRelBasis<FloatOut, FloatIn, Ns, Nc> basis;
    if (location == QUDA_CPU_FIELD_LOCATION) {
      packSpinor<FloatOut, FloatIn, Ns, Nc>(outOrder, inOrder, basis, Vh);
    } else {
      PackSpinor<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, NonRelBasis<FloatOut, FloatIn, Ns, Nc> > pack(outOrder, inOrder, basis, Vh);
      pack.apply(0);
    }
  } else if (srcBasis == QUDA_UKQCD_GAMMA_BASIS && dstBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
    if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
    RelBasis<FloatOut, FloatIn, Ns, Nc> basis;
    if (location == QUDA_CPU_FIELD_LOCATION) {
      packSpinor<FloatOut, FloatIn, Ns, Nc>(outOrder, inOrder, basis, Vh);    
    } else {
      PackSpinor<FloatOut, FloatIn, Ns, Nc, OutOrder, InOrder, RelBasis<FloatOut, FloatIn, Ns, Nc> > pack(outOrder, inOrder, basis, Vh);
      pack.apply(0);
    } 
  } else {
    errorQuda("Basis change not supported");
  }
}


template <int Nc, int Ns, int N, typename dstFloat, typename srcFloat>
void packSpinor(dstFloat *Dst, srcFloat *Src, ColorSpinorField &dst, const ColorSpinorField &src, QudaFieldLocation location) {

  if (dst.Ndim() != src.Ndim()) {
    errorQuda("Number of dimensions %d %d don't match", dst.Ndim(), src.Ndim());
  }

  if (dst.Volume() != src.Volume()) {
    errorQuda("Volumes %d %d don't match", dst.Volume(), src.Volume());
  }

  if (dst.SiteOrder() != src.SiteOrder()) {
    errorQuda("Subset orders %d %d don't match", dst.SiteOrder(), src.SiteOrder());
  }

  if (dst.SiteSubset() != src.SiteSubset()) {
    errorQuda("Subset types do not match %d %d", dst.SiteSubset(), src.SiteSubset());
  }

  int V = dst.Volume();
  QudaSiteSubset subset = dst.SiteSubset();
  QudaSiteOrder siteOrder = dst.SiteOrder();
  int dstLength = dst.TotalLength();
  int srcLength = src.TotalLength();
  QudaGammaBasis dstBasis = dst.GammaBasis();
  QudaGammaBasis srcBasis = src.GammaBasis();
  QudaFieldOrder dstOrder = dst.FieldOrder();
  QudaFieldOrder srcOrder = src.FieldOrder(); 

  // We currently only support parity-ordered fields; even-odd or odd-even
  if (siteOrder == QUDA_LEXICOGRAPHIC_SITE_ORDER) {
    errorQuda("Copying to full fields with lexicographical ordering is not currently supported");
  }

  if (subset == QUDA_FULL_SITE_SUBSET) {
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
    if ((dstOrder == QUDA_FLOAT4_FIELD_ORDER || dstOrder == QUDA_FLOAT2_FIELD_ORDER) &&
	(srcOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER || srcOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER)) {
      if (src.Pad() != 0) errorQuda("Non-zero pad not supported with fieldOrder %d\n", srcOrder);
      if (srcOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	{
	  SpaceSpinorColorOrder<srcFloat, Ns, Nc> inOrder(Src+evenOff, Vh, Vh);
	  FloatNOrder<dstFloat, Ns, Nc, N> outOrder(Dst, Vh, Vh+dst.Pad());
	  packParitySpinor<Ns,Nc>(Dst, Src+evenOff, outOrder, inOrder, Vh, dst.Pad(), dstBasis, srcBasis, location);
	}
	{
	  SpaceSpinorColorOrder<srcFloat, Ns, Nc> inOrder(Src+oddOff, Vh, Vh);
	  FloatNOrder<dstFloat, Ns, Nc, N> outOrder(Dst + dstLength/2, Vh, Vh+dst.Pad());
	  packParitySpinor<Ns,Nc>(Dst + dstLength/2, Src+oddOff, outOrder, inOrder, Vh, dst.Pad(), dstBasis, srcBasis, location);
	}
      } else if (srcOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
	{
	  SpaceColorSpinorOrder<srcFloat, Ns, Nc> inOrder(Src+evenOff, Vh, Vh);
	  FloatNOrder<dstFloat, Ns, Nc, N> outOrder(Dst, Vh, Vh+dst.Pad());
	  packParitySpinor<Ns,Nc>(Dst, Src+evenOff, outOrder, inOrder, Vh, dst.Pad(), dstBasis, srcBasis, location);
	}
	{
	  SpaceColorSpinorOrder<srcFloat, Ns, Nc> inOrder(Src+oddOff, Vh, Vh);
	  FloatNOrder<dstFloat, Ns, Nc, N> outOrder(Dst + dstLength/2, Vh, Vh+dst.Pad());
	  packParitySpinor<Ns,Nc>(Dst + dstLength/2, Src+oddOff, outOrder, inOrder, Vh, dst.Pad(), dstBasis, srcBasis, location);
	}
      }
    } else if ((srcOrder == QUDA_FLOAT4_FIELD_ORDER || srcOrder == QUDA_FLOAT2_FIELD_ORDER) &&
	       (dstOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER || dstOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER)) {
      if (dst.Pad() != 0) errorQuda("Non-zero pad not supported with fieldOrder %d\n", dstOrder);
      if (dstOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	{
	  FloatNOrder<srcFloat, Ns, Nc, N> inOrder(Src+evenOff, Vh, Vh+src.Pad());
	  SpaceSpinorColorOrder<dstFloat, Ns, Nc> outOrder(Dst, Vh, Vh);
	  packParitySpinor<Ns,Nc>(Dst, Src+evenOff, outOrder, inOrder, Vh, src.Pad(), dstBasis, srcBasis, location);
	}
	{
	  FloatNOrder<srcFloat, Ns, Nc, N> inOrder(Src+oddOff, Vh, Vh+src.Pad());
	  SpaceSpinorColorOrder<dstFloat, Ns, Nc> outOrder(Dst + dstLength/2, Vh, Vh);
	  packParitySpinor<Ns,Nc>(Dst + dstLength/2, Src+oddOff, outOrder, inOrder, Vh, src.Pad(), dstBasis, srcBasis, location);
	}
      } else if (dstOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
	{
	  FloatNOrder<srcFloat, Ns, Nc, N> inOrder(Src+evenOff, Vh, Vh+src.Pad());
	  SpaceColorSpinorOrder<dstFloat, Ns, Nc> outOrder(Dst, Vh, Vh);
	  packParitySpinor<Ns,Nc>(Dst, Src+evenOff, outOrder, inOrder, Vh, src.Pad(), dstBasis, srcBasis, location);
	  }
	{	  
	  FloatNOrder<srcFloat, Ns, Nc, N> inOrder(Src+oddOff, Vh, Vh+src.Pad());
	  SpaceColorSpinorOrder<dstFloat, Ns, Nc> outOrder(Dst + dstLength/2, Vh, Vh);
	  packParitySpinor<Ns,Nc>(Dst + dstLength/2, Src+oddOff, outOrder, inOrder, Vh, src.Pad(), dstBasis, srcBasis, location);
	}
      }
    } else {
      errorQuda("Field order conversion from %d to %d not supported", srcOrder, dstOrder);
    }

  } else { // parity field

    if ((dstOrder == QUDA_FLOAT4_FIELD_ORDER || dstOrder == QUDA_FLOAT2_FIELD_ORDER) &&
	(srcOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER || srcOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER)) {
      if (src.Pad() != 0) errorQuda("Non-zero pad not supported with fieldOrder %d\n", srcOrder);
      if (srcOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	SpaceSpinorColorOrder<srcFloat, Ns, Nc> inOrder(Src, V, V);
	FloatNOrder<dstFloat, Ns, Nc, N> outOrder(Dst, V, V+dst.Pad());
	packParitySpinor<Ns,Nc>(Dst, Src, outOrder, inOrder, V, dst.Pad(), dstBasis, srcBasis, location);
      } else if (srcOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
	SpaceColorSpinorOrder<srcFloat, Ns, Nc> inOrder(Src, V, V);
	FloatNOrder<dstFloat, Ns, Nc, N> outOrder(Dst, V, V+dst.Pad());
	packParitySpinor<Ns,Nc>(Dst, Src, outOrder, inOrder, V, dst.Pad(), dstBasis, srcBasis, location);
      }
    } else if ((srcOrder == QUDA_FLOAT4_FIELD_ORDER || srcOrder == QUDA_FLOAT2_FIELD_ORDER) &&
	  (dstOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER || dstOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER)) {
      if (dst.Pad() != 0) errorQuda("Non-zero pad not supported with fieldOrder %d\n", dstOrder);
      if (dstOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	FloatNOrder<srcFloat, Ns, Nc, N> inOrder(Src, V, V+src.Pad());
	SpaceSpinorColorOrder<dstFloat, Ns, Nc> outOrder(Dst, V, V);
	packParitySpinor<Ns,Nc>(Dst, Src, outOrder, inOrder, V, src.Pad(), dstBasis, srcBasis, location);
      } else if (dstOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
	FloatNOrder<srcFloat, Ns, Nc, N> inOrder(Src, V, V+src.Pad());
	SpaceColorSpinorOrder<dstFloat, Ns, Nc> outOrder(Dst, V, V);
	packParitySpinor<Ns,Nc>(Dst, Src, outOrder, inOrder, V, src.Pad(), dstBasis, srcBasis, location);
      }
    } else {
      errorQuda("Field order conversion from %d to %d not supported", srcOrder, dstOrder);
    }

  } // parity or full

}

  /*template <int Nc, int Ns, int N, typename Float, typename FloatN>
void unpackSpinor(Float *dst, FloatN *src, int V, int pad, const int x[], int dstLength, 
		  int srcLength, QudaSiteSubset dstSubset, QudaSiteOrder siteOrder,  
		  QudaGammaBasis dstBasis, QudaGammaBasis srcBasis, QudaFieldOrder dstOrder, 
		  QudaFieldLocation location) {

  if (dstSubset == QUDA_FULL_SITE_SUBSET) {
    if (siteOrder == QUDA_LEXICOGRAPHIC_SITE_ORDER) {
      errorQuda("Copying to full fields with lexicographical ordering is not currently supported");
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
      if (dstOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	{
	  FloatNOrder<FloatN, Ns, Nc, N> inOrder(src+evenOff, Vh, Vh+pad);
	  SpaceSpinorColorOrder<Float, Ns, Nc> outOrder(dst, Vh, Vh);
	  packParitySpinor<Ns,Nc>(dst, src+evenOff, outOrder, inOrder, Vh, pad, dstBasis, srcBasis, location);
	}
	{
	  FloatNOrder<FloatN, Ns, Nc, N> inOrder(src+oddOff, Vh, Vh+pad);
	  SpaceSpinorColorOrder<Float, Ns, Nc> outOrder(dst + dstLength/2, Vh, Vh);
	  packParitySpinor<Ns,Nc>(dst + dstLength/2, src+oddOff, outOrder, inOrder, Vh, pad, dstBasis, srcBasis, location);
	}
      } else if (dstOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
	{
	  FloatNOrder<FloatN, Ns, Nc, N> inOrder(src+evenOff, Vh, Vh+pad);
	  SpaceColorSpinorOrder<Float, Ns, Nc> outOrder(dst, Vh, Vh);
	  packParitySpinor<Ns,Nc>(dst, src+evenOff, outOrder, inOrder, Vh, pad, dstBasis, srcBasis, location);
	}
	{	  
	  FloatNOrder<FloatN, Ns, Nc, N> inOrder(src+oddOff, Vh, Vh+pad);
	  SpaceColorSpinorOrder<Float, Ns, Nc> outOrder(dst + dstLength/2, Vh, Vh);
	  packParitySpinor<Ns,Nc>(dst + dstLength/2, src+oddOff, outOrder, inOrder, Vh, pad, dstBasis, srcBasis, location);
	}
      } else {
	errorQuda("Destination field order not supported");
      }
    }
  } else {
    // dst is defined on a single parity only

  }

  }*/



/*
template <int Nc, int Ns, int N, typename Float, typename FloatN>
void packFullSpinor(FloatN *dst, Float *src, int V, int pad, const int x[], int dstLength,
		    QudaGammaBasis dstBasis, QudaGammaBasis srcBasis) {
  
  int Vh = V/2;
  if (dstBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packSpinorField<Nc,Ns,N>(dst+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packSpinorField<Nc,Ns,N>(dst+dstLength/2+N*i, src+ 2*Nc*Ns*k, Vh+pad);
      }
    }
  } else if (dstBasis == QUDA_UKQCD_GAMMA_BASIS && srcBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
    if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	packNonRelSpinorField<Nc,N>(dst+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	packNonRelSpinorField<Nc,N>(dst+dstLength/2+N*i, src+2*Nc*Ns*k, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
}


template <int Nc, int Ns, int N, typename Float, typename FloatN>
void unpackFullSpinor(Float *dst, FloatN *src, int V, int pad, const int x[], 
		      int srcLength, QudaGammaBasis dstBasis, QudaGammaBasis srcBasis) {
  
  int Vh = V/2;
  if (dstBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackSpinorField<Nc,Ns,N>(dst+2*Ns*Nc*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackSpinorField<Nc,Ns,N>(dst+2*Ns*Nc*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else if (srcBasis == QUDA_UKQCD_GAMMA_BASIS && dstBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
    if (Ns != 4) errorQuda("Can only change basis with Nspin = 4, not Nspin = %d", Ns);
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackNonRelSpinorField<Nc,N>(dst+2*Ns*Nc*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackNonRelSpinorField<Nc,N>(dst+2*Ns*Nc*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
}

template <int Nc, int Ns, int N, typename Float, typename FloatN>
void unpackQLAFullSpinor(Float *dst, FloatN *src, int V, int pad, const int x[], 
			 int srcLength, QudaGammaBasis dstBasis, QudaGammaBasis srcBasis) {
  
  int Vh = V/2;
  if (dstBasis==srcBasis) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackQLASpinorField<Nc,Ns,N>(dst+2*Nc*Ns*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackQLASpinorField<Nc,Ns,N>(dst+2*Nc*Ns*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else if (srcBasis == QUDA_UKQCD_GAMMA_BASIS && dstBasis == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
    for (int i=0; i<V/2; i++) {
      
      int boundaryCrossings = i/(x[0]/2) + i/(x[1]*(x[0]/2)) + i/(x[2]*x[1]*(x[0]/2));
      
      { // even sites
	int k = 2*i + boundaryCrossings%2; 
	unpackNonRelQLASpinorField<Nc,N>(dst+2*Ns*Nc*k, src+N*i, Vh+pad);
      }
      
      { // odd sites
	int k = 2*i + (boundaryCrossings+1)%2;
	unpackNonRelQLASpinorField<Nc,N>(dst+2*Ns*Nc*k, src+srcLength/2+N*i, Vh+pad);
      }
    }
  } else {
    errorQuda("Basis change not supported");
  }
  }*/
