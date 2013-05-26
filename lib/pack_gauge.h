#include <tune_quda.h>

namespace quda {

  /*
    Here we use traits to define the mapping between storage type and
    register type:
    double -> double
    float -> float
    short -> float
    This allows us to wrap the encapsulate the register type into the storage template type
   */
  template<typename> struct mapper { };
  template<> struct mapper<double> { typedef double type; };
  template<> struct mapper<float> { typedef float type; };
  template<> struct mapper<short> { typedef float type; };

  /* Traits used to determine if a variable is half precision or not */
  template< typename T > struct isHalf{ static const bool value = false; };
  template<> struct isHalf<short>{ static const bool value = true; };

  template<typename T1, typename T2> __host__ __device__ inline void copy (T1 &a, const T2 &b) { a = b; }
  template<> __host__ __device__ inline void copy(float &a, const short &b) { a = (float)b/MAX_SHORT; }
  template<> __host__ __device__ inline void copy(short &a, const float &b) { a = (short)(b*MAX_SHORT); }

  /**
     Generic wrapper for Trig functions
  */
  template <bool isHalf>
    struct Trig {
      template<typename T> 
      __device__ __host__ static T Atan2( const T &a, const T &b) { return atan2(a,b); }
      template<typename T> 
      __device__ __host__ static T Sin( const T &a ) { return sin(a); }
      template<typename T> 
      __device__ __host__ static T Cos( const T &a ) { return cos(a); }
    };
  
  /**
     Specialization of Trig functions using shorts
   */
  template <>
    struct Trig<true> {
    template<typename T> 
      __device__ __host__ static T Atan2( const T &a, const T &b) { return atan2(a,b)/M_PI; }
    template<typename T> 
      __device__ __host__ static T Sin( const T &a ) { return sin(a*M_PI); }
    template<typename T> 
      __device__ __host__ static T Cos( const T &a ) { return cos(a*M_PI); }
  };
  


  // a += b*c
  template <typename Float>
    __device__ __host__ inline void accumulateComplexProduct(Float *a, Float *b, Float *c, Float sign) {
    a[0] += sign*(b[0]*c[0] - b[1]*c[1]);
    a[1] += sign*(b[0]*c[1] + b[1]*c[0]);
  }

  // a = conj(b)*c
  template <typename Float>
    __device__ __host__ inline void complexDotProduct(Float *a, Float *b, Float *c) {
    a[0] = b[0]*c[0] + b[1]*c[1];
    a[1] = b[0]*c[1] - b[1]*c[0];
  }

  // a += conj(b) * conj(c)
  template <typename Float>
    __device__ __host__ inline void accumulateConjugateProduct(Float *a, Float *b, Float *c, int sign) {
    a[0] += sign * (b[0]*c[0] - b[1]*c[1]);
    a[1] -= sign * (b[0]*c[1] + b[1]*c[0]);
  }

  // a = conj(b)*conj(c)
  template <typename Float>
    __device__ __host__ inline void complexConjugateProduct(Float *a, Float *b, Float *c) {
    a[0] = b[0]*c[0] - b[1]*c[1];
    a[1] = -b[0]*c[1] - b[1]*c[0];
  }

  /** Generic reconstruction is no reconstruction */
  template <int N, typename Float> 
    struct Reconstruct {
    typedef typename mapper<Float>::type RegType;
    Reconstruct(const GaugeField &u) { ; }
    
    __device__ __host__ inline void Pack(RegType out[N], const RegType in[N]) const {
      for (int i=0; i<N; i++) out[i] = in[i];
    }
    __device__ __host__ inline void Unpack(RegType out[N], const RegType in[N], int idx, int dir) const {
      for (int i=0; i<N; i++) out[i] = in[i];
    }
  };

  /** No reconstruction but we scale the result. This is used for
      half-precision non-unitary fields, e.g., staggered fat link */
  template <typename Float>
    struct Reconstruct<19,Float> {
    typedef typename mapper<Float>::type RegType;
    RegType scale;
    Reconstruct(const GaugeField &u) : scale(u.LinkMax()) { ; }

    __device__ __host__ inline void Pack(RegType out[18], const RegType in[18]) const {
      for (int i=0; i<18; i++) out[i] = in[i] / scale;
    }
    __device__ __host__ inline void Unpack(RegType out[18], const RegType in[18],
					   int idx, int dir) const {
      for (int i=0; i<18; i++) out[i] = scale * in[i];
    }
  };

  template <typename Float>
    struct Reconstruct<12,Float> {
      typedef typename mapper<Float>::type RegType;
      int X[QUDA_MAX_DIM];
      const RegType anisotropy;
      const QudaTboundary tBoundary;

      Reconstruct(const GaugeField &u) : anisotropy(u.Anisotropy()), tBoundary(u.TBoundary()) 
      {	for (int i=0; i<QUDA_MAX_DIM; i++) X[i] = u.X()[i]; }

      __device__ __host__ inline void Pack(RegType out[12], const RegType in[12]) const {
	for (int i=0; i<12; i++) out[i] = in[i];
      }

      __device__ __host__ inline void Unpack(RegType out[12], const RegType in[12],
					     int idx, int dir) const {
	for (int i=0; i<12; i++) out[i] = in[i];
	for (int i=12; i<18; i++) out[i] = 0.0;
	accumulateConjugateProduct(&out[12], &out[2], &out[10], +1);
	accumulateConjugateProduct(&out[12], &out[4], &out[8], -1);
	accumulateConjugateProduct(&out[14], &out[4], &out[6], +1);
	accumulateConjugateProduct(&out[14], &out[0], &out[10], -1);
	accumulateConjugateProduct(&out[16], &out[0], &out[8], +1);
	accumulateConjugateProduct(&out[16], &out[2], &out[6], -1);
	RegType u0 = (dir < 3 ? anisotropy :
		    (idx >= (X[3]-1)*X[0]*X[1]*X[2]/2 ? tBoundary : 1));
	for (int i=12; i<18; i++) out[i]*=u0;
      }
    };

  template <typename Float>
    struct Reconstruct<8,Float> {
      typedef typename mapper<Float>::type RegType;
      int X[QUDA_MAX_DIM];
      const RegType anisotropy;
      const QudaTboundary tBoundary;

      Reconstruct(const GaugeField &u) : anisotropy(u.Anisotropy()), tBoundary(u.TBoundary()) 
      {	for (int i=0; i<QUDA_MAX_DIM; i++) X[i] = u.X()[i]; }

      __device__ __host__ inline void Pack(RegType out[8], const RegType in[18]) const {
	out[0] = Trig<isHalf<Float>::value>::Atan2(in[1], in[0]);
	out[1] = Trig<isHalf<Float>::value>::Atan2(in[13], in[12]);
	for (int i=2; i<8; i++) out[i] = in[i];
      }

      __device__ __host__ inline void Unpack(RegType out[18], const RegType in[8],
					     int idx, int dir) const {
	// First reconstruct first row
	RegType row_sum = 0.0;
	for (int i=2; i<6; i++) {
	  out[i] = in[i];
	  row_sum += in[i]*in[i];
	}

	RegType u0 = (dir < 3 ? anisotropy :
		    (idx >= (X[3]-1)*X[0]*X[1]*X[2]/2 ? tBoundary : 1));
	RegType diff = 1.0/(u0*u0) - row_sum;
	RegType U00_mag = sqrt(diff >= 0 ? diff : 0.0);
    
	out[0] = U00_mag * Trig<isHalf<Float>::value>::Cos(in[0]);
	out[1] = U00_mag * Trig<isHalf<Float>::value>::Sin(in[0]);
	  
	// Now reconstruct first column
	RegType column_sum = 0.0;
	for (int i=0; i<2; i++) column_sum += out[i]*out[i];
	for (int i=6; i<8; i++) {
	  out[i] = in[i];
	  column_sum += in[i]*in[i];
	}
	diff = 1.f/(u0*u0) - column_sum;
	RegType U20_mag = sqrt(diff >= 0 ? diff : 0.0);
	
	out[12] = U20_mag * Trig<isHalf<Float>::value>::Cos(in[1]);
	out[13] = U20_mag * Trig<isHalf<Float>::value>::Sin(in[1]);
	// First column now restored
	
	// finally reconstruct last elements from SU(2) rotation
	RegType r_inv2 = 1.0/(u0*row_sum);
	
	// U11
	RegType A[2];
	complexDotProduct(A, out+0, out+6);
	complexConjugateProduct(out+8, out+12, out+4);
	accumulateComplexProduct(out+8, A, out+2, u0);
	out[8] *= -r_inv2;
	out[9] *= -r_inv2;
	
	// U12
	complexConjugateProduct(out+10, out+12, out+2);
	accumulateComplexProduct(out+10, A, out+4, -u0);
	out[10] *= r_inv2;
	out[11] *= r_inv2;
	
	// U21
	complexDotProduct(A, out+0, out+12);
	complexConjugateProduct(out+14, out+6, out+4);
	accumulateComplexProduct(out+14, A, out+2, -u0);
	out[14] *= r_inv2;
	out[15] *= r_inv2;
	
	// U12
	complexConjugateProduct(out+16, out+6, out+2);
	accumulateComplexProduct(out+16, A, out+4, u0);
	out[16] *= -r_inv2;
	out[17] *= -r_inv2;
      }
    };

  template <typename Float, int length, int N, int reconLen>
    struct FloatNOrder {
      typedef typename mapper<Float>::type RegType;
      Reconstruct<reconLen,Float> reconstruct;
      Float *gauge[2];
      int volumeCB;
      int stride;

    FloatNOrder(Float *gauge_, const GaugeField &u) : 
      reconstruct(u), volumeCB(u.VolumeCB()), stride(u.Stride()) 
      { gauge[0] = gauge_; gauge[1] = (Float*)((char*)gauge_ + u.Bytes()/2); }
      virtual ~FloatNOrder() { ; }
  
      __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity) const {
	const int M = reconLen / N;
	RegType tmp[reconLen];
	for (int i=0; i<M; i++) {
	  for (int j=0; j<N; j++) {
	    int intIdx = i*N + j; // internal dof index
	    int padIdx = intIdx / N;
	    copy(tmp[i*N+j], gauge[parity][dir*stride*M*N + (padIdx*stride + x)*N + intIdx%N]);
	  }
	}
	reconstruct.Unpack(v, tmp, x, dir);
      }
  
      __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {
	const int M = reconLen / N;
	RegType tmp[reconLen];
	reconstruct.Pack(tmp, v);
	for (int i=0; i<M; i++) {
	  for (int j=0; j<N; j++) {
	    int intIdx = i*N + j;
	    int padIdx = intIdx / N;
	    copy(gauge[parity][dir*stride*M*N + (padIdx*stride + x)*N + intIdx%N], tmp[i*N+j]);
	  }
	}
      }

      __device__ __host__ inline void loadGhost(RegType v[length], int x, int dir, int parity) const {
	load(v, volumeCB+x, dir, parity); // an offset of size volumeCB puts us at the padded region
      }
      
      __device__ __host__ inline void saveGhost(const RegType v[length], int x, int dir, int parity) {
	save(v, volumeCB+x, dir, parity); // an offset of size volumeCB puts us at the padded region
      }

      size_t Bytes() const { return 4 * 2 * volumeCB * reconLen * sizeof(Float); }
    };

  /** 
      The LegacyOrder defines the ghost zone storage and ordering for
      all cpuGaugeFields, which use the same ghost zone storage.
  */
  template <typename Float, int length> 
    struct LegacyOrder {
      typedef typename mapper<Float>::type RegType;
      Float *ghost[QUDA_MAX_DIM];
      int faceVolumeCB[QUDA_MAX_DIM];
      int volumeCB;
      LegacyOrder(const GaugeField &u) : volumeCB(u.VolumeCB()) {
	for (int i=0; i<4; i++) {
	  ghost[i] = (Float*)(u.Ghost()[i]);
	  faceVolumeCB[i] = u.SurfaceCB(i)*u.Nface(); // face volume equals surface * depth
	}
      }
      virtual ~LegacyOrder() { ; }
      
      __device__ __host__ inline void loadGhost(RegType v[length], int x, int dir, int parity) const {
	for (int i=0; i<length; i++) v[i] = ghost[dir][(parity*faceVolumeCB[dir] + x)*length + i];
      }
      
      __device__ __host__ inline void saveGhost(const RegType v[length], int x, int dir, int parity) {
	for (int i=0; i<length; i++) ghost[dir][(parity*faceVolumeCB[dir] + x)*length + i] = v[i];
      }
      
      virtual size_t Bytes() const { 
	size_t bytes = 0;
	for (int d=0; d<4; d++) bytes += 2 * faceVolumeCB[d] * length * sizeof(Float);
	return bytes;
      }
    };

  /**
     struct to define QDP ordered gauge fields: 
     [[dim]] [[parity][volumecb][row][col]]
   */
    template <typename Float, int length> struct QDPOrder : public LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge[QUDA_MAX_DIM];
    int volumeCB;
    QDPOrder(void *gauge_, const GaugeField &u) : LegacyOrder<Float,length>(u), volumeCB(u.VolumeCB())
      { for (int i=0; i<4; i++) gauge[i] = ((Float**)gauge_)[i]; }
    virtual ~QDPOrder() { ; }
    
    __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity) const {
      for (int i=0; i<length; i++) {
	v[i] = (RegType)gauge[dir][(parity*volumeCB + x)*length + i];
      }
    }
  
    __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {
      for (int i=0; i<length; i++) {
	gauge[dir][(parity*volumeCB + x)*length + i] = (Float)v[i];
      }
    }

    size_t Bytes() const { return 4 * 2 * volumeCB * length * sizeof(Float); }
  };

  /**
     struct to define MILC ordered gauge fields: 
     [parity][dim][volumecb][row][col]
   */
    template <typename Float, int length> struct MILCOrder : public LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge;
    int volumeCB;
  MILCOrder(void *gauge, const GaugeField &u) : LegacyOrder<Float,length>(u), gauge((Float*)gauge), volumeCB(u.VolumeCB()) { ; }
    virtual ~MILCOrder() { ; }
    
    __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity) const {
      for (int i=0; i<length; i++) {
	v[i] = (RegType)gauge[((parity*volumeCB+x)*4 + dir)*length + i];
      }
    }
  
    __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {
      for (int i=0; i<length; i++) {
	gauge[((parity*volumeCB+x)*4 + dir)*length + i] = (Float)v[i];
      }
    }

    size_t Bytes() const { return 4 * 2 * volumeCB * length * sizeof(Float); }
  };
  
  /**
     struct to define CPS ordered gauge fields: 
     [parity][dim][volumecb][col][row]
   */
    template <typename Float, int length> struct CPSOrder : LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge;
    int volumeCB;
    Float anisotropy;
    const int Nc;
  CPSOrder(void *gauge, const GaugeField &u) : LegacyOrder<Float,length>(u), gauge((Float*)gauge), volumeCB(u.VolumeCB()), anisotropy(u.Anisotropy()), Nc(3) 
    { if (length != 18) errorQuda("Gauge length %d not supported", length); }
    virtual ~CPSOrder() { ; }
    
    // we need to transpose and scale for CPS ordering
    __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity) const {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    v[(i*Nc+j)*2+z] = 
	      (RegType)(gauge[((((parity*volumeCB+x)*4 + dir)*Nc + j)*Nc + i)*2 + z] / anisotropy);
	  }
	}
      }
    }
  
    __device__ __host__ inline void save(const RegType v[18], int x, int dir, int parity) {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    gauge[((((parity*volumeCB+x)*4 + dir)*Nc + j)*Nc + i)*2 + z] = 
	      (Float)(anisotropy * v[(i*Nc+j)*2+z]);
	  }
	}
      }
    }

    size_t Bytes() const { return 4 * 2 * volumeCB * Nc * Nc * 2 * sizeof(Float); }
  };
  
  /**
     struct to define BQCD ordered gauge fields: 
     [mu][parity][volumecb+halos][col][row]
   */
    template <typename Float, int length> struct BQCDOrder : LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge;
    int volumeCB;
    int exVolumeCB; // extended checkerboard volume
    const int Nc;
  BQCDOrder(void *gauge, const GaugeField &u) : LegacyOrder<Float,length>(u), gauge((Float*)gauge), volumeCB(u.VolumeCB()), Nc(3) { 
      if (length != 18) errorQuda("Gauge length %d not supported", length);
      // compute volumeCB + halo region
      exVolumeCB = u.X()[0]/2 + 2;
      for (int i=1; i<4; i++) exVolumeCB *= u.X()[i] + 2; 
    }
    virtual ~BQCDOrder() { ; }
    
    // we need to transpose for BQCD ordering
    __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity) const {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    v[(i*Nc+j)*2+z] = (RegType)gauge[((((dir*2+parity)*exVolumeCB + x)*Nc + j)*Nc + i)*2 + z];
	  }
	}
      }
    }
  
    __device__ __host__ inline void save(const RegType v[18], int x, int dir, int parity) {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    gauge[((((dir*2+parity)*exVolumeCB + x)*Nc + j)*Nc + i)*2 + z] = (Float)v[(i*Nc+j)*2+z];
	  }
	}
      }
    }

    size_t Bytes() const { return 4 * 2 * volumeCB * Nc * Nc * 2 * sizeof(Float); }
  };

  /**
     Generic CPU gauge reordering and packing 
  */
  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    void packGauge(OutOrder outOrder, const InOrder inOrder, int volume, int nDim) {  
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<nDim; d++) {
	for (int x=0; x<volume/2; x++) {
	  RegTypeIn in[length];
	  RegTypeOut out[length];
	  inOrder.load(in, x, d, parity);
	  for (int i=0; i<length; i++) out[i] = in[i];
	  outOrder.save(out, x, d, parity);
	}
      }

    }
  }

  /** 
      Generic CUDA gauge reordering and packing.  Adopts a similar form as
      the CPU version, using the same inlined functions.
  */
  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    __global__ void packGaugeKernel(OutOrder outOrder, const InOrder inOrder, int volume, int nDim) {  
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<nDim; d++) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= volume/2) return;

	RegTypeIn in[length];
	RegTypeOut out[length];
	inOrder.load(in, x, d, parity);
	for (int i=0; i<length; i++) out[i] = in[i];
	outOrder.save(out, x, d, parity);
      }
    }
  }

  /**
     Generic CPU gauge ghost reordering and packing 
  */
  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    void packGhost(OutOrder outOrder, const InOrder inOrder, const int *faceVolumeCB, int nDim) {  
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<nDim; d++) {
	for (int x=0; x<faceVolumeCB[d]; x++) {
	  RegTypeIn in[length];
	  RegTypeOut out[length];
	  inOrder.loadGhost(in, x, d, parity); // assumes we are loading 
	  for (int i=0; i<length; i++) out[i] = in[i];
	  outOrder.saveGhost(out, x, d, parity); // just need to add on volume offset for FloatN gauge
	}
      }

    }
  }

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    class PackGauge : Tunable {
    const InOrder &in;
    OutOrder &out;
    const int volume;
    const int nDim;

  private:
    int sharedBytesPerThread() const { return 0; }
    int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool advanceGridDim(TuneParam &param) const { return false; } // Don't tune the grid dimensions.
    bool advanceBlockDim(TuneParam &param) const {
      bool advance = Tunable::advanceBlockDim(param);
      if (advance) param.grid = dim3( (volume/2+param.block.x-1) / param.block.x, 1, 1);
      return advance;
    }

  public:
  PackGauge(OutOrder &out, const InOrder &in, int volume, int nDim) 
    : out(out), in(in), volume(volume), nDim(nDim) { ; }
    virtual ~PackGauge() { ; }
  
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, QUDA_TUNE_YES, QUDA_DEBUG_VERBOSE);
      packGaugeKernel<FloatOut, FloatIn, length, OutOrder, InOrder> 
	<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(out, in, volume, nDim);
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
      param.grid = dim3( (volume/2+param.block.x-1) / param.block.x, 1, 1);
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const {
      Tunable::defaultTuneParam(param);
      param.grid = dim3( (volume/2+param.block.x-1) / param.block.x, 1, 1);
    }

    long long flops() const { return 0; } 
    long long bytes() const { return in.Bytes() + out.Bytes(); } 
  };

  // FIXME add support for multi-GPU
  // ghost zone when doing D2D

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    void packGauge(OutOrder outOrder, const InOrder inOrder, int volume, const int *faceVolumeCB, int nDim, int ghost) {

    if (!ghost) {
      packGauge<FloatOut, FloatIn, length>(outOrder, inOrder, volume, nDim);
    } else {
      packGhost<FloatOut, FloatIn, length>(outOrder, inOrder, faceVolumeCB, nDim);
    }

  }

  template <typename FloatOut, typename FloatIn, int length, typename InOrder>
    void packGauge(const InOrder &inOrder, FloatOut *Out, GaugeField &out, int ghost) {
    int faceVolumeCB[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) faceVolumeCB[i] = out.SurfaceCB(i) * out.Nface(); 
    if (out.Order() == QUDA_FLOAT_GAUGE_ORDER) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(FloatOut)==typeid(short) && out.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  packGauge<FloatOut,FloatIn,length, FloatNOrder<FloatOut,length,1,19>, InOrder>
	    (FloatNOrder<FloatOut,length,1,19>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
	} else {
	  packGauge<FloatOut,FloatIn,length>
	    (FloatNOrder<FloatOut,length,1,18>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
	}
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
	packGauge<FloatOut,FloatIn,length> 
	  (FloatNOrder<FloatOut,length,1,12>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_8) {
	packGauge<FloatOut,FloatIn,length>
	  (FloatNOrder<FloatOut,length,1,8>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
      }
    } else if (out.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(FloatOut)==typeid(short) && out.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  packGauge<FloatOut,FloatIn,length>
	    (FloatNOrder<FloatOut,length,2,19>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
	} else {
	  packGauge<FloatOut,FloatIn,length>
	    (FloatNOrder<FloatOut,length,2,18>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
	}
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
	packGauge<FloatOut,FloatIn,length> 
	  (FloatNOrder<FloatOut,length,2,12>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);	   
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_8) {
	packGauge<FloatOut,FloatIn,length> 
	  (FloatNOrder<FloatOut,length,2,8>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);	   
      }
    } else if (out.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(FloatOut)==typeid(short) && out.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  packGauge<FloatOut,FloatIn,length>
	    (FloatNOrder<FloatOut,length,1,19>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
	} else {
	  packGauge<FloatOut,FloatIn,length>
	    (FloatNOrder<FloatOut,length,1,18>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
	}
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
	packGauge<FloatOut,FloatIn,length> 
	  (FloatNOrder<FloatOut,length,4,12>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_8) {
	packGauge<FloatOut,FloatIn,length> 
	  (FloatNOrder<FloatOut,length,4,8>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
      }
    } else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {
      packGauge<FloatOut,FloatIn,length>
	(QDPOrder<FloatOut,length>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
    } else if (out.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {
      packGauge<FloatOut,FloatIn,length>
	(CPSOrder<FloatOut,length>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {
      packGauge<FloatOut,FloatIn,length>
	(MILCOrder<FloatOut,length>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
    } else if (out.Order() == QUDA_BQCD_GAUGE_ORDER) {
      packGauge<FloatOut,FloatIn,length>
	(BQCDOrder<FloatOut,length>(Out, out), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), ghost);
    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }

  }

  template <typename FloatOut, typename FloatIn, int length>
    void packGauge(FloatOut *Out, FloatIn *In, GaugeField &out, const GaugeField &in, int ghost) {

    // reconstruction only supported on FloatN fields currently
    if (in.Order() == QUDA_FLOAT_GAUGE_ORDER) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(FloatOut)==typeid(short) && out.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,1,19>(In, in), Out, out, ghost);
	} else {
	  packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,1,18>(In, in), Out, out, ghost);
	}
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
	packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,1,12>(In, in), Out, out, ghost);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
	packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,1,8>(In, in), Out, out, ghost);
      }
    } else if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(FloatOut)==typeid(short) && out.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,2,19>(In, in), Out, out, ghost);
	} else {
	  packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,2,18>(In, in), Out, out, ghost);
	}
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
	packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,2,12>(In, in), Out, out, ghost);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
	packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,2,8>(In, in), Out, out, ghost);
      }      
    } else if (in.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(FloatOut)==typeid(short) && out.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,4,19>(In, in), Out, out, ghost);
	} else {
	  packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,4,18>(In, in), Out, out, ghost);
	}
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
	packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,4,12>(In, in), Out, out, ghost);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
	packGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,4,8>(In, in), Out, out, ghost);
      }
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {
      packGauge<FloatOut,FloatIn,length>(QDPOrder<FloatIn,length>(In, in), Out, out, ghost);
    } else if (in.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {
      packGauge<FloatOut,FloatIn,length>(CPSOrder<FloatIn,length>(In, in), Out, out, ghost);
    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
      packGauge<FloatOut,FloatIn,length>(MILCOrder<FloatIn,length>(In, in), Out, out, ghost);
    } else if (in.Order() == QUDA_BQCD_GAUGE_ORDER) {
      packGauge<FloatOut,FloatIn,length>(BQCDOrder<FloatIn,length>(In, in), Out, out, ghost);
    } else {
      errorQuda("Gauge field %d order not supported", in.Order());
    }

  }

  template <typename FloatOut, typename FloatIn>
    void packGauge(FloatOut *Out, FloatIn *In, GaugeField &out, const GaugeField &in, int ghost) {

    if (in.Ncolor() != 3 && out.Ncolor() != 3) {
      errorQuda("Unsupported number of colors; out.Nc=%d, in.Nc=%d", out.Ncolor(), in.Ncolor());
    }
    
    if (out.LinkType() != QUDA_ASQTAD_MOM_LINKS && out.LinkType() != QUDA_ASQTAD_MOM_LINKS) {
      // we are doing gauge field packing
      packGauge<FloatOut,FloatIn,18>(Out, In, out, in, ghost);
    } else {
      // we are doing momentum field packing
      if (in.Reconstruct() != QUDA_RECONSTRUCT_10 && out.Reconstruct() != 10) {
	errorQuda("Unsupported reconstruction types out=%d in=%d for momentum field", 
		  out.Reconstruct(), in.Reconstruct());
      }
    
      const int length = 10;
      
      // momentum only currently supported on MILC and Float2 fields currently
      if (out.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	  packGauge<FloatOut,FloatIn,10> 
	    (FloatNOrder<FloatOut,10,2,10>(Out, out), FloatNOrder<FloatIn,10,2,10>(In, in), 
	     out.Volume(), out.Ndim());
	} else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
	  packGauge<FloatOut,FloatIn,length> 
	    (FloatNOrder<FloatOut,10,2,10>(Out, out), MILCOrder<FloatIn,10>(In, in), 
	     out.Volume(), out.Ndim());
	} else {
	  errorQuda("Gauge field orders %d not supported", in.Order());
	}
      } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {
	if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	  packGauge<FloatOut,FloatIn,length> 
	    (MILCOrder<FloatOut,10>(Out, out), FloatNOrder<FloatIn,10,2,10>(In, in), 
	     out.Volume(), out.Ndim());
	} else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
	  packGauge<FloatOut,FloatIn,length> 
	    (MILCOrder<FloatOut,10>(Out, out), MILCOrder<FloatIn,10>(In, in), 
	     out.Volume(), out.Ndim());
	} else {
	  errorQuda("Gauge field orders %d not supported", in.Order());
	}
      } else {
	errorQuda("Gauge field orders %d not supported", out.Order());
      }
    }
  }

  // this is the function that is actually called, from here on down we instantiate all required templates
  void packGauge(void *Out, void *In, GaugeField &out, const GaugeField &in, int ghost=0) {
    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	packGauge((double*)Out, (double*)In, out, in, ghost);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	packGauge((double*)Out, (float*)In, out, in, ghost);
      }
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	packGauge((float*)Out, (double*)In, out, in, ghost);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	packGauge((float*)Out, (float*)In, out, in, ghost);
      }
    } else if (out.Precision() == QUDA_HALF_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION){
	packGauge((short*)Out, (double*)In, out, in, ghost);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	packGauge((short*)Out, (float*)In, out, in, ghost);
      }
    } 
  }

  // this is just a wrapper to packGauge - reuse all of its template instantiation
  void packGhost(void *Out, void *In, GaugeField &out, const GaugeField &in) { packGauge(Out, In, out, in, 1); }

  /**
     Generic CPU function find the gauge maximum
  */
  template <typename Float, int Nc, typename Order>
    double maxGauge(const Order order, int volume, int nDim) {  
    typedef typename mapper<Float>::type RegType;
    RegType max = 0.0;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<nDim; d++) {
	for (int x=0; x<volume/2; x++) {
	  RegType v[Nc*Nc*2];
	  order.load(v, x, d, parity);
	  for (int i=0; i<Nc*Nc*2; i++) if (abs(v[i]) > max) { max = v[i]; }
	}
      }

    }

    return max;
  }

  template <typename Float>
    double maxGauge(const GaugeField &u) {

    if (u.Ncolor() != 3) {
      errorQuda("Unsupported number of colors; Nc=%d", u.Ncolor());
    }

    const int Nc = 3;

    double max;
    // max only supported on external fields currently
    if (u.Order() == QUDA_QDP_GAUGE_ORDER) {
      max = maxGauge<Float,Nc>(QDPOrder<Float,2*Nc*Nc>((Float*)u.Gauge_p(), u),u.Volume(),4);
    } else if (u.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {
      max = maxGauge<Float,Nc>(CPSOrder<Float,2*Nc*Nc>((Float*)u.Gauge_p(), u),u.Volume(),4);
    } else if (u.Order() == QUDA_MILC_GAUGE_ORDER) {
      max = maxGauge<Float,Nc>(MILCOrder<Float,2*Nc*Nc>((Float*)u.Gauge_p(), u),u.Volume(),4);
    } else if (u.Order() == QUDA_BQCD_GAUGE_ORDER) {
      max = maxGauge<Float,Nc>(BQCDOrder<Float,2*Nc*Nc>((Float*)u.Gauge_p(), u),u.Volume(),4);
    } else {
      errorQuda("Gauge field %d order not supported", u.Order());
    }

    reduceMaxDouble(max);
    return max;
  }

  double maxGauge(const GaugeField &u) {
    double max = 0;
    if (u.Precision() == QUDA_DOUBLE_PRECISION) {
      max = maxGauge<double>(u);
    } else if (u.Precision() == QUDA_SINGLE_PRECISION) {
      max = maxGauge<float>(u);
    } else {
      errorQuda("Precision %d undefined", u.Precision());
    }
    return max;
  }

}

