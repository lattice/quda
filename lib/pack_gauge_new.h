#include <tune_quda.h>

namespace quda {

  template <typename Float>
    static inline __device__ __host__ Float short2float(short a) {
    return (Float)a/MAX_SHORT;
  }

  template <typename Float>
    static inline __device__ __host__ short float2short(float a) {
    return (short)(a*MAX_SHORT);
  }

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

  template <typename Float, int Nc>
    struct ReconstructNo {
      __device__ __host__ inline void pack(Float out[Nc*Nc*2], Float in[Nc*Nc*2]) const {
	for (int i=0; i<Nc*Nc*2; i++) out[i] = in[i];
      }
      __device__ __host__ inline void unpack(Float out[Nc*Nc*2], Float in[Nc*Nc*2]) const {
	for (int i=0; i<Nc*Nc*2; i++) out[i] = in[i];
      }
      __device__ __host__ inline int Length() { return Nc*Nc*2; }
    };

  /** No reconstruction but we scale the result. This is used for
      half-precision non-unitary fields, e.g., staggered fat link */
  template <typename Float, int Nc>
    struct ReconstructScale {
      Float scale;
      ReconstructScale(Float scale) : scale(scale) { ; }

      __device__ __host__ inline void pack(Float out[Nc*Nc*2], Float in[Nc*Nc*2]) const {
	for (int i=0; i<Nc*Nc*2; i++) out[i] = in[i] / scale;
      }
      __device__ __host__ inline void unpack(Float out[Nc*Nc*2], Float in[Nc*Nc*2]) const {
	for (int i=0; i<Nc*Nc*2; i++) out[i] = scale * in[i];
      }
      __device__ __host__ inline int Length() { return Nc*Nc*2; }
    };

  template <typename Float, int Nc>
    struct Reconstruct12 {
      int X[QUDA_MAX_DIM];
      const Float anisotropy;
      const QudaTboundary tBoundary;

      Reconstruct12(int X_[QUDA_MAX_DIM], Float anisotropy) 
      : anisotropy(anisotropy), tBoundary(tBoundary) {
	for (int i=0; i<QUDA_MAX_DIM; i++) X[i] = X_[i];
      }

      __device__ __host__ inline void Pack(Float out[Nc*(Nc-1)*2], Float in[Nc*Nc*2]) const {
	for (int i=0; i<Nc*(Nc-1)*2; i++) out[i] = in[i];
      }

      __device__ __host__ inline void Unpack(Float out[Nc*Nc*2], Float in[Nc*(Nc-1)*2],
					     int idx, int dir) const {
	for (int i=0; i<12; i++) out[i] = in[i];
	for (int i=12; i<18; i++) out[i] = 0.0;
	accumulateConjugateProduct(&out[12], &out[2], &out[10], +1);
	accumulateConjugateProduct(&out[12], &out[4], &out[8], -1);
	accumulateConjugateProduct(&out[14], &out[4], &out[6], +1);
	accumulateConjugateProduct(&out[14], &out[0], &out[10], -1);
	accumulateConjugateProduct(&out[16], &out[0], &out[8], +1);
	accumulateConjugateProduct(&out[16], &out[2], &out[6], -1);
	Float u0 = (dir < 3 ? anisotropy :
		    (idx >= (X[3]-1)*X[0]*X[1]*X[2]/2 ? tBoundary : 1));
	for (int i=12; i<18; i++) out[i]*=u0;
      }

      __device__ __host__ inline int Length() { return Nc*(Nc-1)*2; }
    };


  template <typename Float, int Nc>
    struct Reconstruct8 {
      int X[QUDA_MAX_DIM];
      const Float anisotropy;
      const QudaTboundary tBoundary;

      Reconstruct8(int X_[QUDA_MAX_DIM], Float anisotropy) 
      : anisotropy(anisotropy), tBoundary(tBoundary) {
	for (int i=0; i<QUDA_MAX_DIM; i++) X[i] = X_[i];
      }

      __device__ __host__ inline void Pack(Float out[8], Float in[Nc*Nc*2]) const {
	out[0] = atan2(in[1], in[0]);
	out[1] = atan2(in[13], in[12]);
	for (int i=2; i<8; i++) out[i] = in[i];
      }

      __device__ __host__ inline void Unpack(Float out[Nc*Nc*2], Float in[8],
					     int idx, int dir) const {
	// First reconstruct first row
	Float row_sum = 0.0;
	for (int i=2; i<6; i++) {
	  out[i] = in[i];
	  row_sum += in[i]*in[i];
	}

	Float u0 = (dir < 3 ? anisotropy :
		    (idx >= (X[3]-1)*X[0]*X[1]*X[2]/2 ? tBoundary : 1));
	Float diff = 1.0/(u0*u0) - row_sum;
	Float U00_mag = sqrt(diff >= 0 ? diff : 0.0);
    
	out[0] = U00_mag * cos(in[0]);
	out[1] = U00_mag * sin(in[0]);
	  
	// Now reconstruct first column
	Float column_sum = 0.0;
	for (int i=0; i<2; i++) column_sum += out[i]*out[i];
	for (int i=6; i<8; i++) {
	  out[i] = in[i];
	  column_sum += in[i]*in[i];
	}
	diff = 1.f/(u0*u0) - column_sum;
	Float U20_mag = sqrt(diff >= 0 ? diff : 0.0);
	
	out[12] = U20_mag * cos(in[1]);
	out[13] = U20_mag * sin(in[1]);
	// First column now restored
	
	// finally reconstruct last elements from SU(2) rotation
	Float r_inv2 = 1.0/(u0*row_sum);
	
	// U11
	Float A[2];
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

      __device__ __host__ inline int Length() { return 8; }
    };

  // Reconstruct10 is a dummy reconstruct type for momentum fields
  template <typename Float>
    struct Reconstruct10 {
      __device__ __host__ inline void pack(Float out[10], Float in[10]) const {
	for (int i=0; i<10; i++) out[i] = in[i];
      }
      __device__ __host__ inline void unpack(Float out[10], Float in[10]) const {
	for (int i=0; i<10; i++) out[i] = in[i];
      }
      __device__ __host__ inline int Length() { return 10; }
    };

  template <typename Float, int Nc, int N, typename Reconstruct>
    struct FloatNOrder {
      Reconstruct reconstruct;
      Float *gauge[2];
      int volumeCB;
      int stride;

    FloatNOrder(Reconstruct reconstruct, Float *gaugeEven, Float *gaugeOdd, int volume, int stride) 
    : reconstruct(reconstruct), volumeCB(volume/2), stride(stride) 
      { gauge[0] = gaugeEven; gauge[1] = gaugeOdd; }
      virtual ~FloatNOrder() { ; }
  
      __device__ __host__ inline void load(Float v[Nc*Nc*2], int x, int dir, int parity) const {
	const int M = reconstruct.Length() / N;
	Float tmp[reconstruct.Length()];
	for (int i=0; i<M; i++) {
	  for (int j=0; j<N; j++) {
	    int intIdx = i*N + j; // internal dof index
	    int padIdx = intIdx / N;
	    if (sizeof(Float)==sizeof(short)) {
	      tmp[i*N+j] = 
		short2float(gauge[parity][dir*volumeCB*M*N + (padIdx*stride + x)*N + intIdx%N]);
	    } else {
	      tmp[i*N+j] = 
		gauge[parity][dir*volumeCB*M*N + (padIdx*stride + x)*N + intIdx%N];
	    }
	  }
	}
	reconstruct.Unpack(v, tmp, x, dir);
      }
  
      __device__ __host__ inline void save(const Float v[Nc*Nc*2], int x, int dir, int parity) {
	const int M = reconstruct.Length() / N;
	Float tmp[reconstruct.Length()];
	reconstruct.Pack(tmp, v);
	for (int i=0; i<M; i++) {
	  for (int j=0; j<N; j++) {
	    int intIdx = i*N + j;
	    int padIdx = intIdx / N;
	    if (sizeof(Float)==sizeof(short)) {
	      gauge[parity][dir*volumeCB*M*N + (padIdx*stride + x)*N + intIdx%N] = 
		float2short(tmp[i*Nc+j]);
	    } else {
	      gauge[parity][dir*volumeCB*M*N + (padIdx*stride + x)*N + intIdx%N] = 
		tmp[i*Nc+j];
	    }
	  }
	}
      }

      size_t Bytes() const { return 4 * 2 * volumeCB * reconstruct.Length() * sizeof(Float); }
    };

  /**
     struct to define QDP ordered gauge fields: 
     [[dim]] [[parity][volumecb][row][col]]
   */
  template <typename Float, int Nc> struct QDPOrder {
    Float *gauge[QUDA_MAX_DIM];
    int volumeCB;
    int stride;
    QDPOrder(Float **gauge_, int volume, int stride) : volumeCB(volume/2), stride(stride) 
    { for (int i=0; i<4; i++) gauge[i] = gauge_[i]; }
    virtual ~QDPOrder() { ; }
    
    __device__ __host__ inline void load(Float v[Nc*Nc*2], int x, int dir, int parity) const {
      for (int i=0; i<2*Nc*Nc; i++) {
	v[i] = gauge[dir][(parity*volumeCB + x)*Nc*Nc*2 + i];
      }
    }
  
    __device__ __host__ inline void save(const Float v[Nc*Nc*2], int x, int dir, int parity) {
      for (int i=0; i<2*Nc*Nc; i++) {
	gauge[dir][(parity*volumeCB + x)*Nc*Nc*2 + i] = v[i];
      }
    }

    size_t Bytes() const { return 4 * 2 * volumeCB * Nc * Nc * 2 * sizeof(Float); }
  };

  /**
     struct to define MILC ordered gauge fields: 
     [parity][dim][volumecb][row][col]
   */
  template <typename Float, int Nc> struct MILCOrder {
    Float *gauge;
    int volumeCB;
    int stride;
    MILCOrder(Float *gauge, int volume, int stride) 
    : gauge(gauge), volumeCB(volume/2), stride(stride) { ; }
    virtual ~MILCOrder() { ; }
    
    __device__ __host__ inline void load(Float v[Nc*Nc*2], int x, int dir, int parity) const {
      for (int i=0; i<2*Nc*Nc; i++) {
	v[i] = gauge[((parity*4+dir)*volumeCB + x)*Nc*Nc*2 + i];
      }
    }
  
    __device__ __host__ inline void save(const Float v[Nc*Nc*2], int x, int dir, int parity) {
      for (int i=0; i<2*Nc*Nc; i++) {
	gauge[((parity*4+dir)*volumeCB + x)*Nc*Nc*2 + i] = v[i];
      }
    }

    size_t Bytes() const { return 4 * 2 * volumeCB * Nc * Nc * 2 * sizeof(Float); }
  };
  
  /**
     struct to define CPS ordered gauge fields: 
     [parity][dim][volumecb][col][row]
   */
  template <typename Float, int Nc> struct CPSOrder {
    Float *gauge;
    int volumeCB;
    int stride;
    Float anisotropy;
  CPSOrder(Float *gauge, int volume, int stride, Float anisotropy) 
    : gauge(gauge), volumeCB(volume/2), stride(stride), anisotropy(anisotropy) { ; }
    virtual ~CPSOrder() { ; }
    
    // we need to transpose and scale for CPS ordering
    __device__ __host__ inline void load(Float v[Nc*Nc*2], int x, int dir, int parity) const {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    v[(i*Nc+j)*2+z] = 
	      gauge[((((parity*4+dir)*volumeCB + x)*Nc + j)*Nc + i)*2 + z] / anisotropy;
	  }
	}
      }
    }
  
    __device__ __host__ inline void save(const Float v[Nc*Nc*2], int x, int dir, int parity) {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    gauge[((((parity*4+dir)*volumeCB + x)*Nc + j)*Nc + i)*2 + z] = 
	      anisotropy * v[(i*Nc+j)*2+z];
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
  template <typename Float, int Nc> struct BQCDOrder {
    Float *gauge;
    int volumeCB;
    int exVolumeCB; // extended checkerboard volume
    int stride;
  BQCDOrder(Float *gauge, int volume, int stride, int X[]) 
    : gauge(gauge), volumeCB(volume/2), stride(stride) { 
      // compute volumeCB + halo region
      exVolumeCB = X[0]/2 + 2;
      for (int i=0; i<4; i++) exVolumeCB *= X[i] + 2; 
    }
    virtual ~BQCDOrder() { ; }
    
    // we need to transpose for BQCD ordering
    __device__ __host__ inline void load(Float v[Nc*Nc*2], int x, int dir, int parity) const {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    v[(i*Nc+j)*2+z] = gauge[((((dir*2+parity)*exVolumeCB + x)*Nc + j)*Nc + i)*2 + z];
	  }
	}
      }
    }
  
    __device__ __host__ inline void save(const Float v[Nc*Nc*2], int x, int dir, int parity) {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    gauge[((((dir*2+parity)*exVolumeCB + x)*Nc + j)*Nc + i)*2 + z] = v[(i*Nc+j)*2+z];
	  }
	}
      }
    }

    size_t Bytes() const { return 4 * 2 * volumeCB * Nc * Nc * 2 * sizeof(Float); }
  };

  /**
     Generic CPU gauge reordering and packing 
  */
  template <typename FloatOut, typename FloatIn, int Nc, typename OutOrder, typename InOrder>
    void packGauge(OutOrder &outOrder, const InOrder &inOrder, int volume, int nDim) {  

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<nDim; d++) {
	for (int x=0; x<volume/2; x++) {
	  FloatIn in[Nc*Nc*2];
	  FloatOut out[Nc*Nc*2];
	  inOrder.load(in, x, d, parity);
	  for (int i=0; i<Nc*Nc*2; i++) out[i] = in[i];
	  outOrder.save(out, x, d, parity);
	}
      }

    }
  }

  /** 
      Generic CUDA gauge reordering and packing.  Adopts a similar form as
      the CPU version, using the same inlined functions.
  */
  template <typename FloatOut, typename FloatIn, int Nc, typename OutOrder, typename InOrder>
    __global__ void packGaugeKernel(OutOrder outOrder, const InOrder inOrder, int volume, int nDim) {  

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<nDim; d++) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= volume/2) return;

	FloatIn in[Nc*Nc*2];
	FloatOut out[Nc*Nc*2];
	inOrder.load(in, x, d, parity);
	for (int i=0; i<Nc*Nc*2; i++) out[i] = in[i];
	outOrder.save(out, x, d, parity);
      }
    }
  }

  template <typename FloatOut, typename FloatIn, int Nc, typename OutOrder, typename InOrder>
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
      packGaugeKernel<FloatOut, FloatIn, Nc, OutOrder, InOrder> 
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

}
