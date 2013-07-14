#include <tune_quda.h>
#include <assert.h>
#include <register_traits.h>

namespace quda {

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
      Float *ghost[4];
      int faceVolumeCB[QUDA_MAX_DIM];
      const int volumeCB;
      const int stride;
      
    FloatNOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) : 
      reconstruct(u), volumeCB(u.VolumeCB()), stride(u.Stride()) {
	if (gauge_) { gauge[0] = gauge_; gauge[1] = (Float*)((char*)gauge_ + u.Bytes()/2);
	} else { gauge[0] = (Float*)u.Gauge_p(); gauge[1] = (Float*)((char*)u.Gauge_p() + u.Bytes()/2);	}
	
	for (int i=0; i<4; i++) {
	  ghost[i] = ghost_ ? ghost_[i] : 0; 
	  faceVolumeCB[i] = u.SurfaceCB(i)*u.Nface(); // face volume equals surface * depth	  
	}
      }
      
    FloatNOrder(const FloatNOrder &order) 
    : reconstruct(order.reconstruct), volumeCB(order.volumeCB), stride(order.stride) {
      gauge[0] = order.gauge[0];
      gauge[1] = order.gauge[1];
      for (int i=0; i<4; i++) {
	ghost[i] = order.ghost[i];
	faceVolumeCB[i] = order.faceVolumeCB[i];
      }
    }

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
	if (!ghost[dir]) { // load from main field not separate array
	  load(v, volumeCB+x, dir, parity); // an offset of size volumeCB puts us at the padded region
	} else {
	  const int M = reconLen / N;
	  RegType tmp[reconLen];
	  for (int i=0; i<M; i++) {
	    for (int j=0; j<N; j++) {
	      int intIdx = i*N + j; // internal dof index
	      int padIdx = intIdx / N;
	      copy(tmp[i*N+j], ghost[dir][(parity*faceVolumeCB[dir]*M + padIdx*faceVolumeCB[dir]+x)*N + intIdx%N]);
	    }
	  }
	  reconstruct.Unpack(v, tmp, x, dir);	 
	}
      }
      
      __device__ __host__ inline void saveGhost(const RegType v[length], int x, int dir, int parity) {
	if (!ghost[dir]) { // store in main field not separate array
	  save(v, volumeCB+x, dir, parity); // an offset of size volumeCB puts us at the padded region
	} else {
	  const int M = reconLen / N;
	  RegType tmp[reconLen];
	  reconstruct.Pack(tmp, v);
	  for (int i=0; i<M; i++) {
	    for (int j=0; j<N; j++) {
	      int intIdx = i*N + j;
	      int padIdx = intIdx / N;
	      copy(ghost[dir][(parity*faceVolumeCB[dir]*M + padIdx*faceVolumeCB[dir]+x)*N + intIdx%N], tmp[i*N+j]);
	    }
	  }
	}
      }

      size_t Bytes() const { return reconLen * sizeof(Float); }
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
      const int volumeCB;
      const int stride;
    LegacyOrder(const GaugeField &u, Float **ghost_) : volumeCB(u.VolumeCB()), stride(u.Stride()) {
	for (int i=0; i<4; i++) {
	  ghost[i] = (ghost_) ? ghost_[i] : (Float*)(u.Ghost()[i]);
	  faceVolumeCB[i] = u.SurfaceCB(i)*u.Nface(); // face volume equals surface * depth
	}
      }
    LegacyOrder(const LegacyOrder &order) : volumeCB(order.volumeCB), stride(order.stride) {
      for (int i=0; i<4; i++) {
	ghost[i] = order.ghost[i];
	faceVolumeCB[i] = order.faceVolumeCB[i];
      }
    }
      virtual ~LegacyOrder() { ; }
      
      __device__ __host__ inline void loadGhost(RegType v[length], int x, int dir, int parity) const {
	for (int i=0; i<length; i++) v[i] = ghost[dir][(parity*faceVolumeCB[dir] + x)*length + i];
      }
      
      __device__ __host__ inline void saveGhost(const RegType v[length], int x, int dir, int parity) {
	for (int i=0; i<length; i++) ghost[dir][(parity*faceVolumeCB[dir] + x)*length + i] = v[i];
      }
    };

  /**
     struct to define QDP ordered gauge fields: 
     [[dim]] [[parity][volumecb][row][col]]
   */
    template <typename Float, int length> struct QDPOrder : public LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge[QUDA_MAX_DIM];
    const int volumeCB;
    QDPOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) 
      : LegacyOrder<Float,length>(u, ghost_), volumeCB(u.VolumeCB())
      { for (int i=0; i<4; i++) gauge[i] = gauge_ ? ((Float**)gauge_)[i] : ((Float**)u.Gauge_p())[i]; }
    QDPOrder(const QDPOrder &order) : LegacyOrder<Float,length>(order), volumeCB(order.volumeCB) {
      for(int i=0; i<4; i++) gauge[i] = order.gauge[i];
    }
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

    size_t Bytes() const { return length * sizeof(Float); }
  };

  /**
     struct to define MILC ordered gauge fields: 
     [parity][dim][volumecb][row][col]
   */
    template <typename Float, int length> struct MILCOrder : public LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge;
    const int volumeCB;
    MILCOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) : 
    LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()), volumeCB(u.VolumeCB()) { ; }
    MILCOrder(const MILCOrder &order) : LegacyOrder<Float,length>(order), gauge(order.gauge), volumeCB(order.volumeCB)
    { ; }
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

    size_t Bytes() const { return length * sizeof(Float); }
  };
  
  /**
     struct to define CPS ordered gauge fields: 
     [parity][dim][volumecb][col][row]
   */
    template <typename Float, int length> struct CPSOrder : LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge;
    const int volumeCB;
    const Float anisotropy;
    const int Nc;
    CPSOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) 
      : LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()), volumeCB(u.VolumeCB()), anisotropy(u.Anisotropy()), Nc(3) 
    { if (length != 18) errorQuda("Gauge length %d not supported", length); }
    CPSOrder(const CPSOrder &order) : LegacyOrder<Float,length>(order), gauge(order.gauge), 
      volumeCB(order.volumeCB), anisotropy(order.anisotropy), Nc(3)
    { ; }
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

    size_t Bytes() const { return Nc * Nc * 2 * sizeof(Float); }
  };
  
  /**
     struct to define BQCD ordered gauge fields: 
     [mu][parity][volumecb+halos][col][row]
   */
    template <typename Float, int length> struct BQCDOrder : LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge;
    const int volumeCB;
    int exVolumeCB; // extended checkerboard volume
    const int Nc;
    BQCDOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) 
      : LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()), volumeCB(u.VolumeCB()), Nc(3) { 
      if (length != 18) errorQuda("Gauge length %d not supported", length);
      // compute volumeCB + halo region
      exVolumeCB = u.X()[0]/2 + 2;
      for (int i=1; i<4; i++) exVolumeCB *= u.X()[i] + 2; 
    }
    BQCDOrder(const BQCDOrder &order) : LegacyOrder<Float,length>(order), gauge(order.gauge), 
      volumeCB(order.volumeCB), exVolumeCB(order.exVolumeCB), Nc(3) {       
      if (length != 18) errorQuda("Gauge length %d not supported", length);
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

    size_t Bytes() const { return Nc * Nc * 2 * sizeof(Float); }
  };

}

