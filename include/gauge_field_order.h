#include <tune_quda.h>
#include <assert.h>
#include <register_traits.h>

namespace quda {

  // a += b*c
  template <typename Float>
    __device__ __host__ inline void accumulateComplexProduct(Float *a, const Float *b, const Float *c, Float sign) {
    a[0] += sign*(b[0]*c[0] - b[1]*c[1]);
    a[1] += sign*(b[0]*c[1] + b[1]*c[0]);
  }

  // a = b*c
  template <typename Float>
    __device__ __host__ inline void complexProduct(Float *a, const Float *b, const Float *c) {
    a[0] = b[0]*c[0] - b[1]*c[1];
    a[1] = b[0]*c[1] + b[1]*c[0];
  }

  // a = conj(b)*c
  template <typename Float>
    __device__ __host__ inline void complexDotProduct(Float *a, const Float *b, const Float *c) {
    a[0] = b[0]*c[0] + b[1]*c[1];
    a[1] = b[0]*c[1] - b[1]*c[0];
  }


  // a = b/c
  template <typename Float> 
    __device__ __host__ inline void complexQuotient(Float *a, const Float *b, const Float *c){
    complexDotProduct(a, c, b);
    Float denom = c[0]*c[0] + c[1]*c[1];
    a[0] /= denom;
    a[1] /= denom;
  }

  // a += conj(b) * conj(c)
  template <typename Float>
    __device__ __host__ inline void accumulateConjugateProduct(Float *a, const Float *b, const Float *c, int sign) {
    a[0] += sign * (b[0]*c[0] - b[1]*c[1]);
    a[1] -= sign * (b[0]*c[1] + b[1]*c[0]);
  }

  // a = conj(b)*conj(c)
  template <typename Float>
    __device__ __host__ inline void complexConjugateProduct(Float *a, const Float *b, const Float *c) {
    a[0] = b[0]*c[0] - b[1]*c[1];
    a[1] = -b[0]*c[1] - b[1]*c[0];
  }

  /** Generic reconstruction is no reconstruction */
  template <int N, typename Float> 
    struct Reconstruct {
      typedef typename mapper<Float>::type RegType;
      Reconstruct(const GaugeField &u) { ; }

      __device__ __host__ inline void Pack(RegType out[N], const RegType in[N], int idx ) const {
        for (int i=0; i<N; i++) out[i] = in[i];
      }
      __device__ __host__ inline void Unpack(RegType out[N], const RegType in[N], int idx, int dir, const RegType phase) const {
        for (int i=0; i<N; i++) out[i] = in[i];
      }


      __device__ __host__ inline void getPhase(RegType *phase, const RegType in[18]) const {
	*phase = 0;
      }

    };

  /** No reconstruction but we scale the result. This is used for
      half-precision non-unitary fields, e.g., staggered fat link */
  template <typename Float>
    struct Reconstruct<19,Float> {
    typedef typename mapper<Float>::type RegType;
    RegType scale;
  Reconstruct(const GaugeField &u) : scale(u.LinkMax()) { ; }

    __device__ __host__ inline void Pack(RegType out[18], const RegType in[18], int idx) const {
      for (int i=0; i<18; i++) out[i] = in[i] / scale;
    }
    __device__ __host__ inline void Unpack(RegType out[18], const RegType in[18],
					   int idx, int dir, const RegType phase) const {
      for (int i=0; i<18; i++) out[i] = scale * in[i];
    }

    __device__ __host__ inline void getPhase(RegType* phase, const RegType in[18]) const { *phase=0; return; }
  };

  template <typename Float>
    __device__ __host__ inline Float timeBoundary(int idx, const int X[QUDA_MAX_DIM], QudaTboundary tBoundary,
						  bool isFirstTimeSlice, bool isLastTimeSlice) {
    if ( idx >= X[3]*X[2]*X[1]*X[0]/2 ) { // halo region on the first time slice
      return isFirstTimeSlice ? tBoundary : 1.0;
    } else if ( idx >= (X[3]-1)*X[0]*X[1]*X[2]/2 ) { // last link on the last time slice
      return isLastTimeSlice ? tBoundary : 1.0;
    } else {
      return 1.0;
    }
  }

  template <typename Float>
    struct Reconstruct<12,Float> {
    typedef typename mapper<Float>::type RegType;
    int X[QUDA_MAX_DIM];
    const RegType anisotropy;
    const QudaTboundary tBoundary;
    bool isFirstTimeSlice;
    bool isLastTimeSlice;

  Reconstruct(const GaugeField &u) : anisotropy(u.Anisotropy()), tBoundary(u.TBoundary()),
      isFirstTimeSlice(comm_coord(3) == 0 ?true : false),
      isLastTimeSlice(comm_coord(3) == comm_dim(3)-1 ? true : false) 
	{ for (int i=0; i<QUDA_MAX_DIM; i++) X[i] = u.X()[i]; }

    __device__ __host__ inline void Pack(RegType out[12], const RegType in[18], int idx) const {
      for (int i=0; i<12; i++) out[i] = in[i];
    }

    __device__ __host__ inline void Unpack(RegType out[18], const RegType in[12],
					   int idx, int dir, const RegType phase) const {
      for (int i=0; i<12; i++) out[i] = in[i];
      for (int i=12; i<18; i++) out[i] = 0.0;
      accumulateConjugateProduct(&out[12], &out[2], &out[10], +1);
      accumulateConjugateProduct(&out[12], &out[4], &out[8], -1);
      accumulateConjugateProduct(&out[14], &out[4], &out[6], +1);
      accumulateConjugateProduct(&out[14], &out[0], &out[10], -1);
      accumulateConjugateProduct(&out[16], &out[0], &out[8], +1);
      accumulateConjugateProduct(&out[16], &out[2], &out[6], -1);

      RegType u0 = dir < 3 ? anisotropy : 
	timeBoundary<RegType>(idx, X, tBoundary,isFirstTimeSlice, isLastTimeSlice);

      for (int i=12; i<18; i++) out[i]*=u0;
    }

    __device__ __host__ inline void getPhase(RegType* phase, const RegType in[18]){ *phase=0; return; }

  };


  // FIX ME - 11 is a misnomer to avoid confusion in template instantiation
  template <typename Float>
    struct Reconstruct<11,Float> {
    typedef typename mapper<Float>::type RegType;

    Reconstruct(const GaugeField &u) { ; }

    __device__ __host__ inline void Pack(RegType out[10], const RegType in[18], int idx) const {
      for (int i=0; i<4; i++) out[i] = in[i+2];
      out[4] = in[10];
      out[5] = in[11];
      out[6] = in[1];
      out[7] = in[9];
      out[8] = in[17];
      out[9] = 0.0;
    }

    __device__ __host__ inline void Unpack(RegType out[18], const RegType in[10],
					   int idx, int dir, const RegType phase) const {
      out[0] = 0.0;
      out[1] = in[6];
      for (int i=0; i<4; i++) out[i+2] = in[i];
      out[6] = -out[2];
      out[7] =  out[3];
      out[8] = 0.0;
      out[9] = in[7];
      out[10] = in[4];
      out[11] = in[5];
      out[12] = -out[4];
      out[13] =  out[5];
      out[14] = -out[10];
      out[15] =  out[11];
      out[16] = 0.0;
      out[17] = in[8];
    }

    __device__ __host__ inline void getPhase(RegType* phase, const RegType in[18])
    { *phase=0; return; }

  };

  template <typename Float>
    struct Reconstruct<13,Float> {
    typedef typename mapper<Float>::type RegType;
    const Reconstruct<12,Float> reconstruct_12;
    const RegType scale; 

  Reconstruct(const GaugeField &u) : reconstruct_12(u), scale(u.Scale()) {}

    __device__ __host__ inline void Pack(RegType out[12], const RegType in[18], int idx) const {
      reconstruct_12.Pack(out, in, idx); 
    }

    __device__ __host__ inline void Unpack(RegType out[18], const RegType in[12], int idx, int dir, const RegType phase) const {
      for(int i=0; i<12; ++i) out[i] = in[i];
      for(int i=12; i<18; ++i) out[i] = 0.0;

      const RegType coeff = 1./scale;

      accumulateConjugateProduct(&out[12], &out[2], &out[10], +coeff);
      accumulateConjugateProduct(&out[12], &out[4], &out[8], -coeff);
      accumulateConjugateProduct(&out[14], &out[4], &out[6], +coeff);
      accumulateConjugateProduct(&out[14], &out[0], &out[10], -coeff);
      accumulateConjugateProduct(&out[16], &out[0], &out[8], +coeff);
      accumulateConjugateProduct(&out[16], &out[2], &out[6], -coeff);

      // Multiply the third row by exp(I*3*phase)
      RegType cos_sin[2];
      Trig<isHalf<RegType>::value>::SinCos(static_cast<RegType>(3.*phase), &cos_sin[1], &cos_sin[0]);     
      RegType tmp[2];
      complexProduct(tmp, cos_sin, &out[12]); out[12] = tmp[0]; out[13] = tmp[1];
      complexProduct(tmp, cos_sin, &out[14]); out[14] = tmp[0]; out[15] = tmp[1];
      complexProduct(tmp, cos_sin, &out[16]); out[16] = tmp[0]; out[17] = tmp[1];
    }

    __device__ __host__ inline void getPhase(RegType *phase, const RegType in[18]) const {
      RegType denom[2];
      // denominator = (U[0][0]*U[1][1] - U[0][1]*U[1][0])*
      complexProduct(denom, in, in+8);
      accumulateComplexProduct(denom, in+2, in+6, static_cast<RegType>(-1.0));

      denom[0] /= scale;
      denom[1] /= (-scale); // complex conjugate

      RegType expI3Phase[2];
      // numerator = U[2][2]
      complexQuotient(expI3Phase, in+16, denom);

      *phase = Trig<isHalf<RegType>::value>::Atan2(expI3Phase[1], expI3Phase[0])/3.;
      return;
    }

  };


  template <typename Float>
    struct Reconstruct<8,Float> {
    typedef typename mapper<Float>::type RegType;
    int X[QUDA_MAX_DIM];
    const RegType anisotropy;
    const QudaTboundary tBoundary;
    bool isFirstTimeSlice;
    bool isLastTimeSlice;

  Reconstruct(const GaugeField &u) : anisotropy(u.Anisotropy()), tBoundary(u.TBoundary()),
      isFirstTimeSlice(comm_coord(3) == 0 ? true : false),
      isLastTimeSlice(comm_coord(3) == comm_dim(3)-1 ? true : false) 
	{ for (int i=0; i<QUDA_MAX_DIM; i++) X[i] = u.X()[i]; }

    __device__ __host__ inline void Pack(RegType out[8], const RegType in[18], int idx) const {
      out[0] = Trig<isHalf<Float>::value>::Atan2(in[1], in[0]);
      out[1] = Trig<isHalf<Float>::value>::Atan2(in[13], in[12]);
      for (int i=2; i<8; i++) out[i] = in[i];
    }

    __device__ __host__ inline void Unpack(RegType out[18], const RegType in[8],
					   int idx, int dir, const RegType phase) const {
      // First reconstruct first row
      RegType row_sum = 0.0;
      for (int i=2; i<6; i++) {
	out[i] = in[i];
	row_sum += in[i]*in[i];
      }

      RegType u0 = dir < 3 ? anisotropy : 
	timeBoundary<RegType>(idx, X, tBoundary,isFirstTimeSlice, isLastTimeSlice);

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

    __device__ __host__ inline void getPhase(RegType* phase, const RegType in[18]){ *phase=0; return; }
  };


  template <typename Float>
    struct Reconstruct<9,Float> {
    typedef typename mapper<Float>::type RegType;
    const Reconstruct<8,Float> reconstruct_8;
    const RegType scale;

  Reconstruct(const GaugeField &u) : reconstruct_8(u), scale(u.Scale()) {}

    __device__ __host__ inline void getPhase(RegType *phase, const RegType in[18]) const {

      RegType denom[2];
      // denominator = (U[0][0]*U[1][1] - U[0][1]*U[1][0])*
      complexProduct(denom, in, in+8);
      accumulateComplexProduct(denom, in+2, in+6, static_cast<RegType>(-1.0));

      denom[0] /= scale;
      denom[1] /= (-scale); // complex conjugate

      RegType expI3Phase[2];
      // numerator = U[2][2]
      complexQuotient(expI3Phase, in+16, denom);

      *phase = Trig<isHalf<RegType>::value>::Atan2(expI3Phase[1], expI3Phase[0])/3.;
    }


    __device__ __host__ inline void Pack(RegType out[8], const RegType in[18], int idx) const {

      RegType phase;
      getPhase(&phase,in);
      RegType cos_sin[2];
      sincos(-phase, &cos_sin[1], &cos_sin[0]);      
      // Rescale the U3 input matrix by exp(-I*phase) to obtain an SU3 matrix multiplied by a real scale factor, 
      // which the macros in read_gauge.h can handle.
      // NB: Only 5 complex matrix elements are used in the reconstruct 8 packing routine, 
      // so only need to rescale those elements.
      RegType su3[18];
      for(int i=0; i<4; ++i){ 
	complexProduct(su3 + 2*i, cos_sin, in + 2*i);
      }
      complexProduct(&su3[12], cos_sin, &in[12]);
      reconstruct_8.Pack(out, su3, idx); 
    }

    __device__ __host__ inline void Unpack(RegType out[18], const RegType in[8], int idx, int dir, const RegType phase) const {
      reconstruct_8.Unpack(out, in, idx, dir, phase);
      RegType cos_sin[2];
      Trig<isHalf<RegType>::value>::SinCos(phase, &cos_sin[1], &cos_sin[0]);     
      RegType tmp[2];
      cos_sin[0] *= scale;
      cos_sin[1] *= scale;

      // rescale the matrix by exp(I*phase)*scale
      complexProduct(tmp, cos_sin, &out[0]);  out[0] = tmp[0]; out[1] = tmp[1];
      complexProduct(tmp, cos_sin, &out[2]);  out[2] = tmp[0]; out[3] = tmp[1];
      complexProduct(tmp, cos_sin, &out[4]);  out[4] = tmp[0]; out[5] = tmp[1];
      complexProduct(tmp, cos_sin, &out[6]);  out[6] = tmp[0]; out[7] = tmp[1];
      complexProduct(tmp, cos_sin, &out[8]);  out[8] = tmp[0]; out[9] = tmp[1];
      complexProduct(tmp, cos_sin, &out[10]); out[10] = tmp[0]; out[11] = tmp[1];
      complexProduct(tmp, cos_sin, &out[12]); out[12] = tmp[0]; out[13] = tmp[1];
      complexProduct(tmp, cos_sin, &out[14]); out[14] = tmp[0]; out[15] = tmp[1];
      complexProduct(tmp, cos_sin, &out[16]); out[16] = tmp[0]; out[17] = tmp[1];
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
      const int geometry;
#if __COMPUTE_CAPABILITY__ >= 200
      const int hasPhase; 
      const size_t phaseOffset;
#endif

    FloatNOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) : 
      reconstruct(u), volumeCB(u.VolumeCB()), stride(u.Stride()), geometry(u.Geometry())
#if __COMPUTE_CAPABILITY__ >= 200
	, hasPhase((u.Reconstruct() == QUDA_RECONSTRUCT_9 || u.Reconstruct() == QUDA_RECONSTRUCT_13) ? 1 : 0), 
	phaseOffset(u.PhaseOffset())
#endif
      {
	if (gauge_) { gauge[0] = gauge_; gauge[1] = (Float*)((char*)gauge_ + u.Bytes()/2);
	} else { gauge[0] = (Float*)u.Gauge_p(); gauge[1] = (Float*)((char*)u.Gauge_p() + u.Bytes()/2);	}

	for (int i=0; i<4; i++) {
	  ghost[i] = ghost_ ? ghost_[i] : 0; 
	  faceVolumeCB[i] = u.SurfaceCB(i)*u.Nface(); // face volume equals surface * depth	  
	}
      }


    FloatNOrder(const FloatNOrder &order) 
    : reconstruct(order.reconstruct), volumeCB(order.volumeCB), stride(order.stride), 
	geometry(order.geometry) 
#if __COMPUTE_CAPABILITY__ >= 200
	, hasPhase(order.hasPhase), phaseOffset(order.phaseOffset) 
#endif
      {
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
        RegType phase = 0.;
#if __COMPUTE_CAPABILITY__ >= 200
        if(hasPhase) copy(phase, gauge[parity][phaseOffset/sizeof(Float) + stride*dir + x]);
        // The phases come after the ghost matrices
#endif
        reconstruct.Unpack(v, tmp, x, dir, 2.*M_PI*phase);
      }

      __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {
        const int M = reconLen / N;
        RegType tmp[reconLen];
        reconstruct.Pack(tmp, v, x);
        for (int i=0; i<M; i++) {
          for (int j=0; j<N; j++) {
            int intIdx = i*N + j;
            int padIdx = intIdx / N;
            copy(gauge[parity][dir*stride*M*N + (padIdx*stride + x)*N + intIdx%N], tmp[i*N+j]);
          }
        }
#if __COMPUTE_CAPABILITY__ >= 200
        if(hasPhase){
          RegType phase;
          reconstruct.getPhase(&phase,v);
          copy(gauge[parity][phaseOffset/sizeof(Float) + dir*stride + x], static_cast<RegType>(phase/(2.*M_PI))); 
        }        
#endif
      }

      __device__ __host__ inline void loadGhost(RegType v[length], int x, int dir, int parity) const {
        if (!ghost[dir]) { // load from main field not separate array
          load(v, volumeCB+x, dir, parity); // an offset of size volumeCB puts us at the padded region
          // This also works perfectly when phases are stored. No need to change this.
        } else {
          const int M = reconLen / N;
          RegType tmp[reconLen];
          for (int i=0; i<M; i++) {
            for (int j=0; j<N; j++) {
              int intIdx = i*N + j; // internal dof index
              int padIdx = intIdx / N;
#if __COMPUTE_CAPABILITY__ < 200
	      const int hasPhase = 0;
#endif
              copy(tmp[i*N+j], ghost[dir][parity*faceVolumeCB[dir]*(M*N + hasPhase) + (padIdx*faceVolumeCB[dir]+x)*N + intIdx%N]);
            }
          }
          RegType phase=0.; 
#if __COMPUTE_CAPABILITY__ >= 200
          if(hasPhase) copy(phase, ghost[dir][parity*faceVolumeCB[dir]*(M*N + 1) + faceVolumeCB[dir]*M*N + x]); 
#endif
          reconstruct.Unpack(v, tmp, x, dir, 2.*M_PI*phase);	 
        }
      }

      __device__ __host__ inline void saveGhost(const RegType v[length], int x, int dir, int parity) {
        if (!ghost[dir]) { // store in main field not separate array
	  save(v, volumeCB+x, dir, parity); // an offset of size volumeCB puts us at the padded region
        } else {
          const int M = reconLen / N;
          RegType tmp[reconLen];
          reconstruct.Pack(tmp, v, x);
          for (int i=0; i<M; i++) {
            for (int j=0; j<N; j++) {
              int intIdx = i*N + j;
              int padIdx = intIdx / N;
#if __COMPUTE_CAPABILITY__ < 200
	      const int hasPhase = 0;
#endif
              copy(ghost[dir][parity*faceVolumeCB[dir]*(M*N + hasPhase) + (padIdx*faceVolumeCB[dir]+x)*N + intIdx%N], tmp[i*N+j]);
            }
          }

#if __COMPUTE_CAPABILITY__ >= 200
          if(hasPhase){
            RegType phase=0.;
            reconstruct.getPhase(&phase, v); 
            copy(ghost[dir][parity*faceVolumeCB[dir]*(M*N + 1) + faceVolumeCB[dir]*M*N + x], static_cast<RegType>(phase/(2.*M_PI)));
          }
#endif
        }
      }

      __device__ __host__ inline void loadGhostEx(RegType v[length], int x, int dir, 
						  int dim, int g, int parity, const int R[]) const {
	const int M = reconLen / N;
	RegType tmp[reconLen];
	for (int i=0; i<M; i++) {
	  for (int j=0; j<N; j++) {
	    int intIdx = i*N + j; // internal dof index
	    int padIdx = intIdx / N;
	    copy(tmp[i*N+j], ghost[dim][((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + hasPhase) 
					+ (padIdx*R[dim]*faceVolumeCB[dim]+x)*N + intIdx%N]);
	  }
	}
	RegType phase=0.; 
	if(hasPhase) copy(phase, ghost[dim][((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + 1) 
					    + R[dim]*faceVolumeCB[dim]*M*N + x]); 

	// this will fail for non-18 reconstruct since it's not doing boundary checking correctly
	reconstruct.Unpack(v, tmp, x, g, 2.*M_PI*phase); 
      }

      __device__ __host__ inline void saveGhostEx(const RegType v[length], int x, 
						  int dir, int dim, int g, int parity, const int R[]) {
	const int M = reconLen / N;
	RegType tmp[reconLen];
	reconstruct.Pack(tmp, v, x);
	for (int i=0; i<M; i++) {
	  for (int j=0; j<N; j++) {
	    int intIdx = i*N + j;
	    int padIdx = intIdx / N;
	    copy(ghost[dim][((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + hasPhase) 
			    + (padIdx*R[dim]*faceVolumeCB[dim]+x)*N + intIdx%N], tmp[i*N+j]);
	  }
	}
	if(hasPhase){
	  RegType phase=0.;
	  reconstruct.getPhase(&phase, v); 
	  copy(ghost[dim][((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + 1) + R[dim]*faceVolumeCB[dim]*M*N + x], 
	       static_cast<RegType>(phase/(2.*M_PI)));
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
      const int geometry;
      const int hasPhase;

      LegacyOrder(const GaugeField &u, Float **ghost_) 
      : volumeCB(u.VolumeCB()), stride(u.Stride()), geometry(u.Geometry()), hasPhase(0) {
	for (int i=0; i<4; i++) {
	  ghost[i] = (ghost_) ? ghost_[i] : (Float*)(u.Ghost()[i]);
	  faceVolumeCB[i] = u.SurfaceCB(i)*u.Nface(); // face volume equals surface * depth
	}
      }

      LegacyOrder(const LegacyOrder &order) 
      : volumeCB(order.volumeCB), stride(order.stride), geometry(order.geometry), hasPhase(0) {
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

      __device__ __host__ inline void loadGhostEx(RegType v[length], int x, int dir, 
						  int dim, int g, int parity, const int R[]) const {
	for (int i=0; i<length; i++) {
	  v[i] = ghost[dim]
	    [(((dir*2+parity)*R[dim]*faceVolumeCB[dim] + x)*geometry+g)*length + i];
	}
      }

      __device__ __host__ inline void saveGhostEx(const RegType v[length], int x, 
						  int dir, int dim, int g, int parity, const int R[]) {
        for (int i=0; i<length; i++) {
	  ghost[dim]
	    [(((dir*2+parity)*R[dim]*faceVolumeCB[dim] + x)*geometry+g)*length + i] = v[i];
	}
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
     struct to define QDPJIT ordered gauge fields: 
     [[dim]] [[parity][complex][row][col][volumecb]]
  */
  template <typename Float, int length> struct QDPJITOrder : public LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge[QUDA_MAX_DIM];
    const int volumeCB;
  QDPJITOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) 
    : LegacyOrder<Float,length>(u, ghost_), volumeCB(u.VolumeCB())
      { for (int i=0; i<4; i++) gauge[i] = gauge_ ? ((Float**)gauge_)[i] : ((Float**)u.Gauge_p())[i]; }
  QDPJITOrder(const QDPJITOrder &order) : LegacyOrder<Float,length>(order), volumeCB(order.volumeCB) {
      for(int i=0; i<4; i++) gauge[i] = order.gauge[i];
    }
    virtual ~QDPJITOrder() { ; }

    __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity) const {
      for (int i=0; i<length; i++) {
	int z = i%2;
	int rolcol = i/2;
	v[i] = (RegType)gauge[dir][((z*(length/2) + rolcol)*2 + parity)*volumeCB + x];
      }
    }

    __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {
      for (int i=0; i<length; i++) {
	int z = i%2;
	int rolcol = i/2;
	gauge[dir][((z*(length/2) + rolcol)*2 + parity)*volumeCB + x] = (Float)v[i];
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
    const int &geometry;
  MILCOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) : 
    LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()), 
      volumeCB(u.VolumeCB()), geometry(LegacyOrder<Float,length>::geometry) { ; }
  MILCOrder(const MILCOrder &order) : LegacyOrder<Float,length>(order), 
      gauge(order.gauge), volumeCB(order.volumeCB), geometry(order.geometry)
      { ; }
    virtual ~MILCOrder() { ; }

    __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity) const {
      for (int i=0; i<length; i++) {
	v[i] = (RegType)gauge[((parity*volumeCB+x)*geometry + dir)*length + i];
      }
    }

    __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {
      for (int i=0; i<length; i++) {
	gauge[((parity*volumeCB+x)*geometry + dir)*length + i] = (Float)v[i];
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
    const int &geometry;
  CPSOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) 
    : LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()), 
      volumeCB(u.VolumeCB()), anisotropy(u.Anisotropy()), Nc(3), 
      geometry(LegacyOrder<Float,length>::geometry) 
      { if (length != 18) errorQuda("Gauge length %d not supported", length); }
  CPSOrder(const CPSOrder &order) : LegacyOrder<Float,length>(order), gauge(order.gauge), 
      volumeCB(order.volumeCB), anisotropy(order.anisotropy), Nc(3), geometry(order.geometry)
      { ; }
    virtual ~CPSOrder() { ; }

    // we need to transpose and scale for CPS ordering
    __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity) const {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    v[(i*Nc+j)*2+z] = 
	      (RegType)(gauge[((((parity*volumeCB+x)*geometry + dir)*Nc + j)*Nc + i)*2 + z] / anisotropy);
	  }
	}
      }
    }

    __device__ __host__ inline void save(const RegType v[18], int x, int dir, int parity) {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    gauge[((((parity*volumeCB+x)*geometry + dir)*Nc + j)*Nc + i)*2 + z] = 
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

  /**
     struct to define TIFR ordered gauge fields: 
     [mu][parity][volumecb][col][row]
  */
  template <typename Float, int length> struct TIFROrder : LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge;
    const int volumeCB;
    const int Nc;
    const Float scale;
  TIFROrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) 
    : LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()), 
      volumeCB(u.VolumeCB()), Nc(3), scale(u.Scale()) { 
      if (length != 18) errorQuda("Gauge length %d not supported", length);
    }
  TIFROrder(const TIFROrder &order) 
    : LegacyOrder<Float,length>(order), gauge(order.gauge), volumeCB(order.volumeCB), Nc(3), scale(order.scale) {       
      if (length != 18) errorQuda("Gauge length %d not supported", length);
    }

    virtual ~TIFROrder() { ; }

    // we need to transpose for TIFR ordering
    __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity) const {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    v[(i*Nc+j)*2+z] = (RegType)gauge[((((dir*2+parity)*volumeCB + x)*Nc + j)*Nc + i)*2 + z] / scale;
	  }
	}
      }
    }

    __device__ __host__ inline void save(const RegType v[18], int x, int dir, int parity) {
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    gauge[((((dir*2+parity)*volumeCB + x)*Nc + j)*Nc + i)*2 + z] = (Float)v[(i*Nc+j)*2+z] * scale;
	  }
	}
      }
    }

    size_t Bytes() const { return Nc * Nc * 2 * sizeof(Float); }
  };


}
