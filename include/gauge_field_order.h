#ifndef _GAUGE_ORDER_H
#define _GAUGE_ORDER_H

/**
 * @file  gauge_field_order.h
 * @brief Main header file for host and device accessors to GaugeFields
 *
 */

#include <tune_quda.h>
#include <assert.h>
#include <register_traits.h>
#include <complex_quda.h>
#include <generics/ldg.h>

namespace quda {

  namespace gauge {


    template<typename Float, int nColor, QudaGaugeFieldOrder order> struct Accessor {
      mutable complex<Float> dummy;
      Accessor(const GaugeField &, void *gauge_=0, void **ghost_=0) { }
      __device__ __host__ complex<Float>& operator()(int d, int parity, int x, int row, int col) const {
#ifndef __CUDA_ARCH__
	errorQuda("Not implemented");
#endif
	return dummy;
      }
    };

    template<typename Float, int nColor, QudaGaugeFieldOrder order> struct GhostAccessor {
      mutable complex<Float> dummy;
      GhostAccessor(const GaugeField &, void *gauge_=0, void **ghost_=0) { }
      __device__ __host__ complex<Float>& operator()(int d, int parity, int x, int row, int col) const {
#ifndef __CUDA_ARCH__
	errorQuda("Not implemented");
#endif
	return dummy;
      }
    };

    template<typename Float, int nColor>
      struct Accessor<Float,nColor,QUDA_QDP_GAUGE_ORDER> {
      complex <Float> *u[QUDA_MAX_GEOMETRY];
      const int cb_offset;
    Accessor(const GaugeField &U, void *gauge_=0, void **ghost_=0)
      : cb_offset((U.Bytes()>>1) / (sizeof(complex<Float>)*U.Geometry())) {
	for (int d=0; d<U.Geometry(); d++)
	  u[d] = gauge_ ? static_cast<complex<Float>**>(gauge_)[d] :
	    static_cast<complex<Float>**>(const_cast<void*>(U.Gauge_p()))[d];
      }
    Accessor(const Accessor<Float,nColor,QUDA_QDP_GAUGE_ORDER> &a) : cb_offset(a.cb_offset) {
	for (int d=0; d<QUDA_MAX_GEOMETRY; d++)
	  u[d] = a.u[d];
      }
      __device__ __host__ inline complex<Float>& operator()(int d, int parity, int x, int row, int col) const
      { return u[d][ parity*cb_offset + (x*nColor + row)*nColor + col]; }
    };

    template<typename Float, int nColor>
      struct GhostAccessor<Float,nColor,QUDA_QDP_GAUGE_ORDER> {
      complex<Float> *ghost[4];
      int ghostOffset[4];
      GhostAccessor(const GaugeField &U, void *gauge_=0, void **ghost_=0) {
	for (int d=0; d<4; d++) {
	  ghost[d] = ghost_ ? static_cast<complex<Float>*>(ghost_[d]) :
	    static_cast<complex<Float>*>(const_cast<void*>(U.Ghost()[d]));
	  ghostOffset[d] = U.Nface()*U.SurfaceCB(d)*U.Ncolor()*U.Ncolor();
	}
      }
      GhostAccessor(const GhostAccessor<Float,nColor,QUDA_QDP_GAUGE_ORDER> &a) {
	for (int d=0; d<4; d++) {
	  ghost[d] = a.ghost[d];
	  ghostOffset[d] = a.ghostOffset[d];
	}
      }
      __device__ __host__ inline complex<Float>& operator()(int d, int parity, int x, int row, int col) const
      { return ghost[d][ parity*ghostOffset[d] + (x*nColor + row)*nColor + col]; }
    };

    template<int nColor, int N>
      __device__ __host__ inline int indexFloatN(int dim, int parity, int x_cb, int row, int col, int stride, int offset_cb) {
      constexpr int M = (2*nColor*nColor) / N;
      int j = ((row*nColor+col)*2) / N; // factor of two for complexity
      int i = ((row*nColor+col)*2) % N;
      int index = ((x_cb + dim*stride*M + j*stride)*2+i) / 2; // back to a complex offset
      index += parity*offset_cb;
      return index;
    };

    template<typename Float, int nColor>
      struct Accessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER> {
      complex<Float> *u;
      const int offset_cb;
      const int stride;
    Accessor(const GaugeField &U, void *gauge_=0, void **ghost_=0)
      : u(gauge_ ? static_cast<complex<Float>*>(gauge_) :
	  static_cast<complex<Float>*>(const_cast<void*>(U.Gauge_p()))),
	offset_cb( (U.Bytes()>>1) / sizeof(complex<Float>)), stride(U.Stride())
	{  }
    Accessor(const Accessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER> &a): u(a.u), offset_cb(a.offset_cb), stride(a.stride) {  }
      __device__ __host__ inline complex<Float>& operator()(int dim, int parity, int x_cb, int row, int col) const
      { return u[parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb]; }
    };

    template<typename Float, int nColor>
      struct GhostAccessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER> {
      const int volumeCB;
      Accessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER> accessor;
    GhostAccessor(const GaugeField &U, void *gauge_, void **ghost_=0)
      : volumeCB(U.VolumeCB()), accessor(U, gauge_, ghost_) {  }
    GhostAccessor(const GhostAccessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER> &a)
      : volumeCB(a.volumeCB), accessor(a.accessor) {  }
      __device__ __host__ inline complex<Float>& operator()(int d, int parity, int x_cb, int row, int col) const
      { return accessor(d, parity, x_cb+volumeCB, row, col); }
    };


    /**
       This is a template driven generic gauge field accessor.  To
       deploy for a specifc field ordering, the two operator()
       accessors have to be specialized for that ordering.
     */
    template <typename Float, int nColor, int nSpinCoarse, QudaGaugeFieldOrder order>
      struct FieldOrder {

      protected:
	/** An internal reference to the actual field we are accessing */
	const int volumeCB;
	const int nDim;
	const int geometry;
	static constexpr int nColorCoarse = nColor / nSpinCoarse;

	const Accessor<Float,nColor,order> accessor;
	const GhostAccessor<Float,nColor,order> ghostAccessor;

      public:
	/**
	 * Constructor for the FieldOrder class
	 * @param field The field that we are accessing
	 */
      FieldOrder(GaugeField &U, void *gauge_=0, void **ghost_=0)
      : volumeCB(U.VolumeCB()), nDim(U.Ndim()), geometry(U.Geometry()),
	  accessor(U, gauge_, ghost_), ghostAccessor(U, gauge_, ghost_)
	{
	  if (U.Reconstruct() != QUDA_RECONSTRUCT_NO)
	    errorQuda("GaugeField ordering not supported with reconstruction");
	}

      FieldOrder(const FieldOrder &o) : volumeCB(o.volumeCB),
	  nDim(o.nDim), geometry(o.geometry),
	  accessor(o.accessor), ghostAccessor(o.ghostAccessor)
	{ }

	virtual ~FieldOrder() { ; }

	/**
	 * Read-only complex-member accessor function
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ const complex<Float>& operator()(int d, int parity, int x, int row, int col) const
	{ return accessor(d,parity,x,row,col); }

	/**
	 * Writable complex-member accessor function
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ complex<Float>& operator() (int d, int parity, int x, int row, int col)
	{ return accessor(d,parity,x,row,col); }

	/**
	 * Read-only complex-member accessor function for the ghost zone
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ const complex<Float>& Ghost(int d, int parity, int x, int row, int col) const
	{ return ghostAccessor(d,parity,x,row,col); }

	/**
	 * Writable complex-member accessor function for the ghost zone
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ complex<Float>& Ghost(int d, int parity, int x, int row, int col)
	{ return ghostAccessor(d,parity,x,row,col); }

    	/**
	 * Specialized read-only complex-member accessor function (for coarse gauge field)
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param s_row row spin index
	 * @param c_row row color index
	 * @param s_col col spin index
	 * @param c_col col color index
	 */
	__device__ __host__ inline const complex<Float>& operator()(int d, int parity, int x, int s_row,
							     int s_col, int c_row, int c_col) const {
	  return (*this)(d, parity, x, s_row*nColorCoarse + c_row, s_col*nColorCoarse + c_col);
	}

	/**
	 * Specialized read-only complex-member accessor function (for coarse gauge field)
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param s_row row spin index
	 * @param c_row row color index
	 * @param s_col col spin index
	 * @param c_col col color index
	 */
	__device__ __host__ inline complex<Float>& operator()(int d, int parity, int x, int s_row,
							     int s_col, int c_row, int c_col) {
	  return (*this)(d, parity, x, s_row*nColorCoarse + c_row, s_col*nColorCoarse + c_col);
	}

    	/**
	 * Specialized read-only complex-member accessor function (for coarse gauge field ghost zone)
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param s_row row spin index
	 * @param c_row row color index
	 * @param s_col col spin index
	 * @param c_col col color index
	 */
	__device__ __host__ inline const complex<Float>& Ghost(int d, int parity, int x, int s_row,
							     int s_col, int c_row, int c_col) const {
	  return Ghost(d, parity, x, s_row*nColorCoarse + c_row, s_col*nColorCoarse + c_col);
	}

	/**
	 * Specialized read-only complex-member accessor function (for coarse gauge field ghost zone)
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param s_row row spin index
	 * @param c_row row color index
	 * @param s_col col spin index
	 * @param c_col col color index
	 */
	__device__ __host__ inline complex<Float>& Ghost(int d, int parity, int x, int s_row,
							     int s_col, int c_row, int c_col) {
	  return Ghost(d, parity, x, s_row*nColorCoarse + c_row, s_col*nColorCoarse + c_col);
	}

	/** Returns the number of field colors */
	__device__ __host__ inline int Ncolor() const { return nColor; }

	/** Returns the field volume */
	__device__ __host__ inline int Volume() const { return 2*volumeCB; }

	/** Returns the field volume */
	__device__ __host__ inline int VolumeCB() const { return volumeCB; }

	/** Returns the field geometric dimension */
	__device__ __host__ inline int Ndim() const { return nDim; }

	/** Returns the field geometry */
	__device__ __host__ inline int Geometry() const { return geometry; }

	/** Returns the number of coarse gauge field spins */
	__device__ __host__ inline int NspinCoarse() const { return nSpinCoarse; }

	/** Returns the number of coarse gauge field colors */
	__device__ __host__ inline int NcolorCoarse() const { return nColorCoarse; }


	__host__ double norm2(int dim) const {
	  double nrm2 = 0.0;
	  for (int parity=0; parity<2; parity++)
	    for (int x_cb=0; x_cb<volumeCB; x_cb++) {
	      for (int row=0; row<nColor; row++)
		for (int col=0; col<nColor; col++)
		  nrm2 += norm((*this)(dim,parity,x_cb,row,col));
	    }
	  return nrm2;
	}

	/** Return the size of the allocation (geometry and parity left out and added as needed in Tunable::bytes) */
	size_t Bytes() const { return static_cast<size_t>(volumeCB) * nColor * nColor * 2ll * sizeof(Float); }
      };


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


	__device__ __host__ inline void getPhase(RegType *phase, const RegType in[N]) const {
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
      }

      /**
	 timeBoundary variant for extended gauge field
	 @param idx extended field linear index
	 @param X the gauge field dimensions
	 @param R the radii dimenions of the extended region
	 @param tBoundary the boundary condition
	 @param isFirstTimeSlice if we're on the first time slice of nodes
	 @param isLastTimeSlide if we're on the last time slice of nodes
	 @param ghostExchange if the field is extended or not (determines indexing type)
      */
      template <typename Float>
      __device__ __host__ inline Float timeBoundary(int idx, const int X[QUDA_MAX_DIM], const int R[QUDA_MAX_DIM],
						    QudaTboundary tBoundary, bool isFirstTimeSlice, bool isLastTimeSlice,
						    QudaGhostExchange ghostExchange) {
	if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED) {
	  if ( idx >= X[3]*X[2]*X[1]*X[0]/2 ) { // halo region on the first time slice
	    return isFirstTimeSlice ? static_cast<Float>(tBoundary) : static_cast<Float>(1.0);
	  } else if ( idx >= (X[3]-1)*X[0]*X[1]*X[2]/2 ) { // last link on the last time slice
	    return isLastTimeSlice ? static_cast<Float>(tBoundary) : static_cast<Float>(1.0);
	  } else {
	    return static_cast<Float>(1.0);
	  }
	} else {
	  if ( idx >= (R[3]-1)*X[0]*X[1]*X[2]/2 && idx < R[3]*X[0]*X[1]*X[2]/2 ) {
	    // the boundary condition is on the R[3]-1 time slice
	    return isFirstTimeSlice ? static_cast<Float>(tBoundary) : static_cast<Float>(1.0);
	  } else if ( idx >= (X[3]-R[3]-1)*X[0]*X[1]*X[2]/2 && idx < (X[3]-R[3])*X[0]*X[1]*X[2]/2 ) {
	    // the boundary condition lies on the X[3]-R[3]-1 time slice
	    return isLastTimeSlice ? static_cast<Float>(tBoundary) : static_cast<Float>(1.0);
	  } else {
	    return static_cast<Float>(1.0);
	  }
	}
      }

      template <typename Float>
      struct Reconstruct<12,Float> {
	typedef typename mapper<Float>::type RegType;
	int X[QUDA_MAX_DIM];
	int R[QUDA_MAX_DIM];
	const RegType anisotropy;
	const QudaTboundary tBoundary;
	bool isFirstTimeSlice;
	bool isLastTimeSlice;
	QudaGhostExchange ghostExchange;

      Reconstruct(const GaugeField &u) : anisotropy(u.Anisotropy()), tBoundary(u.TBoundary()),
	  isFirstTimeSlice(comm_coord(3) == 0 ?true : false),
	  isLastTimeSlice(comm_coord(3) == comm_dim(3)-1 ? true : false),
	  ghostExchange(u.GhostExchange()) {
	  for (int i=0; i<QUDA_MAX_DIM; i++) {
	    X[i] = u.X()[i];
	    R[i] = u.R()[i];
	  }
	}

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
	    timeBoundary<RegType>(idx, X, R, tBoundary,isFirstTimeSlice, isLastTimeSlice, ghostExchange);

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
	Trig<isHalf<RegType>::value,RegType>::SinCos(static_cast<RegType>(3.*phase), &cos_sin[1], &cos_sin[0]);
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

	*phase = Trig<isHalf<RegType>::value,RegType>::Atan2(expI3Phase[1], expI3Phase[0])/3.;
	return;
      }

    };


 template <typename Float>
    struct Reconstruct<8,Float> {
    typedef typename mapper<Float>::type RegType;
    int X[QUDA_MAX_DIM];
    int R[QUDA_MAX_DIM];
    const RegType anisotropy;
    const QudaTboundary tBoundary;
    bool isFirstTimeSlice;
    bool isLastTimeSlice;
    QudaGhostExchange ghostExchange;

  Reconstruct(const GaugeField &u) : anisotropy(u.Anisotropy()), tBoundary(u.TBoundary()),
      isFirstTimeSlice(comm_coord(3) == 0 ? true : false),
      isLastTimeSlice(comm_coord(3) == comm_dim(3)-1 ? true : false),
      ghostExchange(u.GhostExchange()) {
      for (int i=0; i<QUDA_MAX_DIM; i++) {
	X[i] = u.X()[i];
	R[i] = u.R()[i];
      }
    }

    __device__ __host__ inline void Pack(RegType out[8], const RegType in[18], int idx) const {
      out[0] = Trig<isHalf<Float>::value,RegType>::Atan2(in[1], in[0]);
      out[1] = Trig<isHalf<Float>::value,RegType>::Atan2(in[13], in[12]);
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
	timeBoundary<RegType>(idx, X, R, tBoundary,isFirstTimeSlice, isLastTimeSlice, ghostExchange);

      RegType diff = static_cast<RegType>(1.0)/(u0*u0) - row_sum;
      RegType U00_mag = sqrt(diff >= static_cast<RegType>(0.0) ? diff : static_cast<RegType>(0.0));

      out[0] = U00_mag * Trig<isHalf<Float>::value,RegType>::Cos(in[0]);
      out[1] = U00_mag * Trig<isHalf<Float>::value,RegType>::Sin(in[0]);

      // Now reconstruct first column
      RegType column_sum = 0.0;
      for (int i=0; i<2; i++) column_sum += out[i]*out[i];
      for (int i=6; i<8; i++) {
	out[i] = in[i];
	column_sum += in[i]*in[i];
      }
      diff = 1.f/(u0*u0) - column_sum;
      RegType U20_mag = sqrt(diff >= static_cast<RegType>(0.0) ? diff : static_cast<RegType>(0.0));

      out[12] = U20_mag * Trig<isHalf<Float>::value,RegType>::Cos(in[1]);
      out[13] = U20_mag * Trig<isHalf<Float>::value,RegType>::Sin(in[1]);
      // First column now restored

      // finally reconstruct last elements from SU(2) rotation
      RegType r_inv2 = static_cast<RegType>(1.0)/(u0*row_sum);

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

	*phase = Trig<isHalf<RegType>::value,RegType>::Atan2(expI3Phase[1], expI3Phase[0])/3.;
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
	Trig<isHalf<RegType>::value,RegType>::SinCos(phase, &cos_sin[1], &cos_sin[0]);
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
      Float *gauge;
      size_t offset;
      Float *ghost[4];
      const int volumeCB;
      int faceVolumeCB[4];
      const int stride;
      const int geometry;
      const int hasPhase;
      const size_t phaseOffset;
      void *backup_h; //! host memory for backing up the field when tuning
      size_t bytes;

      FloatNOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0)
      : reconstruct(u), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()),
	offset(u.Bytes()/(2*sizeof(Float))),
	volumeCB(u.VolumeCB()), stride(u.Stride()), geometry(u.Geometry()),
	hasPhase((u.Reconstruct() == QUDA_RECONSTRUCT_9 || u.Reconstruct() == QUDA_RECONSTRUCT_13) ? 1 : 0),
	phaseOffset(u.PhaseOffset()), backup_h(0), bytes(u.Bytes())
      {
	for (int i=0; i<4; i++) {
	  ghost[i] = ghost_ ? ghost_[i] : 0;
	  faceVolumeCB[i] = u.SurfaceCB(i)*u.Nface(); // face volume equals surface * depth
	}
      }

    FloatNOrder(const FloatNOrder &order)
      : reconstruct(order.reconstruct), gauge(order.gauge), offset(order.offset),
        volumeCB(order.volumeCB), stride(order.stride), geometry(order.geometry),
	hasPhase(order.hasPhase), phaseOffset(order.phaseOffset), backup_h(0), bytes(order.bytes)
      {
	for (int i=0; i<4; i++) {
	  ghost[i] = order.ghost[i];
	  faceVolumeCB[i] = order.faceVolumeCB[i];
	}
      }
      virtual ~FloatNOrder() { ; }

      __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity) const {
        const int M = reconLen / N;
        RegType tmp[reconLen];
	typedef typename VectorType<Float,N>::type Vector;
	typedef typename VectorType<RegType,N>::type RegVector;

#pragma unroll
        for (int i=0; i<M; i++){
	  // first do vectorized copy from memory
	  Vector vecTmp = vector_load<Vector>(gauge + parity*offset, x + dir*stride*M + stride*i);
	  // second do vectorized copy converting into register type
          copy(reinterpret_cast<RegVector*>(tmp)[i], vecTmp);
        }

        RegType phase = 0.;
        if(hasPhase) copy(phase, (gauge+parity*offset)[phaseOffset/sizeof(Float) + stride*dir + x]);
        // The phases come after the ghost matrices
        reconstruct.Unpack(v, tmp, x, dir, 2.*M_PI*phase);
      }

      __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {
        const int M = reconLen / N;
        RegType tmp[reconLen];
        reconstruct.Pack(tmp, v, x);
	typedef typename VectorType<Float,N>::type Vector;
	typedef typename VectorType<RegType,N>::type RegVector;

#pragma unroll
        for (int i=0; i<M; i++){
	  Vector vecTmp;
	  // first do vectorized copy converting into storage type
	  copy(vecTmp, reinterpret_cast<RegVector*>(tmp)[i]);
	  // second do vectorized copy into memory
	  reinterpret_cast< Vector* >(gauge + parity*offset)[x + dir*stride*M + stride*i] = vecTmp;
        }
        if(hasPhase){
          RegType phase;
          reconstruct.getPhase(&phase,v);
          copy((gauge+parity*offset)[phaseOffset/sizeof(Float) + dir*stride + x], static_cast<RegType>(phase/(2.*M_PI)));
        }
      }

      __device__ __host__ inline void loadGhost(RegType v[length], int x, int dir, int parity) const {
        if (!ghost[dir]) { // load from main field not separate array
          load(v, volumeCB+x, dir, parity); // an offset of size volumeCB puts us at the padded region
          // This also works perfectly when phases are stored. No need to change this.
        } else {
          const int M = reconLen / N;
          RegType tmp[reconLen];
	  typedef typename VectorType<Float,N>::type Vector;
	  typedef typename VectorType<RegType,N>::type RegVector;

#pragma unroll
          for (int i=0; i<M; i++) {
	    // first do vectorized copy from memory into registers
	    Vector vecTmp = vector_load<Vector>(ghost[dir]+parity*faceVolumeCB[dir]*(M*N + hasPhase),
						i*faceVolumeCB[dir]+x);
	    // second do vectorized copy converting into register type
	    copy(reinterpret_cast< RegVector* >(tmp)[i], vecTmp);
          }
          RegType phase=0.;
          if(hasPhase) copy(phase, ghost[dir][parity*faceVolumeCB[dir]*(M*N + 1) + faceVolumeCB[dir]*M*N + x]);
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
	  typedef typename VectorType<Float,N>::type Vector;
	  typedef typename VectorType<RegType,N>::type RegVector;

#pragma unroll
          for (int i=0; i<M; i++) {
	    const int hasPhase = 0;
	    Vector vecTmp;
	    // first do vectorized copy converting into storage type
	    copy(vecTmp, reinterpret_cast< RegVector* >(tmp)[i]);
	    // second do vectorized copy into memory
	    reinterpret_cast< Vector*>
	      (ghost[dir]+parity*faceVolumeCB[dir]*(M*N + hasPhase))[i*faceVolumeCB[dir]+x] = vecTmp;
          }

	  if(hasPhase){
	    RegType phase=0.;
	    reconstruct.getPhase(&phase, v);
	    copy(ghost[dir][parity*faceVolumeCB[dir]*(M*N + 1) + faceVolumeCB[dir]*M*N + x], static_cast<RegType>(phase/(2.*M_PI)));
	  }
	}
      }

      __device__ __host__ inline void loadGhostEx(RegType v[length], int buff_idx, int extended_idx, int dir,
						    int dim, int g, int parity, const int R[]) const {
	const int M = reconLen / N;
	RegType tmp[reconLen];
	typedef typename VectorType<Float,N>::type Vector;
	typedef typename VectorType<RegType,N>::type RegVector;

#pragma unroll
	for (int i=0; i<M; i++) {
	  // first do vectorized copy from memory
	  Vector vecTmp = vector_load<Vector>(ghost[dim] + ((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + hasPhase),
					      +i*R[dim]*faceVolumeCB[dim]+buff_idx);
	  // second do vectorized copy converting into register type
	  copy(reinterpret_cast< RegVector* >(tmp)[i], vecTmp);
	}
	RegType phase=0.;
	if(hasPhase) copy(phase, ghost[dim][((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + 1)
					    + R[dim]*faceVolumeCB[dim]*M*N + buff_idx]);

	// use the extended_idx to determine the boundary condition
	reconstruct.Unpack(v, tmp, extended_idx, g, 2.*M_PI*phase);
      }

      __device__ __host__ inline void saveGhostEx(const RegType v[length], int buff_idx, int extended_idx,
						    int dir, int dim, int g, int parity, const int R[]) {
	const int M = reconLen / N;
	RegType tmp[reconLen];
	// use the extended_idx to determine the boundary condition
	reconstruct.Pack(tmp, v, extended_idx);
	typedef typename VectorType<Float,N>::type Vector;
	typedef typename VectorType<RegType,N>::type RegVector;

#pragma unroll
	  for (int i=0; i<M; i++) {
	    Vector vecTmp;
	    // first do vectorized copy converting into storage type
	    copy(vecTmp, reinterpret_cast< RegVector* >(tmp)[i]);
	    // second do vectorized copy to memory
	    reinterpret_cast< Vector* >
	      (ghost[dim] + ((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + hasPhase))
	      [i*R[dim]*faceVolumeCB[dim]+buff_idx] = vecTmp;
	  }
	  if(hasPhase){
	    RegType phase=0.;
	    reconstruct.getPhase(&phase, v);
	    copy(ghost[dim][((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + 1) + R[dim]*faceVolumeCB[dim]*M*N + buff_idx],
		 static_cast<RegType>(phase/(2.*M_PI)));
	  }
	}

      /**
	 used to backup the field to the host when tuning
      */
      void save() {
	if (backup_h) errorQuda("Already allocated host backup");
	backup_h = safe_malloc(bytes);
	cudaMemcpy(backup_h, gauge, bytes, cudaMemcpyDeviceToHost);
	checkCudaError();
      }

      /**
	 restore the field from the host after tuning
      */
      void load() {
	cudaMemcpy(gauge, backup_h, bytes, cudaMemcpyHostToDevice);
	host_free(backup_h);
	backup_h = 0;
	checkCudaError();
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

      __device__ __host__ inline void loadGhostEx(RegType v[length], int x, int dummy, int dir,
						  int dim, int g, int parity, const int R[]) const {
	for (int i=0; i<length; i++) {
	  v[i] = ghost[dim]
	    [(((dir*2+parity)*R[dim]*faceVolumeCB[dim] + x)*geometry+g)*length + i];
	}
      }

      __device__ __host__ inline void saveGhostEx(const RegType v[length], int x, int dummy,
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
    const int geometry;
  MILCOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) :
    LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()),
      volumeCB(u.VolumeCB()), geometry(u.Geometry()) { ; }
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
    const int geometry;
  CPSOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0)
    : LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()),
      volumeCB(u.VolumeCB()), anisotropy(u.Anisotropy()), Nc(3),
      geometry(u.Geometry())
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

  } // namespace gauge

  // Use traits to reduce the template explosion
  template<typename T,QudaReconstructType,int N=18> struct gauge_mapper { };

  // double precision
  template<int N> struct gauge_mapper<double,QUDA_RECONSTRUCT_NO,N> { typedef gauge::FloatNOrder<double, N, 2, N> type; };
  template<int N> struct gauge_mapper<double,QUDA_RECONSTRUCT_13,N> { typedef gauge::FloatNOrder<double, N, 2, 13> type; };
  template<int N> struct gauge_mapper<double,QUDA_RECONSTRUCT_12,N> { typedef gauge::FloatNOrder<double, N, 2, 12> type; };
  template<int N> struct gauge_mapper<double,QUDA_RECONSTRUCT_9,N> { typedef gauge::FloatNOrder<double, N, 2, 9> type; };
  template<int N> struct gauge_mapper<double,QUDA_RECONSTRUCT_8,N> { typedef gauge::FloatNOrder<double, N, 2, 8> type; };

  // single precision
  template<int N> struct gauge_mapper<float,QUDA_RECONSTRUCT_NO,N> { typedef gauge::FloatNOrder<float, N, 2, N> type; };
  template<int N> struct gauge_mapper<float,QUDA_RECONSTRUCT_13,N> { typedef gauge::FloatNOrder<float, N, 4, 13> type; };
  template<int N> struct gauge_mapper<float,QUDA_RECONSTRUCT_12,N> { typedef gauge::FloatNOrder<float, N, 4, 12> type; };
  template<int N> struct gauge_mapper<float,QUDA_RECONSTRUCT_9,N> { typedef gauge::FloatNOrder<float, N, 4, 9> type; };
  template<int N> struct gauge_mapper<float,QUDA_RECONSTRUCT_8,N> { typedef gauge::FloatNOrder<float, N, 4, 8> type; };

  // half precision
  template<int N> struct gauge_mapper<short,QUDA_RECONSTRUCT_NO,N> { typedef gauge::FloatNOrder<short, N, 2, N> type; };
  template<int N> struct gauge_mapper<short,QUDA_RECONSTRUCT_13,N> { typedef gauge::FloatNOrder<short, N, 4, 13> type; };
  template<int N> struct gauge_mapper<short,QUDA_RECONSTRUCT_12,N> { typedef gauge::FloatNOrder<short, N, 4, 12> type; };
  template<int N> struct gauge_mapper<short,QUDA_RECONSTRUCT_9,N> { typedef gauge::FloatNOrder<short, N, 4, 9> type; };
  template<int N> struct gauge_mapper<short,QUDA_RECONSTRUCT_8,N> { typedef gauge::FloatNOrder<short, N, 4, 8> type; };

  template<typename T, QudaGaugeFieldOrder order, int Nc> struct gauge_order_mapper { };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_QDP_GAUGE_ORDER,Nc> { typedef gauge::QDPOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_FLOAT2_GAUGE_ORDER,Nc> { typedef gauge::FloatNOrder<T, 2*Nc*Nc, 2, 2*Nc*Nc> type; };


  // experiments in reducing template instantation boilerplate
  // can this be replaced with a C++11 variant that uses variadic templates?

#define INSTANTIATE_RECONSTRUCT(func, g, ...)				\
  {									\
    if (!data.isNative())						\
      errorQuda("Field order %d and precision %d is not native", g.Order(), g.Precision()); \
    if( g.Reconstruct() == QUDA_RECONSTRUCT_NO) {			\
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Gauge; \
      func(Gauge(g), g, __VA_ARGS__);					\
    } else if( g.Reconstruct() == QUDA_RECONSTRUCT_12){			\
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Gauge; \
      func(Gauge(g), g, __VA_ARGS__);					\
    } else if( g.Reconstruct() == QUDA_RECONSTRUCT_8){			\
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Gauge; \
      func(Gauge(g), g, __VA_ARGS__);					\
    } else {								\
      errorQuda("Reconstruction type %d of gauge field not supported", g.Reconstruct()); \
    }									\
  }

#define INSTANTIATE_PRECISION(func, lat, ...)				\
  {									\
    if (lat.Precision() == QUDA_DOUBLE_PRECISION) {			\
      func<double>(lat, __VA_ARGS__);					\
    } else if(lat.Precision() == QUDA_SINGLE_PRECISION) {		\
      func<float>(lat, __VA_ARGS__);					\
    } else {								\
      errorQuda("Precision %d not supported", lat.Precision());		\
    }									\
  }

} // namespace quda

#endif // _GAUGE_ORDER_H
