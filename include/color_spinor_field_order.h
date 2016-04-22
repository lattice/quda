#ifndef _COLOR_SPINOR_ORDER_H
#define _COLOR_SPINOR_ORDER_H

/**
 * @file color_spinor_field_order.h
 *
 * @section DESCRIPTION 
 *
 * Define functors to allow for generic accessors regardless of field
 * ordering.  Currently this is used for cpu fields only with limited
 * ordering support, but this will be expanded for device ordering
 *  also.
 */

#include <register_traits.h>
#include <typeinfo>
#include <complex_quda.h>

namespace quda {

  namespace colorspinor {

    template<typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order> struct AccessorCB { 
      AccessorCB(const ColorSpinorField &) { }
      __device__ __host__ inline int index(int parity, int x_cb, int s, int c, int v) const {	
#ifndef __CUDA_ARCH__
	errorQuda("Not implemented");
#endif
	return 0;
      }
    };

    template<typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order> struct GhostAccessorCB {
      GhostAccessorCB(const ColorSpinorField &) { }
      __device__ __host__ inline int index(int dim, int dir, int parity, int x_cb, int s, int c, int v) const {
#ifndef __CUDA_ARCH__
	errorQuda("Not implemented");
#endif
	return 0;
      }
    };

    template<typename Float, int nSpin, int nColor, int nVec> 
      struct AccessorCB<Float,nSpin,nColor,nVec,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> { 
      const int offset_cb;
    AccessorCB(const ColorSpinorField &field) : offset_cb((field.Bytes()>>1) / sizeof(complex<Float>)) { }
      __device__ __host__ inline int index(int parity, int x_cb, int s, int c, int v) const 
      { return parity*offset_cb + ((x_cb*nSpin+s)*nColor+c)*nVec+v; }
    };

    template<typename Float, int nSpin, int nColor, int nVec>
      struct GhostAccessorCB<Float,nSpin,nColor,nVec,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> {
      int ghostOffset[4];
      GhostAccessorCB(const ColorSpinorField &a) {
	for (int d=0; d<4; d++) {
	  ghostOffset[d] = a.Nface()*a.SurfaceCB(d)*a.Ncolor()*a.Nspin();
	}
      }
      __device__ __host__ inline int index(int dim, int dir, int parity, int x_cb, int s, int c, int v) const
      { return parity*ghostOffset[dim] + ((x_cb*nSpin+s)*nColor+c)*nVec+v; }
    };

    template<int nSpin, int nColor, int nVec, int N>
      __device__ __host__ inline int indexFloatN(int parity, int x_cb, int s, int c, int v, int stride, int offset_cb) {
      int j = (((s*nColor+c)*nVec+v)*2) / N; // factor of two for complexity
      int i = (((s*nColor+c)*nVec+v)*2) % N;      
      int index = ((j*stride+x_cb)*2+i) / 2; // back to a complex offset
      index += parity*offset_cb;
      return index;
    };

    template<typename Float, int nSpin, int nColor, int nVec> 
      struct AccessorCB<Float,nSpin,nColor,nVec,QUDA_FLOAT2_FIELD_ORDER> { 
      const int stride;
      const int offset_cb;
    AccessorCB(const ColorSpinorField &field): stride(field.Stride()), 
	offset_cb((field.Bytes()>>1) / sizeof(complex<Float>)) { }
      __device__ __host__ inline int index(int parity, int x_cb, int s, int c, int v) const 
      { return parity*offset_cb + ((s*nColor+c)*nVec+v)*stride+x_cb; }
    };

    template<typename Float, int nSpin, int nColor, int nVec>
      struct GhostAccessorCB<Float,nSpin,nColor,nVec,QUDA_FLOAT2_FIELD_ORDER> {
      int faceVolumeCB[4];
      int ghostOffset[4];
      GhostAccessorCB(const ColorSpinorField &a) {
	for (int d=0; d<4; d++) {
	  faceVolumeCB[d] = a.Nface()*a.SurfaceCB(d);
	  ghostOffset[d] = faceVolumeCB[d]*nColor*nSpin;
	}
      }
      __device__ __host__ inline int index(int dim, int dir, int parity, int x_cb, int s, int c, int v) const
      { return parity*ghostOffset[dim] + ((s*nColor+c)*nVec+v)*faceVolumeCB[dim] + x_cb; }
    };


    template <typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order>
      class FieldOrderCB {

    protected:
      complex<Float> *v;
      complex<Float> *ghost[8];
      mutable int x[QUDA_MAX_DIM];
      const int volumeCB;
      const int nDim;
      const QudaGammaBasis gammaBasis;
      const AccessorCB<Float,nSpin,nColor,nVec,order> accessor;
      const GhostAccessorCB<Float,nSpin,nColor,nVec,order> ghostAccessor;
      const int siteSubset;
      const int nParity;

    public:
      /** 
       * Constructor for the FieldOrderCB class
       * @param field The field that we are accessing
       */
    FieldOrderCB(const ColorSpinorField &field, void *v_=0, void **ghost_=0)
      : v(v_? static_cast<complex<Float>*>(const_cast<void*>(v_))
	  : static_cast<complex<Float>*>(const_cast<void*>(field.V()))),
	volumeCB(field.VolumeCB()),
	nDim(field.Ndim()), gammaBasis(field.GammaBasis()), 
	siteSubset(field.SiteSubset()), nParity(field.SiteSubset()),
	accessor(field), ghostAccessor(field)
      { 
	for (int d=0; d<4; d++) {
	  x[d]=field.X(d); 
	  void * const *_ghost = ghost_ ? ghost_ : field.Ghost();
	  ghost[2*d+0] = static_cast<complex<Float>*>(_ghost[2*d+0]);
	  ghost[2*d+1] = static_cast<complex<Float>*>(_ghost[2*d+1]);
	}
      }

      /**
       * Destructor for the FieldOrderCB class
       */
      virtual ~FieldOrderCB() { ; }

      void resetGhost(void * const *ghost_)
      {
	for (int d=0; d<4; d++) {
	  ghost[2*d+0] = static_cast<complex<Float>*>(ghost_[2*d+0]);
	  ghost[2*d+1] = static_cast<complex<Float>*>(ghost_[2*d+1]);
	}
      }

      /**
       * Read-only complex-member accessor function.  The last
       * parameter n is only used for indexed into the packed
       * null-space vectors.
       * @param x 1-d checkerboard site index
       * @param s spin index
       * @param c color index
       * @param v vector number
       */
      __device__ __host__ inline const complex<Float>& operator()(int parity, int x_cb, int s, int c, int n=0) const
      {	return v[accessor.index(parity,x_cb,s,c,n)]; }

      /**
       * Writable complex-member accessor function.  The last
       * parameter n is only used for indexed into the packed
       * null-space vectors.
       * @param x 1-d checkerboard site index
       * @param s spin index
       * @param c color index
       * @param v vector number
       */
      __device__ __host__ inline complex<Float>& operator()(int parity, int x_cb, int s, int c, int n=0)
      { return v[accessor.index(parity,x_cb,s,c,n)]; }

      /**
       * Read-only complex-member accessor function for the ghost
       * zone.  The last parameter n is only used for indexed into the
       * packed null-space vectors.
       * @param x 1-d checkerboard site index
       * @param s spin index
       * @param c color index
       * @param v vector number
       */
      __device__ __host__ inline const complex<Float>& Ghost(int dim, int dir, int parity, int x_cb, int s, int c, int n=0) const
      {	return ghost[2*dim+dir][ghostAccessor.index(dim,dir,parity,x_cb,s,c,n)]; }

      /**
       * Writable complex-member accessor function for the ghost zone.
       * The last parameter n is only used for indexed into the packed
       * null-space vectors.
       * @param x 1-d checkerboard site index
       * @param s spin index
       * @param c color index
       * @param n vector number
       */
      __device__ __host__ inline complex<Float>& Ghost(int dim, int dir, int parity, int x_cb, int s, int c, int n=0)
      { return ghost[2*dim+dir][ghostAccessor.index(dim,dir,parity,x_cb,s,c,n)]; }

      /**
	 Convert from 1-dimensional index to the n-dimensional spatial index.
	 With full fields, we assume that the field is even-odd ordered.  The
	 lattice coordinates that are computed here are full-field
	 coordinates.
      */
      __device__ __host__ inline void LatticeIndex(int y[QUDA_MAX_DIM], int i) const {
	if (siteSubset == QUDA_FULL_SITE_SUBSET) x[0] /= 2;
	
	for (int d=0; d<nDim; d++) {
	  y[d] = i % x[d];
	  i /= x[d];    
	}
	int parity = i; // parity is the slowest running dimension
	
	// convert into the full-field lattice coordinate
	if (siteSubset == QUDA_FULL_SITE_SUBSET) {
	  for (int d=1; d<nDim; d++) parity += y[d];
	  parity = parity & 1;
	  x[0] *= 2; // restore x[0]
	}
	y[0] = 2*y[0] + parity;  // compute the full x coordinate
      }
      
      /**
	 Convert from n-dimensional spatial index to the 1-dimensional index.
	 With full fields, we assume that the field is even-odd ordered.  The
	 input lattice coordinates are always full-field coordinates.
      */
      __device__ __host__ inline void OffsetIndex(int &i, int y[QUDA_MAX_DIM]) const {
	int parity = 0;
	int savey0 = y[0];
	
	if (siteSubset == QUDA_FULL_SITE_SUBSET) {
	  for (int d=0; d<nDim; d++) parity += y[d];
	  parity = parity & 1;
	  y[0] /= 2;
	  x[0] /= 2; 
	}
	
	i = parity;
	for (int d=nDim-1; d>=0; d--) i = x[d]*i + y[d];
	
	if (siteSubset == QUDA_FULL_SITE_SUBSET) {
	  //y[0] = 2*y[0] + parity;
	  y[0] = savey0;
	  x[0] *= 2; // restore x[0]
	}
      }

      /** Return the length of dimension d */
      __device__ __host__ inline int X(int d) const { return x[d]; }

      /** Return the length of dimension d */
      __device__ __host__ inline const int* X() const { return x; }

      /** Returns the number of field colors */
       __device__ __host__ inline int Ncolor() const { return nColor; }

      /** Returns the number of field spins */
      __device__ __host__ inline int Nspin() const { return nSpin; }

      /** Returns the number of field parities (1 or 2) */
      __device__ __host__ inline int Nparity() const { return nParity; }

      /** Returns the field volume */
      __device__ __host__ inline int VolumeCB() const { return volumeCB; }

      /** Returns the field geometric dimension */
      __device__ __host__ inline int Ndim() const { return nDim; }

      /** Returns the field geometric dimension */
      __device__ __host__ inline QudaGammaBasis GammaBasis() const { return gammaBasis; }

      /** Returns the number of packed vectors (for mg prolongator) */
      __device__ __host__ inline int Nvec() const { return nVec; }

      __host__ double norm2() const {
	double nrm2 = 0.0;
	for (int parity=0; parity<nParity; parity++)
	  for (int x_cb=0; x_cb<volumeCB; x_cb++)
	    for (int s=0; s<nSpin; s++)
	      for (int c=0; c<nColor; c++)
		for (int v=0; v<nVec; v++)
		  nrm2 += norm((*this)(parity,x_cb,s,c,v));
	return nrm2;
      }

      size_t Bytes() const { return nParity * static_cast<size_t>(volumeCB) * nColor * nSpin * nVec * 2ll * sizeof(Float); }
    };

    template <typename Float, int Ns, int Nc, int N>
      struct FloatNOrder {
	typedef typename mapper<Float>::type RegType;
	typedef typename VectorType<Float,N>::type Vector;
	typedef typename VectorType<RegType,N>::type RegVector;
	static const int length = 2 * Ns * Nc;
	static const int M = length / N;
	Float *field;
	size_t offset;
	float *norm;
	size_t norm_offset;
	int volumeCB;
	int faceVolumeCB[4];
	int stride;
	Float *ghost[8];
	int nParity;
      FloatNOrder(const ColorSpinorField &a, Float *field_=0, float *norm_=0, Float **ghost_=0)
      : field(field_ ? field_ : (Float*)a.V()), offset(a.Bytes()/(2*sizeof(Float))),
	norm(norm_ ? norm_ : (float*)a.Norm()), norm_offset(a.NormBytes()/(2*sizeof(float))),
	volumeCB(a.VolumeCB()), stride(a.Stride()), nParity(a.SiteSubset())
	{
	  for (int i=0; i<4; i++) {
	    ghost[2*i+0] = ghost_ ? ghost_[2*i+0] : static_cast<Float*>(a.Ghost()[2*i+0]);
	    ghost[2*i+1] = ghost_ ? ghost_[2*i+1] : static_cast<Float*>(a.Ghost()[2*i+1]);
	    faceVolumeCB[i] = a.SurfaceCB(i)*a.Nface();
	  }
	}
	virtual ~FloatNOrder() { ; }

	__device__ __host__ inline void load(RegType v[length], int x, int parity=0) const {
#pragma unroll
	  for (int i=0; i<M; i++) {
	    // first do vectorized copy from memory
	    Vector vecTmp = vector_load<Vector>(field + parity*offset, x + stride*i);
	    // second do vectorized copy converting into register type
	    copy(reinterpret_cast<RegVector*>(v)[i], vecTmp);
	  }

	  if (sizeof(Float)==sizeof(short))
#pragma unroll
	    for (int i=0; i<length; i++) v[i] *= norm[x+parity*norm_offset];
	}

	__device__ __host__ inline void save(const RegType v[length], int x, int parity=0) {
	  RegType scale = 0.0;
	  RegType tmp[length];

	  if (sizeof(Float)==sizeof(short)) {
#pragma unroll
	    for (int i=0; i<length; i++) scale = fabs(v[i]) > scale ? fabs(v[i]) : scale;
	    norm[x+parity*norm_offset] = scale;
	  }

	  if (sizeof(Float)==sizeof(short))
#pragma unroll
	    for (int i=0; i<length; i++) tmp[i] = v[i] * static_cast<Float>(1.0)/scale;
	  else
#pragma unroll
	    for (int i=0; i<length; i++) tmp[i] = v[i];

#pragma unroll
	  for (int i=0; i<M; i++) {
	    Vector vecTmp;
	    // first do vectorized copy converting into storage type
	    copy(vecTmp, reinterpret_cast<RegVector*>(tmp)[i]);
	    // second do vectorized copy into memory
	    reinterpret_cast< Vector* >(field + parity*offset)[x + stride*i] = vecTmp;
	  }
	}

	// no parity argument since we only presently exchange single parity field
	// add support for half-precision ghosts
	__device__ __host__ inline void loadGhost(RegType v[length], int x, int dim, int dir, int parity=0) const {
#pragma unroll
          for (int i=0; i<M; i++) {
	    // first do vectorized copy from memory into registers
	    Vector vecTmp = vector_load<Vector>(ghost[2*dim+dir]+parity*faceVolumeCB[dim]*M*N,
						i*faceVolumeCB[dim]+x);
	    // second do vectorized copy converting into register type
	    copy(reinterpret_cast< RegVector* >(v)[i], vecTmp);
          }
	}

	// no parity argument since we only presently exchange single parity field
	// add support for half-precision ghosts
	__device__ __host__ inline void saveGhost(RegType v[length], int x, int dim, int dir, int parity=0) const {
#pragma unroll
          for (int i=0; i<M; i++) {
	    Vector vecTmp;
	    // first do vectorized copy converting into storage type
	    copy(vecTmp, reinterpret_cast< RegVector* >(v)[i]);
	    // second do vectorized copy into memory
	    reinterpret_cast< Vector*>
	      (ghost[2*dim+dir]+parity*faceVolumeCB[dim]*M*N)[i*faceVolumeCB[dim]+x] = vecTmp;
          }
	}

	size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };

    template <typename Float, int Ns, int Nc>
      struct SpaceColorSpinorOrder {
	typedef typename mapper<Float>::type RegType;
	static const int length = 2 * Ns * Nc;
	Float *field;
	size_t offset;
	Float *ghost[8];
	int volumeCB;
	int faceVolumeCB[4];
	int stride;
	int nParity;
      SpaceColorSpinorOrder(const ColorSpinorField &a, Float *field_=0, float *dummy=0, Float **ghost_=0)
      : field(field_ ? field_ : (Float*)a.V()), offset(a.Bytes()/(2*sizeof(Float))),
	  volumeCB(a.VolumeCB()), stride(a.Stride()), nParity(a.SiteSubset())
	{
	  if (volumeCB != stride) errorQuda("Stride must equal volume for this field order");
	  for (int i=0; i<4; i++) {
	    ghost[2*i] = ghost_ ? ghost_[2*i] : 0;
	    ghost[2*i+1] = ghost_ ? ghost_[2*i+1] : 0;
	    faceVolumeCB[i] = a.SurfaceCB(i)*a.Nface();
	  }
	}
	virtual ~SpaceColorSpinorOrder() { ; }

	__device__ __host__ inline void load(RegType v[length], int x, int parity=0) const {
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = field[parity*offset + ((x*Nc + c)*Ns + s)*2 + z];
	      }
	    }
	  }
	}

	__device__ __host__ inline void save(const RegType v[length], int x, int parity=0) {
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		field[parity*offset + ((x*Nc + c)*Ns + s)*2 + z] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	}

	__device__ __host__ inline void loadGhost(RegType v[length], int x, int dim, int dir, int parity=0) const {
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = ghost[2*dim+dir][(((parity*faceVolumeCB[dim]+x)*Nc + c)*Ns + s)*2 + z];
	      }
	    }
	  }
	}

	__device__ __host__ inline void saveGhost(const RegType v[length], int x, int dim, int dir, int parity=0) {
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		ghost[2*dim+dir][(((parity*faceVolumeCB[dim]+x)*Nc + c)*Ns + s)*2 + z] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	}

	size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };

    template <typename Float, int Ns, int Nc>
      struct SpaceSpinorColorOrder {
	typedef typename mapper<Float>::type RegType;
	static const int length = 2 * Ns * Nc;
	Float *field;
	size_t offset;
	Float *ghost[8];
	int volumeCB;
	int faceVolumeCB[4];
	int stride;
	int nParity;
      SpaceSpinorColorOrder(const ColorSpinorField &a, Float *field_=0, float *dummy=0, Float **ghost_=0)
      : field(field_ ? field_ : (Float*)a.V()), offset(a.Bytes()/(2*sizeof(Float))),
	  volumeCB(a.VolumeCB()), stride(a.Stride()), nParity(a.SiteSubset())
	{
	  if (volumeCB != stride) errorQuda("Stride must equal volume for this field order");
	  for (int i=0; i<4; i++) {
	    ghost[2*i] = ghost_ ? ghost_[2*i] : 0;
	    ghost[2*i+1] = ghost_ ? ghost_[2*i+1] : 0;
	    faceVolumeCB[i] = a.SurfaceCB(i)*a.Nface();
	  }
	}
	virtual ~SpaceSpinorColorOrder() { ; }

	__device__ __host__ inline void load(RegType v[length], int x, int parity=0) const {
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = field[parity*offset + ((x*Ns + s)*Nc + c)*2 + z];
	      }
	    }
	  }
	}

	__device__ __host__ inline void save(const RegType v[length], int x, int parity=0) {
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		field[parity*offset + ((x*Ns + s)*Nc + c)*2 + z] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	}

	__device__ __host__ inline void loadGhost(RegType v[length], int x, int dim, int dir, int parity=0) const {
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = ghost[2*dim+dir][(((parity*faceVolumeCB[dim]+x)*Ns + s)*Nc + c)*2 + z];
	      }
	    }
	  }
	}

	__device__ __host__ inline void saveGhost(const RegType v[length], int x, int dim, int dir, int parity=0) {
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		ghost[2*dim+dir][(((parity*faceVolumeCB[dim]+x)*Ns + s)*Nc + c)*2 + z] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	}

	size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };


    template <typename Float, int Ns, int Nc>
      struct QDPJITDiracOrder {
	typedef typename mapper<Float>::type RegType;
	Float *field;
	int volumeCB;
	int stride;
	int nParity;
      QDPJITDiracOrder(const ColorSpinorField &a, Float *field_=0)
      : field(field_ ? field_ : (Float*)a.V()), volumeCB(a.VolumeCB()), stride(a.Stride()), nParity(a.SiteSubset())
	{ if (volumeCB != a.Stride()) errorQuda("Stride must equal volume for this field order"); }
	virtual ~QDPJITDiracOrder() { ; }

	__device__ __host__ inline void load(RegType v[Ns*Nc*2], int x, int parity=1) const {
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = field[(((z*Nc + c)*Ns + s)*2 + parity)*volumeCB + x];
	      }
	    }
	  }
	}

	__device__ __host__ inline void save(const RegType v[Ns*Nc*2], int x, int parity=0) {
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		field[(((z*Nc + c)*Ns + s)*2 + parity)*volumeCB + x] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	}

	size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };

  } // namespace colorspinor

    // Use traits to reduce the template explosion
  template<typename T,int Ns, int Nc> struct colorspinor_mapper { };
  
  // double precision
  template<int Nc> struct colorspinor_mapper<double,4,Nc> { typedef colorspinor::FloatNOrder<double, 4, Nc, 2> type; };
  template<int Nc> struct colorspinor_mapper<double,2,Nc> { typedef colorspinor::FloatNOrder<double, 2, Nc, 2> type; };
  template<int Nc> struct colorspinor_mapper<double,1,Nc> { typedef colorspinor::FloatNOrder<double, 1, Nc, 2> type; };
  
  // single precision
  template<int Nc> struct colorspinor_mapper<float,4,Nc> { typedef colorspinor::FloatNOrder<float, 4, Nc, 4> type; };
  template<int Nc> struct colorspinor_mapper<float,2,Nc> { typedef colorspinor::FloatNOrder<float, 2, Nc, 2> type; };
  template<int Nc> struct colorspinor_mapper<float,1,Nc> { typedef colorspinor::FloatNOrder<float, 1, Nc, 2> type; };
  
  // half precision
  template<int Nc> struct colorspinor_mapper<short,4,Nc> { typedef colorspinor::FloatNOrder<short, 4, Nc, 4> type; };
  template<int Nc> struct colorspinor_mapper<short,2,Nc> { typedef colorspinor::FloatNOrder<short, 2, Nc, 2> type; };
  template<int Nc> struct colorspinor_mapper<short,1,Nc> { typedef colorspinor::FloatNOrder<short, 1, Nc, 2> type; };
    

  template<typename T, QudaFieldOrder order, int Ns, int Nc> struct colorspinor_order_mapper { };
  template<typename T, int Ns, int Nc> struct colorspinor_order_mapper<T,QUDA_SPACE_COLOR_SPIN_FIELD_ORDER,Ns,Nc> { typedef colorspinor::SpaceColorSpinorOrder<T, Ns, Nc> type; };
  template<typename T, int Ns, int Nc> struct colorspinor_order_mapper<T,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,Ns,Nc> { typedef colorspinor::SpaceSpinorColorOrder<T, Ns, Nc> type; };
  template<typename T, int Ns, int Nc> struct colorspinor_order_mapper<T,QUDA_FLOAT2_FIELD_ORDER,Ns,Nc> { typedef colorspinor::FloatNOrder<T, Ns, Nc, 2> type; };

} // namespace quda

#endif // _COLOR_SPINOR_ORDER_H
