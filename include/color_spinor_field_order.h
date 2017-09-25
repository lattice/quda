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

// trove requires the warp shuffle instructions introduced with Kepler
#if __COMPUTE_CAPABILITY__ >= 300
#include <trove/ptr.h>
#else
#define DISABLE_TROVE
#endif
#include <register_traits.h>
#include <typeinfo>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <color_spinor.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

namespace quda {

    /**
       @brief colorspinor_wrapper is an internal class that is used to
       wrap instances of colorspinor accessors, currying in a specifc
       location on the field.  The operator() accessors in
       colorspinor-field accessors return instances to this class,
       allowing us to then use operator overloading upon this class
       to interact with the ColorSpinor class.  As a result we can
       include colorspinor-field accessors directly in ColorSpinor
       expressions in kernels without having to declare temporaries
       with explicit calls to the load/save methods in the
       colorspinor-field accessors.
    */
  template <typename Float, typename T>
    struct colorspinor_wrapper {
      T &field;
      const int x_cb;
      const int parity;

      /**
	 @brief colorspinor_wrapper constructor
	 @param[in] a colorspinor field accessor we are wrapping
	 @param[in] x_cb checkerboarded space-time index we are accessing
	 @param[in] parity Parity we are accessing
       */
      __device__ __host__ inline colorspinor_wrapper<Float,T>(T &field, int x_cb, int parity)
	: field(field), x_cb(x_cb), parity(parity) { }

      /**
	 @brief Assignment operator with ColorSpinor instance as input
	 @param[in] C ColorSpinor we want to store in this accessor
       */
      template<typename C>
      __device__ __host__ inline void operator=(const C &a) {
	field.save((Float*)a.data, x_cb, parity);
      }
    };

  template <typename T, int Nc, int Ns>
    template <typename S>
    __device__ __host__ inline void ColorSpinor<T,Nc,Ns>::operator=(const colorspinor_wrapper<T,S> &a) {
    a.field.load((T*)data, a.x_cb, a.parity);
  }

  template <typename T, int Nc, int Ns>
    template <typename S>
    __device__ __host__ inline ColorSpinor<T,Nc,Ns>::ColorSpinor(const colorspinor_wrapper<T,S> &a) {
    a.field.load((T*)data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline void ColorSpinor<T,Nc,2>::operator=(const colorspinor_wrapper<T,S> &a) {
    a.field.load((T*)data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline ColorSpinor<T,Nc,2>::ColorSpinor(const colorspinor_wrapper<T,S> &a) {
    a.field.load((T*)data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline void ColorSpinor<T,Nc,4>::operator=(const colorspinor_wrapper<T,S> &a) {
    a.field.load((T*)data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline ColorSpinor<T,Nc,4>::ColorSpinor(const colorspinor_wrapper<T,S> &a) {
    a.field.load((T*)data, a.x_cb, a.parity);
  }

    /**
       @brief colorspinor_ghost_wrapper is an internal class that is
       used to wrap instances of colorspinor accessors, currying in a
       specifc location on the field.  The Ghost() accessors in
       colorspinor-field accessors return instances to this class,
       allowing us to then use operator overloading upon this class to
       interact with the ColorSpinor class.  As a result we can
       include colorspinor-field accessors directly in ColorSpinor
       expressions in kernels without having to declare temporaries
       with explicit calls to the loadGhost/saveGhost methods in the
       colorspinor-field accessors.
    */
  template <typename Float, typename T>
    struct colorspinor_ghost_wrapper {
      const int dim;
      const int dir;
      const int ghost_idx;
      const int parity;
      T &field;

      /**
	 @brief colorspinor_ghost_wrapper constructor
	 @param[in] a colorspinor field accessor we are wrapping
	 @param[in] dim Dimension of the ghost we are accessing
	 @param[in] dir Direction of the ghost we are accessing
	 @param[in] ghost_idx Checkerboarded space-time ghost index we are accessing
	 @param[in] parity Parity we are accessing
       */
      __device__ __host__ inline colorspinor_ghost_wrapper<Float,T>(T &field, int dim, int dir, int ghost_idx, int parity)
	: field(field), dim(dim), dir(dir), ghost_idx(ghost_idx), parity(parity) { }

      /**
	 @brief Assignment operator with Matrix instance as input
	 @param[in] C ColorSpinor we want to store in this accessot
       */
      template<typename C>
      __device__ __host__ inline void operator=(const C &a) {
	field.saveGhost((Float*)a.data, ghost_idx, dim, dir, parity);
      }
    };

  template <typename T, int Nc, int Ns>
    template <typename S>
    __device__ __host__ inline void ColorSpinor<T,Nc,Ns>::operator=(const colorspinor_ghost_wrapper<T,S> &a) {
    a.field.loadGhost((T*)data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  template <typename T, int Nc, int Ns>
    template <typename S>
    __device__ __host__ inline ColorSpinor<T,Nc,Ns>::ColorSpinor(const colorspinor_ghost_wrapper<T,S> &a) {
    a.field.loadGhost((T*)data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline void ColorSpinor<T,Nc,4>::operator=(const colorspinor_ghost_wrapper<T,S> &a) {
    a.field.loadGhost((T*)data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline ColorSpinor<T,Nc,4>::ColorSpinor(const colorspinor_ghost_wrapper<T,S> &a) {
    a.field.loadGhost((T*)data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  namespace colorspinor {

    template<typename ReduceType, typename Float> struct square { __host__ __device__ ReduceType operator()(quda::complex<Float> x) { return static_cast<ReduceType>(norm(x)); } };

    template<typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order> struct AccessorCB { 
      AccessorCB(const ColorSpinorField &) { errorQuda("Not implemented"); }
      __device__ __host__ inline int index(int parity, int x_cb, int s, int c, int v) const { return 0; }
    };

    template<typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order> struct GhostAccessorCB {
      GhostAccessorCB(const ColorSpinorField &) { errorQuda("Not implemented"); }
      __device__ __host__ inline int index(int dim, int dir, int parity, int x_cb, int s, int c, int v) const
      { return 0; }
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
      GhostAccessorCB(const ColorSpinorField &a, int nFace = 1) {
	for (int d=0; d<4; d++) {
	  ghostOffset[d] = nFace*a.SurfaceCB(d)*nColor*nSpin*nVec;
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
      GhostAccessorCB(const ColorSpinorField &a, int nFace = 1) {
	for (int d=0; d<4; d++) {
	  faceVolumeCB[d] = nFace*a.SurfaceCB(d);
	  ghostOffset[d] = faceVolumeCB[d]*nColor*nSpin*nVec;
	}
      }
      __device__ __host__ inline int index(int dim, int dir, int parity, int x_cb, int s, int c, int v) const
      { return parity*ghostOffset[dim] + ((s*nColor+c)*nVec+v)*faceVolumeCB[dim] + x_cb; }
    };


    template <typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order>
      class FieldOrderCB {

    protected:
      complex<Float> *v;
      mutable complex<Float> *ghost[8];
      mutable int x[QUDA_MAX_DIM];
      const int volumeCB;
      const int nDim;
      const QudaGammaBasis gammaBasis;
      const AccessorCB<Float,nSpin,nColor,nVec,order> accessor;
      const GhostAccessorCB<Float,nSpin,nColor,nVec,order> ghostAccessor;
      const int siteSubset;
      const int nParity;
      const QudaFieldLocation location;

    public:
      /** 
       * Constructor for the FieldOrderCB class
       * @param field The field that we are accessing
       */
    FieldOrderCB(const ColorSpinorField &field, int nFace=1, void *v_=0, void **ghost_=0)
      : v(v_? static_cast<complex<Float>*>(const_cast<void*>(v_))
	  : static_cast<complex<Float>*>(const_cast<void*>(field.V()))),
	volumeCB(field.VolumeCB()),
	nDim(field.Ndim()), gammaBasis(field.GammaBasis()),
	siteSubset(field.SiteSubset()), nParity(field.SiteSubset()),
	location(field.Location()), accessor(field), ghostAccessor(field,nFace)
      { 
	for (int d=0; d<4; d++) {
	  void * const *_ghost = ghost_ ? ghost_ : field.Ghost();
	  ghost[2*d+0] = static_cast<complex<Float>*>(_ghost[2*d+0]);
	  ghost[2*d+1] = static_cast<complex<Float>*>(_ghost[2*d+1]);
	}

	for (int d=0; d<QUDA_MAX_DIM; d++) x[d]=field.X(d);
      }

      /**
       * Destructor for the FieldOrderCB class
       */
      virtual ~FieldOrderCB() { ; }

      void resetGhost(void * const *ghost_) const
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

      /**
       * Returns the L2 norm squared of the field in a given dimension
       * @return L2 norm squared
      */
      __host__ double norm2() const {
	double nrm2 = 0;
	if (location == QUDA_CUDA_FIELD_LOCATION) {
	  thrust::device_ptr<complex<Float> > ptr(v);
	  nrm2 = thrust::transform_reduce(ptr, ptr+nParity*volumeCB*nSpin*nColor*nVec,
					  square<double,Float>(), 0.0, thrust::plus<double>());
	} else {
	  nrm2 = thrust::transform_reduce(thrust::seq, v, v+nParity*volumeCB*nSpin*nColor*nVec,
					  square<double,Float>(), 0.0, thrust::plus<double>());
	}
	comm_allreduce(&nrm2);
	return nrm2;
      }

      size_t Bytes() const { return nParity * static_cast<size_t>(volumeCB) * nColor * nSpin * nVec * 2ll * sizeof(Float); }
    };

    /**
       @brief Accessor routine for ColorSpinorFields in native field order.
       @tparam Float Underlying storage data type of the field
       @tparam Ns Number of spin components
       @tparam Nc Number of colors
       @tparam N Number of real numbers per short vector
       @tparam huge_alloc Template parameter that enables 64-bit
       pointer arithmetic for huge allocations (e.g., packed set of
       vectors).  Default is to use 32-bit pointer arithmetic.
     */
    template <typename Float, int Ns, int Nc, int N, bool huge_alloc=false>
      struct FloatNOrder {
	typedef typename mapper<Float>::type RegType;
	typedef typename VectorType<Float,N>::type Vector;
	typedef typename VectorType<RegType,N>::type RegVector;
	typedef typename AllocType<huge_alloc>::type AllocInt;
	static const int length = 2 * Ns * Nc;
	static const int M = length / N;
	Float *field;
	float *norm;
	const AllocInt offset; // offset can be 32-bit or 64-bit
	const AllocInt norm_offset;
#ifdef USE_TEXTURE_OBJECTS
	typedef typename TexVectorType<RegType,N>::type TexVector;
	cudaTextureObject_t tex;
	cudaTextureObject_t texNorm;
	const int tex_offset;
#endif
	int volumeCB;
	int faceVolumeCB[4];
	int stride;
	Float *ghost[8];
	int nParity;
	void *backup_h; //! host memory for backing up the field when tuning
	size_t bytes;

      FloatNOrder(const ColorSpinorField &a, int nFace=1, Float *field_=0, float *norm_=0, Float **ghost_=0, bool override=false)
      : field(field_ ? field_ : (Float*)a.V()), offset(a.Bytes()/(2*sizeof(Float))),
	  norm(norm_ ? norm_ : (float*)a.Norm()), norm_offset(a.NormBytes()/(2*sizeof(float))),
#ifdef USE_TEXTURE_OBJECTS
	  tex(0), texNorm(0), tex_offset(offset/N),
#endif
	  volumeCB(a.VolumeCB()), stride(a.Stride()), nParity(a.SiteSubset()), backup_h(nullptr), bytes(a.Bytes())
	{
	  for (int i=0; i<4; i++) {
	    ghost[2*i+0] = ghost_ ? ghost_[2*i+0] : static_cast<Float*>(a.Ghost()[2*i+0]);
	    ghost[2*i+1] = ghost_ ? ghost_[2*i+1] : static_cast<Float*>(a.Ghost()[2*i+1]);
	    faceVolumeCB[i] = a.SurfaceCB(i)*nFace;
	  }
#ifdef USE_TEXTURE_OBJECTS
	  if (a.Location() == QUDA_CUDA_FIELD_LOCATION) {
	    tex = static_cast<const cudaColorSpinorField&>(a).Tex();
	    texNorm = static_cast<const cudaColorSpinorField&>(a).TexNorm();
	  }
	  if (!huge_alloc && (this->field != a.V() || (a.Precision() == QUDA_HALF_PRECISION && this->norm != a.Norm()) ) && !override) {
	    errorQuda("Cannot use texture read since data pointer does not equal field pointer - use with huge_alloc=true instead");
	  }
#endif
	}
	virtual ~FloatNOrder() { ; }

	__device__ __host__ inline void load(RegType v[length], int x, int parity=0) const {
#pragma unroll
	  for (int i=0; i<M; i++) {
	    // first do vectorized copy from memory
#if defined(USE_TEXTURE_OBJECTS) && defined(__CUDA_ARCH__)
	    if (!huge_alloc) { // use textures unless we have a huge alloc
	      TexVector vecTmp = tex1Dfetch<TexVector>(tex, parity*tex_offset + stride*i + x);
	      copy(reinterpret_cast<RegVector*>(v)[i], vecTmp);
	    } else
#endif
	    {
	      Vector vecTmp = vector_load<Vector>(field + parity*offset, x + stride*i);
	      copy(reinterpret_cast<RegVector*>(v)[i], vecTmp);
	    }
	  }

	  if (sizeof(Float)==sizeof(short)) {
#if defined(USE_TEXTURE_OBJECTS) && defined(__CUDA_ARCH__)
	    // use textures unless we have a large alloc
	    RegType nrm = !huge_alloc ? tex1Dfetch<float>(texNorm,x+parity*norm_offset) : norm[x+parity*norm_offset];
#else
            RegType nrm = norm[x+parity*norm_offset];
#endif
#pragma unroll
	    for (int i=0; i<length; i++) v[i] *= nrm;
	  }
	}

	__device__ __host__ inline void save(const RegType v[length], int x, int parity=0) {
	  RegType scale = 0.0;
	  RegType tmp[length];

	  if (sizeof(Float)==sizeof(short)) {
#pragma unroll
	    for (int i=0; i<length; i++) scale = fabs(v[i]) > scale ? fabs(v[i]) : scale;
	    norm[x+parity*norm_offset] = scale;
	  }

	  if (sizeof(Float)==sizeof(short)) {
	    RegType scale_inv = static_cast<RegType>(1.0) / scale;
#pragma unroll
	    for (int i=0; i<length; i++) tmp[i] = v[i] * scale_inv;
	  } else {
#pragma unroll
	    for (int i=0; i<length; i++) tmp[i] = v[i];
	  }
#pragma unroll
	  for (int i=0; i<M; i++) {
	    Vector vecTmp;
	    // first do vectorized copy converting into storage type
	    copy(vecTmp, reinterpret_cast<RegVector*>(tmp)[i]);
	    // second do vectorized copy into memory
	    reinterpret_cast< Vector* >(field + parity*offset)[x + stride*i] = vecTmp;
	  }
	}

	/**
	   @brief This accessor routine returns a colorspinor_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline colorspinor_wrapper<RegType,FloatNOrder<Float,Ns,Nc,N> >
	  operator()(int x_cb, int parity) {
	  return colorspinor_wrapper<RegType,FloatNOrder<Float,Ns,Nc,N> >(*this, x_cb, parity);
	}

	/**
	   @brief This accessor routine returns a const colorspinor_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline const colorspinor_wrapper<RegType,FloatNOrder<Float,Ns,Nc,N> >
	  operator()(int x_cb, int parity) const {
	  return colorspinor_wrapper<RegType,FloatNOrder<Float,Ns,Nc,N> >
	    (const_cast<FloatNOrder<Float,Ns,Nc,N>&>(*this), x_cb, parity);
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

	/**
	   @brief This accessor routine returns a colorspinor_ghost_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] dim Dimensions of the ghost we are requesting
	   @param[in] ghost_idx Checkerboarded space-time ghost index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_ghost_wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline colorspinor_ghost_wrapper<Float,FloatNOrder<Float,Ns,Nc,N> >
	  Ghost(int dim, int dir, int ghost_idx, int parity) {
	  return colorspinor_ghost_wrapper<Float,FloatNOrder<Float,Ns,Nc,N> >(*this, dim, dir, ghost_idx, parity);
	}

	/**
	   @brief This accessor routine returns a const
	   colorspinor_ghost_wrapper to this object, allowing us to
	   overload various operators for manipulating at the site
	   level interms of matrix operations.
	   @param[in] dim Dimensions of the ghost we are requesting
	   @param[in] ghost_idx Checkerboarded space-time ghost index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_ghost+wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline const colorspinor_ghost_wrapper<Float,FloatNOrder<Float,Ns,Nc,N> >
	  Ghost(int dim, int dir, int ghost_idx, int parity) const {
	  return colorspinor_ghost_wrapper<Float,FloatNOrder<Float,Ns,Nc,N> >
	    (const_cast<FloatNOrder<Float,Ns,Nc,N>&>(*this), dim, dir, ghost_idx, parity);
	}

	/**
	   @brief Backup the field to the host when tuning
	*/
	void save() {
	  if (backup_h) errorQuda("Already allocated host backup");
	  backup_h = safe_malloc(bytes);
	  cudaMemcpy(backup_h, field, bytes, cudaMemcpyDeviceToHost);
	  checkCudaError();
	}

	/**
	   @brief Restore the field from the host after tuning
	*/
	void load() {
	  cudaMemcpy(field, backup_h, bytes, cudaMemcpyHostToDevice);
	  host_free(backup_h);
	  backup_h = nullptr;
	  checkCudaError();
	}

	size_t Bytes() const { return nParity * volumeCB * (Nc * Ns * 2 * sizeof(Float) + (typeid(Float) == typeid(short) ? sizeof(float) : 0)); }
      };

    /**
       @brief This is just a dummy structure we use for trove to define the
       required structure size
       @tparam real Real number type
       @tparam length Number of elements in the structure
    */
    template <typename real, int length> struct S { real v[length]; };

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
      SpaceColorSpinorOrder(const ColorSpinorField &a, int nFace=1, Float *field_=0, float *dummy=0, Float **ghost_=0)
      : field(field_ ? field_ : (Float*)a.V()), offset(a.Bytes()/(2*sizeof(Float))),
	  volumeCB(a.VolumeCB()), stride(a.Stride()), nParity(a.SiteSubset())
	{
	  if (volumeCB != stride) errorQuda("Stride must equal volume for this field order");
	  for (int i=0; i<4; i++) {
	    ghost[2*i] = ghost_ ? ghost_[2*i] : 0;
	    ghost[2*i+1] = ghost_ ? ghost_[2*i+1] : 0;
	    faceVolumeCB[i] = a.SurfaceCB(i)*nFace;
	  }
	}
	virtual ~SpaceColorSpinorOrder() { ; }

	__device__ __host__ inline void load(RegType v[length], int x, int parity=0) const {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	  typedef S<Float,length> structure;
	  trove::coalesced_ptr<structure> field_((structure*)field);
	  structure v_ = field_[parity*volumeCB + x];
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = (RegType)v_.v[(c*Ns + s)*2 + z];
	      }
	    }
	  }
#else
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = field[parity*offset + ((x*Nc + c)*Ns + s)*2 + z];
	      }
	    }
	  }
#endif
	}

	__device__ __host__ inline void save(const RegType v[length], int x, int parity=0) {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	  typedef S<Float,length> structure;
	  trove::coalesced_ptr<structure> field_((structure*)field);
	  structure v_;
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v_.v[(c*Ns + s)*2 + z] = (Float)v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	  field_[parity*volumeCB + x] = v_;
#else
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		field[parity*offset + ((x*Nc + c)*Ns + s)*2 + z] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
#endif
	}

	/**
	   @brief This accessor routine returns a colorspinor_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline colorspinor_wrapper<RegType,SpaceColorSpinorOrder<Float,Ns,Nc> >
	  operator()(int x_cb, int parity) {
	  return colorspinor_wrapper<RegType,SpaceColorSpinorOrder<Float,Ns,Nc> >(*this, x_cb, parity);
	}

	/**
	   @brief This accessor routine returns a const colorspinor_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline const colorspinor_wrapper<RegType,SpaceColorSpinorOrder<Float,Ns,Nc> >
	  operator()(int x_cb, int parity) const {
	  return colorspinor_wrapper<RegType,SpaceColorSpinorOrder<Float,Ns,Nc> >
	    (const_cast<SpaceColorSpinorOrder<Float,Ns,Nc>&>(*this), x_cb, parity);
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
      SpaceSpinorColorOrder(const ColorSpinorField &a, int nFace=1, Float *field_=0, float *dummy=0, Float **ghost_=0)
      : field(field_ ? field_ : (Float*)a.V()), offset(a.Bytes()/(2*sizeof(Float))),
	  volumeCB(a.VolumeCB()), stride(a.Stride()), nParity(a.SiteSubset())
	{
	  if (volumeCB != stride) errorQuda("Stride must equal volume for this field order");
	  for (int i=0; i<4; i++) {
	    ghost[2*i] = ghost_ ? ghost_[2*i] : 0;
	    ghost[2*i+1] = ghost_ ? ghost_[2*i+1] : 0;
	    faceVolumeCB[i] = a.SurfaceCB(i)*nFace;
	  }
	}
	virtual ~SpaceSpinorColorOrder() { ; }

	__device__ __host__ inline void load(RegType v[length], int x, int parity=0) const {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	  typedef S<Float,length> structure;
	  trove::coalesced_ptr<structure> field_((structure*)field);
	  structure v_ = field_[parity*volumeCB + x];
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = (RegType)v_.v[(s*Nc + c)*2 + z];
	      }
	    }
	  }
#else
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = field[parity*offset + ((x*Ns + s)*Nc + c)*2 + z];
	      }
	    }
	  }
#endif
	}

	__device__ __host__ inline void save(const RegType v[length], int x, int parity=0) {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	  typedef S<Float,length> structure;
	  trove::coalesced_ptr<structure> field_((structure*)field);
	  structure v_;
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v_.v[(s*Nc + c)*2 + z] = (Float)v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	  field_[parity*volumeCB + x] = v_;
#else
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		field[parity*offset + ((x*Ns + s)*Nc + c)*2 + z] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
#endif
	}

	/**
	   @brief This accessor routine returns a colorspinor_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline colorspinor_wrapper<RegType,SpaceSpinorColorOrder<Float,Ns,Nc> >
	  operator()(int x_cb, int parity) {
	  return colorspinor_wrapper<RegType,SpaceSpinorColorOrder<Float,Ns,Nc> >(*this, x_cb, parity);
	}

	/**
	   @brief This accessor routine returns a const colorspinor_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline const colorspinor_wrapper<RegType,SpaceSpinorColorOrder<Float,Ns,Nc> >
	  operator()(int x_cb, int parity) const {
	  return colorspinor_wrapper<RegType,SpaceSpinorColorOrder<Float,Ns,Nc> >
	    (const_cast<SpaceSpinorColorOrder<Float,Ns,Nc>&>(*this), x_cb, parity);
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

    // custom accessor for TIFR z-halo padded arrays
    template <typename Float, int Ns, int Nc>
      struct PaddedSpaceSpinorColorOrder {
	typedef typename mapper<Float>::type RegType;
	static const int length = 2 * Ns * Nc;
	Float *field;
	size_t offset;
	Float *ghost[8];
	int volumeCB;
	int exVolumeCB;
	int faceVolumeCB[4];
	int stride;
	int nParity;
	int dim[4]; // full field dimensions
	int exDim[4]; // full field dimensions
      PaddedSpaceSpinorColorOrder(const ColorSpinorField &a, int nFace=1, Float *field_=0, float *dummy=0, Float **ghost_=0)
      : field(field_ ? field_ : (Float*)a.V()),
	  volumeCB(a.VolumeCB()), exVolumeCB(1), stride(a.Stride()), nParity(a.SiteSubset()),
	  dim{ a.X(0), a.X(1), a.X(2), a.X(3)}, exDim{ a.X(0), a.X(1), a.X(2) + 4, a.X(3)}
	{
	  if (volumeCB != stride) errorQuda("Stride must equal volume for this field order");
	  for (int i=0; i<4; i++) {
	    ghost[2*i] = ghost_ ? ghost_[2*i] : 0;
	    ghost[2*i+1] = ghost_ ? ghost_[2*i+1] : 0;
	    faceVolumeCB[i] = a.SurfaceCB(i)*nFace;
	    exVolumeCB *= exDim[i];
	  }
	  exVolumeCB /= nParity;
	  dim[0] *= (nParity == 1) ? 2 : 1; // need to full dimensions
	  exDim[0] *= (nParity == 1) ? 2 : 1; // need to full dimensions

	  offset = (exVolumeCB*Ns*Nc*2) / 2; // compute manually since Bytes is likely wrong due to z-padding
	}
	virtual ~PaddedSpaceSpinorColorOrder() { ; }

	/**
	   @brief Compute the index into the padded field.  Assumes that
	   parity doesn't change from unpadded to padded.
	*/
	__device__ __host__ int getPaddedIndex(int x_cb, int parity) const {
	  // find coordinates
	  int coord[4];
	  getCoords(coord, x_cb, dim, parity);

	  // get z-extended index
	  coord[2] += 2; // offset for halo
	  return linkIndex(coord, exDim);
	}

	__device__ __host__ inline void load(RegType v[length], int x, int parity=0) const {
	  int y = getPaddedIndex(x, parity);

#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	  typedef S<Float,length> structure;
	  trove::coalesced_ptr<structure> field_((structure*)field);
	  structure v_ = field_[parity*exVolumeCB + y];
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = (RegType)v_.v[(s*Nc + c)*2 + z];
	      }
	    }
	  }
#else
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = field[parity*offset + ((y*Ns + s)*Nc + c)*2 + z];
	      }
	    }
	  }
#endif
	}

	__device__ __host__ inline void save(const RegType v[length], int x, int parity=0) {
	  int y = getPaddedIndex(x, parity);

#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	  typedef S<Float,length> structure;
	  trove::coalesced_ptr<structure> field_((structure*)field);
	  structure v_;
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v_.v[(s*Nc + c)*2 + z] = (Float)v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	  field_[parity*exVolumeCB + y] = v_;
#else
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		field[parity*offset + ((y*Ns + s)*Nc + c)*2 + z] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
#endif
	}

	/**
	   @brief This accessor routine returns a colorspinor_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline colorspinor_wrapper<RegType,PaddedSpaceSpinorColorOrder<Float,Ns,Nc> >
	  operator()(int x_cb, int parity) {
	  return colorspinor_wrapper<RegType,PaddedSpaceSpinorColorOrder<Float,Ns,Nc> >(*this, x_cb, parity);
	}

	/**
	   @brief This accessor routine returns a const colorspinor_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline const colorspinor_wrapper<RegType,PaddedSpaceSpinorColorOrder<Float,Ns,Nc> >
	  operator()(int x_cb, int parity) const {
	  return colorspinor_wrapper<RegType,PaddedSpaceSpinorColorOrder<Float,Ns,Nc> >
	    (const_cast<PaddedSpaceSpinorColorOrder<Float,Ns,Nc>&>(*this), x_cb, parity);
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
      QDPJITDiracOrder(const ColorSpinorField &a, int nFace=1, Float *field_=0)
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

	__device__ __host__ inline void save(const RegType v[Ns*Nc*2], int x, int parity=1) {
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		field[(((z*Nc + c)*Ns + s)*2 + parity)*volumeCB + x] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	}

	/**
	   @brief This accessor routine returns a colorspinor_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline colorspinor_wrapper<RegType,QDPJITDiracOrder<Float,Ns,Nc> >
	  operator()(int x_cb, int parity) {
	  return colorspinor_wrapper<RegType,QDPJITDiracOrder<Float,Ns,Nc> >(*this, x_cb, parity);
	}

	/**
	   @brief This accessor routine returns a const colorspinor_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
	__device__ __host__ inline const colorspinor_wrapper<RegType,QDPJITDiracOrder<Float,Ns,Nc> >
	  operator()(int x_cb, int parity) const {
	  return colorspinor_wrapper<RegType,QDPJITDiracOrder<Float,Ns,Nc> >
	    (const_cast<QDPJITDiracOrder<Float,Ns,Nc>&>(*this), x_cb, parity);
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
