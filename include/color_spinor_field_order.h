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

    template <typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order>
      class FieldOrder {

    protected:
      complex<Float> *v;
      mutable int x[QUDA_MAX_DIM];
      const int volume;
      const int volumeCB;
      const int stride;
      const int nDim;
      const QudaGammaBasis gammaBasis;
      const QudaSiteSubset siteSubset;
      const size_t cb_offset;

      complex<Float> dummy;

    public:
      /** 
       * Constructor for the FieldOrder class
       * @param field The field that we are accessing
       */
    FieldOrder(const ColorSpinorField &field) 
      : v(static_cast<complex<Float>*>(const_cast<void*>(field.V()))), 
	volume(field.Volume()), volumeCB(field.VolumeCB()), stride(field.Stride()),
	nDim(field.Ndim()), gammaBasis(field.GammaBasis()), 
	siteSubset(field.SiteSubset()), cb_offset(field.Bytes()>>1), dummy(0.0,0.0)
      { for (int d=0; d<QUDA_MAX_DIM; d++) x[d]=field.X(d); }

      /**
       * Destructor for the FieldOrder class
       */
      virtual ~FieldOrder() { ; }

      /**
       * Read-only complex-member accessor function
       * @param x 1-d site index
       * @param s spin index
       * @param c color index
       */
      __device__ __host__ const complex<Float>& operator()(int x, int s, int c) const {
#ifndef __CUDA_ARCH__
	errorQuda("Not implemented");
#endif
	return complex<Float>(0.0,0.0);
      }

      /**
       * Writable complex-member accessor function
       * @param x 1-d site index
       * @param s spin index
       * @param c color index
       */
      __device__ __host__ inline complex<Float>& operator()(int x, int s, int c) {
#ifndef __CUDA_ARCH__
	errorQuda("Not implemented");
#endif
	return dummy;
      }

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

      /**
       * Specialized read-only complex-member accessor function (for mg prolongator)
       * @param x 1-d site index
       * @param s spin index
       * @param c color index
       * @param n vector number
       */
      __device__ __host__ inline const complex<Float>& operator()(int x, int s, int c, int n) const {
	return (*this)(x, s, c*nVec + n);      
      }

      /**
       * Specialized writable complex-member accessor function (for mg prolongator)
       * @param x 1-d site index
       * @param s spin index
       * @param c color index
       * @param n vector number
       */
      __device__ __host__ inline complex<Float>& operator()(int x, int s, int c, int n) {
	return (*this)(x, s, c*nVec + n);      
      }


      /** Return the length of dimension d */
      __device__ __host__ inline int X(int d) const { return x[d]; }

      /** Returns the number of field colors */
       __device__ __host__ inline int Ncolor() const { return nColor; }

      /** Returns the number of field spins */
      __device__ __host__ inline int Nspin() const { return nSpin; }

      /** Returns the field volume */
      __device__ __host__ inline int Volume() const { return volume; }

      /** Returns the field geometric dimension */
      __device__ __host__ inline int Ndim() const { return nDim; }

      /** Returns the field geometric dimension */
      __device__ __host__ inline QudaGammaBasis GammaBasis() const { return gammaBasis; }

      /** Returns the field geometric dimension */
      __device__ __host__ inline int SiteSubset() const { return siteSubset; }

      /** Returns the number of packed vectors (for mg prolongator) */
      __device__ __host__ inline int NvecPacked() const { return nVec; }

      /** Returns the number of packed colors (for mg prolongator) */
      __device__ __host__ inline int NcolorPacked() const { return nColor / nVec; }

      /** Returns the number of packed spins (for mg prolongator) */
      __device__ __host__ inline int NspinPacked() const { return nSpin; }    

    };

    template<> __device__ __host__ inline const complex<double>& 
      FieldOrder<double,4,3,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>::operator()(int x, int s, int c) const {
      return v[(x*4+s)*3+c];
    }
    
    template<> __device__ __host__ inline complex<double>&
      FieldOrder<double,4,3,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>::operator()(int x, int s, int c) {
      return v[(x*4+s)*3+c];
    }

    template<> __device__ __host__ inline const complex<float>&
      FieldOrder<float,4,3,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>::operator()(int x, int s, int c) const {
      return v[(x*4+s)*3+c];
    }
    
    template<> __device__ __host__ inline complex<float>&
      FieldOrder<float,4,3,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>::operator()(int x, int s, int c) {
      return v[(x*4+s)*3+c];
    }

    template<typename Float, int nSpin, int nColor, int N>
      inline const complex<Float>& operatorFloatN(int x, int s, int c, int volumeCB, int stride, int cb_offset, const complex<Float> *v) {
      int x_cb = (x >= volumeCB) ? x-volumeCB : x;
      int parity = (x >= volumeCB) ? 1 : 0;
      int j = ((s*nColor+c)*2) / N; // factor of two for complexity
      int i = ((s*nColor+c)*2) % N;      
      int index = ((j*stride+x_cb)*N+i) / 2; // back to a complex offset
      return *(reinterpret_cast<complex<double>*>(reinterpret_cast<char*>(const_cast<complex<Float>*>(v) + index) + parity*cb_offset));
    }

    template<> __device__ __host__ inline const complex<double>& 
      FieldOrder<double,4,3,1,QUDA_FLOAT2_FIELD_ORDER>::operator()(int x, int s, int c) const {
      return operatorFloatN<double,4,3,QUDA_FLOAT2_FIELD_ORDER>(x,s,c,volumeCB,stride,cb_offset,v);
    }

    template<> __device__ __host__ inline complex<double>& 
      FieldOrder<double,4,3,1,QUDA_FLOAT2_FIELD_ORDER>::operator()(int x, int s, int c)  { 
      return const_cast<complex<double>&>(operatorFloatN<double,4,3,QUDA_FLOAT2_FIELD_ORDER>(x,s,c,volumeCB,stride,cb_offset,v));
    }
    
    /*// FIXME remove this factory and propagate the use of traits instead
 template <typename Float>
      FieldOrder<Float>* createOrder(const ColorSpinorField &a, int nVec=1) {
      FieldOrder<Float>* ptr=0;

      if (a.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
	//ptr = new FloatNOrder2<Float,2>(a, nVec);
      } else if (a.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
	//ptr = new FloatNOrder2<Float,4>(a, nVec);
      } else if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	ptr = new SpaceSpinColorOrder<Float>(a, nVec);
      } else if (a.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
	ptr = new SpaceColorSpinOrder<Float>(a, nVec);
      } else if (a.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
	ptr = new QOPDomainWallOrder<Float>(a, nVec);
      } else {
	errorQuda("Order %d not supported in ColorSpinorField", a.FieldOrder());
      }

      return ptr;
      }
    */
    /**
       This traits-driven object creation replaces the factory approach above.
     */
    /*template<typename Float,QudaFieldOrder> struct accessor { };

    template<typename Float> struct accessor<Float,QUDA_FLOAT2_FIELD_ORDER> 
      { typedef FloatNOrder2<Float,2> type; };

    template<typename Float> struct accessor<Float,QUDA_FLOAT4_FIELD_ORDER> 
      { typedef FloatNOrder2<Float,4> type; };

    template<typename Float> struct accessor<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> 
      { typedef SpaceSpinColorOrder<Float> type; };

    template<typename Float> struct accessor<Float,QUDA_SPACE_COLOR_SPIN_FIELD_ORDER> 
      { typedef SpaceColorSpinOrder<Float> type; };

    template<typename Float> struct accessor<Float,QUDA_QOP_DOMAIN_WALL_FIELD_ORDER> 
    { typedef QOPDomainWallOrder<Float> type; };*/

    template <typename Float, int Ns, int Nc, int N>
      struct FloatNOrder {
	typedef typename mapper<Float>::type RegType;
	Float *field;
	float *norm;
	int volumeCB;
	int stride;
      FloatNOrder(const ColorSpinorField &a, Float *field_=0, float *norm_=0)
      : field(field_ ? field_ : (Float*)a.V()), norm(norm_ ? norm_ : (float*)a.Norm()), 
	  volumeCB(a.VolumeCB()), stride(a.Stride()) { ; }
	virtual ~FloatNOrder() { ; }

	__device__ __host__ inline void load(RegType v[Ns*Nc*2], int x) const {
	  if (x >= volumeCB) return;
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		int internal_idx = (s*Nc + c)*2 + z;
		int pad_idx = internal_idx / N;
		copy(v[(s*Nc+c)*2+z], field[(pad_idx * stride + x)*N + internal_idx % N]);
		if (sizeof(Float)==sizeof(short)) v[(s*Nc+c)*2+z] *= norm[x];
	      }
	    }
	  }
	}

	__device__ __host__ inline void save(const RegType v[Ns*Nc*2], int x) {
	  if (x >= volumeCB) return;
	  RegType scale = 0.0;
	  if (sizeof(Float)==sizeof(short)) {
	    for (int i=0; i<2*Ns*Nc; i++) scale = fabs(v[i]) > scale ? fabs(v[i]) : scale;
	    norm[x] = scale;
	  }

	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		int internal_idx = (s*Nc + c)*2 + z;
		int pad_idx = internal_idx / N;
		if (sizeof(Float)==sizeof(short))
		  copy(field[(pad_idx * stride + x)*N + internal_idx % N], v[(s*Nc+c)*2+z] / scale);
		else
		  copy(field[(pad_idx * stride + x)*N + internal_idx % N], v[(s*Nc+c)*2+z]);
	      }
	    }
	  }
	}

	__device__ __host__ const RegType& operator()(int x, int s, int c, int z) const {
	  int internal_idx = (s*Nc + c)*2 + z;
	  int pad_idx = internal_idx / N;    
	  return field[(pad_idx * stride + x)*N + internal_idx % N];
	}

	__device__ __host__ RegType& operator()(int x, int s, int c, int z) {
	  int internal_idx = (s*Nc + c)*2 + z;
	  int pad_idx = internal_idx / N;    
	  return field[(pad_idx * stride + x)*N + internal_idx % N];
	}

	size_t Bytes() const { return volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };

    /**! float4 load specialization to obtain full coalescing. */
    template<> __device__ inline void FloatNOrder<float, 4, 3, 4>::load(float v[24], int x) const {
      if (x >= volumeCB) return;
#pragma unroll
      for (int i=0; i<4*3*2; i+=4) {
	float4 tmp = ((float4*)field)[i/4 * stride + x];
	v[i] = tmp.x; v[i+1] = tmp.y; v[i+2] = tmp.z; v[i+3] = tmp.w;
      }
    }

    /**! float4 save specialization to obtain full coalescing. */
    template<> __device__ inline void FloatNOrder<float, 4, 3, 4>::save(const float v[24], int x) {
      if (x >= volumeCB) return;
#pragma unroll
      for (int i=0; i<4*3*2; i+=4) {
	float4 tmp = make_float4(v[i], v[i+1], v[i+2], v[i+3]);
	((float4*)field)[i/4 * stride + x] = tmp;
      }
    }

    template <typename Float, int Ns, int Nc>
      struct SpaceColorSpinorOrder {
	typedef typename mapper<Float>::type RegType;
	Float *field;
	int volumeCB;
	int stride;
      SpaceColorSpinorOrder(const ColorSpinorField &a, Float *field_=0) 
      : field(field_ ? field_ : (Float*)a.V()), volumeCB(a.VolumeCB()), stride(a.Stride()) 
	{ if (volumeCB != stride) errorQuda("Stride must equal volume for this field order"); }
	virtual ~SpaceColorSpinorOrder() { ; }

	__device__ __host__ inline void load(RegType v[Ns*Nc*2], int x) const {
	  if (x >= volumeCB) return;
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = field[((x*Nc + c)*Ns + s)*2 + z]; 
	      }
	    }
	  }
	}

	__device__ __host__ inline void save(const RegType v[Ns*Nc*2], int x) {
	  if (x >= volumeCB) return;
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		field[((x*Nc + c)*Ns + s)*2 + z] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	}

	__device__ __host__ const RegType& operator()(int x, int s, int c, int z) const {
	  return field[((x*Nc + c)*Ns + s)*2 + z];
	}

	__device__ __host__ RegType& operator()(int x, int s, int c, int z) {
	  return field[((x*Nc + c)*Ns + s)*2 + z];
	}

	size_t Bytes() const { return volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };

    template <typename Float, int Ns, int Nc>
      __device__ inline void load_shared(typename mapper<Float>::type v[Ns*Nc*2], Float *field, int x, int volume) {
      const int tid = threadIdx.x;
      const int vec_length = Ns*Nc*2;

      // the length of the block on the last grid site might not extend to all threads
      const int block_dim = (blockIdx.x == gridDim.x-1) ? 
	volume - (gridDim.x-1)*blockDim.x : blockDim.x;

      extern __shared__ typename mapper<Float>::type s_data[];

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
      __device__ inline void save_shared(Float *field, const typename mapper<Float>::type v[Ns*Nc*2], int x, int volumeCB) {
      const int tid = threadIdx.x;
      const int vec_length = Ns*Nc*2;

      // the length of the block on the last grid site might not extend to all threads
      const int block_dim = (blockIdx.x == gridDim.x-1) ? 
	volumeCB - (gridDim.x-1)*blockDim.x : blockDim.x;

      extern __shared__ typename mapper<Float>::type s_data[];

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
    template<> __host__ __device__ inline void SpaceColorSpinorOrder<float, 4, 3>::load(float v[24], int x) const {
#ifdef __CUDA_ARCH__
      load_shared<float, 4, 3>(v, field, x, volumeCB);
#else
      if (x >= volumeCB) return;
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
    template<> __host__ __device__ inline void SpaceColorSpinorOrder<float, 4, 3>::save(const float v[24], int x) {
#ifdef __CUDA_ARCH__
      save_shared<float, 4, 3>(field, v, x, volumeCB);
#else
      if (x >= volumeCB) return;
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
	typedef typename mapper<Float>::type RegType;
	Float *field;
	int volumeCB;
	int stride;
      SpaceSpinorColorOrder(const ColorSpinorField &a, Float *field_=0) 
      : field(field_ ? field_ : (Float*)a.V()), volumeCB(a.VolumeCB()), stride(a.Stride())
	{ if (volumeCB != stride) errorQuda("Stride must equal volume for this field order"); }
	virtual ~SpaceSpinorColorOrder() { ; }

	__device__ __host__ inline void load(RegType v[Ns*Nc*2], int x) const {
	  if (x >= volumeCB) return;
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = field[((x*Ns + s)*Nc + c)*2 + z];
	      }
	    }
	  }
	}

	__device__ __host__ inline void save(const RegType v[Ns*Nc*2], int x) {
	  if (x >= volumeCB) return;
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		field[((x*Ns + s)*Nc + c)*2 + z] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	}

	__device__ __host__ const RegType& operator()(int x, int s, int c, int z) const {
	  return field[((x*Ns + s)*Nc + c)*2 + z];
	}

	__device__ __host__ RegType& operator()(int x, int s, int c, int z) {
	  return field[((x*Ns + s)*Nc + c)*2 + z];
	}


	size_t Bytes() const { return volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };


    template <typename Float, int Ns, int Nc>
      struct QDPJITDiracOrder {
	typedef typename mapper<Float>::type RegType;
	Float *field;
	int volumeCB;
	int parity;
	int stride;
      QDPJITDiracOrder(const ColorSpinorField &a, Float *field_=0, int parity_=1) 
      : field(field_ ? field_ : (Float*)a.V()), volumeCB(a.VolumeCB()), stride(a.Stride()), parity(parity_)
	{ if (volumeCB != a.Stride()) errorQuda("Stride must equal volume for this field order"); }
	virtual ~QDPJITDiracOrder() { ; }

	__device__ __host__ inline void load(RegType v[Ns*Nc*2], int x) const {
	  if (x >= volumeCB) return;
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		v[(s*Nc+c)*2+z] = field[(((z*Nc + c)*Ns + s)*2 + parity)*volumeCB + x];
	      }
	    }
	  }
	}

	__device__ __host__ inline void save(const RegType v[Ns*Nc*2], int x) {
	  if (x >= volumeCB) return;
	  for (int s=0; s<Ns; s++) {
	    for (int c=0; c<Nc; c++) {
	      for (int z=0; z<2; z++) {
		field[(((z*Nc + c)*Ns + s)*2 + parity)*volumeCB + x] = v[(s*Nc+c)*2+z];
	      }
	    }
	  }
	}

	__device__ __host__ const RegType& operator()(int x, int s, int c, int z) const {
	  return field[((x*Ns + s)*Nc + c)*2 + z];
	}

	__device__ __host__ RegType& operator()(int x, int s, int c, int z) {
	  return field[((x*Ns + s)*Nc + c)*2 + z];
	}


	size_t Bytes() const { return volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };

  } // namespace colorspinor
} // namespace quda

#endif // _COLOR_SPINOR_ORDER_H
