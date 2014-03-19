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

    template <typename Float>
      class FieldOrder {

    protected:
      ColorSpinorField &field; // temporary hack
      complex<Float> *v;
      int x[QUDA_MAX_DIM];
      const int volume;
      const int nDim;
      const int nColor;
      const int nSpin;
      const int nVec;

    public:
      /** 
       * Constructor for the FieldOrder class
       * @param field The field that we are accessing
       */
    FieldOrder(ColorSpinorField &field, int nVec=1) 
      : field(field), v(static_cast<complex<Float>*>(field.V())), 
	volume(field.Volume()), nDim(field.Ndim()), nColor(field.Ncolor()), 
	nSpin(field.Nspin()), nVec(nVec)
      { for (int d=0; d<QUDA_MAX_DIM; d++) x[d]=field.X(d); }

      ColorSpinorField& Field() const { return field; }

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
      __device__ __host__ virtual const complex<Float>& operator()(int x, int s, int c) const = 0;

      /**
       * Writable complex-member accessor function
       * @param x 1-d site index
       * @param s spin index
       * @param c color index
       */
      __device__ __host__ virtual complex<Float>& operator()(int x, int s, int c) = 0;

      /** Return the length of dimension d */
      __device__ __host__ int X(int d) const { return x[d]; }

      /** Returns the number of field colors */
      __device__ __host__ int Ncolor() const { return nColor; }

      /** Returns the number of field spins */
      __device__ __host__ int Nspin() const { return nSpin; }

      /** Returns the field volume */
      __device__ __host__ int Volume() const { return volume; }

      /** Returns the field geometric dimension */
      __device__ __host__ int Ndim() const { return nDim; }

      /**
       * Specialized read-only complex-member accessor function (for mg prolongator)
       * @param x 1-d site index
       * @param s spin index
       * @param c color index
       * @param n vector number
       */
      __device__ __host__ const complex<Float>& operator()(int x, int s, int c, int n) const {
	return (*this)(x, s, c*nVec + n);
      }

      /**
       * Specialized writable complex-member accessor function (for mg prolongator)
       * @param x 1-d site index
       * @param s spin index
       * @param c color index
       * @param n vector number
       */
      __device__ __host__ complex<Float>& operator()(int x, int s, int c, int n) {
	return (*this)(x, s, c*nVec + n);      
      }

      /** Returns the number of packed vectors (for mg prolongator) */
      __device__ __host__ int NvecPacked() const { return nVec; }

      /** Returns the number of packed colors (for mg prolongator) */
      __device__ __host__ int NcolorPacked() const { return nColor / nVec; }

      /** Returns the number of packed spins (for mg prolongator) */
      __device__ __host__ int NspinPacked() const { return nSpin; }    

    };

    // Note this only support N>=2 and cannot work for Float1 ordered fields
    template <typename Float, int N>
      class FloatNOrder2 : public FieldOrder<Float> {
      typedef FieldOrder<Float> F;
      const int volumeCB;
      const int stride;
      const size_t cb_offset;

    private:

    public:
    FloatNOrder2(ColorSpinorField &field, int nVec=1) : F(field, nVec), volumeCB(field.VolumeCB()),
	stride(field.Stride()), cb_offset(field.Bytes()>>1)
      { ; }
      virtual ~FloatNOrder2() { ; }

      /**
	 @param x parity or full-site index (with checkboard ordering).  Need
	 to be careful to support non-zero padding and parity alignment.
	 @param s spin index
	 @param c color index
	 @return element at site (x,s,c)
       */
      __device__ __host__ const complex<Float>& operator()(int x, int s, int c) const {
	int x_cb = (x >= volumeCB) ? x-volumeCB : x;
	int parity = (x >= volumeCB) ? 1 : 0;
	int j = ((s*F::nColor+c)*2) / N; // factor of two for complexity
	int i = ((s*F::nColor+c)*2) % N;
	size_t index = ((j*stride+x_cb)*N+i) / 2; // back to a complex offset
	return *(reinterpret_cast<complex<Float>*>(reinterpret_cast<char*>(F::v + index) + parity*cb_offset));
      }

      /**
	 @param x parity or full-site index (with checkboard ordering).  Need
	 to be careful to support non-zero padding and parity alignment.
	 @param s spin index
	 @param c color index
	 @return element at site (x,s,c)
       */
      __device__ __host__ complex<Float>& operator()(int x, int s, int c) {
	int x_cb = (x >= volumeCB) ? x-volumeCB : x;
	int parity = (x >= volumeCB) ? 1 : 0;
	int j = ((s*F::nColor+c)*2) / N; // factor of two for complexity
	int i = ((s*F::nColor+c)*2) % N;
	size_t index = ((j*stride+x_cb)*N+i) / 2; // back to a complex offset
	return *(reinterpret_cast<complex<Float>*>(reinterpret_cast<char*>(F::v + index) + parity*cb_offset));
      }

    };

    template <typename Float>
      class SpaceSpinColorOrder : public FieldOrder<Float> {
      typedef FieldOrder<Float> F;

    private:

    public:
    SpaceSpinColorOrder(ColorSpinorField &field, int nVec=1) : F(field, nVec)
      { ; }
      virtual ~SpaceSpinColorOrder() { ; }

      __device__ __host__ const complex<Float>& operator()(int x, int s, int c) const {
	unsigned long index = (x*F::nSpin+s)*F::nColor+c;
	return *(F::v + index);
      }

      __device__ __host__ complex<Float>& operator()(int x, int s, int c) {
	unsigned long index = (x*F::nSpin+s)*F::nColor+c;
	return *(F::v + index);
      }

    };

    template <typename Float>
      class SpaceColorSpinOrder : public FieldOrder<Float> {
      typedef FieldOrder<Float> F;

    private:

    public:
    SpaceColorSpinOrder(ColorSpinorField &field, int nVec=1) : F(field, nVec)
      { ; }
      virtual ~SpaceColorSpinOrder() { ; }

       __device__ __host__ const complex<Float>& operator()(int x, int s, int c) const {
	unsigned long index = (x*F::nColor+c)*F::nSpin+s;
	return *(F::v + index);
      }

       __device__ __host__ complex<Float>& operator()(int x, int s, int c) {
	unsigned long index = (x*F::nColor+c)*F::nSpin+s;
	return *(F::v + index);
      }
    };

    template <typename Float>
      class QOPDomainWallOrder : public FieldOrder<Float> {
      typedef FieldOrder<Float> F;

    private:
      complex<Float> **v;
      int volume_4d;
      int Ls;

    public:
    QOPDomainWallOrder(ColorSpinorField &field, int nVec=1) 
      : F(field, nVec), v(static_cast<complex<Float>**>(field.V())), 
	volume_4d(1), Ls(0)
	{ 
	  if (field.Ndim() != 5) errorQuda("Error, wrong number of dimensions for this FieldOrder");
	  for (int i=0; i<4; i++) volume_4d *= field.X()[i];
	  Ls = field.X()[4];
	}
      virtual ~QOPDomainWallOrder() { ; }

       __device__ __host__ const complex<Float>& operator()(int x, int s, int c) const {
	int ls = x / Ls;
	int x_4d = x - ls*volume_4d;
	unsigned long index_4d = (x_4d*F::nColor+c)*F::nSpin+s;
	return v[ls][index_4d];
      }
      
       __device__ __host__ complex<Float>& operator()(int x, int s, int c) {
	int ls = x / Ls;
	int x_4d = x - ls*volume_4d;
	unsigned long index_4d = (x_4d*F::nColor+c)*F::nSpin+s;
	return v[ls][index_4d];
      }
    };

    template <typename Float>
      FieldOrder<Float>* createOrder(const ColorSpinorField &a, int nVec=1) {
      FieldOrder<Float>* ptr=0;

      if (a.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
	ptr = new FloatNOrder2<Float,2>(const_cast<ColorSpinorField&>(a), nVec);
      } else if (a.FieldOrder() == QUDA_FLOAT4_FIELD_ORDER) {
	ptr = new FloatNOrder2<Float,4>(const_cast<ColorSpinorField&>(a), nVec);
      } else if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	ptr = new SpaceSpinColorOrder<Float>(const_cast<ColorSpinorField&>(a), nVec);
      } else if (a.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {
	ptr = new SpaceColorSpinOrder<Float>(const_cast<ColorSpinorField&>(a), nVec);
      } else if (a.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
	ptr = new QOPDomainWallOrder<Float>(const_cast<ColorSpinorField&>(a), nVec);
      } else {
	errorQuda("Order %d not supported in ColorSpinorField", a.FieldOrder());
      }

      return ptr;
    }

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
