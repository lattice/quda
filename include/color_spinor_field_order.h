/*
  Define functors to allow for generic accessors regardless of field
  ordering.  Currently this is used for cpu fields only with limited
  ordering support, but this will be expanded for device ordering
  also.
*/

#include <register_traits.h>

namespace quda {

  template <typename Float>
    class ColorSpinorFieldOrder {

  protected:
    cpuColorSpinorField &field;

  public:
    ColorSpinorFieldOrder(cpuColorSpinorField &field) : field(field) { ; }
    virtual ~ColorSpinorFieldOrder() { ; }
    
    virtual const Float& operator()(const int &x, const int &s, const int &c, const int &z) const = 0;
    virtual Float& operator()(const int &x, const int &s, const int &c, const int &z) = 0;

    int Ncolor() const { return field.Ncolor(); }
    int Nspin() const { return field.Nspin(); }
    int Volume() const { return field.Volume(); }

  };

  template <typename Float>
    class SpaceSpinColorOrder : public ColorSpinorFieldOrder<Float> {

  private:
    cpuColorSpinorField &field; // convenient to have a "local" reference for code brevity

  public:
  SpaceSpinColorOrder(cpuColorSpinorField &field): ColorSpinorFieldOrder<Float>(field), field(field) 
    { ; }
    virtual ~SpaceSpinColorOrder() { ; }

    const Float& operator()(const int &x, const int &s, const int &c, const int &z) const {
      unsigned long index = ((x*field.nSpin+s)*field.nColor+c)*2+z;
      return *((Float*)(field.v) + index);
    }

    Float& operator()(const int &x, const int &s, const int &c, const int &z) {
      unsigned long index = ((x*field.nSpin+s)*field.nColor+c)*2+z;
      return *((Float*)(field.v) + index);
    }
  };

  template <typename Float>
    class SpaceColorSpinOrder : public ColorSpinorFieldOrder<Float> {

  private:
    cpuColorSpinorField &field;  // convenient to have a "local" reference for code brevity

  public:
  SpaceColorSpinOrder(cpuColorSpinorField &field) : ColorSpinorFieldOrder<Float>(field), field(field)
    { ; }
    virtual ~SpaceColorSpinOrder() { ; }

    const Float& operator()(const int &x, const int &s, const int &c, const int &z) const {
      unsigned long index = ((x*field.nColor+c)*field.nSpin+s)*2+z;
      return *((Float*)(field.v) + index);
    }

    Float& operator()(const int &x, const int &s, const int &c, const int &z) {
      unsigned long index = ((x*field.nColor+c)*field.nSpin+s)*2+z;    
      return *((Float*)(field.v) + index);
    }
  };

  template <typename Float>
    class QOPDomainWallOrder : public ColorSpinorFieldOrder<Float> {

  private:
    cpuColorSpinorField &field;  // convenient to have a "local" reference for code brevity
    int volume_4d;
    int Ls;

  public:
  QOPDomainWallOrder(cpuColorSpinorField &field) : ColorSpinorFieldOrder<Float>(field), 
      field(field), volume_4d(1), Ls(0)
      { 
	if (field.Ndim() != 5) errorQuda("Error, wrong number of dimensions for this ColorSpinorFieldOrder");
	for (int i=0; i<4; i++) volume_4d *= field.x[i];
	Ls = field.x[4];
      }
    virtual ~QOPDomainWallOrder() { ; }

    const Float& operator()(const int &x, const int &s, const int &c, const int &z) const {
      int ls = x / Ls;
      int x_4d = x - ls*volume_4d;
      unsigned long index_4d = ((x_4d*field.nColor+c)*field.nSpin+s)*2+z;
      return ((Float**)(field.v))[ls][index_4d];
    }

    Float& operator()(const int &x, const int &s, const int &c, const int &z) {
      int ls = x / Ls;
      int x_4d = x - ls*volume_4d;
      unsigned long index_4d = ((x_4d*field.nColor+c)*field.nSpin+s)*2+z;
      return ((Float**)(field.v))[ls][index_4d];
    }
  };

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

  int x0 = x-tid; // x0 is the base index from where we are reading
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


} // namespace quda
