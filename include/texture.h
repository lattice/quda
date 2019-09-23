#pragma once

#ifdef USE_TEXTURE_OBJECTS

template <typename OutputType, typename InputType> class Texture
{
  typedef typename quda::mapper<InputType>::type RegType;

  private:
  cudaTextureObject_t spinor;

  public:
  Texture() : spinor(0) {}
  Texture(const cudaColorSpinorField *x, bool use_ghost = false)
    : spinor(use_ghost ? x->GhostTex() : x->Tex()) { }
  Texture(const Texture &tex) : spinor(tex.spinor) { }
  ~Texture() { }

  Texture& operator=(const Texture &tex) {
    if (this != &tex) spinor = tex.spinor;
    return *this;
  }

  __device__ inline OutputType fetch(unsigned int idx) const
  {
    OutputType rtn;
    copyFloatN(rtn, tex1Dfetch<RegType>(spinor, idx));
    return rtn;
  }

  __device__ inline OutputType operator[](unsigned int idx) const { return fetch(idx); }
};

__device__ inline double fetch_double(int2 v)
{ return __hiloint2double(v.y, v.x); }

__device__ inline double2 fetch_double2(int4 v)
{ return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z)); }

template <> __device__ inline double2 Texture<double2, double2>::fetch(unsigned int idx) const
{ double2 out; copyFloatN(out, fetch_double2(tex1Dfetch<int4>(spinor, idx))); return out; }

template <> __device__ inline float2 Texture<float2, double2>::fetch(unsigned int idx) const
{ float2 out; copyFloatN(out, fetch_double2(tex1Dfetch<int4>(spinor, idx))); return out; }

#else // !USE_TEXTURE_OBJECTS - use direct reads

template <typename OutputType, typename InputType> class Texture
{
  typedef typename quda::mapper<InputType>::type RegType;

  private:
  const InputType *spinor; // used when textures are disabled

  public:
  Texture() : spinor(0) {}
  Texture(const cudaColorSpinorField *x, bool use_ghost = false) :
      spinor(use_ghost ? (const InputType *)(x->Ghost2()) : (const InputType *)(x->V()))
  {
  }
  Texture(const Texture &tex) : spinor(tex.spinor) {}
  ~Texture() {}

  Texture& operator=(const Texture &tex) {
    if (this != &tex) spinor = tex.spinor;
    return *this;
  }

  __device__ __host__ inline OutputType operator[](unsigned int idx) const
  {
    OutputType out;
    copyFloatN(out, spinor[idx]);
    return out;
  }

  __device__ __host__ inline InputType *Spinor() const { return const_cast<InputType *>(spinor); }
};

#endif

/**
   Checks that the types are set correctly.  The precision used in the
   RegType must match that of the InterType, and the ordering of the
   InterType must match that of the StoreType.  The only exception is
   when fixed precision is used, in which case, RegType can be a double
   and InterType can be single (with StoreType short or char).

   @param RegType Register type used in kernel
   @param InterType Intermediate format - RegType precision with StoreType ordering
   @param StoreType Type used to store field in memory
*/
template <typename RegType, typename InterType, typename StoreType> void checkTypes()
{

  const size_t reg_size = sizeof(((RegType *)0)->x);
  const size_t inter_size = sizeof(((InterType *)0)->x);
  const size_t store_size = sizeof(((StoreType *)0)->x);

  if (reg_size != inter_size && store_size != 2 && store_size != 1 && inter_size != 4)
    errorQuda("Precision of register (%lu) and intermediate (%lu) types must match\n", (unsigned long)reg_size,
        (unsigned long)inter_size);

  if (vecLength<InterType>() != vecLength<StoreType>()) {
    errorQuda("Vector lengths intermediate and register types must match\n");
  }

  if (vecLength<RegType>() == 0) errorQuda("Vector type not supported\n");
  if (vecLength<InterType>() == 0) errorQuda("Vector type not supported\n");
  if (vecLength<StoreType>() == 0) errorQuda("Vector type not supported\n");
}

template <typename RegType, typename StoreType, bool is_fixed> struct SpinorNorm {
  typedef typename bridge_mapper<RegType, StoreType>::type InterType;
  float *norm; // always use direct reads for norm
  unsigned int cb_norm_offset;

  SpinorNorm() : norm(nullptr), cb_norm_offset(0) {}

  SpinorNorm(const ColorSpinorField &x) : norm((float *)x.Norm()), cb_norm_offset(x.NormBytes() / (2 * sizeof(float)))
  {
  }

  SpinorNorm(const SpinorNorm &sn) : norm(sn.norm), cb_norm_offset(sn.cb_norm_offset) {}

  SpinorNorm &operator=(const SpinorNorm &src)
  {
    if (&src != this) {
      norm = src.norm;
      cb_norm_offset = src.cb_norm_offset;
    }
    return *this;
  }

  void set(const cudaColorSpinorField &x)
  {
    norm = (float *)x.Norm();
    cb_norm_offset = x.NormBytes() / (2 * sizeof(float));
  }

  virtual ~SpinorNorm() {}

  __device__ inline float load_norm(const int i, const int parity = 0) const { return norm[cb_norm_offset * parity + i]; }

  template <int M> __device__ inline float store_norm(InterType x[M], int i, int parity)
  {
    float c[M];
#pragma unroll
    for (int j = 0; j < M; j++) c[j] = max_fabs(x[j]);
#pragma unroll
    for (int j = 1; j < M; j++) c[0] = fmaxf(c[j], c[0]);
    norm[cb_norm_offset * parity + i] = c[0];
    return __fdividef(fixedMaxValue<StoreType>::value, c[0]);
  }

  // used to backup the field to the host
  void backup(char **norm_h, size_t norm_bytes)
  {
    if (norm_bytes > 0) {
      *norm_h = new char[norm_bytes];
      cudaMemcpy(*norm_h, norm, norm_bytes, cudaMemcpyDeviceToHost);
    }
    checkCudaError();
  }

  // restore the field from the host
  void restore(char **norm_h, size_t norm_bytes)
  {
    if (norm_bytes > 0) {
      cudaMemcpy(norm, *norm_h, norm_bytes, cudaMemcpyHostToDevice);
      delete[] * norm_h;
      *norm_h = 0;
    }
    checkCudaError();
  }

  float *Norm() { return norm; }
};

template <typename RegType, typename StoreType> struct SpinorNorm<RegType, StoreType, false> {
  typedef typename bridge_mapper<RegType, StoreType>::type InterType;
  SpinorNorm() {}
  SpinorNorm(const ColorSpinorField &x) {}
  SpinorNorm(const SpinorNorm &sn) {}
  SpinorNorm &operator=(const SpinorNorm &src) { return *this; }
  void set(const cudaColorSpinorField &x) {}
  __device__ inline float load_norm(const int i, const int parity = 0) const { return 1.0; }
  template <int M> __device__ inline float store_norm(InterType x[M], int i, int parity) { return 1.0; }
  void backup(char **norm_h, size_t norm_bytes) {}
  void restore(char **norm_h, size_t norm_bytes) {}
  float *Norm() { return nullptr; }
};

/**
   @param RegType Register type used in kernel
   @param InterType Intermediate format - RegType precision with StoreType ordering
   @param StoreType Type used to store field in memory
   @param N Length of vector of RegType elements that this Spinor represents
*/
template <typename RegType_, typename StoreType_, int N>
struct SpinorTexture : SpinorNorm<RegType_, StoreType_, isFixed<StoreType_>::value> {
  typedef RegType_ RegType;
  typedef StoreType_ StoreType;
  typedef typename bridge_mapper<RegType,StoreType>::type InterType;
  typedef SpinorNorm<RegType, StoreType_, isFixed<StoreType>::value> SN;

protected:
  Texture<InterType, StoreType> tex;

  int stride;
  unsigned int cb_offset;
#ifndef BLAS_SPINOR
  Texture<InterType, StoreType> ghostTex;
  int ghost_stride[4];
#endif

public:
  SpinorTexture() :
    SN(),
    tex(),
    stride(0),
    cb_offset(0)
#ifndef BLAS_SPINOR
    ,
    ghostTex()
#endif
  {
  } // default constructor

  // Spinor must only ever called with cudaColorSpinorField references!!!!
  SpinorTexture(const ColorSpinorField &x, int nFace = 1) :
    SN(x),
    tex(&(static_cast<const cudaColorSpinorField &>(x))),
    stride(x.Stride()),
    cb_offset(x.Bytes() / (2 * sizeof(StoreType)))
#ifndef BLAS_SPINOR
    ,
    ghostTex(&(static_cast<const cudaColorSpinorField &>(x)), true)
#endif
  {
    checkTypes<RegType, InterType, StoreType>();
#ifndef BLAS_SPINOR
    for (int d = 0; d < 4; d++) ghost_stride[d] = nFace * x.SurfaceCB(d);
#endif
  }

  SpinorTexture(const SpinorTexture &st) :
    SN(st),
    tex(st.tex),
    stride(st.stride),
    cb_offset(st.cb_offset)
#ifndef BLAS_SPINOR
    ,
    ghostTex(st.ghostTex)
#endif
  {
#ifndef BLAS_SPINOR
    for (int d = 0; d < 4; d++) ghost_stride[d] = st.ghost_stride[d];
#endif
  }

  SpinorTexture &operator=(const SpinorTexture &src)
  {
    if (&src != this) {
      SN::operator=(src);
      tex = src.tex;
      stride = src.stride;
      cb_offset = src.cb_offset;
#ifndef BLAS_SPINOR
      ghostTex = src.ghostTex;
      for (int d = 0; d < 4; d++) ghost_stride[d] = src.ghost_stride[d];
#endif
    }
    return *this;
  }

  void set(const cudaColorSpinorField &x, int nFace = 1)
  {
    SN::set(x);
    tex = Texture<InterType, StoreType>(&x);
    stride = x.Stride();
    cb_offset = x.Bytes() / (2 * sizeof(StoreType));
#ifndef BLAS_SPINOR
    ghostTex = Texture<InterType, StoreType>(&x, true);
    for (int d = 0; d < 4; d++) ghost_stride[d] = nFace * x.SurfaceCB(d);
#endif
    checkTypes<RegType, InterType, StoreType>();
  }

  virtual ~SpinorTexture() {}

  __device__ inline void load(RegType x[], const int i, const int parity = 0) const
  {
    // load data into registers first using the storage order
    constexpr int M = (N * vec_length<RegType>::value) / vec_length<InterType>::value;
    InterType y[M];

    // fixed precision
    if (isFixed<StoreType>::value) {
      const float xN = SN::load_norm(i, parity);
#pragma unroll
      for (int j = 0; j < M; j++) y[j] = xN * tex[cb_offset * parity + i + j * stride];
    } else { // other types
#pragma unroll
      for (int j = 0; j < M; j++) copyFloatN(y[j], tex[cb_offset * parity + i + j * stride]);
    }

    // now convert into desired register order
    convert<RegType, InterType>(x, y, N);
  }

#ifndef BLAS_SPINOR
  /**
     Load the ghost spinor.  For Wilson fermions, we assume that the
     ghost is spin projected
  */
  __device__ inline void loadGhost(RegType x[], const int i, const int dim) const
  {
    // load data into registers first using the storage order
    const int Nspin = (N * vec_length<RegType>::value) / (3 * 2);
    // if Wilson, then load only half the number of components
    constexpr int M = ((N * vec_length<RegType>::value ) / vec_length<InterType>::value) / ((Nspin == 4) ? 2 : 1);

    InterType y[M];

    // fixed precision types (FIXME - these don't look correct?)
    if (isFixed<StoreType>::value) {
      float xN = SN::load_norm(i);
#pragma unroll
      for (int j = 0; j < M; j++) y[j] = xN * ghostTex[i + j * ghost_stride[dim]];
    } else { // other types
#pragma unroll
      for (int j = 0; j < M; j++) copyFloatN(y[j], ghostTex[i + j * ghost_stride[dim]]);
    }

    // now convert into desired register order
    convert<RegType, InterType>(x, y, N);
  }
#endif

  QudaPrecision Precision() const
  {
    QudaPrecision precision = QUDA_INVALID_PRECISION;
    if (sizeof(((StoreType *)0)->x) == sizeof(double))
      precision = QUDA_DOUBLE_PRECISION;
    else if (sizeof(((StoreType *)0)->x) == sizeof(float))
      precision = QUDA_SINGLE_PRECISION;
    else if (sizeof(((StoreType *)0)->x) == sizeof(short))
      precision = QUDA_HALF_PRECISION;
    else if (sizeof(((StoreType *)0)->x) == sizeof(char))
      precision = QUDA_QUARTER_PRECISION;
    else
      errorQuda("Unknown precision type\n");
    return precision;
  }

  int Stride() const { return stride; }
  int Bytes() const { return N * sizeof(RegType); }
};

/**
   @param RegType Register type used in kernel
   @param InterType Intermediate format - RegType precision with StoreType ordering
   @param StoreType Type used to store field in memory
   @param N Length of vector of RegType elements that this Spinor represents
*/
template <typename RegType_, typename StoreType_, int N, int write_>
struct Spinor : public SpinorTexture<RegType_, StoreType_, N> {
  static constexpr bool write = write_;
  typedef RegType_ RegType;
  typedef StoreType_ StoreType;
  typedef typename bridge_mapper<RegType,StoreType>::type InterType;
  typedef SpinorTexture<RegType, StoreType, N> ST;
  typedef SpinorNorm<RegType, StoreType_, isFixed<StoreType>::value> SN;

private:
#ifdef USE_TEXTURE_OBJECTS
  StoreType *spinor;
#endif

public:
  Spinor() :
    ST()
#ifdef USE_TEXTURE_OBJECTS
    ,
    spinor(0)
#endif
  {
  } // default constructor

  // Spinor must only ever called with cudaColorSpinorField references!!!!
  Spinor(const ColorSpinorField &x, int nFace = 1) :
    ST(x, nFace)
#ifdef USE_TEXTURE_OBJECTS
    ,
    spinor((StoreType *)x.V())
#endif
  {
  }

  Spinor(const Spinor &st) :
    ST(st)
#ifdef USE_TEXTURE_OBJECTS
    ,
    spinor(st.spinor)
#endif
  {
  }

  Spinor &operator=(const Spinor &src)
  {
    ST::operator=(src);
    if (&src != this) {
#ifdef USE_TEXTURE_OBJECTS
      spinor = src.spinor;
#endif
    }
    return *this;
  }

  void set(const cudaColorSpinorField &x)
  {
    ST::set(x);
#ifdef USE_TEXTURE_OBJECTS
    spinor = (StoreType *)x.V();
#endif
  }

  ~Spinor() {}

  // default store used for simple fields
  __device__ inline void save(RegType x[], int i, const int parity = 0)
  {
    if (write) {
#ifndef USE_TEXTURE_OBJECTS
      StoreType *spinor = ST::tex.Spinor();
#endif
      constexpr int M = (N * vec_length<RegType>::value) / vec_length<InterType>::value;
      InterType y[M];
      convert<InterType, RegType>(y, x, M);

      if (isFixed<StoreType>::value) {
        float C = SN::store_norm<M>(y, i, parity);
#pragma unroll
        for (int j = 0; j < M; j++) copyFloatN(spinor[ST::cb_offset * parity + i + j * ST::stride], C * y[j]);
      } else {
#pragma unroll
        for (int j = 0; j < M; j++) copyFloatN(spinor[ST::cb_offset * parity + i + j * ST::stride], y[j]);
      }
    }
  }

  // used to backup the field to the host
  void backup(char **spinor_h, char **norm_h, size_t bytes, size_t norm_bytes)
  {
    if (write) {
#ifndef USE_TEXTURE_OBJECTS
      StoreType *spinor = ST::tex.Spinor();
#endif
      *spinor_h = new char[bytes];
      cudaMemcpy(*spinor_h, spinor, bytes, cudaMemcpyDeviceToHost);
      SN::backup(norm_h, norm_bytes);
      checkCudaError();
    }
  }

  // restore the field from the host
  void restore(char **spinor_h, char **norm_h, size_t bytes, size_t norm_bytes)
  {
    if (write) {
#ifndef USE_TEXTURE_OBJECTS
      StoreType *spinor = ST::tex.Spinor();
#endif
      cudaMemcpy(spinor, *spinor_h, bytes, cudaMemcpyHostToDevice);
      SN::restore(norm_h, norm_bytes);
      delete[] * spinor_h;
      *spinor_h = 0;
      checkCudaError();
    }
  }

  void *V()
  {
#ifndef USE_TEXTURE_OBJECTS
    StoreType *spinor = ST::tex.Spinor();
#endif
    return (void *)spinor;
  }
};
