#ifndef _TEXTURE_H
#define _TEXTURE_H

// FIXME - it would not be too hard to get this working on the host as well

#include <convert.h>

#ifdef USE_TEXTURE_OBJECTS

template<typename OutputType, typename InputType>
class Texture {

  typedef typename quda::mapper<InputType>::type RegType;

private: 
#ifndef DIRECT_ACCESS_BLAS
  cudaTextureObject_t spinor;
#else
  const InputType *spinor; // used when textures are disabled
#endif
  
public:
  Texture() : spinor(0) { }   
#ifndef DIRECT_ACCESS_BLAS
  Texture(const cudaColorSpinorField *x, bool use_ghost = false)
    : spinor(use_ghost ? x->GhostTex() : x->Tex()) { }
#else
  Texture(const cudaColorSpinorField *x, bool use_ghost = false)
    : spinor(use_ghost ? (const InputType*)(x->Ghost2()) : (const InputType*)(x->V())) { }
#endif
  Texture(const Texture &tex) : spinor(tex.spinor) { }
  ~Texture() { }

  Texture& operator=(const Texture &tex) {
    if (this != &tex) spinor = tex.spinor;
    return *this;
  }
  
#ifndef DIRECT_ACCESS_BLAS
  __device__ inline OutputType fetch(unsigned int idx) 
  { 
    OutputType rtn;
    copyFloatN(rtn, tex1Dfetch<RegType>(spinor, idx));
    return rtn;
  }
#else
  __device__ inline OutputType fetch(unsigned int idx) 
  { OutputType out; copyFloatN(out, spinor[idx]); return out; } 
#endif

  __device__ inline OutputType operator[](unsigned int idx) { return fetch(idx); }
};

#ifndef DIRECT_ACCESS_BLAS
__device__ inline double fetch_double(int2 v)
{ return __hiloint2double(v.y, v.x); }

__device__ inline double2 fetch_double2(int4 v)
{ return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z)); }

template<> __device__ inline double2 Texture<double2,double2>::fetch(unsigned int idx) 
{ double2 out; copyFloatN(out, fetch_double2(tex1Dfetch<int4>(spinor, idx))); return out; }

template<> __device__ inline float2 Texture<float2,double2>::fetch(unsigned int idx) 
{ float2 out; copyFloatN(out, fetch_double2(tex1Dfetch<int4>(spinor, idx))); return out; }
#endif

#else 

// legacy Texture references

#if (__COMPUTE_CAPABILITY__ >= 130)

  __inline__ __device__ double fetch_double(texture<int2, 1> t, int i)
  {
    int2 v = tex1Dfetch(t,i);
    return __hiloint2double(v.y, v.x);
  }

  __inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
  {
    int4 v = tex1Dfetch(t,i);
    return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
  }
#else
  __inline__ __device__ double fetch_double(texture<int2, 1> t, int i){ return 0.0; }

  __inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
  {
    // do nothing
    return make_double2(0.0, 0.0);
  }
#endif

#define MAX_TEXELS (1<<27)

#define MAX_TEX_ID 4

// dynamically keep track of texture references we've already bound to
bool tex_id_table[MAX_TEX_ID] = { };

template<typename OutputType, typename InputType, int tex_id>
  class Texture {

  private: 
#ifdef DIRECT_ACCESS_BLAS
  const InputType *spinor; // used when textures are disabled
  size_t bytes;
#endif
  static bool bound;
  static int count;

  public:
  Texture()
#ifdef DIRECT_ACCESS_BLAS
  : spinor(0), bytes(0)
#endif
  { count++; } 

 Texture(const cudaColorSpinorField *x, bool use_ghost = false)
#ifdef DIRECT_ACCESS_BLAS 
   : spinor( use_ghost ? (const InputType*)(x->Ghost2()) : (const InputType*)(x->V())) { }
#endif
  { 
    // only bind if bytes > 0
    if (x->Bytes()) { 
      if (tex_id >= 0 && tex_id < MAX_TEX_ID) {
	if (tex_id_table[(tex_id >= 0 && tex_id < MAX_TEX_ID) ? tex_id : 0]) {
	  errorQuda("Already bound to this texture reference");
	} else {
	  tex_id_table[(tex_id >= 0 && tex_id < MAX_TEX_ID) ? tex_id : 0] = true;
	}
      }
      if (use_ghost) bind((const InputType*)(x->Ghost2()), x->GhostBytes());
      else bind((const InputType*)x->V(), x->Bytes()); bound = true;
    } 
    count++;
  }

  Texture(const Texture &tex) 
#ifdef DIRECT_ACCESS_BLAS
  : spinor(tex.spinor), bytes(tex.bytes)
#endif
  { count++; }

  ~Texture() {
    if (bound && !--count) {
      unbind(); bound = false; tex_id_table[(tex_id >= 0 && tex_id < MAX_TEX_ID) ? tex_id : 0]=false;
    }
  }

  Texture& operator=(const Texture &tex) {
#ifdef DIRECT_ACCESS_BLAS
    spinor = tex.spinor;
    bytes = tex.bytes;
#endif
    return *this;
  }

  inline void bind(const InputType*, size_t bytes){ /*errorQuda("Texture id is out of range");*/ }
  inline void unbind() { /*errorQuda("Texture id is out of range");*/ }

  //default should only be called if a tex_id is out of range
  __device__ inline OutputType fetch(unsigned int idx) { OutputType x; x.x =0; return x; };  
  __device__ inline OutputType operator[](unsigned int idx) { return fetch(idx); }
};

  template<typename OutputType, typename InputType, int tex_id>  
    bool Texture<OutputType, InputType, tex_id>::bound = false;

  template<typename OutputType, typename InputType, int tex_id>  
    int Texture<OutputType, InputType, tex_id>::count = 0;

#define DECL_TEX(id)							\
  texture<char2,1,cudaReadModeNormalizedFloat> tex_char2_##id;   \
  texture<char4,1,cudaReadModeNormalizedFloat> tex_char4_##id;   \
  texture<short2,1,cudaReadModeNormalizedFloat> tex_short2_##id;	\
  texture<short4,1,cudaReadModeNormalizedFloat> tex_short4_##id;	\
  texture<float,1> tex_float_##id;					\
  texture<float2,1> tex_float2_##id;					\
  texture<float4,1> tex_float4_##id;					\
  texture<int4,1> tex_double2_##id;


#define DEF_BIND_UNBIND(outtype, intype, id)				\
  template<> inline void Texture<outtype,intype,id>::bind(const intype *ptr, size_t bytes) \
    { cudaBindTexture(0,tex_##intype##_##id, ptr, bytes); }		\
  template<> inline void Texture<outtype,intype,id>::unbind() { cudaUnbindTexture(tex_##intype##_##id); }


#define DEF_FETCH_TEX(outtype, intype, id)				\
  template<> __device__ inline outtype Texture<outtype,intype,id>::fetch(unsigned int idx) \
    { return tex1Dfetch(tex_##intype##_##id,idx); }


#define DEF_FETCH_DIRECT(outtype, intype, id)				\
  template<> __device__ inline outtype Texture<outtype,intype,id>::fetch(unsigned int idx) \
    { outtype out; copyFloatN(out, spinor[idx]); return out; }


#if defined(DIRECT_ACCESS_BLAS)
#define DEF_FETCH DEF_FETCH_DIRECT
#else
#define DEF_FETCH DEF_FETCH_TEX
#endif


#if defined(DIRECT_ACCESS_BLAS) || defined(FERMI_NO_DBLE_TEX)
#define DEF_FETCH_DBLE DEF_FETCH_DIRECT
#else
#define DEF_FETCH_DBLE(outtype, intype, id)				\
  template<> __device__ inline outtype Texture<outtype,double2,id>::fetch(unsigned int idx) \
    { outtype out; copyFloatN(out, fetch_double2(tex_double2_##id,idx)); return out; }
#endif

#if defined(DIRECT_ACCESS_BLAS) || defined(FERMI_NO_DBLE_TEX)
#define DEF_FETCH_DBLE_MIXED DEF_FETCH_DIRECT
#else
#define DEF_FETCH_DBLE_MIXED(outtype, intype, id)                      \
  template<> __device__ inline outtype Texture<outtype,intype,id>::fetch(unsigned int idx) \
  { outtype out; copyFloatN(out, tex1Dfetch(tex_##intype##_##id,idx)); return out; }
#endif


#define DEF_BIND_UNBIND_FETCH(outtype, intype, id)	\
  DEF_BIND_UNBIND(outtype, intype, id)			\
  DEF_FETCH(outtype, intype, id)


#define DEF_ALL(id)				\
  DECL_TEX(id)					\
  DEF_BIND_UNBIND_FETCH(short2, char2, id) \
  DEF_BIND_UNBIND_FETCH(short4, char4, id) \
  DEF_BIND_UNBIND_FETCH(float2, char2, id) \
  DEF_BIND_UNBIND_FETCH(float4, char4, id) \
  DEF_BIND_UNBIND_FETCH(float2, short2, id)	\
  DEF_BIND_UNBIND_FETCH(float4, short4, id)	\
  DEF_BIND_UNBIND_FETCH(float, float, id)	\
  DEF_BIND_UNBIND_FETCH(float2, float2, id)	\
  DEF_BIND_UNBIND_FETCH(float4, float4, id)	\
  DEF_BIND_UNBIND(double2, double2, id)		\
  DEF_BIND_UNBIND(float2, double2, id)		\
  DEF_FETCH_DBLE(double2, double2, id)		\
  DEF_FETCH_DBLE(float2, double2, id)		\
  DEF_BIND_UNBIND(double2, float2, id)		\
  DEF_BIND_UNBIND(double4, float4, id)		\
  DEF_BIND_UNBIND(double2, short2, id)		\
  DEF_BIND_UNBIND(double4, short4, id)		\
  DEF_BIND_UNBIND(double2, char2, id)    \
  DEF_BIND_UNBIND(double4, char4, id)    \
  DEF_FETCH_DBLE_MIXED(double2, float2, id)	\
  DEF_FETCH_DBLE_MIXED(double4, float4, id)	\
  DEF_FETCH_DBLE_MIXED(double2, short2, id)	\
  DEF_FETCH_DBLE_MIXED(double4, short4, id) \
  DEF_FETCH_DBLE_MIXED(double2, char2, id) \
  DEF_FETCH_DBLE_MIXED(double4, char4, id) \

  // Declare the textures and define the member functions of the corresponding templated classes.
  DEF_ALL(0)
  DEF_ALL(1)
  DEF_ALL(2)
  DEF_ALL(3)
  DEF_ALL(4)

#undef DECL_TEX
#undef DEF_BIND_UNBIND
#undef DEF_FETCH_DIRECT
#undef DEF_FETCH_TEX
#undef DEF_FETCH
#undef DEF_FETCH_DBLE
#undef DEF_BIND_UNBIND_FETCH
#undef DEF_ALL

#endif // USE_TEXTURE_OBJECTS


  /**
     Checks that the types are set correctly.  The precision used in the
     RegType must match that of the InterType, and the ordering of the
     InterType must match that of the StoreType.  The only exception is
     when fixed precision is used, in which case, RegType can be a double
     and InterType can be single (with StoreType short).
     
     @param RegType Register type used in kernel
     @param InterType Intermediate format - RegType precision with StoreType ordering
     @param StoreType Type used to store field in memory
  */
  template <typename RegType, typename InterType, typename StoreType>
    void checkTypes() {

    const size_t reg_size = sizeof(((RegType*)0)->x);
    const size_t inter_size = sizeof(((InterType*)0)->x);
    const size_t store_size = sizeof(((StoreType*)0)->x);

    if (reg_size != inter_size  && store_size != 2 && inter_size != 4)
      errorQuda("Precision of register (%lu) and intermediate (%lu) types must match\n",
		(unsigned long)reg_size, (unsigned long)inter_size);
  
    if (vecLength<InterType>() != vecLength<StoreType>()) {
      errorQuda("Vector lengths intermediate and register types must match\n");
    }

    if (vecLength<RegType>() == 0) errorQuda("Vector type not supported\n");
    if (vecLength<InterType>() == 0) errorQuda("Vector type not supported\n");
    if (vecLength<StoreType>() == 0) errorQuda("Vector type not supported\n");

  }

  template <int M, typename FloatN>
  __device__ inline float store_norm(float *norm, FloatN x[M], int i) {
    float c[M];
#pragma unroll
    for (int j=0; j<M; j++) c[j] = max_fabs(x[j]);
#pragma unroll
    for (int j=1; j<M; j++) c[0] = fmaxf(c[j],c[0]);
    norm[i] = c[0];
    return __fdividef(MAX_SHORT, c[0]);
  }

  // Specializations for chars.

  template <int M>
  __device__ inline float store_norm(float *norm, char x[M], int i) {
    float c[M];
#pragma unroll
    for (int j=0; j<M; j++) c[j] = max_fabs(x[j]);
#pragma unroll
    for (int j=1; j<M; j++) c[0] = fmaxf(c[j],c[0]);
    norm[i] = c[0];
    return __fdividef(MAX_CHAR, c[0]);
  }

  template <int M>
  __device__ inline float store_norm(float *norm, char2 x[M], int i) {
    float c[M];
#pragma unroll
    for (int j=0; j<M; j++) c[j] = max_fabs(x[j]);
#pragma unroll
    for (int j=1; j<M; j++) c[0] = fmaxf(c[j],c[0]);
    norm[i] = c[0];
    return __fdividef(MAX_CHAR, c[0]);
  }

  template <int M>
  __device__ inline float store_norm(float *norm, char4 x[M], int i) {
    float c[M];
#pragma unroll
    for (int j=0; j<M; j++) c[j] = max_fabs(x[j]);
#pragma unroll
    for (int j=1; j<M; j++) c[0] = fmaxf(c[j],c[0]);
    norm[i] = c[0];
    return __fdividef(MAX_CHAR, c[0]);
  }

  /**
     @param RegType Register type used in kernel
     @param InterType Intermediate format - RegType precision with StoreType ordering
     @param StoreType Type used to store field in memory
     @param N Length of vector of RegType elements that this Spinor represents
     @param tex_id Which texture reference are we using.  A default of
     -1 disables textures on architectures that don't support texture objects.
  */
template <typename RegType, typename StoreType, int N, int tex_id=-1>
  class SpinorTexture {

  typedef typename bridge_mapper<RegType,StoreType>::type InterType;

  protected:
#ifdef USE_TEXTURE_OBJECTS // texture objects
    Texture<InterType, StoreType> tex;
    Texture<InterType, StoreType> ghostTex;
#else
    StoreType *spinor;
    StoreType *ghost_spinor;
    Texture<InterType, StoreType, tex_id> tex;
    Texture<InterType, StoreType, -1> ghostTex;
#endif
    float *norm; // always use direct reads for norm

    int stride;
    unsigned int cb_offset;
    unsigned int cb_norm_offset;
#ifndef BLAS_SPINOR
    int ghost_stride[4];
#endif

  public:
    SpinorTexture() :
#ifndef USE_TEXTURE_OBJECTS
    spinor(0), ghost_spinor(0),
#endif
  tex(), ghostTex(), norm(0), stride(0), cb_offset(0), cb_norm_offset(0) { } // default constructor

    // Spinor must only ever called with cudaColorSpinorField references!!!!
    SpinorTexture(const ColorSpinorField &x, int nFace=1) :
#ifndef USE_TEXTURE_OBJECTS
    spinor((StoreType*)x.V()), ghost_spinor((StoreType*)x.Ghost2()),
#endif
    tex(&(static_cast<const cudaColorSpinorField&>(x))),
    ghostTex(&(static_cast<const cudaColorSpinorField&>(x)), true),
    norm((float*)x.Norm()), stride(x.Stride()),
    cb_offset(x.Bytes()/(2*sizeof(StoreType))),
    cb_norm_offset(x.NormBytes()/(2*sizeof(float)))
    {
      checkTypes<RegType,InterType,StoreType>();
#ifndef BLAS_SPINOR
      for (int d=0; d<4; d++) ghost_stride[d] = nFace*x.SurfaceCB(d);
#endif
    }

    SpinorTexture(const SpinorTexture &st) :
#ifndef USE_TEXTURE_OBJECTS
    spinor(st.spinor), ghost_spinor(st.ghost_spinor),
#endif
    tex(st.tex), ghostTex(st.ghostTex), norm(st.norm), stride(st.stride),
    cb_offset(st.cb_offset), cb_norm_offset(st.cb_norm_offset)
      {
#ifndef BLAS_SPINOR
  for (int d=0; d<4; d++) ghost_stride[d] = st.ghost_stride[d];
#endif
      }

    SpinorTexture& operator=(const SpinorTexture &src) {
      if (&src != this) {
#ifndef USE_TEXTURE_OBJECTS
	spinor = src.spinor;
        ghost_spinor = src.ghost_spinor;
#endif
	tex = src.tex;
        ghostTex = src.ghostTex;
	norm = src.norm;
	stride = src.stride;
	cb_offset = src.cb_offset;
	cb_norm_offset = src.cb_norm_offset;
#ifndef BLAS_SPINOR
	for (int d=0; d<4; d++) ghost_stride[d] = src.ghost_stride[d];
#endif
      }
      return *this;
    }

  void set(const cudaColorSpinorField &x, int nFace=1){
#ifdef USE_TEXTURE_OBJECTS 
      tex = Texture<InterType, StoreType>(&x);
      ghostTex = Texture<InterType, StoreType>(&x,true);
#else
      spinor = (StoreType*)x.V();
      ghost_spinor = (StoreType*)x.Ghost2();
      tex = Texture<InterType, StoreType, tex_id>(&x);
      ghostTex = Texture<InterType, StoreType, -1>(&x,true);
#endif      
      norm = (float*)x.Norm();
      stride = x.Stride();
      cb_offset = x.Bytes()/(2*sizeof(StoreType));
      cb_norm_offset = x.NormBytes()/(2*sizeof(float));
#ifndef BLAS_SPINOR
      for (int d=0; d<4; d++) ghost_stride[d] = nFace*x.SurfaceCB(d);
#endif
      checkTypes<RegType,InterType,StoreType>();
    }

    virtual ~SpinorTexture() { }

  __device__ inline void load(RegType x[], const int i, const int parity=0) {
      // load data into registers first using the storage order
      constexpr int M = (N * vec_length<RegType>::value ) / vec_length<InterType>::value;
      InterType y[M];

      // If we are using tex references, then we can only use the predeclared texture ids
#ifndef USE_TEXTURE_OBJECTS
      if (tex_id >= 0 && tex_id <= MAX_TEX_ID) {
#endif
	// fixed precision
	if ( isFixed<StoreType>::value ) {
	  float xN = norm[cb_norm_offset*parity + i];
#pragma unroll
	  for (int j=0; j<M; j++) y[j] = xN*tex[cb_offset*parity + i + j*stride];
	} else { // other types
#pragma unroll 
	  for (int j=0; j<M; j++) copyFloatN(y[j], tex[cb_offset*parity + i + j*stride]);
	}
#ifndef USE_TEXTURE_OBJECTS
      } else { // default load when out of tex_id range

	if ( isFixed<StoreType>::value ) {
	  float xN = norm[cb_norm_offset*parity + i];
#pragma unroll
	  for (int j=0; j<M; j++) {
	    copyFloatN(y[j], spinor[cb_offset*parity + i + j*stride]);
	    y[j] *= xN;
	  }
	} else { // other types
#pragma unroll
	  for (int j=0; j<M; j++) copyFloatN(y[j],spinor[cb_offset*parity + i + j*stride]);
	}
      }
#endif // !USE_TEXTURE_OBJECTS

      // now convert into desired register order
      convert<RegType, InterType>(x, y, N);
    }

#ifndef BLAS_SPINOR
  /**
     Load the ghost spinor.  For Wilson fermions, we assume that the
     ghost is spin projected
  */
  __device__ inline void loadGhost(RegType x[], const int i, const int dim) {
    // load data into registers first using the storage order
    const int Nspin = (N * vec_length<RegType>::value) / (3 * 2);
    // if Wilson, then load only half the number of components
    constexpr int M = ((N * vec_length<RegType>::value ) / vec_length<InterType>::value) / ((Nspin == 4) ? 2 : 1);
    
    InterType y[M];
    
    // If we are using tex references, then we can only use the predeclared texture ids
#ifndef USE_TEXTURE_OBJECTS
    if (tex_id >= 0 && tex_id <= MAX_TEX_ID) {
#endif
      // fixed precision types (FIXME - these don't look correct?)
      if ( isFixed<StoreType>::value ) {
	float xN = norm[i];
#pragma unroll
	for (int j=0; j<M; j++) y[j] = xN*ghostTex[i + j*ghost_stride[dim]];
      } else { // other types
#pragma unroll 
	for (int j=0; j<M; j++) copyFloatN(y[j], ghostTex[i + j*ghost_stride[dim]]);
      }
#ifndef USE_TEXTURE_OBJECTS
    } else { // default load when out of tex_id range
      
      if ( isFixed<StoreType>::value ) {
	float xN = norm[i];
#pragma unroll
	for (int j=0; j<M; j++) {
	  copyFloatN(y[j], ghost_spinor[i + j*ghost_stride[dim]]);
	  y[j] *= xN;
	}
      } else { // other types
#pragma unroll
	for (int j=0; j<M; j++) copyFloatN(y[j],ghost_spinor[i + j*ghost_stride[dim]]);
      }
    }
#endif // !USE_TEXTURE_OBJECTS
    
    // now convert into desired register order
    convert<RegType, InterType>(x, y, N);
  }
#endif

    QudaPrecision Precision() const {
      QudaPrecision precision = QUDA_INVALID_PRECISION;
      if (sizeof(((StoreType*)0)->x) == sizeof(double)) precision = QUDA_DOUBLE_PRECISION;
      else if (sizeof(((StoreType*)0)->x) == sizeof(float)) precision = QUDA_SINGLE_PRECISION;
      else if (sizeof(((StoreType*)0)->x) == sizeof(short)) precision = QUDA_HALF_PRECISION;
      else if (sizeof(((StoreType*)0)->x) == sizeof(char)) precision = QUDA_QUARTER_PRECISION;
      else errorQuda("Unknown precision type\n");
      return precision;
    }

    int Stride() const { return stride; }
    int Bytes() const { return N*sizeof(RegType); }
  };

  /**
     @param RegType Register type used in kernel
     @param InterType Intermediate format - RegType precision with StoreType ordering
     @param StoreType Type used to store field in memory
     @param N Length of vector of RegType elements that this Spinor represents
     @param tex_id Which texture reference are we using.  A default of
     -1 disables textures on architectures that don't support texture objects.
  */
template <typename RegType, typename StoreType, int N, int write, int tex_id=-1>
    class Spinor : public SpinorTexture<RegType,StoreType,N,tex_id> {

  typedef typename bridge_mapper<RegType,StoreType>::type InterType;
  typedef SpinorTexture<RegType,StoreType,N,tex_id> ST;

  private:
#ifdef USE_TEXTURE_OBJECTS
    StoreType *spinor;
    StoreType *ghost_spinor;
#define SPINOR spinor
#else
#define SPINOR ST::spinor
#endif
  public:
  Spinor() : ST()
#ifdef USE_TEXTURE_OBJECTS
    , spinor(0), ghost_spinor(0)
#endif
    {} // default constructor

    // Spinor must only ever called with cudaColorSpinorField references!!!!
  Spinor(const ColorSpinorField &x, int nFace=1) : ST(x, nFace)
#ifdef USE_TEXTURE_OBJECTS
    , spinor((StoreType*)x.V()), ghost_spinor((StoreType*)x.Ghost2())
#endif
    {}

    Spinor(const Spinor &st) : ST(st)
#ifdef USE_TEXTURE_OBJECTS
    , spinor(st.spinor), ghost_spinor(st.ghost_spinor)
#endif
    {}

    Spinor& operator=(const Spinor &src) {
      ST::operator=(src);
      if (&src != this) {
#ifdef USE_TEXTURE_OBJECTS
	spinor = src.spinor;
        ghost_spinor = src.ghost_spinor;
#endif
      }
      return *this;
    }

    void set(const cudaColorSpinorField &x){
      ST::set(x);
#ifdef USE_TEXTURE_OBJECTS
      spinor = (StoreType*)x.V();
      ghost_spinor = (StoreType*)x.Ghost2();
#endif
    }

    ~Spinor() { }

    // default store used for simple fields
  __device__ inline void save(RegType x[], int i, const int parity = 0) {
      if (write) {
	constexpr int M = (N * vec_length<RegType>::value ) / vec_length<InterType>::value;
	InterType y[M];
	convert<InterType, RegType>(y, x, M);
	
	if ( isFixed<StoreType>::value ) {
          float C = store_norm<M, InterType>(ST::norm, y, ST::cb_norm_offset*parity + i);
#pragma unroll
          for (int j=0; j<M; j++) copyFloatN(SPINOR[ST::cb_offset*parity + i + j*ST::stride], C*y[j]);
	} else {
#pragma unroll
          for (int j=0; j<M; j++) copyFloatN(SPINOR[ST::cb_offset*parity + i + j*ST::stride], y[j]);
	}
      }
    }

    // used to backup the field to the host
    void backup(char **spinor_h, char **norm_h, size_t bytes, size_t norm_bytes) {
      if (write) {
	*spinor_h = new char[bytes];
	cudaMemcpy(*spinor_h, SPINOR, bytes, cudaMemcpyDeviceToHost);
	if (norm_bytes > 0) {
	  *norm_h = new char[norm_bytes];
          cudaMemcpy(*norm_h, ST::norm, norm_bytes, cudaMemcpyDeviceToHost);
	}
	checkCudaError();
      }
    }

    // restore the field from the host
    void restore(char **spinor_h, char **norm_h, size_t bytes, size_t norm_bytes) {
      if (write) {
	cudaMemcpy(SPINOR, *spinor_h, bytes, cudaMemcpyHostToDevice);
	if (norm_bytes > 0) {
          cudaMemcpy(ST::norm, *norm_h, norm_bytes, cudaMemcpyHostToDevice);
	  delete []*norm_h;
	  *norm_h = 0;
	}
	delete []*spinor_h;
	*spinor_h = 0;
	checkCudaError();
      }
    }

    void* V() { return (void*)SPINOR; }
    float* Norm() { return ST::norm; }
  };


#ifndef USE_TEXTURE_OBJECTS
#undef MAX_TEX_ID
#endif

#endif // _TEXTURE_H
