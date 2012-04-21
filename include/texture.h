#ifndef _TEXTURE_H
#define _TEXTURE_H

#include <convert.h>


// uncomment to disable texture reads
//#define DIRECT_ACCESS_BLAS


#if (__COMPUTE_CAPABILITY__ >= 130)
__inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
  int4 v = tex1Dfetch(t,i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}
#else
__inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
  // do nothing
  return make_double2(0.0, 0.0);
}
#endif


#define MAX_TEXELS (1<<27)

template<typename OutputType, typename InputType, int tex_id=0>
class Texture {
 private: 
 const InputType *spinor; // used when textures are disabled
 //size_t bytes;

 public:
 Texture() { ; }
 Texture(const InputType *x, size_t bytes) : spinor(x)/*, bytes(bytes)*/ { 

   if (bytes) bind(x, MAX_TEXELS*sizeof(InputType)); // only bind if bytes > 0
   //if (bytes) bind(x, bytes); // only bind if bytes > 0
 }
 ~Texture() { /*if (bytes) */ /*unbind()*/; } // unbinding is unnecessary and costly

 Texture& operator=(const Texture &tex) {
   spinor = tex.spinor;
   return *this;
 }

 inline void bind(const InputType*, size_t bytes){ errorQuda("Texture id is out of range"); }
 inline void unbind() { errorQuda("Texture id is out of range"); }

 //default should only be called if a tex_id is out of range
 __device__ inline OutputType fetch(unsigned int idx) { return 0; };  
 __device__ inline OutputType operator[](unsigned int idx) { return fetch(idx); }
};


#define DECL_TEX(id) \
  texture<short2,1,cudaReadModeNormalizedFloat> tex_short2_##id; \
  texture<short4,1,cudaReadModeNormalizedFloat> tex_short4_##id; \
  texture<float,1> tex_float_##id;   \
  texture<float2,1> tex_float2_##id; \
  texture<float4,1> tex_float4_##id; \
  texture<int4,1> tex_double2_##id;


#define DEF_BIND_UNBIND(outtype, intype, id) \
  template<> inline void Texture<outtype,intype,id>::bind(const intype *ptr, size_t bytes) \
  { cudaBindTexture(0,tex_##intype##_##id, ptr, bytes); } \
  template<> inline void Texture<outtype,intype,id>::unbind() { cudaUnbindTexture(tex_##intype##_##id); }


#define DEF_FETCH_TEX(outtype, intype, id) \
  template<> __device__ inline outtype Texture<outtype,intype,id>::fetch(unsigned int idx) \
  { return tex1Dfetch(tex_##intype##_##id,idx); }


#define DEF_FETCH_DIRECT(outtype, intype, id) \
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
#define DEF_FETCH_DBLE(outtype, intype, id) \
  template<> __device__ inline outtype Texture<outtype,double2,id>::fetch(unsigned int idx) \
  { outtype out; copyFloatN(out, fetch_double2(tex_double2_##id,idx)); return out; }
#endif


#define DEF_BIND_UNBIND_FETCH(outtype, intype, id) \
  DEF_BIND_UNBIND(outtype, intype, id)             \
  DEF_FETCH(outtype, intype, id)


#define DEF_ALL(id)                          \
  DECL_TEX(id)                               \
  DEF_BIND_UNBIND_FETCH(float2, short2, id)  \
  DEF_BIND_UNBIND_FETCH(float4, short4, id)  \
  DEF_BIND_UNBIND_FETCH(float, float, id)    \
  DEF_BIND_UNBIND_FETCH(float2, float2, id)  \
  DEF_BIND_UNBIND_FETCH(float4, float4, id)  \
  DEF_BIND_UNBIND(double2, double2, id)      \
  DEF_BIND_UNBIND(float2, double2, id)       \
  DEF_FETCH_DBLE(double2, double2, id)       \
  DEF_FETCH_DBLE(float2, double2, id)


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


/**
   Checks that the types are set correctly.  The precision used in the
   RegType must match that of the InterType, and the ordering of the
   InterType must match that of the StoreType.  The only exception is
   when half precision is used, in which case, RegType can be a double
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
	      reg_size, inter_size);
  
  if (vecLength<InterType>() != vecLength<StoreType>()) {
    errorQuda("Vector lengths intermediate and register types must match\n");
  }

  if (vecLength<RegType>() == 0) errorQuda("Vector type not supported\n");
  if (vecLength<InterType>() == 0) errorQuda("Vector type not supported\n");
  if (vecLength<StoreType>() == 0) errorQuda("Vector type not supported\n");

}

// FIXME: Can we merge the Spinor and SpinorTexture objects so that
// reading from texture is simply a constructor option?

// the number of elements per virtual register
#define REG_LENGTH (sizeof(RegType) / sizeof(((RegType*)0)->x))

/**
  @param RegType Register type used in kernel
  @param InterType Intermediate format - RegType precision with StoreType ordering
  @param StoreType Type used to store field in memory
  @param N Length of vector of RegType elements that this Spinor represents
  @param tex_id Which texture reference are we using
 */
template <typename RegType, typename InterType, typename StoreType, int N, int tex_id>
class SpinorTexture {

 private:
  Texture<InterType, StoreType, tex_id> spinor;
  /* It's faster to always use direct reads for the norm, but leave
     this option in there for the future.*/
#if (__COMPUTE_CAPABILITY__ >= 000)
  float *norm;
#else
  Texture<float, float, tex_id> norm;
#endif
  int stride;

 public:

#if (__COMPUTE_CAPABILITY__ >= 000)
 SpinorTexture() 
   : spinor((StoreType*)0, 0), norm(0), stride(0) {;} // default constructor

 SpinorTexture(const cudaColorSpinorField &x) 
   : spinor((StoreType*)x.V(), x.Bytes()), norm((float*)x.Norm()),
    stride(x.Length()/(N*REG_LENGTH)) { checkTypes<RegType,InterType,StoreType>(); }
#else
 SpinorTexture()
  : spinor((StoreType*)0, 0), norm(0, 0), stride(0) {;} // default constructor

 SpinorTexture(const cudaColorSpinorField &x) 
   : spinor((StoreType*)x.V(), x.Bytes()), norm((float*)x.Norm(), x.NormBytes()),
    stride(x.Length()/(N*REG_LENGTH)) { checkTypes<RegType,InterType,StoreType>(); }
#endif

  ~SpinorTexture() {;}

  SpinorTexture& operator=(const SpinorTexture &src) {
    if (&src != this) {
      spinor = src.spinor;
      norm = src.norm;
      stride = src.stride;
    }
    return *this;
  }


  __device__ inline void load(RegType x[], const int i) {
    // load data into registers first using the storage order
    const int M = (N * sizeof(RegType)) / sizeof(InterType);
    InterType y[M];

    // half precision types
    if (sizeof(InterType) == 2*sizeof(StoreType)) { 
      float xN = norm[i];
#pragma unroll
      for (int j=0; j<M; j++) {
	y[j] = spinor[i + j*stride];
	y[j] *= xN;
      }
    } else { // other types
#pragma unroll 
      for (int j=0; j<M; j++) copyFloatN(y[j], spinor[i + j*stride]);
    }

    // now convert into desired register order
    convert<RegType, InterType>(x, y, N);
  }

  // no save method for Textures

  QudaPrecision Precision() { 
    QudaPrecision precision = QUDA_INVALID_PRECISION;
    if (sizeof(((StoreType*)0)->x) == sizeof(double)) precision = QUDA_DOUBLE_PRECISION;
    else if (sizeof(((StoreType*)0)->x) == sizeof(float)) precision = QUDA_SINGLE_PRECISION;
    else if (sizeof(((StoreType*)0)->x) == sizeof(short)) precision = QUDA_HALF_PRECISION;
    else errorQuda("Unknown precision type\n");
    return precision;
  }

  int Stride() { return stride; }
};

/**
  @param RegType Register type used in kernel
  @param InterType Intermediate format - RegType precision with StoreType ordering
  @param StoreType Type used to store field in memory
  @param N Length of vector of RegType elements that this Spinor represents
 */
template <typename RegType, typename InterType, typename StoreType, int N>
class Spinor {

 private:
  StoreType *spinor;
  float *norm;
  const int stride;

 public:
 Spinor(cudaColorSpinorField &x) : 
  spinor((StoreType*)x.V()), norm((float*)x.Norm()),  stride(x.Length()/(N*REG_LENGTH))
    { checkTypes<RegType,InterType,StoreType>(); } 

 Spinor(const cudaColorSpinorField &x) :
  spinor((StoreType*)x.V()), norm((float*)x.Norm()), stride(x.Length()/(N*REG_LENGTH))
    { checkTypes<RegType,InterType,StoreType>(); } 
  ~Spinor() {;}

  // default load used for simple fields
  __device__ inline void load(RegType x[], const int i) {
    // load data into registers first
    const int M = (N * sizeof(RegType)) / sizeof(InterType);
    InterType y[M];
#pragma unroll
    for (int j=0; j<M; j++) copyFloatN(y[j],spinor[i + j*stride]);

    convert<RegType, InterType>(x, y, N);
  }

  // default store used for simple fields
  __device__ inline void save(RegType x[], int i) {
    const int M = (N * sizeof(RegType)) / sizeof(InterType);
    InterType y[M];
    convert<InterType, RegType>(y, x, M);
#pragma unroll
    for (int j=0; j<M; j++) copyFloatN(spinor[i+j*stride], y[j]);
  }

  // used to backup the field to the host
  void save(char **spinor_h, char **norm_h, size_t bytes, size_t norm_bytes) {
    *spinor_h = new char[bytes];
    cudaMemcpy(*spinor_h, spinor, bytes, cudaMemcpyDeviceToHost);
    if (norm_bytes > 0) {
      *norm_h = new char[norm_bytes];
      cudaMemcpy(*norm_h, norm, norm_bytes, cudaMemcpyDeviceToHost);
    }
    checkCudaError();
  }

  // restore the field from the host
  void load(char **spinor_h, char **norm_h, size_t bytes, size_t norm_bytes) {
    cudaMemcpy(spinor, *spinor_h, bytes, cudaMemcpyHostToDevice);
    if (norm_bytes > 0) {
      cudaMemcpy(norm, *norm_h, norm_bytes, cudaMemcpyHostToDevice);
      delete(*norm_h);
    }
    delete(*spinor_h);
    checkCudaError();
  }

  void* V() { return (void*)spinor; }
  float* Norm() { return norm; }
  QudaPrecision Precision() { 
    QudaPrecision precision = QUDA_INVALID_PRECISION;
    if (sizeof(((StoreType*)0)->x) == sizeof(double)) precision = QUDA_DOUBLE_PRECISION;
    else if (sizeof(((StoreType*)0)->x) == sizeof(float)) precision = QUDA_SINGLE_PRECISION;
    else if (sizeof(((StoreType*)0)->x) == sizeof(short)) precision = QUDA_HALF_PRECISION;
    else errorQuda("Unknown precision type\n");
    return precision;
  }

  int Stride() { return stride; }
};

template <typename OutputType, typename InputType, int M>
  __device__ inline void saveHalf(OutputType *x_o, float *norm, InputType x_i[M], int i, int stride) {
  float c[M];
#pragma unroll
  for (int j=0; j<M; j++) c[j] = max_fabs(x_i[j]);
#pragma unroll
  for (int j=1; j<M; j++) c[0] = fmaxf(c[j],c[0]);
  
  norm[i] = c[0]; // store norm value

  // store spinor values
  float C = __fdividef(MAX_SHORT, c[0]);  
#pragma unroll
  for (int j=0; j<M; j++) {    
    x_o[i+j*stride] = make_shortN(C*x_i[j]); 
  }
}

template <>
__device__ inline void Spinor<float2, float2, short2, 3>::save(float2 x[3], int i) {
  saveHalf<short2, float2, 3>(spinor, norm, x, i, stride);
}

template <>
__device__ inline void Spinor<float4, float4, short4, 6>::save(float4 x[6], int i) {
  saveHalf<short4, float4, 6>(spinor, norm, x, i, stride);
}

template <>
__device__ inline void Spinor<double2, double2, short2, 3>::save(double2 x[3], int i) {
  saveHalf<short2, double2, 3>(spinor, norm, x, i, stride);
}

template <>
__device__ inline void Spinor<double2, double4, short4, 12>::save(double2 x[12], int i) {
  double4 y[6];
  convert<double4, double2>(y, x, 6);
  saveHalf<short4, double4, 6>(spinor, norm, y, i, stride);
}

#endif // _TEXTURE_H
