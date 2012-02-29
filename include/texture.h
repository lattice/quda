#include <convert.h>

#pragma once

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

texture<short2,1,cudaReadModeNormalizedFloat> tex_short2_0;
texture<short2,1,cudaReadModeNormalizedFloat> tex_short2_1;
texture<short2,1,cudaReadModeNormalizedFloat> tex_short2_2;
texture<short2,1,cudaReadModeNormalizedFloat> tex_short2_3;
texture<short2,1,cudaReadModeNormalizedFloat> tex_short2_4;

texture<short4,1,cudaReadModeNormalizedFloat> tex_short4_0;
texture<short4,1,cudaReadModeNormalizedFloat> tex_short4_1;
texture<short4,1,cudaReadModeNormalizedFloat> tex_short4_2;
texture<short4,1,cudaReadModeNormalizedFloat> tex_short4_3;
texture<short4,1,cudaReadModeNormalizedFloat> tex_short4_4;

texture<float,1> tex_float_0;
texture<float,1> tex_float_1;
texture<float,1> tex_float_2;
texture<float,1> tex_float_3;
texture<float,1> tex_float_4;

texture<float2,1> tex_float2_0;
texture<float2,1> tex_float2_1;
texture<float2,1> tex_float2_2;
texture<float2,1> tex_float2_3;
texture<float2,1> tex_float2_4;

texture<float4,1> tex_float4_0;
texture<float4,1> tex_float4_1;
texture<float4,1> tex_float4_2;
texture<float4,1> tex_float4_3;
texture<float4,1> tex_float4_4;

texture<int4,1> tex_int4_0;
texture<int4,1> tex_int4_1;
texture<int4,1> tex_int4_2;
texture<int4,1> tex_int4_3;
texture<int4,1> tex_int4_4;

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

template<> inline void Texture<float2,short2,0>::bind(const short2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_short2_0, ptr, bytes); }
template<> inline void Texture<float2,short2,1>::bind(const short2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_short2_1, ptr, bytes); }
template<> inline void Texture<float2,short2,2>::bind(const short2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_short2_2, ptr, bytes); }
template<> inline void Texture<float2,short2,3>::bind(const short2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_short2_3, ptr, bytes); }
template<> inline void Texture<float2,short2,4>::bind(const short2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_short2_4, ptr, bytes); }

template<> inline void Texture<float4,short4,0>::bind(const short4 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_short4_0, ptr, bytes); }
template<> inline void Texture<float4,short4,1>::bind(const short4 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_short4_1, ptr, bytes); }
template<> inline void Texture<float4,short4,2>::bind(const short4 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_short4_2, ptr, bytes); }
template<> inline void Texture<float4,short4,3>::bind(const short4 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_short4_3, ptr, bytes); }
template<> inline void Texture<float4,short4,4>::bind(const short4 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_short4_4, ptr, bytes); }

template<> inline void Texture<float,float,0>::bind(const float *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float_0, ptr, bytes); }
template<> inline void Texture<float,float,1>::bind(const float *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float_1, ptr, bytes); }
template<> inline void Texture<float,float,2>::bind(const float *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float_2, ptr, bytes); }
template<> inline void Texture<float,float,3>::bind(const float *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float_3, ptr, bytes); }
template<> inline void Texture<float,float,4>::bind(const float *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float_4, ptr, bytes); }

template<> inline void Texture<float2,float2,0>::bind(const float2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float2_0, ptr, bytes); }
template<> inline void Texture<float2,float2,1>::bind(const float2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float2_1, ptr, bytes); }
template<> inline void Texture<float2,float2,2>::bind(const float2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float2_2, ptr, bytes); }
template<> inline void Texture<float2,float2,3>::bind(const float2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float2_3, ptr, bytes); }
template<> inline void Texture<float2,float2,4>::bind(const float2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float2_4, ptr, bytes); }

template<> inline void Texture<float4,float4,0>::bind(const float4 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float4_0, ptr, bytes); }
template<> inline void Texture<float4,float4,1>::bind(const float4 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float4_1, ptr, bytes); }
template<> inline void Texture<float4,float4,2>::bind(const float4 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float4_2, ptr, bytes); }
template<> inline void Texture<float4,float4,3>::bind(const float4 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float4_3, ptr, bytes); }
template<> inline void Texture<float4,float4,4>::bind(const float4 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_float4_4, ptr, bytes); }

template<> inline void Texture<double2,double2,0>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_0, ptr, bytes); }
template<> inline void Texture<double2,double2,1>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_1, ptr, bytes); }
template<> inline void Texture<double2,double2,2>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_2, ptr, bytes); }
template<> inline void Texture<double2,double2,3>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_3, ptr, bytes); }
template<> inline void Texture<double2,double2,4>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_4, ptr, bytes); }

template<> inline void Texture<float2,double2,0>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_0, ptr, bytes); }
template<> inline void Texture<float2,double2,1>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_1, ptr, bytes); }
template<> inline void Texture<float2,double2,2>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_2, ptr, bytes); }
template<> inline void Texture<float2,double2,3>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_3, ptr, bytes); }
template<> inline void Texture<float2,double2,4>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_4, ptr, bytes); }

template<> inline void Texture<float2,short2,0>::unbind() { cudaUnbindTexture(tex_short2_0); }
template<> inline void Texture<float2,short2,1>::unbind() { cudaUnbindTexture(tex_short2_1); }
template<> inline void Texture<float2,short2,2>::unbind() { cudaUnbindTexture(tex_short2_2); }
template<> inline void Texture<float2,short2,3>::unbind() { cudaUnbindTexture(tex_short2_3); }
template<> inline void Texture<float2,short2,4>::unbind() { cudaUnbindTexture(tex_short2_4); }

template<> inline void Texture<float4,short4,0>::unbind() { cudaUnbindTexture(tex_short4_0); }
template<> inline void Texture<float4,short4,1>::unbind() { cudaUnbindTexture(tex_short4_1); }
template<> inline void Texture<float4,short4,2>::unbind() { cudaUnbindTexture(tex_short4_2); }
template<> inline void Texture<float4,short4,3>::unbind() { cudaUnbindTexture(tex_short4_3); }
template<> inline void Texture<float4,short4,4>::unbind() { cudaUnbindTexture(tex_short4_4); }

template<> inline void Texture<float,float,0>::unbind() { cudaUnbindTexture(tex_float_0); }
template<> inline void Texture<float,float,1>::unbind() { cudaUnbindTexture(tex_float_1); }
template<> inline void Texture<float,float,2>::unbind() { cudaUnbindTexture(tex_float_2); }
template<> inline void Texture<float,float,3>::unbind() { cudaUnbindTexture(tex_float_3); }
template<> inline void Texture<float,float,4>::unbind() { cudaUnbindTexture(tex_float_4); }

template<> inline void Texture<float2,float2,0>::unbind() { cudaUnbindTexture(tex_float2_0); }
template<> inline void Texture<float2,float2,1>::unbind() { cudaUnbindTexture(tex_float2_1); }
template<> inline void Texture<float2,float2,2>::unbind() { cudaUnbindTexture(tex_float2_2); }
template<> inline void Texture<float2,float2,3>::unbind() { cudaUnbindTexture(tex_float2_3); }
template<> inline void Texture<float2,float2,4>::unbind() { cudaUnbindTexture(tex_float2_4); }

template<> inline void Texture<float4,float4,0>::unbind() { cudaUnbindTexture(tex_float4_0); }
template<> inline void Texture<float4,float4,1>::unbind() { cudaUnbindTexture(tex_float4_1); }
template<> inline void Texture<float4,float4,2>::unbind() { cudaUnbindTexture(tex_float4_2); }
template<> inline void Texture<float4,float4,3>::unbind() { cudaUnbindTexture(tex_float4_3); }
template<> inline void Texture<float4,float4,4>::unbind() { cudaUnbindTexture(tex_float4_4); }

template<> inline void Texture<double2,double2,0>::unbind() { cudaUnbindTexture(tex_int4_0); }
template<> inline void Texture<double2,double2,1>::unbind() { cudaUnbindTexture(tex_int4_1); }
template<> inline void Texture<double2,double2,2>::unbind() { cudaUnbindTexture(tex_int4_2); }
template<> inline void Texture<double2,double2,3>::unbind() { cudaUnbindTexture(tex_int4_3); }
template<> inline void Texture<double2,double2,4>::unbind() { cudaUnbindTexture(tex_int4_4); }

template<> inline void Texture<float2,double2,0>::unbind() { cudaUnbindTexture(tex_int4_0); }
template<> inline void Texture<float2,double2,1>::unbind() { cudaUnbindTexture(tex_int4_1); }
template<> inline void Texture<float2,double2,2>::unbind() { cudaUnbindTexture(tex_int4_2); }
template<> inline void Texture<float2,double2,3>::unbind() { cudaUnbindTexture(tex_int4_3); }
template<> inline void Texture<float2,double2,4>::unbind() { cudaUnbindTexture(tex_int4_4); }

// short2
template<> __device__ inline float2 Texture<float2,short2,0>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_short2_0,idx); }
template<> __device__ inline float2 Texture<float2,short2,1>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_short2_1,idx); }
template<> __device__ inline float2 Texture<float2,short2,2>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_short2_2,idx); }
template<> __device__ inline float2 Texture<float2,short2,3>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_short2_3,idx); }
template<> __device__ inline float2 Texture<float2,short2,4>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_short2_4,idx); }

// short4
template<> __device__ inline float4 Texture<float4,short4,0>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_short4_0,idx); }
template<> __device__ inline float4 Texture<float4,short4,1>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_short4_1,idx); }
template<> __device__ inline float4 Texture<float4,short4,2>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_short4_2,idx); }
template<> __device__ inline float4 Texture<float4,short4,3>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_short4_3,idx); }
template<> __device__ inline float4 Texture<float4,short4,4>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_short4_4,idx); }

// float
template<> __device__ inline float Texture<float,float,0>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float_0,idx); }
template<> __device__ inline float Texture<float,float,1>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float_1,idx); }
template<> __device__ inline float Texture<float,float,2>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float_2,idx); }
template<> __device__ inline float Texture<float,float,3>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float_3,idx); }
template<> __device__ inline float Texture<float,float,4>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float_4,idx); }

// float2
template<> __device__ inline float2 Texture<float2,float2,0>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float2_0,idx); }
template<> __device__ inline float2 Texture<float2,float2,1>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float2_1,idx); }
template<> __device__ inline float2 Texture<float2,float2,2>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float2_2,idx); }
template<> __device__ inline float2 Texture<float2,float2,3>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float2_3,idx); }
template<> __device__ inline float2 Texture<float2,float2,4>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float2_4,idx); }

// float4
template<> __device__ inline float4 Texture<float4,float4,0>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float4_0,idx); }
template<> __device__ inline float4 Texture<float4,float4,1>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float4_1,idx); }
template<> __device__ inline float4 Texture<float4,float4,2>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float4_2,idx); }
template<> __device__ inline float4 Texture<float4,float4,3>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float4_3,idx); }
template<> __device__ inline float4 Texture<float4,float4,4>::fetch(unsigned int idx) 
{ return tex1Dfetch(tex_float4_4,idx); }

// double2
#ifndef FERMI_NO_DBLE_TEX
template<> __device__ inline double2 Texture<double2,double2,0>::fetch(unsigned int idx) 
{ return fetch_double2(tex_int4_0,idx); }
template<> __device__ inline double2 Texture<double2,double2,1>::fetch(unsigned int idx) 
{ return fetch_double2(tex_int4_1,idx); }
template<> __device__ inline double2 Texture<double2,double2,2>::fetch(unsigned int idx) 
{ return fetch_double2(tex_int4_2,idx); }
template<> __device__ inline double2 Texture<double2,double2,3>::fetch(unsigned int idx) 
{ return fetch_double2(tex_int4_3,idx); }
template<> __device__ inline double2 Texture<double2,double2,4>::fetch(unsigned int idx) 
{ return fetch_double2(tex_int4_4,idx); }

template<> __device__ inline float2 Texture<float2,double2,0>::fetch(unsigned int idx) 
{ double2 x = fetch_double2(tex_int4_0,idx); return make_float2(x.x, x.y); }
template<> __device__ inline float2 Texture<float2,double2,1>::fetch(unsigned int idx) 
{ double2 x = fetch_double2(tex_int4_1,idx); return make_float2(x.x, x.y); }
template<> __device__ inline float2 Texture<float2,double2,2>::fetch(unsigned int idx) 
{ double2 x = fetch_double2(tex_int4_2,idx); return make_float2(x.x, x.y); }
template<> __device__ inline float2 Texture<float2,double2,3>::fetch(unsigned int idx) 
{ double2 x = fetch_double2(tex_int4_3,idx); return make_float2(x.x, x.y); }
template<> __device__ inline float2 Texture<float2,double2,4>::fetch(unsigned int idx) 
{ double2 x = fetch_double2(tex_int4_4,idx); return make_float2(x.x, x.y); }

#else

template<> __device__ inline double2 Texture<double2,double2,0>::fetch(unsigned int idx) 
{ return spinor[idx]; }
template<> __device__ inline double2 Texture<double2,double2,1>::fetch(unsigned int idx) 
{ return spinor[idx]; }
template<> __device__ inline double2 Texture<double2,double2,2>::fetch(unsigned int idx) 
{ return spinor[idx]; }
template<> __device__ inline double2 Texture<double2,double2,3>::fetch(unsigned int idx) 
{ return spinor[idx]; }
template<> __device__ inline double2 Texture<double2,double2,4>::fetch(unsigned int idx) 
{ return spinor[idx]; }

template<> __device__ inline float2 Texture<float2,double2,0>::fetch(unsigned int idx) 
{ double2 x = spinor[idx]; return make_float2(x.x, x.y); }
template<> __device__ inline float2 Texture<float2,double2,1>::fetch(unsigned int idx) 
{ double2 x = spinor[idx]; return make_float2(x.x, x.y); }
template<> __device__ inline float2 Texture<float2,double2,2>::fetch(unsigned int idx) 
{ double2 x = spinor[idx]; return make_float2(x.x, x.y); }
template<> __device__ inline float2 Texture<float2,double2,3>::fetch(unsigned int idx) 
{ double2 x = spinor[idx]; return make_float2(x.x, x.y); }
template<> __device__ inline float2 Texture<float2,double2,4>::fetch(unsigned int idx) 
{ double2 x = spinor[idx]; return make_float2(x.x, x.y); }

#endif //  FERMI_NO_DBLE_TEX

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

