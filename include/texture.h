#pragma once

#define MAX_ELEMENTS (1<<27)
texture<int4,1> tex_int4_0;
texture<int4,1> tex_int4_1;
texture<int4,1> tex_int4_2;
texture<int4,1> tex_int4_3;
texture<int4,1> tex_int4_4;

template<class ScalarType, int tex_id=0>
class Texture {
 public:
 Texture(const ScalarType *x, size_t bytes) { bind(x, bytes); }
 ~Texture() { unbind(); }

 inline void bind(const ScalarType*, size_t bytes){ errorQuda("Texture id is out of range"); }
 inline void unbind() { errorQuda("Texture id is out of range"); }

 __device__ inline ScalarType fetch(unsigned int idx) { return -99; };  //default should only be called if a tex_id is out of range

 __device__ inline ScalarType operator[](unsigned int idx) { return fetch(idx); }
};

template<> inline void Texture<double2,0>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_0, ptr, bytes); }
template<> inline void Texture<double2,1>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_1, ptr, bytes); }
template<> inline void Texture<double2,2>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_2, ptr, bytes); }
template<> inline void Texture<double2,3>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_3, ptr, bytes); }
template<> inline void Texture<double2,4>::bind(const double2 *ptr, size_t bytes) 
{ cudaBindTexture(0,tex_int4_4, ptr, bytes); }

template<> inline void Texture<double2,0>::unbind() { cudaUnbindTexture(tex_int4_0); }
template<> inline void Texture<double2,1>::unbind() { cudaUnbindTexture(tex_int4_1); }
template<> inline void Texture<double2,2>::unbind() { cudaUnbindTexture(tex_int4_2); }
template<> inline void Texture<double2,3>::unbind() { cudaUnbindTexture(tex_int4_3); }
template<> inline void Texture<double2,4>::unbind() { cudaUnbindTexture(tex_int4_4); }

#if (__COMPUTE_CAPABILITY__ < 130) // double precision not supported
__device__ double __hiloint2double(int hi, int lo) { return 0.0; }
#endif

// double2
template<> __device__ inline double2 Texture<double2,0>::fetch(unsigned int idx) 
{ int4 v = tex1Dfetch(tex_int4_0,idx); 
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z)); }
template<> __device__ inline double2 Texture<double2,1>::fetch(unsigned int idx) 
{ int4 v = tex1Dfetch(tex_int4_1,idx); 
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z)); }
template<> __device__ inline double2 Texture<double2,2>::fetch(unsigned int idx) 
{ int4 v = tex1Dfetch(tex_int4_2,idx); 
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z)); }
template<> __device__ inline double2 Texture<double2,3>::fetch(unsigned int idx) 
{ int4 v = tex1Dfetch(tex_int4_3,idx); 
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z)); }
template<> __device__ inline double2 Texture<double2,4>::fetch(unsigned int idx) 
{ int4 v = tex1Dfetch(tex_int4_4,idx); 
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z)); }
