#include <quda_define.h>

#if defined(QUDA_TARGET_CUDA)

inline
__host__ __device__ void quda_sincos(double x, double *s, double *c) { sincos(x,s,c); }

inline
__host__ __device__ void quda_sincos( 
#elif defined(QUDA_TARGET_HIP)

#include <cmath>
#include <hip/math_functions.h>

inline 
__host__ __device__ void quda_sincos(double x, double *s, double *c) 
{
#if defined(_HIP_DEVICE_COMPILE_)
   sincos(x,s,c);
#else
   *s = std::sin(x);
   *c = std::cos(x);
#endif
}

inline
__host__ __device__ void quda_sincos(float x, float *s, float *c)
{
#if defined(_HIP_DEVICE_COMPILE_)
   sincosf(x,s,c);
#else
   *s = std::sin(x);
   *c = std::cos(x);
#endif
}
#else
#error "QUDA TARGET must be CUDA or HIP"
#endif
