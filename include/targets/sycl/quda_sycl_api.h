#pragma once

#include <quda_sycl.h>

namespace quda
{
  namespace device
  {
    sycl::queue get_target_stream(const qudaStream_t &stream);
    sycl::queue defaultQueue(void);
  }
  namespace target {
    namespace sycl {
      void set_error(std::string error_str, const char *api_func, const char *func,
		     const char *file, const char *line, bool allow_error);
    }
  }
}

///// MATH

#include <math_helper.cuh>

//#define rsqrt(x) (1/sqrt(x))
//inline float rsqrt(float x) { return 1.0f/sqrt(x); }
//inline void sincos(float x, float *s, float *c)
//{
  //*s = sin(x);
  //*c = cos(x);
  //*s = sycl::sincos(x, c);
//}

inline float fdividef(float a, float b) { return quda::fdividef(a,b); }

