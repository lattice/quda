#pragma once

#include <quda_sycl.h>

namespace quda
{
  extern void *arg_buf;
  extern size_t arg_buf_size;
  static inline void *get_arg_buf(size_t size)
  {
    if(size > arg_buf_size) {
      if(arg_buf!=nullptr) device_free(arg_buf);
      arg_buf = device_malloc(size);
      arg_buf_size = size;
    }
    return arg_buf;
  }

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

#if 0
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
#endif
