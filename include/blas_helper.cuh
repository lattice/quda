#pragma once

#include <color_spinor_field.h>

//#define QUAD_SUM
#ifdef QUAD_SUM
#include <dbldbl.h>
#endif

// these definitions are used to avoid calling
// std::complex<type>::real/imag which have C++11 ABI incompatibility
// issues with certain versions of GCC

#define REAL(a) (*((double *)&a))
#define IMAG(a) (*((double *)&a + 1))

namespace quda
{

  inline void checkSpinor(const ColorSpinorField &a, const ColorSpinorField &b)
  {
    if (a.Length() != b.Length()) errorQuda("lengths do not match: %lu %lu", a.Length(), b.Length());
    if (a.Stride() != b.Stride()) errorQuda("strides do not match: %lu %lu", a.Stride(), b.Stride());
  }

  inline void checkLength(const ColorSpinorField &a, const ColorSpinorField &b)
  {
    if (a.Length() != b.Length()) errorQuda("lengths do not match: %lu %lu", a.Length(), b.Length());
    if (a.Stride() != b.Stride()) errorQuda("strides do not match: %lu %lu", a.Stride(), b.Stride());
  }

#ifdef QUAD_SUM
#define QudaSumFloat doubledouble
#define QudaSumFloat2 doubledouble2
#define QudaSumFloat3 doubledouble3
  template <> struct scalar<doubledouble> {
    typedef doubledouble type;
  };
  template <> struct scalar<doubledouble2> {
    typedef doubledouble type;
  };
  template <> struct scalar<doubledouble3> {
    typedef doubledouble type;
  };
  template <> struct scalar<doubledouble4> {
    typedef doubledouble type;
  };
  template <> struct vector<doubledouble, 2> {
    typedef doubledouble2 type;
  };
#else
#define QudaSumFloat double
#define QudaSumFloat2 double2
#define QudaSumFloat3 double3
#define QudaSumFloat4 double4
#endif

  __host__ __device__ inline double set(double &x) { return x; }
  __host__ __device__ inline double2 set(double2 &x) { return x; }
  __host__ __device__ inline double3 set(double3 &x) { return x; }
  __host__ __device__ inline double4 set(double4 &x) { return x; }
  __host__ __device__ inline void sum(double &a, double &b) { a += b; }
  __host__ __device__ inline void sum(double2 &a, double2 &b)
  {
    a.x += b.x;
    a.y += b.y;
  }
  __host__ __device__ inline void sum(double3 &a, double3 &b)
  {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
  }
  __host__ __device__ inline void sum(double4 &a, double4 &b)
  {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
  }

#ifdef QUAD_SUM
  __host__ __device__ inline double set(doubledouble &a) { return a.head(); }
  __host__ __device__ inline double2 set(doubledouble2 &a) { return make_double2(a.x.head(), a.y.head()); }
  __host__ __device__ inline double3 set(doubledouble3 &a) { return make_double3(a.x.head(), a.y.head(), a.z.head()); }
  __host__ __device__ inline void sum(double &a, doubledouble &b) { a += b.head(); }
  __host__ __device__ inline void sum(double2 &a, doubledouble2 &b)
  {
    a.x += b.x.head();
    a.y += b.y.head();
  }
  __host__ __device__ inline void sum(double3 &a, doubledouble3 &b)
  {
    a.x += b.x.head();
    a.y += b.y.head();
    a.z += b.z.head();
  }
#endif

} // namespace quda
