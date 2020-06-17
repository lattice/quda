#pragma once

#include <color_spinor_field.h>

//#define QUAD_SUM
#ifdef QUAD_SUM
#include <dbldbl.h>
#endif

namespace quda
{

  template <bool X_ = false, bool Y_ = false, bool Z_ = false, bool W_ = false, bool V_ = false> struct write {
    static constexpr bool X = X_;
    static constexpr bool Y = Y_;
    static constexpr bool Z = Z_;
    static constexpr bool W = W_;
    static constexpr bool V = V_;
  };

#ifdef QUAD_SUM
  using device_reduce_t = doubledouble;
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
  using device_reduce_t = double;
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

  // Vector types used for AoS load-store on CPU
  template <> struct VectorType<double, 24> { using type = vector_type<double, 24>; };
  template <> struct VectorType<float, 24> { using type = vector_type<float, 24>; };
  template <> struct VectorType<double, 6> { using type = vector_type<double, 6>; };
  template <> struct VectorType<float, 6> { using type = vector_type<float, 6>; };

  namespace blas {

    // n_vector defines the granularity of load/store, e.g., sets the
    // size of vector we load from memory
    template <typename store_t, bool GPU, int nSpin, bool site_unroll>
    constexpr int n_vector() { return 0; }

    // native ordering
    template <> constexpr int n_vector<double, true, 4, false>() { return 2; }
    template <> constexpr int n_vector<double, true, 1, false>() { return 2; }

    template <> constexpr int n_vector<double, true, 4, true>() { return 2; }
    template <> constexpr int n_vector<double, true, 1, true>() { return 2; }

    template <> constexpr int n_vector<float, true, 4, false>() { return 4; }
    template <> constexpr int n_vector<float, true, 1, false>() { return 4; }

    template <> constexpr int n_vector<float, true, 4, true>() { return 4; }
    template <> constexpr int n_vector<float, true, 1, true>() { return 2; }

#ifdef FLOAT8
    template <> constexpr int n_vector<short, true, 4, true>() { return 8; }
#else
    template <> constexpr int n_vector<short, true, 4, true>() { return 4; }
#endif
    template <> constexpr int n_vector<short, true, 1, true>() { return 2; }

#ifdef FLOAT8
    template <> constexpr int n_vector<char, true, 4, true>() { return 8; }
#else
    template <> constexpr int n_vector<char, true, 4, true>() { return 4; }
#endif
    template <> constexpr int n_vector<char, true, 1, true>() { return 2; }

    // AoS ordering used on CPU uses when we are not site unrolling
    template <> constexpr int n_vector<double, false, 4, false>() { return 2; }
    template <> constexpr int n_vector<double, false, 1, false>() { return 2; }

    template <> constexpr int n_vector<float, false, 4, false>() { return 4; }
    template <> constexpr int n_vector<float, false, 1, false>() { return 4; }


    template <template <typename...> class Functor,
              template <template <typename...> class, typename store_t, typename y_store_t, int, typename, int> class Blas,
              typename T, typename store_t, typename y_store_t, int NXZ = 1, typename V, typename... Args>
    constexpr void instantiate(const T &a, const T &b, const T &c, V &x, Args &&... args)
    {
      if (x.Nspin() == 4 || x.Nspin() == 2) {
#if defined(NSPIN4) || defined(NSPIN2)
        // Nspin-2 takes Nspin-4 path here, and we check for this later
        Blas<Functor, store_t, y_store_t, 4, T, NXZ>(a, b, c, x, args...);
#else
        errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else {
#if defined(NSPIN1)
        Blas<Functor, store_t, y_store_t, 1, T, NXZ>(a, b, c, x, args...);
#else
        errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      }
    }

    // The instantiate helpers are used to instantiate the precision
    // and spin for the blas and reduce kernels

    template <template <typename...> class Functor,
              template <template <typename...> class, typename store_t, typename y_store_t, int, typename, int> class Blas,
              bool mixed, typename T, typename store_t, int NXZ = 1, typename V, typename... Args>
    constexpr typename std::enable_if<!mixed, void>::type instantiate(const T &a, const T &b, const T &c, V &x, Args &&... args)
    {
      return instantiate<Functor, Blas, T, store_t, store_t, NXZ>(a, b, c, x, args...);
    }

    template <template <typename...> class Functor,
              template <template <typename...> class, typename store_t, typename y_store_t, int, typename, int> class Blas,
              bool mixed, typename T, typename x_store_t, int NXZ = 1, typename V, typename... Args>
    constexpr typename std::enable_if<mixed, void>::type instantiate(const T &a, const T &b, const T &c, V &x, V &y, Args &&... args)
    {
      if (y.Precision() < x.Precision()) errorQuda("Y precision %d not supported", y.Precision());

      // use PromoteType to ensure we don't instantiate unwanted combinations (e.g., x > y)
      if (y.Precision() == QUDA_DOUBLE_PRECISION) {
#if QUDA_PRECISION & 8
        instantiate<Functor, Blas, T, x_store_t, double, NXZ>(a, b, c, x, y, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
#endif
      } else if (y.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
        instantiate<Functor, Blas, T, x_store_t, typename PromoteTypeId<x_store_t, float>::type, NXZ>(a, b, c, x, y, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
      } else if (y.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
        instantiate<Functor, Blas, T, x_store_t, typename PromoteTypeId<x_store_t, short>::type, NXZ>(a, b, c, x, y, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else if (y.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
        instantiate<Functor, Blas, T, x_store_t, typename PromoteTypeId<x_store_t, char>::type, NXZ>(a, b, c, x, y, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Unsupported precision %d\n", y.Precision());
      }
    }

    template <template <typename ...> class Functor,
              template <template <typename ...> class, typename store_t, typename y_store_t, int, typename, int> class Blas,
              bool mixed, int NXZ = 1, typename T, typename V, typename... Args>
    constexpr void instantiate(const T &a, const T &b, const T &c, V &x, Args &&... args)
    {
      if (x.Precision() == QUDA_DOUBLE_PRECISION) {
#if QUDA_PRECISION & 8 || 1
        instantiate<Functor, Blas, mixed, T, double, NXZ>(a, b, c, x, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
#endif
      } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
        instantiate<Functor, Blas, mixed, T, float, NXZ>(a, b, c, x, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
      } else if (x.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
        instantiate<Functor, Blas, mixed, T, short, NXZ>(a, b, c, x, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else if (x.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
        instantiate<Functor, Blas, mixed, T, char, NXZ>(a, b, c, x, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Unsupported precision %d\n", x.Precision());
      }
    }

  } // namespace blas

} // namespace quda
