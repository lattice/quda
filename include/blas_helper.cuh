#pragma once

#include <color_spinor_field.h>
#include <reduce_helper.h>
#include <load_store.h>

//#define QUAD_SUM
#ifdef QUAD_SUM
#include <dbldbl.h>
#endif

namespace quda
{

  template <bool X_ = false, bool Y_ = false, bool Z_ = false, bool W_ = false, bool V_ = false> struct memory_access {
    static constexpr bool X = X_;
    static constexpr bool Y = Y_;
    static constexpr bool Z = Z_;
    static constexpr bool W = W_;
    static constexpr bool V = V_;
  };

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
  template <> struct VectorType<double, 24> {
    using type = vector_type<double, 24>;
  };
  template <> struct VectorType<float, 24> {
    using type = vector_type<float, 24>;
  };
  template <> struct VectorType<short, 24> {
    using type = vector_type<short, 24>;
  };
  template <> struct VectorType<int8_t, 24> {
    using type = vector_type<int8_t, 24>;
  };
  template <> struct VectorType<double, 6> {
    using type = vector_type<double, 6>;
  };
  template <> struct VectorType<float, 6> {
    using type = vector_type<float, 6>;
  };
  template <> struct VectorType<short, 6> {
    using type = vector_type<short, 6>;
  };
  template <> struct VectorType<int8_t, 6> {
    using type = vector_type<int8_t, 6>;
  };

  namespace blas
  {

    template <typename store_t, bool is_fixed> struct SpinorNorm {
      using norm_t = float;
      norm_t *norm;
      unsigned int cb_norm_offset;

      SpinorNorm() : norm(nullptr), cb_norm_offset(0) {}

      SpinorNorm(const ColorSpinorField &x) :
        norm((norm_t *)x.Norm()),
        cb_norm_offset(x.NormBytes() / (2 * sizeof(norm_t)))
      {
      }

      void set(const ColorSpinorField &x)
      {
        norm = (norm_t *)x.Norm();
        cb_norm_offset = x.NormBytes() / (2 * sizeof(norm_t));
      }

      __device__ __host__ inline norm_t load_norm(const int i, const int parity = 0) const
      {
        return norm[cb_norm_offset * parity + i];
      }

      template <typename real, int n>
      __device__ __host__ inline norm_t store_norm(const vector_type<complex<real>, n> &v, int x, int parity)
      {
        norm_t max_[n];
        // two-pass to increase ILP (assumes length divisible by two, e.g. complex-valued)
#pragma unroll
        for (int i = 0; i < n; i++) max_[i] = fmaxf(fabsf((norm_t)v[i].real()), fabsf((norm_t)v[i].imag()));
        norm_t scale = 0.0;
#pragma unroll
        for (int i = 0; i < n; i++) scale = fmaxf(max_[i], scale);
        norm[x + parity * cb_norm_offset] = scale;

        return fdividef(fixedMaxValue<store_t>::value, scale);
      }

      norm_t *Norm() { return norm; }
    };

    template <typename store_type_t> struct SpinorNorm<store_type_t, false> {
      using norm_t = float;
      SpinorNorm() {}
      SpinorNorm(const ColorSpinorField &) {}
      void set(const ColorSpinorField &) {}
      __device__ __host__ inline norm_t load_norm(const int, const int = 0) const { return 1.0; }
      template <typename real, int n>
      __device__ __host__ inline norm_t store_norm(const vector_type<complex<real>, n> &, int, int)
      {
        return 1.0;
      }
      void backup(char **, size_t) {}
      void restore(char **, size_t) {}
      norm_t *Norm() { return nullptr; }
    };

    /**
       @param RegType Register type used in kernel
       @param InterType Intermediate format - RegType precision with StoreType ordering
       @param StoreType Type used to store field in memory
       @param N Length of vector of RegType elements that this Spinor represents
    */
    template <typename store_t, int N> struct Spinor : SpinorNorm<store_t, isFixed<store_t>::value> {
      using SN = SpinorNorm<store_t, isFixed<store_t>::value>;
      using Vector = typename VectorType<store_t, N>::type;
      store_t *spinor;
      int stride;
      unsigned int cb_offset;

      Spinor() : SN(), spinor(nullptr), stride(0), cb_offset(0) {}

      Spinor(const ColorSpinorField &x) :
        SN(x),
        spinor(static_cast<store_t *>(const_cast<ColorSpinorField &>(x).V())),
        stride(x.Stride()),
        cb_offset(x.Bytes() / (2 * sizeof(store_t) * N))
      {
      }

      void set(const ColorSpinorField &x)
      {
        SN::set(x);
        spinor = static_cast<store_t *>(const_cast<ColorSpinorField &>(x).V());
        stride = x.Stride();
        cb_offset = x.Bytes() / (2 * sizeof(store_t) * N);
      }

      template <typename real, int n>
      __device__ __host__ inline void load(vector_type<complex<real>, n> &v, int x, int parity = 0) const
      {
        constexpr int len = 2 * n; // real-valued length
        float nrm = isFixed<store_t>::value ? SN::load_norm(x, parity) : 0.0;

        vector_type<real, len> v_;

        constexpr int M = len / N;
#pragma unroll
        for (int i = 0; i < M; i++) {
          // first load from memory
          Vector vecTmp = vector_load<Vector>(spinor, parity * cb_offset + x + stride * i);
          // now copy into output and scale
#pragma unroll
          for (int j = 0; j < N; j++) copy_and_scale(v_[i * N + j], reinterpret_cast<store_t *>(&vecTmp)[j], nrm);
        }

        for (int i = 0; i < n; i++) { v[i] = complex<real>(v_[2 * i + 0], v_[2 * i + 1]); }
      }

      template <typename real, int n>
      __device__ __host__ inline void save(const vector_type<complex<real>, n> &v, int x, int parity = 0)
      {
        constexpr int len = 2 * n; // real-valued length
        vector_type<real, len> v_;

        if (isFixed<store_t>::value) {
          real scale_inv = SN::template store_norm<real, n>(v, x, parity);
#pragma unroll
          for (int i = 0; i < n; i++) {
            v_[2 * i + 0] = scale_inv * v[i].real();
            v_[2 * i + 1] = scale_inv * v[i].imag();
          }
        } else {
#pragma unroll
          for (int i = 0; i < n; i++) {
            v_[2 * i + 0] = v[i].real();
            v_[2 * i + 1] = v[i].imag();
          }
        }

        constexpr int M = len / N;
#pragma unroll
        for (int i = 0; i < M; i++) {
          Vector vecTmp;
          // first do scalar copy converting into storage type
#pragma unroll
          for (int j = 0; j < N; j++) copy_scaled(reinterpret_cast<store_t *>(&vecTmp)[j], v_[i * N + j]);
          // second do vectorized copy into memory
          vector_store(spinor, parity * cb_offset + x + stride * i, vecTmp);
        }
      }
    };

    // n_vector defines the granularity of load/store, e.g., sets the
    // size of vector we load from memory
    template <typename store_t, bool GPU, int nSpin, bool site_unroll> constexpr int n_vector() { return 0; }

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
    template <> constexpr int n_vector<int8_t, true, 4, true>() { return 8; }
#else
    template <> constexpr int n_vector<int8_t, true, 4, true>() { return 4; }
#endif
    template <> constexpr int n_vector<int8_t, true, 1, true>() { return 2; }

    // Just use float-2/float-4 ordering on CPU when not site unrolling
    template <> constexpr int n_vector<double, false, 4, false>() { return 2; }
    template <> constexpr int n_vector<double, false, 1, false>() { return 2; }
    template <> constexpr int n_vector<float, false, 4, false>() { return 4; }
    template <> constexpr int n_vector<float, false, 1, false>() { return 4; }

    // AoS ordering is used on CPU uses when we are site unrolling
    template <> constexpr int n_vector<double, false, 4, true>() { return 24; }
    template <> constexpr int n_vector<double, false, 1, true>() { return 6; }
    template <> constexpr int n_vector<float, false, 4, true>() { return 24; }
    template <> constexpr int n_vector<float, false, 1, true>() { return 6; }
    template <> constexpr int n_vector<short, false, 4, true>() { return 24; }
    template <> constexpr int n_vector<short, false, 1, true>() { return 6; }
    template <> constexpr int n_vector<int8_t, false, 4, true>() { return 24; }
    template <> constexpr int n_vector<int8_t, false, 1, true>() { return 6; }

    template <template <typename...> class Functor,
              template <template <typename...> class, typename store_t, typename y_store_t, int, typename> class Blas,
              typename T, typename store_t, typename y_store_t, typename V, typename... Args>
#if defined(NSPIN1) || defined(NSPIN2) || defined(NSPIN4)
    constexpr void instantiate(const T &a, const T &b, const T &c, V &x, Args &&... args)
    {
      if (x.Nspin() == 4 || x.Nspin() == 2) {
#if defined(NSPIN4) || defined(NSPIN2)
        // Nspin-2 takes Nspin-4 path here, and we check for this later
        Blas<Functor, store_t, y_store_t, 4, T>(a, b, c, x, args...);
#else
        errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      } else {
#if defined(NSPIN1)
        Blas<Functor, store_t, y_store_t, 1, T>(a, b, c, x, args...);
#else
        errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
#endif
      }
    }
#else
    constexpr void instantiate(const T &, const T &, const T &, V &x, Args &&...)
    {
      errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
    }
#endif

    // The instantiate helpers are used to instantiate the precision
    // and spin for the blas and reduce kernels

    template <template <typename...> class Functor,
              template <template <typename...> class, typename store_t, typename y_store_t, int, typename> class Blas,
              bool mixed, typename T, typename store_t, typename V, typename... Args>
    constexpr std::enable_if_t<!mixed, void> instantiate(const T &a, const T &b, const T &c, V &x,
                                                         Args &&... args)
    {
      return instantiate<Functor, Blas, T, store_t, store_t>(a, b, c, x, args...);
    }

    template <template <typename...> class Functor,
              template <template <typename...> class, typename store_t, typename y_store_t, int, typename> class Blas,
              bool mixed, typename T, typename x_store_t, typename V, typename... Args>
    constexpr std::enable_if_t<mixed, void> instantiate(const T &a, const T &b, const T &c, V &x, V &y,
                                                        Args &&... args)
    {
      if (y.Precision() < x.Precision()) errorQuda("Y precision %d not supported", y.Precision());

      // use PromoteType to ensure we don't instantiate unwanted combinations (e.g., x > y)
      if (y.Precision() == QUDA_DOUBLE_PRECISION) {

#if !(QUDA_PRECISION & 8)
        if (x.Location() == QUDA_CUDA_FIELD_LOCATION)
          errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
#endif
        // always instantiate the double-precision template to allow CPU
        // fields through, and prevent double-precision GPU
        // instantiation using gpu_mapper
        instantiate<Functor, Blas, T, x_store_t, double>(a, b, c, x, y, args...);

      } else if (y.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
        instantiate<Functor, Blas, T, x_store_t, typename PromoteTypeId<x_store_t, float>::type>(a, b, c, x, y,
                                                                                                 args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
      } else if (y.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
        instantiate<Functor, Blas, T, x_store_t, typename PromoteTypeId<x_store_t, short>::type>(a, b, c, x, y,
                                                                                                 args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else if (y.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
        instantiate<Functor, Blas, T, x_store_t, typename PromoteTypeId<x_store_t, int8_t>::type>(a, b, c, x, y,
                                                                                                args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Unsupported precision %d\n", y.Precision());
      }
    }

    template <template <typename...> class Functor,
              template <template <typename...> class, typename store_t, typename y_store_t, int, typename> class Blas,
              bool mixed, typename T, typename V, typename... Args>
    constexpr void instantiate(const T &a, const T &b, const T &c, V &x, Args &&... args)
    {
      if (x.Precision() == QUDA_DOUBLE_PRECISION) {
#if !(QUDA_PRECISION & 8)
        if (x.Location() == QUDA_CUDA_FIELD_LOCATION)
          errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
#endif
        // always instantiate the double-precision template to allow CPU
        // fields through, and prevent double-precision GPU
        // instantiation using double_mapper
        instantiate<Functor, Blas, mixed, T, double>(a, b, c, x, args...);
      } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
        instantiate<Functor, Blas, mixed, T, float>(a, b, c, x, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
      } else if (x.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
        instantiate<Functor, Blas, mixed, T, short>(a, b, c, x, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else if (x.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
        instantiate<Functor, Blas, mixed, T, int8_t>(a, b, c, x, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Unsupported precision %d\n", x.Precision());
      }
    }

    /**
       @brief device_type_mapper In general we want to enable double
       precision blas always on the host, e.g., for running unit tests,
       but may not want to build double precision on the device, e.g., if
       we have a pure single precision build with QUDA_PRECISION=4.
       Thus we do not prevent the double precision template from being
       instantiated when the field precision is queried, but we can
       use device_type_mapper to demote the type prior to any kernel
       being instantiated.
     */
    template <typename T> struct device_type_mapper { using type = T; };
    template <> struct device_type_mapper<double> {
#if QUDA_PRECISION & 8
      using type = double;
#elif QUDA_PRECISION & 4
      using type = float;
#elif QUDA_PRECISION & 2
      using type = short;
#elif QUDA_PRECISION & 1
      using type = int8_t;
#endif
    };

    /**
      @brief host_type_mapper At present we do not support half or
      quarter precision on the host target.  Thus we use
      host_type_mapper to promote any half/quarter precision type to
      double or single to prevent the kernel prior to any kernel being
      instantiated to reduce template bloat.
     */
    template <typename T> struct host_type_mapper { using type = T; };
    template <> struct host_type_mapper<short> {
#if QUDA_PRECISION & 4
      using type = float;
#else
      using type = double;
#endif
    };
    template <> struct host_type_mapper<int8_t> {
#if QUDA_PRECISION & 4
      using type = float;
#else
      using type = double;
#endif
    };

  } // namespace blas

} // namespace quda
