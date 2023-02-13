#pragma once

#include <color_spinor_field.h>
#include <load_store.h>
#include <convert.h>
#include <float_vector.h>
#include <array.h>
#include <math_helper.cuh>

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
    using type = array<double, 24>;
  };
  template <> struct VectorType<float, 24> {
    using type = array<float, 24>;
  };
  template <> struct VectorType<short, 24> {
    using type = array<short, 24>;
  };
  template <> struct VectorType<int8_t, 24> {
    using type = array<int8_t, 24>;
  };
  template <> struct VectorType<double, 6> {
    using type = array<double, 6>;
  };
  template <> struct VectorType<float, 6> {
    using type = array<float, 6>;
  };
  template <> struct VectorType<short, 6> {
    using type = array<short, 6>;
  };
  template <> struct VectorType<int8_t, 6> {
    using type = array<int8_t, 6>;
  };

  namespace blas
  {

    /**
       Helper struct that contains the meta data required for
       read and writing to a spinor field in the BLAS kernels.
       @tparam store_t Type used to store field in memory
       @tparam N Length of vector
    */
    template <typename store_t, int N, bool is_fixed> struct data_t {
      store_t *spinor;
      int stride;
      unsigned int cb_offset;
      data_t() :
        spinor(nullptr),
        stride(0),
        cb_offset(0)
      {}

      data_t(const ColorSpinorField &x) :
        spinor(static_cast<store_t *>(const_cast<ColorSpinorField &>(x).V())),
        stride(x.VolumeCB()),
        cb_offset(x.Bytes() / (2 * sizeof(store_t) * N))
      {}
    };

    /**
       Helper struct that contains the meta data required for read and
       writing to a spinor field in the BLAS kernels.  This is a
       specialized variant for fixed-point fields where need to store
       the meta data for the norm field.
       @tparam store_t Type used to store field in memory
       @tparam N Length of vector
    */
    template <typename store_t, int N> struct data_t<store_t, N, true> {
      using norm_t = float;
      store_t *spinor;
      norm_t *norm;
      int stride;
      unsigned int cb_offset;
      unsigned int cb_norm_offset;
      data_t() :
        spinor(nullptr),
        norm(nullptr),
        stride(0),
        cb_offset(0),
        cb_norm_offset(0)
      {}

      data_t(const ColorSpinorField &x) :
        spinor(static_cast<store_t *>(const_cast<ColorSpinorField &>(x).V())),
        norm(static_cast<norm_t *>(const_cast<ColorSpinorField &>(x).Norm())),
        stride(x.VolumeCB()),
        cb_offset(x.Bytes() / (2 * sizeof(store_t) * N)),
        cb_norm_offset(x.Bytes() / (2 * sizeof(norm_t)))
      {}
    };

    /**
       Specialized accessor struct for the BLAS kernels.
       @tparam store_t Type used to store field in memory
       @tparam N Length of vector
    */
    template <typename store_t, int N> struct Spinor {
      using Vector = typename VectorType<store_t, N>::type;
      using norm_t = float;
      data_t<store_t, N, isFixed<store_t>::value> data;

      Spinor() {}

      Spinor(const ColorSpinorField &x) : data(x) {}

      /**
         @brief Dummy implementation of load_norm for non-fixed-point fields
         @tparam is_fixed Whether fixed point
      */
      template <bool is_fixed>
      __device__ __host__ inline std::enable_if_t<!is_fixed, norm_t> load_norm(const int, const int = 0) const { return 1.0; }

      /**
         @brief Implementation of load_norm for fixed-point fields
         @tparam is_fixed Whether fixed point
         @param[in] i checkerboard site index
         @param[in] parity site parity
      */
      template <bool is_fixed>
      __device__ __host__ inline std::enable_if_t<is_fixed, norm_t> load_norm(const int x, const int parity = 0) const
      {
        return data.norm[data.cb_norm_offset * parity + x];
      }

      /**
         @brief Dummy implementation of store_norm for non fixed-point fields
         @tparam is_fixed Whether fixed point
         @tparam real Precision of vector we wish to store from
         @tparam n Complex vector length
      */
      template <bool is_fixed, typename real, int n>
      __device__ __host__ inline std::enable_if_t<!is_fixed, norm_t> store_norm(const array<complex<real>, n> &, norm_t &) const
      {
        return 1.0;
      }

      /**
         @brief Implementation of store_norm for fixed-point fields
         @tparam is_fixed Whether fixed point
         @tparam real Precision of vector we wish to store from
         @tparam n Complex vector length
         @param[in] v elements we wish to find the max abs of for storing
         @param[in] norm The norm we are 
         @return The scale factor to be applied when packing into fixed point
      */
      template <bool is_fixed, typename real, int n>
      __device__ __host__ inline std::enable_if_t<is_fixed, norm_t> store_norm(const array<complex<real>, n> &v, norm_t &norm) const
      {
        norm_t max_[n];
        // two-pass to increase ILP (assumes length divisible by two, e.g. complex-valued)
#pragma unroll
        for (int i = 0; i < n; i++) max_[i] = fmaxf(fabsf((norm_t)v[i].real()), fabsf((norm_t)v[i].imag()));
        norm_t scale = 0.0;
#pragma unroll
        for (int i = 0; i < n; i++) scale = fmaxf(max_[i], scale);
        norm = scale * fixedInvMaxValue<store_t>::value;
        return fdividef(fixedMaxValue<store_t>::value, scale);
      }

      /**
         @brief Load spinor function
         @tparam real Precision of vector we wish to store from
         @tparam n Complex vector length
         @param[in] v output vector now loaded
         @param[in] x checkerboard site index
         @param[in] parity site parity
      */
      template <typename real, int n>
      __device__ __host__ inline void load(array<complex<real>, n> &v, int x, int parity = 0) const
      {
        constexpr int len = 2 * n; // real-valued length

        if constexpr (!(n == 3 && isHalf<store_t>::value)) {
          norm_t nrm = load_norm<isFixed<store_t>::value>(x, parity);
          array<real, len> v_;

          constexpr int M = len / N;
#pragma unroll
          for (int i = 0; i < M; i++) {
            // first load from memory
            Vector vecTmp = vector_load<Vector>(data.spinor, parity * data.cb_offset + x + data.stride * i);
            // now copy into output and scale
#pragma unroll
            for (int j = 0; j < N; j++) copy_and_scale(v_[i * N + j], reinterpret_cast<store_t *>(&vecTmp)[j], nrm);
          }

          for (int i = 0; i < n; i++) { v[i] = complex<real>(v_[2 * i + 0], v_[2 * i + 1]); }
        } else {
          // specialized path for half precision staggered
          using Vector = int4;
          auto cb_offset = data.cb_norm_offset / 4;
          norm_t nrm;
          array<real, len> v_;

          // first load from memory
          Vector vecTmp = vector_load<Vector>(data.spinor, parity * cb_offset + x);

          // extract norm
          memcpy(&nrm, &vecTmp.w, sizeof(norm_t));

          // now copy into output and scale
#pragma unroll
          for (int i = 0; i < len; i++) copy_and_scale(v_[i], reinterpret_cast<store_t *>(&vecTmp)[i], nrm);

#pragma unroll
          for (int i = 0; i < n; i++) { v[i] = complex<real>(v_[2 * i + 0], v_[2 * i + 1]); }
        }
      }

      /**
         @brief Save spinor function
         @tparam real Precision of vector we wish to store from
         @tparam n Complex vector length
         @param[in] v input vector we wish to store
         @param[in] x checkerboard site index
         @param[in] parity site parity
      */
      template <typename real, int n>
      __device__ __host__ inline void save(const array<complex<real>, n> &v, int x, int parity = 0) const
      {
        constexpr int len = 2 * n; // real-valued length

        if constexpr (!(n == 3 && isHalf<store_t>::value)) {
          array<real, len> v_;

          if constexpr (isFixed<store_t>::value) {
            real scale_inv = store_norm<isFixed<store_t>::value, real, n>(v, data.norm[x + parity * data.cb_norm_offset]);
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
            vector_store(data.spinor, parity * data.cb_offset + x + data.stride * i, vecTmp);
          }
        } else {
          // specialized path for half precision staggered
          using Vector = int4;
          auto cb_offset = data.cb_norm_offset / 4;
          norm_t norm;
          norm_t scale_inv = store_norm<isFixed<store_t>::value, real, n>(v, norm);
          array<real, len> v_;
#pragma unroll
          for (int i = 0; i < n; i++) {
            v_[2 * i + 0] = scale_inv * v[i].real();
            v_[2 * i + 1] = scale_inv * v[i].imag();
          }

          Vector vecTmp;
          memcpy(&vecTmp.w, &norm, sizeof(norm_t)); // pack the norm
#pragma unroll
          for (int i = 0; i < len; i++) copy_scaled(reinterpret_cast<store_t *>(&vecTmp)[i], v_[i]);
          // second do vectorized copy into memory
          vector_store(data.spinor, parity * cb_offset + x, vecTmp);
        }
      }
    };

    /**
       n_vector defines the granularity of load/store, e.g., sets the
       size of vector we load from memory
       @tparam store_t Field storage precision
       @tparam GPU Whether this is GPU (or CPU)?
       @tparam nSpin Number of spino components
       @tparam site_unroll Whether we enforce all site components must
       be unrolled onto the same thread (required for fixed-point precision)
    */
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

    template <> constexpr int n_vector<short, true, 4, true>() { return QUDA_ORDER_FP; }
    template <> constexpr int n_vector<short, true, 1, true>() { return 2; }

    template <> constexpr int n_vector<int8_t, true, 4, true>() { return QUDA_ORDER_FP; }
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
    constexpr void instantiate(const T &a, const T &b, const T &c, V &x_, Args &&... args)
    {
      unwrap_t<V> &x(x_);
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
    constexpr void instantiate(const T &, const T &, const T &, V &x_, Args &&...)
    {
      unwrap_t<V> &x(x_);
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
              bool mixed, typename T, typename x_store_t, typename Vx, typename Vy, typename... Args>
    constexpr std::enable_if_t<mixed, void> instantiate(const T &a, const T &b, const T &c, Vx &x_, Vy &y_,
                                                        Args &&... args)
    {
      unwrap_t<Vx> &x(x_);
      unwrap_t<Vy> &y(y_);

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
    constexpr void instantiate(const T &a, const T &b, const T &c, V &x_, Args &&... args)
    {
      unwrap_t<V> &x(x_);
      if (x.Precision() == QUDA_DOUBLE_PRECISION) {
#if !(QUDA_PRECISION & 8)
        if (x.Location() == QUDA_CUDA_FIELD_LOCATION)
          errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
#endif
        // always instantiate the double-precision template to allow CPU
        // fields through, and prevent double-precision GPU
        // instantiation using double_mapper
        instantiate<Functor, Blas, mixed, T, double>(a, b, c, x_, args...);
      } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
        instantiate<Functor, Blas, mixed, T, float>(a, b, c, x_, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
      } else if (x.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
        instantiate<Functor, Blas, mixed, T, short>(a, b, c, x_, args...);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else if (x.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
        instantiate<Functor, Blas, mixed, T, int8_t>(a, b, c, x_, args...);
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
