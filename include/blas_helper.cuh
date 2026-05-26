#pragma once

#include <cstdio>
#include <cstring>

#include <color_spinor_field.h>
#include <quda_define.h>
#include <load_store.h>
#include <tma_helper.hpp>
#include <target_device.h>
#include <convert.h>
#include <float_vector.h>
#include <array.h>
#include <math_helper.cuh>
#include "instantiate.h"

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
  __host__ __device__ inline double2 set(doubledouble2 &a) { return {a.x.head(), a.y.head()}; }
  __host__ __device__ inline double3 set(doubledouble3 &a) { return {a.x.head(), a.y.head(), a.z.head()}; }
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
    constexpr PrefetchType blas_prefetch_type() noexcept
    {
#if defined(QUDA_BLAS_PREFETCH_TYPE_NONE)
      return PrefetchType::NONE;
#elif defined(QUDA_BLAS_PREFETCH_TYPE_THREAD)
      return PrefetchType::THREAD;
#elif defined(QUDA_BLAS_PREFETCH_TYPE_BULK)
      return PrefetchType::BULK;
#else
#error "Missing or invalid QUDA_BLAS_PREFETCH_TYPE (expect NONE, THREAD, or BULK)"
#endif
    }

    inline constexpr bool blas_prefetch_enabled_v = (blas_prefetch_type() != PrefetchType::NONE);

    /** Append BLAS prefetch mode to \a aux for \c TuneKey when non-NONE. */
    inline void blas_tune_aux_prefetch(char *aux)
    {
      switch (blas_prefetch_type()) {
      case PrefetchType::THREAD: strcat(aux, ",prefetch=thread"); break;
      case PrefetchType::BULK: strcat(aux, ",prefetch=bulk"); break;
      default: break;
      }
    }

    /**
       @brief Append a work-item unroll tag to a BLAS autotune auxiliary string.

       Appends a substring of the form \c ",unroll=N" to \a aux so kernels that vary
       \c QUDA_BLAS_UNROLL_STREAMING (or an effective unroll such as multi-blas vs. NXZ)
       receive distinct \c TuneKey entries.

       @param[in,out] aux Null-terminated auxiliary string buffer (e.g. \c Tunable::aux)
       to append to; must have space for the additional suffix.
       @param[in] unroll Unroll factor to record: typically \c QUDA_BLAS_UNROLL_STREAMING or
       \c QUDA_BLAS_UNROLL_REDUCE, or an effective value such as \c 1 when multi-blas disables unroll.

       @return None.
     */
    inline void blas_tune_aux_work_item_unroll(char *aux, unsigned int unroll)
    {
      char buf[32];
      snprintf(buf, sizeof(buf), ",unroll=%d", unroll);
      strcat(aux, buf);
    }

    /**
       Helper struct that contains the meta data required for
       read and writing to a spinor field in the BLAS kernels.
       @tparam store_t Type used to store field in memory
    */
    template <typename store_t, bool is_fixed> struct data_t {
      store_t *spinor = nullptr;
      int stride = 0;
      unsigned int cb_offset = 0;
      data_t() = default;
      data_t(const ColorSpinorField &x) :
        spinor(x.data<store_t *>()), stride(x.VolumeCB()), cb_offset(x.Bytes() / (2 * sizeof(store_t)))
      {}
    };

    /**
       Helper struct that contains the meta data required for read and
       writing to a spinor field in the BLAS kernels.  This is a
       specialized variant for fixed-point fields where need to store
       the meta data for the norm field.
       @tparam store_t Type used to store field in memory
    */
    template <typename store_t> struct data_t<store_t, true> {
      using norm_t = float;
      store_t *spinor = nullptr;
      norm_t *norm = nullptr;
      int stride = 0;
      unsigned int cb_offset = 0;
      unsigned int cb_norm_offset = 0;
      data_t() = default;
      data_t(const ColorSpinorField &x) :
        spinor(x.data<store_t *>()),
        norm(static_cast<norm_t *>(x.Norm())),
        stride(x.VolumeCB()),
        cb_offset(x.Bytes() / (2 * sizeof(store_t))),
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
      data_t<store_t, isFixed<store_t>::value> data;

      Spinor() = default;
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
        return fdivide(fixedMaxValue<store_t>::value, scale);
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
          constexpr int Nrem = len - M * N;
#pragma unroll
          for (int i = 0; i < M; i++) {
            // first load from memory
            auto vecTmp = vector_load<store_t, N>(data.spinor + parity * data.cb_offset, data.stride * i + x);
            // now copy into output and scale
            copy_and_scale(&v_[i * N], vecTmp, nrm);
          }
          if constexpr (Nrem > 0) {
            // first load from memory
            auto vecTmp = vector_load<store_t, Nrem>(data.spinor + parity * data.cb_offset + data.stride * M * N, x);
            // now copy into output and scale
            copy_and_scale(&v_[M * N], vecTmp, nrm);
          }

#pragma unroll
          for (int i = 0; i < n; i++) { v[i] = complex<real>(v_[2 * i + 0], v_[2 * i + 1]); }
        } else {
          // specialized path for half precision staggered
          auto cb_offset = data.cb_norm_offset / 4;
          norm_t nrm;
          array<real, len> v_;

          // first load from memory
          auto vecTmp = vector_load<store_t, 8>(data.spinor, parity * cb_offset + x);

          // extract norm
          memcpy(&nrm, &vecTmp[6], sizeof(norm_t));

          // now copy into output and scale
          copy_and_scale(&v_[0], vecTmp, nrm);

#pragma unroll
          for (int i = 0; i < n; i++) { v[i] = complex<real>(v_[2 * i + 0], v_[2 * i + 1]); }
        }
      }

      /**
         @brief Prefetch cache lines that \a load would read (for grid-stride latency hiding).
       */
      template <typename real, int n>
      __device__ __host__ inline void prefetch(int x, int parity = 0) const
      {
        if constexpr (blas_prefetch_type() == PrefetchType::NONE) return;

        constexpr int len = 2 * n;

        if constexpr (blas_prefetch_type() == PrefetchType::THREAD) {
          if constexpr (!(n == 3 && isHalf<store_t>::value)) {
            if constexpr (isFixed<store_t>::value)
              prefetch_cache_line(data.norm + data.cb_norm_offset * parity + x);

            constexpr int M = len / N;
            constexpr int Nrem = len - M * N;
#pragma unroll
            for (int i = 0; i < M; i++) {
              using vector_t = typename VectorType<store_t, N>::type;
              prefetch_cache_line(reinterpret_cast<const vector_t *>(data.spinor + parity * data.cb_offset)
                                  + (data.stride * i + x));
            }
            if constexpr (Nrem > 0) {
              using vector_t = typename VectorType<store_t, Nrem>::type;
              prefetch_cache_line(reinterpret_cast<const vector_t *>(data.spinor + parity * data.cb_offset
                                                                     + data.stride * M * N)
                                  + x);
            }
          } else {
            auto cb_offset = data.cb_norm_offset / 4;
            using vector_t = typename VectorType<store_t, 8>::type;
            prefetch_cache_line(reinterpret_cast<const vector_t *>(data.spinor) + (parity * cb_offset + x));
          }
        } else if constexpr (blas_prefetch_type() == PrefetchType::BULK) {
          if (!target::is_thread_zero()) return;
          const unsigned bx = blockDim.x;

          if constexpr (!(n == 3 && isHalf<store_t>::value)) {
            if constexpr (isFixed<store_t>::value)
              prefetch_cache_bulk(data.norm + data.cb_norm_offset * parity + x, bx * sizeof(norm_t));

            constexpr int M = len / N;
            constexpr int Nrem = len - M * N;
#pragma unroll
            for (int i = 0; i < M; i++) {
              using vector_t = typename VectorType<store_t, N>::type;
              prefetch_cache_bulk(reinterpret_cast<const vector_t *>(data.spinor + parity * data.cb_offset)
                                    + (data.stride * i + x),
                                  bx * sizeof(vector_t));
            }
            if constexpr (Nrem > 0) {
              using vector_t = typename VectorType<store_t, Nrem>::type;
              prefetch_cache_bulk(
                reinterpret_cast<const vector_t *>(data.spinor + parity * data.cb_offset + data.stride * M * N) + x,
                bx * sizeof(vector_t));
            }
          } else {
            auto cb_offset = data.cb_norm_offset / 4;
            using vector_t = typename VectorType<store_t, 8>::type;
            prefetch_cache_bulk(reinterpret_cast<const vector_t *>(data.spinor) + (parity * cb_offset + x),
                                bx * sizeof(vector_t));
          }
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

        array<real, len> v_;
#pragma unroll
        for (int i = 0; i < n; i++) {
          v_[2 * i + 0] = v[i].real();
          v_[2 * i + 1] = v[i].imag();
        }

        if constexpr (!(n == 3 && isHalf<store_t>::value)) {
          real scale_inv = 0.0;
          if constexpr (isFixed<store_t>::value)
            scale_inv = store_norm<isFixed<store_t>::value, real, n>(v, data.norm[x + parity * data.cb_norm_offset]);

          constexpr int M = len / N;
          constexpr int Nrem = len - M * N;
#pragma unroll
          for (int i = 0; i < M; i++) {
            array<store_t, N> vecTmp;
            // first do scalar copy converting into storage type
            copy_and_scale<store_t, real, N>(vecTmp, &v_[i * N], scale_inv);
            // second do vectorized copy into memory
            vector_store(data.spinor + parity * data.cb_offset, data.stride * i + x, vecTmp);
          }

          if constexpr (Nrem > 0) {
            array<store_t, Nrem> vecTmp;
            // first do copy converting into storage type
            copy_and_scale<store_t, real, Nrem>(vecTmp, &v_[M * N], scale_inv);
            // second do vectorized copy into memory
            vector_store(data.spinor + parity * data.cb_offset + data.stride * M * N, x, vecTmp);
          }
        } else {
          // specialized path for half precision staggered
          auto cb_offset = data.cb_norm_offset / 4;
          norm_t norm;
          norm_t scale_inv = store_norm<isFixed<store_t>::value, real, n>(v, norm);

          array<store_t, 8> vecTmp;
          memcpy(&vecTmp[6], &norm, sizeof(norm_t)); // pack the norm
          array<store_t, 6> vecTmp2;
          copy_and_scale<store_t, real, 6>(vecTmp2, &v_[0], scale_inv);
          std::memcpy(&vecTmp, &vecTmp2, sizeof(vecTmp2));
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
    template <typename store_t, bool GPU> constexpr int n_vector(int, int) { return 0; }

    // native ordering
    template <> constexpr int n_vector<double, true>(int nSpin, int site_unroll)
    {
      if (site_unroll)
        return nSpin == 4 ? colorspinor::get_vector_order<double>(24) : colorspinor::get_vector_order<double>(6);
      else
        return colorspinor::get_vector_order<double>(4);
    }

    template <> constexpr int n_vector<float, true>(int nSpin, int site_unroll)
    {
      if (site_unroll)
        return nSpin == 4 ? colorspinor::get_vector_order<float>(24) : colorspinor::get_vector_order<float>(6);
      else
        return colorspinor::get_vector_order<float>(8);
    }

    template <> constexpr int n_vector<short, true>(int nSpin, int site_unroll)
    {
      if (site_unroll)
        return nSpin == 4 ? colorspinor::get_vector_order<short>(24) : colorspinor::get_vector_order<short>(6);
      else
        return colorspinor::get_vector_order<short>(16);
    }

    template <> constexpr int n_vector<int8_t, true>(int nSpin, int site_unroll)
    {
      if (site_unroll)
        return nSpin == 4 ? colorspinor::get_vector_order<int8_t>(24) : colorspinor::get_vector_order<int8_t>(6);
      else
        return colorspinor::get_vector_order<int8_t>(16);
    }

    // Just use float-2/float-4 ordering on CPU when not site unrolling
    template <> constexpr int n_vector<double, false>(int nSpin, int site_unroll)
    {
      if (site_unroll) {
        return nSpin * 6;
      } else {
        return 2;
      }
    }

    template <> constexpr int n_vector<float, false>(int nSpin, int site_unroll)
    {
      if (site_unroll) {
        return nSpin * 6;
      } else {
        return 4;
      }
    }

    template <template <typename...> class Functor,
              template <template <typename...> class, typename store_t, typename y_store_t, int, typename> class Blas,
              typename T, typename store_t, typename y_store_t, typename V, typename... Args>
    constexpr void instantiate(const T &a, const T &b, const T &c, V &x_, Args &&... args)
    {
      unwrap_t<V> &x(x_);
      if (x.Nspin() == 4 || x.Nspin() == 2) {
        if constexpr (is_enabled_spin(2) || is_enabled_spin(4)) {
          // Nspin-2 takes Nspin-4 path here, and we check for this later
          Blas<Functor, store_t, y_store_t, 4, T>(a, b, c, x, args...);
        } else {
          errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
        }
      } else {
        if constexpr (is_enabled_spin(1)) {
          Blas<Functor, store_t, y_store_t, 1, T>(a, b, c, x, args...);
        } else {
          errorQuda("blas has not been built for Nspin=%d fields", x.Nspin());
        }
      }
    }

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

        if constexpr (!is_enabled(QUDA_DOUBLE_PRECISION))
          if (x.Location() == QUDA_CUDA_FIELD_LOCATION)
            errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
        // always instantiate the double-precision template to allow CPU
        // fields through, and prevent double-precision GPU
        // instantiation using gpu_mapper
        instantiate<Functor, Blas, T, x_store_t, double>(a, b, c, x, y, args...);

      } else if (y.Precision() == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
          instantiate<Functor, Blas, T, x_store_t, typename PromoteTypeId<x_store_t, float>::type>(a, b, c, x, y,
                                                                                                   args...);
        else
          errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
      } else if (y.Precision() == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION))
          instantiate<Functor, Blas, T, x_store_t, typename PromoteTypeId<x_store_t, short>::type>(a, b, c, x, y,
                                                                                                   args...);
        else
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      } else if (y.Precision() == QUDA_QUARTER_PRECISION) {
        if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
          instantiate<Functor, Blas, T, x_store_t, typename PromoteTypeId<x_store_t, int8_t>::type>(a, b, c, x, y,
                                                                                                    args...);
        else
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
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
        if constexpr (!is_enabled(QUDA_DOUBLE_PRECISION))
          if (x.Location() == QUDA_CUDA_FIELD_LOCATION)
            errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
        // always instantiate the double-precision template to allow CPU
        // fields through, and prevent double-precision GPU
        // instantiation using double_mapper
        instantiate<Functor, Blas, mixed, T, double>(a, b, c, x_, args...);
      } else if (x.Precision() == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
          instantiate<Functor, Blas, mixed, T, float>(a, b, c, x_, args...);
        else
          errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
      } else if (x.Precision() == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION))
          instantiate<Functor, Blas, mixed, T, short>(a, b, c, x_, args...);
        else
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
      } else if (x.Precision() == QUDA_QUARTER_PRECISION) {
        if constexpr (is_enabled(QUDA_QUARTER_PRECISION))
          instantiate<Functor, Blas, mixed, T, int8_t>(a, b, c, x_, args...);
        else
          errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
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

  template <typename A, typename B> void check_size(const A &a, const B &b)
  {
    if (a.size() != b.size()) errorQuda("Mismatched sizes a=%lu b=%lu", a.size(), b.size());
  }

  template <typename A, typename B, typename... Args> void check_size(const A &a, const B &b, const Args &...args)
  {
    check_size(a, b);
    check_size(b, args...);
  }

} // namespace quda
