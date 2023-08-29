#pragma once

/**
 * @file  gauge_field_order.h
 * @brief Main header file for host and device accessors to GaugeFields
 *
 */

#include <cassert>
#include <type_traits>
#include <limits>

#include <register_traits.h>
#include <math_helper.cuh>
#include <convert.h>
#include <complex_quda.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <atomic_helper.h>
#include <gauge_field.h>
#include <index_helper.cuh>
#include <load_store.h>
#include <aos.h>
#include <transform_reduce.h>

namespace quda {

  /**
     @brief gauge_wrapper is an internal class that is used to wrap
     instances of gauge accessors, currying in a specific location on
     the field.  The operator() accessors in gauge-field accessors
     return instances to this class, allowing us to then use operator
     overloading upon this class to interact with the Matrix class.
     As a result we can include gauge-field accessors directly in
     Matrix expressions in kernels without having to declare
     temporaries with explicit calls to the load/save methods in the
     gauge-field accessors.
   */
  template <typename Float, typename T>
    struct gauge_wrapper {
      const int dim;
      const int x_cb;
      const int parity;
      const Float phase;
      T &gauge;

      /**
	 @brief gauge_wrapper constructor
	 @param[in] gauge Gauge field accessor we are wrapping
	 @param[in] dim Dimension we are accessing
	 @param[in] x_cb Checkerboarded space-time index we are accessing
	 @param[in] parity Parity we are accessing
       */
      __device__ __host__ inline gauge_wrapper<Float, T>(T &gauge, int dim, int x_cb, int parity, Float phase = 1.0) :
        dim(dim), x_cb(x_cb), parity(parity), phase(phase), gauge(gauge)
      {
      }

      /**
	 @brief Assignment operator with Matrix instance as input
	 @param[in] M Matrix we want to store in this accessor
       */
      template <typename M> __device__ __host__ inline void operator=(const M &a) const
      {
        gauge.save(a.data, x_cb, dim, parity);
      }
    };

  /**
     @brief Copy constructor for the Matrix class with a gauge_wrapper input.
     @param[in] a Input gauge_wrapper that we use to fill in this matrix instance
   */
  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline void Matrix<T,N>::operator=(const gauge_wrapper<typename RealType<T>::type,S> &a) {
    a.gauge.load(data, a.x_cb, a.dim, a.parity, a.phase);
  }

  /**
     @brief Assignment operator for the Matrix class with a gauge_wrapper input.
     @param[in] a Input gauge_wrapper that we use to fill in this matrix instance
   */
  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline Matrix<T,N>::Matrix(const gauge_wrapper<typename RealType<T>::type,S> &a) {
    a.gauge.load(data, a.x_cb, a.dim, a.parity, a.phase);
  }

  /**
     @brief gauge_ghost_wrapper is an internal class that is used to
     wrap instances of gauge ghost accessors, currying in a specific
     location and dimension on the field.  The Ghost() accessors in
     gauge-field accessors return instances to this class, allowing us
     to then use operator overloading upon this class to interact with
     the Matrix class.  As a result we can include gauge-field ghost
     accessors directly in Matrix expressions in kernels without
     having to declare temporaries with explicit calls to the
     load/save methods in the gauge-field accessors.
   */
  template <typename Float, typename T>
    struct gauge_ghost_wrapper {
      const int dim;
      const int ghost_idx;
      const int parity;
      const Float phase;
      T &gauge;

      /**
	 @brief gauge_wrapper constructor
	 @param[in] gauge Gauge field accessor we are wrapping
	 @param[in] dim Dimension we are accessing
	 @param[in] ghost_idx Ghost index we are accessing
	 @param[in] parity Parity we are accessing
       */
      __device__ __host__ inline gauge_ghost_wrapper<Float, T>(T &gauge, int dim, int ghost_idx, int parity,
                                                               Float phase = 1.0) :
        dim(dim), ghost_idx(ghost_idx), parity(parity), phase(phase), gauge(gauge)
      {
      }

      /**
	 @brief Assignment operator with Matrix instance as input
	 @param[in] M Matrix we want to store in this accessot
       */
      template <typename M> __device__ __host__ inline void operator=(const M &a) const
      {
        gauge.saveGhost(a.data, ghost_idx, dim, parity);
      }
    };

  /**
     @brief Copy constructor for the Matrix class with a gauge_ghost_wrapper input.
     @param[in] a Input gauge_wrapper that we use to fill in this matrix instance
   */
  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline void Matrix<T,N>::operator=(const gauge_ghost_wrapper<typename RealType<T>::type,S> &a) {
    a.gauge.loadGhost(data, a.ghost_idx, a.dim, a.parity, a.phase);
  }

  /**
     @brief Assignment operator for the Matrix class with a gauge_ghost_wrapper input.
     @param[in] a Input gauge_wrapper that we use to fill in this matrix instance
   */
  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline Matrix<T,N>::Matrix(const gauge_ghost_wrapper<typename RealType<T>::type,S> &a) {
    a.gauge.loadGhost(data, a.ghost_idx, a.dim, a.parity, a.phase);
  }

  namespace gauge {

    template <typename Float, typename storeFloat> __host__ __device__ inline constexpr bool fixed_point() { return false; }
    template <> __host__ __device__ inline constexpr bool fixed_point<float, int8_t>() { return true; }
    template<> __host__ __device__ inline constexpr bool fixed_point<float,short>() { return true; }
    template<> __host__ __device__ inline constexpr bool fixed_point<float,int>() { return true; }

    template <typename Float, typename storeFloat> __host__ __device__ inline constexpr bool match() { return false; }
    template<> __host__ __device__ inline constexpr bool match<int,int>() { return true; }
    template<> __host__ __device__ inline constexpr bool match<short,short>() { return true; }

    /**
       @brief fieldorder_wrapper is an internal class that is used to
       wrap instances of FieldOrder accessors, currying in the
       specific location on the field.  This is used as a helper class
       for fixed-point accessors providing the necessary conversion
       and scaling when writing to a fixed-point field.
    */
    template <typename Float, typename storeFloat>
      struct fieldorder_wrapper {
      using value_type = Float;
      using store_type = storeFloat;
      complex<storeFloat> *v;
      const unsigned int idx;

    private:
      const Float scale;
      const Float scale_inv;

    public:
      static constexpr bool fixed = fixed_point<Float, storeFloat>();

      /**
         @brief fieldorder_wrapper constructor
         @param idx Field index
      */
      __device__ __host__ inline fieldorder_wrapper(complex<storeFloat> *v, unsigned int idx, Float scale,
                                                    Float scale_inv) :
        v(v), idx(idx), scale(scale), scale_inv(scale_inv)
      {
      }

      fieldorder_wrapper(const fieldorder_wrapper<Float, storeFloat> &a) = delete;

      fieldorder_wrapper(fieldorder_wrapper<Float, storeFloat> &&a) = default;

      /**
         @brief Assignment operator with fieldorder_wrapper instance as input
         @param a fieldorder_wrapper we are copying from
      */
      __device__ __host__ inline void operator=(const fieldorder_wrapper<Float, storeFloat> &a)
      {
        complex<Float> in = a;
        v[idx] = fixed ?
          complex<storeFloat>(f2i_round<storeFloat>(scale * in.real()), f2i_round<storeFloat>(scale * in.imag())) :
          a.v[a.idx];
      }

      /**
         @brief Assignment operator with fieldorder_wrapper instance as input
         @param a fieldorder_wrapper we are copying from
      */
      template <typename theirFloat, typename theirStoreFloat>
      __device__ __host__ inline void operator=(const fieldorder_wrapper<theirFloat, theirStoreFloat> &a)
      {
        complex<theirFloat> in = a;
        v[idx] = fixed ?
          complex<storeFloat>(f2i_round<storeFloat>(scale * in.real()), f2i_round<storeFloat>(scale * in.imag())) :
          complex<storeFloat>(in.real(), in.imag());
      }

      /**
         @brief Assignment operator with complex number instance as input
         @param a Complex number we want to store in this accessor
      */
      template <typename theirFloat> __device__ __host__ inline void operator=(const complex<theirFloat> &a)
      {
        if constexpr (match<storeFloat, theirFloat>()) {
          v[idx] = complex<storeFloat>(a.x, a.y);
        } else {
          v[idx] = fixed ? complex<storeFloat>(f2i_round<storeFloat>(scale * a.x), f2i_round<storeFloat>(scale * a.y)) :
                           complex<storeFloat>(a.x, a.y);
        }
      }

      /**
         @brief complex cast operator
      */
      __device__ __host__ inline operator complex<Float>() const
      {
        complex<storeFloat> tmp = v[idx];
        if constexpr (fixed) return scale_inv * complex<Float>(static_cast<Float>(tmp.x), static_cast<Float>(tmp.y));
        return complex<Float>(tmp.x, tmp.y);
      }

      /**
       * @brief returns the pointer of this wrapper object
       */
      __device__ __host__ inline auto data() const { return &v[idx]; }

      /**
       * @brief returns the scale of this wrapper object
       */
      __device__ __host__ inline auto get_scale() const { return scale; }

      /**
       * @brief returns the scale_inv of this wrapper object
       */
      __device__ __host__ inline auto get_scale_inv() const { return scale_inv; }

      /**
         @brief negation operator
         @return negation of this complex number
      */
      __device__ __host__ inline complex<Float> operator-() const
      {
        return fixed ? -scale_inv * static_cast<complex<Float>>(v[idx]) : -static_cast<complex<Float>>(v[idx]);
      }

      /**
         @brief Operator+= with complex number instance as input
           @param a Complex number we want to add to this accessor
      */
      template <typename theirFloat> __device__ __host__ inline void operator+=(const complex<theirFloat> &a)
      {
        if constexpr (match<storeFloat, theirFloat>()) {
          v[idx] += complex<storeFloat>(a.x, a.y);
        } else {
          v[idx] += fixed ? complex<storeFloat>(f2i_round<storeFloat>(scale * a.x), f2i_round<storeFloat>(scale * a.y)) :
                            complex<storeFloat>(a.x, a.y);
        }
      }

      /**
         @brief Operator-= with complex number instance as input
         @param a Complex number we want to subtract from this accessor
      */
      template <typename theirFloat> __device__ __host__ inline void operator-=(const complex<theirFloat> &a)
      {
        if constexpr (match<storeFloat, theirFloat>()) {
          v[idx] -= complex<storeFloat>(a.x, a.y);
        } else {
          v[idx] -= fixed ? complex<storeFloat>(f2i_round<storeFloat>(scale * a.x), f2i_round<storeFloat>(scale * a.y)) :
                            complex<storeFloat>(a.x, a.y);
        }
      }
      };

    template<typename Float, typename storeFloat>
    __device__ __host__ inline complex<Float> operator*(const Float &a, const fieldorder_wrapper<Float,storeFloat> &b)
    {
      return a * complex<Float>(b);
    }

    template <typename Float, typename storeFloat>
    __device__ __host__ inline complex<Float> operator*(const complex<Float> &a,
                                                        const fieldorder_wrapper<Float, storeFloat> &b)
    {
      return a * complex<Float>(b);
    }

    template <typename Float, typename storeFloat>
    __device__ __host__ inline complex<Float> operator+(const fieldorder_wrapper<Float, storeFloat> &a,
                                                        const complex<Float> &b)
    {
      return complex<Float>(a) + b;
    }

    template <typename Float, typename storeFloat>
    __device__ __host__ inline complex<Float> operator+(const complex<Float> &a,
                                                        const fieldorder_wrapper<Float, storeFloat> &b)
    {
      return a + complex<Float>(b);
    }

    template <typename Float, typename storeFloat>
    __device__ __host__ inline complex<Float> conj(const fieldorder_wrapper<Float, storeFloat> &a)
    {
      return conj(static_cast<complex<Float>>(a));
    }

    template <typename Float, int nColor, QudaGaugeFieldOrder order, typename storeFloat> struct Accessor {
      static constexpr bool is_mma_compatible = false;
      mutable complex<Float> dummy;
      Accessor(const GaugeField &, void * = nullptr, void ** = nullptr)
      {
        errorQuda("Not implemented for order=%d", order);
      }

      void resetScale(Float) { }

      __device__ __host__ complex<Float> &operator()(int, int, int, int, int) const { return dummy; }
    };

    template <typename Float, int nColor, QudaGaugeFieldOrder order, bool native_ghost, typename storeFloat>
    struct GhostAccessor {
      mutable complex<Float> dummy;
      GhostAccessor(const GaugeField &, void * = nullptr, void ** = nullptr)
      {
        errorQuda("Not implemented for order=%d", order);
      }

      void resetScale(Float) { }

      __device__ __host__ complex<Float> &operator()(int, int, int, int, int) const { return dummy; }
    };

    template <typename Float, int nColor, typename storeFloat>
    struct Accessor<Float, nColor, QUDA_QDP_GAUGE_ORDER, storeFloat> {
      using wrapper = fieldorder_wrapper<Float, storeFloat>;
      static constexpr bool is_mma_compatible = false;
      complex <storeFloat> *u[QUDA_MAX_GEOMETRY];
      const unsigned int volumeCB;
      const int geometry;
      const unsigned int cb_offset;
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();

      Accessor(const GaugeField &U, void *gauge_ = 0, void ** = 0) :
        volumeCB(U.VolumeCB()),
        geometry(U.Geometry()),
        cb_offset((U.Bytes() >> 1) / (sizeof(complex<storeFloat>) * U.Geometry())),
        scale(static_cast<Float>(1.0)),
        scale_inv(static_cast<Float>(1.0))
      {
	for (int d=0; d<U.Geometry(); d++)
	  u[d] = gauge_ ? static_cast<complex<storeFloat>**>(gauge_)[d] :
	    static_cast<complex<storeFloat>**>(const_cast<void*>(U.Gauge_p()))[d];
	resetScale(U.Scale());
      }

      void resetScale(Float max)
      {
        if (fixed) {
          scale = static_cast<Float>(std::numeric_limits<storeFloat>::max()) / max;
          scale_inv = max / static_cast<Float>(std::numeric_limits<storeFloat>::max());
        }
      }

      __device__ __host__ inline wrapper operator()(int d, int parity, int x, int row, int col) const
      {
        return wrapper(u[d], parity * cb_offset + (x * nColor + row) * nColor + col, scale, scale_inv);
      }

      template <typename theirFloat>
      __device__ __host__ inline void atomic_add(int dim, int parity, int x_cb, int row, int col,
                                                 const complex<theirFloat> &val) const
      {
        using vec2 = array<storeFloat, 2>;
        vec2 *u2 = reinterpret_cast<vec2*>(u[dim] + parity*cb_offset + (x_cb*nColor + row)*nColor + col);

        vec2 val_ = (fixed && !match<storeFloat, theirFloat>()) ?
          vec2 {f2i_round<storeFloat>(scale * val.real()), f2i_round<storeFloat>(scale * val.imag())} :
          vec2 {static_cast<storeFloat>(val.real()), static_cast<storeFloat>(val.imag())};

        atomic_fetch_add(u2, val_);
      }

      /**
         @brief Wrapper to transform_reduce which is called by the
         reducer functions, e.g., norm2 and abs_max
         @tparam reducer The reduction operation we which to apply
         @param[in] location The location of execution
         @param[in] dim The dimension of of the field we wish to
         reduce.  If dim = -1, then we reduce over all dimensions.
         @param[in] h The helper functor which acts as the transformer
         in transform_reduce
       */
      template <typename reducer, typename helper>
      __host__ double transform_reduce(QudaFieldLocation location, int dim, helper h) const
      {
        if (dim >= geometry) errorQuda("Request dimension %d exceeds dimensionality of the field %d", dim, geometry);
        int lower = (dim == -1) ? 0 : dim;
        int ndim = (dim == -1 ? geometry : 1);
        std::vector<typename reducer::reduce_t> result(ndim);
        std::vector<complex<storeFloat> *> v(ndim);
        for (int d = 0; d < ndim; d++) v[d] = u[d + lower];
        ::quda::transform_reduce<reducer>(location, result, v, 2 * volumeCB * nColor * nColor, h);
        auto total = reducer::init();
        for (auto &res : result) total = reducer::apply(total, res);
        return total;
      }
    };

    template <typename Float, int nColor, bool native_ghost, typename storeFloat>
    struct GhostAccessor<Float, nColor, QUDA_QDP_GAUGE_ORDER, native_ghost, storeFloat> {
      using wrapper = fieldorder_wrapper<Float, storeFloat>;
      complex<storeFloat> *ghost[8];
      unsigned int ghostOffset[8];
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();

      GhostAccessor(const GaugeField &U, void * = nullptr, void **ghost_ = nullptr) :
        scale(static_cast<Float>(1.0)), scale_inv(static_cast<Float>(1.0))
      {
        for (int d=0; d<4; d++) {
	  ghost[d] = ghost_ ? static_cast<complex<storeFloat>*>(ghost_[d]) :
	    static_cast<complex<storeFloat>*>(const_cast<void*>(U.Ghost()[d]));
	  ghostOffset[d] = U.Nface()*U.SurfaceCB(d)*U.Ncolor()*U.Ncolor();

	  ghost[d+4] = (U.Geometry() != QUDA_COARSE_GEOMETRY) ? nullptr :
	    ghost_ ? static_cast<complex<storeFloat>*>(ghost_[d+4]) :
	    static_cast<complex<storeFloat>*>(const_cast<void*>(U.Ghost()[d+4]));
	  ghostOffset[d+4] = U.Nface()*U.SurfaceCB(d)*U.Ncolor()*U.Ncolor();
	}

	resetScale(U.Scale());
      }

      void resetScale(Float max)
      {
        if (fixed) {
          scale = static_cast<Float>(std::numeric_limits<storeFloat>::max()) / max;
          scale_inv = max / static_cast<Float>(std::numeric_limits<storeFloat>::max());
        }
      }

      __device__ __host__ inline wrapper operator()(int d, int parity, int x, int row, int col) const
      {
        return wrapper(ghost[d], parity * ghostOffset[d] + (x * nColor + row) * nColor + col, scale, scale_inv);
      }
    };

    template <typename Float, int nColor, typename storeFloat>
    struct Accessor<Float, nColor, QUDA_MILC_GAUGE_ORDER, storeFloat> {
      using wrapper = fieldorder_wrapper<Float, storeFloat>;
      static constexpr bool is_mma_compatible = true;
      complex<storeFloat> *u;
      const unsigned int volumeCB;
      const int geometry;
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();

      Accessor(const GaugeField &U, void *gauge_ = nullptr, void ** = nullptr) :
        u(gauge_ ? static_cast<complex<storeFloat> *>(gauge_) :
                   static_cast<complex<storeFloat> *>(const_cast<void *>(U.Gauge_p()))),
        volumeCB(U.VolumeCB()),
        geometry(U.Geometry()),
        scale(static_cast<Float>(1.0)),
        scale_inv(static_cast<Float>(1.0))
      {
        resetScale(U.Scale());
      }

      void resetScale(Float max)
      {
        if (fixed) {
          scale = static_cast<Float>(std::numeric_limits<storeFloat>::max()) / max;
          scale_inv = max / static_cast<Float>(std::numeric_limits<storeFloat>::max());
        }
      }

      /**
       * @brief Creates a fieldorder_wrapper object whose pointer points to the start of
       * the memory chunk corresponds to the matrix at d, parity, x, row, col.
       */
      __device__ __host__ inline auto operator()(int d, int parity, int x, int row, int col) const
      {
        return wrapper(u, (((parity * volumeCB + x) * geometry + d) * nColor + row) * nColor + col, scale, scale_inv);
      }

      template <typename theirFloat>
      __device__ __host__ inline void atomic_add(int dim, int parity, int x_cb, int row, int col,
                                                 const complex<theirFloat> &val) const
      {
        using vec2 = array<storeFloat, 2>;
        vec2 *u2 = reinterpret_cast<vec2*>(u + (((parity*volumeCB+x_cb)*geometry + dim)*nColor + row)*nColor + col);

        vec2 val_ = (fixed && !match<storeFloat, theirFloat>()) ?
          vec2 {f2i_round<storeFloat>(scale * val.real()), f2i_round<storeFloat>(scale * val.imag())} :
          vec2 {static_cast<storeFloat>(val.real()), static_cast<storeFloat>(val.imag())};

        atomic_fetch_add(u2, val_);
      }

      /**
         @brief Wrapper to transform_reduce which is called by the
         reducer functions, e.g., norm2 and abs_max
         @tparam reducer The reduction operation we which to apply
         @param[in] location The location of execution
         @param[in] dim The dimension of of the field we wish to
         reduce.  If dim = -1, then we reduce over all dimensions.
         @param[in] h The helper functor which acts as the transformer
         in transform_reduce
       */
      template <typename reducer, typename helper>
      __host__ double transform_reduce(QudaFieldLocation location, int dim, helper h) const
      {
        if (dim >= geometry) errorQuda("Request dimension %d exceeds dimensionality of the field %d", dim, geometry);
        auto count = (dim == -1 ? geometry : 1) * volumeCB * nColor * nColor; // items per parity
        auto init = reducer::init();
        std::vector<decltype(init)> result = {init, init};
        std::vector<decltype(u)> v
          = {u + 0 * volumeCB * geometry * nColor * nColor, u + 1 * volumeCB * geometry * nColor * nColor};
        if (dim == -1) {
          ::quda::transform_reduce<reducer>(location, result, v, count, h);
        } else {
          ::quda::transform_reduce<reducer>(location, result, v, count, h, milc_mapper(dim, geometry, nColor * nColor));
        }

        return reducer::apply(result[0], result[1]);
      }
    };

    template <typename Float, int nColor, bool native_ghost, typename storeFloat>
    struct GhostAccessor<Float, nColor, QUDA_MILC_GAUGE_ORDER, native_ghost, storeFloat> {
      using wrapper = fieldorder_wrapper<Float, storeFloat>;
      complex<storeFloat> *ghost[8];
      unsigned int ghostOffset[8];
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();

      GhostAccessor(const GaugeField &U, void * = nullptr, void **ghost_ = nullptr) :
        scale(static_cast<Float>(1.0)), scale_inv(static_cast<Float>(1.0))
      {
        for (int d=0; d<4; d++) {
	  ghost[d] = ghost_ ? static_cast<complex<storeFloat>*>(ghost_[d]) :
	    static_cast<complex<storeFloat>*>(const_cast<void*>(U.Ghost()[d]));
	  ghostOffset[d] = U.Nface()*U.SurfaceCB(d)*U.Ncolor()*U.Ncolor();

	  ghost[d+4] = (U.Geometry() != QUDA_COARSE_GEOMETRY) ? nullptr :
	    ghost_ ? static_cast<complex<storeFloat>*>(ghost_[d+4]) :
	    static_cast<complex<storeFloat>*>(const_cast<void*>(U.Ghost()[d+4]));
	  ghostOffset[d+4] = U.Nface()*U.SurfaceCB(d)*U.Ncolor()*U.Ncolor();
	}

	resetScale(U.Scale());
      }

      void resetScale(Float max)
      {
        if (fixed) {
          scale = static_cast<Float>(std::numeric_limits<storeFloat>::max()) / max;
          scale_inv = max / static_cast<Float>(std::numeric_limits<storeFloat>::max());
        }
      }

      /**
       * @brief Creates a fieldorder_wrapper object with the pointer
       * that points to the memory chunk at d, parity, x, row, col
       */
      __device__ __host__ inline wrapper operator()(int d, int parity, int x, int row, int col) const
      {
        return wrapper(ghost[d], parity * ghostOffset[d] + (x * nColor + row) * nColor + col, scale, scale_inv);
      }
    };

    template<int nColor, int N>
      __device__ __host__ inline int indexFloatN(int dim, int parity, int x_cb, int row, int col, int stride, int offset_cb) {
      constexpr int M = (2*nColor*nColor) / N;
      int j = ((row*nColor+col)*2) / N; // factor of two for complexity
      int i = ((row*nColor+col)*2) % N;
      int index = ((x_cb + dim*stride*M + j*stride)*2+i) / 2; // back to a complex offset
      index += parity*offset_cb;
      return index;
    };

    template <typename Float, int nColor, typename storeFloat>
    struct Accessor<Float, nColor, QUDA_FLOAT2_GAUGE_ORDER, storeFloat> {
      using wrapper = fieldorder_wrapper<Float, storeFloat>;
      static constexpr bool is_mma_compatible = false;
      complex<storeFloat> *u;
      const unsigned int offset_cb;
      const unsigned int volumeCB;
      const unsigned int stride;
      const int geometry;
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();

      Accessor(const GaugeField &U, void *gauge_ = nullptr, void ** = nullptr) :
        u(gauge_ ? static_cast<complex<storeFloat> *>(gauge_) :
                   static_cast<complex<storeFloat> *>(const_cast<void *>(U.Gauge_p()))),
        offset_cb((U.Bytes() >> 1) / sizeof(complex<storeFloat>)),
        volumeCB(U.VolumeCB()),
        stride(U.Stride()),
        geometry(U.Geometry()),
        scale(static_cast<Float>(1.0)),
        scale_inv(static_cast<Float>(1.0))
      {
	resetScale(U.Scale());
      }

      void resetScale(Float max)
      {
        if (fixed) {
          scale = static_cast<Float>(std::numeric_limits<storeFloat>::max()) / max;
	  scale_inv = max / static_cast<Float>(std::numeric_limits<storeFloat>::max());
        }
      }

      __device__ __host__ inline wrapper operator()(int dim, int parity, int x_cb, int row, int col) const
      {
        auto index = parity * offset_cb + dim * stride * nColor * nColor + (row * nColor + col) * stride + x_cb;
        return fieldorder_wrapper<Float,storeFloat>(u, index, scale, scale_inv);
      }

      template <typename theirFloat>
      __device__ __host__ void atomic_add(int dim, int parity, int x_cb, int row, int col,
                                          const complex<theirFloat> &val) const
      {
        using vec2 = array<storeFloat, 2>;
        vec2 *u2 = reinterpret_cast<vec2*>(u + parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb);

        vec2 val_ = (fixed && !match<storeFloat, theirFloat>()) ?
          vec2 {f2i_round<storeFloat>(scale * val.real()), f2i_round<storeFloat>(scale * val.imag())} :
          vec2 {static_cast<storeFloat>(val.real()), static_cast<storeFloat>(val.imag())};

        atomic_fetch_add(u2, val_);
      }

      /**
         @brief Wrapper to transform_reduce which is called by the
         reducer functions, e.g., norm2 and abs_max
         @tparam reducer The reduction operation we which to apply
         @param[in] location The location of execution
         @param[in] dim The dimension of of the field we wish to
         reduce.  If dim = -1, then we reduce over all dimensions.
         @param[in] h The helper functor which acts as the transformer
         in transform_reduce
       */
      template <typename reducer, typename helper>
      __host__ double transform_reduce(QudaFieldLocation location, int dim, helper h) const
      {
        if (dim >= geometry) errorQuda("Requested dimension %d exceeds dimensionality of the field %d", dim, geometry);
        auto start = (dim == -1) ? 0 : dim;
        auto count = (dim == -1 ? geometry : 1) * stride * nColor * nColor;
        auto init = reducer::init();
        std::vector<decltype(init)> result = {init, init};
        std::vector<decltype(u)> v = {u + 0 * offset_cb + start * count, u + 1 * offset_cb + start * count};
        ::quda::transform_reduce<reducer>(location, result, v, count, h);
        return reducer::apply(result[0], result[1]);
      }
    };

    template <typename Float, int nColor, bool native_ghost, typename storeFloat>
    struct GhostAccessor<Float, nColor, QUDA_FLOAT2_GAUGE_ORDER, native_ghost, storeFloat> {
      using wrapper = fieldorder_wrapper<Float, storeFloat>;
      complex<storeFloat> *ghost[8];
      const int volumeCB;
      unsigned int ghostVolumeCB[8];
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();
      Accessor<Float, nColor, QUDA_FLOAT2_GAUGE_ORDER, storeFloat> accessor;

      GhostAccessor(const GaugeField &U, void *gauge_, void **ghost_ = 0) :
        volumeCB(U.VolumeCB()),
        scale(static_cast<Float>(1.0)),
        scale_inv(static_cast<Float>(1.0)),
        accessor(U, gauge_, ghost_)
      {
        if constexpr (!native_ghost) assert(ghost_ != nullptr);
        for (int d = 0; d < 4; d++) {
          ghost[d] = !native_ghost ? static_cast<complex<storeFloat>*>(ghost_[d]) : nullptr;
	  ghostVolumeCB[d] = U.Nface()*U.SurfaceCB(d);
	  ghost[d+4] = !native_ghost && U.Geometry() == QUDA_COARSE_GEOMETRY? static_cast<complex<storeFloat>*>(ghost_[d+4]) : nullptr;
	  ghostVolumeCB[d+4] = U.Nface()*U.SurfaceCB(d);
        }
        resetScale(U.Scale());
      }

      void resetScale(Float max)
      {
        accessor.resetScale(max);
        if (fixed) {
          scale = static_cast<Float>(std::numeric_limits<storeFloat>::max()) / max;
          scale_inv = max / static_cast<Float>(std::numeric_limits<storeFloat>::max());
        }
      }

      __device__ __host__ inline wrapper operator()(int d, int parity, int x_cb, int row, int col) const
      {
        if constexpr (native_ghost)
          return accessor(d % 4, parity, x_cb + (d / 4) * ghostVolumeCB[d] + volumeCB, row, col);
        return wrapper(ghost[d], ((parity * nColor + row) * nColor + col) * ghostVolumeCB[d] + x_cb, scale, scale_inv);
      }
    };

    /**
       This is a template driven generic gauge field accessor.  To
       deploy for a specifc field ordering, the two operator()
       accessors have to be specialized for that ordering.

       @tparam Float_ Underlying type returned by the accessors
       @tparam nColor Number of colors for the field
       @tparam nSpinCoarse Number of "spin degrees of freedom" (for coarse-link fields only)
       @tparam order Storage order of the field
       @tparam native_ghost Whether to use native ghosts (inlined into
       @tparam storeFloat_ Underlying storage type for the field
       the padded area for internal-order fields or use a separate array if false)
     */
    template <typename Float_, int nColor, int nSpinCoarse, QudaGaugeFieldOrder order, bool native_ghost = true,
              typename storeFloat_ = Float_>
    struct FieldOrder {

      /** Convenient types */
      using Float = Float_;
      using storeFloat = storeFloat_;
      using wrapper = fieldorder_wrapper<Float, storeFloat>;

      /** An internal reference to the actual field we are accessing */
      const int volumeCB;
      const int nDim;
      const int_fastdiv geometry;
      const QudaFieldLocation location;
      static constexpr int nColorCoarse = nColor / nSpinCoarse;

      using accessor_type = Accessor<Float, nColor, order, storeFloat>;
      static constexpr bool is_mma_compatible = accessor_type::is_mma_compatible;
      accessor_type accessor;
      GhostAccessor<Float, nColor, order, native_ghost, storeFloat> ghostAccessor;

      /** Does this field type support ghost zones? */
      static constexpr bool supports_ghost_zone = true;

      /**
       * Constructor for the FieldOrder class
       * @param field The field that we are accessing
       */
      FieldOrder(const GaugeField &U, storeFloat *gauge_ = 0, storeFloat **ghost_ = 0) :
        volumeCB(U.VolumeCB()),
        nDim(U.Ndim()),
        geometry(U.Geometry()),
        location(U.Location()),
        accessor(U, (void *)gauge_, (void **)ghost_),
        ghostAccessor(U, (void *)gauge_, (void **)ghost_)
      {
        if (U.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("GaugeField ordering not supported with reconstruction");
	}

	void resetScale(double max) {
	  accessor.resetScale(max);
	  ghostAccessor.resetScale(max);
	}

	static constexpr bool fixedPoint() { return fixed_point<Float,storeFloat>(); }

        /**
         * accessor function
         * @param d dimension index
         * @param parity Parity index
         * @param x 1-d site index
         * @param row row index
         * @param c column index
         */
        __device__ __host__ inline auto operator()(int d, int parity, int x, int row, int col) const
        {
          return accessor(d, parity, x, row, col);
        }

        __device__ __host__ inline auto Ghost(int d, int parity, int x) const { return ghostAccessor(d, parity, x); }

        /**
         * accessor function for the ghost zone
         * @param d dimension index
         * @param parity Parity index
         * @param x 1-d site index
         * @param row row index
         * @param c column index
         */
        __device__ __host__ auto Ghost(int d, int parity, int x, int row, int col) const
        {
          return ghostAccessor(d, parity, x, row, col);
        }
        /**
         * @brief This and the following method (eventually) creates a fieldorder_wrapper object whose pointer points to
         * the start of the memory chunk corresponds to the matrix at d, parity, x. Only available for the
         * QUDA_MILC_GAUGE_ORDER order.

         * @param d dimension index
         * @param parity Parity index
         * @param x 1-d site index
         */
        __device__ __host__ auto wrap_ghost(int d, int parity, int x) const
        {
          return ghostAccessor(d, parity, x, 0, 0);
        }

        /**
         * Specialized complex-member accessor function (for coarse gauge field)
         * @param d dimension index
         * @param parity Parity index
         * @param x 1-d site index
         * @param s_row row spin index
         * @param c_row row color index
         * @param s_col col spin index
         * @param c_col col color index
         */
        __device__ __host__ inline auto operator()(int d, int parity, int x, int s_row, int s_col, int c_row,
                                                   int c_col) const
        {
          return (*this)(d, parity, x, s_row * nColorCoarse + c_row, s_col * nColorCoarse + c_col);
        }

        /**
         * Specialized complex-member accessor function (for coarse gauge field ghost zone)
         * @param d dimension index
         * @param parity Parity index
         * @param x 1-d site index
         * @param s_row row spin index
         * @param c_row row color index
         * @param s_col col spin index
         * @param c_col col color index
         */
        __device__ __host__ inline auto Ghost(int d, int parity, int x, int s_row, int s_col, int c_row, int c_col) const
        {
          return Ghost(d, parity, x, s_row * nColorCoarse + c_row, s_col * nColorCoarse + c_col);
        }

        template <typename theirFloat>
        __device__ __host__ inline void atomicAdd(int d, int parity, int x, int s_row, int s_col, int c_row, int c_col,
                                                  const complex<theirFloat> &val) const
        {
          accessor.atomic_add(d, parity, x, s_row*nColorCoarse + c_row, s_col*nColorCoarse + c_col, val);
        }

        /** Returns the number of field colors */
	__device__ __host__ inline int Ncolor() const { return nColor; }

	/** Returns the field volume */
	__device__ __host__ inline int Volume() const { return 2*volumeCB; }

	/** Returns the field volume */
	__device__ __host__ inline int VolumeCB() const { return volumeCB; }

	/** Returns the field geometric dimension */
	__device__ __host__ inline int Ndim() const { return nDim; }

	/** Returns the field geometry */
	__device__ __host__ inline int Geometry() const { return geometry; }

	/** Returns the number of coarse gauge field spins */
	__device__ __host__ inline int NspinCoarse() const { return nSpinCoarse; }

	/** Returns the number of coarse gauge field colors */
	__device__ __host__ inline int NcolorCoarse() const { return nColorCoarse; }

	/**
	 * @brief Returns the L1 norm of the field in a given dimension
	 * @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
	 * @return L1 norm
	 */
	__host__ double norm1(int dim=-1, bool global=true) const {
          commGlobalReductionPush(global);
          double nrm1 = accessor.template transform_reduce<plus<double>>(location, dim,
                                                                         abs_<double, storeFloat>(accessor.scale_inv));
          commGlobalReductionPop();
          return nrm1;
        }

        /**
         * @brief Returns the L2 norm squared of the field in a given dimension
         * @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
         * @return L2 norm squared
         */
        __host__ double norm2(int dim = -1, bool global = true) const
        {
          commGlobalReductionPush(global);
          double nrm2 = accessor.template transform_reduce<plus<double>>(
            location, dim, square_<double, storeFloat>(accessor.scale_inv));
          commGlobalReductionPop();
          return nrm2;
        }

        /**
         * @brief Returns the Linfinity norm of the field in a given dimension
         * @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
         * @return Linfinity norm
         */
        __host__ double abs_max(int dim = -1, bool global = true) const
        {
          commGlobalReductionPush(global);
          double absmax = accessor.template transform_reduce<maximum<Float>>(
            location, dim, abs_max_<Float, storeFloat>(accessor.scale_inv));
          commGlobalReductionPop();
          return absmax;
        }

        /**
         * @brief Returns the minimum absolute value of the field
         * @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
         * @return Minimum norm
         */
        __host__ double abs_min(int dim = -1, bool global = true) const
        {
          commGlobalReductionPush(global);
          double absmin = accessor.template transform_reduce<minimum<Float>>(
            location, dim, abs_min_<Float, storeFloat>(accessor.scale_inv));
          commGlobalReductionPop();
          return absmin;
        }

        /** Return the size of the allocation (geometry and parity left out and added as needed in Tunable::bytes) */
        size_t Bytes() const { return static_cast<size_t>(volumeCB) * nColor * nColor * 2ll * sizeof(storeFloat); }
    };

      /**
         @brief Generic reconstruction helper with no reconstruction
         @tparam N number of real numbers in each packed gauge matrix
         @tparam Float Storage format (e.g., double, float, short)
         @tparam ghostExchange_ optional template the ghostExchange type
         to avoid the run-time overhead (dummy for trivial reconstruct
         type)
      */
      template <int N, typename Float, QudaGhostExchange ghostExchange_, QudaStaggeredPhase = QUDA_STAGGERED_PHASE_NO>
      struct Reconstruct {
        using real = typename mapper<Float>::type;
        using complex = complex<real>;
        real scale;
        real scale_inv;
        Reconstruct(const GaugeField &u) :
          scale(isFixed<Float>::value ? u.LinkMax() : 1.0),
          scale_inv(isFixed<Float>::value ? 1.0 / scale : 1.0)
        {
        }

        __device__ __host__ inline void Pack(real out[N], const complex in[N / 2]) const
        {
          if constexpr (isFixed<Float>::value) {
#pragma unroll
            for (int i = 0; i < N / 2; i++) {
              out[2 * i + 0] = scale_inv * in[i].real();
              out[2 * i + 1] = scale_inv * in[i].imag();
            }
          } else {
#pragma unroll
            for (int i = 0; i < N / 2; i++) {
              out[2 * i + 0] = in[i].real();
              out[2 * i + 1] = in[i].imag();
            }
          }
        }

        template <typename I>
        __device__ __host__ inline void Unpack(complex out[N / 2], const real in[N], int, int, real, const I *,
                                               const int *) const
        {
          if constexpr (isFixed<Float>::value) {
#pragma unroll
            for (int i = 0; i < N / 2; i++) { out[i] = scale * complex(in[2 * i + 0], in[2 * i + 1]); }
          } else {
#pragma unroll
            for (int i = 0; i < N / 2; i++) { out[i] = complex(in[2 * i + 0], in[2 * i + 1]); }
          }
        }
        __device__ __host__ inline real getPhase(const complex[]) const { return 0; }
      };

      /**
         @brief timeBoundary Compute boundary condition correction
         @tparam ghostExhange_ Optional template the ghostExchange type to avoid the run-time overhead
         @param idx extended field linear index
         @param X the gauge field dimensions
         @param R the radii dimenions of the extended region
         @param tBoundary the boundary condition
         @param isFirstTimeSlice if we're on the first time slice of nodes
         @param isLastTimeSlide if we're on the last time slice of nodes
         @param ghostExchange if the field is extended or not (determines indexing type)
      */
      template <QudaGhostExchange ghostExchange_, typename T, typename I>
      __device__ __host__ inline T timeBoundary(int idx, const I X[QUDA_MAX_DIM], const int R[QUDA_MAX_DIM],
          T tBoundary, T scale, int firstTimeSliceBound, int lastTimeSliceBound, bool isFirstTimeSlice,
          bool isLastTimeSlice, QudaGhostExchange ghostExchange = QUDA_GHOST_EXCHANGE_NO)
      {

        // MWTODO: should this return tBoundary : scale or tBoundary*scale : scale

        if (ghostExchange_ == QUDA_GHOST_EXCHANGE_PAD
            || (ghostExchange_ == QUDA_GHOST_EXCHANGE_INVALID && ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED)) {
          if (idx >= firstTimeSliceBound) { // halo region on the first time slice
            return isFirstTimeSlice ? tBoundary : scale;
          } else if (idx >= lastTimeSliceBound) { // last link on the last time slice
            return isLastTimeSlice ? tBoundary : scale;
          } else {
            return scale;
          }
        } else if (ghostExchange_ == QUDA_GHOST_EXCHANGE_EXTENDED
            || (ghostExchange_ == QUDA_GHOST_EXCHANGE_INVALID && ghostExchange == QUDA_GHOST_EXCHANGE_EXTENDED)) {
          if (idx >= (R[3] - 1) * X[0] * X[1] * X[2] / 2 && idx < R[3] * X[0] * X[1] * X[2] / 2) {
            // the boundary condition is on the R[3]-1 time slice
            return isFirstTimeSlice ? tBoundary : scale;
          } else if (idx >= (X[3] - R[3] - 1) * X[0] * X[1] * X[2] / 2 && idx < (X[3] - R[3]) * X[0] * X[1] * X[2] / 2) {
            // the boundary condition lies on the X[3]-R[3]-1 time slice
            return isLastTimeSlice ? tBoundary : scale;
          } else {
            return scale;
          }
        }
        return scale;
      }

      // not actually used - here for reference
      template <typename Float, typename I>
      __device__ __host__ inline Float milcStaggeredPhase(int dim, const int x[], const I R[]) {
        // could consider non-extended variant too?
        Float sign = static_cast<Float>(1.0);
	switch (dim) {
	case 0: if ( ((x[3] - R[3]) & 1) != 0)                             sign = -static_cast<Float>(1.0); break;
	case 1: if ( ((x[0] - R[0] + x[3] - R[3]) & 1) != 0)               sign = -static_cast<Float>(1.0); break;
	case 2: if ( ((x[0] - R[0] + x[1] - R[1] + x[3] - R[3]) & 1) != 0) sign = -static_cast<Float>(1.0); break;
	}
	return sign;
      }

      /**
         @brief Gauge reconstruct 12 helper where we reconstruct the
         third row from the cross product of the first two rows
         @tparam Float Storage format (e.g., double, float, short)
         @tparam ghostExchange_ optional template the ghostExchange
         type to avoid the run-time overhead
      */
      template <typename Float, QudaGhostExchange ghostExchange_> struct Reconstruct<12, Float, ghostExchange_> {
        using real = typename mapper<Float>::type;
        using complex = complex<real>;
        const real anisotropy;
        const real tBoundary;
        const int firstTimeSliceBound;
        const int lastTimeSliceBound;
        const bool isFirstTimeSlice;
        const bool isLastTimeSlice;
        QudaGhostExchange ghostExchange;

        Reconstruct(const GaugeField &u) :
          anisotropy(u.Anisotropy()),
          tBoundary(static_cast<real>(u.TBoundary())),
          firstTimeSliceBound(u.VolumeCB()),
          lastTimeSliceBound((u.X()[3] - 1) * u.X()[0] * u.X()[1] * u.X()[2] / 2),
          isFirstTimeSlice(comm_coord(3) == 0 ? true : false),
          isLastTimeSlice(comm_coord(3) == comm_dim(3) - 1 ? true : false),
          ghostExchange(u.GhostExchange())
        {
        }

        __device__ __host__ inline void Pack(real out[12], const complex in[9]) const
        {
#pragma unroll
          for (int i = 0; i < 6; i++) {
            out[2 * i + 0] = in[i].real();
            out[2 * i + 1] = in[i].imag();
          }
        }

        template <typename I>
        __device__ __host__ inline void Unpack(complex out[9], const real in[12], int idx, int dir, real, const I *X,
                                               const int *R) const
        {
#pragma unroll
          for (int i = 0; i < 6; i++) out[i] = complex(in[2 * i + 0], in[2 * i + 1]);

          const real u0 = dir < 3 ?
            anisotropy :
            timeBoundary<ghostExchange_>(idx, X, R, tBoundary, static_cast<real>(1.0), firstTimeSliceBound,
                                         lastTimeSliceBound, isFirstTimeSlice, isLastTimeSlice, ghostExchange);

          // out[6] = u0*conj(out[1]*out[5] - out[2]*out[4]);
          out[6] = cmul(out[2], out[4]);
          out[6] = cmac(out[1], out[5], -out[6]);
          out[6] = u0 * conj(out[6]);

          // out[7] = u0*conj(out[2]*out[3] - out[0]*out[5]);
          out[7] = cmul(out[0], out[5]);
          out[7] = cmac(out[2], out[3], -out[7]);
          out[7] = u0 * conj(out[7]);

          // out[8] = u0*conj(out[0]*out[4] - out[1]*out[3]);
          out[8] = cmul(out[1], out[3]);
          out[8] = cmac(out[0], out[4], -out[8]);
          out[8] = u0 * conj(out[8]);
        }

        __device__ __host__ inline real getPhase(const complex[]) const { return 0; }
      };

      /**
         @brief Gauge reconstruct helper for Momentum field with 10
         packed elements (really 9 from the Lie algebra, with zero for
         last element).  We label this as 11 to avoid collisions with
         simple load/store of momentum field where we do not seek to
         unpack/pack.
         @tparam Float Storage format (e.g., double, float, short)
         @tparam ghostExchange_ optional template the ghostExchange
         type to avoid the run-time overhead
      */
      template <typename Float, QudaGhostExchange ghostExchange_> struct Reconstruct<11, Float, ghostExchange_> {
        using real = typename mapper<Float>::type;
        using complex = complex<real>;

        Reconstruct(const GaugeField &) { ; }

        __device__ __host__ inline void Pack(real out[10], const complex in[9]) const
        {
#pragma unroll
          for (int i = 0; i < 2; i++) {
            out[2 * i + 0] = in[i + 1].real();
            out[2 * i + 1] = in[i + 1].imag();
          }
          out[4] = in[5].real();
          out[5] = in[5].imag();
          out[6] = in[0].imag();
          out[7] = in[4].imag();
          out[8] = in[8].imag();
          out[9] = 0.0;
        }

        template <typename I>
        __device__ __host__ inline void Unpack(complex out[9], const real in[10], int, int, real, const I *,
                                               const int *) const
        {
          out[0] = complex(0.0, in[6]);
          out[1] = complex(in[0], in[1]);
          out[2] = complex(in[2], in[3]);
          out[3] = complex(-out[1].real(), out[1].imag());
          out[4] = complex(0.0, in[7]);
          out[5] = complex(in[4], in[5]);
          out[6] = complex(-out[2].real(), out[2].imag());
          out[7] = complex(-out[5].real(), out[5].imag());
          out[8] = complex(0.0, in[8]);
        }

        __device__ __host__ inline real getPhase(const complex[]) const { return 0; }
      };

      /**
         @brief Gauge reconstruct 13 helper where we reconstruct the
         third row from the cross product of the first two rows, and
         include a non-trivial phase factor
         @tparam Float Storage format (e.g., double, float, short)
         @tparam ghostExchange_ optional template the ghostExchange
         type to avoid the run-time overhead
      */
      template <typename Float, QudaGhostExchange ghostExchange_, QudaStaggeredPhase stag_phase>
      struct Reconstruct<13, Float, ghostExchange_, stag_phase> {
        using real = typename mapper<Float>::type;
        using complex = complex<real>;
        const Reconstruct<12, Float, ghostExchange_> reconstruct_12;
        const real scale;
        const real scale_inv;

        Reconstruct(const GaugeField &u) :
          reconstruct_12(u), scale(u.Scale() == 0 ? 1.0 : u.Scale()), scale_inv(1.0 / scale)
        {
        }

        __device__ __host__ inline void Pack(real out[12], const complex in[9]) const { reconstruct_12.Pack(out, in); }

        template <typename I>
        __device__ __host__ inline void Unpack(complex out[9], const real in[12], int, int, real phase, const I *,
                                               const int *) const
        {
#pragma unroll
          for (int i = 0; i < 6; i++) out[i] = complex(in[2 * i + 0], in[2 * i + 1]);

          out[6] = cmul(out[2], out[4]);
          out[6] = cmac(out[1], out[5], -out[6]);
          out[6] = scale_inv * conj(out[6]);

          out[7] = cmul(out[0], out[5]);
          out[7] = cmac(out[2], out[3], -out[7]);
          out[7] = scale_inv * conj(out[7]);

          out[8] = cmul(out[1], out[3]);
          out[8] = cmac(out[0], out[4], -out[8]);
          out[8] = scale_inv * conj(out[8]);

          if constexpr (stag_phase == QUDA_STAGGERED_PHASE_NO) { // dynamic phasing
            // Multiply the third row by exp(I*3*phase), since the cross product will end up in a scale factor of exp(-I*2*phase)
            real cos_sin[2];
            sincospi(static_cast<real>(3.0) * phase, &cos_sin[1], &cos_sin[0]);
            complex A(cos_sin[0], cos_sin[1]);
            out[6] = cmul(A, out[6]);
            out[7] = cmul(A, out[7]);
            out[8] = cmul(A, out[8]);
          } else { // phase is +/- 1 so real multiply is sufficient
            out[6] *= phase;
            out[7] *= phase;
            out[8] *= phase;
          }
        }

        __device__ __host__ inline real getPhase(const complex in[9]) const
        {
#if 1 // phase from cross product
          // denominator = (U[0][0]*U[1][1] - U[0][1]*U[1][0])*
          complex denom = conj(in[0] * in[4] - in[1] * in[3]) * scale_inv;
          complex expI3Phase = in[8] / denom; // numerator = U[2][2]

          // dynamic phasing
          if constexpr (stag_phase == QUDA_STAGGERED_PHASE_NO) return arg(expI3Phase) / static_cast<real>(3.0 * M_PI);
          // static phasing
          return expI3Phase.real() > 0 ? 1 : -1;
#else // phase from determinant
          Matrix<complex, 3> a;
#pragma unroll
          for (int i = 0; i < 9; i++) a(i) = scale_inv * in[i];
          const complex det = getDeterminant(a);
          return phase = arg(det) / static_cast<real>(3.0 * M_PI);
#endif
        }
      };

      /**
         @brief Gauge reconstruct 8 helper where we reconstruct the gauge
         matrix from 8 packed elements (maximal compression)
         @tparam Float Storage format (e.g., double, float, short)
         @tparam ghostExchange_ optional template the ghostExchange type
         to avoid the run-time overhead
      */
      template <typename Float, QudaGhostExchange ghostExchange_> struct Reconstruct<8, Float, ghostExchange_> {
        using real = typename mapper<Float>::type;
        using complex = complex<real>;
        const complex anisotropy; // imaginary value stores inverse
        const complex tBoundary;  // imaginary value stores inverse
        const int firstTimeSliceBound;
        const int lastTimeSliceBound;
        const bool isFirstTimeSlice;
        const bool isLastTimeSlice;
        QudaGhostExchange ghostExchange;

        // scale factor is set when using recon-9
        Reconstruct(const GaugeField &u, real scale = 1.0) :
          anisotropy(u.Anisotropy() * scale, 1.0 / (u.Anisotropy() * scale)),
          tBoundary(static_cast<real>(u.TBoundary()) * scale, 1.0 / (static_cast<real>(u.TBoundary()) * scale)),
          firstTimeSliceBound(u.VolumeCB()),
          lastTimeSliceBound((u.X()[3] - 1) * u.X()[0] * u.X()[1] * u.X()[2] / 2),
          isFirstTimeSlice(comm_coord(3) == 0 ? true : false),
          isLastTimeSlice(comm_coord(3) == comm_dim(3) - 1 ? true : false),
          ghostExchange(u.GhostExchange())
        {
        }

        // Pack and unpack are described in https://arxiv.org/pdf/0911.3191.pdf
        // Method was modified to avoid the singularity at unit gauge by
        // compressing the matrix {{b1,b2,b3},{a1,a2,a3},{-c1,-c2,-c3}}
        // instead of {{a1,a2,a3},{b1,b2,b3},{c1,c2,c3}}

        __device__ __host__ inline void Pack(real out[8], const complex in[9]) const
        {
          out[0] = atan2(in[3].imag(), in[3].real()) / static_cast<real>(M_PI);   // a1 -> b1
          out[1] = atan2(-in[6].imag(), -in[6].real()) / static_cast<real>(M_PI); // c1 -> -c1

          out[2] = in[4].real();
          out[3] = in[4].imag(); // a2 -> b2
          out[4] = in[5].real();
          out[5] = in[5].imag(); // a3 -> b3
          out[6] = in[0].real();
          out[7] = in[0].imag(); // b1 -> a1
        }

        template <typename I>
        __device__ __host__ inline void Unpack(complex out[9], const real in[8], int, int, real, const I *, const int *,
                                               const complex, const complex u) const
        {
          real u0 = u.real();
          real u0_inv = u.imag();

#pragma unroll
          for (int i = 1; i <= 3; i++)
            out[i] = complex(in[2 * i + 0], in[2 * i + 1]); // these elements are copied directly

          real tmp[2];
          quda::sincospi(in[0], &tmp[1], &tmp[0]);
          out[0] = complex(tmp[0], tmp[1]);

          quda::sincospi(in[1], &tmp[1], &tmp[0]);
          out[6] = complex(tmp[0], tmp[1]);

          // First, reconstruct first row
          real row_sum = out[1].real() * out[1].real();
          row_sum += out[1].imag() * out[1].imag();
          row_sum += out[2].real() * out[2].real();
          row_sum += out[2].imag() * out[2].imag();
          real row_sum_inv = static_cast<real>(1.0) / row_sum;

          real diff = u0_inv * u0_inv - row_sum;
          real U00_mag = diff > 0.0 ? diff * quda::rsqrt(diff) : static_cast<real>(0.0);

          out[0] *= U00_mag;

          // Second, reconstruct first column
          real column_sum = out[0].real() * out[0].real();
          column_sum += out[0].imag() * out[0].imag();
          column_sum += out[3].real() * out[3].real();
          column_sum += out[3].imag() * out[3].imag();

          diff = u0_inv * u0_inv - column_sum;
          real U20_mag = diff > 0.0 ? diff * quda::rsqrt(diff) : static_cast<real>(0.0);

          out[6] *= U20_mag;

          // Finally, reconstruct last elements from SU(2) rotation
          real r_inv2 = u0_inv * row_sum_inv;
          {
            complex A = cmul(conj(out[0]), out[3]);

            // out[4] = -(conj(out[6])*conj(out[2]) + u0*A*out[1])*r_inv2; // U11
            out[4] = cmul(conj(out[6]), conj(out[2]));
            out[4] = cmac(u0 * A, out[1], out[4]);
            out[4] = -r_inv2 * out[4];

            // out[5] = (conj(out[6])*conj(out[1]) - u0*A*out[2])*r_inv2;  // U12
            out[5] = cmul(conj(out[6]), conj(out[1]));
            out[5] = cmac(-u0 * A, out[2], out[5]);
            out[5] = r_inv2 * out[5];
          }

          {
            complex A = cmul(conj(out[0]), out[6]);

            // out[7] = (conj(out[3])*conj(out[2]) - u0*A*out[1])*r_inv2;  // U21
            out[7] = cmul(conj(out[3]), conj(out[2]));
            out[7] = cmac(-u0 * A, out[1], out[7]);
            out[7] = r_inv2 * out[7];

            // out[8] = -(conj(out[3])*conj(out[1]) + u0*A*out[2])*r_inv2; // U12
            out[8] = cmul(conj(out[3]), conj(out[1]));
            out[8] = cmac(u0 * A, out[2], out[8]);
            out[8] = -r_inv2 * out[8];
          }

          // Rearrange {{b1,b2,b3},{a1,a2,a3},{-c1,-c2,-c3}} back
          // to {{a1,a2,a3},{b1,b2,b3},{c1,c2,c3}}
#pragma unroll
          for (int i = 0; i < 3; i++) {
            const auto tmp = out[i];
            out[i] = out[i + 3];
            out[i + 3] = tmp;
            out[i + 6] = -out[i + 6];
          }
        }

        template <typename I>
        __device__ __host__ inline void
        Unpack(complex out[9], const real in[8], int idx, int dir, real phase, const I *X, const int *R,
               const complex scale = complex(static_cast<real>(1.0), static_cast<real>(1.0))) const
        {
          complex u = dir < 3 ?
            anisotropy :
            timeBoundary<ghostExchange_>(idx, X, R, tBoundary, scale, firstTimeSliceBound, lastTimeSliceBound,
                                         isFirstTimeSlice, isLastTimeSlice, ghostExchange);

          Unpack(out, in, idx, dir, phase, X, R, scale, u);
        }

        __device__ __host__ inline real getPhase(const complex[]) const { return 0; }
      };

      /**
         @brief Gauge reconstruct 9 helper where we reconstruct the gauge
         matrix from 8 packed elements (maximal compression) and include a
         non-trivial phase factor
         @tparam Float Storage format (e.g., double, float, short)
         @tparam ghostExchange_ optional template the ghostExchange type
         to avoid the run-time overhead
      */
      template <typename Float, QudaGhostExchange ghostExchange_, QudaStaggeredPhase stag_phase>
      struct Reconstruct<9, Float, ghostExchange_, stag_phase> {
        using real = typename mapper<Float>::type;
        using complex = complex<real>;
        const Reconstruct<8, Float, ghostExchange_> reconstruct_8;
        const real scale;
        const real scale_inv;

        Reconstruct(const GaugeField &u) :
          reconstruct_8(u), scale(u.Scale() == 0 ? 1.0 : u.Scale()), scale_inv(1.0 / scale)
        {
        }

        __device__ __host__ inline real getPhase(const complex in[9]) const
        {
#if 1 // phase from cross product
          // denominator = (U[0][0]*U[1][1] - U[0][1]*U[1][0])*
          complex denom = conj(in[0] * in[4] - in[1] * in[3]) * scale_inv;
          complex expI3Phase = in[8] / denom; // numerator = U[2][2]
          // dynamic phasing
          if constexpr (stag_phase == QUDA_STAGGERED_PHASE_NO) return arg(expI3Phase) / static_cast<real>(3.0 * M_PI);
          // static phasing
          return expI3Phase.real() > 0 ? 1 : -1;
#else // phase from determinant
          Matrix<complex, 3> a;
#pragma unroll
          for (int i = 0; i < 9; i++) a(i) = scale_inv * in[i];
          const complex det = getDeterminant(a);
          real phase = arg(det) / static_cast<real>(3.0 * M_PI);
          return phase;
#endif
        }

        // Rescale the U3 input matrix by exp(-I*phase) to obtain an SU3 matrix multiplied by a real scale factor,
        __device__ __host__ inline void Pack(real out[8], const complex in[9]) const
        {
          real phase = getPhase(in);
          complex su3[9];

          if constexpr (stag_phase == QUDA_STAGGERED_PHASE_NO) {
            real cos_sin[2];
            sincospi(static_cast<real>(-phase), &cos_sin[1], &cos_sin[0]);
            complex z(cos_sin[0], cos_sin[1]);
            z *= scale_inv;
#pragma unroll
            for (int i = 0; i < 9; i++) su3[i] = cmul(z, in[i]);
          } else {
#pragma unroll
            for (int i = 0; i < 9; i++) { su3[i] = phase * in[i]; }
          }
          reconstruct_8.Pack(out, su3);
        }

        template <typename I>
        __device__ __host__ inline void Unpack(complex out[9], const real in[8], int idx, int dir, real phase,
                                               const I *X, const int *R) const
        {
          reconstruct_8.Unpack(out, in, idx, dir, phase, X, R, complex(static_cast<real>(1.0), static_cast<real>(1.0)),
                               complex(static_cast<real>(1.0), static_cast<real>(1.0)));

          if constexpr (stag_phase == QUDA_STAGGERED_PHASE_NO) { // dynamic phase
            real cos_sin[2];
            sincospi(static_cast<real>(phase), &cos_sin[1], &cos_sin[0]);
            complex z(cos_sin[0], cos_sin[1]);
            z *= scale;
#pragma unroll
            for (int i = 0; i < 9; i++) out[i] = cmul(z, out[i]);
          } else { // stagic phase
#pragma unroll
            for (int i = 0; i < 18; i++) { out[i] *= phase; }
          }
        }
      };

      __host__ __device__ constexpr int ct_sqrt(int n, int i = 1)
      {
        return n == i ? n : (i * i < n ? ct_sqrt(n, i + 1) : i);
      }

      /**
         @brief Return the number of colors of the accessor based on the length of the field
         @param[in] length Number of real numbers per link
         @return Number of colors (=sqrt(length/2))
       */
      __host__ __device__ constexpr int Ncolor(int length) { return ct_sqrt(length / 2); }

      // we default to huge allocations for gauge field (for now)
      constexpr bool default_huge_alloc = true;

      template <QudaStaggeredPhase phase> constexpr bool static_phase()
      {
        switch (phase) {
        case QUDA_STAGGERED_PHASE_MILC:
        case QUDA_STAGGERED_PHASE_CPS:
        case QUDA_STAGGERED_PHASE_TIFR: return true;
        default: return false;
        }
      }

      template <typename Float, int length_, int N, int reconLenParam,
                QudaStaggeredPhase stag_phase = QUDA_STAGGERED_PHASE_NO, bool huge_alloc = default_huge_alloc,
                QudaGhostExchange ghostExchange_ = QUDA_GHOST_EXCHANGE_INVALID, bool use_inphase = false>
      struct FloatNOrder {
        using Accessor
          = FloatNOrder<Float, length_, N, reconLenParam, stag_phase, huge_alloc, ghostExchange_, use_inphase>;

        using store_t = Float;
        static constexpr int length = length_;
        using real = typename mapper<Float>::type;
        using complex = complex<real>;
        typedef typename VectorType<Float, N>::type Vector;
        typedef typename AllocType<huge_alloc>::type AllocInt;
        Reconstruct<reconLenParam, Float, ghostExchange_, stag_phase> reconstruct;
        static constexpr int reconLen = (reconLenParam == 11) ? 10 : reconLenParam;
        static constexpr int hasPhase = (reconLen == 9 || reconLen == 13) ? 1 : 0;
        Float *gauge;
        const AllocInt offset;
        Float *ghost[4];
        QudaGhostExchange ghostExchange;
        int coords[QUDA_MAX_DIM];
        int_fastdiv X[QUDA_MAX_DIM];
        int R[QUDA_MAX_DIM];
        const int volumeCB;
        int faceVolumeCB[4];
        const int stride;
        const int geometry;
        const AllocInt phaseOffset;
        size_t bytes;

        FloatNOrder(const GaugeField &u, Float *gauge_ = 0, Float **ghost_ = 0) :
          reconstruct(u),
          gauge(gauge_ ? gauge_ : (Float *)u.Gauge_p()),
          offset(u.Bytes() / (2 * sizeof(Float) * N)),
          ghostExchange(u.GhostExchange()),
          volumeCB(u.VolumeCB()),
          stride(u.Stride()),
          geometry(u.Geometry()),
          phaseOffset(u.PhaseOffset() / sizeof(Float)),
          bytes(u.Bytes())
        {
          if (geometry == QUDA_COARSE_GEOMETRY)
            errorQuda("This accessor does not support coarse-link fields (lacks support for bidirectional ghost zone");

          // static_assert( !(stag_phase!=QUDA_STAGGERED_PHASE_NO && reconLenParam != 18 && reconLenParam != 12),
          // 	       "staggered phase only presently supported for 18 and 12 reconstruct");
          for (int i = 0; i < 4; i++) {
            X[i] = u.X()[i];
            R[i] = u.R()[i];
            ghost[i] = ghost_ ? ghost_[i] : 0;
            faceVolumeCB[i] = u.SurfaceCB(i) * u.Nface(); // face volume equals surface * depth
          }
        }

      __device__ __host__ inline void load(complex v[length / 2], int x, int dir, int parity, real phase = 1.0) const
      {
        const int M = reconLen / N;
        real tmp[reconLen];

#pragma unroll
        for (int i=0; i<M; i++){
          // first load from memory
          Vector vecTmp = vector_load<Vector>(gauge, parity * offset + (dir * M + i) * stride + x);
          // second do copy converting into register type
#pragma unroll
          for (int j = 0; j < N; j++) copy(tmp[i * N + j], reinterpret_cast<Float *>(&vecTmp)[j]);
        }

        constexpr bool load_phase = (hasPhase && !(static_phase<stag_phase>() && (reconLen == 13 || use_inphase)));
        if constexpr (load_phase) {
          copy(phase, gauge[parity * offset * N + phaseOffset + stride * dir + x]);
          phase *= static_cast<real>(2.0);
        }

        reconstruct.Unpack(v, tmp, x, dir, phase, X, R);
      }

      __device__ __host__ inline void save(const complex v[length / 2], int x, int dir, int parity) const
      {
        const int M = reconLen / N;
        real tmp[reconLen];
        reconstruct.Pack(tmp, v);

#pragma unroll
        for (int i=0; i<M; i++){
	  Vector vecTmp;
	  // first do copy converting into storage type
#pragma unroll
	  for (int j=0; j<N; j++) copy(reinterpret_cast<Float*>(&vecTmp)[j], tmp[i*N+j]);
	  // second do vectorized copy into memory
          vector_store(gauge, parity * offset + x + (dir * M + i) * stride, vecTmp);
        }
        if constexpr (hasPhase) {
          real phase = reconstruct.getPhase(v);
          copy(gauge[parity * offset * N + phaseOffset + dir * stride + x], static_cast<real>(0.5) * phase);
        }
      }

      /**
	 @brief This accessor routine returns a gauge_wrapper to this object,
	 allowing us to overload various operators for manipulating at
	 the site level interms of matrix operations.
	 @param[in] dir Which dimension are we requesting
	 @param[in] x_cb Checkerboarded space-time index we are requesting
	 @param[in] parity Parity we are requesting
	 @return Instance of a gauge_wrapper that curries in access to
	 this field at the above coordinates.
       */
      __device__ __host__ inline auto operator()(int dim, int x_cb, int parity, real phase = 1.0) const
      {
        return gauge_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, x_cb, parity, phase);
      }

      __device__ __host__ inline void loadGhost(complex v[length / 2], int x, int dir, int parity, real inphase = 1.0) const
      {
        if (!ghost[dir]) { // load from main field not separate array
          load(v, volumeCB + x, dir, parity, inphase); // an offset of size volumeCB puts us at the padded region
          // This also works perfectly when phases are stored. No need to change this.
        } else {
          const int M = reconLen / N;
          real tmp[reconLen];

#pragma unroll
          for (int i=0; i<M; i++) {
	    // first do vectorized copy from memory into registers
            Vector vecTmp = vector_load<Vector>(
                ghost[dir] + parity * faceVolumeCB[dir] * (M * N + hasPhase), i * faceVolumeCB[dir] + x);
            // second do copy converting into register type
#pragma unroll
            for (int j = 0; j < N; j++) copy(tmp[i * N + j], reinterpret_cast<Float *>(&vecTmp)[j]);
          }
          real phase = 0.;

          if constexpr (hasPhase) {

            // if(stag_phase == QUDA_STAGGERED_PHASE_MILC )  {
            //   phase = inphase < static_cast<real>(0) ? static_cast<real>(-0.5) : static_cast<real>(0.5);
            // } else {
            copy(phase, ghost[dir][parity * faceVolumeCB[dir] * (M * N + 1) + faceVolumeCB[dir] * M * N + x]);
            phase *= static_cast<real>(2.0);
            // }
          }
          reconstruct.Unpack(v, tmp, x, dir, phase, X, R);
        }
      }

      __device__ __host__ inline void saveGhost(const complex v[length / 2], int x, int dir, int parity) const
      {
        if (!ghost[dir]) { // store in main field not separate array
          save(v, volumeCB + x, dir, parity); // an offset of size volumeCB puts us at the padded region
        } else {
          const int M = reconLen / N;
          real tmp[reconLen];
          reconstruct.Pack(tmp, v);

#pragma unroll
          for (int i=0; i<M; i++) {
	    Vector vecTmp;
	    // first do copy converting into storage type
#pragma unroll
	    for (int j=0; j<N; j++) copy(reinterpret_cast<Float*>(&vecTmp)[j], tmp[i*N+j]);
	    // second do vectorized copy into memory
	    vector_store(ghost[dir]+parity*faceVolumeCB[dir]*(M*N + hasPhase), i*faceVolumeCB[dir]+x, vecTmp);
          }

          if constexpr (hasPhase) {
            real phase = reconstruct.getPhase(v);
            copy(ghost[dir][parity * faceVolumeCB[dir] * (M * N + 1) + faceVolumeCB[dir] * M * N + x],
                 static_cast<real>(0.5) * phase);
          }
        }
      }

      /**
         @brief This accessor routine returns a gauge_ghost_wrapper to this object,
         allowing us to overload various operators for manipulating at
         the site level interms of matrix operations.
         @param[in] dir Which dimension are we requesting
         @param[in] ghost_idx Ghost index we are requesting
         @param[in] parity Parity we are requesting
         @return Instance of a gauge_ghost_wrapper that curries in access to
         this field at the above coordinates.
       */
      __device__ __host__ inline gauge_ghost_wrapper<real, Accessor> Ghost(int dim, int ghost_idx, int parity,
                                                                           real phase = 1.0)
      {
        return gauge_ghost_wrapper<real, Accessor>(*this, dim, ghost_idx, parity, phase);
      }

      /**
         @brief This accessor routine returns a const gauge_ghost_wrapper to this object,
         allowing us to overload various operators for manipulating at
         the site level interms of matrix operations.
         @param[in] dir Which dimension are we requesting
         @param[in] ghost_idx Ghost index we are requesting
         @param[in] parity Parity we are requesting
         @return Instance of a gauge_ghost_wrapper that curries in access to
         this field at the above coordinates.
       */
      __device__ __host__ inline const gauge_ghost_wrapper<real, Accessor> Ghost(int dim, int ghost_idx, int parity,
                                                                                 real phase = 1.0) const
      {
        return gauge_ghost_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, ghost_idx, parity, phase);
      }

      __device__ __host__ inline void loadGhostEx(complex v[length / 2], int buff_idx, int extended_idx, int dir,
                                                  int dim, int g, int parity, const int R[]) const
      {
        const int M = reconLen / N;
        real tmp[reconLen];

#pragma unroll
	for (int i=0; i<M; i++) {
	  // first do vectorized copy from memory
	  Vector vecTmp = vector_load<Vector>(ghost[dim] + ((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + hasPhase),
					      +i*R[dim]*faceVolumeCB[dim]+buff_idx);
	  // second do copy converting into register type
#pragma unroll
	  for (int j=0; j<N; j++) copy(tmp[i*N+j], reinterpret_cast<Float*>(&vecTmp)[j]);
	}
        real phase = 0.;
        if constexpr (hasPhase)
          copy(phase,
               ghost[dim][((dir * 2 + parity) * geometry + g) * R[dim] * faceVolumeCB[dim] * (M * N + 1)
                          + R[dim] * faceVolumeCB[dim] * M * N + buff_idx]);

        // use the extended_idx to determine the boundary condition
        reconstruct.Unpack(v, tmp, extended_idx, g, 2. * phase, X, R);
      }

      __device__ __host__ inline void saveGhostEx(const complex v[length / 2], int buff_idx, int, int dir, int dim,
                                                  int g, int parity, const int R[]) const
      {
        const int M = reconLen / N;
        real tmp[reconLen];
        reconstruct.Pack(tmp, v);

#pragma unroll
	  for (int i=0; i<M; i++) {
	    Vector vecTmp;
	    // first do copy converting into storage type
#pragma unroll
	    for (int j=0; j<N; j++) copy(reinterpret_cast<Float*>(&vecTmp)[j], tmp[i*N+j]);
	    // second do vectorized copy to memory
	    vector_store(ghost[dim] + ((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + hasPhase),
			 i*R[dim]*faceVolumeCB[dim]+buff_idx, vecTmp);
	  }
          if constexpr (hasPhase) {
            real phase = reconstruct.getPhase(v);
            copy(ghost[dim][((dir * 2 + parity) * geometry + g) * R[dim] * faceVolumeCB[dim] * (M * N + 1)
                            + R[dim] * faceVolumeCB[dim] * M * N + buff_idx],
                 static_cast<real>(0.5) * phase);
          }
      }

      size_t Bytes() const { return reconLen * sizeof(Float); }
      };

      /**
         @brief The LegacyOrder defines the ghost zone storage and ordering for
         all cpuGaugeFields, which use the same ghost zone storage.
      */
      template <typename Float, int length_> struct LegacyOrder {
        static constexpr int length = length_;
        using Accessor = LegacyOrder<Float, length>;
        using store_t = Float;
        using real = typename mapper<Float>::type;
        using complex = complex<real>;
        Float *ghost[QUDA_MAX_DIM];
        int faceVolumeCB[QUDA_MAX_DIM];
        const int volumeCB;
        const int stride;
        const int geometry;
        const int hasPhase;

        LegacyOrder(const GaugeField &u, Float **ghost_) :
          volumeCB(u.VolumeCB()),
          stride(u.Stride()),
          geometry(u.Geometry()),
          hasPhase(0)
        {
          if (geometry == QUDA_COARSE_GEOMETRY)
            errorQuda("This accessor does not support coarse-link fields (lacks support for bidirectional ghost zone");

          for (int i = 0; i < 4; i++) {
            ghost[i] = (ghost_) ? ghost_[i] : (Float *)(u.Ghost()[i]);
            faceVolumeCB[i] = u.SurfaceCB(i) * u.Nface(); // face volume equals surface * depth
          }
        }

        __device__ __host__ inline void loadGhost(complex v[length / 2], int x, int dir, int parity, real = 1.0) const
        {
          auto in = &ghost[dir][(parity * faceVolumeCB[dir] + x) * length];
          block_load<complex, length / 2>(v, reinterpret_cast<complex *>(in));
        }

        __device__ __host__ inline void saveGhost(const complex v[length / 2], int x, int dir, int parity)
        {
          auto out = &ghost[dir][(parity * faceVolumeCB[dir] + x) * length];
          block_store<complex, length / 2>(reinterpret_cast<complex *>(out), v);
        }

        /**
           @brief This accessor routine returns a const gauge_ghost_wrapper to this object,
           allowing us to overload various operators for manipulating at
           the site level interms of matrix operations.
           @param[in] dir Which dimension are we requesting
           @param[in] ghost_idx Ghost index we are requesting
           @param[in] parity Parity we are requesting
           @return Instance of a gauge_ghost_wrapper that curries in access to
           this field at the above coordinates.
         */
        __device__ __host__ inline const gauge_ghost_wrapper<real, Accessor> Ghost(int dim, int ghost_idx, int parity,
                                                                                   real phase = 1.0) const
        {
          return gauge_ghost_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, ghost_idx, parity, phase);
        }

        __device__ __host__ inline void loadGhostEx(complex v[length / 2], int x, int, int dir, int dim, int g,
                                                    int parity, const int R[]) const
        {
          auto in = &ghost[dim][(((dir * 2 + parity) * R[dim] * faceVolumeCB[dim] + x) * geometry + g) * length];
          block_load<complex, length / 2>(v, reinterpret_cast<complex *>(in));
        }

        __device__ __host__ inline void saveGhostEx(const complex v[length / 2], int x, int, int dir, int dim, int g,
                                                    int parity, const int R[]) const
        {
          auto out = &ghost[dim][(((dir * 2 + parity) * R[dim] * faceVolumeCB[dim] + x) * geometry + g) * length];
          block_store<complex, length / 2>(reinterpret_cast<complex *>(out), v);
        }
      };

    /**
       struct to define QDP ordered gauge fields:
       [[dim]] [[parity][volumecb][row][col]]
    */
    template <typename Float, int length> struct QDPOrder : public LegacyOrder<Float,length> {
      using Accessor = QDPOrder<Float, length>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      Float *gauge[QUDA_MAX_DIM];
      const int volumeCB;
    QDPOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0)
      : LegacyOrder<Float,length>(u, ghost_), volumeCB(u.VolumeCB())
	{ for (int i=0; i<4; i++) gauge[i] = gauge_ ? ((Float**)gauge_)[i] : ((Float**)u.Gauge_p())[i]; }

        __device__ __host__ inline void load(complex v[length / 2], int x, int dir, int parity, real = 1.0) const
        {
          auto in = &gauge[dir][(parity * volumeCB + x) * length];
          block_load<complex, length / 2>(v, reinterpret_cast<complex *>(in));
      }

      __device__ __host__ inline void save(const complex v[length / 2], int x, int dir, int parity) const
      {
        auto out = &gauge[dir][(parity * volumeCB + x) * length];
        block_store<complex, length / 2>(reinterpret_cast<complex *>(out), v);
      }

      /**
	 @brief This accessor routine returns a gauge_wrapper to this object,
	 allowing us to overload various operators for manipulating at
	 the site level interms of matrix operations.
	 @param[in] dir Which dimension are we requesting
	 @param[in] x_cb Checkerboarded space-time index we are requesting
	 @param[in] parity Parity we are requesting
	 @return Instance of a gauge_wrapper that curries in access to
	 this field at the above coordinates.
       */
      __device__ __host__ inline auto operator()(int dim, int x_cb, int parity) const
      {
        return gauge_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, x_cb, parity);
      }

      size_t Bytes() const { return length * sizeof(Float); }
    };

    /**
       struct to define QDPJIT ordered gauge fields:
       [[dim]] [[parity][complex][row][col][volumecb]]
    */
    template <typename Float, int length> struct QDPJITOrder : public LegacyOrder<Float,length> {
      using Accessor = QDPJITOrder<Float, length>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      Float *gauge[QUDA_MAX_DIM];
      const int volumeCB;
    QDPJITOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0)
      : LegacyOrder<Float,length>(u, ghost_), volumeCB(u.VolumeCB())
	{ for (int i=0; i<4; i++) gauge[i] = gauge_ ? ((Float**)gauge_)[i] : ((Float**)u.Gauge_p())[i]; }

        __device__ __host__ inline void load(complex v[length / 2], int x, int dir, int parity, real = 1.0) const
        {
          for (int i = 0; i < length / 2; i++) {
            v[i].real((real)gauge[dir][((0 * (length / 2) + i) * 2 + parity) * volumeCB + x]);
            v[i].imag((real)gauge[dir][((1 * (length / 2) + i) * 2 + parity) * volumeCB + x]);
          }
      }

      __device__ __host__ inline void save(const complex v[length / 2], int x, int dir, int parity) const
      {
        for (int i = 0; i < length / 2; i++) {
          gauge[dir][((0 * (length / 2) + i) * 2 + parity) * volumeCB + x] = v[i].real();
          gauge[dir][((1 * (length / 2) + i) * 2 + parity) * volumeCB + x] = v[i].imag();
        }
      }

      /**
	 @brief This accessor routine returns a gauge_wrapper to this object,
	 allowing us to overload various operators for manipulating at
	 the site level interms of matrix operations.
	 @param[in] dir Which dimension are we requesting
	 @param[in] x_cb Checkerboarded space-time index we are requesting
	 @param[in] parity Parity we are requesting
	 @return Instance of a gauge_wrapper that curries in access to
	 this field at the above coordinates.
       */
      __device__ __host__ inline auto operator()(int dim, int x_cb, int parity) const
      {
        return gauge_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, x_cb, parity);
      }

      size_t Bytes() const { return length * sizeof(Float); }
    };

  /**
     struct to define MILC ordered gauge fields:
     [parity][dim][volumecb][row][col]
  */
  template <typename Float, int length> struct MILCOrder : public LegacyOrder<Float,length> {
    using Accessor = MILCOrder<Float, length>;
    using real = typename mapper<Float>::type;
    using complex = complex<real>;
    Float *gauge;
    const int volumeCB;
    const int geometry;
  MILCOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) :
    LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()),
      volumeCB(u.VolumeCB()), geometry(u.Geometry()) { ; }

  __device__ __host__ inline void load(complex v[length / 2], int x, int dir, int parity, real = 1.0) const
  {
    auto in = &gauge[((parity * volumeCB + x) * geometry + dir) * length];
    block_load<complex, length / 2>(v, reinterpret_cast<complex *>(in));
    }

    __device__ __host__ inline void save(const complex v[length / 2], int x, int dir, int parity) const
    {
      auto out = &gauge[((parity * volumeCB + x) * geometry + dir) * length];
      block_store<complex, length / 2>(reinterpret_cast<complex *>(out), v);
    }

    /**
       @brief This accessor routine returns a gauge_wrapper to this object,
       allowing us to overload various operators for manipulating at
       the site level interms of matrix operations.
       @param[in] dir Which dimension are we requesting
       @param[in] x_cb Checkerboarded space-time index we are requesting
       @param[in] parity Parity we are requesting
       @return Instance of a gauge_wrapper that curries in access to
       this field at the above coordinates.
    */
    __device__ __host__ inline auto operator()(int dim, int x_cb, int parity) const
    {
      return gauge_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, x_cb, parity);
    }

    size_t Bytes() const { return length * sizeof(Float); }
  };

  /**
     @brief struct to define gauge fields packed into an opaque MILC site struct:

     struct {
       char padding[offset];
       Float [dim][row][col];
     } site;

     site lattice [parity][volumecb];

     We are just passed the size of the struct and the offset to the
     required matrix elements.  Typically, it is expected that this
     accessor will be used with zero-copy memory to the original
     allocation in MILC.
  */
  template <typename Float, int length> struct MILCSiteOrder : public LegacyOrder<Float,length> {
    using Accessor = MILCSiteOrder<Float, length>;
    using real = typename mapper<Float>::type;
    using complex = complex<real>;
    Float *gauge;
    const int volumeCB;
    const int geometry;
    const size_t offset;
    const size_t size;
    MILCSiteOrder(const GaugeField &u, Float *gauge_ = 0, Float **ghost_ = 0) :
      LegacyOrder<Float, length>(u, ghost_),
      gauge(gauge_ ? gauge_ : (Float *)u.Gauge_p()),
      volumeCB(u.VolumeCB()),
      geometry(u.Geometry()),
      offset(u.SiteOffset()),
      size(u.SiteSize())
    {
      if ((uintptr_t)((char *)gauge + offset) % 16 != 0) { errorQuda("MILC structure has misaligned offset"); }
    }

    __device__ __host__ inline void load(complex v[length / 2], int x, int dir, int parity, real = 1.0) const
    {
      // get base pointer
      auto in = reinterpret_cast<const Float *>(reinterpret_cast<const char *>(gauge) + (parity * volumeCB + x) * size
                                                + offset + dir * length * sizeof(Float));
      block_load<complex, length / 2>(v, reinterpret_cast<const complex *>(in));
    }

    __device__ __host__ inline void save(const complex v[length / 2], int x, int dir, int parity) const
    {
      // get base pointer
      auto out = reinterpret_cast<Float *>(reinterpret_cast<char *>(gauge) + (parity * volumeCB + x) * size + offset
                                           + dir * length * sizeof(Float));
      block_store<complex, length / 2>(reinterpret_cast<complex *>(out), v);
    }

    /**
       @brief This accessor routine returns a gauge_wrapper to this object,
       allowing us to overload various operators for manipulating at
       the site level interms of matrix operations.
       @param[in] dir Which dimension are we requesting
       @param[in] x_cb Checkerboarded space-time index we are requesting
       @param[in] parity Parity we are requesting
       @return Instance of a gauge_wrapper that curries in access to
       this field at the above coordinates.
    */
    __device__ __host__ inline auto operator()(int dim, int x_cb, int parity) const
    {
      return gauge_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, x_cb, parity);
    }

    size_t Bytes() const { return length * sizeof(Float); }
  };


  /**
     struct to define CPS ordered gauge fields:
     [parity][dim][volumecb][col][row]
  */
  template <typename Float, int length> struct CPSOrder : LegacyOrder<Float,length> {
    using Accessor = CPSOrder<Float, length>;
    using real = typename mapper<Float>::type;
    using complex = complex<real>;
    Float *gauge;
    const int volumeCB;
    const real anisotropy;
    const real anisotropy_inv;
    static constexpr int Nc = 3;
    const int geometry;
    CPSOrder(const GaugeField &u, Float *gauge_ = 0, Float **ghost_ = 0) :
      LegacyOrder<Float, length>(u, ghost_),
      gauge(gauge_ ? gauge_ : (Float *)u.Gauge_p()),
      volumeCB(u.VolumeCB()),
      anisotropy(u.Anisotropy()),
      anisotropy_inv(1.0 / anisotropy),
      geometry(u.Geometry())
    {
      if constexpr (length != 18) errorQuda("Gauge length %d not supported", length);
    }

    // we need to transpose and scale for CPS ordering
    __device__ __host__ inline void load(complex v[9], int x, int dir, int parity, Float = 1.0) const
    {
      auto in = &gauge[((parity * volumeCB + x) * geometry + dir) * length];
      complex v_[9];
      block_load<complex, length / 2>(v_, reinterpret_cast<complex *>(in));

      for (int i=0; i<Nc; i++) {
        for (int j = 0; j < Nc; j++) { v[i * Nc + j] = v_[j * Nc + i] * anisotropy_inv; }
      }
    }

    __device__ __host__ inline void save(const complex v[9], int x, int dir, int parity) const
    {
      auto out = &gauge[((parity * volumeCB + x) * geometry + dir) * length];
      complex v_[9];
      for (int i=0; i<Nc; i++) {
        for (int j = 0; j < Nc; j++) { v_[i * Nc + j] = v[j * Nc + i] * anisotropy; }
      }

      block_store<complex, length / 2>(reinterpret_cast<complex *>(out), v_);
    }

    /**
       @brief This accessor routine returns a gauge_wrapper to this object,
       allowing us to overload various operators for manipulating at
       the site level interms of matrix operations.
       @param[in] dir Which dimension are we requesting
       @param[in] x_cb Checkerboarded space-time index we are requesting
       @param[in] parity Parity we are requesting
       @return Instance of a gauge_wrapper that curries in access to
       this field at the above coordinates.
    */
    __device__ __host__ inline auto operator()(int dim, int x_cb, int parity) const
    {
      return gauge_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, x_cb, parity);
    }

    size_t Bytes() const { return Nc * Nc * 2 * sizeof(Float); }
  };

    /**
       @brief struct to define BQCD ordered gauge fields:

       Note the convention in BQCD is to store the gauge field
       variables in and extended fields with inline halos
       [mu][parity][volumecb+halos][col][row]
    */
    template <typename Float, int length> struct BQCDOrder : LegacyOrder<Float,length> {
      using Accessor = BQCDOrder<Float, length>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      Float *gauge;
      const int volumeCB;
      int exVolumeCB; // extended checkerboard volume
      static constexpr int Nc = 3;
      BQCDOrder(const GaugeField &u, Float *gauge_ = 0, Float **ghost_ = 0) :
        LegacyOrder<Float, length>(u, ghost_),
        gauge(gauge_ ? gauge_ : (Float *)u.Gauge_p()),
        volumeCB(u.VolumeCB())
      {
        if constexpr (length != 18) errorQuda("Gauge length %d not supported", length);
        // compute volumeCB + halo region
        exVolumeCB = u.X()[0]/2 + 2;
	for (int i=1; i<4; i++) exVolumeCB *= u.X()[i] + 2;
      }

      // we need to transpose for BQCD ordering
      __device__ __host__ inline void load(complex v[9], int x, int dir, int parity, real = 1.0) const
      {
        auto in = &gauge[((dir * 2 + parity) * exVolumeCB + x) * length];
        complex v_[9];
        block_load<complex, 9>(v_, reinterpret_cast<complex *>(in));

        for (int i = 0; i < Nc; i++) {
          for (int j = 0; j < Nc; j++) { v[i * Nc + j] = v_[j * Nc + i]; }
        }
      }

      __device__ __host__ inline void save(const complex v[9], int x, int dir, int parity) const
      {
        auto out = &gauge[((dir * 2 + parity) * exVolumeCB + x) * length];
        complex v_[9];
        for (int i = 0; i < Nc; i++) {
          for (int j = 0; j < Nc; j++) { v_[i * Nc + j] = v[j * Nc + i]; }
        }

        block_store<complex, 9>(reinterpret_cast<complex *>(out), v_);
      }

      /**
         @brief This accessor routine returns a gauge_wrapper to this object,
         allowing us to overload various operators for manipulating at
         the site level interms of matrix operations.
         @param[in] dir Which dimension are we requesting
         @param[in] x_cb Checkerboarded space-time index we are requesting
         @param[in] parity Parity we are requesting
         @return Instance of a gauge_wrapper that curries in access to
         this field at the above coordinates.
       */
      __device__ __host__ inline auto operator()(int dim, int x_cb, int parity) const
      {
        return gauge_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, x_cb, parity);
      }

      size_t Bytes() const { return Nc * Nc * 2 * sizeof(Float); }
    };

    /**
       @brief struct to define TIFR ordered gauge fields:
       [mu][parity][volumecb][col][row]
    */
    template <typename Float, int length> struct TIFROrder : LegacyOrder<Float,length> {
      using Accessor = TIFROrder<Float, length>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      Float *gauge;
      const int volumeCB;
      static constexpr int Nc = 3;
      const real scale;
      const real scale_inv;
      TIFROrder(const GaugeField &u, Float *gauge_ = 0, Float **ghost_ = 0) :
        LegacyOrder<Float, length>(u, ghost_),
        gauge(gauge_ ? gauge_ : (Float *)u.Gauge_p()),
        volumeCB(u.VolumeCB()),
        scale(u.Scale()),
        scale_inv(1.0 / scale)
      {
        if constexpr (length != 18) errorQuda("Gauge length %d not supported", length);
      }

      // we need to transpose for TIFR ordering
      __device__ __host__ inline void load(complex v[9], int x, int dir, int parity, real = 1.0) const
      {
        auto in = &gauge[((dir * 2 + parity) * volumeCB + x) * length];
        complex v_[9];
        block_load<complex, 9>(v_, reinterpret_cast<complex *>(in));

        for (int i = 0; i < Nc; i++) {
          for (int j = 0; j < Nc; j++) { v[i * Nc + j] = v_[j * Nc + i] * scale_inv; }
        }
      }

      __device__ __host__ inline void save(const complex v[9], int x, int dir, int parity) const
      {
        auto out = &gauge[((dir * 2 + parity) * volumeCB + x) * length];
        complex v_[9];
        for (int i = 0; i < Nc; i++) {
          for (int j = 0; j < Nc; j++) { v_[i * Nc + j] = v[j * Nc + i] * scale; }
        }

        block_store<complex, 9>(reinterpret_cast<complex *>(out), v_);
      }

      /**
         @brief This accessor routine returns a gauge_wrapper to this object,
         allowing us to overload various operators for manipulating at
         the site level interms of matrix operations.
         @param[in] dir Which dimension are we requesting
         @param[in] x_cb Checkerboarded space-time index we are requesting
         @param[in] parity Parity we are requesting
         @return Instance of a gauge_wrapper that curries in access to
         this field at the above coordinates.
       */
      __device__ __host__ inline auto operator()(int dim, int x_cb, int parity) const
      {
        return gauge_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, x_cb, parity);
      }

      size_t Bytes() const { return Nc * Nc * 2 * sizeof(Float); }
    };

    /**
       struct to define TIFR ordered gauge fields (with inlined z halo of depth two):
       [mu][parity][t][z+4][y][x/2][col][row]
    */
    template <typename Float, int length> struct TIFRPaddedOrder : LegacyOrder<Float,length> {
      using Accessor = TIFRPaddedOrder<Float, length>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      Float *gauge;
      const int volumeCB;
      int exVolumeCB;
      static constexpr int Nc = 3;
      const real scale;
      const real scale_inv;
      const int dim[4];
      const int exDim[4];
      TIFRPaddedOrder(const GaugeField &u, Float *gauge_ = 0, Float **ghost_ = 0) :
        LegacyOrder<Float, length>(u, ghost_),
        gauge(gauge_ ? gauge_ : (Float *)u.Gauge_p()),
        volumeCB(u.VolumeCB()),
        exVolumeCB(1),
        scale(u.Scale()),
        scale_inv(1.0 / scale),
        dim {u.X()[0], u.X()[1], u.X()[2], u.X()[3]},
        exDim {u.X()[0], u.X()[1], u.X()[2] + 4, u.X()[3]}
      {
        if constexpr (length != 18) errorQuda("Gauge length %d not supported", length);

        // exVolumeCB is the padded checkboard volume
        for (int i=0; i<4; i++) exVolumeCB *= exDim[i];
	exVolumeCB /= 2;
      }

      /**
	 @brief Compute the index into the padded field.  Assumes that
	 parity doesn't change from unpadded to padded.
       */
      __device__ __host__ inline int getPaddedIndex(int x_cb, int parity) const {
	// find coordinates
	int coord[4];
	getCoords(coord, x_cb, dim, parity);

	// get z-extended index
	coord[2] += 2; // offset for halo
	return linkIndex(coord, exDim);
      }

      // we need to transpose for TIFR ordering
      __device__ __host__ inline void load(complex v[9], int x, int dir, int parity, real = 1.0) const
      {
        int y = getPaddedIndex(x, parity);
        auto in = &gauge[((dir * 2 + parity) * exVolumeCB + y) * length];
        complex v_[9];
        block_load<complex, 9>(v_, reinterpret_cast<complex *>(in));

        for (int i = 0; i < Nc; i++) {
          for (int j = 0; j < Nc; j++) { v[i * Nc + j] = v_[j * Nc + i] * scale_inv; }
        }
      }

      __device__ __host__ inline void save(const complex v[9], int x, int dir, int parity) const
      {
        int y = getPaddedIndex(x, parity);
        auto out = &gauge[((dir * 2 + parity) * exVolumeCB + y) * length];

        complex v_[9];
        for (int i = 0; i < Nc; i++) {
          for (int j = 0; j < Nc; j++) { v_[i * Nc + j] = v[j * Nc + i] * scale; }
        }

        block_store<complex, 9>(reinterpret_cast<complex *>(out), v_);
      }

      /**
         @brief This accessor routine returns a gauge_wrapper to this object,
         allowing us to overload various operators for manipulating at
         the site level interms of matrix operations.
         @param[in] dir Which dimension are we requesting
         @param[in] x_cb Checkerboarded space-time index we are requesting
         @param[in] parity Parity we are requesting
         @return Instance of a gauge_wrapper that curries in access to
         this field at the above coordinates.
       */
      __device__ __host__ inline auto operator()(int dim, int x_cb, int parity) const
      {
        return gauge_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, x_cb, parity);
      }

      size_t Bytes() const { return Nc * Nc * 2 * sizeof(Float); }
    };

  } // namespace gauge

  template <typename real_out_t, typename store_out_t, typename real_in_t, typename store_in_t, bool block_float, typename norm_t>
  __device__ __host__ inline auto
  operator*(const gauge::fieldorder_wrapper<real_out_t, store_out_t> &a,
            const colorspinor::fieldorder_wrapper<real_in_t, store_in_t, block_float, norm_t> &b)
  {
    return complex<real_out_t>(a) * complex<real_in_t>(b);
  }

  // Use traits to reduce the template explosion
  template <typename T, QudaReconstructType, int N = 18, QudaStaggeredPhase stag = QUDA_STAGGERED_PHASE_NO,
            bool huge_alloc = gauge::default_huge_alloc, QudaGhostExchange ghostExchange = QUDA_GHOST_EXCHANGE_INVALID,
            bool use_inphase = false, QudaGaugeFieldOrder order = QUDA_NATIVE_GAUGE_ORDER>
  struct gauge_mapper {
  };

  // double precision
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<double, QUDA_RECONSTRUCT_NO, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<double, N, 2, N, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<double, QUDA_RECONSTRUCT_13, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<double, N, 2, 13, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<double, QUDA_RECONSTRUCT_12, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<double, N, 2, 12, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<double, QUDA_RECONSTRUCT_10, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<double, N, 2, 11, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<double, QUDA_RECONSTRUCT_9, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<double, N, 2, 9, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<double, QUDA_RECONSTRUCT_8, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<double, N, 2, 8, stag, huge_alloc, ghostExchange, use_inphase> type;
  };

  // single precision
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<float, QUDA_RECONSTRUCT_NO, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<float, N, 2, N, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<float, QUDA_RECONSTRUCT_13, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<float, N, 4, 13, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<float, QUDA_RECONSTRUCT_12, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<float, N, 4, 12, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<float, QUDA_RECONSTRUCT_10, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<float, N, 2, 11, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<float, QUDA_RECONSTRUCT_9, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<float, N, 4, 9, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<float, QUDA_RECONSTRUCT_8, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<float, N, 4, 8, stag, huge_alloc, ghostExchange, use_inphase> type;
  };

  // half precision
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<short, QUDA_RECONSTRUCT_NO, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<short, N, 2, N, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<short, QUDA_RECONSTRUCT_13, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<short, N, 4, 13, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<short, QUDA_RECONSTRUCT_12, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<short, N, 4, 12, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<short, QUDA_RECONSTRUCT_10, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<short, N, 2, 11, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<short, QUDA_RECONSTRUCT_9, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<short, N, QUDA_ORDER_FP, 9, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<short, QUDA_RECONSTRUCT_8, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<short, N, QUDA_ORDER_FP, 8, stag, huge_alloc, ghostExchange, use_inphase> type;
  };

  // quarter precision
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<int8_t, QUDA_RECONSTRUCT_NO, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<int8_t, N, 2, N, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<int8_t, QUDA_RECONSTRUCT_13, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<int8_t, N, 4, 13, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<int8_t, QUDA_RECONSTRUCT_12, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<int8_t, N, 4, 12, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<int8_t, QUDA_RECONSTRUCT_10, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<int8_t, N, 2, 11, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<int8_t, QUDA_RECONSTRUCT_9, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<int8_t, N, QUDA_ORDER_FP, 9, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<int8_t, QUDA_RECONSTRUCT_8, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<int8_t, N, QUDA_ORDER_FP, 8, stag, huge_alloc, ghostExchange, use_inphase> type;
  };

  template <typename T, QudaReconstructType recon, int N, QudaStaggeredPhase stag, bool huge_alloc,
            QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<T, recon, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_MILC_GAUGE_ORDER> {
    typedef gauge::MILCOrder<T, N> type;
  };

  template <typename T, QudaReconstructType recon, int N, QudaStaggeredPhase stag, bool huge_alloc,
            QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<T, recon, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_QDP_GAUGE_ORDER> {
    typedef gauge::QDPOrder<T, N> type;
  };

  template<typename T, QudaGaugeFieldOrder order, int Nc> struct gauge_order_mapper { };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_QDP_GAUGE_ORDER,Nc> { typedef gauge::QDPOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_QDPJIT_GAUGE_ORDER,Nc> { typedef gauge::QDPJITOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_MILC_GAUGE_ORDER,Nc> { typedef gauge::MILCOrder<T, 2*Nc*Nc> type; };
  template <typename T, int Nc> struct gauge_order_mapper<T, QUDA_CPS_WILSON_GAUGE_ORDER, Nc> {
    typedef gauge::CPSOrder<T, 2 * Nc * Nc> type;
  };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_BQCD_GAUGE_ORDER,Nc> { typedef gauge::BQCDOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_TIFR_GAUGE_ORDER,Nc> { typedef gauge::TIFROrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_TIFR_PADDED_GAUGE_ORDER,Nc> { typedef gauge::TIFRPaddedOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_FLOAT2_GAUGE_ORDER,Nc> { typedef gauge::FloatNOrder<T, 2*Nc*Nc, 2, 2*Nc*Nc> type; };

} // namespace quda
