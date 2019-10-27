#ifndef _GAUGE_ORDER_H
#define _GAUGE_ORDER_H

/**
 * @file  gauge_field_order.h
 * @brief Main header file for host and device accessors to GaugeFields
 *
 */

// trove requires the warp shuffle instructions introduced with Kepler
#if __COMPUTE_CAPABILITY__ >= 300
#include <trove/ptr.h>
#else
#define DISABLE_TROVE
#endif
#ifndef __CUDACC_RTC__
#include <assert.h>
#endif
#include <type_traits>

#include <register_traits.h>
#include <complex_quda.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <type_traits>
#include <limits>
#include <atomic.cuh>
#include <thrust_helper.cuh>
#include <gauge_field.h>
#include <index_helper.cuh>

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
          gauge(gauge),
          dim(dim),
          x_cb(x_cb),
          parity(parity),
          phase(phase)
      {
      }

      /**
	 @brief Assignment operator with Matrix instance as input
	 @param[in] M Matrix we want to store in this accessor
       */
      template<typename M>
      __device__ __host__ inline void operator=(const M &a) {
	gauge.save((Float*)a.data, x_cb, dim, parity);
      }
    };

  /**
     @brief Copy constructor for the Matrix class with a gauge_wrapper input.
     @param[in] a Input gauge_wrapper that we use to fill in this matrix instance
   */
  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline void Matrix<T,N>::operator=(const gauge_wrapper<typename RealType<T>::type,S> &a) {
    a.gauge.load((typename RealType<T>::type *)data, a.x_cb, a.dim, a.parity, a.phase);
  }

  /**
     @brief Assignment operator for the Matrix class with a gauge_wrapper input.
     @param[in] a Input gauge_wrapper that we use to fill in this matrix instance
   */
  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline Matrix<T,N>::Matrix(const gauge_wrapper<typename RealType<T>::type,S> &a) {
    a.gauge.load((typename RealType<T>::type *)data, a.x_cb, a.dim, a.parity, a.phase);
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
      __device__ __host__ inline gauge_ghost_wrapper<Float, T>(
          T &gauge, int dim, int ghost_idx, int parity, Float phase = 1.0) :
          gauge(gauge),
          dim(dim),
          ghost_idx(ghost_idx),
          parity(parity),
          phase(phase)
      {
      }

      /**
	 @brief Assignment operator with Matrix instance as input
	 @param[in] M Matrix we want to store in this accessot
       */
      template<typename M>
      __device__ __host__ inline void operator=(const M &a) {
	gauge.saveGhost((Float*)a.data, ghost_idx, dim, parity);
      }
    };

  /**
     @brief Copy constructor for the Matrix class with a gauge_ghost_wrapper input.
     @param[in] a Input gauge_wrapper that we use to fill in this matrix instance
   */
  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline void Matrix<T,N>::operator=(const gauge_ghost_wrapper<typename RealType<T>::type,S> &a) {
    a.gauge.loadGhost((typename RealType<T>::type *)data, a.ghost_idx, a.dim, a.parity, a.phase);
  }

  /**
     @brief Assignment operator for the Matrix class with a gauge_ghost_wrapper input.
     @param[in] a Input gauge_wrapper that we use to fill in this matrix instance
   */
  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline Matrix<T,N>::Matrix(const gauge_ghost_wrapper<typename RealType<T>::type,S> &a) {
    a.gauge.loadGhost((typename RealType<T>::type *)data, a.ghost_idx, a.dim, a.parity, a.phase);
  }

  namespace gauge {

    template<typename ReduceType, typename Float> struct square_ {
      square_(ReduceType scale) { }
      __host__ __device__ inline ReduceType operator()(const quda::complex<Float> &x)
      { return static_cast<ReduceType>(norm(x)); }
    };

    template<typename ReduceType> struct square_<ReduceType,char> {
      const ReduceType scale;
      square_(const ReduceType scale) : scale(scale) { }
      __host__ __device__ inline ReduceType operator()(const quda::complex<char> &x)
      { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
    };

    template<typename ReduceType> struct square_<ReduceType,short> {
      const ReduceType scale;
      square_(const ReduceType scale) : scale(scale) { }
      __host__ __device__ inline ReduceType operator()(const quda::complex<short> &x)
      { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
    };

    template<typename ReduceType> struct square_<ReduceType,int> {
      const ReduceType scale;
      square_(const ReduceType scale) : scale(scale) { }
      __host__ __device__ inline ReduceType operator()(const quda::complex<int> &x)
      { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
    };

    template<typename Float, typename storeFloat> struct abs_ {
      abs_(const Float scale) { }
      __host__ __device__ Float operator()(const quda::complex<storeFloat> &x) { return abs(x); }
    };

    template<typename Float> struct abs_<Float,char> {
      Float scale;
      abs_(const Float scale) : scale(scale) { }
      __host__ __device__ Float operator()(const quda::complex<char> &x)
      { return abs(scale * complex<Float>(x.real(), x.imag())); }
    };

    template<typename Float> struct abs_<Float,short> {
      Float scale;
      abs_(const Float scale) : scale(scale) { }
      __host__ __device__ Float operator()(const quda::complex<short> &x)
      { return abs(scale * complex<Float>(x.real(), x.imag())); }
    };

    template<typename Float> struct abs_<Float,int> {
      Float scale;
      abs_(const Float scale) : scale(scale) { }
      __host__ __device__ Float operator()(const quda::complex<int> &x)
      { return abs(scale * complex<Float>(x.real(), x.imag())); }
    };

    template <typename Float, typename storeFloat> __host__ __device__ inline constexpr bool fixed_point() { return false; }
    template<> __host__ __device__ inline constexpr bool fixed_point<float,char>() { return true; }
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
	complex<storeFloat> *v;
	const int idx;
	const Float scale;
	const Float scale_inv;
	static constexpr bool fixed = fixed_point<Float,storeFloat>();

	/**
	   @brief fieldorder_wrapper constructor
	   @param idx Field index
	*/
        __device__ __host__ inline fieldorder_wrapper(complex<storeFloat> *v, int idx, Float scale, Float scale_inv)
	  : v(v), idx(idx), scale(scale), scale_inv(scale_inv) {}

	__device__ __host__ inline Float real() const {
          if (!fixed) {
            return v[idx].real();
          } else {
            return scale_inv*static_cast<Float>(v[idx].real());
          }
        }

	__device__ __host__ inline Float imag() const {
          if (!fixed) {
            return v[idx].imag();
          } else {
            return scale_inv*static_cast<Float>(v[idx].imag());
          }
        }

	/**
	   @brief negation operator
           @return negation of this complex number
	*/
	__device__ __host__ inline complex<Float> operator-() const {
	  return fixed ? -scale_inv*static_cast<complex<Float> >(v[idx]) : -static_cast<complex<Float> >(v[idx]);
	}

	/**
	   @brief Assignment operator with fieldorder_wrapper instance as input
	   @param a fieldorder_wrapper we are copying from
	*/
	__device__ __host__ inline void operator=(const fieldorder_wrapper<Float,storeFloat> &a) {
	  v[idx] = fixed ? complex<storeFloat>(round(scale * a.real()), round(scale * a.imag())) : a.v[a.idx];
	}

	/**
	   @brief Assignment operator with complex number instance as input
	   @param a Complex number we want to store in this accessor
	*/
        template<typename theirFloat>
	__device__ __host__ inline void operator=(const complex<theirFloat> &a) {
	  if (match<storeFloat,theirFloat>()) {
	    v[idx] = complex<storeFloat>(a.x, a.y);
	  } else {
	    v[idx] = fixed ? complex<storeFloat>(round(scale * a.x), round(scale * a.y)) : complex<storeFloat>(a.x, a.y);
	  }
	}

	/**
	   @brief Operator+= with complex number instance as input
	   @param a Complex number we want to add to this accessor
	*/
        template<typename theirFloat>
	__device__ __host__ inline void operator+=(const complex<theirFloat> &a) {
	  if (match<storeFloat,theirFloat>()) {
	    v[idx] += complex<storeFloat>(a.x, a.y);
	  } else {
	    v[idx] += fixed ? complex<storeFloat>(round(scale * a.x), round(scale * a.y)) : complex<storeFloat>(a.x, a.y);
	  }
	}

	/**
	   @brief Operator-= with complex number instance as input
	   @param a Complex number we want to subtract from this accessor
	*/
	template<typename theirFloat>
	__device__ __host__ inline void operator-=(const complex<theirFloat> &a) {
	  if (match<storeFloat,theirFloat>()) {
	    v[idx] -= complex<storeFloat>(a.x, a.y);
	  } else {
	    v[idx] -= fixed ? complex<storeFloat>(round(scale * a.x), round(scale * a.y)) : complex<storeFloat>(a.x, a.y);
	  }
	}

      };

    template<typename Float, typename storeFloat>
    __device__ __host__ inline complex<Float> operator*(const Float &a, const fieldorder_wrapper<Float,storeFloat> &b)
    {
      if (fixed_point<Float,storeFloat>()) return a*complex<Float>(b.real(), b.imag());
      else return a*complex<Float>(b.v[b.idx].real(),b.v[b.idx].imag());
    }

    template<typename Float, typename storeFloat>
    __device__ __host__ inline complex<Float> operator+(const fieldorder_wrapper<Float,storeFloat> &a, const complex<Float> &b) {
      if (fixed_point<Float,storeFloat>()) return complex<Float>(a.real(), a.imag()) + b;
      else return complex<Float>(a.v[a.idx].real(),a.v[a.idx].imag()) + b;
    }

    template<typename Float, typename storeFloat>
    __device__ __host__ inline complex<Float> operator+(const complex<Float> &a, const fieldorder_wrapper<Float,storeFloat> &b) {
      if (fixed_point<Float,storeFloat>()) return a + complex<Float>(b.real(), b.imag());
      else return a + complex<Float>(b.v[b.idx].real(),b.v[b.idx].imag());;
    }

    template<typename Float, int nColor, QudaGaugeFieldOrder order, typename storeFloat, bool use_tex>
    struct Accessor {
      mutable complex<Float> dummy;
      Accessor(const GaugeField &, void *gauge_=0, void **ghost_=0) {
	errorQuda("Not implemented for order=%d", order);
      }

      void resetScale(Float dummy) { }

      __device__ __host__ complex<Float>& operator()(int d, int parity, int x, int row, int col) const {
	return dummy;
      }
    };

    template<typename Float, int nColor, QudaGaugeFieldOrder order, bool native_ghost, typename storeFloat, bool use_tex>
    struct GhostAccessor {
      mutable complex<Float> dummy;
      GhostAccessor(const GaugeField &, void *gauge_=0, void **ghost_=0) {
	errorQuda("Not implemented for order=%d", order);
      }

      void resetScale(Float dummy) { }

      __device__ __host__ complex<Float>& operator()(int d, int parity, int x, int row, int col) const {
	return dummy;
      }
    };

    template<typename Float, int nColor, typename storeFloat, bool use_tex>
      struct Accessor<Float,nColor,QUDA_QDP_GAUGE_ORDER,storeFloat,use_tex> {
      complex <storeFloat> *u[QUDA_MAX_GEOMETRY];
      const int volumeCB;
      const int geometry;
      const int cb_offset;
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();

      Accessor(const GaugeField &U, void *gauge_=0, void **ghost_=0)
	: volumeCB(U.VolumeCB()), geometry(U.Geometry()), cb_offset((U.Bytes()>>1) / (sizeof(complex<storeFloat>)*U.Geometry())),
	scale(static_cast<Float>(1.0)), scale_inv(static_cast<Float>(1.0))
      {
	for (int d=0; d<U.Geometry(); d++)
	  u[d] = gauge_ ? static_cast<complex<storeFloat>**>(gauge_)[d] :
	    static_cast<complex<storeFloat>**>(const_cast<void*>(U.Gauge_p()))[d];
	resetScale(U.Scale());
      }

    Accessor(const Accessor<Float,nColor,QUDA_QDP_GAUGE_ORDER,storeFloat,use_tex> &a)
      : volumeCB(a.volumeCB), geometry(a.geometry), cb_offset(a.cb_offset), scale(a.scale), scale_inv(a.scale_inv) {
	for (int d=0; d<QUDA_MAX_GEOMETRY; d++)
	  u[d] = a.u[d];
      }

      void resetScale(Float max) {
	if (fixed) {
	  scale = static_cast<Float>(std::numeric_limits<storeFloat>::max() / max);
	  scale_inv = static_cast<Float>(max / std::numeric_limits<storeFloat>::max());
	}
      }

      __device__ __host__ inline complex<Float> operator()(int d, int parity, int x, int row, int col) const
      {
	complex<storeFloat> tmp = u[d][ parity*cb_offset + (x*nColor + row)*nColor + col];

	if (fixed) {
	  return scale_inv*complex<Float>(static_cast<Float>(tmp.x), static_cast<Float>(tmp.y));
	} else {
	  return complex<Float>(tmp.x,tmp.y);
	}
      }

      __device__ __host__ inline fieldorder_wrapper<Float,storeFloat> operator()(int d, int parity, int x, int row, int col)
	{ return fieldorder_wrapper<Float,storeFloat>(u[d], parity*cb_offset + (x*nColor + row)*nColor + col,
						      scale, scale_inv); }

      template<typename theirFloat>
      __device__ __host__ inline void atomic_add(int dim, int parity, int x_cb, int row, int col,
                                                 const complex<theirFloat> &val) const {
#ifdef __CUDA_ARCH__
	typedef typename vector<storeFloat,2>::type vec2;
	vec2 *u2 = reinterpret_cast<vec2*>(u[dim] + parity*cb_offset + (x_cb*nColor + row)*nColor + col);
	if (fixed && !match<storeFloat,theirFloat>()) {
	  complex<storeFloat> val_(round(scale * val.real()), round(scale * val.imag()));
	  atomicAdd(u2, (vec2&)val_);
	} else {
	  atomicAdd(u2, (vec2&)val);
	}
#else
	if (fixed && !match<storeFloat,theirFloat>()) {
	  complex<storeFloat> val_(round(scale * val.real()), round(scale * val.imag()));
#pragma omp atomic update
	  u[dim][ parity*cb_offset + (x_cb*nColor + row)*nColor + col].x += val_.x;
#pragma omp atomic update
	  u[dim][ parity*cb_offset + (x_cb*nColor + row)*nColor + col].y += val_.y;
	} else {
#pragma omp atomic update
	  u[dim][ parity*cb_offset + (x_cb*nColor + row)*nColor + col].x += static_cast<storeFloat>(val.x);
#pragma omp atomic update
	  u[dim][ parity*cb_offset + (x_cb*nColor + row)*nColor + col].y += static_cast<storeFloat>(val.y);
	}
#endif
      }

      template<typename helper, typename reducer>
        __host__ double transform_reduce(QudaFieldLocation location, int dim, helper h, reducer r, double init) const {
	if (dim >= geometry) errorQuda("Request dimension %d exceeds dimensionality of the field %d", dim, geometry);
        int lower = (dim == -1) ? 0 : dim;
        int upper = (dim == -1) ? geometry : dim+1;
        double result = init;
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          thrust_allocator alloc;
          for (int d=lower; d<upper; d++) {
            thrust::device_ptr<complex<storeFloat> > ptr(u[d]);
            result = thrust::transform_reduce(thrust::cuda::par(alloc), ptr, ptr+2*volumeCB*nColor*nColor, h, result, r);
          }
        } else {
          for (int d=lower; d<upper; d++) {
            result = thrust::transform_reduce(thrust::seq, u[d], u[d]+2*volumeCB*nColor*nColor, h, result, r);
          }
        }
        return result;
      }

    };

    template<typename Float, int nColor, bool native_ghost, typename storeFloat, bool use_tex>
      struct GhostAccessor<Float,nColor,QUDA_QDP_GAUGE_ORDER,native_ghost,storeFloat,use_tex> {
      complex<storeFloat> *ghost[8];
      int ghostOffset[8];
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();

      GhostAccessor(const GaugeField &U, void *gauge_=0, void **ghost_=0)
	: scale(static_cast<Float>(1.0)), scale_inv(static_cast<Float>(1.0)) {
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

    GhostAccessor(const GhostAccessor<Float,nColor,QUDA_QDP_GAUGE_ORDER,native_ghost,storeFloat,use_tex> &a)
	: scale(a.scale), scale_inv(a.scale_inv) {
	for (int d=0; d<8; d++) {
	  ghost[d] = a.ghost[d];
	  ghostOffset[d] = a.ghostOffset[d];
	}
      }

      void resetScale(Float max) {
	if (fixed) {
	  scale = static_cast<Float>(std::numeric_limits<storeFloat>::max() / max);
	  scale_inv = static_cast<Float>(max / std::numeric_limits<storeFloat>::max());
	}
      }

      __device__ __host__ inline complex<Float> operator()(int d, int parity, int x, int row, int col) const
      {
	complex<storeFloat> tmp = ghost[d][ parity*ghostOffset[d] + (x*nColor + row)*nColor + col];
	if (fixed) {
	  return scale_inv*complex<Float>(static_cast<Float>(tmp.x), static_cast<Float>(tmp.y));
	} else {
	  return complex<Float>(tmp.x,tmp.y);
	}
      }

      __device__ __host__ inline fieldorder_wrapper<Float,storeFloat> operator()(int d, int parity, int x, int row, int col)
	{ return fieldorder_wrapper<Float,storeFloat>(ghost[d], parity*ghostOffset[d] + (x*nColor + row)*nColor + col,
						      scale, scale_inv); }
    };

    template<typename Float, int nColor, typename storeFloat, bool use_tex>
      struct Accessor<Float,nColor,QUDA_MILC_GAUGE_ORDER,storeFloat,use_tex> {
      complex<storeFloat> *u;
      const int volumeCB;
      const int geometry;
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();

      Accessor(const GaugeField &U, void *gauge_=0, void **ghost_=0)
      : u(gauge_ ? static_cast<complex<storeFloat>*>(gauge_) :
	  static_cast<complex<storeFloat>*>(const_cast<void *>(U.Gauge_p()))),
	volumeCB(U.VolumeCB()), geometry(U.Geometry()),
	scale(static_cast<Float>(1.0)), scale_inv(static_cast<Float>(1.0)) {
	resetScale(U.Scale());
      }

    Accessor(const Accessor<Float,nColor,QUDA_MILC_GAUGE_ORDER,storeFloat,use_tex> &a)
	: u(a.u), volumeCB(a.volumeCB), geometry(a.geometry), scale(a.scale), scale_inv(a.scale_inv)
      { }

      void resetScale(Float max) {
	if (fixed) {
	  scale = static_cast<Float>(std::numeric_limits<storeFloat>::max() / max);
	  scale_inv = static_cast<Float>(max / std::numeric_limits<storeFloat>::max());
	}
      }

      __device__ __host__ inline complex<Float> operator()(int d, int parity, int x, int row, int col) const
      {
	complex<storeFloat> tmp = u[(((parity*volumeCB+x)*geometry + d)*nColor + row)*nColor + col];
	if (fixed) {
	  return scale_inv*complex<Float>(static_cast<Float>(tmp.x), static_cast<Float>(tmp.y));
	} else {
	  return complex<Float>(tmp.x,tmp.y);
	}
      }

      __device__ __host__ inline fieldorder_wrapper<Float,storeFloat> operator()(int d, int parity, int x, int row, int col)
	{ return fieldorder_wrapper<Float,storeFloat>
	    (u, (((parity*volumeCB+x)*geometry + d)*nColor + row)*nColor + col, scale, scale_inv); }

      template <typename theirFloat>
      __device__ __host__ inline void atomic_add(int dim, int parity, int x_cb, int row, int col, const complex<theirFloat> &val) const {
#ifdef __CUDA_ARCH__
	typedef typename vector<storeFloat,2>::type vec2;
	vec2 *u2 = reinterpret_cast<vec2*>(u + (((parity*volumeCB+x_cb)*geometry + dim)*nColor + row)*nColor + col);
	if (fixed && !match<storeFloat,theirFloat>()) {
	  complex<storeFloat> val_(round(scale * val.real()), round(scale * val.imag()));
	  atomicAdd(u2, (vec2&)val_);
	} else {
	  atomicAdd(u2, (vec2&)val);
	}
#else
	if (fixed && !match<storeFloat,theirFloat>()) {
	  complex<storeFloat> val_(round(scale * val.real()), round(scale * val.imag()));
#pragma omp atomic update
	  u[(((parity*volumeCB+x_cb)*geometry + dim)*nColor + row)*nColor + col].x += val_.x;
#pragma omp atomic update
	  u[(((parity*volumeCB+x_cb)*geometry + dim)*nColor + row)*nColor + col].y += val_.y;
	} else {
#pragma omp atomic update
	  u[(((parity*volumeCB+x_cb)*geometry + dim)*nColor + row)*nColor + col].x += static_cast<storeFloat>(val.x);
#pragma omp atomic update
	  u[(((parity*volumeCB+x_cb)*geometry + dim)*nColor + row)*nColor + col].y += static_cast<storeFloat>(val.y);
	}
#endif
      }

      template<typename helper, typename reducer>
      __host__ double transform_reduce(QudaFieldLocation location, int dim, helper h, reducer r, double init) const {
	if (dim >= geometry) errorQuda("Request dimension %d exceeds dimensionality of the field %d", dim, geometry);
        int lower = (dim == -1) ? 0 : dim;
        int upper = (dim == -1) ? geometry : dim+1;
        double result = init;
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          thrust_allocator alloc;
          thrust::device_ptr<complex<storeFloat> > ptr(u);
          result = thrust::transform_reduce(thrust::cuda::par(alloc),
                                            ptr+(0*geometry+lower)*volumeCB*nColor*nColor,
                                            ptr+(0*geometry+upper)*volumeCB*nColor*nColor, h, result, r);
          result = thrust::transform_reduce(thrust::cuda::par(alloc),
                                            ptr+(1*geometry+lower)*volumeCB*nColor*nColor,
                                            ptr+(1*geometry+upper)*volumeCB*nColor*nColor, h, result, r);
        } else {
          result = thrust::transform_reduce(thrust::seq,
                                            u+(0*geometry+lower)*volumeCB*nColor*nColor,
                                            u+(0*geometry+upper)*volumeCB*nColor*nColor, h, result, r);
          result  = thrust::transform_reduce(thrust::seq,
                                             u+(1*geometry+lower)*volumeCB*nColor*nColor,
                                             u+(1*geometry+upper)*volumeCB*nColor*nColor, h, result, r);
        }
        return result;
      }

    };

    template<typename Float, int nColor, bool native_ghost, typename storeFloat, bool use_tex>
      struct GhostAccessor<Float,nColor,QUDA_MILC_GAUGE_ORDER,native_ghost,storeFloat,use_tex> {
      complex<storeFloat> *ghost[8];
      int ghostOffset[8];
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();

      GhostAccessor(const GaugeField &U, void *gauge_=0, void **ghost_=0)
	: scale(static_cast<Float>(1.0)), scale_inv(static_cast<Float>(1.0)) {
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

    GhostAccessor(const GhostAccessor<Float,nColor,QUDA_MILC_GAUGE_ORDER,native_ghost,storeFloat,use_tex> &a)
	: scale(a.scale), scale_inv(a.scale_inv) {
	for (int d=0; d<8; d++) {
	  ghost[d] = a.ghost[d];
	  ghostOffset[d] = a.ghostOffset[d];
	}
      }

      void resetScale(Float max) {
	if (fixed) {
	  scale = static_cast<Float>(std::numeric_limits<storeFloat>::max() / max);
	  scale_inv = static_cast<Float>(max / std::numeric_limits<storeFloat>::max());
	}
      }

      __device__ __host__ inline complex<Float> operator()(int d, int parity, int x, int row, int col) const
      {
	complex<storeFloat> tmp = ghost[d][ parity*ghostOffset[d] + (x*nColor + row)*nColor + col];
	if (fixed) {
	  return scale_inv*complex<Float>(static_cast<Float>(tmp.x), static_cast<Float>(tmp.y));
	} else {
	  return complex<Float>(tmp.x,tmp.y);
	}
      }

      __device__ __host__ inline fieldorder_wrapper<Float,storeFloat> operator()(int d, int parity, int x, int row, int col)
	{ return fieldorder_wrapper<Float,storeFloat>
	    (ghost[d], parity*ghostOffset[d] + (x*nColor + row)*nColor + col, scale, scale_inv); }
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

    template<typename Float, int nColor, typename storeFloat, bool use_tex>
      struct Accessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER, storeFloat, use_tex> {
      complex<storeFloat> *u;
      const int offset_cb;
#ifdef USE_TEXTURE_OBJECTS
      typedef typename TexVectorType<Float,2>::type TexVector;
      cudaTextureObject_t tex;
#endif
      const int volumeCB;
      const int stride;
      const int geometry;
      Float max;
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();

    Accessor(const GaugeField &U, void *gauge_=0, void **ghost_=0, bool override=false)
      : u(gauge_ ? static_cast<complex<storeFloat>*>(gauge_) :
	  static_cast<complex<storeFloat>*>(const_cast<void*>(U.Gauge_p()))),
	offset_cb( (U.Bytes()>>1) / sizeof(complex<storeFloat>)),
#ifdef USE_TEXTURE_OBJECTS
        tex(0),
#endif
        volumeCB(U.VolumeCB()), stride(U.Stride()), geometry(U.Geometry()),
        max(static_cast<Float>(1.0)), scale(static_cast<Float>(1.0)), scale_inv(static_cast<Float>(1.0))
      {
	resetScale(U.Scale());
#ifdef USE_TEXTURE_OBJECTS
	if (U.Location() == QUDA_CUDA_FIELD_LOCATION) tex = static_cast<const cudaGaugeField&>(U).Tex();
	if (use_tex && this->u != U.Gauge_p() && !override) {
	  errorQuda("Cannot use texture read since data pointer does not equal field pointer - use with use_tex=false instead");
	}
#endif
      }

    Accessor(const Accessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER,storeFloat,use_tex> &a)
      : u(a.u), offset_cb(a.offset_cb),
#ifdef USE_TEXTURE_OBJECTS
        tex(a.tex),
#endif
        volumeCB(a.volumeCB), stride(a.stride), geometry(a.geometry),
	scale(a.scale), scale_inv(a.scale_inv) {  }

      void resetScale(Float max_) {
	if (fixed) {
	  max = max_;
	  scale = static_cast<Float>(std::numeric_limits<storeFloat>::max() / max);
	  scale_inv = static_cast<Float>(max / std::numeric_limits<storeFloat>::max());
	}
      }

      __device__ __host__ inline const complex<Float> operator()(int dim, int parity, int x_cb, int row, int col) const
      {
#if defined(USE_TEXTURE_OBJECTS) && defined(__CUDA_ARCH__)
	if (use_tex) {
	  TexVector vecTmp = tex1Dfetch<TexVector>(tex, parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb);
	  if (fixed) {
	    return max*complex<Float>(vecTmp.x, vecTmp.y);
	  } else {
	    return complex<Float>(vecTmp.x, vecTmp.y);
	  }
	} else
#endif
	{
	  complex<storeFloat> tmp = u[parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb];
	  if (fixed) {
	    return scale_inv*complex<Float>(static_cast<Float>(tmp.x), static_cast<Float>(tmp.y));
	  } else {
	    return complex<Float>(tmp.x, tmp.y);
	  }
	}
      }

      __device__ __host__ inline fieldorder_wrapper<Float,storeFloat> operator()(int dim, int parity, int x_cb, int row, int col)
      {
	int index = parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb;
	return fieldorder_wrapper<Float,storeFloat>(u, index, scale, scale_inv);
      }

      template <typename theirFloat>
      __device__ __host__ void atomic_add(int dim, int parity, int x_cb, int row, int col, const complex<theirFloat> &val) const {
#ifdef __CUDA_ARCH__
	typedef typename vector<storeFloat,2>::type vec2;
	vec2 *u2 = reinterpret_cast<vec2*>(u + parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb);
	if (fixed && !match<storeFloat,theirFloat>()) {
	  complex<storeFloat> val_(round(scale * val.real()), round(scale * val.imag()));
	  atomicAdd(u2, (vec2&)val_);
	} else {
	  atomicAdd(u2, (vec2&)val);
	}
#else
        if (fixed && !match<storeFloat,theirFloat>()) {
	  complex<storeFloat> val_(round(scale * val.real()), round(scale * val.imag()));
#pragma omp atomic update
	  u[parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb].x += val_.x;
#pragma omp atomic update
	  u[parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb].y += val_.y;
	  } else {
#pragma omp atomic update
	  u[parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb].x += static_cast<storeFloat>(val.x);
#pragma omp atomic update
	  u[parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb].y += static_cast<storeFloat>(val.y);
	}
#endif
      }

      template<typename helper, typename reducer>
        __host__ double transform_reduce(QudaFieldLocation location, int dim, helper h, reducer r, double init) const {
	if (dim >= geometry) errorQuda("Request dimension %d exceeds dimensionality of the field %d", dim, geometry);
        int lower = (dim == -1) ? 0 : dim;
        int upper = (dim == -1) ? geometry : dim+1;
        double result = init;
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          thrust_allocator alloc;
          thrust::device_ptr<complex<storeFloat> > ptr(u);
          result = thrust::transform_reduce(thrust::cuda::par(alloc),
                                            ptr+0*offset_cb+lower*stride*nColor*nColor,
                                            ptr+0*offset_cb+upper*stride*nColor*nColor, h, result, r);
          result = thrust::transform_reduce(thrust::cuda::par(alloc),
                                            ptr+1*offset_cb+lower*stride*nColor*nColor,
                                            ptr+1*offset_cb+upper*stride*nColor*nColor, h, result, r);
        } else {
          result = thrust::transform_reduce(thrust::seq,
                                            u+0*offset_cb+lower*stride*nColor*nColor,
                                            u+0*offset_cb+upper*stride*nColor*nColor, h, result, r);
          result = thrust::transform_reduce(thrust::seq,
                                            u+1*offset_cb+lower*stride*nColor*nColor,
                                            u+1*offset_cb+upper*stride*nColor*nColor, h, result, r);
        }
        return result;
      }

    };

    template<typename Float, int nColor, bool native_ghost, typename storeFloat, bool use_tex>
      struct GhostAccessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER,native_ghost,storeFloat,use_tex> {
      complex<storeFloat> *ghost[8];
      const int volumeCB;
      int ghostVolumeCB[8];
      Float scale;
      Float scale_inv;
      static constexpr bool fixed = fixed_point<Float,storeFloat>();
      Accessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER,storeFloat,use_tex> accessor;

      GhostAccessor(const GaugeField &U, void *gauge_, void **ghost_=0)
	: volumeCB(U.VolumeCB()), accessor(U, gauge_, ghost_),
	  scale(static_cast<Float>(1.0)), scale_inv(static_cast<Float>(1.0))
      {
	if (!native_ghost) assert(ghost_ != nullptr);
	for (int d=0; d<4; d++) {
	  ghost[d] = !native_ghost ? static_cast<complex<storeFloat>*>(ghost_[d]) : nullptr;
	  ghostVolumeCB[d] = U.Nface()*U.SurfaceCB(d);
	  ghost[d+4] = !native_ghost && U.Geometry() == QUDA_COARSE_GEOMETRY? static_cast<complex<storeFloat>*>(ghost_[d+4]) : nullptr;
	  ghostVolumeCB[d+4] = U.Nface()*U.SurfaceCB(d);
	}
	resetScale(U.Scale());
      }

    GhostAccessor(const GhostAccessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER,native_ghost,storeFloat,use_tex> &a)
	: volumeCB(a.volumeCB), scale(a.scale), scale_inv(a.scale_inv), accessor(a.accessor)
      {
	for (int d=0; d<8; d++) {
	  ghost[d] = a.ghost[d];
	  ghostVolumeCB[d] = a.ghostVolumeCB[d];
	}
      }

      void resetScale(Float max) {
	accessor.resetScale(max);
	if (fixed) {
	  scale = static_cast<Float>(std::numeric_limits<storeFloat>::max() / max);
	  scale_inv = static_cast<Float>(max / std::numeric_limits<storeFloat>::max());
	}
      }

      __device__ __host__ inline const complex<Float> operator()(int d, int parity, int x_cb, int row, int col) const
      {
	if (native_ghost) {
	  return accessor(d%4, parity, x_cb+(d/4)*ghostVolumeCB[d]+volumeCB, row, col);
	} else {
	  complex<storeFloat> tmp = ghost[d][ ((parity*nColor + row)*nColor+col)*ghostVolumeCB[d] + x_cb ];
	  if (fixed) {
	    return scale_inv*complex<Float>(static_cast<Float>(tmp.x), static_cast<Float>(tmp.y));
	  } else {
	    return complex<Float>(tmp.x, tmp.y);
	  }
	}
      }

      __device__ __host__ inline fieldorder_wrapper<Float,storeFloat> operator()(int d, int parity, int x_cb, int row, int col)
      {
	if (native_ghost)
	  return accessor(d%4, parity, x_cb+(d/4)*ghostVolumeCB[d]+volumeCB, row, col);
	else
	  return fieldorder_wrapper<Float,storeFloat>
	    (ghost[d], ((parity*nColor + row)*nColor+col)*ghostVolumeCB[d] + x_cb, scale, scale_inv);
      }
    };


    /**
       This is a template driven generic gauge field accessor.  To
       deploy for a specifc field ordering, the two operator()
       accessors have to be specialized for that ordering.

       @tparam Float Underlying type returned by the accessors
       @tparam nColor Number of colors for the field
       @tparam nSpinCoarse Number of "spin degrees of freedom" (for coarse-link fields only)
       @tparam order Storage order of the field
       @tparam native_ghost Whether to use native ghosts (inlined into
       the padded area for internal-order fields or use a separate array if false)
     */
  template <typename Float, int nColor, int nSpinCoarse, QudaGaugeFieldOrder order,
    bool native_ghost=true, typename storeFloat=Float, bool use_tex=false>
      struct FieldOrder {

	/** An internal reference to the actual field we are accessing */
	const int volumeCB;
	const int nDim;
	const int_fastdiv geometry;
	const QudaFieldLocation location;
	static constexpr int nColorCoarse = nColor / nSpinCoarse;

	Accessor<Float,nColor,order,storeFloat,use_tex> accessor;
	GhostAccessor<Float,nColor,order,native_ghost,storeFloat,use_tex> ghostAccessor;

	/**
	 * Constructor for the FieldOrder class
	 * @param field The field that we are accessing
	 */
      FieldOrder(GaugeField &U, void *gauge_=0, void **ghost_=0)
      : volumeCB(U.VolumeCB()), nDim(U.Ndim()), geometry(U.Geometry()),
	  location(U.Location()),
	  accessor(U, gauge_, ghost_), ghostAccessor(U, gauge_, ghost_)
	{
	  if (U.Reconstruct() != QUDA_RECONSTRUCT_NO)
	    errorQuda("GaugeField ordering not supported with reconstruction");
	}

      FieldOrder(const FieldOrder &o) : volumeCB(o.volumeCB),
	  nDim(o.nDim), geometry(o.geometry), location(o.location),
	  accessor(o.accessor), ghostAccessor(o.ghostAccessor)
	{ }

	virtual ~FieldOrder() { ; }

	void resetScale(double max) {
	  accessor.resetScale(max);
	  ghostAccessor.resetScale(max);
	}

	static constexpr bool fixedPoint() { return fixed_point<Float,storeFloat>(); }

	/**
	 * Read-only complex-member accessor function
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ complex<Float> operator()(int d, int parity, int x, int row, int col) const
	{ return accessor(d,parity,x,row,col); }

	/**
	 * Writable complex-member accessor function
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ fieldorder_wrapper<Float,storeFloat> operator() (int d, int parity, int x, int row, int col)
	{ return accessor(d,parity,x,row,col); }

	/**
	 * Read-only complex-member accessor function for the ghost zone
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ complex<Float> Ghost(int d, int parity, int x, int row, int col) const
	{ return ghostAccessor(d,parity,x,row,col); }

	/**
	 * Writable complex-member accessor function for the ghost zone
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ fieldorder_wrapper<Float,storeFloat> Ghost(int d, int parity, int x, int row, int col)
	{ return ghostAccessor(d,parity,x,row,col); }

    	/**
	 * Specialized read-only complex-member accessor function (for coarse gauge field)
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param s_row row spin index
	 * @param c_row row color index
	 * @param s_col col spin index
	 * @param c_col col color index
	 */
	__device__ __host__ inline const complex<Float> operator()(int d, int parity, int x, int s_row,
								   int s_col, int c_row, int c_col) const {
	  return (*this)(d, parity, x, s_row*nColorCoarse + c_row, s_col*nColorCoarse + c_col);
	}

	/**
	 * Specialized read-only complex-member accessor function (for coarse gauge field)
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param s_row row spin index
	 * @param c_row row color index
	 * @param s_col col spin index
	 * @param c_col col color index
	 */
	__device__ __host__ inline fieldorder_wrapper<Float,storeFloat> operator()
	  (int d, int parity, int x, int s_row, int s_col, int c_row, int c_col) {
	  return (*this)(d, parity, x, s_row*nColorCoarse + c_row, s_col*nColorCoarse + c_col);
	}

    	/**
	 * Specialized read-only complex-member accessor function (for coarse gauge field ghost zone)
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param s_row row spin index
	 * @param c_row row color index
	 * @param s_col col spin index
	 * @param c_col col color index
	 */
	__device__ __host__ inline complex<Float> Ghost(int d, int parity, int x, int s_row,
							int s_col, int c_row, int c_col) const {
	  return Ghost(d, parity, x, s_row*nColorCoarse + c_row, s_col*nColorCoarse + c_col);
	}

	/**
	 * Specialized read-only complex-member accessor function (for coarse gauge field ghost zone)
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param s_row row spin index
	 * @param c_row row color index
	 * @param s_col col spin index
	 * @param c_col col color index
	 */
	__device__ __host__ inline fieldorder_wrapper<Float,storeFloat>
	  Ghost(int d, int parity, int x, int s_row, int s_col, int c_row, int c_col) {
	  return Ghost(d, parity, x, s_row*nColorCoarse + c_row, s_col*nColorCoarse + c_col);
	}

        template <typename theirFloat>
	__device__ __host__ inline void atomicAdd(int d, int parity, int x, int s_row, int s_col,
						  int c_row, int c_col, const complex<theirFloat> &val) {
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
	  double nrm1 = accessor.transform_reduce(location, dim, abs_<double,storeFloat>(accessor.scale_inv),
                                                  thrust::plus<double>(), 0.0);
	  if (global) comm_allreduce(&nrm1);
	  return nrm1;
	}

	/**
	 * @brief Returns the L2 norm squared of the field in a given dimension
	 * @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
	 * @return L2 norm squared
	 */
	__host__ double norm2(int dim=-1, bool global=true) const {
	  double nrm2 = accessor.transform_reduce(location, dim, square_<double,storeFloat>(accessor.scale_inv),
                                                  thrust::plus<double>(), 0.0);
	  if (global) comm_allreduce(&nrm2);
	  return nrm2;
	}

	/**
	 * @brief Returns the Linfinity norm of the field in a given dimension
	 * @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
	 * @return Linfinity norm
	 */
	__host__ double abs_max(int dim=-1, bool global=true) const {
	  double absmax = accessor.transform_reduce(location, dim, abs_<Float,storeFloat>(accessor.scale_inv),
                                                    thrust::maximum<Float>(), 0.0);
	  if (global) comm_allreduce_max(&absmax);
	  return absmax;
	}

	/**
	 * @brief Returns the minimum absolute value of the field
	 * @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
	 * @return Minimum norm
	 */
	__host__ double abs_min(int dim=-1, bool global=true) const {
	  double absmin = accessor.transform_reduce(location, dim, abs_<Float,storeFloat>(accessor.scale_inv),
                                                    thrust::minimum<Float>(), std::numeric_limits<double>::max());
	  if (global) comm_allreduce_min(&absmin);
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
        typedef typename mapper<Float>::type RegType;
        RegType scale;
        RegType scale_inv;
        Reconstruct(const GaugeField &u) :
          scale(isFixed<Float>::value ? u.LinkMax() : 1.0),
          scale_inv(isFixed<Float>::value ? 1.0 / scale : 1.0)
        {
        }

        Reconstruct(const Reconstruct<N, Float, ghostExchange_> &recon) : scale(recon.scale), scale_inv(recon.scale_inv)
        {
        }

        __device__ __host__ inline void Pack(RegType out[N], const RegType in[N], int idx) const
        {
          if (isFixed<Float>::value) {
#pragma unroll
            for (int i = 0; i < N; i++) out[i] = scale_inv * in[i];
          } else {
#pragma unroll
            for (int i = 0; i < N; i++) out[i] = in[i];
          }
        }

        template <typename I>
        __device__ __host__ inline void Unpack(
            RegType out[N], const RegType in[N], int idx, int dir, const RegType phase, const I *X, const int *R) const
        {
          if (isFixed<Float>::value) {
#pragma unroll
            for (int i = 0; i < N; i++) out[i] = scale * in[i];
          } else {
#pragma unroll
            for (int i = 0; i < N; i++) out[i] = in[i];
          }
        }
        __device__ __host__ inline RegType getPhase(const RegType in[N]) const { return 0; }
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
        typedef typename mapper<Float>::type RegType;
        typedef complex<RegType> Complex;
        const RegType anisotropy;
        const RegType tBoundary;
        const int firstTimeSliceBound;
        const int lastTimeSliceBound;
        const bool isFirstTimeSlice;
        const bool isLastTimeSlice;
        QudaGhostExchange ghostExchange;

        Reconstruct(const GaugeField &u) :
            anisotropy(u.Anisotropy()),
            tBoundary(static_cast<RegType>(u.TBoundary())),
            firstTimeSliceBound(u.VolumeCB()),
            lastTimeSliceBound((u.X()[3] - 1) * u.X()[0] * u.X()[1] * u.X()[2] / 2),
            isFirstTimeSlice(comm_coord(3) == 0 ? true : false),
            isLastTimeSlice(comm_coord(3) == comm_dim(3) - 1 ? true : false),
            ghostExchange(u.GhostExchange())
        {
        }

        Reconstruct(const Reconstruct<12, Float, ghostExchange_> &recon) :
            anisotropy(recon.anisotropy),
            tBoundary(recon.tBoundary),
            firstTimeSliceBound(recon.firstTimeSliceBound),
            lastTimeSliceBound(recon.lastTimeSliceBound),
            isFirstTimeSlice(recon.isFirstTimeSlice),
            isLastTimeSlice(recon.isLastTimeSlice),
            ghostExchange(recon.ghostExchange)
        {
        }

        __device__ __host__ inline void Pack(RegType out[12], const RegType in[18], int idx) const {
#pragma unroll
	  for (int i=0; i<12; i++) out[i] = in[i];
	}

	template<typename I>
	__device__ __host__ inline void Unpack(RegType out[18], const RegType in[12], int idx, int dir,
					       const RegType phase, const I *X, const int *R) const {
          Complex Out[9];
#pragma unroll
          for (int i = 0; i < 6; i++) Out[i] = Complex(in[2 * i + 0], in[2 * i + 1]);

          const RegType u0 = dir < 3 ?
              anisotropy :
              timeBoundary<ghostExchange_>(idx, X, R, tBoundary, static_cast<RegType>(1.0), firstTimeSliceBound,
                  lastTimeSliceBound, isFirstTimeSlice, isLastTimeSlice, ghostExchange);

          // Out[6] = u0*conj(Out[1]*Out[5] - Out[2]*Out[4]);
          Out[6] = cmul(Out[2], Out[4]);
          Out[6] = cmac(Out[1], Out[5], -Out[6]);
          Out[6] = u0 * conj(Out[6]);

          // Out[7] = u0*conj(Out[2]*Out[3] - Out[0]*Out[5]);
          Out[7] = cmul(Out[0], Out[5]);
          Out[7] = cmac(Out[2], Out[3], -Out[7]);
          Out[7] = u0 * conj(Out[7]);

          // Out[8] = u0*conj(Out[0]*Out[4] - Out[1]*Out[3]);
          Out[8] = cmul(Out[1], Out[3]);
          Out[8] = cmac(Out[0], Out[4], -Out[8]);
          Out[8] = u0 * conj(Out[8]);

#pragma unroll
          for (int i = 0; i < 9; i++) {
            out[2 * i + 0] = Out[i].real();
            out[2 * i + 1] = Out[i].imag();
          }
        }

        __device__ __host__ inline RegType getPhase(const RegType in[18]) { return 0; }
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
        typedef typename mapper<Float>::type RegType;

	Reconstruct(const GaugeField &u) { ; }
        Reconstruct(const Reconstruct<11, Float, ghostExchange_> &recon) {}

        __device__ __host__ inline void Pack(RegType out[10], const RegType in[18], int idx) const {
#pragma unroll
	  for (int i=0; i<4; i++) out[i] = in[i+2];
	  out[4] = in[10];
	  out[5] = in[11];
	  out[6] = in[1];
	  out[7] = in[9];
	  out[8] = in[17];
	  out[9] = 0.0;
	}

	template<typename I>
	__device__ __host__ inline void Unpack(RegType out[18], const RegType in[10], int idx, int dir,
					       const RegType phase, const I *X, const int *R) const {
	  out[0] = 0.0;
	  out[1] = in[6];
#pragma unroll
	  for (int i=0; i<4; i++) out[i+2] = in[i];
	  out[6] = -out[2];
	  out[7] =  out[3];
	  out[8] = 0.0;
	  out[9] = in[7];
	  out[10] = in[4];
	  out[11] = in[5];
	  out[12] = -out[4];
	  out[13] =  out[5];
	  out[14] = -out[10];
	  out[15] =  out[11];
	  out[16] = 0.0;
	  out[17] = in[8];
	}

	__device__ __host__ inline RegType getPhase(const RegType in[18]) { return 0; }
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
        typedef typename mapper<Float>::type RegType;
        typedef complex<RegType> Complex;
        const Reconstruct<12, Float, ghostExchange_> reconstruct_12;
        const RegType scale;
        const RegType scale_inv;

        Reconstruct(const GaugeField &u) : reconstruct_12(u), scale(u.Scale()), scale_inv(1.0 / scale) {}
        Reconstruct(const Reconstruct<13, Float, ghostExchange_, stag_phase> &recon) :
            reconstruct_12(recon.reconstruct_12),
            scale(recon.scale),
            scale_inv(recon.scale_inv)
        {
        }

        __device__ __host__ inline void Pack(RegType out[12], const RegType in[18], int idx) const
        {
          reconstruct_12.Pack(out, in, idx);
        }

        template <typename I>
        __device__ __host__ inline void Unpack(RegType out[18], const RegType in[12], int idx, int dir,
            const RegType phase, const I *X, const int *R) const
        {
          Complex Out[9];
#pragma unroll
          for (int i = 0; i < 6; i++) Out[i] = Complex(in[2 * i + 0], in[2 * i + 1]);

          Out[6] = cmul(Out[2], Out[4]);
          Out[6] = cmac(Out[1], Out[5], -Out[6]);
          Out[6] = scale_inv * conj(Out[6]);

          Out[7] = cmul(Out[0], Out[5]);
          Out[7] = cmac(Out[2], Out[3], -Out[7]);
          Out[7] = scale_inv * conj(Out[7]);

          Out[8] = cmul(Out[1], Out[3]);
          Out[8] = cmac(Out[0], Out[4], -Out[8]);
          Out[8] = scale_inv * conj(Out[8]);

          if (stag_phase == QUDA_STAGGERED_PHASE_NO) { // dynamic phasing
            // Multiply the third row by exp(I*3*phase), since the cross product will end up in a scale factor of exp(-I*2*phase)
            RegType cos_sin[2];
            Trig<isFixed<RegType>::value, RegType>::SinCos(static_cast<RegType>(3. * phase), &cos_sin[1], &cos_sin[0]);
            Complex A(cos_sin[0], cos_sin[1]);
            Out[6] = cmul(A, Out[6]);
            Out[7] = cmul(A, Out[7]);
            Out[8] = cmul(A, Out[8]);
          } else { // phase is +/- 1 so real multiply is sufficient
            Out[6] *= phase;
            Out[7] *= phase;
            Out[8] *= phase;
          }

#pragma unroll
          for (int i = 0; i < 9; i++) {
            out[2 * i + 0] = Out[i].real();
            out[2 * i + 1] = Out[i].imag();
          }
        }

        __device__ __host__ inline RegType getPhase(const RegType in[18]) const
        {
#if 1 // phase from cross product
          Complex In[9];
#pragma unroll
          for (int i = 0; i < 9; i++) In[i] = Complex(in[2 * i + 0], in[2 * i + 1]);
          // denominator = (U[0][0]*U[1][1] - U[0][1]*U[1][0])*
          Complex denom = conj(In[0] * In[4] - In[1] * In[3]) * scale_inv;
          Complex expI3Phase = In[8] / denom; // numerator = U[2][2]

          if (stag_phase == QUDA_STAGGERED_PHASE_NO) { // dynamic phasing
            return arg(expI3Phase) / static_cast<RegType>(3.0);
          } else {
            return expI3Phase.real() > 0 ? 1 : -1;
          }
#else // phase from determinant
          Matrix<Complex, 3> a;
#pragma unroll
          for (int i = 0; i < 9; i++) a(i) = Complex(in[2 * i] * scale_inv, in[2 * i + 1] * scale_inv);
          const Complex det = getDeterminant(a);
          return phase = arg(det) / 3;
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
        typedef typename mapper<Float>::type RegType;
        typedef complex<RegType> Complex;
        const Complex anisotropy; // imaginary value stores inverse
        const Complex tBoundary;  // imaginary value stores inverse
        const int firstTimeSliceBound;
        const int lastTimeSliceBound;
        const bool isFirstTimeSlice;
        const bool isLastTimeSlice;
        QudaGhostExchange ghostExchange;

        // scale factor is set when using recon-9
        Reconstruct(const GaugeField &u, RegType scale = 1.0) :
            anisotropy(u.Anisotropy() * scale, 1.0 / (u.Anisotropy() * scale)),
            tBoundary(static_cast<RegType>(u.TBoundary()) * scale, 1.0 / (static_cast<RegType>(u.TBoundary()) * scale)),
            firstTimeSliceBound(u.VolumeCB()),
            lastTimeSliceBound((u.X()[3] - 1) * u.X()[0] * u.X()[1] * u.X()[2] / 2),
            isFirstTimeSlice(comm_coord(3) == 0 ? true : false),
            isLastTimeSlice(comm_coord(3) == comm_dim(3) - 1 ? true : false),
            ghostExchange(u.GhostExchange())
        {
        }

        Reconstruct(const Reconstruct<8, Float, ghostExchange_> &recon) :
            anisotropy(recon.anisotropy),
            tBoundary(recon.tBoundary),
            firstTimeSliceBound(recon.firstTimeSliceBound),
            lastTimeSliceBound(recon.lastTimeSliceBound),
            isFirstTimeSlice(recon.isFirstTimeSlice),
            isLastTimeSlice(recon.isLastTimeSlice),
            ghostExchange(recon.ghostExchange)
        {
        }

        __device__ __host__ inline void Pack(RegType out[8], const RegType in[18], int idx) const
        {
          out[0] = Trig<isFixed<Float>::value, RegType>::Atan2(in[1], in[0]);
          out[1] = Trig<isFixed<Float>::value, RegType>::Atan2(in[13], in[12]);
#pragma unroll
      for (int i=2; i<8; i++) out[i] = in[i];
    }

    template <typename I>
    __device__ __host__ inline void Unpack(RegType out[18], const RegType in[8], int idx, int dir, const RegType phase,
        const I *X, const int *R, const Complex scale, const Complex u) const
    {

      RegType u0 = u.real();
      RegType u0_inv = u.imag();

      Complex Out[9];
#pragma unroll
      for (int i = 1; i <= 3; i++) Out[i] = Complex(in[2 * i + 0], in[2 * i + 1]); // these elements are copied directly

      Trig<isFixed<Float>::value, RegType>::SinCos(in[0], &Out[0].y, &Out[0].x);
      Trig<isFixed<Float>::value, RegType>::SinCos(in[1], &Out[6].y, &Out[6].x);

      // First, reconstruct first row
      RegType row_sum = Out[1].real() * Out[1].real();
      row_sum += Out[1].imag() * Out[1].imag();
      row_sum += Out[2].real() * Out[2].real();
      row_sum += Out[2].imag() * Out[2].imag();
      RegType row_sum_inv = static_cast<RegType>(1.0) / row_sum;

      RegType diff = u0_inv * u0_inv - row_sum;
      RegType U00_mag = diff > 0.0 ? diff * rsqrt(diff) : static_cast<RegType>(0.0);

      Out[0] *= U00_mag;

      // Second, reconstruct first column
      RegType column_sum = Out[0].real() * Out[0].real();
      column_sum += Out[0].imag() * Out[0].imag();
      column_sum += Out[3].real() * Out[3].real();
      column_sum += Out[3].imag() * Out[3].imag();

      diff = u0_inv * u0_inv - column_sum;
      RegType U20_mag = diff > 0.0 ? diff * rsqrt(diff) : static_cast<RegType>(0.0);

      Out[6] *= U20_mag;

      // Finally, reconstruct last elements from SU(2) rotation
      RegType r_inv2 = u0_inv * row_sum_inv;
      {
        Complex A = cmul(conj(Out[0]), Out[3]);

        // Out[4] = -(conj(Out[6])*conj(Out[2]) + u0*A*Out[1])*r_inv2; // U11
        Out[4] = cmul(conj(Out[6]), conj(Out[2]));
        Out[4] = cmac(u0 * A, Out[1], Out[4]);
        Out[4] = -r_inv2 * Out[4];

        // Out[5] = (conj(Out[6])*conj(Out[1]) - u0*A*Out[2])*r_inv2;  // U12
        Out[5] = cmul(conj(Out[6]), conj(Out[1]));
        Out[5] = cmac(-u0 * A, Out[2], Out[5]);
        Out[5] = r_inv2 * Out[5];
      }

      {
        Complex A = cmul(conj(Out[0]), Out[6]);

        // Out[7] = (conj(Out[3])*conj(Out[2]) - u0*A*Out[1])*r_inv2;  // U21
        Out[7] = cmul(conj(Out[3]), conj(Out[2]));
        Out[7] = cmac(-u0 * A, Out[1], Out[7]);
        Out[7] = r_inv2 * Out[7];

        // Out[8] = -(conj(Out[3])*conj(Out[1]) + u0*A*Out[2])*r_inv2; // U12
        Out[8] = cmul(conj(Out[3]), conj(Out[1]));
        Out[8] = cmac(u0 * A, Out[2], Out[8]);
        Out[8] = -r_inv2 * Out[8];
      }

#pragma unroll
      for (int i = 0; i < 9; i++) {
        out[2 * i + 0] = Out[i].real();
        out[2 * i + 1] = Out[i].imag();
      }
    }

    template <typename I>
    __device__ __host__ inline void Unpack(RegType out[18], const RegType in[8], int idx, int dir, const RegType phase,
        const I *X, const int *R, const Complex scale = Complex(static_cast<RegType>(1.0), static_cast<RegType>(1.0))) const
    {
      Complex u = dir < 3 ? anisotropy :
                            timeBoundary<ghostExchange_>(idx, X, R, tBoundary, scale, firstTimeSliceBound,
                                lastTimeSliceBound, isFirstTimeSlice, isLastTimeSlice, ghostExchange);
      Unpack(out, in, idx, dir, phase, X, R, scale, u);
    }

    __device__ __host__ inline RegType getPhase(const RegType in[18]){ return 0; }
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
        typedef typename mapper<Float>::type RegType;
        typedef complex<RegType> Complex;
        const Reconstruct<8, Float, ghostExchange_> reconstruct_8;
        const RegType scale;
        const RegType scale_inv;

        Reconstruct(const GaugeField &u) : reconstruct_8(u), scale(u.Scale()), scale_inv(1.0 / scale) {}

        Reconstruct(const Reconstruct<9, Float, ghostExchange_, stag_phase> &recon) :
            reconstruct_8(recon.reconstruct_8),
            scale(recon.scale),
            scale_inv(recon.scale_inv)
        {
        }

        __device__ __host__ inline RegType getPhase(const RegType in[18]) const
        {
#if 1 // phase from cross product
          Complex In[9];
#pragma unroll
          for (int i = 0; i < 9; i++) In[i] = Complex(in[2 * i + 0], in[2 * i + 1]);
          // denominator = (U[0][0]*U[1][1] - U[0][1]*U[1][0])*
          Complex denom = conj(In[0] * In[4] - In[1] * In[3]) * scale_inv;
          Complex expI3Phase = In[8] / denom; // numerator = U[2][2]
          if (stag_phase == QUDA_STAGGERED_PHASE_NO) {
            return arg(expI3Phase) / static_cast<RegType>(3.0);
          } else {
            return expI3Phase.real() > 0 ? 1 : -1;
          }
#else // phase from determinant
	Matrix<Complex,3> a;
#pragma unroll
        for (int i = 0; i < 9; i++) a(i) = Complex(in[2 * i], in[2 * i + 1]) * scale_inv;
        const Complex det = getDeterminant( a );
	RegType phase = arg(det)/3;
	return phase;
#endif
        }

        // Rescale the U3 input matrix by exp(-I*phase) to obtain an SU3 matrix multiplied by a real scale factor,
        __device__ __host__ inline void Pack(RegType out[8], const RegType in[18], int idx) const
        {
          RegType phase = getPhase(in);
          RegType su3[18];

          if (stag_phase == QUDA_STAGGERED_PHASE_NO) {
            RegType cos_sin[2];
            Trig<isFixed<RegType>::value, RegType>::SinCos(static_cast<RegType>(-phase), &cos_sin[1], &cos_sin[0]);
            Complex z(cos_sin[0], cos_sin[1]);
            z *= scale_inv;
#pragma unroll
            for (int i = 0; i < 9; i++) {
              Complex su3_ = cmul(z, Complex(in[2 * i + 0], in[2 * i + 1]));
              su3[2 * i + 0] = su3_.real();
              su3[2 * i + 1] = su3_.imag();
            }
          } else {
#pragma unroll
            for (int i = 0; i < 18; i++) { su3[i] = phase * in[i]; }
          }
          reconstruct_8.Pack(out, su3, idx);
        }

        template <typename I>
        __device__ __host__ inline void Unpack(
            RegType out[18], const RegType in[8], int idx, int dir, const RegType phase, const I *X, const int *R) const
        {
          reconstruct_8.Unpack(out, in, idx, dir, phase, X, R,
              Complex(static_cast<RegType>(1.0), static_cast<RegType>(1.0)),
              Complex(static_cast<RegType>(1.0), static_cast<RegType>(1.0)));

          if (stag_phase == QUDA_STAGGERED_PHASE_NO) { // dynamic phase
            RegType cos_sin[2];
            Trig<isFixed<RegType>::value, RegType>::SinCos(static_cast<RegType>(phase), &cos_sin[1], &cos_sin[0]);
            Complex z(cos_sin[0], cos_sin[1]);
            z *= scale;
#pragma unroll
            for (int i = 0; i < 9; i++) {
              Complex Out = cmul(z, Complex(out[2 * i + 0], out[2 * i + 1]));
              out[2 * i + 0] = Out.real();
              out[2 * i + 1] = Out.imag();
            };
          } else { // stagic phase
#pragma unroll
            for (int i = 0; i < 18; i++) { out[i] *= phase; }
          }
        }
      };

      __host__ __device__ inline constexpr int ct_sqrt(int n, int i = 1)
      {
        return n == i ? n : (i * i < n ? ct_sqrt(n, i + 1) : i);
      }

      /**
         @brief Return the number of colors of the accessor based on the length of the field
         @param[in] length Number of real numbers per link
         @return Number of colors (=sqrt(length/2))
       */
      __host__ __device__ inline constexpr int Ncolor(int length) { return ct_sqrt(length / 2); }

      // we default to huge allocations for gauge field (for now)
      constexpr bool default_huge_alloc = true;

      template <QudaStaggeredPhase phase> __host__ __device__ inline bool static_phase()
      {
        switch (phase) {
        case QUDA_STAGGERED_PHASE_MILC:
        case QUDA_STAGGERED_PHASE_CPS:
        case QUDA_STAGGERED_PHASE_TIFR: return true;
        default: return false;
        }
      }

      template <typename Float, int length, int N, int reconLenParam,
          QudaStaggeredPhase stag_phase = QUDA_STAGGERED_PHASE_NO, bool huge_alloc = default_huge_alloc,
          QudaGhostExchange ghostExchange_ = QUDA_GHOST_EXCHANGE_INVALID, bool use_inphase = false>
      struct FloatNOrder {
        using Accessor
            = FloatNOrder<Float, length, N, reconLenParam, stag_phase, huge_alloc, ghostExchange_, use_inphase>;

        typedef typename mapper<Float>::type RegType;
        typedef typename VectorType<Float, N>::type Vector;
        typedef typename AllocType<huge_alloc>::type AllocInt;
        Reconstruct<reconLenParam, Float, ghostExchange_, stag_phase> reconstruct;
        static const int reconLen = (reconLenParam == 11) ? 10 : reconLenParam;
        static const int hasPhase = (reconLen == 9 || reconLen == 13) ? 1 : 0;
        Float *gauge;
        const AllocInt offset;
#ifdef USE_TEXTURE_OBJECTS
      typedef typename TexVectorType<RegType,N>::type TexVector;
      cudaTextureObject_t tex;
      const int tex_offset;
#endif
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
      void *backup_h; //! host memory for backing up the field when tuning
      size_t bytes;

    FloatNOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0, bool override=false)
      : reconstruct(u), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()),
	offset(u.Bytes()/(2*sizeof(Float))),
#ifdef USE_TEXTURE_OBJECTS
	tex(0), tex_offset(offset/N),
#endif
	ghostExchange(u.GhostExchange()),
	volumeCB(u.VolumeCB()), stride(u.Stride()), geometry(u.Geometry()),
	phaseOffset(u.PhaseOffset()), backup_h(nullptr), bytes(u.Bytes())
      {
	if (geometry == QUDA_COARSE_GEOMETRY)
	  errorQuda("This accessor does not support coarse-link fields (lacks support for bidirectional ghost zone");

        // static_assert( !(stag_phase!=QUDA_STAGGERED_PHASE_NO && reconLenParam != 18 && reconLenParam != 12),
        // 	       "staggered phase only presently supported for 18 and 12 reconstruct");
        for (int i = 0; i < 4; i++) {
          X[i] = u.X()[i];
          R[i] = u.R()[i];
	  ghost[i] = ghost_ ? ghost_[i] : 0;
	  faceVolumeCB[i] = u.SurfaceCB(i)*u.Nface(); // face volume equals surface * depth
        }
#ifdef USE_TEXTURE_OBJECTS
	if (u.Location() == QUDA_CUDA_FIELD_LOCATION) tex = static_cast<const cudaGaugeField&>(u).Tex();
	if (!huge_alloc && this->gauge != u.Gauge_p() && !override) {
	  errorQuda("Cannot use texture read since data pointer does not equal field pointer - use with huge_alloc=true instead");
	}
#endif
      }

    FloatNOrder(const FloatNOrder &order)
      : reconstruct(order.reconstruct), gauge(order.gauge), offset(order.offset),
#ifdef USE_TEXTURE_OBJECTS
	tex(order.tex), tex_offset(order.tex_offset),
#endif
	ghostExchange(order.ghostExchange),
        volumeCB(order.volumeCB), stride(order.stride), geometry(order.geometry),
	phaseOffset(order.phaseOffset), backup_h(nullptr), bytes(order.bytes)
      {
	for (int i=0; i<4; i++) {
	  X[i] = order.X[i];
	  R[i] = order.R[i];
	  ghost[i] = order.ghost[i];
	  faceVolumeCB[i] = order.faceVolumeCB[i];
	}
      }
      virtual ~FloatNOrder() { ; }

      __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity, Float inphase = 1.0) const
      {
        const int M = reconLen / N;
        RegType tmp[reconLen];

#pragma unroll
        for (int i=0; i<M; i++){
          // first do texture load from memory
#if defined(USE_TEXTURE_OBJECTS) && defined(__CUDA_ARCH__)
	  if (!huge_alloc) { // use textures unless we have a huge alloc
            TexVector vecTmp = tex1Dfetch<TexVector>(tex, parity * tex_offset + (dir * M + i) * stride + x);
            // now insert into output array
#pragma unroll
	    for (int j=0; j<N; j++) copy(tmp[i*N+j], reinterpret_cast<RegType*>(&vecTmp)[j]);
	  } else
#endif
	  {
            // first load from memory
            Vector vecTmp = vector_load<Vector>(gauge + parity * offset, (dir * M + i) * stride + x);
            // second do copy converting into register type
#pragma unroll
	    for (int j=0; j<N; j++) copy(tmp[i*N+j], reinterpret_cast<Float*>(&vecTmp)[j]);
	  }
	}

        RegType phase = 0.; // TODO - add texture support for phases

        if (hasPhase) {
          if (static_phase<stag_phase>() && (reconLen == 13 || use_inphase)) {
            phase = inphase;
          } else {
            copy(phase, (gauge + parity * offset)[phaseOffset / sizeof(Float) + stride * dir + x]);
            phase *= static_cast<RegType>(2.0) * static_cast<RegType>(M_PI);
          }
        }

        reconstruct.Unpack(v, tmp, x, dir, phase, X, R);
      }

      __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {

        const int M = reconLen / N;
        RegType tmp[reconLen];
        reconstruct.Pack(tmp, v, x);

#pragma unroll
        for (int i=0; i<M; i++){
	  Vector vecTmp;
	  // first do copy converting into storage type
#pragma unroll
	  for (int j=0; j<N; j++) copy(reinterpret_cast<Float*>(&vecTmp)[j], tmp[i*N+j]);
	  // second do vectorized copy into memory
          vector_store(gauge + parity * offset, x + (dir * M + i) * stride, vecTmp);
        }
        if(hasPhase){
          RegType phase = reconstruct.getPhase(v);
          copy((gauge+parity*offset)[phaseOffset/sizeof(Float) + dir*stride + x], static_cast<RegType>(phase/(2.*M_PI)));
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
      __device__ __host__ inline gauge_wrapper<RegType, Accessor> operator()(
          int dim, int x_cb, int parity, Float phase = 1.0)
      {
        return gauge_wrapper<RegType, Accessor>(*this, dim, x_cb, parity, phase);
      }

      /**
	 @brief This accessor routine returns a const gauge_wrapper to this object,
	 allowing us to overload various operators for manipulating at
	 the site level interms of matrix operations.
	 @param[in] dir Which dimension are we requesting
	 @param[in] x_cb Checkerboarded space-time index we are requesting
	 @param[in] parity Parity we are requesting
	 @return Instance of a gauge_wrapper that curries in access to
	 this field at the above coordinates.
       */
      __device__ __host__ inline const gauge_wrapper<RegType, Accessor> operator()(
          int dim, int x_cb, int parity, Float phase = 1.0) const
      {
        return gauge_wrapper<RegType, Accessor>(const_cast<Accessor &>(*this), dim, x_cb, parity, phase);
      }

      __device__ __host__ inline void loadGhost(RegType v[length], int x, int dir, int parity, Float inphase = 1.0) const
      {
        if (!ghost[dir]) { // load from main field not separate array
          load(v, volumeCB + x, dir, parity, inphase); // an offset of size volumeCB puts us at the padded region
          // This also works perfectly when phases are stored. No need to change this.
        } else {
          const int M = reconLen / N;
          RegType tmp[reconLen];

#pragma unroll
          for (int i=0; i<M; i++) {
	    // first do vectorized copy from memory into registers
            Vector vecTmp = vector_load<Vector>(
                ghost[dir] + parity * faceVolumeCB[dir] * (M * N + hasPhase), i * faceVolumeCB[dir] + x);
            // second do copy converting into register type
#pragma unroll
            for (int j = 0; j < N; j++) copy(tmp[i * N + j], reinterpret_cast<Float *>(&vecTmp)[j]);
          }
          RegType phase = 0.;

          if (hasPhase) {

            // if(stag_phase == QUDA_STAGGERED_PHASE_MILC )  {
            //   phase = inphase < static_cast<Float>(0) ? static_cast<Float>(-1./(2.*M_PI)) : static_cast<Float>(1./2.*M_PI);
            // } else {
            copy(phase, ghost[dir][parity * faceVolumeCB[dir] * (M * N + 1) + faceVolumeCB[dir] * M * N + x]);
            phase *= static_cast<RegType>(2.0) * static_cast<RegType>(M_PI);
            // }
          }
          reconstruct.Unpack(v, tmp, x, dir, phase, X, R);
        }
      }

      __device__ __host__ inline void saveGhost(const RegType v[length], int x, int dir, int parity) {
        if (!ghost[dir]) { // store in main field not separate array
	  save(v, volumeCB+x, dir, parity); // an offset of size volumeCB puts us at the padded region
        } else {
          const int M = reconLen / N;
          RegType tmp[reconLen];
          reconstruct.Pack(tmp, v, x);

#pragma unroll
          for (int i=0; i<M; i++) {
	    Vector vecTmp;
	    // first do copy converting into storage type
#pragma unroll
	    for (int j=0; j<N; j++) copy(reinterpret_cast<Float*>(&vecTmp)[j], tmp[i*N+j]);
	    // second do vectorized copy into memory
	    vector_store(ghost[dir]+parity*faceVolumeCB[dir]*(M*N + hasPhase), i*faceVolumeCB[dir]+x, vecTmp);
          }

	  if (hasPhase) {
	    RegType phase = reconstruct.getPhase(v);
	    copy(ghost[dir][parity*faceVolumeCB[dir]*(M*N + 1) + faceVolumeCB[dir]*M*N + x], static_cast<RegType>(phase/(2.*M_PI)));
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
	 @return Instance of a gauge_wrapper that curries in access to
	 this field at the above coordinates.
       */
      __device__ __host__ inline gauge_ghost_wrapper<RegType, Accessor> Ghost(
          int dim, int ghost_idx, int parity, Float phase = 1.0)
      {
        return gauge_ghost_wrapper<RegType, Accessor>(*this, dim, ghost_idx, parity, phase);
      }

      /**
	 @brief This accessor routine returns a const gauge_wrapper to this object,
	 allowing us to overload various operators for manipulating at
	 the site level interms of matrix operations.
	 @param[in] dir Which dimension are we requesting
	 @param[in] ghost_idx Ghost index we are requesting
	 @param[in] parity Parity we are requesting
	 @return Instance of a gauge_wrapper that curries in access to
	 this field at the above coordinates.
       */
      __device__ __host__ inline const gauge_ghost_wrapper<RegType, Accessor> Ghost(
          int dim, int ghost_idx, int parity, Float phase = 1.0) const
      {
        return gauge_ghost_wrapper<RegType, Accessor>(const_cast<Accessor &>(*this), dim, ghost_idx, parity, phase);
      }

      __device__ __host__ inline void loadGhostEx(RegType v[length], int buff_idx, int extended_idx, int dir,
						  int dim, int g, int parity, const int R[]) const {
	const int M = reconLen / N;
	RegType tmp[reconLen];

#pragma unroll
	for (int i=0; i<M; i++) {
	  // first do vectorized copy from memory
	  Vector vecTmp = vector_load<Vector>(ghost[dim] + ((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + hasPhase),
					      +i*R[dim]*faceVolumeCB[dim]+buff_idx);
	  // second do copy converting into register type
#pragma unroll
	  for (int j=0; j<N; j++) copy(tmp[i*N+j], reinterpret_cast<Float*>(&vecTmp)[j]);
	}
	RegType phase=0.;
	if(hasPhase) copy(phase, ghost[dim][((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + 1)
					    + R[dim]*faceVolumeCB[dim]*M*N + buff_idx]);

	// use the extended_idx to determine the boundary condition
	reconstruct.Unpack(v, tmp, extended_idx, g, 2.*M_PI*phase, X, R);
      }

      __device__ __host__ inline void saveGhostEx(const RegType v[length], int buff_idx, int extended_idx,
						  int dir, int dim, int g, int parity, const int R[]) {
	const int M = reconLen / N;
	RegType tmp[reconLen];
	// use the extended_idx to determine the boundary condition
	reconstruct.Pack(tmp, v, extended_idx);

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
	  if (hasPhase) {
	    RegType phase = reconstruct.getPhase(v);
	    copy(ghost[dim][((dir*2+parity)*geometry+g)*R[dim]*faceVolumeCB[dim]*(M*N + 1) + R[dim]*faceVolumeCB[dim]*M*N + buff_idx],
		 static_cast<RegType>(phase/(2.*M_PI)));
	  }
	}

      /**
	 @brief Backup the field to the host when tuning
      */
      void save() {
	if (backup_h) errorQuda("Already allocated host backup");
	backup_h = safe_malloc(bytes);
	cudaMemcpy(backup_h, gauge, bytes, cudaMemcpyDeviceToHost);
	checkCudaError();
      }

      /**
	 @brief Restore the field from the host after tuning
      */
      void load() {
	cudaMemcpy(gauge, backup_h, bytes, cudaMemcpyHostToDevice);
	host_free(backup_h);
	backup_h = nullptr;
	checkCudaError();
      }

      size_t Bytes() const { return reconLen * sizeof(Float); }
      };

  /**
     @brief This is just a dummy structure we use for trove to define the
     required structure size
     @param real Real number type
     @param length Number of elements in the structure
  */
  template <typename real, int length> struct S { real v[length]; };

  /**
      The LegacyOrder defines the ghost zone storage and ordering for
      all cpuGaugeFields, which use the same ghost zone storage.
  */
  template <typename Float, int length>
    struct LegacyOrder {
      typedef typename mapper<Float>::type RegType;
      Float *ghost[QUDA_MAX_DIM];
      int faceVolumeCB[QUDA_MAX_DIM];
      const int volumeCB;
      const int stride;
      const int geometry;
      const int hasPhase;

      LegacyOrder(const GaugeField &u, Float **ghost_)
      : volumeCB(u.VolumeCB()), stride(u.Stride()), geometry(u.Geometry()), hasPhase(0) {
	if (geometry == QUDA_COARSE_GEOMETRY)
	  errorQuda("This accessor does not support coarse-link fields (lacks support for bidirectional ghost zone");

	for (int i=0; i<4; i++) {
	  ghost[i] = (ghost_) ? ghost_[i] : (Float*)(u.Ghost()[i]);
	  faceVolumeCB[i] = u.SurfaceCB(i)*u.Nface(); // face volume equals surface * depth
	}
      }

      LegacyOrder(const LegacyOrder &order)
      : volumeCB(order.volumeCB), stride(order.stride), geometry(order.geometry), hasPhase(0) {
	for (int i=0; i<4; i++) {
	  ghost[i] = order.ghost[i];
	  faceVolumeCB[i] = order.faceVolumeCB[i];
	}
      }

      virtual ~LegacyOrder() { ; }

      __device__ __host__ inline void loadGhost(RegType v[length], int x, int dir, int parity) const {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	typedef S<Float,length> structure;
	trove::coalesced_ptr<structure> ghost_((structure*)ghost[dir]);
	structure v_ = ghost_[parity*faceVolumeCB[dir] + x];
	for (int i=0; i<length; i++) v[i] = (RegType)v_.v[i];
#else
	for (int i=0; i<length; i++) v[i] = (RegType)ghost[dir][(parity*faceVolumeCB[dir] + x)*length + i];
#endif
      }

      __device__ __host__ inline void saveGhost(const RegType v[length], int x, int dir, int parity) {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	typedef S<Float,length> structure;
	trove::coalesced_ptr<structure> ghost_((structure*)ghost[dir]);
	structure v_;
	for (int i=0; i<length; i++) v_.v[i] = (Float)v[i];
	ghost_[parity*faceVolumeCB[dir] + x] = v_;
#else
	for (int i=0; i<length; i++) ghost[dir][(parity*faceVolumeCB[dir] + x)*length + i] = (Float)v[i];
#endif
      }

      __device__ __host__ inline void loadGhostEx(RegType v[length], int x, int dummy, int dir,
						  int dim, int g, int parity, const int R[]) const {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	typedef S<Float,length> structure;
	trove::coalesced_ptr<structure> ghost_((structure*)ghost[dim]);
	structure v_ = ghost_[((dir*2+parity)*R[dim]*faceVolumeCB[dim] + x)*geometry+g];
	for (int i=0; i<length; i++) v[i] = (RegType)v_.v[i];
#else
	for (int i=0; i<length; i++) {
	  v[i] = (RegType)ghost[dim][(((dir*2+parity)*R[dim]*faceVolumeCB[dim] + x)*geometry+g)*length + i];
	}
#endif
      }

      __device__ __host__ inline void saveGhostEx(const RegType v[length], int x, int dummy,
						  int dir, int dim, int g, int parity, const int R[]) {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	typedef S<Float,length> structure;
	trove::coalesced_ptr<structure> ghost_((structure*)ghost[dim]);
	structure v_;
	for (int i=0; i<length; i++) v_.v[i] = (Float)v[i];
	ghost_[((dir*2+parity)*R[dim]*faceVolumeCB[dim] + x)*geometry+g] = v_;
#else
	for (int i=0; i<length; i++) {
	  ghost[dim]
	    [(((dir*2+parity)*R[dim]*faceVolumeCB[dim] + x)*geometry+g)*length + i] = (Float)v[i];
	}
#endif
      }

    };

    /**
       struct to define QDP ordered gauge fields:
       [[dim]] [[parity][volumecb][row][col]]
    */
    template <typename Float, int length> struct QDPOrder : public LegacyOrder<Float,length> {
      typedef typename mapper<Float>::type RegType;
      Float *gauge[QUDA_MAX_DIM];
      const int volumeCB;
    QDPOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0)
      : LegacyOrder<Float,length>(u, ghost_), volumeCB(u.VolumeCB())
	{ for (int i=0; i<4; i++) gauge[i] = gauge_ ? ((Float**)gauge_)[i] : ((Float**)u.Gauge_p())[i]; }
    QDPOrder(const QDPOrder &order) : LegacyOrder<Float,length>(order), volumeCB(order.volumeCB) {
	for(int i=0; i<4; i++) gauge[i] = order.gauge[i];
      }
      virtual ~QDPOrder() { ; }

      __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity, Float inphase = 1.0) const
      {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	typedef S<Float,length> structure;
	trove::coalesced_ptr<structure> gauge_((structure*)gauge[dir]);
	structure v_ = gauge_[parity*volumeCB + x];
	for (int i=0; i<length; i++) v[i] = (RegType)v_.v[i];
#else
	for (int i=0; i<length; i++) {
	  v[i] = (RegType)gauge[dir][(parity*volumeCB + x)*length + i];
	}
#endif
      }

      __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	typedef S<Float,length> structure;
	trove::coalesced_ptr<structure> gauge_((structure*)gauge[dir]);
	structure v_;
	for (int i=0; i<length; i++) v_.v[i] = (Float)v[i];
	gauge_[parity*volumeCB + x] = v_;
#else
	for (int i=0; i<length; i++) {
	  gauge[dir][(parity*volumeCB + x)*length + i] = (Float)v[i];
	}
#endif
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
      __device__ __host__ inline gauge_wrapper<RegType, QDPOrder<Float, length>> operator()(int dim, int x_cb, int parity)
      {
        return gauge_wrapper<RegType, QDPOrder<Float, length>>(*this, dim, x_cb, parity);
      }

      /**
	 @brief This accessor routine returns a const gauge_wrapper to this object,
	 allowing us to overload various operators for manipulating at
	 the site level interms of matrix operations.
	 @param[in] dir Which dimension are we requesting
	 @param[in] x_cb Checkerboarded space-time index we are requesting
	 @param[in] parity Parity we are requesting
	 @return Instance of a gauge_wrapper that curries in access to
	 this field at the above coordinates.
       */
      __device__ __host__ inline const gauge_wrapper<RegType, QDPOrder<Float, length>> operator()(
          int dim, int x_cb, int parity) const
      {
        return gauge_wrapper<RegType, QDPOrder<Float, length>>(
            const_cast<QDPOrder<Float, length> &>(*this), dim, x_cb, parity);
      }

      size_t Bytes() const { return length * sizeof(Float); }
    };

    /**
       struct to define QDPJIT ordered gauge fields:
       [[dim]] [[parity][complex][row][col][volumecb]]
    */
    template <typename Float, int length> struct QDPJITOrder : public LegacyOrder<Float,length> {
      typedef typename mapper<Float>::type RegType;
      Float *gauge[QUDA_MAX_DIM];
      const int volumeCB;
    QDPJITOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0)
      : LegacyOrder<Float,length>(u, ghost_), volumeCB(u.VolumeCB())
	{ for (int i=0; i<4; i++) gauge[i] = gauge_ ? ((Float**)gauge_)[i] : ((Float**)u.Gauge_p())[i]; }
    QDPJITOrder(const QDPJITOrder &order) : LegacyOrder<Float,length>(order), volumeCB(order.volumeCB) {
	for(int i=0; i<4; i++) gauge[i] = order.gauge[i];
      }
      virtual ~QDPJITOrder() { ; }

      __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity, Float inphase = 1.0) const
      {
        for (int i = 0; i < length; i++) {
          int z = i % 2;
          int rolcol = i/2;
	  v[i] = (RegType)gauge[dir][((z*(length/2) + rolcol)*2 + parity)*volumeCB + x];
        }
      }

      __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {
	for (int i=0; i<length; i++) {
	  int z = i%2;
	  int rolcol = i/2;
	  gauge[dir][((z*(length/2) + rolcol)*2 + parity)*volumeCB + x] = (Float)v[i];
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
      __device__ __host__ inline gauge_wrapper<RegType, QDPJITOrder<Float, length>> operator()(
          int dim, int x_cb, int parity)
      {
        return gauge_wrapper<RegType, QDPJITOrder<Float, length>>(*this, dim, x_cb, parity);
      }

      /**
	 @brief This accessor routine returns a const gauge_wrapper to this object,
	 allowing us to overload various operators for manipulating at
	 the site level interms of matrix operations.
	 @param[in] dir Which dimension are we requesting
	 @param[in] x_cb Checkerboarded space-time index we are requesting
	 @param[in] parity Parity we are requesting
	 @return Instance of a gauge_wrapper that curries in access to
	 this field at the above coordinates.
       */
      __device__ __host__ inline const gauge_wrapper<RegType, QDPJITOrder<Float, length>> operator()(
          int dim, int x_cb, int parity) const
      {
        return gauge_wrapper<RegType, QDPJITOrder<Float, length>>(
            const_cast<QDPJITOrder<Float, length> &>(*this), dim, x_cb, parity);
      }

      size_t Bytes() const { return length * sizeof(Float); }
    };

  /**
     struct to define MILC ordered gauge fields:
     [parity][dim][volumecb][row][col]
  */
  template <typename Float, int length> struct MILCOrder : public LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge;
    const int volumeCB;
    const int geometry;
  MILCOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) :
    LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()),
      volumeCB(u.VolumeCB()), geometry(u.Geometry()) { ; }
  MILCOrder(const MILCOrder &order) : LegacyOrder<Float,length>(order),
      gauge(order.gauge), volumeCB(order.volumeCB), geometry(order.geometry)
      { ; }
    virtual ~MILCOrder() { ; }

    __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity, Float inphase = 1.0) const
    {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
      typedef S<Float,length> structure;
      trove::coalesced_ptr<structure> gauge_((structure*)gauge);
      structure v_ = gauge_[(parity*volumeCB+x)*geometry + dir];
      for (int i=0; i<length; i++) v[i] = (RegType)v_.v[i];
#else
      for (int i=0; i<length; i++) {
	v[i] = (RegType)gauge[((parity*volumeCB+x)*geometry + dir)*length + i];
      }
#endif
    }

    __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
      typedef S<Float,length> structure;
      trove::coalesced_ptr<structure> gauge_((structure*)gauge);
      structure v_;
      for (int i=0; i<length; i++) v_.v[i] = (Float)v[i];
      gauge_[(parity*volumeCB+x)*geometry + dir] = v_;
#else
      for (int i=0; i<length; i++) {
	gauge[((parity*volumeCB+x)*geometry + dir)*length + i] = (Float)v[i];
      }
#endif
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
    __device__ __host__ inline gauge_wrapper<RegType, MILCOrder<Float, length>> operator()(int dim, int x_cb, int parity)
    {
      return gauge_wrapper<RegType, MILCOrder<Float, length>>(*this, dim, x_cb, parity);
    }

    /**
       @brief This accessor routine returns a const gauge_wrapper to this object,
       allowing us to overload various operators for manipulating at
       the site level interms of matrix operations.
       @param[in] dir Which dimension are we requesting
       @param[in] x_cb Checkerboarded space-time index we are requesting
       @param[in] parity Parity we are requesting
       @return Instance of a gauge_wrapper that curries in access to
       this field at the above coordinates.
    */
    __device__ __host__ inline const gauge_wrapper<RegType, MILCOrder<Float, length>> operator()(
        int dim, int x_cb, int parity) const
    {
      return gauge_wrapper<RegType, MILCOrder<Float, length>>(
          const_cast<MILCOrder<Float, length> &>(*this), dim, x_cb, parity);
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
    typedef typename mapper<Float>::type RegType;
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

    MILCSiteOrder(const MILCSiteOrder &order) :
      LegacyOrder<Float, length>(order),
      gauge(order.gauge),
      volumeCB(order.volumeCB),
      geometry(order.geometry),
      offset(order.offset),
      size(order.size)
    {
    }

    virtual ~MILCSiteOrder() { ; }

    __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity, Float inphase = 1.0) const
    {
      // get base pointer
      const Float *gauge0 = reinterpret_cast<const Float*>(reinterpret_cast<const char*>(gauge) + (parity*volumeCB+x)*size + offset);

#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
      typedef S<Float,length> structure;
      trove::coalesced_ptr<structure> gauge_((structure*)gauge0);
      structure v_ = gauge_[dir];
      for (int i=0; i<length; i++) v[i] = (RegType)v_.v[i];
#else
      for (int i=0; i<length; i++) {
	v[i] = (RegType)gauge0[dir*length + i];
      }
#endif
    }

    __device__ __host__ inline void save(const RegType v[length], int x, int dir, int parity) {
      // get base pointer
      Float *gauge0 = reinterpret_cast<Float*>(reinterpret_cast<char*>(gauge) + (parity*volumeCB+x)*size + offset);

#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
      typedef S<Float,length> structure;
      trove::coalesced_ptr<structure> gauge_((structure*)gauge0);
      structure v_;
      for (int i=0; i<length; i++) v_.v[i] = (Float)v[i];
      gauge_[dir] = v_;
#else
      for (int i=0; i<length; i++) {
	gauge0[dir*length + i] = (Float)v[i];
      }
#endif
    }

    size_t Bytes() const { return length * sizeof(Float); }
  };


  /**
     struct to define CPS ordered gauge fields:
     [parity][dim][volumecb][col][row]
  */
  template <typename Float, int length> struct CPSOrder : LegacyOrder<Float,length> {
    typedef typename mapper<Float>::type RegType;
    Float *gauge;
    const int volumeCB;
    const Float anisotropy;
    static constexpr int Nc = 3;
    const int geometry;
  CPSOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0)
    : LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()),
      volumeCB(u.VolumeCB()), anisotropy(u.Anisotropy()), geometry(u.Geometry())
      { if (length != 18) errorQuda("Gauge length %d not supported", length); }
  CPSOrder(const CPSOrder &order) : LegacyOrder<Float,length>(order), gauge(order.gauge),
      volumeCB(order.volumeCB), anisotropy(order.anisotropy), geometry(order.geometry)
      { ; }
    virtual ~CPSOrder() { ; }

    // we need to transpose and scale for CPS ordering
    __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity, Float inphase = 1.0) const
    {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
      typedef S<Float,length> structure;
      trove::coalesced_ptr<structure> gauge_((structure*)gauge);
      structure v_ = gauge_[((parity*volumeCB+x)*geometry + dir)];
      for (int i=0; i<Nc; i++)
	for (int j=0; j<Nc; j++)
	  for (int z=0; z<2; z++)
	    v[(i*Nc+j)*2+z] = (RegType)v_.v[(j*Nc+i)*2+z] / anisotropy;
#else
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    v[(i*Nc+j)*2+z] =
	      (RegType)(gauge[((((parity*volumeCB+x)*geometry + dir)*Nc + j)*Nc + i)*2 + z] / anisotropy);
	  }
	}
      }
#endif
    }

    __device__ __host__ inline void save(const RegType v[18], int x, int dir, int parity) {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
      typedef S<Float,length> structure;
      trove::coalesced_ptr<structure> gauge_((structure*)gauge);
      structure v_;
      for (int i=0; i<Nc; i++)
	for (int j=0; j<Nc; j++)
	  for (int z=0; z<2; z++)
	    v_.v[(j*Nc+i)*2+z] = (Float)(anisotropy * v[(i*Nc+j)*2+z]);
      gauge_[((parity*volumeCB+x)*geometry + dir)] = v_;
#else
      for (int i=0; i<Nc; i++) {
	for (int j=0; j<Nc; j++) {
	  for (int z=0; z<2; z++) {
	    gauge[((((parity*volumeCB+x)*geometry + dir)*Nc + j)*Nc + i)*2 + z] =
	      (Float)(anisotropy * v[(i*Nc+j)*2+z]);
	  }
	}
      }
#endif
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
    __device__ __host__ inline gauge_wrapper<RegType, CPSOrder<Float, length>> operator()(int dim, int x_cb, int parity)
    {
      return gauge_wrapper<RegType, CPSOrder<Float, length>>(*this, dim, x_cb, parity);
    }

    /**
       @brief This accessor routine returns a const gauge_wrapper to this object,
       allowing us to overload various operators for manipulating at
       the site level interms of matrix operations.
       @param[in] dir Which dimension are we requesting
       @param[in] x_cb Checkerboarded space-time index we are requesting
       @param[in] parity Parity we are requesting
       @return Instance of a gauge_wrapper that curries in access to
       this field at the above coordinates.
    */
    __device__ __host__ inline const gauge_wrapper<RegType, CPSOrder<Float, length>> operator()(
        int dim, int x_cb, int parity) const
    {
      return gauge_wrapper<RegType, CPSOrder<Float, length>>(
          const_cast<CPSOrder<Float, length> &>(*this), dim, x_cb, parity);
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
      typedef typename mapper<Float>::type RegType;
      Float *gauge;
      const int volumeCB;
      int exVolumeCB; // extended checkerboard volume
      static constexpr int Nc = 3;
    BQCDOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0)
      : LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()), volumeCB(u.VolumeCB()) {
	if (length != 18) errorQuda("Gauge length %d not supported", length);
	// compute volumeCB + halo region
	exVolumeCB = u.X()[0]/2 + 2;
	for (int i=1; i<4; i++) exVolumeCB *= u.X()[i] + 2;
      }
    BQCDOrder(const BQCDOrder &order) : LegacyOrder<Float,length>(order), gauge(order.gauge),
	volumeCB(order.volumeCB), exVolumeCB(order.exVolumeCB) {
	if (length != 18) errorQuda("Gauge length %d not supported", length);
      }

      virtual ~BQCDOrder() { ; }

      // we need to transpose for BQCD ordering
      __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity, Float inphase = 1.0) const
      {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
      typedef S<Float,length> structure;
      trove::coalesced_ptr<structure> gauge_((structure*)gauge);
      structure v_ = gauge_[(dir*2+parity)*exVolumeCB + x];
      for (int i=0; i<Nc; i++)
	for (int j=0; j<Nc; j++)
	  for (int z=0; z<2; z++)
	    v[(i*Nc+j)*2+z] = (RegType)v_.v[(j*Nc+i)*2+z];
#else
	for (int i=0; i<Nc; i++) {
	  for (int j=0; j<Nc; j++) {
	    for (int z=0; z<2; z++) {
	      v[(i*Nc+j)*2+z] = (RegType)gauge[((((dir*2+parity)*exVolumeCB + x)*Nc + j)*Nc + i)*2 + z];
	    }
	  }
	}
#endif
      }

      __device__ __host__ inline void save(const RegType v[18], int x, int dir, int parity) {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	typedef S<Float,length> structure;
	trove::coalesced_ptr<structure> gauge_((structure*)gauge);
	structure v_;
	for (int i=0; i<Nc; i++)
	  for (int j=0; j<Nc; j++)
	    for (int z=0; z<2; z++)
	      v_.v[(j*Nc+i)*2+z] = (Float)(v[(i*Nc+j)*2+z]);
	gauge_[(dir*2+parity)*exVolumeCB + x] = v_;
#else
	for (int i=0; i<Nc; i++) {
	  for (int j=0; j<Nc; j++) {
	    for (int z=0; z<2; z++) {
	      gauge[((((dir*2+parity)*exVolumeCB + x)*Nc + j)*Nc + i)*2 + z] = (Float)v[(i*Nc+j)*2+z];
	    }
	  }
	}
#endif
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
      __device__ __host__ inline gauge_wrapper<RegType, BQCDOrder<Float, length>> operator()(int dim, int x_cb, int parity)
      {
        return gauge_wrapper<RegType, BQCDOrder<Float, length>>(*this, dim, x_cb, parity);
      }

      /**
	 @brief This accessor routine returns a const gauge_wrapper to this object,
	 allowing us to overload various operators for manipulating at
	 the site level interms of matrix operations.
	 @param[in] dir Which dimension are we requesting
	 @param[in] x_cb Checkerboarded space-time index we are requesting
	 @param[in] parity Parity we are requesting
	 @return Instance of a gauge_wrapper that curries in access to
	 this field at the above coordinates.
       */
      __device__ __host__ inline const gauge_wrapper<RegType, BQCDOrder<Float, length>> operator()(
          int dim, int x_cb, int parity) const
      {
        return gauge_wrapper<RegType, BQCDOrder<Float, length>>(
            const_cast<BQCDOrder<Float, length> &>(*this), dim, x_cb, parity);
      }

      size_t Bytes() const { return Nc * Nc * 2 * sizeof(Float); }
    };

    /**
       struct to define TIFR ordered gauge fields:
       [mu][parity][volumecb][col][row]
    */
    template <typename Float, int length> struct TIFROrder : LegacyOrder<Float,length> {
      typedef typename mapper<Float>::type RegType;
      Float *gauge;
      const int volumeCB;
      static constexpr int Nc = 3;
      const Float scale;
    TIFROrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0)
      : LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()),
	volumeCB(u.VolumeCB()), scale(u.Scale()) {
	if (length != 18) errorQuda("Gauge length %d not supported", length);
      }
    TIFROrder(const TIFROrder &order)
      : LegacyOrder<Float,length>(order), gauge(order.gauge), volumeCB(order.volumeCB), scale(order.scale) {
	if (length != 18) errorQuda("Gauge length %d not supported", length);
      }

      virtual ~TIFROrder() { ; }

      // we need to transpose for TIFR ordering
      __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity, Float inphase = 1.0) const
      {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
      typedef S<Float,length> structure;
      trove::coalesced_ptr<structure> gauge_((structure*)gauge);
      structure v_ = gauge_[(dir*2+parity)*volumeCB + x];
      for (int i=0; i<Nc; i++)
	for (int j=0; j<Nc; j++)
	  for (int z=0; z<2; z++)
	    v[(i*Nc+j)*2+z] = (RegType)v_.v[(j*Nc+i)*2+z] / scale;
#else
	for (int i=0; i<Nc; i++) {
	  for (int j=0; j<Nc; j++) {
	    for (int z=0; z<2; z++) {
	      v[(i*Nc+j)*2+z] = (RegType)gauge[((((dir*2+parity)*volumeCB + x)*Nc + j)*Nc + i)*2 + z] / scale;
	    }
	  }
	}
#endif
      }

      __device__ __host__ inline void save(const RegType v[18], int x, int dir, int parity) {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	typedef S<Float,length> structure;
	trove::coalesced_ptr<structure> gauge_((structure*)gauge);
	structure v_;
	for (int i=0; i<Nc; i++)
	  for (int j=0; j<Nc; j++)
	    for (int z=0; z<2; z++)
	      v_.v[(j*Nc+i)*2+z] = (Float)(v[(i*Nc+j)*2+z]) * scale;
	gauge_[(dir*2+parity)*volumeCB + x] = v_;
#else
	for (int i=0; i<Nc; i++) {
	  for (int j=0; j<Nc; j++) {
	    for (int z=0; z<2; z++) {
	      gauge[((((dir*2+parity)*volumeCB + x)*Nc + j)*Nc + i)*2 + z] = (Float)v[(i*Nc+j)*2+z] * scale;
	    }
	  }
	}
#endif
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
      __device__ __host__ inline gauge_wrapper<RegType, TIFROrder<Float, length>> operator()(int dim, int x_cb, int parity)
      {
        return gauge_wrapper<RegType, TIFROrder<Float, length>>(*this, dim, x_cb, parity);
      }

      /**
	 @brief This accessor routine returns a const gauge_wrapper to this object,
	 allowing us to overload various operators for manipulating at
	 the site level interms of matrix operations.
	 @param[in] dir Which dimension are we requesting
	 @param[in] x_cb Checkerboarded space-time index we are requesting
	 @param[in] parity Parity we are requesting
	 @return Instance of a gauge_wrapper that curries in access to
	 this field at the above coordinates.
       */
      __device__ __host__ inline const gauge_wrapper<RegType, TIFROrder<Float, length>> operator()(
          int dim, int x_cb, int parity) const
      {
        return gauge_wrapper<RegType, TIFROrder<Float, length>>(
            const_cast<TIFROrder<Float, length> &>(*this), dim, x_cb, parity);
      }

      size_t Bytes() const { return Nc * Nc * 2 * sizeof(Float); }
    };

    /**
       struct to define TIFR ordered gauge fields (with inlined z halo of depth two):
       [mu][parity][t][z+4][y][x/2][col][row]
    */
    template <typename Float, int length> struct TIFRPaddedOrder : LegacyOrder<Float,length> {
      typedef typename mapper<Float>::type RegType;
      Float *gauge;
      const int volumeCB;
      int exVolumeCB;
      static constexpr int Nc = 3;
      const Float scale;
      const int dim[4];
      const int exDim[4];
    TIFRPaddedOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0)
      : LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()),
	volumeCB(u.VolumeCB()), exVolumeCB(1), scale(u.Scale()),
	dim{ u.X()[0], u.X()[1], u.X()[2], u.X()[3] },
	exDim{ u.X()[0], u.X()[1], u.X()[2] + 4, u.X()[3] } {
	if (length != 18) errorQuda("Gauge length %d not supported", length);

	// exVolumeCB is the padded checkboard volume
	for (int i=0; i<4; i++) exVolumeCB *= exDim[i];
	exVolumeCB /= 2;
      }

    TIFRPaddedOrder(const TIFRPaddedOrder &order)
      : LegacyOrder<Float,length>(order), gauge(order.gauge), volumeCB(order.volumeCB), exVolumeCB(order.exVolumeCB), scale(order.scale),
	  dim{order.dim[0], order.dim[1], order.dim[2], order.dim[3]},
	  exDim{order.exDim[0], order.exDim[1], order.exDim[2], order.exDim[3]} {
	if (length != 18) errorQuda("Gauge length %d not supported", length);
      }

      virtual ~TIFRPaddedOrder() { ; }

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
      __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity, Float inphase = 1.0) const
      {

        int y = getPaddedIndex(x, parity);

#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	typedef S<Float,length> structure;
	trove::coalesced_ptr<structure> gauge_((structure*)gauge);
	structure v_ = gauge_[(dir*2+parity)*exVolumeCB + y];
	for (int i=0; i<Nc; i++)
	  for (int j=0; j<Nc; j++)
	    for (int z=0; z<2; z++)
	      v[(i*Nc+j)*2+z] = (RegType)v_.v[(j*Nc+i)*2+z] / scale;
#else
	for (int i=0; i<Nc; i++) {
	  for (int j=0; j<Nc; j++) {
	    for (int z=0; z<2; z++) {
	      v[(i*Nc+j)*2+z] = (RegType)gauge[((((dir*2+parity)*exVolumeCB + y)*Nc + j)*Nc + i)*2 + z] / scale;
	    }
	  }
	}
#endif
      }

      __device__ __host__ inline void save(const RegType v[18], int x, int dir, int parity) {

	int y = getPaddedIndex(x, parity);

#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	typedef S<Float,length> structure;
	trove::coalesced_ptr<structure> gauge_((structure*)gauge);
	structure v_;
	for (int i=0; i<Nc; i++)
	  for (int j=0; j<Nc; j++)
	    for (int z=0; z<2; z++)
	      v_.v[(j*Nc+i)*2+z] = (Float)(v[(i*Nc+j)*2+z]) * scale;
  gauge_[(dir*2+parity)*exVolumeCB + y] = v_;
#else
	for (int i=0; i<Nc; i++) {
	  for (int j=0; j<Nc; j++) {
	    for (int z=0; z<2; z++) {
	      gauge[((((dir*2+parity)*exVolumeCB + y)*Nc + j)*Nc + i)*2 + z] = (Float)v[(i*Nc+j)*2+z] * scale;
	    }
	  }
	}
#endif
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
      __device__ __host__ inline gauge_wrapper<RegType, TIFRPaddedOrder<Float, length>> operator()(
          int dim, int x_cb, int parity)
      {
        return gauge_wrapper<RegType, TIFRPaddedOrder<Float, length>>(*this, dim, x_cb, parity);
      }

      /**
	 @brief This accessor routine returns a const gauge_wrapper to this object,
	 allowing us to overload various operators for manipulating at
	 the site level interms of matrix operations.
	 @param[in] dir Which dimension are we requesting
	 @param[in] x_cb Checkerboarded space-time index we are requesting
	 @param[in] parity Parity we are requesting
	 @return Instance of a gauge_wrapper that curries in access to
	 this field at the above coordinates.
       */
      __device__ __host__ inline const gauge_wrapper<RegType, TIFRPaddedOrder<Float, length>> operator()(
          int dim, int x_cb, int parity) const
      {
        return gauge_wrapper<RegType, TIFRPaddedOrder<Float, length>>(
            const_cast<TIFRPaddedOrder<Float, length> &>(*this), dim, x_cb, parity);
      }

      size_t Bytes() const { return Nc * Nc * 2 * sizeof(Float); }
    };

  } // namespace gauge

  template <typename otherFloat, typename storeFloat>
    __device__ __host__ inline void complex<double>::operator=(const gauge::fieldorder_wrapper<otherFloat,storeFloat> &a) {
    x = a.real();
    y = a.imag();
  }

  template <typename otherFloat, typename storeFloat>
    __device__ __host__ inline void complex<float>::operator=(const gauge::fieldorder_wrapper<otherFloat,storeFloat> &a) {
    x = a.real();
    y = a.imag();
  }

  template <typename otherFloat, typename storeFloat>
    __device__ __host__ inline complex<double>::complex(const gauge::fieldorder_wrapper<otherFloat,storeFloat> &a) {
    x = a.real();
    y = a.imag();
  }

  template <typename otherFloat, typename storeFloat>
    __device__ __host__ inline complex<float>::complex(const gauge::fieldorder_wrapper<otherFloat,storeFloat> &a) {
    x = a.real();
    y = a.imag();
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
    typedef gauge::FloatNOrder<short, N, 4, 9, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<short, QUDA_RECONSTRUCT_8, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<short, N, 4, 8, stag, huge_alloc, ghostExchange, use_inphase> type;
  };

  // quarter precision
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<char, QUDA_RECONSTRUCT_NO, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<char, N, 2, N, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<char, QUDA_RECONSTRUCT_13, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<char, N, 4, 13, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<char, QUDA_RECONSTRUCT_12, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<char, N, 4, 12, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<char, QUDA_RECONSTRUCT_10, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<char, N, 2, 11, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<char, QUDA_RECONSTRUCT_9, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<char, N, 4, 9, stag, huge_alloc, ghostExchange, use_inphase> type;
  };
  template <int N, QudaStaggeredPhase stag, bool huge_alloc, QudaGhostExchange ghostExchange, bool use_inphase>
  struct gauge_mapper<char, QUDA_RECONSTRUCT_8, N, stag, huge_alloc, ghostExchange, use_inphase, QUDA_NATIVE_GAUGE_ORDER> {
    typedef gauge::FloatNOrder<char, N, 4, 8, stag, huge_alloc, ghostExchange, use_inphase> type;
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
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_BQCD_GAUGE_ORDER,Nc> { typedef gauge::BQCDOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_TIFR_GAUGE_ORDER,Nc> { typedef gauge::TIFROrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_TIFR_PADDED_GAUGE_ORDER,Nc> { typedef gauge::TIFRPaddedOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_FLOAT2_GAUGE_ORDER,Nc> { typedef gauge::FloatNOrder<T, 2*Nc*Nc, 2, 2*Nc*Nc> type; };

} // namespace quda

#endif // _GAUGE_ORDER_H
