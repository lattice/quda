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
#include <tune_quda.h>
#include <assert.h>
#include <register_traits.h>
#include <complex_quda.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <fast_intdiv.h>
#include <type_traits>
#include <atomic.cuh>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

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
      T &gauge;

      /**
	 @brief gauge_wrapper constructor
	 @param[in] gauge Gauge field accessor we are wrapping
	 @param[in] dim Dimension we are accessing
	 @param[in] x_cb Checkerboarded space-time index we are accessing
	 @param[in] parity Parity we are accessing
       */
      __device__ __host__ inline gauge_wrapper<Float,T>(T &gauge, int dim, int x_cb, int parity)
	: gauge(gauge), dim(dim), x_cb(x_cb), parity(parity) { }

      /**
	 @brief Assignment operator with Matrix instance as input
	 @param[in] M Matrix we want to store in this accessot
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
    a.gauge.load((typename RealType<T>::type*)data, a.x_cb, a.dim, a.parity);
  }

  /**
     @brief Assignment operator for the Matrix class with a gauge_wrapper input.
     @param[in] a Input gauge_wrapper that we use to fill in this matrix instance
   */
  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline Matrix<T,N>::Matrix(const gauge_wrapper<typename RealType<T>::type,S> &a) {
    a.gauge.load((typename RealType<T>::type*)data, a.x_cb, a.dim, a.parity);
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
      T &gauge;

      /**
	 @brief gauge_wrapper constructor
	 @param[in] gauge Gauge field accessor we are wrapping
	 @param[in] dim Dimension we are accessing
	 @param[in] ghost_idx Ghost index we are accessing
	 @param[in] parity Parity we are accessing
       */
      __device__ __host__ inline gauge_ghost_wrapper<Float,T>(T &gauge, int dim, int ghost_idx, int parity)
	: gauge(gauge), dim(dim), ghost_idx(ghost_idx), parity(parity) { }

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
    a.gauge.loadGhost((typename RealType<T>::type*)data, a.ghost_idx, a.dim, a.parity);
  }

  /**
     @brief Assignment operator for the Matrix class with a gauge_ghost_wrapper input.
     @param[in] a Input gauge_wrapper that we use to fill in this matrix instance
   */
  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline Matrix<T,N>::Matrix(const gauge_ghost_wrapper<typename RealType<T>::type,S> &a) {
    a.gauge.loadGhost((typename RealType<T>::type*)data, a.ghost_idx, a.dim, a.parity);
  }

  namespace gauge {

    template<typename ReduceType, typename Float> struct square { __host__ __device__ ReduceType operator()(quda::complex<Float> x) { return static_cast<ReduceType>(norm(x)); } };

    template<typename Float, int nColor, QudaGaugeFieldOrder order> struct Accessor {
      mutable complex<Float> dummy;
      Accessor(const GaugeField &, void *gauge_=0, void **ghost_=0) {
	errorQuda("Not implemented for order=%d", order);
      }
      __device__ __host__ complex<Float>& operator()(int d, int parity, int x, int row, int col) const {
	return dummy;
      }
    };

    template<typename Float, int nColor, QudaGaugeFieldOrder order, bool native_ghost>
    struct GhostAccessor {
      mutable complex<Float> dummy;
      GhostAccessor(const GaugeField &, void *gauge_=0, void **ghost_=0) {
	errorQuda("Not implemented for order=%d", order);
      }
      __device__ __host__ complex<Float>& operator()(int d, int parity, int x, int row, int col) const {
	return dummy;
      }
    };

    template<typename Float, int nColor>
      struct Accessor<Float,nColor,QUDA_QDP_GAUGE_ORDER> {
      complex <Float> *u[QUDA_MAX_GEOMETRY];
      const int cb_offset;
    Accessor(const GaugeField &U, void *gauge_=0, void **ghost_=0)
      : cb_offset((U.Bytes()>>1) / (sizeof(complex<Float>)*U.Geometry())) {
	for (int d=0; d<U.Geometry(); d++)
	  u[d] = gauge_ ? static_cast<complex<Float>**>(gauge_)[d] :
	    static_cast<complex<Float>**>(const_cast<void*>(U.Gauge_p()))[d];
      }
    Accessor(const Accessor<Float,nColor,QUDA_QDP_GAUGE_ORDER> &a) : cb_offset(a.cb_offset) {
	for (int d=0; d<QUDA_MAX_GEOMETRY; d++)
	  u[d] = a.u[d];
      }
      __device__ __host__ inline complex<Float>& operator()(int d, int parity, int x, int row, int col) const
      { return u[d][ parity*cb_offset + (x*nColor + row)*nColor + col]; }

      __device__ __host__ inline void atomic_add(int dim, int parity, int x_cb, int row, int col, complex<Float> &val) const {
#ifdef __CUDA_ARCH__
	typedef typename vector<Float,2>::type vec2;
	vec2 *u2 = reinterpret_cast<vec2*>(u[dim] + parity*cb_offset + (x_cb*nColor + row)*nColor + col);
	atomicAdd(u2, (vec2&)val);
#else
	u[dim][ parity*cb_offset + (x_cb*nColor + row)*nColor + col] += val;
#endif
      }

      __host__ double device_norm2(int dim) const {
	errorQuda("Not implemented");
	return 0.0;
      }
    };

    template<typename Float, int nColor, bool native_ghost>
      struct GhostAccessor<Float,nColor,QUDA_QDP_GAUGE_ORDER,native_ghost> {
      complex<Float> *ghost[8];
      int ghostOffset[8];
      GhostAccessor(const GaugeField &U, void *gauge_=0, void **ghost_=0) {
	for (int d=0; d<4; d++) {
	  ghost[d] = ghost_ ? static_cast<complex<Float>*>(ghost_[d]) :
	    static_cast<complex<Float>*>(const_cast<void*>(U.Ghost()[d]));
	  ghostOffset[d] = U.Nface()*U.SurfaceCB(d)*U.Ncolor()*U.Ncolor();

	  ghost[d+4] = (U.Geometry() != QUDA_COARSE_GEOMETRY) ? nullptr :
	    ghost_ ? static_cast<complex<Float>*>(ghost_[d+4]) :
	    static_cast<complex<Float>*>(const_cast<void*>(U.Ghost()[d+4]));
	  ghostOffset[d+4] = U.Nface()*U.SurfaceCB(d)*U.Ncolor()*U.Ncolor();
	}
      }
      GhostAccessor(const GhostAccessor<Float,nColor,QUDA_QDP_GAUGE_ORDER,native_ghost> &a) {
	for (int d=0; d<8; d++) {
	  ghost[d] = a.ghost[d];
	  ghostOffset[d] = a.ghostOffset[d];
	}
      }
      __device__ __host__ inline complex<Float>& operator()(int d, int parity, int x, int row, int col) const
      { return ghost[d][ parity*ghostOffset[d] + (x*nColor + row)*nColor + col]; }
    };

    template<typename Float, int nColor>
      struct Accessor<Float,nColor,QUDA_MILC_GAUGE_ORDER> {
      complex<Float> *u;
      const int volumeCB;
      const int geometry;
    Accessor(const GaugeField &U, void *gauge_=0, void **ghost_=0)
      : u(gauge_ ? static_cast<complex<Float>*>(gauge_) :
	  static_cast<complex<Float>*>(const_cast<void *>(U.Gauge_p()))),
	volumeCB(U.VolumeCB()), geometry(U.Geometry()) { }
    Accessor(const Accessor<Float,nColor,QUDA_MILC_GAUGE_ORDER> &a)
      : u(a.u), volumeCB(a.volumeCB), geometry(a.geometry) { }
      __device__ __host__ inline complex<Float>& operator()(int d, int parity, int x, int row, int col) const
      { return 	u[(((parity*volumeCB+x)*geometry + d)*nColor + row)*nColor + col]; }

      __device__ __host__ inline void atomic_add(int dim, int parity, int x_cb, int row, int col, complex<Float> &val) const {
#ifdef __CUDA_ARCH__
	typedef typename vector<Float,2>::type vec2;
	vec2 *u2 = reinterpret_cast<vec2*>(u + (((parity*volumeCB+x_cb)*geometry + dim)*nColor + row)*nColor + col);
	atomicAdd(u2, (vec2&)val);
#else
	u[(((parity*volumeCB+x_cb)*geometry + dim)*nColor + row)*nColor + col] += val;
#endif
      }

      __host__ double device_norm2(int dim) const {
	errorQuda("Not implemented");
	return 0.0;
      }
    };

    template<typename Float, int nColor, bool native_ghost>
      struct GhostAccessor<Float,nColor,QUDA_MILC_GAUGE_ORDER,native_ghost> {
      complex<Float> *ghost[8];
      int ghostOffset[8];
      GhostAccessor(const GaugeField &U, void *gauge_=0, void **ghost_=0) {
	for (int d=0; d<4; d++) {
	  ghost[d] = ghost_ ? static_cast<complex<Float>*>(ghost_[d]) :
	    static_cast<complex<Float>*>(const_cast<void*>(U.Ghost()[d]));
	  ghostOffset[d] = U.Nface()*U.SurfaceCB(d)*U.Ncolor()*U.Ncolor();

	  ghost[d+4] = (U.Geometry() != QUDA_COARSE_GEOMETRY) ? nullptr :
	    ghost_ ? static_cast<complex<Float>*>(ghost_[d+4]) :
	    static_cast<complex<Float>*>(const_cast<void*>(U.Ghost()[d+4]));
	  ghostOffset[d+4] = U.Nface()*U.SurfaceCB(d)*U.Ncolor()*U.Ncolor();
	}
      }
      GhostAccessor(const GhostAccessor<Float,nColor,QUDA_MILC_GAUGE_ORDER,native_ghost> &a) {
	for (int d=0; d<8; d++) {
	  ghost[d] = a.ghost[d];
	  ghostOffset[d] = a.ghostOffset[d];
	}
      }
      __device__ __host__ inline complex<Float>& operator()(int d, int parity, int x, int row, int col) const
      { return ghost[d][ parity*ghostOffset[d] + (x*nColor + row)*nColor + col]; }
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

    template<typename Float, int nColor>
      struct Accessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER> {
      complex<Float> *u;
      const int offset_cb;
      const int stride;
      const int geometry;
    Accessor(const GaugeField &U, void *gauge_=0, void **ghost_=0)
      : u(gauge_ ? static_cast<complex<Float>*>(gauge_) :
	  static_cast<complex<Float>*>(const_cast<void*>(U.Gauge_p()))),
	offset_cb( (U.Bytes()>>1) / sizeof(complex<Float>)), stride(U.Stride()), geometry(U.Geometry())
	{  }
    Accessor(const Accessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER> &a)
      : u(a.u), offset_cb(a.offset_cb), stride(a.stride), geometry(a.geometry) {  }

      __device__ __host__ inline complex<Float>& operator()(int dim, int parity, int x_cb, int row, int col) const
      { return u[parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb]; }

      __device__ __host__ void atomic_add(int dim, int parity, int x_cb, int row, int col, complex<Float> &val) const {
#ifdef __CUDA_ARCH__
	typedef typename vector<Float,2>::type vec2;
	vec2 *u2 = reinterpret_cast<vec2*>(u + parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb);
	atomicAdd(u2, (vec2&)val);
#else
	u[parity*offset_cb + dim*stride*nColor*nColor + (row*nColor+col)*stride + x_cb] += val;
#endif
      }

      __host__ double device_norm2(int dim) const {
	if (dim >= geometry) errorQuda("Request dimension %d exceeds dimensionality of the field %d", dim, geometry);
	thrust::device_ptr<complex<Float> > ptr(u);
	double even = thrust::transform_reduce(ptr+0*offset_cb+(dim+0)*stride*nColor*nColor,
					       ptr+0*offset_cb+(dim+1)*stride*nColor*nColor,
					       square<double,Float>(), 0.0, thrust::plus<double>());
	double odd  = thrust::transform_reduce(ptr+1*offset_cb+(dim+0)*stride*nColor*nColor,
					       ptr+1*offset_cb+(dim+1)*stride*nColor*nColor,
					       square<double,Float>(), 0.0, thrust::plus<double>());
	return even + odd;
      }
    };

    template<typename Float, int nColor, bool native_ghost>
      struct GhostAccessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER,native_ghost> {
      complex<Float> *ghost[8];
      const int volumeCB;
      int ghostVolumeCB[8];
      Accessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER> accessor;
    GhostAccessor(const GaugeField &U, void *gauge_, void **ghost_=0)
      : volumeCB(U.VolumeCB()), accessor(U, gauge_, ghost_)
      {
	if (!native_ghost) assert(ghost_ != nullptr);
	for (int d=0; d<4; d++) {
	  ghost[d] = !native_ghost ? static_cast<complex<Float>*>(ghost_[d]) : nullptr;
	  ghostVolumeCB[d] = U.Nface()*U.SurfaceCB(d);
	  ghost[d+4] = !native_ghost && U.Geometry() == QUDA_COARSE_GEOMETRY? static_cast<complex<Float>*>(ghost_[d+4]) : nullptr;
	  ghostVolumeCB[d+4] = U.Nface()*U.SurfaceCB(d);
	}
      }
    GhostAccessor(const GhostAccessor<Float,nColor,QUDA_FLOAT2_GAUGE_ORDER,native_ghost> &a)
      : volumeCB(a.volumeCB), accessor(a.accessor)
      {
	for (int d=0; d<8; d++) {
	  ghost[d] = a.ghost[d];
	  ghostVolumeCB[d] = a.ghostVolumeCB[d];
	}
      }
      __device__ __host__ inline complex<Float>& operator()(int d, int parity, int x_cb, int row, int col) const
      {
	if (native_ghost)
	  return accessor(d%4, parity, x_cb+(d/4)*ghostVolumeCB[d]+volumeCB, row, col);
	else
	  return ghost[d][ ((parity*nColor + row)*nColor+col)*ghostVolumeCB[d] + x_cb ];
      }
    };


    /**
       This is a template driven generic gauge field accessor.  To
       deploy for a specifc field ordering, the two operator()
       accessors have to be specialized for that ordering.
     */
  template <typename Float, int nColor, int nSpinCoarse, QudaGaugeFieldOrder order, bool native_ghost=true>
      struct FieldOrder {

	/** An internal reference to the actual field we are accessing */
	const int volumeCB;
	const int nDim;
	const int geometry;
	static constexpr int nColorCoarse = nColor / nSpinCoarse;
	QudaFieldLocation location;

	const Accessor<Float,nColor,order> accessor;
	const GhostAccessor<Float,nColor,order,native_ghost> ghostAccessor;

      public:
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
	  nDim(o.nDim), geometry(o.geometry),
	  accessor(o.accessor), ghostAccessor(o.ghostAccessor)
	{ }

	virtual ~FieldOrder() { ; }

	/**
	 * Read-only complex-member accessor function
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ const complex<Float>& operator()(int d, int parity, int x, int row, int col) const
	{ return accessor(d,parity,x,row,col); }

	/**
	 * Writable complex-member accessor function
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ complex<Float>& operator() (int d, int parity, int x, int row, int col)
	{ return accessor(d,parity,x,row,col); }

	/**
	 * Read-only complex-member accessor function for the ghost zone
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ const complex<Float>& Ghost(int d, int parity, int x, int row, int col) const
	{ return ghostAccessor(d,parity,x,row,col); }

	/**
	 * Writable complex-member accessor function for the ghost zone
	 * @param d dimension index
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param row row index
	 * @param c column index
	 */
	__device__ __host__ complex<Float>& Ghost(int d, int parity, int x, int row, int col)
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
	__device__ __host__ inline const complex<Float>& operator()(int d, int parity, int x, int s_row,
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
	__device__ __host__ inline complex<Float>& operator()(int d, int parity, int x, int s_row,
							     int s_col, int c_row, int c_col) {
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
	__device__ __host__ inline const complex<Float>& Ghost(int d, int parity, int x, int s_row,
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
	__device__ __host__ inline complex<Float>& Ghost(int d, int parity, int x, int s_row,
							     int s_col, int c_row, int c_col) {
	  return Ghost(d, parity, x, s_row*nColorCoarse + c_row, s_col*nColorCoarse + c_col);
	}

	__device__ __host__ inline void atomicAdd(int d, int parity, int x, int s_row, int s_col,
						  int c_row, int c_col, complex<Float> &val) {
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
	 * @brief Returns the L2 norm squared of the field in a given dimension
	 * @param[in] dim Which dimension we are taking the norm of
	 * @return L2 norm squared
	 */
	__host__ double norm2(int dim) const {
	  double nrm2 = 0;
	  if (location == QUDA_CUDA_FIELD_LOCATION) {
	    // call device version - specialized for ordering
	    nrm2 = accessor.device_norm2(dim);
	  } else {
	    // do simple norm on host memory
	    for (int parity=0; parity<2; parity++)
	      for (int x_cb=0; x_cb<volumeCB; x_cb++) {
		for (int row=0; row<nColor; row++)
		  for (int col=0; col<nColor; col++)
		    nrm2 += norm((*this)(dim,parity,x_cb,row,col));
	      }
	  }
	  comm_allreduce(&nrm2);
	  return nrm2;
	}

	/** Return the size of the allocation (geometry and parity left out and added as needed in Tunable::bytes) */
	size_t Bytes() const { return static_cast<size_t>(volumeCB) * nColor * nColor * 2ll * sizeof(Float); }
      };


    /** Generic reconstruction is no reconstruction */
    template <int N, typename Float>
      struct Reconstruct {
      typedef typename mapper<Float>::type RegType;
    Reconstruct(const GaugeField &u) { }
    Reconstruct(const Reconstruct<N,Float> &recon) { }

      __device__ __host__ inline void Pack(RegType out[N], const RegType in[N], int idx ) const {
#pragma unroll
	for (int i=0; i<N; i++) out[i] = in[i];
      }
      template<typename I>
      __device__ __host__ inline void Unpack(RegType out[N], const RegType in[N], int idx, int dir,
					       const RegType phase, const I *X, const int *R) const {
#pragma unroll
	for (int i=0; i<N; i++) out[i] = in[i];
      }
      __device__ __host__ inline RegType getPhase(const RegType in[N]) const { return 0; }
    };

    /** No reconstruction but we scale the result. This is used for
	half-precision non-unitary fields, e.g., staggered fat link */
    template <typename Float>
      struct Reconstruct<19,Float> {
      typedef typename mapper<Float>::type RegType;
      RegType scale;
    Reconstruct(const GaugeField &u) : scale(u.LinkMax()) { }
    Reconstruct(const Reconstruct<19,Float> &recon) : scale(recon.scale) { }

      __device__ __host__ inline void Pack(RegType out[18], const RegType in[18], int idx) const {
#pragma unroll
	for (int i=0; i<18; i++) out[i] = in[i] / scale;
      }
      template<typename I>
      __device__ __host__ inline void Unpack(RegType out[18], const RegType in[18], int idx, int dir,
					     const RegType phase, const I *X, const int *R) const {
#pragma unroll
	for (int i=0; i<18; i++) out[i] = scale * in[i];
      }
      __device__ __host__ inline RegType getPhase(const RegType in[18]) const { return 0; }
    };

      /**
	 timeBoundary variant for extended gauge field
	 @param idx extended field linear index
	 @param X the gauge field dimensions
	 @param R the radii dimenions of the extended region
	 @param tBoundary the boundary condition
	 @param isFirstTimeSlice if we're on the first time slice of nodes
	 @param isLastTimeSlide if we're on the last time slice of nodes
	 @param ghostExchange if the field is extended or not (determines indexing type)
      */
      template <typename Float, typename I>
      __device__ __host__ inline Float timeBoundary(int idx, const I X[QUDA_MAX_DIM], const int R[QUDA_MAX_DIM],
						    QudaTboundary tBoundary, bool isFirstTimeSlice, bool isLastTimeSlice,
						    QudaGhostExchange ghostExchange=QUDA_GHOST_EXCHANGE_NO) {
	if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED) {
	  if ( idx >= X[3]*X[2]*X[1]*X[0]/2 ) { // halo region on the first time slice
	    return isFirstTimeSlice ? static_cast<Float>(tBoundary) : static_cast<Float>(1.0);
	  } else if ( idx >= (X[3]-1)*X[0]*X[1]*X[2]/2 ) { // last link on the last time slice
	    return isLastTimeSlice ? static_cast<Float>(tBoundary) : static_cast<Float>(1.0);
	  } else {
	    return static_cast<Float>(1.0);
	  }
	} else {
	  if ( idx >= (R[3]-1)*X[0]*X[1]*X[2]/2 && idx < R[3]*X[0]*X[1]*X[2]/2 ) {
	    // the boundary condition is on the R[3]-1 time slice
	    return isFirstTimeSlice ? static_cast<Float>(tBoundary) : static_cast<Float>(1.0);
	  } else if ( idx >= (X[3]-R[3]-1)*X[0]*X[1]*X[2]/2 && idx < (X[3]-R[3])*X[0]*X[1]*X[2]/2 ) {
	    // the boundary condition lies on the X[3]-R[3]-1 time slice
	    return isLastTimeSlice ? static_cast<Float>(tBoundary) : static_cast<Float>(1.0);
	  } else {
	    return static_cast<Float>(1.0);
	  }
	}
      }

      // not actually used - here for reference
      template <typename Float, typename I>
      __device__ __host__ inline Float milcStaggeredPhase(int dim, const int x[], const I R[]) {
	// could consider non-extended varient too?
	Float sign = static_cast<Float>(1.0);
	switch (dim) {
	case 0: if ( ((x[3] - R[3]) & 1) != 0)                             sign = -static_cast<Float>(1.0); break;
	case 1: if ( ((x[0] - R[0] + x[3] - R[3]) & 1) != 0)               sign = -static_cast<Float>(1.0); break;
	case 2: if ( ((x[0] - R[0] + x[1] - R[1] + x[3] - R[3]) & 1) != 0) sign = -static_cast<Float>(1.0); break;
	}
	return sign;
      }

      template <typename Float>
      struct Reconstruct<12,Float> {
	typedef typename mapper<Float>::type RegType;
	typedef complex<RegType> Complex;
	const RegType anisotropy;
	const QudaTboundary tBoundary;
	bool isFirstTimeSlice;
	bool isLastTimeSlice;
	QudaGhostExchange ghostExchange;

      Reconstruct(const GaugeField &u) : anisotropy(u.Anisotropy()), tBoundary(u.TBoundary()),
	  isFirstTimeSlice(comm_coord(3) == 0 ? true : false),
	  isLastTimeSlice(comm_coord(3) == comm_dim(3)-1 ? true : false),
	  ghostExchange(u.GhostExchange()) { }

      Reconstruct(const Reconstruct<12,Float> &recon) : anisotropy(recon.anisotropy),
	  tBoundary(recon.tBoundary), isFirstTimeSlice(recon.isFirstTimeSlice),
	  isLastTimeSlice(recon.isLastTimeSlice), ghostExchange(recon.ghostExchange) { }

	__device__ __host__ inline void Pack(RegType out[12], const RegType in[18], int idx) const {
#pragma unroll
	  for (int i=0; i<12; i++) out[i] = in[i];
	}

	template<typename I>
	__device__ __host__ inline void Unpack(RegType out[18], const RegType in[12], int idx, int dir,
					       const RegType phase, const I *X, const int *R) const {
	  const Complex *In = reinterpret_cast<const Complex*>(in);
	  Complex *Out = reinterpret_cast<Complex*>(out);

	  const RegType u0 = dir < 3 ? anisotropy :
	    timeBoundary<RegType>(idx, X, R, tBoundary,isFirstTimeSlice, isLastTimeSlice, ghostExchange);

#pragma unroll
	  for(int i=0; i<6; ++i) Out[i] = In[i];

	  Out[6] = u0*conj(Out[1]*Out[5] - Out[2]*Out[4]);
	  Out[7] = u0*conj(Out[2]*Out[3] - Out[0]*Out[5]);
	  Out[8] = u0*conj(Out[0]*Out[4] - Out[1]*Out[3]);
	}

	__device__ __host__ inline RegType getPhase(const RegType in[18]) { return 0; }
      };

      // FIX ME - 11 is a misnomer to avoid confusion in template instantiation
      template <typename Float>
      struct Reconstruct<11,Float> {
	typedef typename mapper<Float>::type RegType;

	Reconstruct(const GaugeField &u) { ; }
	Reconstruct(const Reconstruct<11,Float> &recon) { }

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

      template <typename Float>
	struct Reconstruct<13,Float> {
	typedef typename mapper<Float>::type RegType;
	typedef complex<RegType> Complex;
	const Reconstruct<12,Float> reconstruct_12;
	const RegType scale;

      Reconstruct(const GaugeField &u) : reconstruct_12(u), scale(u.Scale()) { }
      Reconstruct(const Reconstruct<13,Float> &recon) : reconstruct_12(recon.reconstruct_12),
	  scale(recon.scale) { }

	__device__ __host__ inline void Pack(RegType out[12], const RegType in[18], int idx) const {
	  reconstruct_12.Pack(out, in, idx);
      }

      template<typename I>
      __device__ __host__ inline void Unpack(RegType out[18], const RegType in[12], int idx, int dir,
					     const RegType phase, const I *X, const int *R) const {
	const Complex *In = reinterpret_cast<const Complex*>(in);
	Complex *Out = reinterpret_cast<Complex*>(out);
	const RegType coeff = static_cast<RegType>(1.0)/scale;

#pragma unroll
	for(int i=0; i<6; ++i) Out[i] = In[i];

	Out[6] = coeff*conj(Out[1]*Out[5] - Out[2]*Out[4]);
	Out[7] = coeff*conj(Out[2]*Out[3] - Out[0]*Out[5]);
	Out[8] = coeff*conj(Out[0]*Out[4] - Out[1]*Out[3]);

	// Multiply the third row by exp(I*3*phase), since the cross product will end up in a scale factor of exp(-I*2*phase)
	RegType cos_sin[2];
	Trig<isHalf<RegType>::value,RegType>::SinCos(static_cast<RegType>(3.*phase), &cos_sin[1], &cos_sin[0]);
	Complex A(cos_sin[0], cos_sin[1]);

	Out[6] *= A;
	Out[7] *= A;
	Out[8] *= A;
      }

      __device__ __host__ inline RegType getPhase(const RegType in[18]) const {
#if 1 // phase from cross product
	const Complex *In = reinterpret_cast<const Complex*>(in);
	// denominator = (U[0][0]*U[1][1] - U[0][1]*U[1][0])*
	Complex denom = conj(In[0]*In[4] - In[1]*In[3]) / scale;
	Complex expI3Phase = In[8] / denom; // numerator = U[2][2]
	RegType phase = arg(expI3Phase)/static_cast<RegType>(3.0);
#else // phase from determinant
	Matrix<Complex,3> a;
#pragma unroll
	for (int i=0; i<9; i++) a(i) = Complex(in[2*i]/scale, in[2*i+1]/scale);
	const Complex det = getDeterminant( a );
	RegType phase = arg(det)/3;
#endif
	return phase;
      }

    };


    template <typename Float>
    struct Reconstruct<8,Float> {
    typedef typename mapper<Float>::type RegType;
    typedef complex<RegType> Complex;
    const RegType anisotropy;
    const QudaTboundary tBoundary;
    bool isFirstTimeSlice;
    bool isLastTimeSlice;
    QudaGhostExchange ghostExchange;

    Reconstruct(const GaugeField &u) : anisotropy(u.Anisotropy()), tBoundary(u.TBoundary()),
      isFirstTimeSlice(comm_coord(3) == 0 ? true : false),
      isLastTimeSlice(comm_coord(3) == comm_dim(3)-1 ? true : false),
      ghostExchange(u.GhostExchange()) { }

    Reconstruct(const Reconstruct<8,Float> &recon) : anisotropy(recon.anisotropy),
      tBoundary(recon.tBoundary), isFirstTimeSlice(recon.isFirstTimeSlice),
      isLastTimeSlice(recon.isLastTimeSlice), ghostExchange(recon.ghostExchange) { }

    __device__ __host__ inline void Pack(RegType out[8], const RegType in[18], int idx) const {
      out[0] = Trig<isHalf<Float>::value,RegType>::Atan2(in[1], in[0]);
      out[1] = Trig<isHalf<Float>::value,RegType>::Atan2(in[13], in[12]);
#pragma unroll
      for (int i=2; i<8; i++) out[i] = in[i];
    }

    template<typename I>
    __device__ __host__ inline void Unpack(RegType out[18], const RegType in[8], int idx, int dir, const RegType phase,
					   const I *X, const int *R, const RegType scale=1.0) const {
      const Complex *In = reinterpret_cast<const Complex*>(in);
      Complex *Out = reinterpret_cast<Complex*>(out);

      // First reconstruct first row
      Out[1] = In[1];
      Out[2] = In[2];
      RegType row_sum = norm(Out[1]) + norm(Out[2]);

      RegType u0 = dir < 3 ? anisotropy :
	timeBoundary<RegType>(idx, X, R, tBoundary,isFirstTimeSlice, isLastTimeSlice, ghostExchange);
      u0 *= scale;

      RegType diff = static_cast<RegType>(1.0)/(u0*u0) - row_sum;
      RegType U00_mag = sqrt(diff >= static_cast<RegType>(0.0) ? diff : static_cast<RegType>(0.0));

      Out[0] = U00_mag * Complex(Trig<isHalf<Float>::value,RegType>::Cos(in[0]), Trig<isHalf<Float>::value,RegType>::Sin(in[0]));

      // Now reconstruct first column
      Out[3] = In[3];
      RegType column_sum = norm(Out[0]) + norm(Out[3]);

      diff = static_cast<RegType>(1.0)/(u0*u0) - column_sum;
      RegType U20_mag = sqrt(diff >= static_cast<RegType>(0.0) ? diff : static_cast<RegType>(0.0));

      Out[6] = U20_mag * Complex( Trig<isHalf<Float>::value,RegType>::Cos(in[1]), Trig<isHalf<Float>::value,RegType>::Sin(in[1]));
      // First column now restored

      // finally reconstruct last elements from SU(2) rotation
      RegType r_inv2 = static_cast<RegType>(1.0)/(u0*row_sum);

      Complex A = conj(Out[0])*Out[3];
      Out[4] = -(conj(Out[6])*conj(Out[2]) + u0*A*Out[1])*r_inv2; // U11
      Out[5] = (conj(Out[6])*conj(Out[1]) - u0*A*Out[2])*r_inv2;  // U12

      A = conj(Out[0])*Out[6];
      Out[7] = (conj(Out[3])*conj(Out[2]) - u0*A*Out[1])*r_inv2;  // U21
      Out[8] = -(conj(Out[3])*conj(Out[1]) + u0*A*Out[2])*r_inv2; // U12
    }

    __device__ __host__ inline RegType getPhase(const RegType in[18]){ return 0; }
  };


    template <typename Float>
      struct Reconstruct<9,Float> {
      typedef typename mapper<Float>::type RegType;
      typedef complex<RegType> Complex;
      const Reconstruct<8,Float> reconstruct_8;
      const RegType scale;

    Reconstruct(const GaugeField &u) : reconstruct_8(u), scale(u.Scale()) {}

    Reconstruct(const Reconstruct<9,Float> &recon) : reconstruct_8(recon.reconstruct_8),
	scale(recon.scale) { }

      __device__ __host__ inline RegType getPhase(const RegType in[18]) const {
#if 1 // phase from cross product
	const Complex *In = reinterpret_cast<const Complex*>(in);
	// denominator = (U[0][0]*U[1][1] - U[0][1]*U[1][0])*
	Complex denom = conj(In[0]*In[4] - In[1]*In[3]) / scale;
	Complex expI3Phase = In[8] / denom; // numerator = U[2][2]
	RegType phase = arg(expI3Phase)/static_cast<RegType>(3.0);
#else // phase from determinant
	Matrix<Complex,3> a;
#pragma unroll
	for (int i=0; i<9; i++) a(i) = Complex(in[2*i]/scale, in[2*i+1]/scale);
	const Complex det = getDeterminant( a );
	RegType phase = arg(det)/3;
#endif
	return phase;
      }

	// Rescale the U3 input matrix by exp(-I*phase) to obtain an SU3 matrix multiplied by a real scale factor,
      __device__ __host__ inline void Pack(RegType out[8], const RegType in[18], int idx) const {
	RegType phase = getPhase(in);
	RegType cos_sin[2];
	Trig<isHalf<RegType>::value,RegType>::SinCos(static_cast<RegType>(-phase), &cos_sin[1], &cos_sin[0]);
	Complex z(cos_sin[0], cos_sin[1]);
	Complex su3[9];
#pragma unroll
	for (int i=0; i<9; i++) su3[i] = z * reinterpret_cast<const Complex*>(in)[i];
	reconstruct_8.Pack(out, reinterpret_cast<RegType*>(su3), idx);
      }

      template<typename I>
      __device__ __host__ inline void Unpack(RegType out[18], const RegType in[8], int idx, int dir,
					     const RegType phase, const I *X, const int *R) const {
	reconstruct_8.Unpack(out, in, idx, dir, phase, X, R, scale);
	RegType cos_sin[2];
	Trig<isHalf<RegType>::value,RegType>::SinCos(static_cast<RegType>(phase), &cos_sin[1], &cos_sin[0]);
	Complex z(cos_sin[0], cos_sin[1]);
#pragma unroll
	for (int i=0; i<9; i++) reinterpret_cast<Complex*>(out)[i] *= z;
      }

    };

  __host__ __device__ inline constexpr int ct_sqrt(int n, int i = 1){
    return n == i ? n : (i * i < n ? ct_sqrt(n, i + 1) : i);
  }

  /**
     @brief Return the number of colors of the accessor based on the length of the field
     @param[in] length Number of real numbers per link
     @return Number of colors (=sqrt(length/2))
   */
  __host__ __device__ inline constexpr int Ncolor(int length) { return ct_sqrt(length/2); }

  // we default to huge allocations for gauge field (for now)
  constexpr bool default_huge_alloc = true;

  template <typename Float, int length, int N, int reconLenParam, QudaStaggeredPhase stag_phase=QUDA_STAGGERED_PHASE_NO, bool huge_alloc=default_huge_alloc>
    struct FloatNOrder {
      typedef typename mapper<Float>::type RegType;
      typedef typename VectorType<Float,N>::type Vector;
      typedef typename AllocType<huge_alloc>::type AllocInt;
      Reconstruct<reconLenParam,Float> reconstruct;
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

	static_assert( !(stag_phase!=QUDA_STAGGERED_PHASE_NO && reconLenParam != 18 && reconLenParam != 12),
		       "staggered phase only presently supported for 18 and 12 reconstruct");
	for (int i=0; i<4; i++) {
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

      __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity) const {
        const int M = reconLen / N;
        RegType tmp[reconLen];

#pragma unroll
        for (int i=0; i<M; i++){
	  // first do vectorized copy from memory
#if defined(USE_TEXTURE_OBJECTS) && defined(__CUDA_ARCH__)
	  if (!huge_alloc) { // use textures unless we have a huge alloc
	    TexVector vecTmp = tex1Dfetch<TexVector>(tex, parity*tex_offset + dir*stride*M + stride*i + x);
#pragma unroll
	    for (int j=0; j<N; j++) copy(tmp[i*N+j], reinterpret_cast<RegType*>(&vecTmp)[j]);
	  } else
#endif
	  {
	    Vector vecTmp = vector_load<Vector>(gauge + parity*offset, dir*stride*M + stride*i + x);
	    // second do copy converting into register type
#pragma unroll
	    for (int j=0; j<N; j++) copy(tmp[i*N+j], reinterpret_cast<Float*>(&vecTmp)[j]);
	  }
	}

        RegType phase = 0.; // TODO - add texture support for phases
	if (hasPhase) copy(phase, (gauge+parity*offset)[phaseOffset/sizeof(Float) + stride*dir + x]);

        // The phases come after the ghost matrices
        reconstruct.Unpack(v, tmp, x, dir, 2.*M_PI*phase, X, R);

	// FIXME - this is a hack from hell - needs to be moved into the reconstruct type
	if (stag_phase == QUDA_STAGGERED_PHASE_MILC && reconLenParam == 12) {
	  Float sign = (dir == 0 && ((coords[3] - R[3]) & 1) != 0) ||
	    ( dir == 1 && ((coords[0] - R[0] + coords[3] - R[3]) & 1) != 0) ||
	    ( dir == 2 && ((coords[0] - R[0] + coords[1] - R[1] + coords[3] - R[3]) & 1) != 0) ? -1.0 : 1.0;

#pragma unroll
	  for (int i=12; i<18; i++) v[i] *= sign;
	}
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
	  vector_store(gauge + parity*offset, x + dir*stride*M + stride*i, vecTmp);
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
      __device__ __host__ inline gauge_wrapper<Float,FloatNOrder<Float,length,N,reconLenParam,stag_phase,huge_alloc> >
	   operator()(int dim, int x_cb, int parity) {
	return gauge_wrapper<Float,FloatNOrder<Float,length,N,reconLenParam,stag_phase,huge_alloc> >(*this, dim, x_cb, parity);
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
      __device__ __host__ inline const gauge_wrapper<Float,FloatNOrder<Float,length,N,reconLenParam,stag_phase,huge_alloc> >
	   operator()(int dim, int x_cb, int parity) const {
	return gauge_wrapper<Float,FloatNOrder<Float,length,N,reconLenParam,stag_phase,huge_alloc> >
	(const_cast<FloatNOrder<Float,length,N,reconLenParam,stag_phase,huge_alloc>&>(*this), dim, x_cb, parity);
      }

      __device__ __host__ inline void loadGhost(RegType v[length], int x, int dir, int parity) const {
        if (!ghost[dir]) { // load from main field not separate array
          load(v, volumeCB+x, dir, parity); // an offset of size volumeCB puts us at the padded region
          // This also works perfectly when phases are stored. No need to change this.
        } else {
          const int M = reconLen / N;
          RegType tmp[reconLen];

#pragma unroll
          for (int i=0; i<M; i++) {
	    // first do vectorized copy from memory into registers
	    Vector vecTmp = vector_load<Vector>(ghost[dir]+parity*faceVolumeCB[dir]*(M*N + hasPhase),
						i*faceVolumeCB[dir]+x);
	    // second do copy converting into register type
#pragma unroll
	    for (int j=0; j<N; j++) copy(tmp[i*N+j], reinterpret_cast<Float*>(&vecTmp)[j]);
          }
          RegType phase=0.;
          if(hasPhase) copy(phase, ghost[dir][parity*faceVolumeCB[dir]*(M*N + 1) + faceVolumeCB[dir]*M*N + x]);
          reconstruct.Unpack(v, tmp, x, dir, 2.*M_PI*phase, X, R);
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
      __device__ __host__ inline gauge_ghost_wrapper<Float,FloatNOrder<Float,length,N,reconLenParam,stag_phase,huge_alloc> >
	   Ghost(int dim, int ghost_idx, int parity) {
	return gauge_ghost_wrapper<Float,FloatNOrder<Float,length,N,reconLenParam,stag_phase,huge_alloc> >(*this, dim, ghost_idx, parity);
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
      __device__ __host__ inline const gauge_ghost_wrapper<Float,FloatNOrder<Float,length,N,reconLenParam,stag_phase,huge_alloc> >
	   Ghost(int dim, int ghost_idx, int parity) const {
	return gauge_ghost_wrapper<Float,FloatNOrder<Float,length,N,reconLenParam,stag_phase,huge_alloc> >
	(const_cast<FloatNOrder<Float,length,N,reconLenParam,stag_phase,huge_alloc>&>(*this), dim, ghost_idx, parity);
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

      __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity) const {
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
      __device__ __host__ inline gauge_wrapper<Float,QDPOrder<Float,length> >
	   operator()(int dim, int x_cb, int parity) {
	return gauge_wrapper<Float,QDPOrder<Float,length> >(*this, dim, x_cb, parity);
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
      __device__ __host__ inline const gauge_wrapper<Float,QDPOrder<Float,length> >
	   operator()(int dim, int x_cb, int parity) const {
	return gauge_wrapper<Float,QDPOrder<Float,length> >
	(const_cast<QDPOrder<Float,length>&>(*this), dim, x_cb, parity);
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

      __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity) const {
	for (int i=0; i<length; i++) {
	  int z = i%2;
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
      __device__ __host__ inline gauge_wrapper<Float,QDPJITOrder<Float,length> >
	   operator()(int dim, int x_cb, int parity) {
	return gauge_wrapper<Float,QDPJITOrder<Float,length> >(*this, dim, x_cb, parity);
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
      __device__ __host__ inline const gauge_wrapper<Float,QDPJITOrder<Float,length> >
	   operator()(int dim, int x_cb, int parity) const {
	return gauge_wrapper<Float,QDPJITOrder<Float,length> >
	(const_cast<QDPJITOrder<Float,length>&>(*this), dim, x_cb, parity);
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

    __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity) const {
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
    __device__ __host__ inline gauge_wrapper<Float,MILCOrder<Float,length> >
      operator()(int dim, int x_cb, int parity) {
      return gauge_wrapper<Float,MILCOrder<Float,length> >(*this, dim, x_cb, parity);
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
    __device__ __host__ inline const gauge_wrapper<Float,MILCOrder<Float,length> >
      operator()(int dim, int x_cb, int parity) const {
      return gauge_wrapper<Float,MILCOrder<Float,length> >
	(const_cast<MILCOrder<Float,length>&>(*this), dim, x_cb, parity);
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
  MILCSiteOrder(const GaugeField &u, Float *gauge_=0, Float **ghost_=0) :
    LegacyOrder<Float,length>(u, ghost_), gauge(gauge_ ? gauge_ : (Float*)u.Gauge_p()),
      volumeCB(u.VolumeCB()), geometry(u.Geometry()),
      offset(u.SiteOffset()), size(u.SiteSize()) { ; }
  MILCSiteOrder(const MILCSiteOrder &order) : LegacyOrder<Float,length>(order),
      gauge(order.gauge), volumeCB(order.volumeCB), geometry(order.geometry),
      offset(order.offset), size(order.size)
      { ; }
    virtual ~MILCSiteOrder() { ; }

    __device__ __host__ inline void load(RegType v[length], int x, int dir, int parity) const {
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
    __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity) const {
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
    __device__ __host__ inline gauge_wrapper<Float,CPSOrder<Float,length> >
      operator()(int dim, int x_cb, int parity) {
      return gauge_wrapper<Float,CPSOrder<Float,length> >(*this, dim, x_cb, parity);
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
    __device__ __host__ inline const gauge_wrapper<Float,CPSOrder<Float,length> >
      operator()(int dim, int x_cb, int parity) const {
      return gauge_wrapper<Float,CPSOrder<Float,length> >
	(const_cast<CPSOrder<Float,length>&>(*this), dim, x_cb, parity);
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
      __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity) const {
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
      __device__ __host__ inline gauge_wrapper<Float,BQCDOrder<Float,length> >
	   operator()(int dim, int x_cb, int parity) {
	return gauge_wrapper<Float,BQCDOrder<Float,length> >(*this, dim, x_cb, parity);
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
      __device__ __host__ inline const gauge_wrapper<Float,BQCDOrder<Float,length> >
	   operator()(int dim, int x_cb, int parity) const {
	return gauge_wrapper<Float,BQCDOrder<Float,length> >
	(const_cast<BQCDOrder<Float,length>&>(*this), dim, x_cb, parity);
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
      __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity) const {
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
      __device__ __host__ inline gauge_wrapper<Float,TIFROrder<Float,length> >
	   operator()(int dim, int x_cb, int parity) {
	return gauge_wrapper<Float,TIFROrder<Float,length> >(*this, dim, x_cb, parity);
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
      __device__ __host__ inline const gauge_wrapper<Float,TIFROrder<Float,length> >
	   operator()(int dim, int x_cb, int parity) const {
	return gauge_wrapper<Float,TIFROrder<Float,length> >
	(const_cast<TIFROrder<Float,length>&>(*this), dim, x_cb, parity);
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
      __device__ __host__ inline void load(RegType v[18], int x, int dir, int parity) const {

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
      __device__ __host__ inline gauge_wrapper<Float,TIFRPaddedOrder<Float,length> >
	   operator()(int dim, int x_cb, int parity) {
	return gauge_wrapper<Float,TIFRPaddedOrder<Float,length> >(*this, dim, x_cb, parity);
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
      __device__ __host__ inline const gauge_wrapper<Float,TIFRPaddedOrder<Float,length> >
	   operator()(int dim, int x_cb, int parity) const {
	return gauge_wrapper<Float,TIFRPaddedOrder<Float,length> >
	(const_cast<TIFRPaddedOrder<Float,length>&>(*this), dim, x_cb, parity);
      }

      size_t Bytes() const { return Nc * Nc * 2 * sizeof(Float); }
    };

  } // namespace gauge

  // Use traits to reduce the template explosion
  template<typename T,QudaReconstructType,int N=18,QudaStaggeredPhase stag=QUDA_STAGGERED_PHASE_NO,bool huge_alloc=gauge::default_huge_alloc> struct gauge_mapper { };

  // double precision
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<double,QUDA_RECONSTRUCT_NO,N,stag,huge_alloc> { typedef gauge::FloatNOrder<double, N, 2, N, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<double,QUDA_RECONSTRUCT_13,N,stag,huge_alloc> { typedef gauge::FloatNOrder<double, N, 2, 13, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<double,QUDA_RECONSTRUCT_12,N,stag,huge_alloc> { typedef gauge::FloatNOrder<double, N, 2, 12, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<double,QUDA_RECONSTRUCT_9,N,stag,huge_alloc> { typedef gauge::FloatNOrder<double, N, 2, 9, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<double,QUDA_RECONSTRUCT_8,N,stag,huge_alloc> { typedef gauge::FloatNOrder<double, N, 2, 8, stag, huge_alloc> type; };

  // single precision
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<float,QUDA_RECONSTRUCT_NO,N,stag,huge_alloc> { typedef gauge::FloatNOrder<float, N, 2, N, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<float,QUDA_RECONSTRUCT_13,N,stag,huge_alloc> { typedef gauge::FloatNOrder<float, N, 4, 13, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<float,QUDA_RECONSTRUCT_12,N,stag,huge_alloc> { typedef gauge::FloatNOrder<float, N, 4, 12, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<float,QUDA_RECONSTRUCT_9,N,stag,huge_alloc> { typedef gauge::FloatNOrder<float, N, 4, 9, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<float,QUDA_RECONSTRUCT_8,N,stag,huge_alloc> { typedef gauge::FloatNOrder<float, N, 4, 8, stag, huge_alloc> type; };

  // half precision
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<short,QUDA_RECONSTRUCT_NO,N,stag,huge_alloc> { typedef gauge::FloatNOrder<short, N, 2, N, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<short,QUDA_RECONSTRUCT_13,N,stag,huge_alloc> { typedef gauge::FloatNOrder<short, N, 4, 13, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<short,QUDA_RECONSTRUCT_12,N,stag,huge_alloc> { typedef gauge::FloatNOrder<short, N, 4, 12, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<short,QUDA_RECONSTRUCT_9,N,stag,huge_alloc> { typedef gauge::FloatNOrder<short, N, 4, 9, stag, huge_alloc> type; };
  template<int N,QudaStaggeredPhase stag,bool huge_alloc> struct gauge_mapper<short,QUDA_RECONSTRUCT_8,N,stag,huge_alloc> { typedef gauge::FloatNOrder<short, N, 4, 8, stag, huge_alloc> type; };

  template<typename T, QudaGaugeFieldOrder order, int Nc> struct gauge_order_mapper { };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_QDP_GAUGE_ORDER,Nc> { typedef gauge::QDPOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_QDPJIT_GAUGE_ORDER,Nc> { typedef gauge::QDPJITOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_MILC_GAUGE_ORDER,Nc> { typedef gauge::MILCOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_BQCD_GAUGE_ORDER,Nc> { typedef gauge::BQCDOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_TIFR_GAUGE_ORDER,Nc> { typedef gauge::TIFROrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_TIFR_PADDED_GAUGE_ORDER,Nc> { typedef gauge::TIFRPaddedOrder<T, 2*Nc*Nc> type; };
  template<typename T, int Nc> struct gauge_order_mapper<T,QUDA_FLOAT2_GAUGE_ORDER,Nc> { typedef gauge::FloatNOrder<T, 2*Nc*Nc, 2, 2*Nc*Nc> type; };

  // experiments in reducing template instantation boilerplate
  // can this be replaced with a C++11 variant that uses variadic templates?

#define INSTANTIATE_RECONSTRUCT(func, g, ...)				\
  {									\
    if (!data.isNative())						\
      errorQuda("Field order %d and precision %d is not native", g.Order(), g.Precision()); \
    if( g.Reconstruct() == QUDA_RECONSTRUCT_NO) {			\
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type Gauge; \
      func(Gauge(g), g, __VA_ARGS__);					\
    } else if( g.Reconstruct() == QUDA_RECONSTRUCT_12){			\
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type Gauge; \
      func(Gauge(g), g, __VA_ARGS__);					\
    } else if( g.Reconstruct() == QUDA_RECONSTRUCT_8){			\
      typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_8>::type Gauge; \
      func(Gauge(g), g, __VA_ARGS__);					\
    } else {								\
      errorQuda("Reconstruction type %d of gauge field not supported", g.Reconstruct()); \
    }									\
  }

#define INSTANTIATE_PRECISION(func, lat, ...)				\
  {									\
    if (lat.Precision() == QUDA_DOUBLE_PRECISION) {			\
      func<double>(lat, __VA_ARGS__);					\
    } else if(lat.Precision() == QUDA_SINGLE_PRECISION) {		\
      func<float>(lat, __VA_ARGS__);					\
    } else {								\
      errorQuda("Precision %d not supported", lat.Precision());		\
    }									\
  }

} // namespace quda

#endif // _GAUGE_ORDER_H
