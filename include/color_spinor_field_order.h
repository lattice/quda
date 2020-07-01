#ifndef _COLOR_SPINOR_ORDER_H
#define _COLOR_SPINOR_ORDER_H

/**
 * @file color_spinor_field_order.h
 *
 * @section DESCRIPTION
 *
 * Define functors to allow for generic accessors regardless of field
 * ordering.  Currently this is used for cpu fields only with limited
 * ordering support, but this will be expanded for device ordering
 *  also.
 */

#include <register_traits.h>
#include <convert.h>
#include <typeinfo>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <color_spinor.h>
#include <color_spinor_field.h>
#include <trove_helper.cuh>
#include <texture_helper.cuh>
#include <transform_reduce.h>

namespace quda {

  /**
     @brief colorspinor_wrapper is an internal class that is used to
     wrap instances of colorspinor accessors, currying in a specifc
     location on the field.  The operator() accessors in
     colorspinor-field accessors return instances to this class,
     allowing us to then use operator overloading upon this class
     to interact with the ColorSpinor class.  As a result we can
     include colorspinor-field accessors directly in ColorSpinor
     expressions in kernels without having to declare temporaries
     with explicit calls to the load/save methods in the
     colorspinor-field accessors.
  */
  template <typename Float, typename T>
    struct colorspinor_wrapper {
      T &field;
      const int x_cb;
      const int parity;

      /**
         @brief colorspinor_wrapper constructor
         @param[in] a colorspinor field accessor we are wrapping
         @param[in] x_cb checkerboarded space-time index we are accessing
         @param[in] parity Parity we are accessing
      */
      __device__ __host__ inline colorspinor_wrapper<Float, T>(T &field, int x_cb, int parity) :
          field(field),
          x_cb(x_cb),
          parity(parity)
      {
      }

      /**
         @brief Assignment operator with ColorSpinor instance as input
         @param[in] C ColorSpinor we want to store in this accessor
      */
      template <typename C> __device__ __host__ inline void operator=(const C &a) { field.save(a.data, x_cb, parity); }
    };

  template <typename T, int Nc, int Ns>
    template <typename S>
    __device__ __host__ inline void ColorSpinor<T,Nc,Ns>::operator=(const colorspinor_wrapper<T,S> &a) {
    a.field.load(data, a.x_cb, a.parity);
  }

  template <typename T, int Nc, int Ns>
    template <typename S>
    __device__ __host__ inline ColorSpinor<T,Nc,Ns>::ColorSpinor(const colorspinor_wrapper<T,S> &a) {
    a.field.load(data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline void ColorSpinor<T,Nc,2>::operator=(const colorspinor_wrapper<T,S> &a) {
    a.field.load(data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline ColorSpinor<T,Nc,2>::ColorSpinor(const colorspinor_wrapper<T,S> &a) {
    a.field.load(data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline void ColorSpinor<T,Nc,4>::operator=(const colorspinor_wrapper<T,S> &a) {
    a.field.load(data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline ColorSpinor<T,Nc,4>::ColorSpinor(const colorspinor_wrapper<T,S> &a) {
    a.field.load(data, a.x_cb, a.parity);
  }

  /**
     @brief colorspinor_ghost_wrapper is an internal class that is
     used to wrap instances of colorspinor accessors, currying in a
     specifc location on the field.  The Ghost() accessors in
     colorspinor-field accessors return instances to this class,
     allowing us to then use operator overloading upon this class to
     interact with the ColorSpinor class.  As a result we can
     include colorspinor-field accessors directly in ColorSpinor
     expressions in kernels without having to declare temporaries
     with explicit calls to the loadGhost/saveGhost methods in the
     colorspinor-field accessors.
  */
  template <typename Float, typename T>
    struct colorspinor_ghost_wrapper {
      const int dim;
      const int dir;
      const int ghost_idx;
      const int parity;
      T &field;

      /**
         @brief colorspinor_ghost_wrapper constructor
         @param[in] a colorspinor field accessor we are wrapping
         @param[in] dim Dimension of the ghost we are accessing
         @param[in] dir Direction of the ghost we are accessing
         @param[in] ghost_idx Checkerboarded space-time ghost index we are accessing
         @param[in] parity Parity we are accessing
      */
      __device__ __host__ inline colorspinor_ghost_wrapper<Float, T>(
          T &field, int dim, int dir, int ghost_idx, int parity) :
          field(field),
          dim(dim),
          dir(dir),
          ghost_idx(ghost_idx),
          parity(parity)
      {
      }

      /**
         @brief Assignment operator with Matrix instance as input
         @param[in] C ColorSpinor we want to store in this accessot
      */
      template<typename C>
      __device__ __host__ inline void operator=(const C &a) {
        field.saveGhost(a.data, ghost_idx, dim, dir, parity);
      }
    };

  template <typename T, int Nc, int Ns>
    template <typename S>
    __device__ __host__ inline void ColorSpinor<T,Nc,Ns>::operator=(const colorspinor_ghost_wrapper<T,S> &a) {
    a.field.loadGhost(data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  template <typename T, int Nc, int Ns>
    template <typename S>
    __device__ __host__ inline ColorSpinor<T,Nc,Ns>::ColorSpinor(const colorspinor_ghost_wrapper<T,S> &a) {
    a.field.loadGhost(data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  template <typename T, int Nc>
  template <typename S>
  __device__ __host__ inline void ColorSpinor<T, Nc, 2>::operator=(const colorspinor_ghost_wrapper<T, S> &a)
  {
    a.field.loadGhost(data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  template <typename T, int Nc>
  template <typename S>
  __device__ __host__ inline ColorSpinor<T, Nc, 2>::ColorSpinor(const colorspinor_ghost_wrapper<T, S> &a)
  {
    a.field.loadGhost(data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline void ColorSpinor<T,Nc,4>::operator=(const colorspinor_ghost_wrapper<T,S> &a) {
    a.field.loadGhost(data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  template <typename T, int Nc>
    template <typename S>
    __device__ __host__ inline ColorSpinor<T,Nc,4>::ColorSpinor(const colorspinor_ghost_wrapper<T,S> &a) {
    a.field.loadGhost(data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  namespace colorspinor {

    template<typename ReduceType, typename Float> struct square_ {
      square_(ReduceType scale) { }
      __host__ __device__ inline ReduceType operator()(const quda::complex<Float> &x)
      { return static_cast<ReduceType>(norm(x)); }
    };

    template<typename ReduceType> struct square_<ReduceType,short> {
      const ReduceType scale;
      square_(ReduceType scale) : scale(scale) { }
      __host__ __device__ inline ReduceType operator()(const quda::complex<short> &x)
      { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
    };

    template<typename ReduceType> struct square_<ReduceType,char> {
      const ReduceType scale;
      square_(ReduceType scale) : scale(scale) { }
      __host__ __device__ inline ReduceType operator()(const quda::complex<char> &x)
      { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
    };

    template<typename Float, typename storeFloat> struct abs_ {
      abs_(const Float scale) { }
      __host__ __device__ Float operator()(const quda::complex<storeFloat> &x) { return abs(x); }
    };

    template<typename Float> struct abs_<Float,short> {
      Float scale;
      abs_(const Float scale) : scale(scale) { }
      __host__ __device__ Float operator()(const quda::complex<short> &x)
      { return abs(scale * complex<Float>(x.real(), x.imag())); }
    };

    template<typename Float> struct abs_<Float,char> {
      Float scale;
      abs_(const Float scale) : scale(scale) { }
      __host__ __device__ Float operator()(const quda::complex<char> &x)
      { return abs(scale * complex<Float>(x.real(), x.imag())); }
    };

    template <typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order> struct AccessorCB {
      AccessorCB(const ColorSpinorField &) { errorQuda("Not implemented"); }
      AccessorCB() { errorQuda("Not implemented"); }
      __device__ __host__ inline int index(int parity, int x_cb, int s, int c, int v) const { return 0; }
    };

    template<typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order> struct GhostAccessorCB {
      GhostAccessorCB(const ColorSpinorField &) { errorQuda("Not implemented"); }
      GhostAccessorCB() { errorQuda("Not implemented"); }
      __device__ __host__ inline int index(int dim, int dir, int parity, int x_cb, int s, int c, int v) const
      { return 0; }
    };

    template <typename Float, int nSpin, int nColor, int nVec>
    struct AccessorCB<Float, nSpin, nColor, nVec, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> {
      const int offset_cb;
    AccessorCB(const ColorSpinorField &field) : offset_cb((field.Bytes()>>1) / sizeof(complex<Float>)) { }
    AccessorCB() : offset_cb(0) { }
    __device__ __host__ inline int index(int parity, int x_cb, int s, int c, int v) const
    {
      return parity * offset_cb + ((x_cb * nSpin + s) * nColor + c) * nVec + v;
    }
    };

    template<typename Float, int nSpin, int nColor, int nVec>
      struct GhostAccessorCB<Float,nSpin,nColor,nVec,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> {
      int faceVolumeCB[4];
      int ghostOffset[4];
      GhostAccessorCB(const ColorSpinorField &a, int nFace = 1) {
        for (int d=0; d<4; d++) {
          faceVolumeCB[d] = nFace*a.SurfaceCB(d);
          ghostOffset[d] = faceVolumeCB[d]*nColor*nSpin*nVec;
        }
      }
      GhostAccessorCB() : ghostOffset{ } { }
      __device__ __host__ inline int index(int dim, int dir, int parity, int x_cb, int s, int c, int v) const
      { return parity*ghostOffset[dim] + ((x_cb*nSpin+s)*nColor+c)*nVec+v; }
    };

    template<int nSpin, int nColor, int nVec, int N>
      __device__ __host__ inline int indexFloatN(int x_cb, int s, int c, int v, int stride) {
      int k = ((s*nColor+c)*nVec+v)*2; // factor of two for complexity
      int j = k / N; // factor of two for complexity
      int i = k % N;
      return ((j*stride+x_cb)*N+i) / 2; // back to a complex offset
    };

    template <typename Float, int nSpin, int nColor, int nVec>
    struct AccessorCB<Float, nSpin, nColor, nVec, QUDA_FLOAT2_FIELD_ORDER> {
      const int stride;
      const int offset_cb;
      AccessorCB(const ColorSpinorField &field) :
        stride(field.Stride()),
        offset_cb((field.Bytes() >> 1) / sizeof(complex<Float>))
      {
      }
    AccessorCB(): stride(0), offset_cb(0) { }
    __device__ __host__ inline int index(int parity, int x_cb, int s, int c, int v) const
    {
      return parity * offset_cb + ((s * nColor + c) * nVec + v) * stride + x_cb;
    }
    };

    template<typename Float, int nSpin, int nColor, int nVec>
      struct GhostAccessorCB<Float,nSpin,nColor,nVec,QUDA_FLOAT2_FIELD_ORDER> {
      int faceVolumeCB[4];
      int ghostOffset[4];
      GhostAccessorCB(const ColorSpinorField &a, int nFace = 1) {
        for (int d=0; d<4; d++) {
          faceVolumeCB[d] = nFace*a.SurfaceCB(d);
          ghostOffset[d] = faceVolumeCB[d]*nColor*nSpin*nVec;
        }
      }
      GhostAccessorCB() : faceVolumeCB{ }, ghostOffset{ } { }
      __device__ __host__ inline int index(int dim, int dir, int parity, int x_cb, int s, int c, int v) const
      { return parity*ghostOffset[dim] + ((s*nColor+c)*nVec+v)*faceVolumeCB[dim] + x_cb; }
    };

    template <typename Float, int nSpin, int nColor, int nVec>
    struct AccessorCB<Float, nSpin, nColor, nVec, QUDA_FLOAT4_FIELD_ORDER> {
      const int stride;
      const int offset_cb;
      AccessorCB(const ColorSpinorField &field) :
        stride(field.Stride()),
        offset_cb((field.Bytes() >> 1) / sizeof(complex<Float>))
      {
      }
    AccessorCB() : stride(0), offset_cb(0) { }
    __device__ __host__ inline int index(int parity, int x_cb, int s, int c, int v) const
    {
      return parity * offset_cb + indexFloatN<nSpin, nColor, nVec, 4>(x_cb, s, c, v, stride);
    }
    };

    template<typename Float, int nSpin, int nColor, int nVec>
      struct GhostAccessorCB<Float,nSpin,nColor,nVec,QUDA_FLOAT4_FIELD_ORDER> {
      int faceVolumeCB[4];
      int ghostOffset[4];
      GhostAccessorCB(const ColorSpinorField &a, int nFace = 1) {
  for (int d=0; d<4; d++) {
    faceVolumeCB[d] = nFace*a.SurfaceCB(d);
    ghostOffset[d] = faceVolumeCB[d]*nColor*nSpin*nVec;
  }
      }
    GhostAccessorCB() : faceVolumeCB{ }, ghostOffset{ } { }
      __device__ __host__ inline int index(int dim, int dir, int parity, int x_cb, int s, int c, int v) const
      { return parity*ghostOffset[dim] + indexFloatN<nSpin,nColor,nVec,4>(x_cb, s, c, v, faceVolumeCB[dim]); }
    };


    template <typename Float, typename storeFloat> __host__ __device__ inline constexpr bool fixed_point() { return false; }
    template<> __host__ __device__ inline constexpr bool fixed_point<float,char>() { return true; }
    template<> __host__ __device__ inline constexpr bool fixed_point<float,short>() { return true; }
    template<> __host__ __device__ inline constexpr bool fixed_point<float,int>() { return true; }

    template <typename Float, typename storeFloat> __host__ __device__ inline constexpr bool match() { return false; }
    template<> __host__ __device__ inline constexpr bool match<char,char>() { return true; }
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

  __device__ __host__ inline void real(const Float &a) {
    if (!fixed) {
      v[idx].real(storeFloat(a));
    } else { // we need to scale and then round
      v[idx].real(storeFloat(round(scale * a)));
    }
  }
  __device__ __host__ inline void imag(const Float &a) {
    if (!fixed) {
      v[idx].imag(storeFloat(a));
    } else { // we need to scale and then round
      v[idx].imag(storeFloat(round(scale * a)));
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
     @brief Assignment operator with real number instance as input
     @param a real number we want to store in this accessor
  */
        template<typename theirFloat>
  __device__ __host__ inline void operator=(const theirFloat &a) { *this = complex<theirFloat>(a,static_cast<theirFloat>(0.0)); }

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


    template <typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order,
      typename storeFloat=Float, typename ghostFloat=storeFloat, bool disable_ghost=false, bool block_float=false, bool use_tex=false>
      class FieldOrderCB {

      typedef float norm_type;

  protected:
      complex<storeFloat> *v;
    const AccessorCB<storeFloat,nSpin,nColor,nVec,order> accessor;
      // since these variables are mutually exclusive, we use a union to minimize the accessor footprint
      union {
        norm_type *norm;
        Float scale;
      };
      union {
        Float scale_inv;
        int norm_offset;
      };
#ifndef DISABLE_GHOST
      mutable complex<ghostFloat> *ghost[8];
      mutable norm_type *ghost_norm[8];
      mutable int x[QUDA_MAX_DIM];
      const int volumeCB;
      const int nDim;
      const QudaGammaBasis gammaBasis;
      const int siteSubset;
      const int nParity;
      const QudaFieldLocation location;
    const GhostAccessorCB<ghostFloat,nSpin,nColor,nVec,order> ghostAccessor;
      Float ghost_scale;
      Float ghost_scale_inv;
#endif
      static constexpr bool fixed = fixed_point<Float,storeFloat>();
      static constexpr bool ghost_fixed = fixed_point<Float,ghostFloat>();
      static constexpr bool block_float_ghost = !fixed && ghost_fixed;

    public:
      /**
       * Constructor for the FieldOrderCB class
       * @param field The field that we are accessing
       */
    FieldOrderCB(const ColorSpinorField &field, int nFace=1, void *v_=0, void **ghost_=0)
      : v(v_? static_cast<complex<storeFloat>*>(const_cast<void*>(v_))
	  : static_cast<complex<storeFloat>*>(const_cast<void*>(field.V()))),
        accessor(field), scale(static_cast<Float>(1.0)), scale_inv(static_cast<Float>(1.0))
#ifndef DISABLE_GHOST
        , volumeCB(field.VolumeCB()), nDim(field.Ndim()), gammaBasis(field.GammaBasis()),
	siteSubset(field.SiteSubset()), nParity(field.SiteSubset()),
        location(field.Location()), ghostAccessor(field,nFace),
        ghost_scale(static_cast<Float>(1.0)), ghost_scale_inv(static_cast<Float>(1.0))
#endif
      {
#ifndef DISABLE_GHOST
        for (int d=0; d<QUDA_MAX_DIM; d++) x[d]=field.X(d);
        resetGhost(field, ghost_ ? ghost_ : field.Ghost());
#endif
        resetScale(field.Scale());

#ifdef DISABLE_GHOST
        if (!disable_ghost) errorQuda("DISABLE_GHOST macro set but corresponding disable_ghost template not set");
#endif

        if (block_float) {
          // only if we have block_float format do we set these (only block_orthogonalize.cu at present)
          norm = static_cast<norm_type *>(const_cast<void *>(field.Norm()));
          norm_offset = field.NormBytes() / (2 * sizeof(norm_type));
        }
      }

#ifndef DISABLE_GHOST
      void resetGhost(const ColorSpinorField &a, void * const *ghost_) const
      {
        for (int dim=0; dim<4; dim++) {
          for (int dir=0; dir<2; dir++) {
          ghost[2*dim+dir] = static_cast<complex<ghostFloat>*>(ghost_[2*dim+dir]);
          ghost_norm[2 * dim + dir] = !block_float_ghost ? nullptr :
            reinterpret_cast<norm_type *>(static_cast<char *>(ghost_[2 * dim + dir]) + a.GhostNormOffset(dim, dir) * sizeof(norm_type)
                                          - a.GhostOffset(dim, dir) * sizeof(ghostFloat));
          }
        }
      }
#endif

      void resetScale(Float max) {
        if (fixed) {
          scale = static_cast<Float>(std::numeric_limits<storeFloat>::max() / max);
          scale_inv = static_cast<Float>(max / std::numeric_limits<storeFloat>::max());
        }
#ifndef DISABLE_GHOST
        if (ghost_fixed) {
          if (block_float_ghost && max != static_cast<Float>(1.0))
              errorQuda("Block-float accessor requires max=1.0 not max=%e\n", max);
          ghost_scale = static_cast<Float>(std::numeric_limits<ghostFloat>::max() / max);
          ghost_scale_inv = static_cast<Float>(max / std::numeric_limits<ghostFloat>::max());
        }
#endif
      }

      /**
       * Read-only complex-member accessor function.  The last
       * parameter n is only used for indexed into the packed
       * null-space vectors.
       * @param x 1-d checkerboard site index
       * @param s spin index
       * @param c color index
       * @param v vector number
       */
      __device__ __host__ inline const complex<Float> operator()(int parity, int x_cb, int s, int c, int n=0) const
      {
#ifdef __CUDA_ARCH__
        if (!fixed) {
          return complex<Float>( use_tex ? __ldg(v + accessor.index(parity,x_cb,s,c,n)) : v[accessor.index(parity,x_cb,s,c,n)]);
	} else {
	  complex<storeFloat> tmp = use_tex ? __ldg(v + accessor.index(parity,x_cb,s,c,n)) : v[accessor.index(parity,x_cb,s,c,n)];
	  Float norm_ = block_float ? (use_tex ? __ldg(norm + parity*norm_offset+x_cb) : norm[parity*norm_offset+x_cb]) : scale_inv;
	  return norm_*complex<Float>(static_cast<Float>(tmp.x), static_cast<Float>(tmp.y));
        }
#else
        if (!fixed) {
          return complex<Float>( v[accessor.index(parity,x_cb,s,c,n)] );
	} else {
	  complex<storeFloat> tmp = v[accessor.index(parity,x_cb,s,c,n)];
	  Float norm_ = block_float ? norm[parity*norm_offset+x_cb] : scale_inv;
	  return norm_*complex<Float>(static_cast<Float>(tmp.x), static_cast<Float>(tmp.y));
        }
#endif
      }


      /**
       * Writable complex-member accessor function.  The last
       * parameter n is only used for indexed into the packed
       * null-space vectors.
       * @param x 1-d checkerboard site index
       * @param s spin index
       * @param c color index
       * @param v vector number
       */
      __device__ __host__ inline fieldorder_wrapper<Float,storeFloat> operator()(int parity, int x_cb, int s, int c, int n=0)
  { return fieldorder_wrapper<Float,storeFloat>(v, accessor.index(parity,x_cb,s,c,n), scale, scale_inv); }

#ifndef DISABLE_GHOST
      /**
       * Read-only complex-member accessor function for the ghost
       * zone.  The last parameter n is only used for indexed into the
       * packed null-space vectors.
       * @param x 1-d checkerboard site index
       * @param s spin index
       * @param c color index
       * @param v vector number
       */
      __device__ __host__ inline const complex<Float> Ghost(int dim, int dir, int parity, int x_cb, int s, int c, int n=0) const
      {
#ifdef __CUDA_ARCH__
        if (!ghost_fixed) {
          return complex<Float>( use_tex ? __ldg(ghost[2*dim+dir]+ghostAccessor.index(dim,dir,parity,x_cb,s,c,n)) :
				 ghost[2*dim+dir][ghostAccessor.index(dim,dir,parity,x_cb,s,c,n)] );
        } else {
          Float scale = ghost_scale_inv;
          if (block_float_ghost) scale *= (use_tex ? __ldg(ghost_norm[2*dim+dir]+parity*ghostAccessor.faceVolumeCB[dim] + x_cb) :
					   ghost_norm[2*dim+dir][parity*ghostAccessor.faceVolumeCB[dim] + x_cb]);
          complex<ghostFloat> tmp = (use_tex ? __ldg(ghost[2*dim+dir] + ghostAccessor.index(dim,dir,parity,x_cb,s,c,n)) :
				     ghost[2*dim+dir][ghostAccessor.index(dim,dir,parity,x_cb,s,c,n)]);
          return scale*complex<Float>(static_cast<Float>(tmp.x), static_cast<Float>(tmp.y));
        }
#else
        if (!ghost_fixed) {
          return complex<Float>( ghost[2*dim+dir][ghostAccessor.index(dim,dir,parity,x_cb,s,c,n)] );
        } else {
          Float scale = ghost_scale_inv;
          if (block_float_ghost) scale *= ghost_norm[2*dim+dir][parity*ghostAccessor.faceVolumeCB[dim] + x_cb];
          complex<ghostFloat> tmp = ghost[2*dim+dir][ghostAccessor.index(dim,dir,parity,x_cb,s,c,n)];
          return scale*complex<Float>(static_cast<Float>(tmp.x), static_cast<Float>(tmp.y));
        }
#endif
      }

      /**
       * Writable complex-member accessor function for the ghost zone.
       * The last parameter n is only used for indexed into the packed
       * null-space vectors.
       * @param x 1-d checkerboard site index
       * @param s spin index
       * @param c color index
       * @param n vector number
       * @param max site-element max (only when using block-float format)
       */
	__device__ __host__ inline fieldorder_wrapper<Float,ghostFloat> Ghost(int dim, int dir, int parity, int x_cb, int s, int c, int n=0, Float max=0)
      {
        if (block_float_ghost && s==0 && c==0 && n==0) ghost_norm[2*dim+dir][parity*ghostAccessor.faceVolumeCB[dim] + x_cb] = max;
        const int idx = ghostAccessor.index(dim,dir,parity,x_cb,s,c,n);
        return fieldorder_wrapper<Float,ghostFloat>(ghost[2*dim+dir], idx,
              block_float_ghost ? ghost_scale/max : ghost_scale,
              block_float_ghost ? ghost_scale_inv*max : ghost_scale_inv);

      }

      /**
   Convert from 1-dimensional index to the n-dimensional spatial index.
   With full fields, we assume that the field is even-odd ordered.  The
   lattice coordinates that are computed here are full-field
   coordinates.
      */
      __device__ __host__ inline void LatticeIndex(int y[QUDA_MAX_DIM], int i) const {
        if (siteSubset == QUDA_FULL_SITE_SUBSET) x[0] /= 2;

        for (int d=0; d<nDim; d++) {
          y[d] = i % x[d];
          i /= x[d];
        }
        int parity = i; // parity is the slowest running dimension

        // convert into the full-field lattice coordinate
        if (siteSubset == QUDA_FULL_SITE_SUBSET) {
          for (int d=1; d<nDim; d++) parity += y[d];
          parity = parity & 1;
          x[0] *= 2; // restore x[0]
        }
        y[0] = 2*y[0] + parity;  // compute the full x coordinate
      }

      /**
	 Convert from n-dimensional spatial index to the 1-dimensional index.
	 With full fields, we assume that the field is even-odd ordered.  The
	 input lattice coordinates are always full-field coordinates.
      */
      __device__ __host__ inline void OffsetIndex(int &i, int y[QUDA_MAX_DIM]) const {
	int parity = 0;
	int savey0 = y[0];

	if (siteSubset == QUDA_FULL_SITE_SUBSET) {
	  for (int d=0; d<nDim; d++) parity += y[d];
	  parity = parity & 1;
	  y[0] /= 2;
	  x[0] /= 2;
	}

	i = parity;
	for (int d=nDim-1; d>=0; d--) i = x[d]*i + y[d];

	if (siteSubset == QUDA_FULL_SITE_SUBSET) {
	  //y[0] = 2*y[0] + parity;
	  y[0] = savey0;
	  x[0] *= 2; // restore x[0]
	}
      }

      /** Return the length of dimension d */
      __device__ __host__ inline int X(int d) const { return x[d]; }

      /** Return the length of dimension d */
      __device__ __host__ inline const int* X() const { return x; }
#endif

      /** Returns the number of field colors */
       __device__ __host__ inline int Ncolor() const { return nColor; }

      /** Returns the number of field spins */
      __device__ __host__ inline int Nspin() const { return nSpin; }

      /** Returns the number of packed vectors (for mg prolongator) */
      __device__ __host__ inline int Nvec() const { return nVec; }

#ifndef DISABLE_GHOST
      /** Returns the number of field parities (1 or 2) */
      __device__ __host__ inline int Nparity() const { return nParity; }

      /** Returns the field volume */
      __device__ __host__ inline int VolumeCB() const { return volumeCB; }

      /** Returns the field geometric dimension */
      __device__ __host__ inline int Ndim() const { return nDim; }

      /** Returns the field geometric dimension */
      __device__ __host__ inline QudaGammaBasis GammaBasis() const { return gammaBasis; }

      /**
       * Returns the L2 norm squared of the field in a given dimension
       * @param[in] global Whether to do a global or process local norm2 reduction
       * @return L2 norm squared
      */
      __host__ double norm2(bool global = true) const
      {
        double nrm2 = ::quda::transform_reduce(location, v, nParity * volumeCB * nSpin * nColor * nVec,
                                               square_<double, storeFloat>(scale_inv), 0.0, plus<double>());
        if (global) comm_allreduce(&nrm2);
        return nrm2;
      }

      /**
       * Returns the Linfinity norm of the field
       * @param[in] global Whether to do a global or process local Linfinity reduction
       * @return Linfinity norm
      */
      __host__ double abs_max(bool global = true) const
      {
        double absmax = ::quda::transform_reduce(location, v, nParity * volumeCB * nSpin * nColor * nVec,
                                                 abs_<double, storeFloat>(scale_inv), 0.0, maximum<double>());
        if (global) comm_allreduce_max(&absmax);
        return absmax;
      }

      size_t Bytes() const { return nParity * static_cast<size_t>(volumeCB) * nColor * nSpin * nVec * 2ll * sizeof(storeFloat); }
#endif
    };

    /**
       @brief Accessor routine for ColorSpinorFields in native field order.
       @tparam Float Underlying storage data type of the field
       @tparam Ns Number of spin components
       @tparam Nc Number of colors
       @tparam N Number of real numbers per short vector
       @tparam spin_project Whether the ghosts are spin projected or not
       @tparam huge_alloc Template parameter that enables 64-bit
       pointer arithmetic for huge allocations (e.g., packed set of
       vectors).  Default is to use 32-bit pointer arithmetic.
     */
    template <typename Float, int Ns, int Nc, int N_, bool spin_project = false, bool huge_alloc = false>
    struct FloatNOrder {
      static_assert((2 * Ns * Nc) % N_ == 0, "Internal degrees of freedom not divisible by short-vector length");
      static constexpr int length = 2 * Ns * Nc;
      static constexpr int length_ghost = spin_project ? length / 2 : length;
      static constexpr int N = N_;
      static constexpr int M = length / N;
      // if spin projecting, check that short vector length is compatible, if not halve the vector length
      static constexpr int N_ghost = !spin_project ? N : (Ns * Nc) % N == 0 ? N : N / 2;
      static constexpr int M_ghost = length_ghost / N_ghost;
      using Accessor = FloatNOrder<Float, Ns, Nc, N, spin_project, huge_alloc>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      using Vector = typename VectorType<Float, N>::type;
      using GhostVector = typename VectorType<Float, N_ghost>::type;
      using AllocInt = typename AllocType<huge_alloc>::type;
      using norm_type = float;
      Float *field;
      norm_type *norm;
      const AllocInt offset; // offset can be 32-bit or 64-bit
      const AllocInt norm_offset;
#ifdef USE_TEXTURE_OBJECTS
      typedef typename TexVectorType<real, N>::type TexVector;
      cudaTextureObject_t tex;
      cudaTextureObject_t texNorm;
#endif
      int volumeCB;
      int faceVolumeCB[4];
      int stride;
      mutable Float *ghost[8];
      mutable norm_type *ghost_norm[8];
      int nParity;
      void *backup_h; //! host memory for backing up the field when tuning
      size_t bytes;

      FloatNOrder(const ColorSpinorField &a, int nFace = 1, Float *field_ = 0, norm_type *norm_ = 0, Float **ghost_ = 0,
                  bool override = false) :
        field(field_ ? field_ : (Float *)a.V()),
        offset(a.Bytes() / (2 * sizeof(Float) * N)),
        norm(norm_ ? norm_ : (norm_type *)a.Norm()),
        norm_offset(a.NormBytes() / (2 * sizeof(norm_type))),
#ifdef USE_TEXTURE_OBJECTS
        tex(0),
        texNorm(0),
#endif
        volumeCB(a.VolumeCB()),
        stride(a.Stride()),
        nParity(a.SiteSubset()),
        backup_h(nullptr),
        bytes(a.Bytes())
  {
    for (int i=0; i<4; i++) {
      faceVolumeCB[i] = a.SurfaceCB(i)*nFace;
    }
    resetGhost(a, ghost_ ? (void **)ghost_ : a.Ghost());
#ifdef USE_TEXTURE_OBJECTS
    if (a.Location() == QUDA_CUDA_FIELD_LOCATION) {
      tex = static_cast<const cudaColorSpinorField&>(a).Tex();
      texNorm = static_cast<const cudaColorSpinorField&>(a).TexNorm();
    }
    if (!huge_alloc && (this->field != a.V() || (a.Precision() == QUDA_HALF_PRECISION && this->norm != a.Norm()) || (a.Precision() == QUDA_QUARTER_PRECISION && this->norm != a.Norm())) && !override) {
      errorQuda("Cannot use texture read since data pointer does not equal field pointer - use with huge_alloc=true instead");
    }
#endif
  }

  void resetGhost(const ColorSpinorField &a, void *const *ghost_) const
  {
    for (int dim = 0; dim < 4; dim++) {
      for (int dir = 0; dir < 2; dir++) {
        ghost[2 * dim + dir] = comm_dim_partitioned(dim) ? static_cast<Float *>(ghost_[2 * dim + dir]) : nullptr;
        ghost_norm[2 * dim + dir] = !comm_dim_partitioned(dim) ?
          nullptr :
          reinterpret_cast<norm_type *>(static_cast<char *>(ghost_[2 * dim + dir])
                                        + nParity * length_ghost * faceVolumeCB[dim] * sizeof(Float));
      }
    }
  }

  __device__ __host__ inline void load(complex out[length / 2], int x, int parity = 0) const
  {
    real v[length];
    norm_type nrm;
    if (isFixed<Float>::value) {
#if defined(USE_TEXTURE_OBJECTS) && defined(__CUDA_ARCH__)
      // use textures unless we have a large alloc
      nrm = !huge_alloc ? tex1Dfetch_<float>(texNorm, x + parity * norm_offset) : norm[x + parity * norm_offset];
#else
      nrm = vector_load<float>(norm, x + parity * norm_offset);
#endif
    }

#pragma unroll
    for (int i=0; i<M; i++) {
#if defined(USE_TEXTURE_OBJECTS) && defined(__CUDA_ARCH__)
      if (!huge_alloc) { // use textures unless we have a huge alloc
        // first do texture load from memory
        TexVector vecTmp = tex1Dfetch_<TexVector>(tex, parity * offset + stride * i + x);
        // now insert into output array
#pragma unroll
        for (int j = 0; j < N; j++) copy(v[i * N + j], reinterpret_cast<real *>(&vecTmp)[j]);
        if (isFixed<Float>::value) {
#pragma unroll
          for (int j = 0; j < N; j++) v[i * N + j] *= nrm;
        }
      } else
#endif
      {
        // first load from memory
        Vector vecTmp = vector_load<Vector>(field, parity * offset + x + stride * i);
        // now copy into output and scale
#pragma unroll
        for (int j = 0; j < N; j++) copy_and_scale(v[i * N + j], reinterpret_cast<Float *>(&vecTmp)[j], nrm);
      }
    }

#pragma unroll
    for (int i = 0; i < length / 2; i++) out[i] = complex(v[2 * i + 0], v[2 * i + 1]);
  }

  __device__ __host__ inline void save(const complex in[length / 2], int x, int parity = 0)
  {
    real v[length];

#pragma unroll
    for (int i = 0; i < length / 2; i++) {
      v[2 * i + 0] = in[i].real();
      v[2 * i + 1] = in[i].imag();
    }

    if (isFixed<Float>::value) {
      norm_type max_[length / 2];
      // two-pass to increase ILP (assumes length divisible by two, e.g. complex-valued)
#pragma unroll
      for (int i = 0; i < length / 2; i++) max_[i] = fmaxf(fabsf((norm_type)v[i]), fabsf((norm_type)v[i + length / 2]));
      norm_type scale = 0.0;
#pragma unroll
      for (int i = 0; i < length / 2; i++) scale = fmaxf(max_[i], scale);
      norm[x+parity*norm_offset] = scale;

#ifdef __CUDA_ARCH__
      real scale_inv = __fdividef(fixedMaxValue<Float>::value, scale);
#else
      real scale_inv = fixedMaxValue<Float>::value / scale;
#endif
#pragma unroll
      for (int i = 0; i < length; i++) v[i] = v[i] * scale_inv;
    }

#pragma unroll
    for (int i=0; i<M; i++) {
      Vector vecTmp;
      // first do scalar copy converting into storage type
#pragma unroll
      for (int j = 0; j < N; j++) copy_scaled(reinterpret_cast<Float *>(&vecTmp)[j], v[i * N + j]);
      // second do vectorized copy into memory
      vector_store(field, parity * offset + x + stride * i, vecTmp);
    }
  }

  /**
     @brief This accessor routine returns a colorspinor_wrapper to this object,
     allowing us to overload various operators for manipulating at
     the site level interms of matrix operations.
     @param[in] x_cb Checkerboarded space-time index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline colorspinor_wrapper<real, Accessor> operator()(int x_cb, int parity)
  {
    return colorspinor_wrapper<real, Accessor>(*this, x_cb, parity);
  }

  /**
     @brief This accessor routine returns a const colorspinor_wrapper to this object,
     allowing us to overload various operators for manipulating at
     the site level interms of matrix operations.
     @param[in] x_cb Checkerboarded space-time index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline const colorspinor_wrapper<real, Accessor> operator()(int x_cb, int parity) const
  {
    return colorspinor_wrapper<real, Accessor>(const_cast<Accessor &>(*this), x_cb, parity);
  }

  __device__ __host__ inline void loadGhost(complex out[length_ghost / 2], int x, int dim, int dir, int parity = 0) const
  {
    real v[length_ghost];
    norm_type nrm;
    if (isFixed<Float>::value) { nrm = vector_load<float>(ghost_norm[2 * dim + dir], parity * faceVolumeCB[dim] + x); }

#pragma unroll
    for (int i = 0; i < M_ghost; i++) {
      GhostVector vecTmp = vector_load<GhostVector>(ghost[2 * dim + dir],
                                                    parity * faceVolumeCB[dim] * M_ghost + i * faceVolumeCB[dim] + x);
#pragma unroll
      for (int j = 0; j < N_ghost; j++) copy_and_scale(v[i * N_ghost + j], reinterpret_cast<Float *>(&vecTmp)[j], nrm);
    }

#pragma unroll
    for (int i = 0; i < length_ghost / 2; i++) out[i] = complex(v[2 * i + 0], v[2 * i + 1]);
  }

  __device__ __host__ inline void saveGhost(const complex in[length_ghost / 2], int x, int dim, int dir,
                                            int parity = 0) const
  {
    real v[length_ghost];
#pragma unroll
    for (int i = 0; i < length_ghost / 2; i++) {
      v[2 * i + 0] = in[i].real();
      v[2 * i + 1] = in[i].imag();
    }

    if (isFixed<Float>::value) {
      norm_type max_[length_ghost / 2];
      // two-pass to increase ILP (assumes length divisible by two, e.g. complex-valued)
#pragma unroll
      for (int i = 0; i < length_ghost / 2; i++)
        max_[i] = fmaxf( (norm_type)fabsf( (norm_type)v[i] ),
                         (norm_type)fabsf( (norm_type)v[i + length_ghost / 2] ) );
      norm_type scale = 0.0;
#pragma unroll
      for (int i = 0; i < length_ghost / 2; i++) scale = fmaxf(max_[i], scale);
      ghost_norm[2 * dim + dir][parity * faceVolumeCB[dim] + x] = scale;

#ifdef __CUDA_ARCH__
      real scale_inv = __fdividef(fixedMaxValue<Float>::value, scale);
#else
      real scale_inv = fixedMaxValue<Float>::value / scale;
#endif
#pragma unroll
      for (int i = 0; i < length_ghost; i++) v[i] = v[i] * scale_inv;
    }

#pragma unroll
    for (int i = 0; i < M_ghost; i++) {
      GhostVector vecTmp;
      // first do scalar copy converting into storage type
#pragma unroll
      for (int j = 0; j < N_ghost; j++) copy_scaled(reinterpret_cast<Float *>(&vecTmp)[j], v[i * N_ghost + j]);
      // second do vectorized copy into memory
      vector_store(ghost[2 * dim + dir], parity * faceVolumeCB[dim] * M_ghost + i * faceVolumeCB[dim] + x, vecTmp);
    }
  }

  /**
     @brief This accessor routine returns a colorspinor_ghost_wrapper to this object,
     allowing us to overload various operators for manipulating at
     the site level interms of matrix operations.
     @param[in] dim Dimensions of the ghost we are requesting
     @param[in] ghost_idx Checkerboarded space-time ghost index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_ghost_wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline colorspinor_ghost_wrapper<real, Accessor> Ghost(int dim, int dir, int ghost_idx, int parity)
  {
    return colorspinor_ghost_wrapper<real, Accessor>(*this, dim, dir, ghost_idx, parity);
  }

  /**
     @brief This accessor routine returns a const
     colorspinor_ghost_wrapper to this object, allowing us to
     overload various operators for manipulating at the site
     level interms of matrix operations.
     @param[in] dim Dimensions of the ghost we are requesting
     @param[in] ghost_idx Checkerboarded space-time ghost index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_ghost+wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline const colorspinor_ghost_wrapper<real, Accessor> Ghost(int dim, int dir, int ghost_idx,
                                                                                   int parity) const
  {
    return colorspinor_ghost_wrapper<real, Accessor>(const_cast<Accessor &>(*this), dim, dir, ghost_idx, parity);
  }

  /**
     @brief Backup the field to the host when tuning
  */
  void save() {
    if (backup_h) errorQuda("Already allocated host backup");
    backup_h = safe_malloc(bytes);
    cudaMemcpy(backup_h, field, bytes, cudaMemcpyDeviceToHost);
    checkCudaError();
  }

  /**
     @brief Restore the field from the host after tuning
  */
  void load() {
    cudaMemcpy(field, backup_h, bytes, cudaMemcpyHostToDevice);
    host_free(backup_h);
    backup_h = nullptr;
    checkCudaError();
  }

  size_t Bytes() const
  {
    return nParity * volumeCB * (Nc * Ns * 2 * sizeof(Float) + (isFixed<Float>::value ? sizeof(norm_type) : 0));
  }
    };

    /**
       @brief This is just a dummy structure we use for trove to define the
       required structure size
       @tparam real Real number type
       @tparam length Number of elements in the structure
    */
    template <typename real, int length> struct S { real v[length]; };

    template <typename Float, int Ns, int Nc>
      struct SpaceColorSpinorOrder {
      using Accessor = SpaceColorSpinorOrder<Float, Ns, Nc>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      static const int length = 2 * Ns * Nc;
      Float *field;
      size_t offset;
      Float *ghost[8];
      int volumeCB;
      int faceVolumeCB[4];
      int stride;
      int nParity;
      SpaceColorSpinorOrder(const ColorSpinorField &a, int nFace=1, Float *field_=0, float *dummy=0, Float **ghost_=0)
      : field(field_ ? field_ : (Float*)a.V()), offset(a.Bytes()/(2*sizeof(Float))),
    volumeCB(a.VolumeCB()), stride(a.Stride()), nParity(a.SiteSubset())
  {
    if (volumeCB != stride) errorQuda("Stride must equal volume for this field order");
    for (int i=0; i<4; i++) {
      ghost[2*i] = ghost_ ? ghost_[2*i] : 0;
      ghost[2*i+1] = ghost_ ? ghost_[2*i+1] : 0;
      faceVolumeCB[i] = a.SurfaceCB(i)*nFace;
    }
  }

  __device__ __host__ inline void load(complex v[length / 2], int x, int parity = 0) const
  {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
    typedef S<Float,length> structure;
    trove::coalesced_ptr<structure> field_((structure*)field);
    structure v_ = field_[parity*volumeCB + x];
    for (int s=0; s<Ns; s++) {
      for (int c = 0; c < Nc; c++) { v[s * Nc + c] = complex(v_.v[(c * Ns + s) * 2 + 0], v_.v[(c * Ns + s) * 2 + 1]); }
    }
#else
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        v[s * Nc + c] = complex(field[parity * offset + ((x * Nc + c) * Ns + s) * 2 + 0],
                                field[parity * offset + ((x * Nc + c) * Ns + s) * 2 + 1]);
      }
    }
#endif
  }

  __device__ __host__ inline void save(const complex v[length / 2], int x, int parity = 0)
  {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
    typedef S<Float,length> structure;
    trove::coalesced_ptr<structure> field_((structure*)field);
    structure v_;
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        v_.v[(c*Ns + s)*2 + 0] = (Float)v[s*Nc+c].real();
        v_.v[(c*Ns + s)*2 + 1] = (Float)v[s*Nc+c].imag();
      }
    }
    field_[parity*volumeCB + x] = v_;
#else
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        field[parity*offset + ((x*Nc + c)*Ns + s)*2 + 0] = v[s*Nc+c].real();
        field[parity*offset + ((x*Nc + c)*Ns + s)*2 + 1] = v[s*Nc+c].imag();
      }
    }
#endif
  }

  /**
     @brief This accessor routine returns a colorspinor_wrapper to this object,
     allowing us to overload various operators for manipulating at
     the site level interms of matrix operations.
     @param[in] x_cb Checkerboarded space-time index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline colorspinor_wrapper<real, Accessor> operator()(int x_cb, int parity)
  {
    return colorspinor_wrapper<real, Accessor>(*this, x_cb, parity);
  }

  /**
     @brief This accessor routine returns a const colorspinor_wrapper to this object,
     allowing us to overload various operators for manipulating at
     the site level interms of matrix operations.
     @param[in] x_cb Checkerboarded space-time index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline const colorspinor_wrapper<real, Accessor> operator()(int x_cb, int parity) const
  {
    return colorspinor_wrapper<real, Accessor>(const_cast<Accessor &>(*this), x_cb, parity);
  }

  __device__ __host__ inline void loadGhost(complex v[length / 2], int x, int dim, int dir, int parity = 0) const
  {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        v[s * Nc + c] = complex(ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Nc + c) * Ns + s) * 2 + 0],
                                ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Nc + c) * Ns + s) * 2 + 1]);
      }
    }
  }

  __device__ __host__ inline void saveGhost(const complex v[length / 2], int x, int dim, int dir, int parity = 0)
  {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Nc + c) * Ns + s) * 2 + 0] = v[s * Nc + c].real();
        ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Nc + c) * Ns + s) * 2 + 1] = v[s * Nc + c].imag();
      }
    }
  }

  size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };

    template <typename Float, int Ns, int Nc>
      struct SpaceSpinorColorOrder {
      using Accessor = SpaceSpinorColorOrder<Float, Ns, Nc>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      static const int length = 2 * Ns * Nc;
      Float *field;
      size_t offset;
      Float *ghost[8];
      int volumeCB;
      int faceVolumeCB[4];
      int stride;
      int nParity;
      SpaceSpinorColorOrder(const ColorSpinorField &a, int nFace=1, Float *field_=0, float *dummy=0, Float **ghost_=0)
      : field(field_ ? field_ : (Float*)a.V()), offset(a.Bytes()/(2*sizeof(Float))),
    volumeCB(a.VolumeCB()), stride(a.Stride()), nParity(a.SiteSubset())
  {
    if (volumeCB != stride) errorQuda("Stride must equal volume for this field order");
    for (int i=0; i<4; i++) {
      ghost[2*i] = ghost_ ? ghost_[2*i] : 0;
      ghost[2*i+1] = ghost_ ? ghost_[2*i+1] : 0;
      faceVolumeCB[i] = a.SurfaceCB(i)*nFace;
    }
  }

  __device__ __host__ inline void load(complex v[length / 2], int x, int parity = 0) const
  {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
    typedef S<Float,length> structure;
    trove::coalesced_ptr<structure> field_((structure*)field);
    structure v_ = field_[parity*volumeCB + x];
    for (int s=0; s<Ns; s++) {
      for (int c = 0; c < Nc; c++) { v[s * Nc + c] = complex(v_.v[(s * Nc + c) * 2 + 0], v_.v[(s * Nc + c) * 2 + 1]); }
    }
#else
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        v[s * Nc + c] = complex(field[parity * offset + ((x * Ns + s) * Nc + c) * 2 + 0],
                                field[parity * offset + ((x * Ns + s) * Nc + c) * 2 + 1]);
      }
    }
#endif
  }

  __device__ __host__ inline void save(const complex v[length / 2], int x, int parity = 0)
  {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
    typedef S<Float,length> structure;
    trove::coalesced_ptr<structure> field_((structure*)field);
    structure v_;
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        v_.v[(s * Nc + c) * 2 + 0] = v[s * Nc + c].real();
        v_.v[(s * Nc + c) * 2 + 1] = v[s * Nc + c].imag();
      }
    }
    field_[parity*volumeCB + x] = v_;
#else
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        field[parity * offset + ((x * Ns + s) * Nc + c) * 2 + 0] = v[s * Nc + c].real();
        field[parity * offset + ((x * Ns + s) * Nc + c) * 2 + 1] = v[s * Nc + c].imag();
      }
    }
#endif
  }

  /**
     @brief This accessor routine returns a colorspinor_wrapper to this object,
     allowing us to overload various operators for manipulating at
     the site level interms of matrix operations.
     @param[in] x_cb Checkerboarded space-time index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline colorspinor_wrapper<real, Accessor> operator()(int x_cb, int parity)
  {
    return colorspinor_wrapper<real, Accessor>(*this, x_cb, parity);
  }

  /**
     @brief This accessor routine returns a const colorspinor_wrapper to this object,
     allowing us to overload various operators for manipulating at
     the site level interms of matrix operations.
     @param[in] x_cb Checkerboarded space-time index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline const colorspinor_wrapper<real, Accessor> operator()(int x_cb, int parity) const
  {
    return colorspinor_wrapper<real, Accessor>(const_cast<Accessor &>(*this), x_cb, parity);
  }

  __device__ __host__ inline void loadGhost(complex v[length / 2], int x, int dim, int dir, int parity = 0) const
  {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        v[s * Nc + c] = complex(ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 0],
                                ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 1]);
      }
    }
  }

  __device__ __host__ inline void saveGhost(const complex v[length / 2], int x, int dim, int dir, int parity = 0)
  {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 0] = v[s * Nc + c].real();
        ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 1] = v[s * Nc + c].imag();
      }
    }
  }

  size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };

    // custom accessor for TIFR z-halo padded arrays
    template <typename Float, int Ns, int Nc>
      struct PaddedSpaceSpinorColorOrder {
      using Accessor = PaddedSpaceSpinorColorOrder<Float, Ns, Nc>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      static const int length = 2 * Ns * Nc;
      Float *field;
      size_t offset;
      Float *ghost[8];
      int volumeCB;
      int exVolumeCB;
      int faceVolumeCB[4];
      int stride;
      int nParity;
      int dim[4];   // full field dimensions
      int exDim[4]; // full field dimensions
      PaddedSpaceSpinorColorOrder(const ColorSpinorField &a, int nFace=1, Float *field_=0, float *dummy=0, Float **ghost_=0)
      : field(field_ ? field_ : (Float*)a.V()),
    volumeCB(a.VolumeCB()), exVolumeCB(1), stride(a.Stride()), nParity(a.SiteSubset()),
    dim{ a.X(0), a.X(1), a.X(2), a.X(3)}, exDim{ a.X(0), a.X(1), a.X(2) + 4, a.X(3)}
  {
    if (volumeCB != stride) errorQuda("Stride must equal volume for this field order");
    for (int i=0; i<4; i++) {
      ghost[2*i] = ghost_ ? ghost_[2*i] : 0;
      ghost[2*i+1] = ghost_ ? ghost_[2*i+1] : 0;
      faceVolumeCB[i] = a.SurfaceCB(i)*nFace;
      exVolumeCB *= exDim[i];
    }
    exVolumeCB /= nParity;
    dim[0] *= (nParity == 1) ? 2 : 1; // need to full dimensions
    exDim[0] *= (nParity == 1) ? 2 : 1; // need to full dimensions

    offset = exVolumeCB*Ns*Nc*2; // compute manually since Bytes is likely wrong due to z-padding
  }

  /**
     @brief Compute the index into the padded field.  Assumes that
     parity doesn't change from unpadded to padded.
  */
  __device__ __host__ int getPaddedIndex(int x_cb, int parity) const {
    // find coordinates
    int coord[4];
    getCoords(coord, x_cb, dim, parity);

    // get z-extended index
    coord[2] += 2; // offset for halo
    return linkIndex(coord, exDim);
  }

  __device__ __host__ inline void load(complex v[length / 2], int x, int parity = 0) const
  {
    int y = getPaddedIndex(x, parity);

#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
    typedef S<Float,length> structure;
    trove::coalesced_ptr<structure> field_((structure*)field);
    structure v_ = field_[parity*exVolumeCB + y];
    for (int s=0; s<Ns; s++) {
      for (int c = 0; c < Nc; c++) { v[s * Nc + c] = complex(v_.v[(s * Nc + c) * 2 + 0], v_.v[(s * Nc + c) * 2 + 1]); }
    }
#else
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        v[s * Nc + c] = complex(field[parity * offset + ((y * Ns + s) * Nc + c) * 2 + 0],
                                field[parity * offset + ((y * Ns + s) * Nc + c) * 2 + 1]);
      }
    }
#endif
  }

  __device__ __host__ inline void save(const complex v[length / 2], int x, int parity = 0)
  {
    int y = getPaddedIndex(x, parity);

#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
    typedef S<Float,length> structure;
    trove::coalesced_ptr<structure> field_((structure*)field);
    structure v_;
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        v_.v[(s * Nc + c) * 2 + 0] = v[s * Nc + c].real();
        v_.v[(s * Nc + c) * 2 + 1] = v[s * Nc + c].imag();
      }
    }
    field_[parity*exVolumeCB + y] = v_;
#else
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        field[parity * offset + ((y * Ns + s) * Nc + c) * 2 + 0] = v[s * Nc + c].real();
        field[parity * offset + ((y * Ns + s) * Nc + c) * 2 + 1] = v[s * Nc + c].imag();
      }
    }
#endif
  }

  /**
     @brief This accessor routine returns a colorspinor_wrapper to this object,
     allowing us to overload various operators for manipulating at
     the site level interms of matrix operations.
     @param[in] x_cb Checkerboarded space-time index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline colorspinor_wrapper<real, Accessor> operator()(int x_cb, int parity)
  {
    return colorspinor_wrapper<real, Accessor>(*this, x_cb, parity);
  }

  /**
     @brief This accessor routine returns a const colorspinor_wrapper to this object,
     allowing us to overload various operators for manipulating at
     the site level interms of matrix operations.
     @param[in] x_cb Checkerboarded space-time index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline const colorspinor_wrapper<real, Accessor> operator()(int x_cb, int parity) const
  {
    return colorspinor_wrapper<real, Accessor>(const_cast<Accessor &>(*this), x_cb, parity);
  }

  __device__ __host__ inline void loadGhost(complex v[length / 2], int x, int dim, int dir, int parity = 0) const
  {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        v[s * Nc + c] = complex(ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 0],
                                ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 1]);
      }
    }
  }

  __device__ __host__ inline void saveGhost(const complex v[length / 2], int x, int dim, int dir, int parity = 0)
  {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 0] = v[s * Nc + c].real();
        ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 1] = v[s * Nc + c].imag();
      }
    }
  }

  size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };


    template <typename Float, int Ns, int Nc>
      struct QDPJITDiracOrder {
      using Accessor = QDPJITDiracOrder<Float, Ns, Nc>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      Float *field;
      int volumeCB;
      int stride;
      int nParity;
      QDPJITDiracOrder(const ColorSpinorField &a, int nFace=1, Float *field_=0)
      : field(field_ ? field_ : (Float*)a.V()), volumeCB(a.VolumeCB()), stride(a.Stride()), nParity(a.SiteSubset())
  { if (volumeCB != stride) errorQuda("Stride must equal volume for this field order"); }

  __device__ __host__ inline void load(complex v[Ns * Nc], int x, int parity = 0) const
  {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        v[s * Nc + c] = complex(field[(((0 * Nc + c) * Ns + s) * 2 + (1 - parity)) * volumeCB + x],
                                field[(((1 * Nc + c) * Ns + s) * 2 + (1 - parity)) * volumeCB + x]);
      }
    }
  }

  __device__ __host__ inline void save(const complex v[Ns * Nc], int x, int parity = 0)
  {
    for (int s=0; s<Ns; s++) {
      for (int c=0; c<Nc; c++) {
        field[(((0 * Nc + c) * Ns + s) * 2 + (1 - parity)) * volumeCB + x] = v[s * Nc + c].real();
        field[(((1 * Nc + c) * Ns + s) * 2 + (1 - parity)) * volumeCB + x] = v[s * Nc + c].imag();
      }
    }
  }

  /**
     @brief This accessor routine returns a colorspinor_wrapper to this object,
     allowing us to overload various operators for manipulating at
     the site level interms of matrix operations.
     @param[in] x_cb Checkerboarded space-time index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline colorspinor_wrapper<real, Accessor> operator()(int x_cb, int parity)
  {
    return colorspinor_wrapper<real, Accessor>(*this, x_cb, parity);
  }

  /**
     @brief This accessor routine returns a const colorspinor_wrapper to this object,
     allowing us to overload various operators for manipulating at
     the site level interms of matrix operations.
     @param[in] x_cb Checkerboarded space-time index we are requesting
     @param[in] parity Parity we are requesting
     @return Instance of a colorspinor_wrapper that curries in access to
     this field at the above coordinates.
  */
  __device__ __host__ inline const colorspinor_wrapper<real, Accessor> operator()(int x_cb, int parity) const
  {
    return colorspinor_wrapper<real, Accessor>(const_cast<Accessor &>(*this), x_cb, parity);
  }

  size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
      };

  } // namespace colorspinor

  template <typename otherFloat, typename storeFloat>
    __device__ __host__ inline void complex<double>::operator=(const colorspinor::fieldorder_wrapper<otherFloat,storeFloat> &a) {
    x = a.real();
    y = a.imag();
  }

  template <typename otherFloat, typename storeFloat>
    __device__ __host__ inline void complex<float>::operator=(const colorspinor::fieldorder_wrapper<otherFloat,storeFloat> &a) {
    x = a.real();
    y = a.imag();
  }

  template <typename otherFloat, typename storeFloat>
    __device__ __host__ inline complex<double>::complex(const colorspinor::fieldorder_wrapper<otherFloat,storeFloat> &a) {
    x = a.real();
    y = a.imag();
  }

  template <typename otherFloat, typename storeFloat>
    __device__ __host__ inline complex<float>::complex(const colorspinor::fieldorder_wrapper<otherFloat,storeFloat> &a) {
    x = a.real();
    y = a.imag();
  }

  // Use traits to reduce the template explosion
  template <typename T, int Ns, int Nc, bool project = false, bool huge_alloc = false> struct colorspinor_mapper {
  };

  // double precision
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<double, 4, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<double, 4, Nc, 2, false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<double, 4, Nc, true, huge_alloc> {
    typedef colorspinor::FloatNOrder<double, 4, Nc, 2, true, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<double, 2, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<double, 2, Nc, 2, false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<double, 1, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<double, 1, Nc, 2, false, huge_alloc> type;
  };

  // single precision
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<float, 4, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<float, 4, Nc, 4, false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<float, 4, Nc, true, huge_alloc> {
    typedef colorspinor::FloatNOrder<float, 4, Nc, 4, true, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<float, 2, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<float, 2, Nc, 2, false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<float, 1, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<float, 1, Nc, 2, false, huge_alloc> type;
  };

  // half precision
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<short, 4, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<short, 4, Nc, 4, false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<short, 4, Nc, true, huge_alloc> {
    typedef colorspinor::FloatNOrder<short, 4, Nc, 4, true, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<short, 2, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<short, 2, Nc, 2, false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<short, 1, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<short, 1, Nc, 2, false, huge_alloc> type;
  };

  // quarter precision
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<char, 4, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<char, 4, Nc, 4, false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<char, 4, Nc, true, huge_alloc> {
    typedef colorspinor::FloatNOrder<char, 4, Nc, 4, true, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<char, 2, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<char, 2, Nc, 2, false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<char, 1, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<char, 1, Nc, 2, false, huge_alloc> type;
  };

  template<typename T, QudaFieldOrder order, int Ns, int Nc> struct colorspinor_order_mapper { };
  template<typename T, int Ns, int Nc> struct colorspinor_order_mapper<T,QUDA_SPACE_COLOR_SPIN_FIELD_ORDER,Ns,Nc> { typedef colorspinor::SpaceColorSpinorOrder<T, Ns, Nc> type; };
  template<typename T, int Ns, int Nc> struct colorspinor_order_mapper<T,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,Ns,Nc> { typedef colorspinor::SpaceSpinorColorOrder<T, Ns, Nc> type; };
  template<typename T, int Ns, int Nc> struct colorspinor_order_mapper<T,QUDA_FLOAT2_FIELD_ORDER,Ns,Nc> { typedef colorspinor::FloatNOrder<T, Ns, Nc, 2> type; };

} // namespace quda

#endif // _COLOR_SPINOR_ORDER_H
