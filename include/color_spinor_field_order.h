#pragma once

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

#include <limits>
#include <register_traits.h>
#include <convert.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <color_spinor.h>
#include <color_spinor_field.h>
#include <load_store.h>
#include <aos.h>
#include <transform_reduce.h>

namespace quda
{

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
  template <typename Float, typename T> struct colorspinor_wrapper {
    const T &field;
    const int x_cb;
    const int parity;

    /**
       @brief colorspinor_wrapper constructor
       @param[in] a colorspinor field accessor we are wrapping
       @param[in] x_cb checkerboarded space-time index we are accessing
       @param[in] parity Parity we are accessing
    */
    __device__ __host__ inline colorspinor_wrapper<Float, T>(const T &field, int x_cb, int parity) :
      field(field), x_cb(x_cb), parity(parity)
    {
    }

    /**
       @brief Assignment operator with ColorSpinor instance as input
       @param[in] C ColorSpinor we want to store in this accessor
    */
    template <typename C> __device__ __host__ inline void operator=(const C &a) const
    {
      field.save(a.data, x_cb, parity);
    }
  };

  template <typename T, int Nc, int Ns>
  template <typename S>
  __device__ __host__ inline void ColorSpinor<T, Nc, Ns>::operator=(const colorspinor_wrapper<T, S> &a)
  {
    a.field.load(data, a.x_cb, a.parity);
  }

  template <typename T, int Nc, int Ns>
  template <typename S>
  __device__ __host__ inline ColorSpinor<T, Nc, Ns>::ColorSpinor(const colorspinor_wrapper<T, S> &a)
  {
    a.field.load(data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
  template <typename S>
  __device__ __host__ inline void ColorSpinor<T, Nc, 2>::operator=(const colorspinor_wrapper<T, S> &a)
  {
    a.field.load(data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
  template <typename S>
  __device__ __host__ inline ColorSpinor<T, Nc, 2>::ColorSpinor(const colorspinor_wrapper<T, S> &a)
  {
    a.field.load(data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
  template <typename S>
  __device__ __host__ inline void ColorSpinor<T, Nc, 4>::operator=(const colorspinor_wrapper<T, S> &a)
  {
    a.field.load(data, a.x_cb, a.parity);
  }

  template <typename T, int Nc>
  template <typename S>
  __device__ __host__ inline ColorSpinor<T, Nc, 4>::ColorSpinor(const colorspinor_wrapper<T, S> &a)
  {
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
  template <typename Float, typename T> struct colorspinor_ghost_wrapper {
    const int dim;
    const int dir;
    const int ghost_idx;
    const int parity;
    const T &field;

    /**
       @brief colorspinor_ghost_wrapper constructor
       @param[in] a colorspinor field accessor we are wrapping
       @param[in] dim Dimension of the ghost we are accessing
       @param[in] dir Direction of the ghost we are accessing
       @param[in] ghost_idx Checkerboarded space-time ghost index we are accessing
       @param[in] parity Parity we are accessing
    */
    __device__ __host__ inline colorspinor_ghost_wrapper<Float, T>(const T &field, int dim, int dir, int ghost_idx,
                                                                   int parity) :
      dim(dim), dir(dir), ghost_idx(ghost_idx), parity(parity), field(field)
    {
    }

    /**
       @brief Assignment operator with Matrix instance as input
       @param[in] C ColorSpinor we want to store in this accessot
    */
    template <typename C> __device__ __host__ inline void operator=(const C &a) const
    {
      field.saveGhost(a.data, ghost_idx, dim, dir, parity);
    }
  };

  template <typename T, int Nc, int Ns>
  template <typename S>
  __device__ __host__ inline void ColorSpinor<T, Nc, Ns>::operator=(const colorspinor_ghost_wrapper<T, S> &a)
  {
    a.field.loadGhost(data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  template <typename T, int Nc, int Ns>
  template <typename S>
  __device__ __host__ inline ColorSpinor<T, Nc, Ns>::ColorSpinor(const colorspinor_ghost_wrapper<T, S> &a)
  {
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
  __device__ __host__ inline void ColorSpinor<T, Nc, 4>::operator=(const colorspinor_ghost_wrapper<T, S> &a)
  {
    a.field.loadGhost(data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  template <typename T, int Nc>
  template <typename S>
  __device__ __host__ inline ColorSpinor<T, Nc, 4>::ColorSpinor(const colorspinor_ghost_wrapper<T, S> &a)
  {
    a.field.loadGhost(data, a.ghost_idx, a.dim, a.dir, a.parity);
  }

  namespace colorspinor
  {

    template <typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order> struct AccessorCB {
      AccessorCB(const ColorSpinorField &) { errorQuda("Not implemented"); }
      AccessorCB() { errorQuda("Not implemented"); }
      AccessorCB(const AccessorCB &) { errorQuda("Not implemented"); }
      AccessorCB &operator=(const AccessorCB &) { errorQuda("Not implemented"); }
      constexpr int index(int, int, int, int, int, int) const { return 0; }
    };

    template <typename Float, int nSpin, int nColor, int nVec, QudaFieldOrder order> struct GhostAccessorCB {
      GhostAccessorCB(const ColorSpinorField &) { errorQuda("Not implemented"); }
      GhostAccessorCB() { errorQuda("Not implemented"); }
      GhostAccessorCB(const GhostAccessorCB &) { errorQuda("Not implemented"); }
      GhostAccessorCB &operator=(const GhostAccessorCB &) { errorQuda("Not implemented"); }
      constexpr int index(int, int, int, int, int, int, int) const { return 0; }
    };

    template <typename Float, int nSpin, int nColor, int nVec>
    struct AccessorCB<Float, nSpin, nColor, nVec, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> {
      int offset_cb = 0;
      AccessorCB(const ColorSpinorField &field) : offset_cb((field.Bytes() >> 1) / sizeof(complex<Float>)) { }
      AccessorCB() = default;
      AccessorCB(const AccessorCB &) = default;
      AccessorCB &operator=(const AccessorCB &) = default;

      /**
       * @brief This method returns the index for the pointer that
       * points to the start of the memory chunk corresponds to the
       * matrix at parity, x_cb, s, c, v.
       * @param parity Parity index
       * @param x_cb 1-d checkboarding site index
       * @param s spin index
       * @param c color index
       * @param v vector index
       */
      constexpr int index(int parity, int x_cb, int s, int c, int v, int) const
      {
        return parity * offset_cb + ((x_cb * nSpin + s) * nColor + c) * nVec + v;
      }

      template <int nSpinBlock>
      __device__ __host__ inline void load(complex<Float> out[nSpinBlock * nColor * nVec], complex<Float> *in,
                                           int parity, int x_cb, int chi, int) const
      {
        using vec_t = typename VectorType<Float, 2>::type;
        constexpr int N = nSpin * nColor * nVec;
        constexpr int M = nSpinBlock * nColor * nVec;
#pragma unroll
        for (int i = 0; i < M; i++) {
          vec_t tmp
            = vector_load<vec_t>(reinterpret_cast<const vec_t *>(in + parity * offset_cb), x_cb * N + chi * M + i);
          memcpy(&out[i], &tmp, sizeof(vec_t));
        }
      }
    };

    template <typename Float, int nSpin, int nColor, int nVec>
    struct GhostAccessorCB<Float, nSpin, nColor, nVec, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> {
      int faceVolumeCB[4] = {};
      int ghostOffset[4] = {};
      GhostAccessorCB(const ColorSpinorField &a, int nFace = 1)
      {
        for (int d = 0; d < 4; d++) {
          faceVolumeCB[d] = nFace * a.SurfaceCB(d);
          ghostOffset[d] = faceVolumeCB[d] * nColor * nSpin * nVec;
        }
      }
      GhostAccessorCB() = default;
      GhostAccessorCB(const GhostAccessorCB &) = default;
      GhostAccessorCB &operator=(const GhostAccessorCB &) = default;

      constexpr int index(int dim, int parity, int x_cb, int s, int c, int v) const
      {
        return parity * ghostOffset[dim] + ((x_cb * nSpin + s) * nColor + c) * nVec + v;
      }
    };

    template <int nSpin, int nColor, int nVec, int N> // note this will not work for N=1
    constexpr int indexFloatN(int x_cb, int s, int c, int v, int stride)
    {
      int k = (s * nColor + c) * nVec + v;
      int j = k / (N / 2);
      int i = k % (N / 2);
      return (j * stride + x_cb) * (N / 2) + i;
    };

    template <typename Float, int nSpin, int nColor, int nVec>
    struct AccessorCB<Float, nSpin, nColor, nVec, QUDA_FLOAT2_FIELD_ORDER> {
      int offset_cb = 0;
      AccessorCB(const ColorSpinorField &field) : offset_cb((field.Bytes() >> 1) / sizeof(complex<Float>)) { }
      AccessorCB() = default;
      AccessorCB(const AccessorCB &) = default;
      AccessorCB &operator=(const AccessorCB &) = default;

      constexpr int index(int parity, int x_cb, int s, int c, int v, int stride) const
      {
        return parity * offset_cb + ((s * nColor + c) * nVec + v) * stride + x_cb;
      }

      template <int nSpinBlock>
      __device__ __host__ inline void load(complex<Float> out[nSpinBlock * nColor * nVec], complex<Float> *in,
                                           int parity, int x_cb, int chi, int stride) const
      {
        using vec_t = typename VectorType<Float, 2>::type;
        constexpr int M = nSpinBlock * nColor * nVec;
#pragma unroll
        for (int i = 0; i < M; i++) {
          vec_t tmp = vector_load<vec_t>(reinterpret_cast<const vec_t *>(in + parity * offset_cb),
                                         (chi * M + i) * stride + x_cb);
          memcpy(&out[i], &tmp, sizeof(vec_t));
        }
      }
    };

    // specialized varient for packed half precision staggered
    template <> struct AccessorCB<short, 1, 3, 1, QUDA_FLOAT2_FIELD_ORDER> {
      int offset_cb = 0;
      AccessorCB(const ColorSpinorField &field) : offset_cb((field.Bytes() >> 1) / sizeof(complex<short>)) { }
      AccessorCB() = default;
      AccessorCB(const AccessorCB &) = default;
      AccessorCB &operator=(const AccessorCB &) = default;

      constexpr int index(int parity, int x_cb, int s, int c, int v, int stride) const
      {
        return parity * offset_cb + ((s * 3 + c) * 1 + v) * stride + x_cb;
      }

      template <int nSpinBlock>
      __device__ __host__ inline void load(complex<short> out[3], complex<short> *in, int parity, int x_cb, int, int) const
      {
        using vec_t = typename VectorType<float, 4>::type;
        vec_t tmp = vector_load<vec_t>(reinterpret_cast<const vec_t *>(in + parity * offset_cb), x_cb);
        memcpy(out, &tmp, 3 * sizeof(complex<short>));
      }
    };

    template <typename Float, int nSpin, int nColor, int nVec>
    struct GhostAccessorCB<Float, nSpin, nColor, nVec, QUDA_FLOAT2_FIELD_ORDER> {
      int faceVolumeCB[4] = {};
      int ghostOffset[4] = {};
      GhostAccessorCB(const ColorSpinorField &a, int nFace = 1)
      {
        for (int d = 0; d < 4; d++) {
          faceVolumeCB[d] = nFace * a.SurfaceCB(d);
          ghostOffset[d] = faceVolumeCB[d] * nColor * nSpin * nVec;
        }
      }
      GhostAccessorCB() = default;
      GhostAccessorCB(const GhostAccessorCB &) = default;
      GhostAccessorCB &operator=(const GhostAccessorCB &) = default;

      constexpr int index(int dim, int parity, int x_cb, int s, int c, int v) const
      {
        return parity * ghostOffset[dim] + ((s * nColor + c) * nVec + v) * faceVolumeCB[dim] + x_cb;
      }
    };

    template <typename Float, int nSpin, int nColor, int nVec>
    struct AccessorCB<Float, nSpin, nColor, nVec, QUDA_FLOAT4_FIELD_ORDER> {
      int offset_cb = 0;
      AccessorCB(const ColorSpinorField &field) : offset_cb((field.Bytes() >> 1) / sizeof(complex<Float>)) { }
      AccessorCB() = default;
      AccessorCB(const AccessorCB &) = default;
      AccessorCB &operator=(const AccessorCB &) = default;

      constexpr int index(int parity, int x_cb, int s, int c, int v, int stride) const
      {
        return parity * offset_cb + indexFloatN<nSpin, nColor, nVec, 4>(x_cb, s, c, v, stride);
      }

      template <int nSpinBlock>
      __device__ __host__ inline void load(complex<Float> out[nSpinBlock * nColor * nVec], complex<Float> *in,
                                           int parity, int x_cb, int chi, int stride) const
      {
        using vec_t = typename VectorType<Float, 4>::type;
        constexpr int M = (nSpinBlock * nColor * nVec * 2) / 4;
#pragma unroll
        for (int i = 0; i < M; i++) {
          vec_t tmp = vector_load<vec_t>(reinterpret_cast<const vec_t *>(in + parity * offset_cb),
                                         (chi * M + i) * stride + x_cb);
          memcpy(&out[i * 2], &tmp, sizeof(vec_t));
        }
      }
    };

    template <typename Float, int nSpin, int nColor, int nVec>
    struct GhostAccessorCB<Float, nSpin, nColor, nVec, QUDA_FLOAT4_FIELD_ORDER> {
      int faceVolumeCB[4] = {};
      int ghostOffset[4] = {};
      GhostAccessorCB(const ColorSpinorField &a, int nFace = 1)
      {
        for (int d = 0; d < 4; d++) {
          faceVolumeCB[d] = nFace * a.SurfaceCB(d);
          ghostOffset[d] = faceVolumeCB[d] * nColor * nSpin * nVec;
        }
      }
      GhostAccessorCB() = default;
      GhostAccessorCB(const GhostAccessorCB &) = default;
      GhostAccessorCB &operator=(const GhostAccessorCB &) = default;

      constexpr int index(int dim, int parity, int x_cb, int s, int c, int v) const
      {
        return parity * ghostOffset[dim] + indexFloatN<nSpin, nColor, nVec, 4>(x_cb, s, c, v, faceVolumeCB[dim]);
      }
    };

    template <typename Float, int nSpin, int nColor, int nVec>
    struct AccessorCB<Float, nSpin, nColor, nVec, QUDA_FLOAT8_FIELD_ORDER> {
      int offset_cb = 0;
      AccessorCB(const ColorSpinorField &field) : offset_cb((field.Bytes() >> 1) / sizeof(complex<Float>)) { }
      AccessorCB() = default;
      AccessorCB(const AccessorCB &) = default;
      AccessorCB &operator=(const AccessorCB &) = default;

      constexpr int index(int parity, int x_cb, int s, int c, int v, int stride) const
      {
        return parity * offset_cb + indexFloatN<nSpin, nColor, nVec, 8>(x_cb, s, c, v, stride);
      }

      template <int nSpinBlock>
      __device__ __host__ inline void load(complex<Float> out[nSpinBlock * nColor * nVec], complex<Float> *in,
                                           int parity, int x_cb, int chi, int stride) const
      {
        using vec_t = typename VectorType<Float, 8>::type;

        // in case the vector length isn't divisible by 8, load in the entire vector and then pick the chirality
        // (the compiler will remove any unused loads)
        constexpr int N = nSpin * nColor * nVec * 2; // real numbers in the loaded vector
        constexpr int M = N / 8;
        Float tmp[N];
#pragma unroll
        for (int i = 0; i < M; i++) {
          vec_t ld_tmp = vector_load<vec_t>(reinterpret_cast<const vec_t *>(in + parity * offset_cb), i * stride + x_cb);
          memcpy(&tmp[i * 8], &ld_tmp, sizeof(vec_t));
        }
        constexpr int N_chi = N / (nSpin / nSpinBlock);
#pragma unroll
        for (int i = 0; i < N_chi; i++)
          out[i] = complex<Float>(tmp[chi * N_chi + 2 * i + 0], tmp[chi * N_chi + 2 * i + 1]);
      }
    };

    template <typename Float, int nSpin, int nColor, int nVec>
    struct GhostAccessorCB<Float, nSpin, nColor, nVec, QUDA_FLOAT8_FIELD_ORDER> {
      int faceVolumeCB[4] = {};
      int ghostOffset[4] = {};
      GhostAccessorCB(const ColorSpinorField &a, int nFace = 1)
      {
        for (int d = 0; d < 4; d++) {
          faceVolumeCB[d] = nFace * a.SurfaceCB(d);
          ghostOffset[d] = faceVolumeCB[d] * nColor * nSpin * nVec;
        }
      }
      GhostAccessorCB() = default;
      GhostAccessorCB(const GhostAccessorCB &) = default;
      GhostAccessorCB &operator=(const GhostAccessorCB &) = default;

      constexpr int index(int dim, int parity, int x_cb, int s, int c, int v) const
      {
        return parity * ghostOffset[dim] + indexFloatN<nSpin, nColor, nVec, 8>(x_cb, s, c, v, faceVolumeCB[dim]);
      }
    };

    template <typename Float, typename storeFloat> constexpr bool fixed_point() { return false; }
    template <> constexpr bool fixed_point<double, int8_t>() { return true; }
    template <> constexpr bool fixed_point<double, short>() { return true; }
    template <> constexpr bool fixed_point<double, int>() { return true; }
    template <> constexpr bool fixed_point<float, int8_t>() { return true; }
    template <> constexpr bool fixed_point<float, short>() { return true; }
    template <> constexpr bool fixed_point<float, int>() { return true; }

    /**
       @brief fieldorder_wrapper is an internal class that is used to
       wrap instances of FieldOrder accessors, currying in the
       specific location on the field.  This is used as a helper class
       for fixed-point accessors providing the necessary conversion
       and scaling when writing to a fixed-point field.
    */
    template <typename Float, typename storeFloat, bool block_float_, typename norm_t> struct fieldorder_wrapper {
      using value_type = Float;      /**< Compute type */
      using store_type = storeFloat; /**< Storage type */
      complex<storeFloat> *v;        /**< Field memory address this wrapper encompasses */
      const int idx;                 /**< Index into field */
    private:
      const Float scale;             /**< Float to fixed-point scale factor */
      const Float scale_inv;         /**< Fixed-point to float scale factor */
    public:
      norm_t *norm;                  /**< Address of norm field (if it exists) */
      const int norm_idx;            /**< Index into norm field */
      const bool norm_write;         /**< Whether we need to write to the norm field */
      static constexpr bool fixed = fixed_point<Float, storeFloat>(); /**< Whether this is a fixed point field */
      static constexpr bool block_float = block_float_;               /**< Whether this is a block float field */

      /**
         @brief fieldorder_wrapper constructor
         @param idx Field index
      */
      __device__ __host__ inline fieldorder_wrapper(complex<storeFloat> *v, int idx, Float scale, Float scale_inv,
                                                    norm_t *norm = nullptr, int norm_idx = 0, bool norm_write = false) :
        v(v), idx(idx), scale(scale), scale_inv(scale_inv), norm(norm), norm_idx(norm_idx), norm_write(norm_write)
      {
      }

      fieldorder_wrapper(const fieldorder_wrapper<Float, storeFloat, block_float_, norm_t> &a) = delete;

      fieldorder_wrapper(fieldorder_wrapper<Float, storeFloat, block_float_, norm_t> &&a) = default;

      /**
         @brief Assignment operator with complex number instance as input
         @param a Complex number we want to store in this accessor
      */
      template <typename theirFloat> __device__ __host__ inline void operator=(const complex<theirFloat> &a) const
      {
        // we only ever write the norm out if we are doing block
        // float format, and if specifically requested (in general,
        // this will be a specific thread that requests this (norm_write = true)
        if (block_float && norm_write) norm[norm_idx] = scale_inv;
        if (std::is_same_v<storeFloat, theirFloat>) {
          v[idx] = complex<storeFloat>(a.real(), a.imag());
        } else {
          v[idx] = fixed ?
            complex<storeFloat>(f2i_round<storeFloat>(scale * a.real()), f2i_round<storeFloat>(scale * a.imag())) :
            complex<storeFloat>(a.real(), a.imag());
        }
      }

      /**
         @brief Assignment operator with fieldorder_wrapper instance as input
         @param a fieldorder_wrapper we are copying from
      */
      __device__ __host__ inline void operator=(const fieldorder_wrapper<Float, storeFloat, block_float_, norm_t> &a) const
      {
        *this = complex<Float>(a);
      }

      /**
         @brief Assignment operator with fieldorder_wrapper instance as input
         @param a fieldorder_wrapper we are copying from
      */
      template <typename theirFloat, typename theirStoreFloat, bool their_block_float, typename their_norm_t>
      __device__ __host__ inline void
      operator=(const fieldorder_wrapper<theirFloat, theirStoreFloat, their_block_float, their_norm_t> &a) const
      {
        *this = complex<Float>(a);
      }

      /**
         @brief Assignment operator with real number instance as input
         @param a real number we want to store in this accessor
      */
      template <typename theirFloat> __device__ __host__ inline void operator=(const theirFloat &a) const
      {
        *this = complex<Float>(static_cast<Float>(a), static_cast<Float>(0.0));
      }

      /**
         @brief complex cast operator
      */
      __device__ __host__ inline operator complex<Float>() const
      {
        if (!fixed) {
          complex<storeFloat> tmp = v[idx];
          return complex<Float>(tmp.real(), tmp.imag());
        } else {
          complex<storeFloat> tmp = v[idx];
          Float norm_ = block_float ? norm[norm_idx] : scale_inv;
          return norm_ * complex<Float>(static_cast<Float>(tmp.real()), static_cast<Float>(tmp.imag()));
        }
      }

      /**
         @brief complex cast operator to a different precision
      */
      template <typename theirFloat> __device__ __host__ inline operator complex<theirFloat>() const
      {
        auto out = static_cast<complex<Float>>(*this);
        return complex<theirFloat>(out.real(), out.imag());
      }

      /**
       * @brief returns the pointer of this wrapper object
       */
      __device__ __host__ inline auto data() const { return &v[idx]; }

      /**
       * @brief returns the scale of this wrapper object
       */
      __device__ __host__ inline auto get_scale() const
      {
        static_assert(block_float == false, "Orders with block_float == true should not call the get_scale method.");
        return block_float ? static_cast<Float>(1) / norm[norm_idx] : scale;
      }

      /**
       * @brief returns the scale_inv of this wrapper object
       */
      __device__ __host__ inline auto get_scale_inv() const { return block_float ? norm[norm_idx] : scale_inv; }

      /**
         @brief Operator+= with complex number instance as input
         @param a Complex number we want to add to this accessor
      */
      template <typename theirFloat> __device__ __host__ inline void operator+=(const complex<theirFloat> &a) const
      {
        *this = complex<Float>(*this) + complex<Float>(a);
      }

      /**
         @brief Operator-= with complex number instance as input
         @param a Complex number we want to subtract from this accessor
      */
      template <typename theirFloat> __device__ __host__ inline void operator-=(const complex<theirFloat> &a) const
      {
        *this += (-a);
      }
    };

    template <typename Float, typename storeFloat, bool block_float, typename norm_t>
    __device__ __host__ inline complex<Float>
    operator*(const Float &a, const fieldorder_wrapper<Float, storeFloat, block_float, norm_t> &b)
    {
      return a * complex<Float>(b);
    }

    template <typename Float, typename storeFloat, bool block_float, typename norm_t>
    __device__ __host__ inline complex<Float>
    operator*(const fieldorder_wrapper<Float, storeFloat, block_float, norm_t> &a, const Float &b)
    {
      return complex<Float>(a) * b;
    }

    template <typename Float, typename storeFloat, bool block_float, typename norm_t>
    __device__ __host__ inline complex<Float>
    operator*(const complex<Float> &a, const fieldorder_wrapper<Float, storeFloat, block_float, norm_t> &b)
    {
      return a * complex<Float>(b);
    }

    template <typename Float, typename storeFloat, bool block_float, typename norm_t>
    __device__ __host__ inline complex<Float>
    operator*(const fieldorder_wrapper<Float, storeFloat, block_float, norm_t> &a, const complex<Float> &b)
    {
      return complex<Float>(a) * b;
    }

    template <typename Float, typename storeFloat, bool block_float, typename norm_t>
    __device__ __host__ inline complex<Float> conj(const fieldorder_wrapper<Float, storeFloat, block_float, norm_t> &a)
    {
      return conj(static_cast<complex<Float>>(a));
    }

    template <typename Float, int nSpin_, int nColor_, int nVec, QudaFieldOrder order, typename storeFloat,
              typename ghostFloat, bool disable_ghost = false>
    class GhostOrder
    {
    protected:
      GhostOrder() = default;
      GhostOrder(const GhostOrder &) = default;
      GhostOrder(const ColorSpinorField &, int, void *const *) { }
      GhostOrder &operator=(const GhostOrder &) = default;

    public:
      /** Does this field type support ghost zones? */
      static constexpr bool supports_ghost_zone = false;
    };

    template <typename store_t, typename norm_t, bool ghost_fixed, bool block_float_ghost> struct ghost_t {
      complex<store_t> *ghost[8] = {};
      constexpr auto &operator[](int idx) { return ghost[idx]; }
    };

    template <typename store_t, typename norm_t> struct ghost_t<store_t, norm_t, true, true> {
      complex<store_t> *ghost[8] = {};
      norm_t *norm_[8] = {};
      constexpr auto &operator[](int idx) { return ghost[idx]; }
      constexpr auto &norm(int idx) { return norm_[idx]; }
    };

    template <typename store_t, typename norm_t> struct ghost_t<store_t, norm_t, true, false> {
      complex<store_t> *ghost[8] = {};
      norm_t scale = 1.0;
      norm_t scale_inv = 1.0;
      constexpr auto &operator[](int idx) { return ghost[idx]; }
    };

    template <typename Float, int nSpin_, int nColor_, int nVec, QudaFieldOrder order, typename storeFloat, typename ghostFloat>
    class GhostOrder<Float, nSpin_, nColor_, nVec, order, storeFloat, ghostFloat, false>
    {
      using norm_t = float;
      static constexpr int nSpin = nSpin_;
      static constexpr int nColor = nColor_;
      static constexpr bool fixed = fixed_point<Float, storeFloat>();
      static constexpr bool ghost_fixed = fixed_point<Float, ghostFloat>();
      static constexpr bool block_float_ghost = !fixed && ghost_fixed;

      mutable ghost_t<ghostFloat, norm_t, ghost_fixed, block_float_ghost> ghost;
      int nParity = 0;
      using ghost_accessor_t = GhostAccessorCB<ghostFloat, nSpin, nColor, nVec, order>;
      ghost_accessor_t ghostAccessor;

    public:
      /** Does this field type support ghost zones? */
      static constexpr bool supports_ghost_zone = true;

      GhostOrder() = default;
      GhostOrder(const GhostOrder &) = default;

      GhostOrder(const ColorSpinorField &field, int nFace, void *const *ghost_ = nullptr) :
        nParity(field.SiteSubset()), ghostAccessor(field, nFace)
      {
        resetGhost(ghost_ ? ghost_ : field.Ghost());
        resetScale(field.Scale());
      }

      GhostOrder &operator=(const GhostOrder &) = default;

      void resetScale(Float max)
      {
        if (block_float_ghost && max != static_cast<Float>(1.0))
          errorQuda("Block-float accessor requires max=1.0 not max=%e", max);
        if constexpr (ghost_fixed && !block_float_ghost) {
          ghost.scale = static_cast<Float>(std::numeric_limits<ghostFloat>::max() / max);
          ghost.scale_inv = static_cast<Float>(max / std::numeric_limits<ghostFloat>::max());
        }
      }

      void resetGhost(void *const *ghost_) const
      {
        for (int dim = 0; dim < 4; dim++) {
          for (int dir = 0; dir < 2; dir++) {
            ghost[2 * dim + dir] = static_cast<complex<ghostFloat> *>(ghost_[2 * dim + dir]);
            if constexpr (block_float_ghost)
              ghost.norm(2 * dim + dir) = reinterpret_cast<norm_t *>(
                static_cast<char *>(ghost_[2 * dim + dir])
                + nParity * nColor * nSpin * nVec * 2 * ghostAccessor.faceVolumeCB[dim] * sizeof(ghostFloat));
          }
        }
      }

      /**
       * Complex-member accessor function for the ghost zone.  The
       * parameter n is only used for indexed into the packed
       * null-space vectors.
       * @param x 1-d checkerboard site index
       * @param s spin index
       * @param c color index
       * @param n vector number
       * @param max site-element max (only when writing in block-float format)
       */
      __device__ __host__ inline auto Ghost(int dim, int dir, int parity, int x_cb, int s, int c, int n = 0,
                                            Float max = 0) const
      {
        norm_t *norm_ptr = nullptr;
        norm_t scale = 1.0;
        norm_t scale_inv = 1.0;
        if constexpr (ghost_fixed) {
          if constexpr (block_float_ghost) {
            norm_ptr = ghost.norm(2 * dim + dir);
            scale = fdividef(fixedMaxValue<ghostFloat>::value, max);
            scale_inv = fixedInvMaxValue<ghostFloat>::value * max;
          } else {
            scale = ghost.scale;
            scale_inv = ghost.scale_inv;
          }
        }
        return fieldorder_wrapper<Float, ghostFloat, block_float_ghost, norm_t>(
          ghost[2 * dim + dir], ghostAccessor.index(dim, parity, x_cb, s, c, n), scale, scale_inv, norm_ptr,
          parity * ghostAccessor.faceVolumeCB[dim] + x_cb, s == 0 && c == 0 && n == 0);
      }

      /** Returns the number of field parities (1 or 2) */
      constexpr int Nparity() const { return nParity; }

      /**
         @brief Wrapper to transform_reduce which is called by the
         reducer functions, e.g., norm2 and abs_max
         @tparam reducer The reduction operation we which to apply
         @param[in] Dimension of the ghost we are concerned with
         @param[in] location The location of execution
         @param[in] nParity Number of parities of the field
         @param[in] volumeCB Checkerboard volume
         @param[in] h The helper functor which acts as the transformer
         in transform_reduce
       */
      template <typename reducer, typename helper>
      auto transform_reduce(int dim, QudaFieldLocation location, int nParity, helper h) const
      {
        // separate norms for forwards and backwards
        if constexpr (fixed && block_float_ghost) {
          errorQuda("Reduction not defined");
        } else {
          std::vector<complex<ghostFloat> *> g {ghost[2 * dim + 0], ghost[2 * dim + 1]};
          std::vector<typename reducer::reduce_t> result(2);
          ::quda::transform_reduce<reducer>(
            location, result, g, unsigned(nParity * ghostAccessor.faceVolumeCB[dim] * nSpin * nColor * nVec), h);
          return result;
        }
      }

      /**
       * Returns the L2 norm squared of the ghost elements in a given dimension
       * @param[in] field Field instance we use to source some meta data
       * @param[in] dim Dimension of the ghost we are concerned with
       * @param[in] global Whether to do a global or process local norm2 reduction
       * @return L2 norm squared
       */
      auto ghost_norm2(const ColorSpinorField &field, int dim, bool global = true) const
      {
        commGlobalReductionPush(global);
        Float scale_inv = 1.0;
        if constexpr (fixed && !block_float_ghost) scale_inv = ghost.scale_inv;
        auto nrm2 = transform_reduce<plus<double>>(dim, field.Location(), field.SiteSubset(),
                                                   square_<double, ghostFloat>(scale_inv));
        commGlobalReductionPop();
        return nrm2;
      }

      /**
       * Returns the Linfinity norm of the field
       * @param[in] field Field instance we use to source some meta data
       * @param[in] dim Dimension of the ghost we are concerned with
       * @param[in] global Whether to do a global or process local norm2 reduction
       * @return Linfinity norm
       */
      auto ghost_abs_max(const ColorSpinorField &field, bool global = true) const
      {
        commGlobalReductionPush(global);
        Float scale_inv = 1.0;
        if constexpr (fixed && !block_float_ghost) scale_inv = ghost.scale_inv;
        auto absmax = transform_reduce<maximum<Float>>(field.Location(), field.SiteSubset(),
                                                       abs_max_<Float, ghostFloat>(scale_inv));
        commGlobalReductionPop();
        return absmax;
      }
    };

    template <typename real, typename store_t, bool fixed, bool block_float> struct field {
      complex<store_t> *v = nullptr;
    };

    template <typename real, typename store_t> struct field<real, store_t, true, false> {
      complex<store_t> *v = nullptr;
      real scale = 1.0;
      real scale_inv = 1.0;
    };

    template <typename real, typename store_t> struct field<real, store_t, true, true> {
      using norm_t = float;
      complex<store_t> *v = nullptr;
      norm_t *norm = nullptr;
      int norm_offset = 0;
    };

    template <typename Float, int nSpin_, int nColor_, int nVec, QudaFieldOrder order, typename storeFloat = Float,
              typename ghostFloat = storeFloat, bool disable_ghost = false, bool block_float = false>
    class FieldOrderCB : public GhostOrder<Float, nSpin_, nColor_, nVec, order, storeFloat, ghostFloat, disable_ghost>
    {
      static_assert((block_float && nVec == 1) || !block_float, "Not supported");
      using GhostOrder = GhostOrder<Float, nSpin_, nColor_, nVec, order, storeFloat, ghostFloat, disable_ghost>;
      using norm_t = float;

    public:
      static constexpr bool fixed = fixed_point<Float, storeFloat>();
      static constexpr int nSpin = nSpin_;
      static constexpr int nColor = nColor_;

      field<Float, storeFloat, fixed, block_float> v;
      unsigned int volumeCB = 0;

    protected:
      using accessor_t = AccessorCB<storeFloat, nSpin, nColor, nVec, order>;
      accessor_t accessor;

    public:
      using real = Float;
      FieldOrderCB() = default;
      FieldOrderCB(const FieldOrderCB &) = default;

      /**
       * Constructor for the FieldOrderCB class
       * @param field The field that we are accessing
       */
      FieldOrderCB(const ColorSpinorField &field, int nFace = 1, void *const v_ = 0, void *const *ghost_ = 0) :
        GhostOrder(field, nFace, ghost_), volumeCB(field.VolumeCB()), accessor(field)
      {
        v.v = v_ ? static_cast<complex<storeFloat> *>(const_cast<void *>(v_)) :
                   static_cast<complex<storeFloat> *>(const_cast<void *>(field.V()));
        resetScale(field.Scale());

        if constexpr (fixed && block_float) {
          if constexpr (nColor == 3 && nSpin == 1 && nVec == 1 && order == 2)
            // special case where the norm is packed into the per site struct
            v.norm = reinterpret_cast<norm_t *>(const_cast<void *>(field.V()));
          else
            v.norm = static_cast<norm_t *>(const_cast<void *>(field.Norm()));
          v.norm_offset = field.Bytes() / (2 * sizeof(norm_t));
        }
      }

      FieldOrderCB &operator=(const FieldOrderCB &) = default;

      void resetScale(Float max)
      {
        if (block_float && max != static_cast<Float>(1.0))
          errorQuda("Block-float accessor requires max=1.0 not max=%e", max);
        if constexpr (fixed && !block_float) {
          v.scale = static_cast<Float>(std::numeric_limits<storeFloat>::max() / max);
          v.scale_inv = static_cast<Float>(max / std::numeric_limits<storeFloat>::max());
        }
        if constexpr (GhostOrder::supports_ghost_zone) GhostOrder::resetScale(max);
      }

      /**
       * Read-only accessor function.  This specialized load returns
       * the entire site vector for a given chirality
       * @tparam nSpinBlock The number of spin components in a chiral block
       * @param[out] out, The loaded site vector
       * @param[in] parity The site parity
       * @param[in] x_cb 1-d checkerboard site index
       * @param[in] chi The desired chirality
       */
      template <int nSpinBlock>
      __device__ __host__ inline void load(complex<Float> out[nSpinBlock * nColor * nVec], int parity, int x_cb,
                                           int chi) const
      {
        if (!fixed) {
          accessor.template load<nSpinBlock>((complex<storeFloat> *)out, v.v, parity, x_cb, chi, volumeCB);
        } else {
          complex<storeFloat> tmp[nSpinBlock * nColor * nVec];
          accessor.template load<nSpinBlock>(tmp, v.v, parity, x_cb, chi, volumeCB);

          Float norm_ = 0.0;
          if constexpr (fixed) {
            if constexpr (block_float) {
              if constexpr (nColor == 3 && nSpin == 1 && nVec == 1)
                // special case where the norm is packed into the per site struct
                norm_ = v.norm[parity * v.norm_offset + 4 * x_cb + 3];
              else
                norm_ = v.norm[parity * v.norm_offset + x_cb];
            } else {
              norm_ = v.scale_inv;
            }
          }
#pragma unroll
          for (int s = 0; s < nSpinBlock; s++) {
#pragma unroll
            for (int c = 0; c < nColor; c++) {
#pragma unroll
              for (int v = 0; v < nVec; v++) {
                int k = (s * nColor + c) * nVec + v;
                out[k] = norm_ * complex<Float>(static_cast<Float>(tmp[k].real()), static_cast<Float>(tmp[k].imag()));
              }
            }
          }
        }
      }

      /**
       * Complex-member accessor function.  The parameter n is only
       * used for indexed into the packed null-space vectors.
       * @param x 1-d checkerboard site index
       * @param s spin index
       * @param c color index
       * @param n vector number
       */
      __device__ __host__ inline auto operator()(int parity, int x_cb, int s, int c, int n = 0) const
      {
        Float scale = 1.0;
        Float scale_inv = 1.0;
        norm_t *norm = nullptr;
        int norm_offset = 0;
        if constexpr (fixed) {
          if constexpr (block_float) {
            norm = v.norm;
            norm_offset = v.norm_offset;
          } else {
            scale = v.scale;
            scale_inv = v.scale_inv;
          }
        }
        return fieldorder_wrapper<Float, storeFloat, block_float, norm_t>(
          v.v, accessor.index(parity, x_cb, s, c, n, volumeCB), scale, scale_inv, norm, parity * norm_offset + x_cb);
      }

      /** Returns the number of field colors */
      constexpr int Ncolor() const { return nColor; }

      /** Returns the number of field spins */
      constexpr int Nspin() const { return nSpin; }

      /** Returns the number of packed vectors (for mg prolongator) */
      constexpr int Nvec() const { return nVec; }

      /** Returns the field volume */
      constexpr int VolumeCB() const { return volumeCB; }

      /**
         @brief Wrapper to transform_reduce which is called by the
         reducer functions, e.g., norm2 and abs_max
         @tparam reducer The reduction operation we which to apply
         @param[in] location The location of execution
         @param[in] nParity Number of parities of the field
         @param[in] volumeCB Checkerboard volume
         @param[in] h The helper functor which acts as the transformer
         in transform_reduce
       */
      template <typename reducer, typename helper>
      auto transform_reduce(QudaFieldLocation location, int nParity, helper h) const
      {
        std::vector<decltype(v.v)> v_eo(nParity);
        for (auto i = 0u; i < v_eo.size(); i++) v_eo[i] = v.v + i * accessor.offset_cb;
        std::vector<typename reducer::reduce_t> result(nParity);

        ::quda::transform_reduce<reducer>(location, result, v_eo, volumeCB * nSpin * nColor * nVec, h);

        auto total = reducer::init();
        for (auto &res : result) total = reducer::apply(total, res);
        return total;
      }

      /**
       * Returns the L2 norm squared of the field
       * @param[in] field  Field instance we use to soruce some metadata
       * @param[in] global Whether to do a global or process local norm2 reduction
       * @return L2 norm squared
       */
      auto norm2(const ColorSpinorField &field, bool global = true) const
      {
        commGlobalReductionPush(global);
        Float scale_inv = 1.0;
        if constexpr (fixed && !block_float) scale_inv = v.scale_inv;
        auto nrm2
          = transform_reduce<plus<double>>(field.Location(), field.SiteSubset(), square_<double, storeFloat>(scale_inv));
        commGlobalReductionPop();
        return nrm2;
      }

      /**
       * Returns the Linfinity norm of the field
       * @param[in] field  Field instance we use to soruce some metadata
       * @param[in] global Whether to do a global or process local Linfinity reduction
       * @return Linfinity norm
       */
      auto abs_max(const ColorSpinorField &field, bool global = true) const
      {
        commGlobalReductionPush(global);
        Float scale_inv = 1.0;
        if constexpr (fixed && !block_float) scale_inv = v.scale_inv;
        auto absmax = transform_reduce<maximum<Float>>(field.Location(), field.SiteSubset(),
                                                       abs_max_<Float, storeFloat>(scale_inv));
        commGlobalReductionPop();
        return absmax;
      }
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
      int volumeCB;
      int faceVolumeCB[4];
      mutable Float *ghost[8];
      mutable norm_type *ghost_norm[8];
      int nParity;
      void *backup_h; //! host memory for backing up the field when tuning
      size_t bytes;

      FloatNOrder(const ColorSpinorField &a, int nFace = 1, Float *buffer = 0, Float **ghost_ = 0) :
        field(buffer ? buffer : (Float *)a.V()),
        norm(buffer ? reinterpret_cast<norm_type *>(reinterpret_cast<char *>(buffer) + a.NormOffset()) :
                      const_cast<norm_type *>(reinterpret_cast<const norm_type *>(a.Norm()))),
        offset(a.Bytes() / (2 * sizeof(Float) * N)),
        norm_offset(a.Bytes() / (2 * sizeof(norm_type))),
        volumeCB(a.VolumeCB()),
        nParity(a.SiteSubset()),
        backup_h(nullptr),
        bytes(a.Bytes())
      {
        for (int i = 0; i < 4; i++) { faceVolumeCB[i] = a.SurfaceCB(i) * nFace; }
        resetGhost(ghost_ ? (void **)ghost_ : a.Ghost());
      }

      void resetGhost(void *const *ghost_) const
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
        norm_type nrm = isFixed<Float>::value ? vector_load<float>(norm, x + parity * norm_offset) : 0.0;

#pragma unroll
        for (int i = 0; i < M; i++) {
          // first load from memory
          Vector vecTmp = vector_load<Vector>(field, parity * offset + x + volumeCB * i);
          // now copy into output and scale
#pragma unroll
          for (int j = 0; j < N; j++) copy_and_scale(v[i * N + j], reinterpret_cast<Float *>(&vecTmp)[j], nrm);
        }

#pragma unroll
        for (int i = 0; i < length / 2; i++) out[i] = complex(v[2 * i + 0], v[2 * i + 1]);
      }

      __device__ __host__ inline void save(const complex in[length / 2], int x, int parity = 0) const
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
          for (int i = 0; i < length / 2; i++)
            max_[i] = fmaxf(fabsf((norm_type)v[i]), fabsf((norm_type)v[i + length / 2]));
          norm_type scale = 0.0;
#pragma unroll
          for (int i = 0; i < length / 2; i++) scale = fmaxf(max_[i], scale);
          norm[x + parity * norm_offset] = scale * fixedInvMaxValue<Float>::value;

          real scale_inv = fdividef(fixedMaxValue<Float>::value, scale);
#pragma unroll
          for (int i = 0; i < length; i++) v[i] = v[i] * scale_inv;
        }

#pragma unroll
        for (int i = 0; i < M; i++) {
          Vector vecTmp;
          // first do scalar copy converting into storage type
#pragma unroll
          for (int j = 0; j < N; j++) copy_scaled(reinterpret_cast<Float *>(&vecTmp)[j], v[i * N + j]);
          // second do vectorized copy into memory
          vector_store(field, parity * offset + x + volumeCB * i, vecTmp);
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
      __device__ __host__ inline auto operator()(int x_cb, int parity) const
      {
        return colorspinor_wrapper<real, Accessor>(*this, x_cb, parity);
      }

      __device__ __host__ inline void loadGhost(complex out[length_ghost / 2], int x, int dim, int dir, int parity = 0) const
      {
        real v[length_ghost];
        norm_type nrm
          = isFixed<Float>::value ? vector_load<float>(ghost_norm[2 * dim + dir], parity * faceVolumeCB[dim] + x) : 0.0;

#pragma unroll
        for (int i = 0; i < M_ghost; i++) {
          GhostVector vecTmp = vector_load<GhostVector>(
            ghost[2 * dim + dir], parity * faceVolumeCB[dim] * M_ghost + i * faceVolumeCB[dim] + x);
#pragma unroll
          for (int j = 0; j < N_ghost; j++)
            copy_and_scale(v[i * N_ghost + j], reinterpret_cast<Float *>(&vecTmp)[j], nrm);
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
            max_[i] = fmaxf((norm_type)fabsf((norm_type)v[i]), (norm_type)fabsf((norm_type)v[i + length_ghost / 2]));
          norm_type scale = 0.0;
#pragma unroll
          for (int i = 0; i < length_ghost / 2; i++) scale = fmaxf(max_[i], scale);
          ghost_norm[2 * dim + dir][parity * faceVolumeCB[dim] + x] = scale * fixedInvMaxValue<Float>::value;

          real scale_inv = fdividef(fixedMaxValue<Float>::value, scale);
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
      __device__ __host__ inline auto Ghost(int dim, int dir, int ghost_idx, int parity) const
      {
        return colorspinor_ghost_wrapper<real, Accessor>(*this, dim, dir, ghost_idx, parity);
      }

      /**
         @brief Backup the field to the host when tuning
      */
      void save()
      {
        if (backup_h) errorQuda("Already allocated host backup");
        backup_h = safe_malloc(bytes);
        qudaMemcpy(backup_h, field, bytes, qudaMemcpyDeviceToHost);
      }

      /**
         @brief Restore the field from the host after tuning
      */
      void load()
      {
        qudaMemcpy(field, backup_h, bytes, qudaMemcpyHostToDevice);
        host_free(backup_h);
        backup_h = nullptr;
      }

      size_t Bytes() const
      {
        return nParity * volumeCB * (Nc * Ns * 2 * sizeof(Float) + (isFixed<Float>::value ? sizeof(norm_type) : 0));
      }
    };

    /**
       @brief Accessor routine for ColorSpinorFields in native field
       order.  Specialization for half-precision staggered QCD fields
       where we pack each site into a 128-bit word (int4).
       @tparam N Number of real numbers per short vector.  Ignored in this specialization.
       @tparam spin_project Whether the ghosts are spin projected or not
       @tparam huge_alloc Template parameter that enables 64-bit
       pointer arithmetic for huge allocations (e.g., packed set of
       vectors).  Default is to use 32-bit pointer arithmetic.
     */
    template <int N_, bool spin_project, bool huge_alloc>
    struct FloatNOrder<short, 1, 3, N_, spin_project, huge_alloc> {
      using Float = short;
      static constexpr int Ns = 1;
      static constexpr int Nc = 3;
      static constexpr int length = 2 * Ns * Nc;
      static constexpr int length_ghost = 2 * Ns * Nc;
      using Accessor = FloatNOrder<Float, Ns, Nc, N_, spin_project, huge_alloc>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      using Vector = int4;      // 128-bit packed type
      using GhostVector = int4; // 128-bit packed type
      using AllocInt = typename AllocType<huge_alloc>::type;
      using norm_type = float;
      Float *field;
      const AllocInt offset; // offset can be 32-bit or 64-bit
      int volumeCB;
      int faceVolumeCB[4];
      mutable Float *ghost[8];
      int nParity;
      void *backup_h; //! host memory for backing up the field when tuning
      size_t bytes;

      FloatNOrder(const ColorSpinorField &a, int nFace = 1, Float *buffer = 0, Float **ghost_ = 0) :
        field(buffer ? buffer : (Float *)a.V()),
        offset(a.Bytes() / (2 * sizeof(Vector))),
        volumeCB(a.VolumeCB()),
        nParity(a.SiteSubset()),
        backup_h(nullptr),
        bytes(a.Bytes())
      {
        for (int i = 0; i < 4; i++) { faceVolumeCB[i] = a.SurfaceCB(i) * nFace; }
        resetGhost(ghost_ ? (void **)ghost_ : a.Ghost());
      }

      void resetGhost(void *const *ghost_) const
      {
        for (int dim = 0; dim < 4; dim++) {
          for (int dir = 0; dir < 2; dir++) {
            ghost[2 * dim + dir] = comm_dim_partitioned(dim) ? static_cast<Float *>(ghost_[2 * dim + dir]) : nullptr;
          }
        }
      }

      __device__ __host__ inline void load(complex out[length / 2], int x, int parity = 0) const
      {
        real v[length];
        Vector vecTmp = vector_load<Vector>(field, parity * offset + x);

        // extract the norm
        norm_type nrm;
        memcpy(&nrm, &vecTmp.w, sizeof(norm_type));

        // now copy into output and scale
#pragma unroll
        for (int i = 0; i < length; i++) copy_and_scale(v[i], reinterpret_cast<Float *>(&vecTmp)[i], nrm);

#pragma unroll
        for (int i = 0; i < length / 2; i++) out[i] = complex(v[2 * i + 0], v[2 * i + 1]);
      }

      __device__ __host__ inline void save(const complex in[length / 2], int x, int parity = 0) const
      {
        real v[length];

#pragma unroll
        for (int i = 0; i < length / 2; i++) {
          v[2 * i + 0] = in[i].real();
          v[2 * i + 1] = in[i].imag();
        }

        norm_type max_[length / 2];
        // two-pass to increase ILP (assumes length divisible by two, e.g. complex-valued)
#pragma unroll
        for (int i = 0; i < length / 2; i++)
          max_[i] = fmaxf(fabsf((norm_type)v[i]), fabsf((norm_type)v[i + length / 2]));
        norm_type scale = 0.0;
#pragma unroll
        for (int i = 0; i < length / 2; i++) scale = fmaxf(max_[i], scale);
        norm_type nrm = scale * fixedInvMaxValue<Float>::value;

        real scale_inv = fdividef(fixedMaxValue<Float>::value, scale);
#pragma unroll
        for (int i = 0; i < length; i++) v[i] = v[i] * scale_inv;

        Vector vecTmp;
        memcpy(&vecTmp.w, &nrm, sizeof(norm_type)); // pack the norm

        // pack the spinor elements
#pragma unroll
        for (int i = 0; i < length; i++) copy_scaled(reinterpret_cast<Float *>(&vecTmp)[i], v[i]);

        vector_store(field, parity * offset + x, vecTmp);
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
      __device__ __host__ inline auto operator()(int x_cb, int parity) const
      {
        return colorspinor_wrapper<real, Accessor>(*this, x_cb, parity);
      }

      __device__ __host__ inline void loadGhost(complex out[length_ghost / 2], int x, int dim, int dir, int parity = 0) const
      {
        real v[length_ghost];
        GhostVector vecTmp = vector_load<GhostVector>(ghost[2 * dim + dir], parity * faceVolumeCB[dim] + x);

        // extract the norm
        norm_type nrm;
        memcpy(&nrm, &vecTmp.w, sizeof(norm_type));

#pragma unroll
        for (int i = 0; i < length_ghost; i++) copy_and_scale(v[i], reinterpret_cast<Float *>(&vecTmp)[i], nrm);

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

        norm_type max_[length_ghost / 2];
        // two-pass to increase ILP (assumes length divisible by two, e.g. complex-valued)
#pragma unroll
        for (int i = 0; i < length_ghost / 2; i++)
          max_[i] = fmaxf(fabsf((norm_type)v[i]), fabsf((norm_type)v[i + length_ghost / 2]));
        norm_type scale = 0.0;
#pragma unroll
        for (int i = 0; i < length_ghost / 2; i++) scale = fmaxf(max_[i], scale);
        norm_type nrm = scale * fixedInvMaxValue<Float>::value;

        real scale_inv = fdividef(fixedMaxValue<Float>::value, scale);
#pragma unroll
        for (int i = 0; i < length_ghost; i++) v[i] = v[i] * scale_inv;

        GhostVector vecTmp;
        memcpy(&vecTmp.w, &nrm, sizeof(norm_type)); // pack the norm

        // pack the spinor elements
#pragma unroll
        for (int i = 0; i < length_ghost; i++) copy_scaled(reinterpret_cast<Float *>(&vecTmp)[i], v[i]);
        vector_store(ghost[2 * dim + dir], parity * faceVolumeCB[dim] + x, vecTmp);
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
      __device__ __host__ inline auto Ghost(int dim, int dir, int ghost_idx, int parity) const
      {
        return colorspinor_ghost_wrapper<real, Accessor>(*this, dim, dir, ghost_idx, parity);
      }

      /**
         @brief Backup the field to the host when tuning
      */
      void save()
      {
        if (backup_h) errorQuda("Already allocated host backup");
        backup_h = safe_malloc(bytes);
        qudaMemcpy(backup_h, field, bytes, qudaMemcpyDeviceToHost);
      }

      /**
         @brief Restore the field from the host after tuning
      */
      void load()
      {
        qudaMemcpy(field, backup_h, bytes, qudaMemcpyHostToDevice);
        host_free(backup_h);
        backup_h = nullptr;
      }

      size_t Bytes() const
      {
        return nParity * volumeCB * (Nc * Ns * 2 * sizeof(Float) + (isFixed<Float>::value ? sizeof(norm_type) : 0));
      }
    };

    template <typename Float, int Ns, int Nc> struct SpaceColorSpinorOrder {
      using Accessor = SpaceColorSpinorOrder<Float, Ns, Nc>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      static const int length = 2 * Ns * Nc;
      Float *field;
      size_t offset;
      Float *ghost[8];
      int volumeCB;
      int faceVolumeCB[4];
      int nParity;
      SpaceColorSpinorOrder(const ColorSpinorField &a, int nFace = 1, Float *field_ = 0, float * = 0, Float **ghost_ = 0) :
        field(field_ ? field_ : (Float *)a.V()),
        offset(a.Bytes() / (2 * sizeof(Float))),
        volumeCB(a.VolumeCB()),
        nParity(a.SiteSubset())
      {
        for (int i = 0; i < 4; i++) {
          ghost[2 * i] = ghost_ ? ghost_[2 * i] : 0;
          ghost[2 * i + 1] = ghost_ ? ghost_[2 * i + 1] : 0;
          faceVolumeCB[i] = a.SurfaceCB(i) * nFace;
        }
      }

      __device__ __host__ inline void load(complex v[length / 2], int x, int parity = 0) const
      {
        auto in = &field[(parity * volumeCB + x) * length];
        complex v_[length / 2];
        block_load<complex, length / 2>(v_, reinterpret_cast<const complex *>(in));

        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) { v[s * Nc + c] = v_[c * Ns + s]; }
        }
      }

      __device__ __host__ inline void save(const complex v[length / 2], int x, int parity = 0) const
      {
        auto out = &field[(parity * volumeCB + x) * length];
        complex v_[length / 2];
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) { v_[c * Ns + s] = v[s * Nc + c]; }
        }

        block_store<complex, length / 2>(reinterpret_cast<complex *>(out), v_);
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
      __device__ __host__ inline auto operator()(int x_cb, int parity) const
      {
        return colorspinor_wrapper<real, Accessor>(*this, x_cb, parity);
      }

      __device__ __host__ inline void loadGhost(complex v[length / 2], int x, int dim, int dir, int parity = 0) const
      {
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) {
            v[s * Nc + c]
              = complex(ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Nc + c) * Ns + s) * 2 + 0],
                        ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Nc + c) * Ns + s) * 2 + 1]);
          }
        }
      }

      __device__ __host__ inline void saveGhost(const complex v[length / 2], int x, int dim, int dir, int parity = 0) const
      {
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) {
            ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Nc + c) * Ns + s) * 2 + 0] = v[s * Nc + c].real();
            ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Nc + c) * Ns + s) * 2 + 1] = v[s * Nc + c].imag();
          }
        }
      }

      size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
    };

    template <typename Float, int Ns, int Nc> struct SpaceSpinorColorOrder {
      using Accessor = SpaceSpinorColorOrder<Float, Ns, Nc>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      static const int length = 2 * Ns * Nc;
      Float *field;
      size_t offset;
      Float *ghost[8];
      int volumeCB;
      int faceVolumeCB[4];
      int nParity;
      SpaceSpinorColorOrder(const ColorSpinorField &a, int nFace = 1, Float *field_ = 0, float * = 0, Float **ghost_ = 0) :
        field(field_ ? field_ : (Float *)a.V()),
        offset(a.Bytes() / (2 * sizeof(Float))),
        volumeCB(a.VolumeCB()),
        nParity(a.SiteSubset())
      {
        for (int i = 0; i < 4; i++) {
          ghost[2 * i] = ghost_ ? ghost_[2 * i] : 0;
          ghost[2 * i + 1] = ghost_ ? ghost_[2 * i + 1] : 0;
          faceVolumeCB[i] = a.SurfaceCB(i) * nFace;
        }
      }

      __device__ __host__ inline void load(complex v[length / 2], int x, int parity = 0) const
      {
        auto in = &field[(parity * volumeCB + x) * length];
        block_load<complex, length / 2>(v, reinterpret_cast<const complex *>(in));
      }

      __device__ __host__ inline void save(const complex v[length / 2], int x, int parity = 0) const
      {
        auto out = &field[(parity * volumeCB + x) * length];
        block_store<complex, length / 2>(reinterpret_cast<complex *>(out), v);
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
      __device__ __host__ inline auto operator()(int x_cb, int parity) const
      {
        return colorspinor_wrapper<real, Accessor>(*this, x_cb, parity);
      }

      __device__ __host__ inline void loadGhost(complex v[length / 2], int x, int dim, int dir, int parity = 0) const
      {
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) {
            v[s * Nc + c]
              = complex(ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 0],
                        ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 1]);
          }
        }
      }

      __device__ __host__ inline void saveGhost(const complex v[length / 2], int x, int dim, int dir, int parity = 0) const
      {
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) {
            ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 0] = v[s * Nc + c].real();
            ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 1] = v[s * Nc + c].imag();
          }
        }
      }

      size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
    };

    // custom accessor for TIFR z-halo padded arrays
    template <typename Float, int Ns, int Nc> struct PaddedSpaceSpinorColorOrder {
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
      int nParity;
      int dim[4];   // full field dimensions
      int exDim[4]; // full field dimensions
      PaddedSpaceSpinorColorOrder(const ColorSpinorField &a, int nFace = 1, Float *field_ = 0, float * = 0,
                                  Float **ghost_ = 0) :
        field(field_ ? field_ : (Float *)a.V()),
        volumeCB(a.VolumeCB()),
        exVolumeCB(1),
        nParity(a.SiteSubset()),
        dim {a.X(0), a.X(1), a.X(2), a.X(3)},
        exDim {a.X(0), a.X(1), a.X(2) + 4, a.X(3)}
      {
        for (int i = 0; i < 4; i++) {
          ghost[2 * i] = ghost_ ? ghost_[2 * i] : 0;
          ghost[2 * i + 1] = ghost_ ? ghost_[2 * i + 1] : 0;
          faceVolumeCB[i] = a.SurfaceCB(i) * nFace;
          exVolumeCB *= exDim[i];
        }
        exVolumeCB /= nParity;
        dim[0] *= (nParity == 1) ? 2 : 1;   // need to full dimensions
        exDim[0] *= (nParity == 1) ? 2 : 1; // need to full dimensions

        offset = exVolumeCB * Ns * Nc * 2; // compute manually since Bytes is likely wrong due to z-padding
      }

      /**
         @brief Compute the index into the padded field.  Assumes that
         parity doesn't change from unpadded to padded.
      */
      __device__ __host__ int getPaddedIndex(int x_cb, int parity) const
      {
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
        auto in = &field[(parity * exVolumeCB + y) * length];
        block_load<complex, length / 2>(v, reinterpret_cast<const complex *>(in));
      }

      __device__ __host__ inline void save(const complex v[length / 2], int x, int parity = 0) const
      {
        int y = getPaddedIndex(x, parity);
        auto out = &field[(parity * exVolumeCB + y) * length];
        block_store<complex, length / 2>(reinterpret_cast<complex *>(out), v);
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
      __device__ __host__ inline auto operator()(int x_cb, int parity) const
      {
        return colorspinor_wrapper<real, Accessor>(*this, x_cb, parity);
      }

      __device__ __host__ inline void loadGhost(complex v[length / 2], int x, int dim, int dir, int parity = 0) const
      {
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) {
            v[s * Nc + c]
              = complex(ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 0],
                        ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 1]);
          }
        }
      }

      __device__ __host__ inline void saveGhost(const complex v[length / 2], int x, int dim, int dir, int parity = 0) const
      {
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) {
            ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 0] = v[s * Nc + c].real();
            ghost[2 * dim + dir][(((parity * faceVolumeCB[dim] + x) * Ns + s) * Nc + c) * 2 + 1] = v[s * Nc + c].imag();
          }
        }
      }

      size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
    };

    template <typename Float, int Ns, int Nc> struct QDPJITDiracOrder {
      using Accessor = QDPJITDiracOrder<Float, Ns, Nc>;
      using real = typename mapper<Float>::type;
      using complex = complex<real>;
      Float *field;
      int volumeCB;
      int nParity;
      QDPJITDiracOrder(const ColorSpinorField &a, int = 1, Float *field_ = 0, float * = 0) :
        field(field_ ? field_ : (Float *)a.V()), volumeCB(a.VolumeCB()), nParity(a.SiteSubset())
      {
      }

      __device__ __host__ inline void load(complex v[Ns * Nc], int x, int parity = 0) const
      {
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) {
            v[s * Nc + c] = complex(field[(((0 * Nc + c) * Ns + s) * 2 + (1 - parity)) * volumeCB + x],
                                    field[(((1 * Nc + c) * Ns + s) * 2 + (1 - parity)) * volumeCB + x]);
          }
        }
      }

      __device__ __host__ inline void save(const complex v[Ns * Nc], int x, int parity = 0) const
      {
        for (int s = 0; s < Ns; s++) {
          for (int c = 0; c < Nc; c++) {
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
      __device__ __host__ inline auto operator()(int x_cb, int parity) const
      {
        return colorspinor_wrapper<real, Accessor>(*this, x_cb, parity);
      }

      size_t Bytes() const { return nParity * volumeCB * Nc * Ns * 2 * sizeof(Float); }
    };

  } // namespace colorspinor

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
    typedef colorspinor::FloatNOrder<float, 2, Nc, colorspinor::getNative<float>(2), false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<float, 1, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<float, 1, Nc, 2, false, huge_alloc> type;
  };

  // half precision
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<short, 4, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<short, 4, Nc, colorspinor::getNative<short>(4), false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<short, 4, Nc, true, huge_alloc> {
    typedef colorspinor::FloatNOrder<short, 4, Nc, colorspinor::getNative<short>(4), true, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<short, 2, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<short, 2, Nc, colorspinor::getNative<short>(2), false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<short, 1, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<short, 1, Nc, 2, false, huge_alloc> type;
  };

  // quarter precision
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<int8_t, 4, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<int8_t, 4, Nc, colorspinor::getNative<int8_t>(4), false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<int8_t, 4, Nc, true, huge_alloc> {
    typedef colorspinor::FloatNOrder<int8_t, 4, Nc, colorspinor::getNative<int8_t>(4), true, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<int8_t, 2, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<int8_t, 2, Nc, colorspinor::getNative<int8_t>(2), false, huge_alloc> type;
  };
  template <int Nc, bool huge_alloc> struct colorspinor_mapper<int8_t, 1, Nc, false, huge_alloc> {
    typedef colorspinor::FloatNOrder<int8_t, 1, Nc, 2, false, huge_alloc> type;
  };

  template <typename T, QudaFieldOrder order, int Ns, int Nc> struct colorspinor_order_mapper {
  };
  template <typename T, int Ns, int Nc> struct colorspinor_order_mapper<T, QUDA_SPACE_COLOR_SPIN_FIELD_ORDER, Ns, Nc> {
    typedef colorspinor::SpaceColorSpinorOrder<T, Ns, Nc> type;
  };
  template <typename T, int Ns, int Nc> struct colorspinor_order_mapper<T, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER, Ns, Nc> {
    typedef colorspinor::SpaceSpinorColorOrder<T, Ns, Nc> type;
  };
  template <typename T, int Ns, int Nc> struct colorspinor_order_mapper<T, QUDA_FLOAT2_FIELD_ORDER, Ns, Nc> {
    typedef colorspinor::FloatNOrder<T, Ns, Nc, 2> type;
  };

} // namespace quda
