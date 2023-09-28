#pragma once

/**
 * @file  clover_field_order.h
 * @brief Main header file for host and device accessors to CloverFields
 *
 */

#include <limits>
#include <register_traits.h>
#include <convert.h>
#include <clover_field.h>
#include <complex_quda.h>
#include <quda_matrix.h>
#include <color_spinor.h>
#include <load_store.h>
#include <aos.h>
#include <transform_reduce.h>

namespace quda {

  /**
     @brief clover_wrapper is an internal class that is used to
     wrap instances of clover accessors, currying in a specifc
     location and chirality on the field.  The operator() accessors in
     clover-field accessors return instances to this class,
     allowing us to then use operator overloading upon this class
     to interact with the HMatrix class.  As a result we can
     include clover-field accessors directly in HMatrix
     expressions in kernels without having to declare temporaries
     with explicit calls to the load/save methods in the
     clover-field accessors.
  */
  template <typename Float, typename T>
    struct clover_wrapper {
    const T &field;
    const int x_cb;
    const int parity;
    const int chirality;

    /**
       @brief clover_wrapper constructor
       @param[in] a clover field accessor we are wrapping
       @param[in] x_cb checkerboarded space-time index we are accessing
       @param[in] parity Parity we are accessing
       @param[in] chirality Chirality we are accessing
    */
    __device__ __host__ inline clover_wrapper<Float, T>(const T &field, int x_cb, int parity, int chirality) :
      field(field), x_cb(x_cb), parity(parity), chirality(chirality)
    {
    }

      /**
	 @brief Assignment operator with H matrix instance as input
	 @param[in] C ColorSpinor we want to store in this accessor
      */
      template <typename C> __device__ __host__ inline void operator=(const C &a) const
      {
        field.save(a.data, x_cb, parity, chirality);
      }
    };

  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline void HMatrix<T,N>::operator=(const clover_wrapper<T,S> &a) {
    if (a.chirality == 0) a.field.load(data, a.x_cb, a.parity, 0);
    else                  a.field.load(data, a.x_cb, a.parity, 1);
  }

  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline HMatrix<T,N>::HMatrix(const clover_wrapper<T,S> &a) {
    if (a.chirality == 0) a.field.load(data, a.x_cb, a.parity, 0);
    else                  a.field.load(data, a.x_cb, a.parity, 1);
  }

  namespace clover {

    template <typename real, int block, bool enable_reconstruct> struct reconstruct_t {

      real diagonal;

      constexpr reconstruct_t(real diagonal) : diagonal(diagonal) { }

      /** Length of compressed block */
      static constexpr int compressed_block_size() { return 28; }

      // map from in-kernel internal order to storaage order for a chiral block with Nc = 3
      constexpr auto pack_idx(int i) const
      {
        constexpr int order[] = {0,  1,  2,  3,  28, 29,                             // diagonal elements
                                 4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, // off diagonals
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35};
        return order[i];
      }

      // inverse of pack_idx
      constexpr auto unpack_idx(int i) const
      {
        constexpr int order[] = {0,  1,  2,  3,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 4,  5,  30, 31, 32, 33, 34, 35};
        return order[i];
      }

      constexpr auto compress_idx(int i) const
      {
        constexpr int order[] = {0,  1,  2,                                      // uncompressed diagonals
                                 0,  1,  2,                                      // compressed diagonals
                                 6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, // uncompressed off diagonals
                                 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, // uncompressed off diagonals
                                 6,  7,  8,  9,  16, 17};                        // compressed off diagonals
        return order[i];
      }

      constexpr auto pack_compress_idx(int i) const
      {
        constexpr int order[] = {0,  1,  2,                                          // uncompressed diagonals
                                 0,  1,  2,                                          // compressed diagonals
                                 4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, // uncompressed off diagonals
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,         // uncompressed off diagonals
                                 4,  5,  6,  7,  14, 15};                            // compressed off diagonals
        return order[i];
      }

      template <typename T> constexpr T decompress(const T &in, int k) const
      {
        switch (k) {
        case 0:
        case 1:
        case 2: return diagonal + in;
        case 3:
        case 4:
        case 5: return diagonal - in;
        case 30:
        case 31:
        case 32:
        case 33:
        case 34:
        case 35: return -in;
        default: return in;
        }
      }

      template <typename T1, typename T2> __device__ __host__ inline void unpack(T1 &out, const T2 &in) const
      {
#pragma unroll
        for (int i = 0; i < compressed_block_size(); i++) out[unpack_idx(i)] = in[i];

          // first reconstruct second set of diagonal elements before we reconstruct the first set
#pragma unroll
        for (int i = 0; i < 3; i++) out[i + 3] = diagonal - out[i];
#pragma unroll
        for (int i = 0; i < 3; i++) out[i + 0] = diagonal + out[i];

        out[30] = -out[6];
        out[31] = -out[7];
        out[32] = -out[8];
        out[33] = -out[9];
        out[34] = -out[16];
        out[35] = -out[17];
      }

      template <typename T1, typename T2> __device__ __host__ inline void pack(T1 &out, const T2 &in) const
      {
#pragma unroll
        for (int i = 0; i < compressed_block_size(); i++) out[i] = in[unpack_idx(i)];
          // remove diagonal constant
#pragma unroll
        for (int i = 0; i < 3; i++) out[i] -= diagonal;
        out[3] = 0.0; // intentionally zero this so that it can't contribute to the max element
      }
    };

    template <typename real, int block> struct reconstruct_t<real, block, false> {

      real diagonal;
      constexpr reconstruct_t(real diagonal) : diagonal(diagonal) { }
      static constexpr auto compressed_block_size() { return block; } /** Length of compressed block */
      constexpr auto pack_idx(int i) const { return i; }
      constexpr auto compress_idx(int i) const { return i; }
      constexpr auto pack_compress_idx(int i) const { return i; }

      template <typename T> constexpr T decompress(const T &in, int) const { return in; }

      template <typename T1, typename T2> __device__ __host__ inline void unpack(T1 &out, const T2 &in) const
      {
#pragma unroll
        for (int i = 0; i < block; i++) out[i] = in[i];
      }

      template <typename T1, typename T2> __device__ __host__ inline void pack(T1 &out, const T2 &in) const
      {
#pragma unroll
        for (int i = 0; i < block; i++) out[i] = in[i];
      }
    };

    /**
       The internal ordering for each clover matrix has chirality as the
       slowest running dimension, with the internal 36 degrees of
       freedom stored as follows (s=spin, c = color)

       i |  row  |  col  |
           s   c   s   c   z
       0   0   0   0   0   0
       1   0   1   0   1   0
       2   0   2   0   2   0
       3   1   0   1   0   0
       4   1   1   1   1   0
       5   1   2   1   2   0
       6   0   1   0   0   0
       7   0   1   0   0   1
       8   0   2   0   0   0
       9   0   2   0   0   1
       10  1   0   0   0   0
       11  1   0   0   0   1
       12  1   1   0   0   0
       13  1   1   0   0   1
       14  1   2   0   0   0
       15  1   2   0   0   1
       16  0   2   0   1   0
       17  0   2   0   1   1
       18  1   0   0   1   0
       19  1   0   0   1   1
       20  1   1   0   1   0
       21  1   1   0   1   1
       22  1   2   0   1   0
       23  1   2   0   1   1
       24  1   0   0   2   0
       25  1   0   0   2   1
       26  1   1   0   2   0
       27  1   1   0   2   1
       28  1   2   0   2   0
       29  1   2   0   2   1
       30  1   1   1   0   0
       31  1   1   1   0   1
       32  1   2   1   0   0
       33  1   2   1   0   1
       34  1   2   1   1   0
       35  1   2   1   1   1

       For each chirality the first 6 entires are the pure real
       diagonal entries.  The following 30 entries correspond to the
       15 complex numbers on the strictly lower triangular.

       E.g., N = 6 (2 spins x 3 colors) and
       # entries = 1/2 * N * (N-1)

       The storage order on the strictly lower triangular is column
       major, which complicates the indexing, since we have to count
       backwards from the end of the array.

       // psuedo code in lieu of implementation
       int row = s_row*3 + c_row;
       int col = s_col*3 + c_col;
       if (row == col) {
         return complex(a[row])
       } else if (col < row) {

         // below we find the offset into each chiral half.  First
         // compute the offset into the strictly lower triangular
         // part, counting from the lower right.  This requires we
         // change to prime coordinates.
         int row' = N - row;
         int col' = N - col;

         // The linear offset (in bottom-right coordinates) to the
         // required element is simply 1/2*col'*(col'-1) + col - row.
         // Subtract this offset from the number of elements: N=6,
         // means 15 elements (14 with C-style indexing)), multiply by
         // two to account for complexity and then add on number of
         // real diagonals at the end

         int k = 2 * ( (1/2 N*(N-1) -1) - (1/2 * col' * (col'-1) + col - row) + N;
         return complex(a[2*k], a[2*k+1]);
       } else {
         conj(swap(col,row));
       }

    */

    template<typename Float, int nColor, int nSpin, QudaCloverFieldOrder order> struct Accessor {
      Accessor(const CloverField &, bool = false) { errorQuda("Not implemented for order %d", order); }
      constexpr complex<Float> &operator()(int, int, int, int, int, int) const { return complex<Float>(0.0); }

      /**
         @brief Wrapper to transform_reduce which is called by the
         reducer functions, e.g., norm2 and abs_max.  This is the
         dummy implementation in the non-specialized Accessor class.
         @tparam reducer The reduction operation we which to apply
         @tparam helper The helper functor which acts as the transformer
         in transform_reduce
      */
      template <typename reducer, typename helper> constexpr double transform_reduce(QudaFieldLocation, helper) const
      {
        return 0.0;
      }
    };

    template <int N> constexpr int indexFloatN(int k, int stride, int x)
    {
      int j = k / N;
      int i = k % N;
      return (j * stride + x) * N + i;
    };

    template <typename Float, int nColor, int nSpin, QudaCloverFieldOrder order> struct FloatNAccessor {
      const Float *a;
      const int stride;
      const size_t offset_cb;
      const int compressed_block_size;
      static constexpr int N = nColor * nSpin / 2;
      reconstruct_t<Float, N * N, clover::reconstruct()> recon;
      FloatNAccessor(const CloverField &A, bool inverse = false) :
        a(static_cast<Float *>(const_cast<void *>(A.V(inverse)))),
        stride(A.VolumeCB()),
        offset_cb(A.Bytes() / (2 * sizeof(Float))),
        compressed_block_size(A.compressed_block_size()),
        recon(A.Diagonal())
      {
      }

      __device__ __host__ inline complex<Float> operator()(int parity, int x, int chirality, int s_row, int s_col,
                                                           int c_row, int c_col) const
      {
        int row = s_row * nColor + c_row;
        int col = s_col * nColor + c_col;
        const Float *a_ = a + parity * offset_cb + stride * chirality * compressed_block_size;

        if (row == col) {
          auto a = a_[indexFloatN<order>(recon.pack_compress_idx(row), stride, x)];
          return static_cast<Float>(2.0) * complex<Float>(recon.decompress(a, row));
        } else if (col < row) {
          // switch coordinates to count from bottom right instead of top left of matrix
	  int k = N*(N-1)/2 - (N-col)*(N-col-1)/2 + row - col - 1;
          int idx = N + 2*k;

          auto a = complex<Float>(a_[indexFloatN<order>(recon.pack_compress_idx(idx + 0), stride, x)],
                                  a_[indexFloatN<order>(recon.pack_compress_idx(idx + 1), stride, x)]);
          return static_cast<Float>(2.0)
            * complex<Float>(recon.decompress(a.real(), idx), recon.decompress(a.imag(), idx + 1));
        } else {
          // requesting upper triangular so return conjugate transpose
	  // switch coordinates to count from bottom right instead of top left of matrix
	  int k = N*(N-1)/2 - (N-row)*(N-row-1)/2 + col - row - 1;
          int idx = N + 2*k;

          auto a = complex<Float>(a_[indexFloatN<order>(recon.pack_compress_idx(idx + 0), stride, x)],
                                  a_[indexFloatN<order>(recon.pack_compress_idx(idx + 1), stride, x)]);
          return static_cast<Float>(2.0)
            * complex<Float>(recon.decompress(a.real(), idx), -recon.decompress(a.imag(), idx + 1));
        }
      }

      /**
         @brief Wrapper to transform_reduce which is called by the
         reducer functions, e.g., norm2 and abs_max
         @tparam reducer The reduction operation we which to apply
         @param[in] location The location of execution
         @param[in] h The helper functor which acts as the transformer
         in transform_reduce
       */
      template <typename reducer, typename helper>
      __host__ double transform_reduce(QudaFieldLocation location, helper h) const
      {
        // just use offset_cb, since factor of two from parity is equivalent to complexity
        return ::quda::transform_reduce<reducer>(location, reinterpret_cast<const complex<Float> *>(a), offset_cb, h);
      }

      constexpr Float scale() const { return static_cast<Float>(2.0); } // normalization of native storage
    };

    template <typename Float, int nColor, int nSpin>
    struct Accessor<Float, nColor, nSpin, QUDA_FLOAT2_CLOVER_ORDER>
      : FloatNAccessor<Float, nColor, nSpin, QUDA_FLOAT2_CLOVER_ORDER> {
      Accessor(const CloverField &A, bool inverse = false) :
        FloatNAccessor<Float, nColor, nSpin, QUDA_FLOAT2_CLOVER_ORDER>(A, inverse)
      {
      }
    };

    template <typename Float, int nColor, int nSpin>
    struct Accessor<Float, nColor, nSpin, QUDA_FLOAT4_CLOVER_ORDER>
      : FloatNAccessor<Float, nColor, nSpin, QUDA_FLOAT4_CLOVER_ORDER> {
      Accessor(const CloverField &A, bool inverse = false) :
        FloatNAccessor<Float, nColor, nSpin, QUDA_FLOAT4_CLOVER_ORDER>(A, inverse)
      {
      }
    };

    template <typename Float, int nColor, int nSpin>
    struct Accessor<Float, nColor, nSpin, QUDA_FLOAT8_CLOVER_ORDER>
      : FloatNAccessor<Float, nColor, nSpin, QUDA_FLOAT8_CLOVER_ORDER> {
      Accessor(const CloverField &A, bool inverse = false) :
        FloatNAccessor<Float, nColor, nSpin, QUDA_FLOAT8_CLOVER_ORDER>(A, inverse)
      {
      }
    };

    template <typename Float, int nColor, int nSpin> struct Accessor<Float, nColor, nSpin, QUDA_PACKED_CLOVER_ORDER> {
      Float *a;
      size_t offset_cb;
      const int N = nSpin * nColor / 2;
      const complex<Float> zero;
      Accessor(const CloverField &A, bool inverse = false) :
        a(static_cast<Float *>(const_cast<void *>(A.V(inverse)))),
        offset_cb(A.Bytes() / (2 * sizeof(Float))),
        zero(complex<Float>(0.0, 0.0))
      {
      }

      __device__ __host__ inline complex<Float> operator()(int parity, int x, int chirality, int s_row, int s_col,
                                                           int c_row, int c_col) const
      {
        unsigned int row = s_row * nColor + c_row;
        unsigned int col = s_col * nColor + c_col;

        if (row == col) {
          complex<Float> tmp = a[parity * offset_cb + (x * 2 + chirality) * N * N + row];
          return tmp;
        } else if (col < row) {
          // switch coordinates to count from bottom right instead of top left of matrix
	  int k = N*(N-1)/2 - (N-col)*(N-col-1)/2 + row - col - 1;
          int idx = (x*2 + chirality)*N*N + N + 2*k;
          return complex<Float>(a[parity * offset_cb + idx], a[parity * offset_cb + idx + 1]);
        } else {
          // switch coordinates to count from bottom right instead of top left of matrix
	  int k = N*(N-1)/2 - (N-row)*(N-row-1)/2 + col - row - 1;
          int idx = (x*2 + chirality)*N*N + N + 2*k;
          return complex<Float>(a[parity * offset_cb + idx], -a[parity * offset_cb + idx + 1]);
        }
      }

      /**
         @brief Wrapper to transform_reduce which is called by the
         reducer functions, e.g., norm2 and abs_max
         @tparam reducer The reduction operation we which to apply
         @param[in] location The location of execution
         @param[in] h The helper functor which acts as the transformer
         in transform_reduce
       */
      template <typename reducer, typename helper>
      __host__ double transform_reduce(QudaFieldLocation location, helper h) const
      {
        return ::quda::transform_reduce<reducer>(location, reinterpret_cast<complex<Float> *>(a), offset_cb, h);
      }

      constexpr Float scale() const { return static_cast<Float>(1.0); }
    };

    /**
       This is a template driven generic clover field accessor.  To
       deploy for a specifc field ordering, the two operator()
       accessors have to be specialized for that ordering.
     */
    template <typename Float, int nColor, int nSpin, QudaCloverFieldOrder order>
      struct FieldOrder {

      /** Does this field type support ghost zones? */
      static constexpr bool supports_ghost_zone = false;

    protected:
      /** An internal reference to the actual field we are accessing */
      CloverField &A;
      const int volumeCB;
      const Accessor<Float, nColor, nSpin, order> accessor;
      bool inverse;
      const QudaFieldLocation location;

    public:
      /**
       * Constructor for the FieldOrder class
       * @param field The field that we are accessing
       */
      FieldOrder(CloverField &A, bool inverse = false) :
        A(A), volumeCB(A.VolumeCB()), accessor(A, inverse), inverse(inverse), location(A.Location())
      {
      }

      CloverField &Field() { return A; }

      /**
       * @brief Read-only complex-member accessor function.  This is a
       * special variant that is compatible with the equivalent
       * gauge::FieldOrder accessor so these can be used
       * interchangebly in templated code
       *
       * @param parity Parity index
       * @param x_cb checkerboard site index
       * @param chirality Chirality index
       * @param s_row row spin index
       * @param c_row row color index
       * @param s_col col spin index
       * @param c_col col color index
       */
      __device__ __host__ inline complex<Float> operator()(int parity, int x, int chirality, int s_row, int s_col,
                                                           int c_row, int c_col) const
      {
        return accessor(parity, x, chirality, s_row, s_col, c_row, c_col);
        }

        /** Returns the number of field colors */
        constexpr int Ncolor() const { return nColor; }

        /** Returns the field volume */
        constexpr int Volume() const { return 2 * volumeCB; }

        /** Returns the field volume */
        constexpr int VolumeCB() const { return volumeCB; }

        /** Return the size of the allocation (parity left out and added as needed in Tunable::bytes) */
        size_t Bytes() const {
          return static_cast<size_t>(volumeCB) * A.compressed_block_size() * 2ll * sizeof(Float); // 2 from chirality
        }

        /**
	 * @brief Returns the L1 norm of the field
	 * @param[in] dim Which dimension we are taking the norm of (dummy for clover)
	 * @return L1 norm
	 */
        __host__ double norm1(int = -1, bool global = true) const
        {
          commGlobalReductionPush(global);
          double nrm1
            = accessor.scale() * accessor.template transform_reduce<plus<double>>(location, abs_<double, Float>());
          commGlobalReductionPop();
          return nrm1;
        }

        /**
         * @brief Returns the L2 norm squared of the field
         * @param[in] dim Which dimension we are taking the norm of (dummy for clover)
         * @return L1 norm
         */
        __host__ double norm2(int = -1, bool global = true) const
        {
          commGlobalReductionPush(global);
          double nrm2 = accessor.scale() * accessor.scale()
            * accessor.template transform_reduce<plus<double>>(location, square_<double, Float>());
          commGlobalReductionPop();
          return nrm2;
        }

        /**
         * @brief Returns the Linfinity norm of the field
         * @param[in] dim Which dimension we are taking the Linfinity norm of (dummy for clover)
         * @return Linfinity norm
         */
        __host__ double abs_max(int = -1, bool global = true) const
        {
          commGlobalReductionPush(global);
          double absmax
            = accessor.scale() * accessor.template transform_reduce<maximum<Float>>(location, abs_max_<Float, Float>());
          commGlobalReductionPop();
          return absmax;
        }

        /**
         * @brief Returns the minimum absolute value of the field
         * @param[in] dim Which dimension we are taking the minimum abs of (dummy for clover)
         * @return Minimum norm
         */
        __host__ double abs_min(int = -1, bool global = true) const
        {
          commGlobalReductionPush(global);
          double absmin
            = accessor.scale() * accessor.template transform_reduce<minimum<Float>>(location, abs_min_<Float, Float>());
          commGlobalReductionPop();
          return absmin;
        }
      };

    /**
       @brief Accessor routine for CloverFields in native field order.
       @tparam Float Underlying storage data type of the field
       @tparam length Total number of elements per packed clover matrix (e.g., 72)
       @tparam N Number of real numbers per short vector
       @tparam add_rho Whether to add the constant rho onto the
       diagonal.  This is used to enable Hasenbusch mass
       preconditioning.
       @tparam huge_alloc Template parameter that enables 64-bit
       pointer arithmetic for huge allocations (e.g., packed set of
       vectors).  Default is to use 32-bit pointer arithmetic.
    */
      template <typename Float, int length, int N, bool add_rho = false,
                bool enable_reconstruct_ = clover::reconstruct(), bool huge_alloc = false>
      struct FloatNOrder {
        static constexpr bool enable_reconstruct = enable_reconstruct_;
        using Accessor = FloatNOrder<Float, length, N, add_rho, enable_reconstruct, huge_alloc>;
        using real = typename mapper<Float>::type;
        typedef typename VectorType<Float, N>::type Vector;
        typedef typename AllocType<huge_alloc>::type AllocInt;
        typedef float norm_type;
        static constexpr int Ns = 4;
        static constexpr int Nc = 3;
        static constexpr int block = (Nc * Ns / 2) * (Nc * Ns / 2); // elements in a chiral block
        static_assert(2 * block == length, "2 * block != length");
        static_assert(!enable_reconstruct || (enable_reconstruct && Nc == 3), "Reconstruct requires Nc=3");
        reconstruct_t<real, block, enable_reconstruct> recon;
        static constexpr int compressed_block = reconstruct_t<real, block, enable_reconstruct>::compressed_block_size();
        static constexpr int M = (compressed_block + N - 1) / N; /** number of short vectors per chiral block we need to read */
        static constexpr int M_offset = compressed_block / N;    /** the block offset that contains the second chiral block */
        static constexpr int M_rem = compressed_block % N;       /** the remainder of the chiral block not divisible by N */
        Float *clover;
        norm_type nrm;
        norm_type nrm_inv;

        const bool is_inverse;
        const AllocInt offset; // offset can be 32-bit or 64-bit
        const int volumeCB;

        const QudaTwistFlavorType twist_flavor;
        const real mu2;
        const real epsilon2;
        const real rho;

        size_t bytes;
	void *backup_h; //! host memory for backing up the field when tuning

        FloatNOrder(const CloverField &clover, bool is_inverse, Float *clover_ = nullptr) :
          recon(clover.Diagonal()),
          nrm(clover.max_element(is_inverse)
              / (2 * (isFixed<Float>::value ? fixedMaxValue<Float>::value : 1))), // factor of two in normalization
          nrm_inv(1.0 / nrm),
          is_inverse(is_inverse),
          offset(clover.Bytes() / (2 * sizeof(Float) * N)),
          volumeCB(clover.VolumeCB()),
          twist_flavor(clover.TwistFlavor()),
          mu2(clover.Mu2()),
          epsilon2(clover.Epsilon2()),
          rho(clover.Rho()),
          bytes(clover.Bytes()),
          backup_h(nullptr)
        {
          if (clover.Order() != N) errorQuda("Invalid clover order %d for FloatN (N=%d) accessor", clover.Order(), N);
          if (clover.Reconstruct() != enable_reconstruct)
            errorQuda("Accessor reconstruct = %d does not match field reconstruct %d", enable_reconstruct,
                      clover.Reconstruct());
          if (clover.max_element(is_inverse) == 0.0 && isFixed<Float>::value)
            errorQuda("%p max_element(%d) appears unset", &clover, is_inverse);
          if (clover.Diagonal() == 0.0 && clover.Reconstruct()) errorQuda("%p diagonal appears unset", &clover);
          this->clover = clover_ ? clover_ : (Float *)(clover.V(is_inverse));
        }

        QudaTwistFlavorType TwistFlavor() const { return twist_flavor; }
        real Mu2() const { return mu2; }
        real Epsilon2() const { return epsilon2; }

        /**
           @brief This accessor routine returns a const clover_wrapper to this object,
           allowing us to overload various operators for manipulating at
           the site level interms of matrix operations.
           @param[in] x_cb Checkerboarded space-time index we are requesting
           @param[in] parity Parity we are requesting
           @param[in] chirality Chirality we are requesting
           @return Instance of a clover_wrapper that curries in access to
           this field at the above coordinates.
        */
        __device__ __host__ inline auto operator()(int x_cb, int parity, int chirality) const
        {
          return clover_wrapper<real, Accessor>(*this, x_cb, parity, chirality);
        }

        /**
           @brief Load accessor for a single compressed chiral block
           @param[out] v Vector of loaded elements
           @param[in] x Checkerboarded site index
           @param[in] parity Field parity
           @param[in] chirality Chiral block index
         */
        __device__ __host__ inline void raw_load(real v[compressed_block], int x, int parity, int chirality) const
        {
          // the chiral block size may not be exactly divisible by N, in which case we need to over-size the read 
          array<real, M * N> tmp;             // array storing the elements read in

#pragma unroll
          for (int i = 0; i < M; i++) {
            // first load from memory
            Vector vecTmp = vector_load<Vector>(clover, parity * offset + x + volumeCB * (chirality * M_offset + i));

            // second do scalar copy converting into register type
#pragma unroll
            for (int j = 0; j < N; j++) { copy_and_scale(tmp[i * N + j], reinterpret_cast<Float *>(&vecTmp)[j], nrm); }
          }

#pragma unroll
          for (int i = 0; i < compressed_block; i++) v[i] = tmp[i + chirality * compressed_block % N];
        }

        __device__ __host__ inline void raw_load(real v[2 * compressed_block], int x, int parity) const
        {
#pragma unroll
          for (int ch = 0; ch < 2; ch++) raw_load(v + ch * compressed_block, x, parity, ch);
        }

        /**
	   @brief Load accessor for a single chiral block
	   @param[out] v Vector of loaded elements
	   @param[in] x Checkerboarded site index
	   @param[in] parity Field parity
	   @param[in] chirality Chiral block index
	 */
        __device__ __host__ inline void load(real v[block], int x, int parity, int chirality) const
        {
          array<real, compressed_block> tmp;
          raw_load(tmp.data, x, parity, chirality);
          recon.unpack(v, tmp);

          if (add_rho) {
#pragma unroll
            for (int i = 0; i < 6; i++) v[i] += rho;
          }
        }

        /**
           @brief Store accessor for a single chiral block
           @param[out] v Vector of elements to be stored
           @param[in] x Checkerboarded site index
           @param[in] parity Field parity
           @param[in] chirality Chiral block index
         */
        __device__ __host__ inline void raw_save(const real v[compressed_block], int x, int parity, int chirality) const
        {
          // the chiral block size may not be exactly divisible by N,
          // in which case we write the divisble part first and then
          // deal with the remainder afterwards
          array<real, compressed_block> tmp;
#pragma unroll
          for (int i = 0; i < compressed_block; i++) tmp[i] = isFixed<Float>::value ? v[i] * nrm_inv : v[i];

#pragma unroll
          for (int i = 0; i < M_offset; i++) {
            Vector vecTmp;
            // first do scalar copy converting into storage type
#pragma unroll
            for (int j = 0; j < N; j++)
              copy_scaled(reinterpret_cast<Float *>(&vecTmp)[j], tmp[chirality * M_rem + i * N + j]);
            // second do vectorized copy into memory
            vector_store(clover, parity * offset + x + volumeCB * (chirality * M + i), vecTmp);
          }

          if (M_rem) {
            typename VectorType<Float, std::max(M_rem, 1)>::type vecTmp;
            // first do scalar copy converting into storage type
#pragma unroll
            for (int j = 0; j < M_rem; j++)
              copy_scaled(reinterpret_cast<Float *>(&vecTmp)[j], tmp[(1 - chirality) * M_offset * N + j]);

            char *ptr = reinterpret_cast<char *>(reinterpret_cast<Vector *>(clover) + parity * offset + x);
            ptr += (volumeCB * (M_offset * N) + chirality * M_rem) * sizeof(Float);
            vector_store(ptr, 0, vecTmp); // second do vectorized copy into memory
          }
        }

        __device__ __host__ inline void raw_save(const real v[2 * compressed_block], int x, int parity) const
        {
#pragma unroll
          for (int ch = 0; ch < 2; ch++) raw_save(v + ch * compressed_block, x, parity, ch);
        }

        /**
           @brief Store accessor for a single chiral block
           @param[out] v Vector of elements to be stored
           @param[in] x Checkerboarded site index
           @param[in] parity Field parity
           @param[in] chirality Chiral block index
         */
        __device__ __host__ inline void save(const real v[block], int x, int parity, int chirality) const
        {
          array<real, compressed_block> tmp;
          recon.pack(tmp, v);
          raw_save(tmp.data, x, parity, chirality);
        }

        /**
           @brief Load accessor for the clover matrix
           @param[out] v Vector of loaded elements
           @param[in] x Checkerboarded site index
           @param[in] parity Field parity
           @param[in] chirality Chiral block index
         */
        __device__ __host__ inline void load(real v[], int x, int parity) const
        {
#pragma unroll
          for (int chirality = 0; chirality < 2; chirality++) load(&v[chirality * block], x, parity, chirality);
        }

        /**
           @brief Store accessor for the clover matrix
           @param[out] v Vector of elements to be stored
           @param[in] x Checkerboarded site index
           @param[in] parity Field parity
           @param[in] chirality Chiral block index
         */
        __device__ __host__ inline void save(const real v[], int x, int parity) const
        {
#pragma unroll
          for (int chirality = 0; chirality < 2; chirality++) save(&v[chirality * block], x, parity, chirality);
        }

        /**
	   @brief Backup the field to the host when tuning
	*/
	void save() {
	  if (backup_h) errorQuda("Already allocated host backup");
	  backup_h = safe_malloc(bytes);
          qudaMemcpy(backup_h, clover, bytes, qudaMemcpyDeviceToHost);
        }

        /**
           @brief Restore the field from the host after tuning
        */
        void load()
        {
          qudaMemcpy(clover, backup_h, bytes, qudaMemcpyHostToDevice);
          host_free(backup_h);
          backup_h = nullptr;
        }

        size_t Bytes() const { return 2 * decltype(recon)::compressed_block_size() * sizeof(Float); }
      };

    /**
       QDP ordering for clover fields
    */
      template <typename Float, int length = 72> struct QDPOrder {
        static constexpr bool enable_reconstruct = false;
        typedef typename mapper<Float>::type RegType;
        Float *clover;
        const int volumeCB;
        const int offset;

        const QudaTwistFlavorType twist_flavor;
        const Float mu2;
        const Float epsilon2;

        QDPOrder(const CloverField &clover, bool inverse, Float *clover_ = nullptr, void * = nullptr) :
          volumeCB(clover.VolumeCB()),
          offset(clover.Bytes() / (2 * sizeof(Float))),
          twist_flavor(clover.TwistFlavor()),
          mu2(clover.Mu2()),
          epsilon2(clover.Epsilon2())
        {
          if (clover.Order() != QUDA_PACKED_CLOVER_ORDER) {
            errorQuda("Invalid clover order %d for this accessor", clover.Order());
          }
          this->clover = clover_ ? clover_ : (Float *)(clover.V(inverse));
        }

        QudaTwistFlavorType TwistFlavor() const { return twist_flavor; }
        Float Mu2() const { return mu2; }
        Float Epsilon2() const { return epsilon2; }

        __device__ __host__ inline void load(RegType v[length], int x, int parity) const {
	  // factor of 0.5 comes from basis change
          Float v_[length];
          block_load<Float, length>(v_, &clover[parity * offset + x * length]);
#pragma unroll
          for (int i = 0; i < length; i++) v[i] = 0.5 * v_[i];
        }

        __device__ __host__ inline void save(const RegType v[length], int x, int parity) const
        {
          Float v_[length];
#pragma unroll
          for (int i = 0; i < length; i++) v_[i] = 2.0 * v[i];
          block_store<Float, length>(&clover[parity * offset + x * length], v_);
        }

        size_t Bytes() const { return length*sizeof(Float); }
      };

    /**
       QDPJIT ordering for clover fields
    */
      template <typename Float, int length = 72> struct QDPJITOrder {
        static constexpr bool enable_reconstruct = false;
        typedef typename mapper<Float>::type RegType;
        Float *diag;    /**< Pointers to the off-diagonal terms (two parities) */
        Float *offdiag; /**< Pointers to the diagonal terms (two parities) */
        const int volumeCB;
        const QudaTwistFlavorType twist_flavor;
        const Float mu2;
        const Float epsilon2;

        QDPJITOrder(const CloverField &clover, bool inverse, Float *clover_ = nullptr, void * = nullptr) :
          volumeCB(clover.VolumeCB()),
          twist_flavor(clover.TwistFlavor()),
          mu2(clover.Mu2()),
          epsilon2(clover.Epsilon2())
        {
          if (clover.Order() != QUDA_QDPJIT_CLOVER_ORDER) {
            errorQuda("Invalid clover order %d for this accessor", clover.Order());
          }
          offdiag = clover_ ? ((Float **)clover_)[0] : ((Float **)clover.V(inverse))[0];
          diag = clover_ ? ((Float **)clover_)[1] : ((Float **)clover.V(inverse))[1];
        }

        QudaTwistFlavorType TwistFlavor() const { return twist_flavor; }
        Float Mu2() const { return mu2; }
        Float Epsilon2() const { return epsilon2; }

        __device__ __host__ inline void load(RegType v[length], int x, int parity) const {
	  // the factor of 0.5 comes from a basis change
#pragma unroll
          for (int chirality = 0; chirality < 2; chirality++) {
            // set diagonal elements
#pragma unroll
            for (int i = 0; i < 6; i++) {
              v[chirality*36 + i] = 0.5*diag[((i*2 + chirality)*2 + parity)*volumeCB + x];
            }

            // the off diagonal elements
#pragma unroll
            for (int i = 0; i < 30; i++) {
              int z = i%2;
	      int off = i/2;
	      const int idtab[15]={0,1,3,6,10,2,4,7,11,5,8,12,9,13,14};
	      v[chirality*36 + 6 + i] = 0.5*offdiag[(((z*15 + idtab[off])*2 + chirality)*2 + parity)*volumeCB + x];
            }
          }
        }

        __device__ __host__ inline void save(const RegType v[length], int x, int parity) const
        {
          // the factor of 2.0 comes from undoing the basis change
#pragma unroll
          for (int chirality = 0; chirality < 2; chirality++) {
            // set diagonal elements
#pragma unroll
            for (int i = 0; i < 6; i++) {
              diag[((i*2 + chirality)*2 + parity)*volumeCB + x] = 2.0*v[chirality*36 + i];
            }

            // the off diagonal elements
#pragma unroll
            for (int i = 0; i < 30; i++) {
              int z = i%2;
	      int off = i/2;
	      const int idtab[15]={0,1,3,6,10,2,4,7,11,5,8,12,9,13,14};
	      offdiag[(((z*15 + idtab[off])*2 + chirality)*2 + parity)*volumeCB + x] = 2.0*v[chirality*36 + 6 + i];
            }
          }
        }

        size_t Bytes() const { return length*sizeof(Float); }
      };

    /**
       BQCD ordering for clover fields
       struct for reordering a BQCD clover matrix into the order that is
       expected by QUDA.  As well as reordering the clover matrix
       elements, we are also changing basis.
    */
      template <typename Float, int length = 72> struct BQCDOrder {
        static constexpr bool enable_reconstruct = false;
        typedef typename mapper<Float>::type RegType;
        Float *clover[2];
        const int volumeCB;
        const QudaTwistFlavorType twist_flavor;
        const Float mu2;
        const Float epsilon2;

        BQCDOrder(const CloverField &clover, bool inverse, Float *clover_ = nullptr, void * = nullptr) :
          volumeCB(clover.Stride()),
          twist_flavor(clover.TwistFlavor()),
          mu2(clover.Mu2()),
          epsilon2(clover.Epsilon2())
        {
          if (clover.Order() != QUDA_BQCD_CLOVER_ORDER) {
            errorQuda("Invalid clover order %d for this accessor", clover.Order());
          }
          this->clover[0] = clover_ ? clover_ : (Float *)(clover.V(inverse));
          this->clover[1] = (Float *)((char *)this->clover[0] + clover.Bytes() / 2);
        }

        QudaTwistFlavorType TwistFlavor() const { return twist_flavor; }
        Float Mu2()	const	{return mu2;}
        Float Epsilon2() const { return epsilon2; }

        /**
	   @param v The output clover matrix in QUDA order
	   @param x The checkerboarded lattice site
	   @param parity The parity of the lattice site
	*/
	__device__ __host__ inline void load(RegType v[length], int x, int parity) const {
	  int bq[36] = { 21, 32, 33, 0,  1, 20,                   // diagonal
			 28, 29, 30, 31, 6, 7,  14, 15, 22, 23,   // column 1  6
			 34, 35, 8, 9, 16, 17, 24, 25,            // column 2  16
			 10, 11, 18, 19, 26, 27,                  // column 3  24
			 2,  3,  4,  5,                           // column 4  30
			 12, 13};

          // flip the sign of the imaginary components
          int sign[36];
#pragma unroll
          for (int i = 0; i < 6; i++) sign[i] = 1;
#pragma unroll
          for (int i = 6; i < 36; i += 2) {
            if ( (i >= 10 && i<= 15) || (i >= 18 && i <= 29) )  { sign[i] = -1; sign[i+1] = -1; }
	    else { sign[i] = 1; sign[i+1] = -1; }
          }

          const int M = length / 2;
#pragma unroll
          for (int chirality = 0; chirality < 2; chirality++)
#pragma unroll
            for (int i = 0; i < M; i++)
              v[chirality * M + i] = sign[i] * clover[parity][x * length + chirality * M + bq[i]];
        }

        // FIXME implement the save routine for BQCD ordered fields
        __device__ __host__ inline void save(RegType[length], int, int) const { }

        size_t Bytes() const { return length*sizeof(Float); }
      };

  } // namespace clover

  // Use traits to reduce the template explosion
  template <typename Float, int N = 72, bool add_rho = false, bool enable_reconstruct = clover::reconstruct()>
  struct clover_mapper {
  };

  // double precision uses Float2
  template <int N, bool add_rho, bool enable_reconstruct> struct clover_mapper<double, N, add_rho, enable_reconstruct> {
    using type = clover::FloatNOrder<double, N, 2, add_rho, enable_reconstruct>;
  };

  // single precision uses Float4
  template <int N, bool add_rho, bool enable_reconstruct> struct clover_mapper<float, N, add_rho, enable_reconstruct> {
    using type = clover::FloatNOrder<float, N, 4, add_rho, enable_reconstruct>;
  };

  // half precision uses QUDA_ORDER_FP (Float8 default)
  template <int N, bool add_rho, bool enable_reconstruct> struct clover_mapper<short, N, add_rho, enable_reconstruct> {
    using type = clover::FloatNOrder<short, N, QUDA_ORDER_FP, add_rho, enable_reconstruct>;
  };

  // quarter precision uses QUDA_ORDER_FP (Float8 default)
  template <int N, bool add_rho, bool enable_reconstruct> struct clover_mapper<int8_t, N, add_rho, enable_reconstruct> {
    using type = clover::FloatNOrder<int8_t, N, QUDA_ORDER_FP, add_rho, enable_reconstruct>;
  };

} // namespace quda
