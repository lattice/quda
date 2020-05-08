#ifndef _CLOVER_ORDER_H
#define _CLOVER_ORDER_H

/**
 * @file  clover_field_order.h
 * @brief Main header file for host and device accessors to CloverFields
 *
 */

#include <register_traits.h>
#include <convert.h>
#include <clover_field.h>
#include <complex_quda.h>
#include <quda_matrix.h>
#include <color_spinor.h>
#include <trove_helper.cuh>
#include <texture_helper.cuh>
#include <transform_reduce.h>

namespace quda {

  /**
     @brief clover_wrapper is an internal class that is used to
     wrap instances of colorspinor accessors, currying in a specifc
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
      T &field;
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
      __device__ __host__ inline clover_wrapper<Float,T>(T &field, int x_cb, int parity, int chirality)
	: field(field), x_cb(x_cb), parity(parity), chirality(chirality) { }

      /**
	 @brief Assignment operator with H matrix instance as input
	 @param[in] C ColorSpinor we want to store in this accessor
      */
      template<typename C>
      __device__ __host__ inline void operator=(const C &a) {
        field.save(a.data, x_cb, parity, chirality);
      }
    };

  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline void HMatrix<T,N>::operator=(const clover_wrapper<T,S> &a) {
    a.field.load(data, a.x_cb, a.parity, a.chirality);
  }

  template <typename T, int N>
    template <typename S>
    __device__ __host__ inline HMatrix<T,N>::HMatrix(const clover_wrapper<T,S> &a) {
    a.field.load(data, a.x_cb, a.parity, a.chirality);
  }

  namespace clover {

    template<typename ReduceType, typename Float> struct square_ {
      __host__ __device__ inline ReduceType operator()(const quda::complex<Float> &x)
      { return static_cast<ReduceType>(norm(x)); }
    };

    template<typename ReduceType, typename Float> struct abs_ {
      __host__ __device__ inline ReduceType operator()(const quda::complex<Float> &x)
      { return static_cast<ReduceType>(abs(x)); }
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
      mutable complex<Float> dummy;
      Accessor(const CloverField &A, bool inverse=false) {
	errorQuda("Not implemented for order %d", order);
      }

      __device__ __host__ inline complex<Float>& operator()(int parity, int x, int s_row, int s_col,
							    int c_row, int c_col) const {
	return dummy;
      }

      template <typename helper, typename reducer>
      __host__ double transform_reduce(QudaFieldLocation location, helper h, double i, reducer r) const
      {
        return 0.0;
      }
    };

    template<typename Float, int nColor, int nSpin>
      struct Accessor<Float,nColor,nSpin,QUDA_FLOAT2_CLOVER_ORDER> {
      Float *a;
      int stride;
      size_t offset_cb;
      static constexpr int N = nSpin * nColor / 2;
    Accessor(const CloverField &A, bool inverse=false)
      : a(static_cast<Float*>(const_cast<void*>(A.V(inverse)))), stride(A.Stride()),
	offset_cb(A.Bytes()/(2*sizeof(Float))) { }

      __device__ __host__ inline complex<Float> operator()(int parity, int x, int s_row, int s_col, int c_row, int c_col) const {
	// if not in the diagonal chiral block then return 0.0
	if (s_col / 2 != s_row / 2) { return complex<Float>(0.0); }

	const int chirality = s_col / 2;

	int row = s_row%2 * nColor + c_row;
	int col = s_col%2 * nColor + c_col;
	Float *a_ = a+parity*offset_cb+stride*chirality*N*N;

	if (row == col) {
	  return 2*a_[ row*stride+x ];
	} else if (col < row) {
	  // switch coordinates to count from bottom right instead of top left of matrix
	  int k = N*(N-1)/2 - (N-col)*(N-col-1)/2 + row - col - 1;
          complex<Float> *off = reinterpret_cast<complex<Float>*>(a_ + N);

          return 2*off[k*stride + x];
	} else {
	  // requesting upper triangular so return conjugate transpose
	  // switch coordinates to count from bottom right instead of top left of matrix
	  int k = N*(N-1)/2 - (N-row)*(N-row-1)/2 + col - row - 1;
          complex<Float> *off = reinterpret_cast<complex<Float>*>(a_ + N);
          return 2*conj(off[k*stride + x]);
	}

      }

      template <typename helper, typename reducer>
      __host__ double transform_reduce(QudaFieldLocation location, helper h, double init, reducer r) const
      {
        // just use offset_cb, since factor of two from parity is equivalent to complexity
        double result = ::quda::transform_reduce(location, reinterpret_cast<complex<Float> *>(a), offset_cb, h, init, r);
        return 2.0 * result; // factor of two is normalization
      }
    };

    template<int N>
      __device__ __host__ inline int indexFloatN(int k, int stride, int x) {
      int j = k / N;
      int i = k % N;
      return (j*stride+x)*N + i;
    };

    template<typename Float, int nColor, int nSpin>
      struct Accessor<Float,nColor,nSpin,QUDA_FLOAT4_CLOVER_ORDER> {
      Float *a;
      int stride;
      size_t offset_cb;
      static constexpr int N = nSpin * nColor / 2;
    Accessor(const CloverField &A, bool inverse=false)
      : a(static_cast<Float*>(const_cast<void*>(A.V(inverse)))), stride(A.Stride()),
	offset_cb(A.Bytes()/(2*sizeof(Float))) { }

      __device__ __host__ inline complex<Float> operator()(int parity, int x, int s_row, int s_col, int c_row, int c_col) const {
	// if not in the diagonal chiral block then return 0.0
	if (s_col / 2 != s_row / 2) { return complex<Float>(0.0); }

	const int chirality = s_col / 2;

	int row = s_row%2 * nColor + c_row;
	int col = s_col%2 * nColor + c_col;
	Float *a_ = a+parity*offset_cb+stride*chirality*N*N;

	if (row == col) {
	  return 2*a_[ indexFloatN<QUDA_FLOAT4_CLOVER_ORDER>(row, stride, x) ];
	} else if (col < row) {
	  // switch coordinates to count from bottom right instead of top left of matrix
	  int k = N*(N-1)/2 - (N-col)*(N-col-1)/2 + row - col - 1;
          int idx = N + 2*k;

          return 2*complex<Float>(a_[ indexFloatN<QUDA_FLOAT4_CLOVER_ORDER>(idx+0,stride,x) ],
				  a_[ indexFloatN<QUDA_FLOAT4_CLOVER_ORDER>(idx+1,stride,x) ]);
	} else {
	  // requesting upper triangular so return conjugate transpose
	  // switch coordinates to count from bottom right instead of top left of matrix
	  int k = N*(N-1)/2 - (N-row)*(N-row-1)/2 + col - row - 1;
          int idx = N + 2*k;

          return 2*complex<Float>( a_[ indexFloatN<QUDA_FLOAT4_CLOVER_ORDER>(idx+0,stride,x) ],
				  -a_[ indexFloatN<QUDA_FLOAT4_CLOVER_ORDER>(idx+1,stride,x) ]);
	}

      }

      template <typename helper, typename reducer>
      __host__ double transform_reduce(QudaFieldLocation location, helper h, double init, reducer r) const
      {
        // just use offset_cb, since factor of two from parity is equivalent to complexity
        double result = ::quda::transform_reduce(location, reinterpret_cast<complex<Float> *>(a), offset_cb, h, init, r);
        return 2.0 * result; // factor of two is normalization
      }
    };

    template<typename Float, int nColor, int nSpin> 
      struct Accessor<Float,nColor,nSpin,QUDA_PACKED_CLOVER_ORDER> { 
      Float *a[2];
      const int N = nSpin * nColor / 2;
      complex<Float> zero;
      Accessor(const CloverField &A, bool inverse=false) {
	// even
	a[0] = static_cast<Float*>(const_cast<void*>(A.V(inverse)));
	// odd
	a[1] = static_cast<Float*>(const_cast<void*>(A.V(inverse))) + A.Bytes()/(2*sizeof(Float));
	zero = complex<Float>(0.0,0.0);
      }

      __device__ __host__ inline complex<Float> operator()(int parity, int x, int s_row, int s_col, int c_row, int c_col) const {
	// if not in the diagonal chiral block then return 0.0
	if (s_col / 2 != s_row / 2) { return zero; }

	const int chirality = s_col / 2;

	unsigned int row = s_row%2 * nColor + c_row;
	unsigned int col = s_col%2 * nColor + c_col;

	if (row == col) {
	  complex<Float> tmp = a[parity][(x*2 + chirality)*N*N + row];
	  return tmp;
	} else if (col < row) {
	  // switch coordinates to count from bottom right instead of top left of matrix
	  int k = N*(N-1)/2 - (N-col)*(N-col-1)/2 + row - col - 1;
          int idx = (x*2 + chirality)*N*N + N + 2*k;
          return complex<Float>(a[parity][idx], a[parity][idx+1]);
	} else {
	  // switch coordinates to count from bottom right instead of top left of matrix
	  int k = N*(N-1)/2 - (N-row)*(N-row-1)/2 + col - row - 1;
          int idx = (x*2 + chirality)*N*N + N + 2*k;
          return complex<Float>(a[parity][idx], -a[parity][idx+1]);
	}
      }

      template <typename helper, typename reducer>
      __host__ double transform_reduce(QudaFieldLocation location, helper h, double init, reducer r) const
      {
        errorQuda("Not implemented");
	return 0.0;
      }
    };

    /*
      FIXME the below is the old optimization used for reading the
      clover field, making use of the symmetry to reduce the number of
      reads.

#define READ_CLOVER2_DOUBLE_STR(clover_, chi)                           \
    double2 C0, C1, C2, C3, C4, C5, C6, C7, C8, C9;                       \
    double2 C10, C11, C12, C13, C14, C15, C16, C17;                       \
    double2* clover = (double2*)clover_;                                  \
    load_streaming_double2(C0, &clover[sid + (18*chi+0)*param.cl_stride]); \
    load_streaming_double2(C1, &clover[sid + (18*chi+1)*param.cl_stride]); \
    double diag = 0.5*(C0.x + C1.y);                                      \
    double diag_inv = 1.0/diag;                                           \
    C2 = make_double2(diag*(2-C0.y*diag_inv), diag*(2-C1.x*diag_inv));    \
    load_streaming_double2(C3, &clover[sid + (18*chi+3)*param.cl_stride]);        \
    load_streaming_double2(C4, &clover[sid + (18*chi+4)*param.cl_stride]);        \
    load_streaming_double2(C5, &clover[sid + (18*chi+5)*param.cl_stride]);        \
    load_streaming_double2(C6, &clover[sid + (18*chi+6)*param.cl_stride]);        \
    load_streaming_double2(C7, &clover[sid + (18*chi+7)*param.cl_stride]);        \
    load_streaming_double2(C8, &clover[sid + (18*chi+8)*param.cl_stride]);        \
    load_streaming_double2(C9, &clover[sid + (18*chi+9)*param.cl_stride]);        \
    load_streaming_double2(C10, &clover[sid + (18*chi+10)*param.cl_stride]);      \
    load_streaming_double2(C11, &clover[sid + (18*chi+11)*param.cl_stride]);      \
    load_streaming_double2(C12, &clover[sid + (18*chi+12)*param.cl_stride]);      \
    load_streaming_double2(C13, &clover[sid + (18*chi+13)*param.cl_stride]);      \
    load_streaming_double2(C14, &clover[sid + (18*chi+14)*param.cl_stride]); \
    C15 = make_double2(-C3.x,-C3.y);                                      \
    C16 = make_double2(-C4.x,-C4.y);                                      \
    C17 = make_double2(-C8.x,-C8.y);                                      \
    */

    /**
       This is a template driven generic clover field accessor.  To
       deploy for a specifc field ordering, the two operator()
       accessors have to be specialized for that ordering.
     */
    template <typename Float, int nColor, int nSpin, QudaCloverFieldOrder order>
      struct FieldOrder {

      protected:
	/** An internal reference to the actual field we are accessing */
	CloverField &A;
	const int volumeCB;
	const Accessor<Float,nColor,nSpin,order> accessor;
	bool inverse;
	const QudaFieldLocation location;

      public:
	/** 
	 * Constructor for the FieldOrder class
	 * @param field The field that we are accessing
	 */
      FieldOrder(CloverField &A, bool inverse=false)
      : A(A), volumeCB(A.VolumeCB()), accessor(A,inverse), inverse(inverse), location(A.Location())
	{ }
	
	CloverField& Field() { return A; }
	
    	/**
	 * @brief Read-only complex-member accessor function
	 *
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param s_row row spin index
	 * @param c_row row color index
	 * @param s_col col spin index
	 * @param c_col col color index
	 */
	__device__ __host__ inline const complex<Float> operator()(int parity, int x, int s_row,
								   int s_col, int c_row, int c_col) const {
	  return accessor(parity, x, s_row, s_col, c_row, c_col);
	}
	
    	/**
	 * @brief Read-only complex-member accessor function.  This is a
	 * special variant that is compatible with the equivalent
	 * gauge::FieldOrder accessor so these can be used
	 * interchangebly in templated code
	 *
	 * @param dummy Dummy parameter that is ignored
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param s_row row spin index
	 * @param c_row row color index
	 * @param s_col col spin index
	 * @param c_col col color index
	 */
	__device__ __host__ inline complex<Float> operator()(int dummy, int parity, int x, int s_row,
							     int s_col, int c_row, int c_col) const {
	  return accessor(parity,x,s_row,s_col,c_row,c_col);
	}

	/**
	 * @brief Complex-member accessor function
	 *
	 * @param parity Parity index
	 * @param x 1-d site index
	 * @param s_row row spin index
	 * @param c_row row color index
	 * @param s_col col spin index
	 * @param c_col col color index
	 */
	/*
	__device__ __host__ inline complex<Float>& operator()(int parity, int x, int s_row,
							     int s_col, int c_row, int c_col) {
	  //errorQuda("Clover accessor not implemented as a lvalue");
	  return accessor(parity, x, s_row, s_col, c_row, c_col);
	  }
	*/
	
	/** Returns the number of field colors */
	__device__ __host__ inline int Ncolor() const { return nColor; }

	/** Returns the field volume */
	__device__ __host__ inline int Volume() const { return 2*volumeCB; }

	/** Returns the field volume */
	__device__ __host__ inline int VolumeCB() const { return volumeCB; }

	/** Return the size of the allocation (parity left out and added as needed in Tunable::bytes) */
	size_t Bytes() const {
	  constexpr int n = (nSpin * nColor) / 2;
	  constexpr int chiral_block = n * n / 2;
	  return static_cast<size_t>(volumeCB) * chiral_block * 2ll * 2ll * sizeof(Float); // 2 from complex, 2 from chirality
	}

	/**
	 * @brief Returns the L1 norm of the field
	 * @param[in] dim Which dimension we are taking the norm of (dummy for clover)
	 * @return L1 norm
	 */
	__host__ double norm1(int dim=-1, bool global=true) const {
          double nrm1 = accessor.transform_reduce(location, abs_<double, Float>(), 0.0, plus<double>());
          if (global) comm_allreduce(&nrm1);
          return nrm1;
        }

        /**
	 * @brief Returns the L2 norm suared of the field
	 * @param[in] dim Which dimension we are taking the norm of (dummy for clover)
	 * @return L1 norm
	 */
	__host__ double norm2(int dim=-1, bool global=true) const {
          double nrm2 = accessor.transform_reduce(location, square_<double, Float>(), 0.0, plus<double>());
          if (global) comm_allreduce(&nrm2);
          return nrm2;
        }

        /**
	 * @brief Returns the Linfinity norm of the field
	 * @param[in] dim Which dimension we are taking the Linfinity norm of (dummy for clover)
	 * @return Linfinity norm
	 */
	__host__ double abs_max(int dim=-1, bool global=true) const {
          double absmax = accessor.transform_reduce(location, abs_<Float, Float>(), 0.0, maximum<Float>());
          if (global) comm_allreduce_max(&absmax);
          return absmax;
        }

        /**
	 * @brief Returns the minimum absolute value of the field
	 * @param[in] dim Which dimension we are taking the minimum abs of (dummy for clover)
	 * @return Minimum norm
	 */
	__host__ double abs_min(int dim=-1, bool global=true) const {
          double absmax = accessor.transform_reduce(location, abs_<Float, Float>(), std::numeric_limits<double>::max(),
                                                    minimum<Float>());
          if (global) comm_allreduce_min(&absmax);
          return absmax;
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
    template <typename Float, int length, int N, bool add_rho=false, bool huge_alloc=false>
    struct FloatNOrder {
      using Accessor = FloatNOrder<Float, length, N, add_rho, huge_alloc>;
      using real = typename mapper<Float>::type;
      typedef typename VectorType<Float, N>::type Vector;
      typedef typename AllocType<huge_alloc>::type AllocInt;
      typedef float norm_type;
      static const int M = length / (N * 2); // number of short vectors per chiral block
      static const int block = length / 2;   // chiral block size
      Float *clover;
      norm_type *norm;
      const AllocInt offset; // offset can be 32-bit or 64-bit
      const AllocInt norm_offset;
#ifdef USE_TEXTURE_OBJECTS
	typedef typename TexVectorType<real, N>::type TexVector;
	cudaTextureObject_t tex;
	cudaTextureObject_t normTex;
#endif
	const int volumeCB;
	const int stride;

	const bool twisted;
	const real mu2;
        const real rho;

        size_t bytes;
	size_t norm_bytes;
	void *backup_h; //! host memory for backing up the field when tuning
	void *backup_norm_h; //! host memory for backing up norm when tuning

        FloatNOrder(const CloverField &clover, bool is_inverse, Float *clover_ = 0, norm_type *norm_ = 0,
                    bool override = false) :
          offset(clover.Bytes() / (2 * sizeof(Float) * N)),
          norm_offset(clover.NormBytes() / (2 * sizeof(norm_type))),
#ifdef USE_TEXTURE_OBJECTS
          tex(0),
          normTex(0),
#endif
          volumeCB(clover.VolumeCB()),
          stride(clover.Stride()),
          twisted(clover.Twisted()),
          mu2(clover.Mu2()),
          rho(clover.Rho()),
          bytes(clover.Bytes()),
          norm_bytes(clover.NormBytes()),
          backup_h(nullptr),
          backup_norm_h(nullptr)
	{
	  this->clover = clover_ ? clover_ : (Float*)(clover.V(is_inverse));
          this->norm = norm_ ? norm_ : (norm_type *)(clover.Norm(is_inverse));
#ifdef USE_TEXTURE_OBJECTS
	  if (clover.Location() == QUDA_CUDA_FIELD_LOCATION) {
	    if (is_inverse) {
	      tex = static_cast<const cudaCloverField&>(clover).InvTex();
	      normTex = static_cast<const cudaCloverField&>(clover).InvNormTex();
	    } else {
	      tex = static_cast<const cudaCloverField&>(clover).Tex();
	      normTex = static_cast<const cudaCloverField&>(clover).NormTex();
	    }
	    if (!huge_alloc && (this->clover != clover.V(is_inverse) ||
				((clover.Precision() == QUDA_HALF_PRECISION || clover.Precision() == QUDA_QUARTER_PRECISION) && this->norm != clover.Norm(is_inverse)) ) && !override) {
	      errorQuda("Cannot use texture read since data pointer does not equal field pointer - use with huge_alloc=true instead");
	    }
	  }
#endif
	}

	bool Twisted() const { return twisted; }
	real Mu2() const { return mu2; }

	/**
	   @brief This accessor routine returns a clover_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @param[in] chirality Chirality we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
        __device__ __host__ inline clover_wrapper<real, Accessor> operator()(int x_cb, int parity, int chirality)
        {
          return clover_wrapper<real, Accessor>(*this, x_cb, parity, chirality);
        }

        /**
	   @brief This accessor routine returns a const colorspinor_wrapper to this object,
	   allowing us to overload various operators for manipulating at
	   the site level interms of matrix operations.
	   @param[in] x_cb Checkerboarded space-time index we are requesting
	   @param[in] parity Parity we are requesting
	   @param[in] chirality Chirality we are requesting
	   @return Instance of a colorspinor_wrapper that curries in access to
	   this field at the above coordinates.
	*/
        __device__ __host__ inline const clover_wrapper<real, Accessor> operator()(
            int x_cb, int parity, int chirality) const
        {
          return clover_wrapper<real, Accessor>(const_cast<Accessor &>(*this), x_cb, parity, chirality);
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
          norm_type nrm;
          if (isFixed<Float>::value) {
#if defined(USE_TEXTURE_OBJECTS) && defined(__CUDA_ARCH__)
            nrm = !huge_alloc ? tex1Dfetch_<float>(normTex, parity * norm_offset + chirality * stride + x) :
                                norm[parity * norm_offset + chirality * stride + x];
#else
            nrm = vector_load<float>(norm, parity * norm_offset + chirality * stride + x);
#endif
          }

#pragma unroll
	  for (int i=0; i<M; i++) {
#if defined(USE_TEXTURE_OBJECTS) && defined(__CUDA_ARCH__)
	    if (!huge_alloc) { // use textures unless we have a huge alloc
                               // first do texture load from memory
              TexVector vecTmp = tex1Dfetch_<TexVector>(tex, parity * offset + stride * (chirality * M + i) + x);
              // now insert into output array
#pragma unroll
              for (int j = 0; j < N; j++) {
                copy(v[i * N + j], reinterpret_cast<real *>(&vecTmp)[j]);
                if (isFixed<Float>::value) v[i * N + j] *= nrm;
              }
            } else
#endif
	    {
              // first load from memory
              Vector vecTmp = vector_load<Vector>(clover, parity * offset + x + stride * (chirality * M + i));
              // second do scalar copy converting into register type
#pragma unroll
              for (int j = 0; j < N; j++) { copy_and_scale(v[i * N + j], reinterpret_cast<Float *>(&vecTmp)[j], nrm); }
            }
	  }

          if (add_rho) for (int i=0; i<6; i++) v[i] += rho;
        }
  
	/**
	   @brief Store accessor for a single chiral block
	   @param[out] v Vector of elements to be stored
	   @param[in] x Checkerboarded site index
	   @param[in] parity Field parity
	   @param[in] chirality Chiral block index
	 */
	__device__ __host__ inline void save(const real v[block], int x, int parity, int chirality)
        {
          real tmp[block];

          // find the norm of each chiral block
          if (isFixed<Float>::value) {
            norm_type scale = 0.0;
#pragma unroll
            for (int i = 0; i < block; i++) scale = fabsf((norm_type)v[i]) > scale ? fabsf((norm_type)v[i]) : scale;
            norm[parity*norm_offset + chirality*stride + x] = scale;

#ifdef __CUDA_ARCH__
            real scale_inv = __fdividef(fixedMaxValue<Float>::value, scale);
#else
            real scale_inv = fixedMaxValue<Float>::value / scale;
#endif
#pragma unroll
            for (int i = 0; i < block; i++) tmp[i] = v[i] * scale_inv;
          } else {
#pragma unroll
            for (int i = 0; i < block; i++) tmp[i] = v[i];
          }

#pragma unroll
          for (int i = 0; i < M; i++) {
            Vector vecTmp;
            // first do scalar copy converting into storage type
            for (int j = 0; j < N; j++) copy_scaled(reinterpret_cast<Float *>(&vecTmp)[j], tmp[i * N + j]);
            // second do vectorized copy into memory
            vector_store(clover, parity * offset + x + stride * (chirality * M + i), vecTmp);
          }
        }

	/**
	   @brief Load accessor for the clover matrix
	   @param[out] v Vector of loaded elements
	   @param[in] x Checkerboarded site index
	   @param[in] parity Field parity
	   @param[in] chirality Chiral block index
	 */
	__device__ __host__ inline void load(real v[length], int x, int parity) const {
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
	__device__ __host__ inline void save(const real v[length], int x, int parity) {
#pragma unroll
          for (int chirality = 0; chirality < 2; chirality++) save(&v[chirality * block], x, parity, chirality);
        }

	/**
	   @brief Backup the field to the host when tuning
	*/
	void save() {
	  if (backup_h) errorQuda("Already allocated host backup");
	  backup_h = safe_malloc(bytes);
	  cudaMemcpy(backup_h, clover, bytes, cudaMemcpyDeviceToHost);
	  if (norm_bytes) {
	    backup_norm_h = safe_malloc(norm_bytes);
	    cudaMemcpy(backup_norm_h, norm, norm_bytes, cudaMemcpyDeviceToHost);
	  }
	  checkCudaError();
	}

	/**
	   @brief Restore the field from the host after tuning
	*/
	void load() {
	  cudaMemcpy(clover, backup_h, bytes, cudaMemcpyHostToDevice);
	  host_free(backup_h);
	  backup_h = nullptr;
	  if (norm_bytes) {
	    cudaMemcpy(norm, backup_norm_h, norm_bytes, cudaMemcpyHostToDevice);
	    host_free(backup_norm_h);
	    backup_norm_h = nullptr;
	  }
	  checkCudaError();
	}

	size_t Bytes() const {
	  size_t bytes = length*sizeof(Float);
          if (isFixed<Float>::value) bytes += 2 * sizeof(norm_type);
          return bytes;
	}
      };

    /**
       @brief This is just a dummy structure we use for trove to define the
       required structure size
       @tparam real Real number type
       @tparam length Number of elements in the structure
    */
    template <typename real, int length> struct S { real v[length]; };

    /**
       QDP ordering for clover fields
    */
    template <typename Float, int length>
      struct QDPOrder {
	typedef typename mapper<Float>::type RegType;
	Float *clover;
	const int volumeCB;
	const int stride;
	const int offset;

	const bool twisted;
	const Float mu2;

      QDPOrder(const CloverField &clover, bool inverse, Float *clover_=0) 
      : volumeCB(clover.VolumeCB()), stride(volumeCB), offset(clover.Bytes()/(2*sizeof(Float))),
	twisted(clover.Twisted()), mu2(clover.Mu2()) {
	this->clover = clover_ ? clover_ : (Float*)(clover.V(inverse));
      }

	bool  Twisted()	const	{return twisted;}
	Float Mu2()	const	{return mu2;}

	__device__ __host__ inline void load(RegType v[length], int x, int parity) const {
	  // factor of 0.5 comes from basis change
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	  typedef S<Float,length> structure;
	  trove::coalesced_ptr<structure> clover_((structure*)clover);
	  structure v_ = clover_[parity*volumeCB + x];
	  for (int i=0; i<length; i++) v[i] = 0.5*(RegType)v_.v[i];
#else
	  for (int i=0; i<length; i++) v[i] = 0.5*clover[parity*offset + x*length+i];
#endif
	}
  
	__device__ __host__ inline void save(const RegType v[length], int x, int parity) {
#if defined( __CUDA_ARCH__) && !defined(DISABLE_TROVE)
	  typedef S<Float,length> structure;
	  trove::coalesced_ptr<structure> clover_((structure*)clover);
	  structure v_;
	  for (int i=0; i<length; i++) v_.v[i] = 2.0*(Float)v[i];
	  clover_[parity*volumeCB + x] = v_;
#else
	  for (int i=0; i<length; i++) clover[parity*offset + x*length+i] = 2.0*v[i];
#endif
	}

	size_t Bytes() const { return length*sizeof(Float); }
      };

    /**
       QDPJIT ordering for clover fields
    */
    template <typename Float, int length>
      struct QDPJITOrder {
	typedef typename mapper<Float>::type RegType;
	Float *diag; 	   /**< Pointers to the off-diagonal terms (two parities) */
	Float *offdiag;   /**< Pointers to the diagonal terms (two parities) */
	const int volumeCB;
	const int stride;

	const bool twisted;
	const Float mu2;

      QDPJITOrder(const CloverField &clover, bool inverse, Float *clover_=0) 
      : volumeCB(clover.VolumeCB()), stride(volumeCB), twisted(clover.Twisted()), mu2(clover.Mu2()) {
	offdiag = clover_ ? ((Float**)clover_)[0] : ((Float**)clover.V(inverse))[0];
	diag = clover_ ? ((Float**)clover_)[1] : ((Float**)clover.V(inverse))[1];
      }
	
      bool  Twisted()	const	{return twisted;}
      Float Mu2()	const	{return mu2;}

	__device__ __host__ inline void load(RegType v[length], int x, int parity) const {
	  // the factor of 0.5 comes from a basis change
	  for (int chirality=0; chirality<2; chirality++) {
	    // set diagonal elements
	    for (int i=0; i<6; i++) {
	      v[chirality*36 + i] = 0.5*diag[((i*2 + chirality)*2 + parity)*volumeCB + x];
	    }

	    // the off diagonal elements
	    for (int i=0; i<30; i++) {
	      int z = i%2;
	      int off = i/2;
	      const int idtab[15]={0,1,3,6,10,2,4,7,11,5,8,12,9,13,14};
	      v[chirality*36 + 6 + i] = 0.5*offdiag[(((z*15 + idtab[off])*2 + chirality)*2 + parity)*volumeCB + x];
	    }

	  }
	}
  
	__device__ __host__ inline void save(const RegType v[length], int x, int parity) {
	  // the factor of 2.0 comes from undoing the basis change
	  for (int chirality=0; chirality<2; chirality++) {
	    // set diagonal elements
	    for (int i=0; i<6; i++) {
	      diag[((i*2 + chirality)*2 + parity)*volumeCB + x] = 2.0*v[chirality*36 + i];
	    }

	    // the off diagonal elements
	    for (int i=0; i<30; i++) {
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
    template <typename Float, int length>
      struct BQCDOrder {
	typedef typename mapper<Float>::type RegType;
	Float *clover[2];
	const int volumeCB;
	const int stride;

	const bool twisted;
	const Float mu2;

      BQCDOrder(const CloverField &clover, bool inverse, Float *clover_=0) 
      : volumeCB(clover.Stride()), stride(volumeCB), twisted(clover.Twisted()), mu2(clover.Mu2()) {
	this->clover[0] = clover_ ? clover_ : (Float*)(clover.V(inverse));
	this->clover[1] = (Float*)((char*)this->clover[0] + clover.Bytes()/2);
      }


	bool  Twisted()	const	{return twisted;}
	Float Mu2()	const	{return mu2;}

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
	  for (int i=0; i<6; i++) sign[i] = 1;
	  for (int i=6; i<36; i+=2) {
	    if ( (i >= 10 && i<= 15) || (i >= 18 && i <= 29) )  { sign[i] = -1; sign[i+1] = -1; }
	    else { sign[i] = 1; sign[i+1] = -1; }
	  }
	
	  const int M=length/2;
	  for (int chirality=0; chirality<2; chirality++) 
	    for (int i=0; i<M; i++) 
	      v[chirality*M+i] = sign[i] * clover[parity][x*length+chirality*M+bq[i]];
	
	}
  
	// FIXME implement the save routine for BQCD ordered fields
	__device__ __host__ inline void save(RegType v[length], int x, int parity) {

	};

	size_t Bytes() const { return length*sizeof(Float); }
      };

  } // namespace clover

  // Use traits to reduce the template explosion
  template<typename Float,int N=72, bool add_rho=false> struct clover_mapper { };

  // double precision uses Float2
  template<int N, bool add_rho> struct clover_mapper<double,N,add_rho> { typedef clover::FloatNOrder<double, N, 2, add_rho> type; };

  // single precision uses Float4
  template<int N, bool add_rho> struct clover_mapper<float,N,add_rho> { typedef clover::FloatNOrder<float, N, 4, add_rho> type; };

  // half precision uses Float4
  template<int N, bool add_rho> struct clover_mapper<short,N,add_rho> { typedef clover::FloatNOrder<short, N, 4, add_rho> type; };

  // quarter precision uses Float4
  template<int N, bool add_rho> struct clover_mapper<char,N,add_rho> { typedef clover::FloatNOrder<char, N, 4, add_rho> type; };

} // namespace quda

#endif //_CLOVER_ORDER_H


