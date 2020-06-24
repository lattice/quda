#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#include <register_traits.h>
#include <float_vector.h>
#include <complex_quda.h>

namespace quda {


  template<class T>
    struct Zero
    {
      //static const T val;
      __device__ __host__ inline
        static T val();
    };

  template<>
    __device__ __host__ inline
    float2 Zero<float2>::val()
    {
      return make_float2(0.,0.);
    }

  template<>
    __device__ __host__ inline
    double2 Zero<double2>::val()
    {
      return make_double2(0.,0.);
    }



  template<class T>
    struct Identity
    {
      __device__  __host__ inline
        static T val();
    };

  template<>
    __device__ __host__ inline
    float2 Identity<float2>::val(){
      return make_float2(1.,0.);
    }

  template<>
    __device__ __host__ inline
    double2 Identity<double2>::val(){
      return make_double2(1.,0.);
    }

  template<typename Float, typename T> struct gauge_wrapper;
  template<typename Float, typename T> struct gauge_ghost_wrapper;
  template<typename Float, typename T> struct clover_wrapper;
  template<typename T, int N> struct HMatrix;

  template<class T, int N>
    class Matrix
    {
      typedef typename RealType<T>::type real;

    private:
        __device__ __host__ inline int index(int i, int j) const { return i*N + j; }

      public:
        T data[N*N];

        __device__ __host__ constexpr int size() const { return N; }

        __device__ __host__ inline Matrix() {
#pragma unroll
	  for (int i=0; i<N*N; i++) zero(data[i]);
	}

	__device__ __host__ inline Matrix(const Matrix<T,N> &a) {
#pragma unroll
	  for (int i=0; i<N*N; i++) data[i] = a.data[i];
	}

        template <class U> __device__ __host__ inline Matrix(const Matrix<U, N> &a)
        {
#pragma unroll
          for (int i = 0; i < N * N; i++) data[i] = a.data[i];
        }

        __device__ __host__ inline Matrix(const T data_[])
        {
#pragma unroll
	  for (int i=0; i<N*N; i++) data[i] = data_[i];
        }

        __device__ __host__ inline Matrix(const HMatrix<real, N> &a);

        __device__ __host__ inline T const & operator()(int i, int j) const {
          return data[index(i,j)];
        }

        __device__ __host__ inline T & operator()(int i, int j) {
          return data[index(i,j)];
        }

        __device__ __host__ inline T const & operator()(int i) const {
          int j = i % N;
          int k = i / N;
          return data[index(j,k)];
        }

        __device__ __host__ inline T& operator()(int i) {
          int j = i % N;
          int k = i / N;
          return data[index(j,k)];
        }

	template<class U>
	  __device__ __host__ inline void operator=(const Matrix<U,N> & b) {
#pragma unroll
	  for (int i=0; i<N*N; i++) data[i] = b.data[i];
	}

	template<typename S>
	  __device__ __host__ inline Matrix(const gauge_wrapper<real, S> &s);

	template<typename S>
	  __device__ __host__ inline void operator=(const gauge_wrapper<real, S> &s);

	template<typename S>
	  __device__ __host__ inline Matrix(const gauge_ghost_wrapper<real, S> &s);

	template<typename S>
	  __device__ __host__ inline void operator=(const gauge_ghost_wrapper<real, S> &s);

        /**
           @brief Compute the matrix L1 norm - this is the maximum of
           the absolute column sums.
           @return Compute L1 norm
        */
        __device__ __host__ inline real L1() {
          real l1 = 0;
#pragma unroll
          for (int j=0; j<N; j++) {
            real col_sum = 0;
#pragma unroll
            for (int i=0; i<N; i++) {
              col_sum += abs(data[i*N + j]);
            }
            l1 = col_sum > l1 ? col_sum : l1;
          }
          return l1;
        }

        /**
           @brief Compute the matrix L2 norm.  We actually compute the
           Frobenius norm which is an upper bound on the L2 norm.
           @return Computed L2 norm
        */
        __device__ __host__ inline real L2() {
          real l2 = 0;
#pragma unroll
          for (int j=0; j<N; j++) {
#pragma unroll
            for (int i=0; i<N; i++) {
              l2 += norm(data[i*N + j]);
            }
          }
          return sqrt(l2);
        }

        /**
           @brief Compute the matrix Linfinity norm - this is the maximum of
           the absolute row sums.
           @return Computed Linfinity norm
        */
        __device__ __host__ inline real Linf() {
          real linf = 0;
#pragma unroll
          for (int i=0; i<N; i++) {
            real row_sum = 0;
#pragma unroll
            for (int j=0; j<N; j++) {
              row_sum += abs(data[i*N + j]);
            }
            linf = row_sum > linf ? row_sum : linf;
          }
          return linf;
        }

        /**
	   Return 64-bit XOR checksum computed from the
	   elements of the matrix.  Compute the checksum on each
	   64-bit word that constitutes the Matrix
	 */
	__device__ __host__ inline uint64_t checksum() const {
          // ensure length is rounded up to 64-bit multiple
          constexpr int length = (N*N*sizeof(T) + sizeof(uint64_t) - 1)/ sizeof(uint64_t);
          uint64_t base_[length] = { };
          T *data_ = reinterpret_cast<T*>( static_cast<void*>(base_) );
          for (int i=0; i<N*N; i++) data_[i] = data[i];
          uint64_t checksum_ = base_[0];
          for (int i=1; i<length; i++) checksum_ ^= base_[i];
          return checksum_;
        }

        __device__ __host__ inline bool isUnitary(double max_error) const
        {
          const auto identity = conj(*this) * *this;

#pragma unroll
          for (int i=0; i<N; ++i){
            if( fabs(identity(i,i).real() - 1.0) > max_error ||
                fabs(identity(i,i).imag()) > max_error) return false;

#pragma unroll
            for (int j=i+1; j<N; ++j){
              if( fabs(identity(i,j).real()) > max_error ||
                  fabs(identity(i,j).imag()) > max_error ||
                  fabs(identity(j,i).real()) > max_error ||
                  fabs(identity(j,i).imag()) > max_error ){
                return false;
              }
            }
          }

#pragma unroll
          for (int i=0; i<N; i++) {
#pragma unroll
            for (int j=0; j<N; j++) {
              if (std::isnan((*this)(i,j).real()) ||
                  std::isnan((*this)(i,j).imag())) return false;
            }
          }

          return true;
        }

    };

  /**
     @brief wrapper class that enables us to write to Hmatrices in packed format
     @tparam T Underlying real storage type
     @tparam Hmat The type of HMatrix we are wrapping
  */
  template <typename T, typename Hmat>
  struct HMatrix_wrapper {
    Hmat &mat;
    const int i;
    const int j;
    const int idx;

    __device__ __host__ inline HMatrix_wrapper(Hmat &mat, int i, int j, int idx) : mat(mat), i(i), j(j), idx(idx) { }

    __device__ __host__ inline void operator=(const complex<T> &a) {
      if (i==j) {
	mat.data[idx] = a.real();
      } else if (j<i) {
	mat.data[idx+0] = a.real();
	mat.data[idx+1] = a.imag();
      } else {
	mat.data[idx+0] = a.real();
	mat.data[idx+1] = -a.imag();
      }
    }

    __device__ __host__ inline void operator+=(const complex<T> &a) {
      if (i==j) {
	mat.data[idx] += a.real();
      } else if (j<i) {
	mat.data[idx+0] += a.real();
	mat.data[idx+1] += a.imag();
      } else {
	mat.data[idx+0] += a.real();
	mat.data[idx+1] += -a.imag();
      }
    }
  };

  /**
     @brief Specialized container for Hermitian matrices (e.g., used for wrapping clover matrices)
   */
  template<class T, int N>
    class HMatrix
    {
      friend HMatrix_wrapper<T,HMatrix<T,N> >;
      private:
      // compute index into triangular-packed Hermitian matrix
      __device__ __host__ inline int index(int i, int j) const {
	if (i==j) {
	  return i;
	} else if (j<i) {
	  int k = N*(N-1)/2 - (N-j)*(N-j-1)/2 + i - j - 1;
	  return N + 2*k;
	} else { // i>j
	  // switch coordinates to count from bottom right instead of top left of matrix
	  int k = N*(N-1)/2 - (N-i)*(N-i-1)/2 + j - i - 1;
	  return N + 2*k;
	}
      }

      public:
      T data[N*N]; // store in real-valued array

      __device__ __host__ inline HMatrix() {
#pragma unroll
	for (int i=0; i<N*N; i++) zero(data[i]);
      }

      __device__ __host__ inline HMatrix(const HMatrix<T,N> &a) {
#pragma unroll
	for (int i=0; i<N*N; i++) data[i] = a.data[i];
      }

      __device__ __host__ inline HMatrix(const T data_[]) {
#pragma unroll
	for (int i=0; i<N*N; i++) data[i] = data_[i];
      }

      __device__ __host__ inline complex<T> const operator()(int i, int j) const {
	const int idx = index(i,j);
	if (i==j) {
	  return complex<T>(data[idx],0.0);
	} else if (j<i) {
	  return complex<T>(data[idx], data[idx+1]);
	} else {
	  return complex<T>(data[idx],-data[idx+1]);
	}
      }

      __device__ __host__ inline HMatrix_wrapper<T,HMatrix<T,N> > operator() (int i, int j) {
	return HMatrix_wrapper<T,HMatrix<T,N> >(*this, i, j, index(i,j));
      }

      template<class U>
	__device__ __host__ inline void operator=(const HMatrix<U,N> & b) {
#pragma unroll
	for (int i=0; i<N*N; i++) data[i] = b.data[i];
      }

      template<typename S>
	__device__ __host__ inline HMatrix(const clover_wrapper<T, S> &s);

      template<typename S>
	__device__ __host__ inline void operator=(const clover_wrapper<T, S> &s);

      /**
	 @brief Hermitian matrix square
	 @return Matrix square
      */
      __device__ __host__ inline HMatrix<T,N> square() const {
	HMatrix<T,N> result;
	complex<T> tmp;
#pragma unroll
	for (int i=0; i<N; i++) {
#pragma unroll
	  for (int k=0; k<N; k++) if (i<=k) { // else compiler can't handle triangular unroll
	    tmp.x  = (*this)(i,0).real() * (*this)(0,k).real();
	    tmp.x -= (*this)(i,0).imag() * (*this)(0,k).imag();
	    tmp.y  = (*this)(i,0).real() * (*this)(0,k).imag();
	    tmp.y += (*this)(i,0).imag() * (*this)(0,k).real();
#pragma unroll
	    for (int j=1; j<N; j++) {
	      tmp.x += (*this)(i,j).real() * (*this)(j,k).real();
	      tmp.x -= (*this)(i,j).imag() * (*this)(j,k).imag();
	      tmp.y += (*this)(i,j).real() * (*this)(j,k).imag();
	      tmp.y += (*this)(i,j).imag() * (*this)(j,k).real();
	    }
	    result(i,k) = tmp;
	  }
	}
	return result;
      }

      /**
         @brief Compute the absolute max element of the Hermitian matrix
         @return Absolute Max element
      */
      __device__ __host__ inline T max() const
      {
        HMatrix<T, N> result;
        T max = static_cast<T>(0.0);
#pragma unroll
        for (int i = 0; i < N * N; i++) max = (abs(data[i]) > max ? abs(data[i]) : max);
        return max;
      }

      __device__ __host__ void print() const {
	for (int i=0; i<N; i++) {
	  printf("i=%d ", i);
	  for (int j=0; j<N; j++) {
	    printf(" (%e, %e)", (*this)(i,j).real(), (*this)(i,j).imag());
	  }
	  printf("\n");
	}
	printf("\n");
      }

    };

  template<class T,int N>
    __device__ __host__ Matrix<T,N>::Matrix(const HMatrix<typename RealType<T>::type,N> &a) {
#pragma unroll
    for (int i=0; i<N; i++) {
#pragma unroll
      for (int j=0; j<N; j++) {
	(*this)(i,j) = a(i,j);
      }
    }
  }

  template<class T>
    __device__ __host__ inline T getTrace(const Matrix<T,3>& a)
    {
      return a(0,0) + a(1,1) + a(2,2);
    }


  template< template<typename,int> class Mat, class T>
    __device__ __host__ inline  T getDeterminant(const Mat<T,3> & a){

      T result;
      result = a(0,0)*(a(1,1)*a(2,2) - a(2,1)*a(1,2))
        - a(0,1)*(a(1,0)*a(2,2) - a(1,2)*a(2,0))
        + a(0,2)*(a(1,0)*a(2,1) - a(1,1)*a(2,0));

      return result;
    }

  template< template<typename,int> class Mat, class T, int N>
    __device__ __host__ inline Mat<T,N> operator+(const Mat<T,N> & a, const Mat<T,N> & b)
    {
      Mat<T,N> result;
#pragma unroll
      for (int i=0; i<N*N; i++) result.data[i] = a.data[i] + b.data[i];
      return result;
    }


  template< template<typename,int> class Mat, class T, int N>
    __device__ __host__ inline Mat<T,N> operator+=(Mat<T,N> & a, const Mat<T,N> & b)
    {
#pragma unroll
      for (int i=0; i<N*N; i++) a.data[i] += b.data[i];
      return a;
    }

  template< template<typename,int> class Mat, class T, int N>
    __device__ __host__ inline Mat<T,N> operator+=(Mat<T,N> & a, const T & b)
    {
#pragma unroll
      for (int i=0; i<N; i++) a(i,i) += b;
      return a;
    }

  template< template<typename,int> class Mat, class T, int N>
    __device__ __host__ inline Mat<T,N> operator-=(Mat<T,N> & a, const Mat<T,N> & b)
    {
#pragma unroll
      for (int i=0; i<N*N; i++) a.data[i] -= b.data[i];
      return a;
    }

  template< template<typename,int> class Mat, class T, int N>
    __device__ __host__ inline Mat<T,N> operator-(const Mat<T,N> & a, const Mat<T,N> & b)
    {
      Mat<T,N> result;
#pragma unroll
      for (int i=0; i<N*N; i++) result.data[i] = a.data[i] - b.data[i];
      return result;
    }

  template< template<typename,int> class Mat, class T, int N, class S>
    __device__ __host__ inline Mat<T,N> operator*(const S & scalar, const Mat<T,N> & a){
      Mat<T,N> result;
#pragma unroll
      for (int i=0; i<N*N; ++i) result.data[i] = scalar*a.data[i];
      return result;
    }

  template< template<typename,int> class Mat, class T, int N, class S>
    __device__ __host__ inline Mat<T,N> operator*(const Mat<T,N> & a, const S & scalar){
      return scalar*a;
    }

  template< template<typename,int> class Mat, class T, int N, class S>
    __device__ __host__ inline Mat<T,N> operator *=(Mat<T,N> & a, const S & scalar){
      a = scalar*a;
      return a;
    }

  template< template<typename,int> class Mat, class T, int N>
    __device__ __host__ inline Mat<T,N> operator-(const Mat<T,N> & a){
      Mat<T,N> result;
#pragma unroll
      for (int i=0; i<(N*N); ++i) result.data[i] = -a.data[i];
      return result;
    }


  /**
     @brief Generic implementation of matrix multiplication
  */
  template< template<typename,int> class Mat, class T, int N>
    __device__ __host__ inline Mat<T,N> operator*(const Mat<T,N> &a, const Mat<T,N> &b)
    {
      Mat<T,N> result;
#pragma unroll
      for (int i=0; i<N; i++) {
#pragma unroll
	for (int k=0; k<N; k++) {
	  result(i,k) = a(i,0) * b(0,k);
#pragma unroll
	  for (int j=1; j<N; j++) {
	    result(i,k) += a(i,j) * b(j,k);
	  }
	}
      }
      return result;
    }

  /**
     @brief Specialization of complex matrix multiplication that will issue optimal fma instructions
   */
  template< template<typename> class complex, typename T, int N>
    __device__ __host__ inline Matrix<complex<T>,N> operator*(const Matrix<complex<T>,N> &a, const Matrix<complex<T>,N> &b)
    {
      Matrix<complex<T>,N> result;
#pragma unroll
      for (int i=0; i<N; i++) {
#pragma unroll
	for (int k=0; k<N; k++) {
	  result(i,k).x  = a(i,0).real() * b(0,k).real();
	  result(i,k).x -= a(i,0).imag() * b(0,k).imag();
	  result(i,k).y  = a(i,0).real() * b(0,k).imag();
	  result(i,k).y += a(i,0).imag() * b(0,k).real();
#pragma unroll
	  for (int j=1; j<N; j++) {
	    result(i,k).x += a(i,j).real() * b(j,k).real();
	    result(i,k).x -= a(i,j).imag() * b(j,k).imag();
	    result(i,k).y += a(i,j).real() * b(j,k).imag();
	    result(i,k).y += a(i,j).imag() * b(j,k).real();
	  }
	}
      }
      return result;
    }

  template<class T, int N>
    __device__ __host__ inline Matrix<T,N> operator *=(Matrix<T,N> & a, const Matrix<T,N>& b){

    Matrix<T,N> c = a;
    a = c*b;
    return a;
  }


  // This is so that I can multiply real and complex matrice
  template<class T, class U, int N>
    __device__ __host__ inline
    Matrix<typename PromoteTypeId<T,U>::Type,N> operator*(const Matrix<T,N> &a, const Matrix<U,N> &b)
    {
      Matrix<typename PromoteTypeId<T,U>::Type,N> result;
#pragma unroll
      for (int i=0; i<N; i++) {
#pragma unroll
	for (int k=0; k<N; k++) {
	  result(i,k) = a(i,0) * b(0,k);
#pragma unroll
	  for (int j=1; j<N; j++) {
	    result(i,k) += a(i,j) * b(j,k);
	  }
	}
      }
      return result;
    }


  template<class T>
    __device__ __host__ inline
    Matrix<T,2> operator*(const Matrix<T,2> & a, const Matrix<T,2> & b)
    {
      Matrix<T,2> result;
      result(0,0) = a(0,0)*b(0,0) + a(0,1)*b(1,0);
      result(0,1) = a(0,0)*b(0,1) + a(0,1)*b(1,1);
      result(1,0) = a(1,0)*b(0,0) + a(1,1)*b(1,0);
      result(1,1) = a(1,0)*b(0,1) + a(1,1)*b(1,1);
      return result;
    }


  template<class T, int N>
    __device__ __host__ inline
    Matrix<T,N> conj(const Matrix<T,N> & other){
      Matrix<T,N> result;
#pragma unroll
      for (int i=0; i<N; ++i){
#pragma unroll
        for (int j=0; j<N; ++j){
          result(i,j) = conj( other(j,i) );
        }
      }
      return result;
    }


  template<class T>
    __device__  __host__ inline
    Matrix<T,3> inverse(const Matrix<T,3> &u)
    {
      const T det = getDeterminant(u);
      const T det_inv = static_cast<typename T::value_type>(1.0)/det;
      Matrix<T,3> uinv;

      T temp;

      temp = u(1,1)*u(2,2) - u(1,2)*u(2,1);
      uinv(0,0) = (det_inv*temp);

      temp = u(0,2)*u(2,1) - u(0,1)*u(2,2);
      uinv(0,1) = (temp*det_inv);

      temp = u(0,1)*u(1,2)  - u(0,2)*u(1,1);
      uinv(0,2) = (temp*det_inv);

      temp = u(1,2)*u(2,0) - u(1,0)*u(2,2);
      uinv(1,0) = (det_inv*temp);

      temp = u(0,0)*u(2,2) - u(0,2)*u(2,0);
      uinv(1,1) = (temp*det_inv);

      temp = u(0,2)*u(1,0) - u(0,0)*u(1,2);
      uinv(1,2) = (temp*det_inv);

      temp = u(1,0)*u(2,1) - u(1,1)*u(2,0);
      uinv(2,0) = (det_inv*temp);

      temp = u(0,1)*u(2,0) - u(0,0)*u(2,1);
      uinv(2,1) = (temp*det_inv);

      temp = u(0,0)*u(1,1) - u(0,1)*u(1,0);
      uinv(2,2) = (temp*det_inv);

      return uinv;
    }



  template<class T, int N>
    __device__ __host__ inline
    void setIdentity(Matrix<T,N>* m){

#pragma unroll
      for (int i=0; i<N; ++i){
        (*m)(i,i) = 1;
#pragma unroll
        for (int j=i+1; j<N; ++j){
          (*m)(i,j) = (*m)(j,i) = 0;
        }
      }
    }


  template<int N>
    __device__ __host__ inline
    void setIdentity(Matrix<float2,N>* m){

#pragma unroll
      for (int i=0; i<N; ++i){
        (*m)(i,i) = make_float2(1,0);
#pragma unroll
        for (int j=i+1; j<N; ++j){
          (*m)(i,j) = (*m)(j,i) = make_float2(0.,0.);
        }
      }
    }


  template<int N>
    __device__ __host__ inline
    void setIdentity(Matrix<double2,N>* m){

#pragma unroll
      for (int i=0; i<N; ++i){
        (*m)(i,i) = make_double2(1,0);
#pragma unroll
        for (int j=i+1; j<N; ++j){
          (*m)(i,j) = (*m)(j,i) = make_double2(0.,0.);
        }
      }
    }


  // Need to write more generic code for this!
  template<class T, int N>
    __device__ __host__ inline
    void setZero(Matrix<T,N>* m){

#pragma unroll
      for (int i=0; i<N; ++i){
#pragma unroll
        for (int j=0; j<N; ++j){
          (*m)(i,j) = 0;
        }
      }
    }


  template<int N>
    __device__ __host__ inline
    void setZero(Matrix<float2,N>* m){

#pragma unroll
      for (int i=0; i<N; ++i){
#pragma unroll
        for (int j=0; j<N; ++j){
          (*m)(i,j) = make_float2(0.,0.);
        }
      }
    }


  template<int N>
    __device__ __host__ inline
    void setZero(Matrix<double2,N>* m){

#pragma unroll
      for (int i=0; i<N; ++i){
#pragma unroll
        for (int j=0; j<N; ++j){
          (*m)(i,j) = make_double2(0.,0.);
        }
      }
    }


  template<typename Complex,int N>
    __device__ __host__ inline void makeAntiHerm(Matrix<Complex,N> &m) {
    typedef typename Complex::value_type real;
    // first make the matrix anti-hermitian
    Matrix<Complex,N> am = m - conj(m);

    // second make it traceless
    real imag_trace = 0.0;
#pragma unroll
    for (int i=0; i<N; i++) imag_trace += am(i,i).y;
#pragma unroll
    for (int i=0; i<N; i++) {
      am(i,i).y -= imag_trace/N;
    }
    m = 0.5*am;
  }

  template <typename Complex, int N> __device__ __host__ inline void makeHerm(Matrix<Complex, N> &m)
  {
    typedef typename Complex::value_type real;
    // first make the matrix anti-hermitian
    Matrix<Complex, N> am = conj(m) - m;

    // second make it traceless
    real imag_trace = 0.0;
#pragma unroll
    for (int i = 0; i < N; i++) imag_trace += am(i, i).y;
#pragma unroll
    for (int i = 0; i < N; i++) { am(i, i).y -= imag_trace / N; }
    // third scale out anti hermitian part
    Complex i_2(0.0, 0.5);
    m = i_2 * am;
  }

  // Matrix and array are very similar
  // Maybe I should factor out the similar
  // code. However, I want to make sure that
  // the compiler knows to store the
  // data elements in registers, so I won't do
  // it right now.
  template<class T, int N>
    class Array
    {
      private:
        T data[N];

      public:
        // access function
        __device__ __host__ inline
          T const & operator[](int i) const{
            return data[i];
          }

        // assignment function
        __device__ __host__ inline
          T & operator[](int i){
            return data[i];
          }
    };


  template<class T, int N>
    __device__  __host__ inline
    void copyColumn(const Matrix<T,N>& m, int c, Array<T,N>* a)
    {
#pragma unroll
      for (int i=0; i<N; ++i){
        (*a)[i] = m(i,c); // c is the column index
      }
    }


  // Need some print utilities
  template<class T, int N>
    std::ostream & operator << (std::ostream & os, const Matrix<T,N> & m){
#pragma unroll
      for (int i=0; i<N; ++i){
#pragma unroll
        for (int j=0; j<N; ++j){
          os << m(i,j) << " ";
        }
        if(i<N-1) os << std::endl;
      }
      return os;
    }


  template<class T, int N>
    std::ostream & operator << (std::ostream & os, const Array<T,N> & a){
      for (int i=0; i<N; ++i){
        os << a[i] << " ";
      }
      return os;
    }

  template<class Cmplx>
    __device__  __host__ inline
    void computeLinkInverse(Matrix<Cmplx,3>* uinv, const Matrix<Cmplx,3>& u)
    {
      const Cmplx & det = getDeterminant(u);
      const Cmplx & det_inv = static_cast<typename Cmplx::value_type>(1.0)/det;

      Cmplx temp;

      temp = u(1,1)*u(2,2) - u(1,2)*u(2,1);
      (*uinv)(0,0) = (det_inv*temp);

      temp = u(0,2)*u(2,1) - u(0,1)*u(2,2);
      (*uinv)(0,1) = (temp*det_inv);

      temp = u(0,1)*u(1,2)  - u(0,2)*u(1,1);
      (*uinv)(0,2) = (temp*det_inv);

      temp = u(1,2)*u(2,0) - u(1,0)*u(2,2);
      (*uinv)(1,0) = (det_inv*temp);

      temp = u(0,0)*u(2,2) - u(0,2)*u(2,0);
      (*uinv)(1,1) = (temp*det_inv);

      temp = u(0,2)*u(1,0) - u(0,0)*u(1,2);
      (*uinv)(1,2) = (temp*det_inv);

      temp = u(1,0)*u(2,1) - u(1,1)*u(2,0);
      (*uinv)(2,0) = (det_inv*temp);

      temp = u(0,1)*u(2,0) - u(0,0)*u(2,1);
      (*uinv)(2,1) = (temp*det_inv);

      temp = u(0,0)*u(1,1) - u(0,1)*u(1,0);
      (*uinv)(2,2) = (temp*det_inv);
    }
  // template this!
  inline void copyArrayToLink(Matrix<float2,3>* link, float* array){
#pragma unroll
    for (int i=0; i<3; ++i){
#pragma unroll
      for (int j=0; j<3; ++j){
        (*link)(i,j).x = array[(i*3+j)*2];
        (*link)(i,j).y = array[(i*3+j)*2 + 1];
      }
    }
  }

  template<class Cmplx, class Real>
    inline void copyArrayToLink(Matrix<Cmplx,3>* link, Real* array){
#pragma unroll
      for (int i=0; i<3; ++i){
#pragma unroll
        for (int j=0; j<3; ++j){
          (*link)(i,j).x = array[(i*3+j)*2];
          (*link)(i,j).y = array[(i*3+j)*2 + 1];
        }
      }
    }


  // and this!
  inline void copyLinkToArray(float* array, const Matrix<float2,3>& link){
#pragma unroll
    for (int i=0; i<3; ++i){
#pragma unroll
      for (int j=0; j<3; ++j){
        array[(i*3+j)*2] = link(i,j).x;
        array[(i*3+j)*2 + 1] = link(i,j).y;
      }
    }
  }

  // and this!
  template<class Cmplx, class Real>
    inline void copyLinkToArray(Real* array, const Matrix<Cmplx,3>& link){
#pragma unroll
      for (int i=0; i<3; ++i){
#pragma unroll
        for (int j=0; j<3; ++j){
          array[(i*3+j)*2] = link(i,j).x;
          array[(i*3+j)*2 + 1] = link(i,j).y;
        }
      }
    }

  template<class T>
  __device__ __host__ inline Matrix<T,3> getSubTraceUnit(const Matrix<T,3>& a){
    T tr = (a(0,0) + a(1,1) + a(2,2)) / 3.0;
    Matrix<T,3> res;
    res(0,0) = a(0,0) - tr; res(0,1) = a(0,1); res(0,2) = a(0,2);
    res(1,0) = a(1,0); res(1,1) = a(1,1) - tr; res(1,2) = a(1,2);
    res(2,0) = a(2,0); res(2,1) = a(2,1); res(2,2) = a(2,2) - tr;
    return res;
  }

  template<class T>
  __device__ __host__ inline void SubTraceUnit(Matrix<T,3>& a){
    T tr = (a(0,0) + a(1,1) + a(2,2)) / static_cast<T>(3.0);
    a(0,0) -= tr; a(1,1) -= tr; a(2,2) -= tr;
  }

  template<class T>
  __device__ __host__ inline double getRealTraceUVdagger(const Matrix<T,3>& a, const Matrix<T,3>& b){
    double sum = (double)(a(0,0).x * b(0,0).x  + a(0,0).y * b(0,0).y);
    sum += (double)(a(0,1).x * b(0,1).x  + a(0,1).y * b(0,1).y);
    sum += (double)(a(0,2).x * b(0,2).x  + a(0,2).y * b(0,2).y);
    sum += (double)(a(1,0).x * b(1,0).x  + a(1,0).y * b(1,0).y);
    sum += (double)(a(1,1).x * b(1,1).x  + a(1,1).y * b(1,1).y);
    sum += (double)(a(1,2).x * b(1,2).x  + a(1,2).y * b(1,2).y);
    sum += (double)(a(2,0).x * b(2,0).x  + a(2,0).y * b(2,0).y);
    sum += (double)(a(2,1).x * b(2,1).x  + a(2,1).y * b(2,1).y);
    sum += (double)(a(2,2).x * b(2,2).x  + a(2,2).y * b(2,2).y);
    return sum;
  }

  // and this!
  template<class Cmplx>
    __host__ __device__ inline
    void printLink(const Matrix<Cmplx,3>& link){
      printf("(%lf, %lf)\t", link(0,0).x, link(0,0).y);
      printf("(%lf, %lf)\t", link(0,1).x, link(0,1).y);
      printf("(%lf, %lf)\n", link(0,2).x, link(0,2).y);
      printf("(%lf, %lf)\t", link(1,0).x, link(1,0).y);
      printf("(%lf, %lf)\t", link(1,1).x, link(1,1).y);
      printf("(%lf, %lf)\n", link(1,2).x, link(1,2).y);
      printf("(%lf, %lf)\t", link(2,0).x, link(2,0).y);
      printf("(%lf, %lf)\t", link(2,1).x, link(2,1).y);
      printf("(%lf, %lf)\n", link(2,2).x, link(2,2).y);
      printf("\n");
    }

  template<class Cmplx>
  __device__ __host__
    double ErrorSU3(const Matrix<Cmplx,3>& matrix)
    {
      const Matrix<Cmplx,3> identity_comp = conj(matrix)*matrix;
      double error = 0.0;
      Cmplx temp(0,0);
      int i=0;
      int j=0;

      //error = ||U^dagger U - I||_L2
#pragma unroll
      for (i=0; i<3; ++i)
#pragma unroll
	for (j=0; j<3; ++j)
	  if(i==j) {
	    temp = identity_comp(i,j);
	    temp -= 1.0;
	    error += norm(temp);
	  }
	  else {
	    error += norm(identity_comp(i,j));
	  }
      //error is L2 norm, should be (very close) to zero.
      return error;
    }

    template <class T> __device__ __host__ inline auto exponentiate_iQ(const Matrix<T, 3> &Q)
    {
      // Use Cayley-Hamilton Theorem for SU(3) exp{iQ}.
      // This algorithm is outlined in
      // http://arxiv.org/pdf/hep-lat/0311018v1.pdf
      // Equation numbers in the paper are referenced by [eq_no].

      //Declarations
      typedef decltype(Q(0, 0).x) matType;

      matType inv3 = 1.0 / 3.0;
      matType c0, c1, c0_max, Tr_re;
      matType f0_re, f0_im, f1_re, f1_im, f2_re, f2_im;
      matType theta;
      matType u_p, w_p; // u, w parameters.
      Matrix<T,3> temp1;
      Matrix<T,3> temp2;
      //[14] c0 = det(Q) = 1/3Tr(Q^3)
      const T & det_Q = getDeterminant(Q);
      c0 = det_Q.x;
      //[15] c1 = 1/2Tr(Q^2)
      // Q = Q^dag => Tr(Q^2) = Tr(QQ^dag) = sum_ab [Q_ab * Q_ab^*]
      temp1 = Q;
      temp1 = temp1 * Q;
      Tr_re = getTrace(temp1).x;
      c1 = 0.5*Tr_re;

      //We now have the coeffiecients c0 and c1.
      //We now find: exp(iQ) = f0*I + f1*Q + f2*Q^2
      //      where       fj = fj(c0,c1), j=0,1,2.

      //[17]
      auto sqrt_c1_inv3 = sqrt(c1 * inv3);
      c0_max = 2 * (c1 * inv3 * sqrt_c1_inv3); // reuse the sqrt factor for a fast 1.5 power

      //[25]
      theta  = acos(c0/c0_max);

      sincos(theta * inv3, &w_p, &u_p);
      //[23]
      u_p *= sqrt_c1_inv3;

      //[24]
      w_p *= sqrt(c1);

      //[29] Construct objects for fj = hj/(9u^2 - w^2).
      matType u_sq = u_p * u_p;
      matType w_sq = w_p * w_p;
      matType denom_inv = 1.0 / (9 * u_sq - w_sq);
      matType exp_iu_re, exp_iu_im;
      sincos(u_p, &exp_iu_im, &exp_iu_re);
      matType exp_2iu_re = exp_iu_re * exp_iu_re - exp_iu_im * exp_iu_im;
      matType exp_2iu_im = 2 * exp_iu_re * exp_iu_im;
      matType cos_w = cos(w_p);
      matType sinc_w;
      matType hj_re = 0.0;
      matType hj_im = 0.0;

      //[33] Added one more term to the series given in the paper.
      if (w_p < 0.05 && w_p > -0.05) {
	//1 - 1/6 x^2 (1 - 1/20 x^2 (1 - 1/42 x^2(1 - 1/72*x^2)))
	sinc_w = 1.0 - (w_sq/6.0)*(1 - (w_sq*0.05)*(1 - (w_sq/42.0)*(1 - (w_sq/72.0))));
      }
      else sinc_w = sin(w_p)/w_p;

      //[34] Test for c0 < 0.
      int parity = 0;
      if(c0 < 0) {
	c0 *= -1.0;
	parity = 1;
	//calculate fj with c0 > 0 and then convert all fj.
      }

      //Get all the numerators for fj,
      //[30] f0
      hj_re = (u_sq - w_sq)*exp_2iu_re + 8*u_sq*cos_w*exp_iu_re + 2*u_p*(3*u_sq + w_sq)*sinc_w*exp_iu_im;
      hj_im = (u_sq - w_sq)*exp_2iu_im - 8*u_sq*cos_w*exp_iu_im + 2*u_p*(3*u_sq + w_sq)*sinc_w*exp_iu_re;
      f0_re = hj_re*denom_inv;
      f0_im = hj_im*denom_inv;

      //[31] f1
      hj_re = 2*u_p*exp_2iu_re - 2*u_p*cos_w*exp_iu_re + (3*u_sq - w_sq)*sinc_w*exp_iu_im;
      hj_im = 2*u_p*exp_2iu_im + 2*u_p*cos_w*exp_iu_im + (3*u_sq - w_sq)*sinc_w*exp_iu_re;
      f1_re = hj_re*denom_inv;
      f1_im = hj_im*denom_inv;

      //[32] f2
      hj_re = exp_2iu_re - cos_w*exp_iu_re - 3*u_p*sinc_w*exp_iu_im;
      hj_im = exp_2iu_im + cos_w*exp_iu_im - 3*u_p*sinc_w*exp_iu_re;
      f2_re = hj_re*denom_inv;
      f2_im = hj_im*denom_inv;

      //[34] If c0 < 0, apply tranformation  fj(-c0,c1) = (-1)^j f^*j(c0,c1)
      if (parity == 1) {
	f0_im *= -1.0;
	f1_re *= -1.0;
	f2_im *= -1.0;
      }

      T f0_c;
      T f1_c;
      T f2_c;

      f0_c.x = f0_re;
      f0_c.y = f0_im;

      f1_c.x = f1_re;
      f1_c.y = f1_im;

      f2_c.x = f2_re;
      f2_c.y = f2_im;

      //[19] Construct exp{iQ}
      Matrix<T, 3> exp_iQ;
      setZero(&exp_iQ);
      Matrix<T,3> UnitM;
      setIdentity(&UnitM);
      // +f0*I
      temp1 = f0_c * UnitM;
      exp_iQ = temp1;

      // +f1*Q
      temp1 = f1_c * Q;
      exp_iQ += temp1;

      // +f2*Q^2
      temp1 = Q * Q;
      temp2 = f2_c * temp1;
      exp_iQ += temp2;

      //exp(iQ) is now defined.
      return exp_iQ;
    }

    /**
       Direct port of the TIFR expsu3 algorithm
    */
    template <typename Float> __device__ __host__ void expsu3(Matrix<complex<Float>, 3> &q)
    {
      typedef complex<Float> Complex;

      Complex a2 = (q(3) * q(1) + q(7) * q(5) + q(6) * q(2) - (q(0) * q(4) + (q(0) + q(4)) * q(8))) / (Float)3.0;
      Complex a3 = q(0) * q(4) * q(8) + q(1) * q(5) * q(6) + q(2) * q(3) * q(7) - q(6) * q(4) * q(2)
        - q(3) * q(1) * q(8) - q(0) * q(7) * q(5);

      Complex sg2h3 = sqrt(a3 * a3 - (Float)4. * a2 * a2 * a2);
      Complex cp = exp(log((Float)0.5 * (a3 + sg2h3)) / (Float)3.0);
      Complex cm = a2 / cp;

      Complex r1 = exp(Complex(0.0, 1.0) * (Float)(2.0 * M_PI / 3.0));
      Complex r2 = exp(-Complex(0.0, 1.0) * (Float)(2.0 * M_PI / 3.0));

      Complex w1[3];

      w1[0] = cm + cp;
      w1[1] = r1 * cp + r2 * cm;
      w1[2] = r2 * cp + r1 * cm;
      Complex z1 = q(1) * q(6) - q(0) * q(7);
      Complex z2 = q(3) * q(7) - q(4) * q(6);

      Complex al = w1[0];
      Complex wr21 = (z1 + al * q(7)) / (z2 + al * q(6));
      Complex wr31 = (al - q(0) - wr21 * q(3)) / q(6);

      al = w1[1];
      Complex wr22 = (z1 + al * q(7)) / (z2 + al * q(6));
      Complex wr32 = (al - q(0) - wr22 * q(3)) / q(6);

      al = w1[2];
      Complex wr23 = (z1 + al * q(7)) / (z2 + al * q(6));
      Complex wr33 = (al - q(0) - wr23 * q(3)) / q(6);

      z1 = q(3) * q(2) - q(0) * q(5);
      z2 = q(1) * q(5) - q(4) * q(2);

      al = w1[0];
      Complex wl21 = conj((z1 + al * q(5)) / (z2 + al * q(2)));
      Complex wl31 = conj((al - q(0) - conj(wl21) * q(1)) / q(2));

      al = w1[1];
      Complex wl22 = conj((z1 + al * q(5)) / (z2 + al * q(2)));
      Complex wl32 = conj((al - q(0) - conj(wl22) * q(1)) / q(2));

      al = w1[2];
      Complex wl23 = conj((z1 + al * q(5)) / (z2 + al * q(2)));
      Complex wl33 = conj((al - q(0) - conj(wl23) * q(1)) / q(2));

      Complex xn1 = (Float)1. + wr21 * conj(wl21) + wr31 * conj(wl31);
      Complex xn2 = (Float)1. + wr22 * conj(wl22) + wr32 * conj(wl32);
      Complex xn3 = (Float)1. + wr23 * conj(wl23) + wr33 * conj(wl33);

      Complex d1 = exp(w1[0]);
      Complex d2 = exp(w1[1]);
      Complex d3 = exp(w1[2]);
      Complex y11 = d1 / xn1;
      Complex y12 = d2 / xn2;
      Complex y13 = d3 / xn3;
      Complex y21 = wr21 * d1 / xn1;
      Complex y22 = wr22 * d2 / xn2;
      Complex y23 = wr23 * d3 / xn3;
      Complex y31 = wr31 * d1 / xn1;
      Complex y32 = wr32 * d2 / xn2;
      Complex y33 = wr33 * d3 / xn3;
      q(0) = y11 + y12 + y13;
      q(1) = y21 + y22 + y23;
      q(2) = y31 + y32 + y33;
      q(3) = y11 * conj(wl21) + y12 * conj(wl22) + y13 * conj(wl23);
      q(4) = y21 * conj(wl21) + y22 * conj(wl22) + y23 * conj(wl23);
      q(5) = y31 * conj(wl21) + y32 * conj(wl22) + y33 * conj(wl23);
      q(6) = y11 * conj(wl31) + y12 * conj(wl32) + y13 * conj(wl33);
      q(7) = y21 * conj(wl31) + y22 * conj(wl32) + y23 * conj(wl33);
      q(8) = y31 * conj(wl31) + y32 * conj(wl32) + y33 * conj(wl33);
    }

} // end namespace quda
