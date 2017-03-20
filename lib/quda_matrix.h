#ifndef _QUDA_MATRIX_H_
#define _QUDA_MATRIX_H_

#include <cstdio>

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cuda.h>

#include <float_vector.h>

#include <complex_quda.h>

namespace quda {

  template<class T, class U>
    struct PromoteTypeId
    {
      typedef T Type;
    };

  template<>
    struct PromoteTypeId<complex<float>, float>
    {
      typedef complex<float> Type;
    };

  template<>
    struct PromoteTypeId<float, complex<float> >
    {
      typedef complex<float> Type;
    };

  template<>
    struct PromoteTypeId<complex<double>, double>
    {
      typedef complex<double> Type;
    };

  template<>
    struct PromoteTypeId<double, complex<double> >
    {
      typedef complex<double> Type;
    };

  template<>
    struct PromoteTypeId<double,int>
    {
      typedef double Type;
    };

  template<>
    struct PromoteTypeId<int,double>
    {
      typedef double Type;
    };

  template<>
    struct PromoteTypeId<float,int>
    {
      typedef float Type;
    };

  template<>
    struct PromoteTypeId<int,float>
    {
      typedef float Type;
    };



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

  template<int N>
    __device__ __host__ inline
    int index(int i, int j)
    {
      return i*N + j;
    }

  template<typename Float, typename T> struct gauge_wrapper;

  template<class T, int N>
    class Matrix
    {
      public:
        T data[N*N];

	__device__ __host__ Matrix() { for (int i=0; i<N*N; i++) zero(data[i]); }

	__device__ __host__ Matrix(const T data_[]) { for (int i=0; i<N*N; i++) data[i] = data_[i]; }

        __device__ __host__ inline T const & operator()(int i, int j) const{
          return data[index<N>(i,j)];
        }

        __device__ __host__ inline T & operator()(int i, int j){
          return data[index<N>(i,j)];
        }

        __device__ __host__ inline T const & operator()(int i) const{
          int j = i % N;
          int k = i / N;
          return data[index<N>(j,k)];
        }

        __device__ __host__ inline T& operator()(int i) {
          int j = i % N;
          int k = i / N;
          return data[index<N>(j,k)];
        }

	template<class U>
	  __device__ __host__ inline void operator=(const Matrix<U,N> & b) {
	  for(int i=0; i<N*N; i++) data[i] = b.data[i];
	}

	template<typename S>
	  __device__ __host__ inline Matrix(const gauge_wrapper<typename RealType<T>::type, S> &s);

	template<typename S>
	  __device__ __host__ inline void operator=(const gauge_wrapper<typename RealType<T>::type, S> &s);
    };

  template<class T>
    __device__ __host__ inline T getTrace(const Matrix<T,3>& a)
    {
      return a(0,0) + a(1,1) + a(2,2);
    }


  template<class T>
    __device__ __host__ inline  T getDeterminant(const Matrix<T,3> & a){

      T result;
      result = a(0,0)*(a(1,1)*a(2,2) - a(2,1)*a(1,2))
        - a(0,1)*(a(1,0)*a(2,2) - a(1,2)*a(2,0))
        + a(0,2)*(a(1,0)*a(2,1) - a(1,1)*a(2,0));

      return result;
    }

  template<class T, int N>
    __device__ __host__ inline Matrix<T,N> operator+(const Matrix<T,N> & a, const Matrix<T,N> & b)
    {
      Matrix<T,N> result;
      for(int i=0; i<N*N; i++){
        result.data[i] = a.data[i] + b.data[i];
      }
      return result;
    }


  template<class T, int N>
    __device__ __host__ inline Matrix<T,N> operator+=(Matrix<T,N> & a, const Matrix<T,N> & b)
    {
      for(int i=0; i<N*N; i++){
        a.data[i] += b.data[i];
      }
      return a;
    }


  template<class T, int N> 
    __device__ __host__ inline Matrix<T,N> operator-=(Matrix<T,N> & a, const Matrix<T,N> & b)
    {
      for(int i=0; i<N*N; i++){
        a.data[i] -= b.data[i];
      }
      return a;
    }


  template<class T, int N>
    __device__ __host__ inline Matrix<T,N> operator-(const Matrix<T,N> & a, const Matrix<T,N> & b)
    {
      Matrix<T,N> result;
      for(int i=0; i<N*N; i++){
        result.data[i] = a.data[i] - b.data[i];
      }
      return result;
    }



  template<class T, int N, class S>
    __device__ __host__ inline Matrix<T,N> operator*(const S & scalar, const Matrix<T,N> & a){
      Matrix<T,N> result;
      for(int i=0; i<N*N; ++i){
        result.data[i] = scalar*a.data[i];
      }
      return result;
    }


  template<class T, int N, class S>
    __device__ __host__ inline Matrix<T,N> operator*(const Matrix<T,N> & a, const S & scalar){
      return scalar*a;
    }

  template<class T, int N, class S>
    __device__ __host__ inline Matrix<T,N> operator *=(Matrix<T,N> & a, const S & scalar){
      a = scalar*a;
      return a;
    }

  template<class T, int N>
    __device__ __host__ inline Matrix<T,N> operator-(const Matrix<T,N> & a){
      Matrix<T,N> result;
      for(int i=0; i<(N*N); ++i){
        result.data[i] = -1*a.data[i]; 
      }
      return result;
    }


  /**
     @brief Generic implementation of matrix multiplication
  */
  template<class T, int N>
    __device__ __host__ inline
    Matrix<T,N> operator*(const Matrix<T,N> &a, const Matrix<T,N> &b)
    {
      Matrix<T,N> result;
      for (int i=0; i<N; i++) {
	for (int k=0; k<N; k++) {
	  result(i,k) = 0.0;
	  for (int j=0; j<N; j++) {
	    result(i,k) += a(i,j) * b(j,k);
	  }
	}
      }
      return result;
    }

  /**
     @brief Specialization of N=3 matrix multiplication
   */
  template<class T>  __device__ __host__ inline
    Matrix<T,3> operator*(const Matrix<T,3> &a, const Matrix<T,3> &b)
    {
      Matrix<T,3> result;
#pragma unroll
      for (int i=0; i<3; i++) {
#pragma unroll
	for (int k=0; k<3; k++) {
	  result(i,k)  = a(i,0) * b(0,k);
	  result(i,k) += a(i,1) * b(1,k);
	  result(i,k) += a(i,2) * b(2,k);
	}
      }
      return result;
    }

  /**
     @brief FP32 specialization of N=3 matrix multiplication that will issue optimal fma instructions
   */
  template<>  __device__ __host__ inline
    Matrix<complex<float>,3> operator*(const Matrix<complex<float>,3> &a, const Matrix<complex<float>,3> &b)
    {
      Matrix<complex<float>,3> result;
#pragma unroll
      for (int i=0; i<3; i++) {
#pragma unroll
	for (int k=0; k<3; k++) {
	  result(i,k).x  = a(i,0).real() * b(0,k).real();
	  result(i,k).x -= a(i,0).imag() * b(0,k).imag();
	  result(i,k).y  = a(i,0).real() * b(0,k).imag();
	  result(i,k).y += a(i,0).imag() * b(0,k).real();

	  result(i,k).x += a(i,1).real() * b(1,k).real();
	  result(i,k).x -= a(i,1).imag() * b(1,k).imag();
	  result(i,k).y += a(i,1).real() * b(1,k).imag();
	  result(i,k).y += a(i,1).imag() * b(1,k).real();

	  result(i,k).x += a(i,2).real() * b(2,k).real();
	  result(i,k).x -= a(i,2).imag() * b(2,k).imag();
	  result(i,k).y += a(i,2).real() * b(2,k).imag();
	  result(i,k).y += a(i,2).imag() * b(2,k).real();
	}
      }
      return result;
    }

  /**
     @brief FP64 specialization of N=3 matrix multiplication that will issue optimal fma instructions
   */
  template<> __device__ __host__ inline
    Matrix<complex<double>,3> operator*(const Matrix<complex<double>,3> &a, const Matrix<complex<double>,3> &b)
    {
      Matrix<complex<double>,3> result;
#pragma unroll
      for (int i=0; i<3; i++) {
#pragma unroll
	for (int k=0; k<3; k++) {
	  result(i,k).x  = a(i,0).real() * b(0,k).real();
	  result(i,k).x -= a(i,0).imag() * b(0,k).imag();
	  result(i,k).y  = a(i,0).real() * b(0,k).imag();
	  result(i,k).y += a(i,0).imag() * b(0,k).real();

	  result(i,k).x += a(i,1).real() * b(1,k).real();
	  result(i,k).x -= a(i,1).imag() * b(1,k).imag();
	  result(i,k).y += a(i,1).real() * b(1,k).imag();
	  result(i,k).y += a(i,1).imag() * b(1,k).real();

	  result(i,k).x += a(i,2).real() * b(2,k).real();
	  result(i,k).x -= a(i,2).imag() * b(2,k).imag();
	  result(i,k).y += a(i,2).real() * b(2,k).imag();
	  result(i,k).y += a(i,2).imag() * b(2,k).real();
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
      for (int i=0; i<N; i++) {
	for (int k=0; k<N; k++) {
	  result(i,k) = 0.0;
	  for (int j=0; j<N; j++) {
	    result(i,k) += a(i,j) * b(j,k);
	  }
	}
      }
      return result;
    }


  template<class T, class U>
    __device__ __host__ inline
    Matrix<typename PromoteTypeId<T,U>::Type,3> operator*(const Matrix<T,3> &a, const Matrix<U,3> &b)
    {
      Matrix<typename PromoteTypeId<T,U>::Type,3> result;
#pragma unroll
      for (int i=0; i<3; i++) {
#pragma unroll
	for (int k=0; k<3; k++) {
	  result(i,k) = a(i,0) * b(0,k) + a(i,1) * b(1,k) + a(i,2) * b(2,k);
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
      for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
          result(i,j) = conj( other(j,i) );
        }
      }
      return result;
    }


  template<class T> 
    __device__  __host__ inline
    void computeMatrixInverse(const Matrix<T,3>& u, Matrix<T,3>* uinv)
    {

      const T & det = getDeterminant(u);
      const T & det_inv = static_cast<typename T::value_type>(1.0)/det;

      T temp;

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

      return;
    } 



  template<class T, int N>
    __device__ __host__ inline
    void setIdentity(Matrix<T,N>* m){

      for(int i=0; i<N; ++i){
        (*m)(i,i) = 1;
        for(int j=i+1; j<N; ++j){
          (*m)(i,j) = (*m)(j,i) = 0;
        }
      }
      return;
    }


  template<int N>
    __device__ __host__ inline
    void setIdentity(Matrix<float2,N>* m){

      for(int i=0; i<N; ++i){
        (*m)(i,i) = make_float2(1,0);
        for(int j=i+1; j<N; ++j){
          (*m)(i,j) = (*m)(j,i) = make_float2(0.,0.);    
        }
      }
      return;
    }


  template<int N>
    __device__ __host__ inline
    void setIdentity(Matrix<double2,N>* m){

      for(int i=0; i<N; ++i){
        (*m)(i,i) = make_double2(1,0);
        for(int j=i+1; j<N; ++j){
          (*m)(i,j) = (*m)(j,i) = make_double2(0.,0.);    
        }
      }
      return;
    }


  // Need to write more generic code for this!
  template<class T, int N>
    __device__ __host__ inline
    void setZero(Matrix<T,N>* m){

      for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
          (*m)(i,j) = 0;
        }
      }
      return;
    }


  template<int N>
    __device__ __host__ inline
    void setZero(Matrix<float2,N>* m){

      for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
          (*m)(i,j) = make_float2(0.,0.);
        }
      }
      return;
    }


  template<int N>
    __device__ __host__ inline
    void setZero(Matrix<double2,N>* m){

      for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
          (*m)(i,j) = make_double2(0.,0.);
        }
      }
      return;
    }


  template<typename Complex,int N>
    __device__ __host__ inline void makeAntiHerm(Matrix<Complex,N> &m) {
    typedef typename Complex::value_type real;
    // first make the matrix anti-hermitian
    Matrix<Complex,N> am = m - conj(m);

    // second make it traceless
    real imag_trace = 0.0;
    for (int i=0; i<N; i++) imag_trace += am(i,i).y;
    for (int i=0; i<N; i++) {
      am(i,i).y -= imag_trace/N;
    }
    m = 0.5*am;
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
      for(int i=0; i<N; ++i){
        (*a)[i] = m(i,c); // c is the column index
      }
      return;
    }


  template<class T, int N>
    __device__ __host__ inline
    void outerProd(const Array<T,N>& a, const Array<T,N> & b, Matrix<T,N>* m){
      for(int i=0; i<N; ++i){
        const T conjb_i = conj(b[i]);
        for(int j=0; j<N; ++j){
          (*m)(j,i) = a[j]*conjb_i; // we reverse the ordering of indices because it cuts down on the number of function calls
        }
      }
      return;
    }

  template<class T, int N>
    __device__ __host__ inline 
    void outerProd(const T (&a)[N], const T (&b)[N], Matrix<T,N>* m){
      for(int i=0; i<N; ++i){
        const T conjb_i = conj(b[i]);
        for(int j=0; j<N; ++j){
          (*m)(j,i) = a[j]*conjb_i; // we reverse the ordering of indices because it cuts down on the number of function calls
        }
      }
      return;
    }


  // Need some print utilities
  template<class T, int N>
    std::ostream & operator << (std::ostream & os, const Matrix<T,N> & m){
      for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
          os << m(i,j) << " ";
        }
        if(i<N-1) os << std::endl;
      }
      return os;
    }


  template<class T, int N>
    std::ostream & operator << (std::ostream & os, const Array<T,N> & a){
      for(int i=0; i<N; ++i){
        os << a[i] << " ";
      }
      return os;
    }


  template<class T, class U>
    __device__ inline
    void loadLinkVariableFromArray(const T* const array, const int dir, const int idx, const int stride, Matrix<U,3> *link)
    {
      for(int i=0; i<9; ++i){
        link->data[i] = array[idx + (dir*9 + i)*stride];
      }
      return;
    }


  template<class T, class U, int N>
    __device__ inline 
    void loadMatrixFromArray(const T* const array, const int idx, const int stride, Matrix<U,N> *mat)
    {
      for(int i=0; i<(N*N); ++i){
        mat->data[i] = array[idx + i*stride];
      }
    }


  __device__ inline  
    void loadLinkVariableFromArray(const float2* const array, const int dir, const int idx, const int stride, Matrix<complex<double>,3> *link)
    { 
      float2 single_temp; 
      for(int i=0; i<9; ++i){
        single_temp = array[idx + (dir*9 + i)*stride];
        link->data[i].x = single_temp.x;
        link->data[i].y = single_temp.y;
      }
      return;
    }



  template<class T, int N, class U>
    __device__ inline 
    void writeMatrixToArray(const Matrix<T,N>& mat, const int idx, const int stride, U* const array)
    {
      for(int i=0; i<(N*N); ++i){
        array[idx + i*stride] = mat.data[i];
      }
    }

  __device__ inline 
    void appendMatrixToArray(const Matrix<complex<double>,3>& mat, const int idx, const int stride, double2* const array)
    {
      for(int i=0; i<9; ++i){
        array[idx + i*stride].x += mat.data[i].x;
        array[idx + i*stride].y += mat.data[i].y;
      }
    }

  __device__ inline 
    void appendMatrixToArray(const Matrix<complex<float>,3>& mat, const int idx, const int stride, float2* const array)
    {
      for(int i=0; i<9; ++i){
        array[idx + i*stride].x += mat.data[i].x;
        array[idx + i*stride].y += mat.data[i].y;
      }
    }


  template<class T, class U>
    __device__ inline
    void writeLinkVariableToArray(const Matrix<T,3> & link, const int dir, const int idx, const int stride, U* const array)
    {
      for(int i=0; i<9; ++i){ 
        array[idx + (dir*9 + i)*stride] = link.data[i];
      }
      return;
    }




  __device__ inline 
    void writeLinkVariableToArray(const Matrix<complex<double>,3> & link, const int dir, const int idx, const int stride, float2* const array)
    {
      float2 single_temp;

      for(int i=0; i<9; ++i){ 
        single_temp.x = link.data[i].x;
        single_temp.y = link.data[i].y;
        array[idx + (dir*9 + i)*stride] = single_temp;
      }
      return;
    }


  template<class T>
    __device__ inline
    void loadMomentumFromArray(const T* const array, const int dir, const int idx, const int stride, Matrix<T,3> *mom)
    {
      T temp2[5];
      temp2[0] = array[idx + dir*stride*5];
      temp2[1] = array[idx + dir*stride*5 + stride];
      temp2[2] = array[idx + dir*stride*5 + 2*stride];
      temp2[3] = array[idx + dir*stride*5 + 3*stride];
      temp2[4] = array[idx + dir*stride*5 + 4*stride];

      mom->data[0].x = 0.;
      mom->data[0].y = temp2[3].x;
      mom->data[1] = temp2[0];
      mom->data[2] = temp2[1];

      mom->data[3].x = -mom->data[1].x;
      mom->data[3].y =  mom->data[1].y;
      mom->data[4].x = 0.;
      mom->data[4].y = temp2[3].y;
      mom->data[5]   = temp2[2];

      mom->data[6].x = -mom->data[2].x;
      mom->data[6].y =  mom->data[2].y;

      mom->data[7].x = -mom->data[5].x;
      mom->data[7].y =  mom->data[5].y;

      mom->data[8].x = 0.;
      mom->data[8].y = temp2[4].x;

      return;
    }



  template<class T, class U>
    __device__  inline 
    void writeMomentumToArray(const Matrix<T,3> & mom, const int dir, const int idx, const U coeff, const int stride, T* const array)
    {
      typedef typename T::value_type real;
      T temp2;
      temp2.x = (mom.data[1].x - mom.data[3].x)*0.5*coeff;
      temp2.y = (mom.data[1].y + mom.data[3].y)*0.5*coeff;
      array[idx + dir*stride*5] = temp2;

      temp2.x = (mom.data[2].x - mom.data[6].x)*0.5*coeff;
      temp2.y = (mom.data[2].y + mom.data[6].y)*0.5*coeff;
      array[idx + dir*stride*5 + stride] = temp2;

      temp2.x = (mom.data[5].x - mom.data[7].x)*0.5*coeff;
      temp2.y = (mom.data[5].y + mom.data[7].y)*0.5*coeff;
      array[idx + dir*stride*5 + stride*2] = temp2;

      const real temp = (mom.data[0].y + mom.data[4].y + mom.data[8].y)*0.3333333333333333333333333;
      temp2.x =  (mom.data[0].y-temp)*coeff;
      temp2.y =  (mom.data[4].y-temp)*coeff;
      array[idx + dir*stride*5 + stride*3] = temp2;

      temp2.x = (mom.data[8].y - temp)*coeff;
      temp2.y = 0.0;
      array[idx + dir*stride*5 + stride*4] = temp2;

      return;
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

      return;
    } 
  // template this! 
  inline void copyArrayToLink(Matrix<float2,3>* link, float* array){
    for(int i=0; i<3; ++i){
      for(int j=0; j<3; ++j){
        (*link)(i,j).x = array[(i*3+j)*2];
        (*link)(i,j).y = array[(i*3+j)*2 + 1];
      }
    }
    return;
  }

  template<class Cmplx, class Real>
    inline void copyArrayToLink(Matrix<Cmplx,3>* link, Real* array){
      for(int i=0; i<3; ++i){
        for(int j=0; j<3; ++j){
          (*link)(i,j).x = array[(i*3+j)*2];
          (*link)(i,j).y = array[(i*3+j)*2 + 1];
        }
      }
      return;
    }


  // and this!
  inline void copyLinkToArray(float* array, const Matrix<float2,3>& link){
    for(int i=0; i<3; ++i){
      for(int j=0; j<3; ++j){
        array[(i*3+j)*2] = link(i,j).x;
        array[(i*3+j)*2 + 1] = link(i,j).y;
      }
    }
    return;
  }

  // and this!
  template<class Cmplx, class Real>
    inline void copyLinkToArray(Real* array, const Matrix<Cmplx,3>& link){
      for(int i=0; i<3; ++i){
        for(int j=0; j<3; ++j){
          array[(i*3+j)*2] = link(i,j).x;
          array[(i*3+j)*2 + 1] = link(i,j).y;
        }
      }
      return;
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
    bool isUnitary(const Matrix<Cmplx,3>& matrix, double max_error)
  {
    const Matrix<Cmplx,3> identity = conj(matrix)*matrix;

    for(int i=0; i<3; ++i){
      if( fabs(identity(i,i).x - 1.0) > max_error || fabs(identity(i,i).y) > max_error) return false;
      for(int j=i+1; j<3; ++j){
	if( fabs(identity(i,j).x) > max_error || fabs(identity(i,j).y) > max_error
	    ||  fabs(identity(j,i).x) > max_error || fabs(identity(j,i).y) > max_error ){
	  return false;
	}
      }
    }

    for (int i=0; i<3; i++) {
      for (int j=0; j<3; j++) {
	if (isnan(matrix(i,j).x) || isnan(matrix(i,j).y)) return false;
      }
    }

    return true;
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
      for(i=0; i<3; ++i) 
	for(j=0; j<3; ++j) 
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
  
  template<class T> 
    __device__  __host__ inline
    void exponentiate_iQ(const Matrix<T,3>& Q, Matrix<T,3>* exp_iQ)
    {
      // Use Cayley-Hamilton Theorem for SU(3) exp{iQ}.
      // This algorithm is outlined in
      // http://arxiv.org/pdf/hep-lat/0311018v1.pdf
      // Equation numbers in the paper are referenced by [eq_no].

      //Declarations
      typedef decltype(Q(0,0).x) undMatType;

      undMatType inv3 = 1.0/3.0;      
      undMatType c0, c1, c0_max, Tr_re;
      undMatType f0_re, f0_im, f1_re, f1_im, f2_re, f2_im;
      undMatType theta;
      undMatType u_p, w_p;  //u, w parameters.
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
      c0_max = 2*pow(c1*inv3,1.5);
      
      //[25]
      theta  = acos(c0/c0_max);
      
      //[23]
      u_p = sqrt(c1*inv3)*cos(theta*inv3);
      
      //[24]
      w_p = sqrt(c1)*sin(theta*inv3);
      
      //[29] Construct objects for fj = hj/(9u^2 - w^2).
      undMatType u_sq = u_p*u_p;
      undMatType w_sq = w_p*w_p;
      undMatType denom_inv = 1.0/(9*u_sq - w_sq);
      undMatType exp_iu_re = cos(u_p);
      undMatType exp_iu_im = sin(u_p);
      undMatType exp_2iu_re = exp_iu_re*exp_iu_re - exp_iu_im*exp_iu_im;
      undMatType exp_2iu_im = 2*exp_iu_re*exp_iu_im;
      undMatType cos_w = cos(w_p);
      undMatType sinc_w;
      undMatType hj_re = 0.0;
      undMatType hj_im = 0.0;
  
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
      setZero(exp_iQ);
      Matrix<T,3> UnitM;
      setIdentity(&UnitM);
      // +f0*I
      temp1 = f0_c * UnitM;
      *exp_iQ = temp1;
      
      // +f1*Q
      temp1 = f1_c * Q;
      *exp_iQ += temp1;

      // +f2*Q^2
      temp1 = Q * Q;
      temp2 = f2_c * temp1;
      *exp_iQ += temp2;

      //exp(iQ) is now defined.
      return;
    }


} // end namespace quda
#endif // _QUDA_MATRIX_H_
