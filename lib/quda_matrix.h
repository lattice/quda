#ifndef _QUDA_MATRIX_H_
#define _QUDA_MATRIX_H_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cuda.h>

namespace hisq{

  // Given a real type T, returns the corresponding complex type
  template<class T>
    struct ComplexTypeId;

  template<>
    struct ComplexTypeId<float>
    {
      typedef float2 Type;
    };

  template<>
    struct ComplexTypeId<double>
    {
      typedef double2 Type;
    };

  template<class T> 
    struct RealTypeId; 

  template<>
    struct RealTypeId<float2>
    {
      typedef float Type;
    };

  template<>
    struct RealTypeId<double2>
    {
      typedef double Type;
    };


  template<class T, class U>
    struct PromoteTypeId
    {
      typedef T Type;
    };


  template<>
    struct PromoteTypeId<float2, float>
    {
      typedef float2 Type;
    };


  template<>
    struct PromoteTypeId<float, float2>
    {
      typedef float2 Type;
    };


  template<>
    struct PromoteTypeId<double2, double>
    {
      typedef double2 Type;
    };


  template<>
    struct PromoteTypeId<double, double2>
    {
      typedef double2 Type;
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




  template<class Cmplx>
    __device__ __host__ 
    Cmplx makeComplex(const typename RealTypeId<Cmplx>::Type & a, const typename RealTypeId<Cmplx>::Type & b)
    {
      Cmplx z;
      z.x = a; z.y = b;
      return z;
    }


  __device__ __host__
    double2 makeComplex(const double & a, const double & b){
      return make_double2(a,b);
    }

  __device__ __host__
    float2 makeComplex(const float & a, const float & b){
      return make_float2(a,b);
    } 

  /*
  //  doesn't seem to work
  template<class Cmplx> 
  __device__ __host__ 
  const Cmplx & operator+=(Cmplx & a, const Cmplx & b){
  a.x += b.x; 
  a.y += b.y;
  return a;
  }
   */

  template<class Cmplx>
    __device__ __host__ Cmplx operator+(const Cmplx & a, const Cmplx & b){
      return makeComplex(a.x+b.x,a.y+b.y);
    }

  template<class Cmplx>
    __device__ __host__ Cmplx operator-(const Cmplx & a, const Cmplx & b)
    {
      return makeComplex(a.x-b.x,a.y-b.y);
    }

  template<class Cmplx>
    __device__ __host__ Cmplx operator*(const Cmplx & a, const typename RealTypeId<Cmplx>::Type & scalar)
    {
      return makeComplex(a.x*scalar,a.y*scalar);
    }

  template<class Cmplx>
    __device__ __host__ Cmplx operator/(const Cmplx & a, const typename RealTypeId<Cmplx>::Type & scalar)
    {
      return makeComplex(a.x/scalar,a.y/scalar);
    }

  template<class Cmplx>
    __device__ __host__ Cmplx operator+(const Cmplx & a, const typename RealTypeId<Cmplx>::Type & scalar)
    {
      return makeComplex(a.x+scalar,a.y);
    }

  template<class Cmplx>
    __device__ __host__ Cmplx operator+(const typename RealTypeId<Cmplx>::Type & scalar, const Cmplx & a)
    {
      return makeComplex(a.x+scalar,a.y);
    }

  template<class Cmplx>
    __device__ __host__ Cmplx operator-(const Cmplx & a, const typename RealTypeId<Cmplx>::Type & scalar)
    {
      return makeComplex(a.x-scalar,a.y);
    }

  template<class Cmplx>
    __device__ __host__ Cmplx operator-(const typename RealTypeId<Cmplx>::Type & scalar, const Cmplx & a)
    {
      return makeComplex(scalar-a.x,-a.y);
    }


  template<class Cmplx>
    __device__ __host__ Cmplx operator*(const typename RealTypeId<Cmplx>::Type & scalar, const Cmplx & b)
    {
      return operator*(b,scalar);
    }

  template<class Cmplx>
    __device__ __host__ Cmplx operator*(const Cmplx & a, const Cmplx & b)
    {
      return makeComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
    }

  template<class Cmplx>
    __device__ __host__ Cmplx conj(const Cmplx & a)
    {
      return makeComplex(a.x,-a.y);
    }

  __device__ __host__ double conj(const double & a)
  {
    return a;
  }

  __device__ __host__ float conj(const float & a)
  {
    return a;
  }



  template<class Cmplx>
    __device__ __host__
    Cmplx getPreciseInverse(const Cmplx & z){
      typename RealTypeId<Cmplx>::Type ratio, max, denom;
      if( fabs(z.x) > fabs(z.y) ){ max = z.x; ratio = z.y/max; }else{ max=z.y; ratio = z.x/max; }
      denom = max*max*(1 + ratio*ratio);
      return makeComplex(z.x/denom, -z.y/denom);
    }


  // for printing
  std::ostream & operator << (std::ostream & os, const float2 & z){
    os << "(" << z.x << "," << z.y << ")";
    return os;
  }

  std::ostream & operator << (std::ostream & os, const double2 & z){
    os << "(" << z.x << "," << z.y << ")";
    return os;
  }



  template<class T>
    struct Zero
    {
      //static const T val;
      __device__ __host__
        static T val();
    };

  template<>
    __device__ __host__
    float2 Zero<float2>::val()  
    {
      return make_float2(0.,0.);
    }

  template<>
    __device__ __host__
    double2 Zero<double2>::val()
    {
      return make_double2(0.,0.);
    }



  template<class T>
    struct Identity
    {
      __device__  __host__
        static T val();
    };

  template<>
    __device__ __host__
    float2 Identity<float2>::val(){
      return make_float2(1.,0.);
    }

  template<>
    __device__ __host__
    double2 Identity<double2>::val(){
      return make_double2(1.,0.);
    }

  template<int N>
    inline  __device__ __host__
    int index(int i, int j)
    {
      return i*N + j;
    }

  template<class T, int N>
    class Matrix
    {
      public:
        T data[N*N];

        __device__ __host__ T const & operator()(int i, int j) const{
          return data[index<N>(i,j)];
        }

        __device__ __host__ T & operator()(int i, int j){
          return data[index<N>(i,j)];
        }
    };

  template<class T>
    __device__ __host__ T getTrace(const Matrix<T,3>& a)
    {
      return a(0,0) + a(1,1) + a(2,2);
    }


  template<class T>
    __device__ __host__  T getDeterminant(const Matrix<T,3> & a){

      T result;
      result = a(0,0)*(a(1,1)*a(2,2) - a(2,1)*a(1,2))
        - a(0,1)*(a(1,0)*a(2,2) - a(1,2)*a(2,0))
        + a(0,2)*(a(1,0)*a(2,1) - a(1,1)*a(2,0));

      return result;
    }

  template<class T, int N>
    __device__ __host__ Matrix<T,N> operator+(const Matrix<T,N> & a, const Matrix<T,N> & b)
    {
      Matrix<T,N> result;
      for(int i=0; i<N*N; i++){
        result.data[i] = a.data[i] + b.data[i];
      }
      return result;
    }



  template<class T, int N>
    __device__ __host__ Matrix<T,N> operator-(const Matrix<T,N> & a, const Matrix<T,N> & b)
    {
      Matrix<T,N> result;
      for(int i=0; i<N*N; i++){
        result.data[i] = a.data[i] - b.data[i];
      }
      return result;
    }



  template<class T, int N, class S>
    __device__ __host__ Matrix<T,N> operator*(const S & scalar, const Matrix<T,N> & a){
      Matrix<T,N> result;
      for(int i=0; i<N*N; ++i){
        result.data[i] = scalar*a.data[i];
      }
      return result;
    }


  template<class T, int N, class S>
    __device__ __host__ Matrix<T,N> operator*(const Matrix<T,N> & a, const S & scalar){
      return scalar*a;
    }



  template<class T>
    __device__ __host__
    Matrix<T,3> operator*(const Matrix<T,3> & a, const Matrix<T,3> & b)
    {
      // The compiler has a hard time unrolling nested loops,
      // so here I do it by hand. 
      // I could do something more sophisticated in the future.
      Matrix<T,3> result;
      result(0,0) = a(0,0)*b(0,0) + a(0,1)*b(1,0) + a(0,2)*b(2,0);
      result(0,1) = a(0,0)*b(0,1) + a(0,1)*b(1,1) + a(0,2)*b(2,1);
      result(0,2) = a(0,0)*b(0,2) + a(0,1)*b(1,2) + a(0,2)*b(2,2);
      result(1,0) = a(1,0)*b(0,0) + a(1,1)*b(1,0) + a(1,2)*b(2,0);
      result(1,1) = a(1,0)*b(0,1) + a(1,1)*b(1,1) + a(1,2)*b(2,1);
      result(1,2) = a(1,0)*b(0,2) + a(1,1)*b(1,2) + a(1,2)*b(2,2);
      result(2,0) = a(2,0)*b(0,0) + a(2,1)*b(1,0) + a(2,2)*b(2,0);
      result(2,1) = a(2,0)*b(0,1) + a(2,1)*b(1,1) + a(2,2)*b(2,1);
      result(2,2) = a(2,0)*b(0,2) + a(2,1)*b(1,2) + a(2,2)*b(2,2);
      return result;
    }


  // This is so that I can multiply real and complex matrices
  template<class T, class U>
    __device__ __host__
    Matrix<typename PromoteTypeId<T,U>::Type,3> operator*(const Matrix<T,3> & a, const Matrix<U,3> & b)
    {
      Matrix<typename PromoteTypeId<T,U>::Type,3> result;
      result(0,0) = a(0,0)*b(0,0) + a(0,1)*b(1,0) + a(0,2)*b(2,0);
      result(0,1) = a(0,0)*b(0,1) + a(0,1)*b(1,1) + a(0,2)*b(2,1);
      result(0,2) = a(0,0)*b(0,2) + a(0,1)*b(1,2) + a(0,2)*b(2,2);
      result(1,0) = a(1,0)*b(0,0) + a(1,1)*b(1,0) + a(1,2)*b(2,0);
      result(1,1) = a(1,0)*b(0,1) + a(1,1)*b(1,1) + a(1,2)*b(2,1);
      result(1,2) = a(1,0)*b(0,2) + a(1,1)*b(1,2) + a(1,2)*b(2,2);
      result(2,0) = a(2,0)*b(0,0) + a(2,1)*b(1,0) + a(2,2)*b(2,0);
      result(2,1) = a(2,0)*b(0,1) + a(2,1)*b(1,1) + a(2,2)*b(2,1);
      result(2,2) = a(2,0)*b(0,2) + a(2,1)*b(1,2) + a(2,2)*b(2,2);
      return result;
    }



  template<class T>
    __device__ __host__
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
    __device__ __host__
    Matrix<T,N> conj(const Matrix<T,N> & other){
      Matrix<T,N> result;
      for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
          result(i,j) = conj(other(j,i));
        }
      }
      return result;
    }


  template<class T> 
    __device__  __host__
    void computeMatrixInverse(const Matrix<T,3>& u, Matrix<T,3>* uinv)
    {

      const T & det = getDeterminant(u);
      const T & det_inv = getPreciseInverse(det);

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
    inline __device__ __host__
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
    inline __device__ __host__
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
    inline __device__ __host__
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
    inline __device__ __host__
    void setZero(Matrix<T,N>* m){

      for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
          (*m)(i,j) = 0;
        }
      }
      return;
    }


  template<int N>
    inline __device__ __host__
    void setZero(Matrix<float2,N>* m){

      for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
          (*m)(i,j) = make_float2(0.,0.);
        }
      }
      return;
    }


  template<int N>
    inline __device__ __host__
    void setZero(Matrix<double2,N>* m){

      for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
          (*m)(i,j) = make_double2(0.,0.);
        }
      }
      return;
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
        __device__ __host__
          T const & operator[](int i) const{
            return data[i];
          }

        // assignment function
        __device__ __host__ 
          T & operator[](int i){
            return data[i];
          }
    };


  template<class T, int N>
    __device__  __host__ 
    void copyColumn(const Matrix<T,N>& m, int c, Array<T,N>* a)
    {
      for(int i=0; i<N; ++i){
        (*a)[i] = m(i,c); // c is the column index
      }
      return;
    }


  template<class T, int N>
    __device__ __host__
    void outerProd(const Array<T,N>& a, const Array<T,N> & b, Matrix<T,N>* m){
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


  template<class T>
    __device__
    void loadLinkVariableFromArray(const T* const array, int dir, int idx, int stride, Matrix<T,3> *link)
    {
      for(int i=0; i<9; ++i){
        link->data[i] = array[idx + (dir*9 + i)*stride];
      }
      return;
    }


    __device__  
   void loadLinkVariableFromArray(const float2* const array, int dir, int idx, int stride, Matrix<double2,3> *link)
   { 
		 float2 single_temp; 
	   for(int i=0; i<9; ++i){
        single_temp = array[idx + (dir*9 + i)*stride];
				link->data[i].x = single_temp.x;
				link->data[i].y = single_temp.y;
		 }
		 return;
   }
   




  template<class T>
    __device__
    void writeLinkVariableToArray(const Matrix<T,3> & link,  int dir, int idx, int stride, T* const array)
    {
      for(int i=0; i<9; ++i){ 
        array[idx + (dir*9 + i)*stride] = link.data[i];
      }
      return;
    }




    __device__ 
	  void writeLinkVariableToArray(const Matrix<double2,3> & link, int dir, int idx, int stride, float2* const array)
   {
	    float2 single_temp;

      for(int i=0; i<9; ++i){ 
				single_temp.x = link.data[i].x;
				single_temp.y = link.data[i].y;
        array[idx + (dir*9 + i)*stride] = single_temp;
      }
     return;
   }


  template<class Cmplx> 
    __device__  __host__
    void computeLinkInverse(Matrix<Cmplx,3>* uinv, const Matrix<Cmplx,3>& u)
    {

      const Cmplx & det = getDeterminant(u);
      const Cmplx & det_inv = getPreciseInverse(det);

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
        void copyArrayToLink(Matrix<float2,3>* link, float* array){
          for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
              (*link)(i,j).x = array[(i*3+j)*2];
              (*link)(i,j).y = array[(i*3+j)*2 + 1];
            }
          }
          return;
        }
	
	template<class Cmplx, class Real>
        void copyArrayToLink(Matrix<Cmplx,3>* link, Real* array){
          for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
              (*link)(i,j).x = array[(i*3+j)*2];
              (*link)(i,j).y = array[(i*3+j)*2 + 1];
            }
          }
          return;
        }
	
        
        // and this!
        void copyLinkToArray(float* array, const Matrix<float2,3>& link){
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
        void copyLinkToArray(Real* array, const Matrix<Cmplx,3>& link){
          for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
              array[(i*3+j)*2] = link(i,j).x;
              array[(i*3+j)*2 + 1] = link(i,j).y;
            }
          }
          return;
        }



        // and this!
	template<class Cmplx>
	__host__ __device__
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

} // end namespace hisq
#endif // _QUDA_MATRIX_H_
