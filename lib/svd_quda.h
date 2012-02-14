#ifndef _SVD_QUDA_H_
#define _SVD_QUDA_H_

#include "quda_matrix.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#define DEVICEHOST __device__ __host__
#define SVDPREC 1e-10
#define LOG2 0.69314718055994530942


namespace hisq{

  template<class Cmplx> 
    inline DEVICEHOST
    typename RealTypeId<Cmplx>::Type cabs(const Cmplx & z)
    {
      typename RealTypeId<Cmplx>::Type max, ratio, square;
      if(fabs(z.x) > fabs(z.y)){ max = z.x; ratio = z.y/max; }else{ max=z.y; ratio = z.x/max; }
      square = max*max*(1.0 + ratio*ratio);
      return sqrt(square);
    }


/*
  template<class Cmplx>
    inline DEVICEHOST
    typename RealTypeId<Cmplx>::Type cabs(const Cmplx & z)
    {
      return sqrt(z.x*z.x + z.y*z.y);
    }
*/

  template<class T, class U> 
    inline DEVICEHOST typename PromoteTypeId<T,U>::Type quadSum(const T & a, const U & b){
      typename PromoteTypeId<T,U>::Type ratio, square, max;
      if(fabs(a) > fabs(b)){ max = a; ratio = b/a; }else{ max=b; ratio = a/b; }
      square = max*max*(1.0 + ratio*ratio);
      return sqrt(square);
    }

/*
  template<class T, class U>
    inline DEVICEHOST
    T quadSum(const T & a, const U & b)
    {
      return sqrt(a*a + b*b);
    }
*/



  // In future iterations of the code, I would like to use templates to compute the norm
  DEVICEHOST
    float getNorm(const Array<float2,3>& a){
      float temp1, temp2, temp3;
      temp1 = cabs(a[0]);
      temp2 = cabs(a[1]);
      temp3 = quadSum(temp1,temp2);
      temp1 = cabs(a[2]);
      return quadSum(temp1,temp3);
    }


  DEVICEHOST
    double getNorm(const Array<double2,3>& a){
      double temp1, temp2, temp3;
      temp1 = cabs(a[0]);
      temp2 = cabs(a[1]);
      temp3 = quadSum(temp1,temp2);
      temp1 = cabs(a[2]);
      return quadSum(temp1,temp3);
    }


  template<class T>
    DEVICEHOST 
    void constructHHMat(const T & tau, const Array<T,3> & v, Matrix<T,3> & hh)
    {
      Matrix<T,3> temp1, temp2;
      outerProd(v,v,&temp1);

      temp2 = conj(tau)*temp1;    

      setIdentity(&temp1);

      hh =  temp1 - temp2; 
      return;
    }


  template<class Real>
    DEVICEHOST
    void getLambdaMax(const Matrix<Real,3> & b, Real & lambda_max){

      Real m11 = b(1,1)*b(1,1) + b(0,1)*b(0,1);
      Real m22 = b(2,2)*b(2,2) + b(1,2)*b(1,2);
      Real m12 = b(1,1)*b(1,2);
      Real dm  = (m11 - m22)/2.0;

      Real norm1 = quadSum(dm, m12);
      if( dm >= 0.0 ){ 
        lambda_max = m22 - (m12*m12)/(norm1 + dm);
      }else{
        lambda_max = m22 + (m12*m12)/(norm1 - dm);
      }
      return;
    }


  template<class Real>
    DEVICEHOST
    void getGivensRotation(const Real & alpha, const Real & beta, Real & c, Real & s)
    {
      Real ratio;
      if( beta == 0.0 ){
        c = 1.0; s = 0.0;
      }else if( fabs(beta) > fabs(alpha) ){
        ratio = -alpha/beta;
        s = rsqrt(1.0 + ratio*ratio);
        c = ratio*s;
      }else{
        ratio = -beta/alpha;
        c = rsqrt(1.0 + ratio*ratio);
        s = ratio*c; 
      }
      return;
    }

  template<class Real>
    inline DEVICEHOST
    void accumGivensRotation(int index, const Real & c, const Real & s, Matrix<Real,3> & m){
      int index_p1 = index+1;
      Real alpha, beta;
      for(int i=0; i<3; ++i){
        alpha = m(i,index);
        beta  = m(i,index_p1);
        m(i,index) = alpha*c - beta*s;
        m(i,index_p1) = alpha*s + beta*c;
      }
      return;
    }

  template<class Real>
    inline DEVICEHOST
    void assignGivensRotation(const Real & c, const Real & s, Matrix<Real,2> & m){
      m(0,0) = c;
      m(0,1) = s;
      m(1,0) = -s;
      m(1,1) =  c;
      return;
    }

  template<class Real>
    inline DEVICEHOST
    void swap(Real & a, Real & b){
      Real temp = a;
      a = b;
      b = temp;
      return;
    }

  template<class Real>
    inline DEVICEHOST
    void smallSVD(Matrix<Real,2> & u, Matrix<Real,2> & v, Matrix<Real,2> & m){
      // set u and v to the 2x2 identity matrix
      setIdentity(&u);
      setIdentity(&v);

      Real c, s;

      if( m(0,0) == 0.0 ){

        getGivensRotation(m(0,1), m(1,1), c, s);

        m(0,0) = m(0,1)*c - m(1,1)*s;
        m(0,1) = 0.0;
        m(1,1) = 0.0;

        // exchange columns in v 
        v(0,0) = 0.0;
        v(0,1) = 1.0;
        v(1,0) = 1.0;
        v(1,1) = 0.0;

        // u is a Givens rotation 
        assignGivensRotation(c, s, u);

      }else if( m(1,1) == 0.0 ){

        getGivensRotation(m(0,0), m(0,1), c, s);  

        m(0,0) = m(0,0)*c - m(0,1)*s;
        m(0,1) = 0.0;
        m(1,1) = 0.0;

        assignGivensRotation(c,s,v);

      }else if( m(0,1) != 0.0 ){

        // need to calculate (m(1,1)**2 + m(0,1)**2 - m(0,0)**2)/(2*m(0,0)*m(0,1))
        Real abs01 = fabs(m(0,1));
        Real abs11 = fabs(m(1,1));
        Real min, max;
        if( abs01 > abs11 ){ min = abs11; max = abs01; } 
        else { min = abs01; max = abs11; }


        Real ratio = min/max;
        Real alpha = 2.0*log(max) + log(1.0 + ratio*ratio);

        Real abs00 = fabs(m(0,0));
        Real beta = 2.0*log(abs00);

        int sign;
        Real temp;

        if( alpha > beta ){
          sign = 1;
          temp = alpha + log(1.0 - exp(beta-alpha));
        }else{
          sign = -1;	
          temp = beta + log(1.0 - exp(alpha-beta));
        }
        temp -= LOG2 + log(abs00) + log(abs01);
        temp = sign*exp(temp);

        if( m(0,0) < 0.0 ){ temp *= -1.0; }
        if( m(0,1) < 0.0 ){ temp *= -1.0; }

        beta = quadSum(1.0, temp);

        if( temp >= 0.0 ){
          temp = 1.0/(temp + beta);
        }else{
          temp = 1.0/(temp - beta);
        }

        // Calculate beta = sqrt(1 + temp**2)
        beta = quadSum(1.0, temp);

        c = 1.0/beta;
        s = temp*c;

        Matrix<Real,2> p;

        p(0,0) = c*m(0,0) - s*m(0,1);
        p(1,0) =          - s*m(1,1);
        p(0,1) = s*m(0,0) + c*m(0,1);
        p(1,1) = c*m(1,1);

        assignGivensRotation(c, s, v);

        // Make the column with the largest norm the first column
        alpha = quadSum(p(0,0),p(1,0));
        beta  = quadSum(p(0,1),p(1,1));

        if(alpha < beta){
          swap(p(0,0),p(0,1));
          swap(p(1,0),p(1,1));

          swap(v(0,0),v(0,1));
          swap(v(1,0),v(1,1));
        }    

        getGivensRotation(p(0,0), p(1,0), c, s);

        m(0,0) = p(0,0)*c - s*p(1,0);
        m(1,1) = p(0,1)*s + c*p(1,1);
        m(0,1) = 0.0;

        assignGivensRotation(c,s,u);	
      }

      return;
    }

  // Change this so that the bidiagonal terms are not returned 
  // as a complex matrix
  template<class Cmplx>
    DEVICEHOST
    void getRealBidiagMatrix(const Matrix<Cmplx,3> & mat, 
        Matrix<Cmplx,3> & u,
        Matrix<Cmplx,3> & v)
    {
      Matrix<Cmplx,3> p;
      Array<Cmplx,3> vec;
      Matrix<Cmplx,3> temp;


      const Cmplx COMPLEX_UNITY = makeComplex<Cmplx>(1,0);
      const Cmplx COMPLEX_ZERO  = makeComplex<Cmplx>(0,0);
      // Step 1: build the first left reflector v,
      //	      calculate the first left rotation
      //	      apply to the original matrix
      typename RealTypeId<Cmplx>::Type x = cabs(mat(1,0));
      typename RealTypeId<Cmplx>::Type y = cabs(mat(2,0));
      typename RealTypeId<Cmplx>::Type norm1 = quadSum(x,y);
      typename RealTypeId<Cmplx>::Type beta;
      Cmplx w, tau, z;

      if(norm1 == 0 && mat(0,0).y == 0){
        p = mat;
      }else{
        Array<Cmplx,3> temp_vec;
        copyColumn(mat, 0, &temp_vec);

        beta = getNorm(temp_vec); 

        if(mat(0,0).x > 0.0){ beta = -beta; }

        w = mat(0,0) - beta; 
        norm1 = cabs(w);
        w = conj(w)/norm1; 

        // Assign the vector elements
        vec[0] = COMPLEX_UNITY; 
        vec[1] = mat(1,0)*w/norm1;
        vec[2] = mat(2,0)*w/norm1;

        // Now compute tau
        tau.x =  (beta - mat(0,0).x)/beta;
        tau.y =  mat(0,0).y/beta;
        // construct the Householder matrix
        constructHHMat(tau, vec, temp);
        p = conj(temp)*mat;    
        u = temp;
      }

      // Step 2: build the first right reflector
      typename RealTypeId<Cmplx>::Type norm2 = cabs(p(0,2));
      if(norm2 != 0.0 || p(0,1).y != 0.0){
        norm1 = cabs(p(0,1));
        beta  = quadSum(norm1,norm2);
        vec[0] = COMPLEX_ZERO;
        vec[1] = COMPLEX_UNITY;  

        if( p(0,1).x > 0.0 ){ beta = -beta; }
        w = p(0,1)-beta;
        norm1 = cabs(w);
        w = conj(w)/norm1; 
        z = conj(p(0,2))/norm1;
        vec[2] = z*conj(w);

        tau = (beta - p(0,1))/beta;
        // construct the Householder matrix
        constructHHMat(tau, vec, temp);
        p = p*temp;
        v = temp;
      }

      // Step 3: build the second left reflector
      norm2 = cabs(p(2,1));
      if(norm2 != 0.0 || p(1,1).y != 0.0){
        norm1 = cabs(p(1,1));
        beta  = quadSum(norm1,norm2);

        // Set the first two elements of the vector
        vec[0] = COMPLEX_ZERO;
        vec[1] = COMPLEX_UNITY;

        if( p(1,1).x > 0 ){ beta = -beta; }
        w = p(1,1) - beta;
        norm1 = cabs(w);
        w = conj(w)/norm1;
        z = p(2,1)/norm1;
        vec[2] = z*w;

        tau.x  = (beta - p(1,1).x)/beta;
        tau.y  = p(1,1).y/beta;
        // I could very easily change the definition of tau
        // so that we wouldn't need to call conj(temp) below.
        // I would have to be careful to make sure this change 
        // is consistent with the other parts of the code.
        constructHHMat(tau, vec, temp);
        p = conj(temp)*p;
        u = u*temp;
      }

      // Step 4: build the second right reflector
      setIdentity(&temp);
      if( p(1,2).y != 0.0 ){
        beta = cabs(p(1,2));
        if( p(1,2).x > 0.0 ){ beta = -beta; }
        temp(2,2) = conj(p(1,2))/beta;
	p(2,2) = p(2,2)*temp(2,2); // This is the only element of p needed below
        v = v*temp;
      }


      // Step 5: build the third left reflector
      if( p(2,2).y != 0.0 ){
        beta = cabs(p(2,2));
        if( p(2,2).x > 0.0 ){ beta = -beta; }
        temp(2,2) = p(2,2)/beta;
        u = u*temp;
      }
      return;
    }



  template<class Real>
    DEVICEHOST
    void bdSVD(Matrix<Real,3>& u, Matrix<Real,3>& v, Matrix<Real,3>& b, int max_it)
    {

      Real c,s;

      // set u and v matrices equal to the identity
      setIdentity(&u);
      setIdentity(&v);

      Real alpha, beta;

      int it=0;
      do{  

        if( fabs(b(0,1)) < SVDPREC*( fabs(b(0,0)) + fabs(b(1,1)) ) ){ b(0,1) = 0.0; } 
        if( fabs(b(1,2)) < SVDPREC*( fabs(b(0,0)) + fabs(b(2,2)) ) ){ b(1,2) = 0.0; }

        if( b(0,1) != 0.0 && b(1,2) != 0.0 ){
          if( b(0,0) == 0.0 ){

            getGivensRotation(-b(0,1), b(1,1), s, c);

            for(int i=0; i<3; ++i){
              alpha = u(i,0);
              beta = u(i,1);
              u(i,0) = alpha*c - beta*s;
              u(i,1) = alpha*s + beta*c;
            }

            b(1,1) = b(0,1)*s + b(1,1)*c;
            b(0,1) = 0.0;

            b(0,2) = -b(1,2)*s;
            b(1,2) *= c;

            getGivensRotation(-b(0,2), b(2,2), s, c);

            for(int i=0; i<3; ++i){
              alpha = u(i,0);
              beta = u(i,2);
              u(i,0) = alpha*c - beta*s;
              u(i,2) = alpha*s + beta*c;
            }
            b(2,2) = b(0,2)*s + b(2,2)*c;
            b(0,2) = 0.0;

          }else if( b(1,1) == 0.0 ){
            // same block
            getGivensRotation(-b(1,2), b(2,2), s, c);
            for(int i=0; i<3; ++i){
              alpha = u(i,1);
              beta = u(i,2);
              u(i,1) = alpha*c - beta*s;
              u(i,2) = alpha*s + beta*c;
            }
            b(2,2) = b(1,2)*s + b(2,2)*c;
            b(1,2) = 0.0;
            // end block
          }else if( b(2,2) == 0.0 ){

            getGivensRotation(b(1,1), -b(1,2), c, s);
            for(int i=0; i<3; ++i){
              alpha = v(i,1);
              beta = v(i,2);
              v(i,1) = alpha*c + beta*s;
              v(i,2) = -alpha*s + beta*c;
            }
            b(1,1) = b(1,1)*c + b(1,2)*s;
            b(1,2) = 0.0;

            b(0,2) = -b(0,1)*s;
            b(0,1) *= c;

            // apply second rotation, to remove b02
            getGivensRotation(b(0,0), -b(0,2), c, s);
            for(int i=0; i<3; ++i){
              alpha = v(i,0);
              beta = v(i,2);
              v(i,0) = alpha*c + beta*s;
              v(i,2) = -alpha*s + beta*c;
            }
            b(0,0) = b(0,0)*c + b(0,2)*s;
            b(0,2) = 0.0;

          }else{ // Else entering normal QR iteration

            Real lambda_max; 
            getLambdaMax(b, lambda_max); // defined above

            alpha = b(0,0)*b(0,0) - lambda_max;
            beta  = b(0,0)*b(0,1);

            // c*beta + s*alpha = 0
            getGivensRotation(alpha, beta, c, s);
            // Multiply right v matrix 
            accumGivensRotation(0, c, s, v);
            // apply to bidiagonal matrix, this generates b(1,0)
            alpha = b(0,0);
            beta  = b(0,1);  

            b(0,0) = alpha*c - beta*s;
            b(0,1) = alpha*s + beta*c;
            b(1,0) = -b(1,1)*s;
            b(1,1) = b(1,1)*c;

            // Calculate the second Givens rotation (this time on the left)
            getGivensRotation(b(0,0), b(1,0), c, s);

            // Multiply left u matrix
            accumGivensRotation(0, c, s, u);

            b(0,0) = b(0,0)*c - b(1,0)*s;
            alpha  = b(0,1);
            beta   = b(1,1);
            b(0,1) = alpha*c - beta*s;
            b(1,1) = alpha*s + beta*c;
            b(0,2) = -b(1,2)*s;
            b(1,2) = b(1,2)*c;
            // Following from the definition of the Givens rotation, b(1,0) should be equal to zero.
            b(1,0) = 0.0; 

            // Calculate the third Givens rotation (on the right)
            getGivensRotation(b(0,1), b(0,2), c, s);

            // Multiply the right v matrix
            accumGivensRotation(1, c, s, v);

            b(0,1) = b(0,1)*c - b(0,2)*s;
            alpha = b(1,1);
            beta  = b(1,2);

            b(1,1) = alpha*c - beta*s;
            b(1,2) = alpha*s + beta*c;
            b(2,1) = -b(2,2)*s;
            b(2,2) *= c;
            b(0,2) = 0.0;


            // Calculate the fourth Givens rotation (on the left)
            getGivensRotation(b(1,1), b(2,1), c, s);
            // Multiply left u matrix
            accumGivensRotation(1, c, s, u);
            // Eliminate b(2,1)
            b(1,1) = b(1,1)*c - b(2,1)*s;
            alpha = b(1,2);
            beta  = b(2,2);
            b(1,2) = alpha*c - beta*s;
            b(2,2) = alpha*s + beta*c;
            b(2,1) = 0.0;

          } // end of normal QR iteration


        }else if( b(0,1) == 0.0 ){ 
          Matrix<Real,2> m_small, u_small, v_small;

          m_small(0,0) = b(1,1);
          m_small(0,1) = b(1,2);
          m_small(1,0) = b(2,1);
          m_small(1,1) = b(2,2);

          smallSVD(u_small, v_small, m_small); 

          b(1,1) = m_small(0,0);	
          b(1,2) = m_small(0,1);	
          b(2,1) = m_small(1,0);	
          b(2,2) = m_small(1,1);	


          for(int i=0; i<3; ++i){
            alpha = u(i,1);
            beta  = u(i,2);
            u(i,1) = alpha*u_small(0,0) + beta*u_small(1,0);
            u(i,2) = alpha*u_small(0,1) + beta*u_small(1,1);

            alpha = v(i,1);
            beta  = v(i,2);
            v(i,1) = alpha*v_small(0,0) + beta*v_small(1,0);
            v(i,2) = alpha*v_small(0,1) + beta*v_small(1,1);
          }


        }else if( b(1,2) == 0.0 ){
          Matrix<Real,2> m_small, u_small, v_small;

          m_small(0,0) = b(0,0);
          m_small(0,1) = b(0,1);
          m_small(1,0) = b(1,0);
          m_small(1,1) = b(1,1);	

          smallSVD(u_small, v_small, m_small); 

          b(0,0) = m_small(0,0);
          b(0,1) = m_small(0,1);
          b(1,0) = m_small(1,0);
          b(1,1) = m_small(1,1);

          for(int i=0; i<3; ++i){
            alpha = u(i,0);
            beta  = u(i,1);
            u(i,0) = alpha*u_small(0,0) + beta*u_small(1,0);
            u(i,1) = alpha*u_small(0,1) + beta*u_small(1,1);

            alpha = v(i,0);
            beta  = v(i,1);
            v(i,0) = alpha*v_small(0,0) + beta*v_small(1,0);
            v(i,1) = alpha*v_small(0,1) + beta*v_small(1,1);
          }

        } // end if b(1,2) == 0
      } while( (b(0,1) != 0.0 || b(1,2) != 0.0) && it < max_it);


      for(int i=0; i<3; ++i){
        if( b(i,i) < 0.0) {
          b(i,i) *= -1;
          for(int j=0; j<3; ++j){
            v(j,i) *= -1;
          }
        }
      }
      return;
    }



  template<class Cmplx>
    DEVICEHOST
    void computeSVD(const Matrix<Cmplx,3> & m, 
        Matrix<Cmplx,3>&  u,
        Matrix<Cmplx,3>&  v,
        typename RealTypeId<Cmplx>::Type singular_values[3])
    {
        
      getRealBidiagMatrix<Cmplx>(m, u, v);
      Matrix<typename RealTypeId<Cmplx>::Type,3> bd, u_real, v_real;
      // Make real
      for(int i=0; i<3; ++i){
        for(int j=0; j<3; ++j){
          bd(i,j) = (conj(u)*m*(v))(i,j).x;
        }
      }

      bdSVD(u_real, v_real, bd, 40);
      for(int i=0; i<3; ++i){
        singular_values[i] = bd(i,i);
      }

      u = u*u_real;
      v = v*v_real;

      return;
    }

} // end namespace hisq


#endif // _SVD_QUDA_H
