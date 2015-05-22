#include <quda_internal.h>
#include <quda_matrix.h>
#include <tune_quda.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <cub/cub.cuh> 
#include <launch_kernel.cuh>

#include <device_functions.h>


#include <comm_quda.h>


#include <pgauge_monte.h> 
#include <gauge_tools.h>


#include <random.h>


#define BORDER_RADIUS 2

#ifndef PI
#define PI    3.1415926535897932384626433832795    // pi
#endif
#ifndef PII
#define PII   6.2831853071795864769252867665590    // 2 * pi
#endif

namespace quda {






static  __inline__ __device__ double atomicAdd(double *addr, double val){
  double old=*addr, assumed;
  do {
    assumed = old;
    old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
            __double_as_longlong(assumed),
            __double_as_longlong(val+assumed)));
  } while( __double_as_longlong(assumed)!=__double_as_longlong(old) );
  
  return old;
}

static  __inline__ __device__ double2 atomicAdd(double2 *addr, double2 val){
    double2 old=*addr;
    old.x = atomicAdd((double*)addr, val.x);
    old.y = atomicAdd((double*)addr+1, val.y);
    return old;
  }


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200
//CUDA 6.5 NOT DETECTING ATOMICADD FOR FLOAT TYPE!!!!!!!
static __inline__ __device__ float atomicAdd(float *address, float val)
{
  return __fAtomicAdd(address, val);
}
#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 200 */



template <typename T>
struct Summ {
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b){
        return a + b;
    }
};
template <>
struct Summ<double2>{
    __host__ __device__ __forceinline__ double2 operator()(const double2 &a, const double2 &b){
        return make_double2(a.x+b.x, a.y+b.y);
    }
};




/**
    @brief Calculate the SU(2) index block in the SU(Nc) matrix
    @param block number to calculate the index's, the total number of blocks is NCOLORS * ( NCOLORS - 1) / 2.
    @return Returns two index's in int2 type, accessed by .x and .y.
*/
template<int NCOLORS>
__host__ __device__ static inline   int2 IndexBlock(int block){
    int2 id;
    int i1;
    int found = 0;
    int del_i = 0;
    int index = -1;
    while ( del_i < (NCOLORS-1) && found == 0 ){
        del_i++;
        for ( i1 = 0; i1 < (NCOLORS-del_i); i1++ ){
            index++;
            if ( index == block ){
                found = 1;
                break;
            }
        }
    }
    id.y = i1 + del_i;
    id.x = i1;
    return id;
}
/**
    @brief Calculate the SU(2) index block in the SU(Nc) matrix
    @param block number to calculate de index's, the total number of blocks is NCOLORS * ( NCOLORS - 1) / 2.
    @param p store the first index
    @param q store the second index
*/
template<int NCOLORS>
__host__ __device__ static inline void   IndexBlock(int block, int &p, int &q){
  if (NCOLORS == 3){
    if(block == 0){p=0;q=1;}
    else if(block == 1){p=1;q=2;}
    else{p=0;q=2;}
  }
  else if(NCOLORS>3){
      int i1;
      int found = 0;
      int del_i = 0;
      int index = -1;
      while ( del_i < (NCOLORS-1) && found == 0 ){
          del_i++;
          for ( i1 = 0; i1 < (NCOLORS-del_i); i1++ ){
              index++;
              if ( index == block ){
                  found = 1;
                  break;
              }
          }
      }
      q = i1 + del_i;
      p = i1;
  }
}

/**
    @brief Generate full SU(2) matrix (four real numbers instead of 2x2 complex matrix) and update link matrix.
    Get from MILC code.
    @param al weight
    @param localstate CURAND rng state
*/
template <class T>
__device__ static inline Matrix<T,2> generate_su2_matrix_milc(T al, cuRNGState& localState){
    T xr1, xr2, xr3, xr4, d, r;
    int k;
    xr1 = Random<T>(localState);
    xr1 = (log((xr1 + 1.e-10)));
    xr2 = Random<T>(localState);
    xr2 = (log((xr2 + 1.e-10)));
    xr3 = Random<T>(localState);
    xr4 = Random<T>(localState);
    xr3 = cos(PII*xr3);
    d = -(xr2  + xr1 * xr3 * xr3 ) / al;
    //now  beat each  site into submission
    int nacd = 0;
    if ((1.00 - 0.5 * d) > xr4 * xr4) nacd=1;
    if(nacd == 0 && al > 2.0){ //k-p algorithm
        for(k = 0; k < 20; k++){
            //get four random numbers (add a small increment to prevent taking log(0.)
            xr1 = Random<T>(localState);
            xr1 = (log((xr1 + 1.e-10)));
            xr2 = Random<T>(localState);
            xr2 = (log((xr2 + 1.e-10)));
            xr3 = Random<T>(localState);
            xr4 = Random<T>(localState);
            xr3 = cos(PII * xr3);
            d = -(xr2 + xr1 * xr3 * xr3) / al;
            if((1.00 - 0.5 * d) > xr4 * xr4) break;
        }
    } //endif nacd
    Matrix<T,2> a;
    if(nacd == 0 && al <= 2.0){ //creutz algorithm
        xr3 = exp(-2.0 * al);
        xr4 = 1.0 - xr3;
        for(k = 0;k < 20 ; k++){
            //get two random numbers
            xr1 = Random<T>(localState);
            xr2 = Random<T>(localState);
            r = xr3 + xr4 * xr1; 
            a(0,0) = 1.00 + log(r) / al;
            if((1.0 -a(0,0) * a(0,0)) > xr2 * xr2) break;
        }
        d = 1.0 - a(0,0);
    } //endif nacd
    //generate the four su(2) elements 
    //find a0  = 1 - d
    a(0,0) = 1.0 - d;
    //compute r
    xr3 = 1.0 - a(0,0) * a(0,0);
    xr3 = abs(xr3);
    r = sqrt(xr3);
    //compute a3
    a(1,1) = (2.0 * Random<T>(localState) - 1.0) * r;
    //compute a1 and a2
    xr1 = xr3 - a(1,1) * a(1,1);
    xr1 = abs(xr1);
    xr1 = sqrt(xr1);
    //xr2 is a random number between 0 and 2*pi
    xr2 = PII * Random<T>(localState);
    a(0,1) = xr1 * cos(xr2);
    a(1,0) = xr1 * sin(xr2);
    return a;
}


/**
    @brief Return SU(2) subgroup (4 real numbers) from SU(3) matrix
    @param tmp1 input SU(3) matrix
    @param block to retrieve from 0 to 2.
    @return 4 real numbers
*/
template < class T>
__host__ __device__ static inline Matrix<T,2> get_block_su2( Matrix<typename ComplexTypeId<T>::Type,3> tmp1, int block ){
    Matrix<T,2> r;
    switch(block){
  case 0:
      r(0,0) = tmp1(0,0).x + tmp1(1,1).x;
      r(0,1) = tmp1(0,1).y + tmp1(1,0).y;
      r(1,0) = tmp1(0,1).x - tmp1(1,0).x;
      r(1,1) = tmp1(0,0).y - tmp1(1,1).y;
        break;
  case 1:
      r(0,0) = tmp1(1,1).x + tmp1(2,2).x;
      r(0,1) = tmp1(1,2).y + tmp1(2,1).y;
      r(1,0) = tmp1(1,2).x - tmp1(2,1).x;
      r(1,1) = tmp1(1,1).y - tmp1(2,2).y;
        break;
  case 2:
      r(0,0) = tmp1(0,0).x + tmp1(2,2).x;
      r(0,1) = tmp1(0,2).y + tmp1(2,0).y;
      r(1,0) = tmp1(0,2).x - tmp1(2,0).x;
      r(1,1) = tmp1(0,0).y - tmp1(2,2).y;
        break;
    }
    return r;
}

/**
    @brief Return SU(2) subgroup (4 real numbers) from SU(Nc) matrix
    @param tmp1 input SU(Nc) matrix
    @param id the two indices to retrieve SU(2) block
    @return 4 real numbers
*/
template <class T, int NCOLORS>
__host__ __device__ static inline Matrix<T,2> get_block_su2( Matrix<typename ComplexTypeId<T>::Type,NCOLORS> tmp1, int2 id ){
    Matrix<T,2> r;
    r(0,0) = tmp1(id.x,id.x).x + tmp1(id.y,id.y).x;
    r(0,1) = tmp1(id.x,id.y).y + tmp1(id.y,id.x).y;
    r(1,0) = tmp1(id.x,id.y).x - tmp1(id.y,id.x).x;
    r(1,1) = tmp1(id.x,id.x).y - tmp1(id.y,id.y).y;
    return r;
}

/**
    @brief Create a SU(Nc) identity matrix and fills with the SU(2) block
    @param rr SU(2) matrix represented only by four real numbers
    @param id the two indices to fill in the SU(3) matrix
    @return SU(Nc) matrix
*/
template <class T, int NCOLORS>
__host__ __device__ static inline Matrix<typename ComplexTypeId<T>::Type,NCOLORS> block_su2_to_sun( Matrix<T,2> rr, int2 id ){
    Matrix<typename ComplexTypeId<T>::Type,NCOLORS> tmp1;
    setIdentity(&tmp1);
    tmp1(id.x,id.x) = makeComplex( rr(0,0), rr(1,1) );
    tmp1(id.x,id.y) = makeComplex( rr(1,0), rr(0,1) );
    tmp1(id.y,id.x) = makeComplex(-rr(1,0), rr(0,1) );
    tmp1(id.y,id.y) = makeComplex( rr(0,0),-rr(1,1) );
    return tmp1;
}
/**
    @brief Update the SU(Nc) link with the new SU(2) matrix, link <- u * link
    @param u SU(2) matrix represented by four real numbers
    @param link SU(Nc) matrix
    @param id indices
*/
template <class T, int NCOLORS>
__host__ __device__ static inline void mul_block_sun( Matrix<T,2> u, Matrix<typename ComplexTypeId<T>::Type,NCOLORS> &link, int2 id ){
    typename ComplexTypeId<T>::Type tmp;
    for(int j = 0; j < NCOLORS; j++){
        tmp = makeComplex( u(0,0), u(1,1) ) * link(id.x, j) + makeComplex( u(1,0), u(0,1) ) * link(id.y, j);
        link(id.y, j) = makeComplex(-u(1,0), u(0,1) ) * link(id.x, j) + makeComplex( u(0,0),-u(1,1) ) * link(id.y, j);
        link(id.x, j) = tmp;
    }
}

/**
    @brief Update the SU(3) link with the new SU(2) matrix, link <- u * link
    @param U SU(3) matrix
    @param a00 element (0,0) of the SU(2) matrix
    @param a01 element (0,1) of the SU(2) matrix
    @param a10 element (1,0) of the SU(2) matrix
    @param a11 element (1,1) of the SU(2) matrix
    @param block of the SU(3) matrix, 0,1 or 2
*/
template <class Cmplx>
__host__ __device__ static inline void block_su2_to_su3( Matrix<Cmplx,3> &U, Cmplx a00, Cmplx a01, Cmplx a10, Cmplx a11, int block ){
    Cmplx tmp;
    switch(block){
  case 0:
      tmp = a00 * U(0,0) + a01 * U(1,0);
      U(1,0) = a10 * U(0,0) + a11 * U(1,0);
      U(0,0) = tmp;
      tmp = a00 * U(0,1) + a01 * U(1,1);
      U(1,1) = a10 * U(0,1) + a11 * U(1,1);
      U(0,1) = tmp;
      tmp = a00 * U(0,2) + a01 * U(1,2);
      U(1,2) = a10 * U(0,2) + a11 * U(1,2);
      U(0,2) = tmp;
        break;
  case 1:
      tmp = a00 * U(1,0) + a01 * U(2,0);
      U(2,0) = a10 * U(1,0) + a11 * U(2,0);
      U(1,0) = tmp;
      tmp = a00 * U(1,1) + a01 * U(2,1);
      U(2,1) = a10 * U(1,1) + a11 * U(2,1);
      U(1,1) = tmp;
      tmp = a00 * U(1,2) + a01 * U(2,2);
      U(2,2) = a10 * U(1,2) + a11 * U(2,2);
      U(1,2) = tmp;
        break;
  case 2:
      tmp = a00 * U(0,0) + a01 * U(2,0);
      U(2,0) = a10 * U(0,0) + a11 * U(2,0);
      U(0,0) = tmp;
      tmp = a00 * U(0,1) + a01 * U(2,1);
      U(2,1) = a10 * U(0,1) + a11 * U(2,1);
      U(0,1) = tmp;
      tmp = a00 * U(0,2) + a01 * U(2,2);
      U(2,2) = a10 * U(0,2) + a11 * U(2,2);
      U(0,2) = tmp;
        break;
    }
}



// v * u^dagger
template <class Float>
__host__ __device__ static inline Matrix<Float,2> mulsu2UVDagger(Matrix<Float,2> v, Matrix<Float,2> u){
    Matrix<Float,2> b;
    b(0,0) = v(0,0)*u(0,0) + v(0,1)*u(0,1) + v(1,0)*u(1,0) + v(1,1)*u(1,1);
    b(0,1) = v(0,1)*u(0,0) - v(0,0)*u(0,1) + v(1,0)*u(1,1) - v(1,1)*u(1,0);
    b(1,0) = v(1,0)*u(0,0) - v(0,0)*u(1,0) + v(1,1)*u(0,1) - v(0,1)*u(1,1);
    b(1,1) = v(1,1)*u(0,0) - v(0,0)*u(1,1) + v(0,1)*u(1,0) - v(1,0)*u(0,1);
    return b;
}

/**
    @brief Link update by pseudo-heatbath
    @param U link to be updated
    @param F staple
    @param localstate CURAND rng state
*/
template <class Float, int NCOLORS>
__device__ inline void heatBathSUN( Matrix<typename ComplexTypeId<Float>::Type,NCOLORS>& U, Matrix<typename ComplexTypeId<Float>::Type,NCOLORS> F, \
  cuRNGState& localState, Float BetaOverNc ){

    typedef typename ComplexTypeId<Float>::Type Cmplx;
if (NCOLORS == 3){
    //////////////////////////////////////////////////////////////////
    /* 
      for( int block = 0; block < NCOLORS; block++ ) {
      Matrix<typename ComplexTypeId<T>::Type,3> tmp1 = U * F;
      Matrix<T,2> r = get_block_su2<T>(tmp1, block);
      T k = sqrt(r(0,0)*r(0,0)+r(0,1)*r(0,1)+r(1,0)*r(1,0)+r(1,1)*r(1,1));
      T ap = BetaOverNc * k;
      k = (T)1.0 / k;
      r *= k;
      //Matrix<T,2> a = generate_su2_matrix<T4, T>(ap, localState);
      Matrix<T,2> a = generate_su2_matrix_milc<T>(ap, localState);
      r = mulsu2UVDagger_4<T>( a, r);
      ///////////////////////////////////////
      block_su2_to_su3<T>( U, complex( r(0,0), r(1,1) ), complex( r(1,0), r(0,1) ), complex(-r(1,0), r(0,1) ), complex( r(0,0),-r(1,1) ), block ); 
      //FLOP_min = (198 + 4 + 15 + 28 + 28 + 84) * 3 = 1071
      }*/
    //////////////////////////////////////////////////////////////////
     
    for( int block = 0; block < NCOLORS; block++ ) {
        int p,q;
        IndexBlock<NCOLORS>(block, p, q);
        Cmplx a0 = makeComplex((Float)0.0, (Float)0.0);
        Cmplx a1 = a0;
        Cmplx a2 = a0;
        Cmplx a3 = a0;
         
        for(int j = 0; j < NCOLORS; j++){
            a0 += U(p,j) * F(j,p);
            a1 += U(p,j) * F(j,q);
            a2 += U(q,j) * F(j,p);
            a3 += U(q,j) * F(j,q);
        }
        Matrix<Float,2> r;
        r(0,0) = a0.x + a3.x;
        r(0,1) = a1.y + a2.y;
        r(1,0) = a1.x - a2.x;
        r(1,1) = a0.y - a3.y;
        Float k = sqrt(r(0,0)*r(0,0)+r(0,1)*r(0,1)+r(1,0)*r(1,0)+r(1,1)*r(1,1));;
        Float ap = BetaOverNc * k;
        k = 1.0 / k;
        r *= k;
        Matrix<Float,2> a = generate_su2_matrix_milc<Float>(ap, localState);
        r = mulsu2UVDagger<Float>( a, r);
        ///////////////////////////////////////
        a0 = makeComplex( r(0,0), r(1,1) );
        a1 = makeComplex( r(1,0), r(0,1) );
        a2 = makeComplex(-r(1,0), r(0,1) );
        a3 = makeComplex( r(0,0),-r(1,1) );
        Cmplx tmp0;
         
        for(int j = 0; j < NCOLORS; j++){
        tmp0 = a0 * U(p,j) + a1 * U(q,j);
        U(q,j) = a2 * U(p,j) + a3 * U(q,j);
        U(p,j) = tmp0;
        }   
        //FLOP_min = (NCOLORS * 64 + 19 + 28 + 28) * 3 = NCOLORS * 192 + 225
    }
    //////////////////////////////////////////////////////////////////
}
else if(NCOLORS>3){
    //////////////////////////////////////////////////////////////////
    //TESTED IN SU(4) SP THIS IS WORST
      Matrix<typename ComplexTypeId<Float>::Type,NCOLORS> M = U * F;
      for( int block = 0; block < NCOLORS * ( NCOLORS - 1) / 2; block++ ) {
      int2 id = IndexBlock<NCOLORS>( block );
      Matrix<Float,2> r = get_block_su2<Float>(M, id);  
      Float k = sqrt(r(0,0)*r(0,0)+r(0,1)*r(0,1)+r(1,0)*r(1,0)+r(1,1)*r(1,1));
      Float ap = BetaOverNc * k;
      k = 1.0 / k;
      r *= k;
      Matrix<Float,2> a = generate_su2_matrix_milc<Float>(ap, localState);
      Matrix<Float,2> rr = mulsu2UVDagger<Float>( a, r);
      ///////////////////////////////////////   
      mul_block_sun<Float, NCOLORS>( rr, U, id);
      mul_block_sun<Float, NCOLORS>( rr, M, id);
      ///////////////////////////////////////     
      }
    /*//TESTED IN SU(4) SP THIS IS FASTER
    for( int block = 0; block < NCOLORS * ( NCOLORS - 1) / 2; block++ ) {
      int2 id = IndexBlock<NCOLORS>( block );
        complex a0 = complex::zero();
        complex a1 = complex::zero();
        complex a2 = complex::zero();
        complex a3 = complex::zero();
         
        for(int j = 0; j < NCOLORS; j++){
            a0 += U(id.x, j) * F.e[j][id.x];
            a1 += U(id.x, j) * F.e[j][id.y];
            a2 += U(id.y, j) * F.e[j][id.x];
            a3 += U(id.y, j) * F.e[j][id.y];
        }
        Matrix<T,2> r;
        r(0,0) = a0.x + a3.x;
        r(0,1) = a1.y + a2.y;
        r(1,0) = a1.x - a2.x;
        r(1,1) = a0.y - a3.y;
        T k = sqrt(r(0,0)*r(0,0)+r(0,1)*r(0,1)+r(1,0)*r(1,0)+r(1,1)*r(1,1));
        T ap = BetaOverNc * k;
        k = (T)1.0 / k;
        r *= k;
        //Matrix<T,2> a = generate_su2_matrix<T4, T>(ap, localState);
        Matrix<T,2> a = generate_su2_matrix_milc<T>(ap, localState);
        r = mulsu2UVDagger<T>( a, r);
        mul_block_sun<T>( r, U, id);*/
        /*///////////////////////////////////////
          a0 = complex( r(0,0), r(1,1) );
          a1 = complex( r(1,0), r(0,1) );
          a2 = complex(-r(1,0), r(0,1) );
          a3 = complex( r(0,0),-r(1,1) );
          complex tmp0;
           
          for(int j = 0; j < NCOLORS; j++){
          tmp0 = a0 * U(id.x, j) + a1 * U(id.y, j);
          U(id.y, j) = a2 * U(id.x, j) + a3 * U(id.y, j);
          U(id.x, j) = tmp0;
          } */  
   // }

}
    //////////////////////////////////////////////////////////////////
}

//////////////////////////////////////////////////////////////////////////
/**
    @brief Link update by overrelaxation
    @param U link to be updated
    @param F staple
*/
template <class Float, int NCOLORS>
__device__ inline void overrelaxationSUN( Matrix<typename ComplexTypeId<Float>::Type,NCOLORS>& U, Matrix<typename ComplexTypeId<Float>::Type,NCOLORS> F ){

    typedef typename ComplexTypeId<Float>::Type Cmplx;
if (NCOLORS == 3){
    //////////////////////////////////////////////////////////////////
    /* 
      for( int block = 0; block < 3; block++ ) {
      Matrix<typename ComplexTypeId<T>::Type,3> tmp1 = U * F;
      Matrix<T,2> r = get_block_su2<T>(tmp1, block);
      //normalize and conjugate
      Float norm = 1.0 / sqrt(r(0,0)*r(0,0)+r(0,1)*r(0,1)+r(1,0)*r(1,0)+r(1,1)*r(1,1));;
      r(0,0) *= norm;
      r(0,1) *= -norm;
      r(1,0) *= -norm;
      r(1,1) *= -norm;
      ///////////////////////////////////////
      complex a00 = complex( r(0,0), r(1,1) );
      complex a01 = complex( r(1,0), r(0,1) );
      complex a10 = complex(-r(1,0), r(0,1) );
      complex a11 = complex( r(0,0),-r(1,1) );
      block_su2_to_su3<T>( U, a00, a01, a10, a11, block );
      block_su2_to_su3<T>( U, a00, a01, a10, a11, block );

      //FLOP = (198 + 17 + 84 * 2) * 3 = 1149
      }*/
    ///////////////////////////////////////////////////////////////////
    //This version does not need to multiply all matrix at each block: tmp1 = U * F;
    //////////////////////////////////////////////////////////////////
     
    for( int block = 0; block < 3; block++ ) {
        int p,q;
        IndexBlock<NCOLORS>(block, p, q);
        Cmplx a0 = makeComplex((Float)0., (Float)0.);
        Cmplx a1 = a0;
        Cmplx a2 = a0;
        Cmplx a3 = a0;
         
        for(int j = 0; j < NCOLORS; j++){
            a0 += U(p,j) * F(j,p);
            a1 += U(p,j) * F(j,q);
            a2 += U(q,j) * F(j,p);
            a3 += U(q,j) * F(j,q);
        }
        Matrix<Float,2> r;
        r(0,0) = a0.x + a3.x;
        r(0,1) = a1.y + a2.y;
        r(1,0) = a1.x - a2.x;
        r(1,1) = a0.y - a3.y;
        //normalize and conjugate
        //r = r.conj_normalize();
        Float norm = 1.0 / sqrt(r(0,0)*r(0,0)+r(0,1)*r(0,1)+r(1,0)*r(1,0)+r(1,1)*r(1,1));;
        r(0,0) *= norm;
        r(0,1) *= -norm;
        r(1,0) *= -norm;
        r(1,1) *= -norm;


        ///////////////////////////////////////
        a0 = makeComplex( r(0,0), r(1,1) );
        a1 = makeComplex( r(1,0), r(0,1) );
        a2 = makeComplex(-r(1,0), r(0,1) );
        a3 = makeComplex( r(0,0),-r(1,1) );
        Cmplx tmp0, tmp1;
         
        for(int j = 0; j < NCOLORS; j++){
        tmp0 = a0 * U(p,j) + a1 * U(q,j);
        tmp1 = a2 * U(p,j) + a3 * U(q,j);
        U(p,j) = a0 * tmp0 + a1 * tmp1;
        U(q,j) = a2 * tmp0 + a3 * tmp1;
        }
        //FLOP = (NCOLORS * 88 + 17) * 3
    }
    ///////////////////////////////////////////////////////////////////
}
else if(NCOLORS>3){
    ///////////////////////////////////////////////////////////////////
    Matrix<typename ComplexTypeId<Float>::Type,NCOLORS> M = U * F;
    for( int block = 0; block < NCOLORS * ( NCOLORS - 1) / 2; block++ ) {
        int2 id = IndexBlock<NCOLORS>( block );
        Matrix<Float,2> r = get_block_su2<Float, NCOLORS>(M, id);
        //normalize and conjugate
        Float norm = 1.0 / sqrt(r(0,0)*r(0,0)+r(0,1)*r(0,1)+r(1,0)*r(1,0)+r(1,1)*r(1,1));;
        r(0,0) *= norm;
        r(0,1) *= -norm;
        r(1,0) *= -norm;
        r(1,1) *= -norm;  
        mul_block_sun<Float, NCOLORS>( r, U, id);
        mul_block_sun<Float, NCOLORS>( r, U, id);
        mul_block_sun<Float, NCOLORS>( r, M, id);
        mul_block_sun<Float, NCOLORS>( r, M, id);
        ///////////////////////////////////////
    }
    /*  //TESTED IN SU(4) SP THIS IS WORST
        for( int block = 0; block < NCOLORS * ( NCOLORS - 1) / 2; block++ ) {
      int2 id = IndexBlock<NCOLORS>( block );
        complex a0 = complex::zero();
        complex a1 = complex::zero();
        complex a2 = complex::zero();
        complex a3 = complex::zero();
         
        for(int j = 0; j < NCOLORS; j++){
    a0 += U(id.x, j) * F.e[j][id.x];
    a1 += U(id.x, j) * F.e[j][id.y];
    a2 += U(id.y, j) * F.e[j][id.x];
    a3 += U(id.y, j) * F.e[j][id.y];
        }
        Matrix<T,2> r;
        r(0,0) = a0.x + a3.x;
        r(0,1) = a1.y + a2.y;
        r(1,0) = a1.x - a2.x;
        r(1,1) = a0.y - a3.y;
        //normalize and conjugate
        Float norm = 1.0 / sqrt(r(0,0)*r(0,0)+r(0,1)*r(0,1)+r(1,0)*r(1,0)+r(1,1)*r(1,1));;
        r(0,0) *= norm;
        r(0,1) *= -norm;
        r(1,0) *= -norm;
        r(1,1) *= -norm;
        //mul_block_sun<T>( r, U, id);
        //mul_block_sun<T>( r, U, id);
        ///////////////////////////////////////
        a0 = complex( r(0,0), r(1,1) );
        a1 = complex( r(1,0), r(0,1) );
        a2 = complex(-r(1,0), r(0,1) );
        a3 = complex( r(0,0),-r(1,1) );
        complex tmp0, tmp1;
         
        for(int j = 0; j < NCOLORS; j++){
        tmp0 = a0 * U(id.x, j) + a1 * U(id.y, j);
        tmp1 = a2 * U(id.x, j) + a3 * U(id.y, j);
        U(id.x, j) = a0 * tmp0 + a1 * tmp1;
        U(id.y, j) = a2 * tmp0 + a3 * tmp1;
        }
        }
    */
}
}








__device__ __host__ inline int linkIndex2(int x[], int dx[], const int X[4]) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
  int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
  return idx;
}



static __device__ __host__ inline int linkIndex3(int x[], int dx[], const int X[4]) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = (x[i] + dx[i] + X[i]) % X[i];
  int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
  return idx;
}
static __device__ __host__ inline int linkIndex(int x[], const int X[4]) {
  int idx = (((x[3]*X[2] + x[2])*X[1] + x[1])*X[0] + x[0]) >> 1;
  return idx;
}
static __device__ __host__ inline int linkIndexM1(int x[], const int X[4], const int mu) {
  int y[4];
  for (int i=0; i<4; i++) y[i] = x[i];
  y[mu] = (y[mu] -1 + X[mu]) % X[mu];
  int idx = (((y[3]*X[2] + y[2])*X[1] + y[1])*X[0] + y[0]) >> 1;
  return idx;
}



static __device__ __host__ inline void getCoords3(int x[4], int cb_index, const int X[4], int parity) {
  /*x[3] = cb_index/(X[2]*X[1]*X[0]/2);
  x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
  x[1] = (cb_index/(X[0]/2)) % X[1];
  x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);*/
  int za = (cb_index / (X[0]/2));
  int zb =  (za / X[1]);
  x[1] = za - zb * X[1];
  x[3] = (zb / X[2]);
  x[2] = zb - x[3] * X[2];
  int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
  x[0] = (2 * cb_index + x1odd)  - za * X[0];
  return;
}









template <typename Gauge, typename Float, int NCOLORS>
struct MonteArg {
  int threads; // number of active threads required
  int X[4]; // grid dimensions
#ifdef MULTI_GPU
  int border[4]; 
#endif
  Gauge dataOr;
  cudaGaugeField &data;
  Float BetaOverNc;
  cuRNGState *rngstate;
  MonteArg(const Gauge &dataOr, cudaGaugeField &data, Float Beta, cuRNGState *rngstate)
    : dataOr(dataOr), data(data), rngstate(rngstate) {
    BetaOverNc = Beta / NCOLORS;
#ifdef MULTI_GPU
    for(int dir=0; dir<4; ++dir){
      if(comm_dim_partitioned(dir)) border[dir] = BORDER_RADIUS;
      else border[dir] = 0;
    }
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir] - border[dir]*2;
#else
    for(int dir=0; dir<4; ++dir) X[dir] = data.X()[dir];
#endif
    threads = X[0]*X[1]*X[2]*X[3] >> 1;
  }
};


template<typename Float, typename Gauge, int NCOLORS, bool HeatbathOrRelax>
__global__ void compute_heatBath(MonteArg<Gauge, Float, NCOLORS> arg, int mu, int parity){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx >= arg.threads) return;
    typedef typename ComplexTypeId<Float>::Type Cmplx;
    int id = idx;
    int X[4]; 
    #pragma unroll
    for(int dr=0; dr<4; ++dr) X[dr] = arg.X[dr];

    int x[4];
    getCoords3(x, idx, X, parity);
#ifdef MULTI_GPU
    #pragma unroll
    for(int dr=0; dr<4; ++dr) {
         x[dr] += arg.border[dr];
         X[dr] += 2*arg.border[dr];
    }
    idx = linkIndex(x,X);
#endif

    Matrix<Cmplx,NCOLORS> staple;
    setZero(&staple);

    Matrix<Cmplx,NCOLORS> U; 
    for(int nu = 0; nu < 4; nu++)  if(mu != nu) {
      int dx[4] = {0, 0, 0, 0};
      Matrix<Cmplx,NCOLORS> link; 
      arg.dataOr.load((Float*)(link.data), idx, nu, parity);
      dx[nu]++;
      arg.dataOr.load((Float*)(U.data), linkIndex2(x,dx,X), mu, 1-parity);
      link *= U;
      dx[nu]--;
      dx[mu]++;
      arg.dataOr.load((Float*)(U.data), linkIndex2(x,dx,X), nu, 1-parity);
      link *= conj(U);
      staple += link;
      dx[mu]--;
      dx[nu]--;
      arg.dataOr.load((Float*)(link.data), linkIndex2(x,dx,X), nu, 1-parity);
      arg.dataOr.load((Float*)(U.data), linkIndex2(x,dx,X), mu, 1-parity);
      link = conj(link) * U;
      dx[mu]++;
      arg.dataOr.load((Float*)(U.data), linkIndex2(x,dx,X), nu, parity);
      link *= U;
      staple += link;
    }
    arg.dataOr.load((Float*)(U.data), idx, mu, parity);
    if(HeatbathOrRelax) {
      cuRNGState localState = arg.rngstate[ id ];
      heatBathSUN<Float, NCOLORS>( U, conj(staple), localState, arg.BetaOverNc );
      arg.rngstate[ id ] = localState;
    }
    else{
      overrelaxationSUN<Float, NCOLORS>( U, conj(staple) );
    }   
    arg.dataOr.save((Float*)(U.data), idx, mu, parity);
}


template<typename Float, typename Gauge, int NCOLORS, int NElems, bool HeatbathOrRelax>
class GaugeHB : Tunable {
  MonteArg<Gauge, Float, NCOLORS> arg;
  int mu;
  int parity;
  mutable char aux_string[128]; // used as a label in the autotuner
  private:
  unsigned int sharedBytesPerThread() const { return 0; }
  unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
  bool tuneSharedBytes() const { return false; } // Don't tune shared memory
  bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
  unsigned int minThreads() const { return arg.threads; }

  public:
  GaugeHB(MonteArg<Gauge, Float, NCOLORS> &arg)
    : arg(arg), mu(0), parity(0) { }
  /*GaugeHB(MonteArg<Gauge, Float, NCOLORS> &arg, int _mu, int _parity)
    : arg(arg) {
    mu = _mu;
    parity = _parity;
  }*/
  ~GaugeHB () { }
  void SetParam(int _mu, int _parity){
    mu = _mu;
    parity = _parity;
  }
  void apply(const cudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      compute_heatBath<Float, Gauge, NCOLORS, HeatbathOrRelax ><<<tp.grid,tp.block, 0, stream>>>(arg, mu, parity);
  }

  TuneKey tuneKey() const {
    std::stringstream vol;
    vol << arg.X[0] << "x";
    vol << arg.X[1] << "x";
    vol << arg.X[2] << "x";
    vol << arg.X[3];
    sprintf(aux_string,"threads=%d,prec=%d",arg.threads, sizeof(Float));
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux_string);
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { arg.data.backup(); }
  void postTune() { arg.data.restore(); }
  long long flops() const { 

    //NEED TO CHECK THIS!!!!!!
    if(NCOLORS == 3){
      long long flop = 2268LL;
      if(HeatbathOrRelax){
        flop += 801LL;
      }
      else{
        flop += 843LL;
      }
      flop *= arg.threads;
      return flop;
    }
    else{
      long long flop = NCOLORS * NCOLORS * NCOLORS * 84LL;
      if(HeatbathOrRelax){
        flop += NCOLORS * NCOLORS * NCOLORS + (NCOLORS * ( NCOLORS - 1) / 2) * (46LL + 48LL+56LL * NCOLORS);
      }
      else{
        flop += NCOLORS * NCOLORS * NCOLORS + (NCOLORS * ( NCOLORS - 1) / 2) * (17LL+112LL * NCOLORS);
      }
      flop *= arg.threads;
      return flop;
    }
   }
  long long bytes() const { 
    //NEED TO CHECK THIS!!!!!!
    if(NCOLORS == 3){
      long long byte = 20LL * NElems * sizeof(Float) ;
      if(HeatbathOrRelax) byte += 2LL * sizeof(cuRNGState);
      byte *= arg.threads;
      return byte;
    }
    else{
      long long byte = 20LL * NCOLORS * NCOLORS * 2 * sizeof(Float);
      if(HeatbathOrRelax) byte += 2LL * sizeof(cuRNGState);
      byte *= arg.threads;
      return byte;
    }
   }
}; 









template<typename Float, int NElems, int NCOLORS, typename Gauge>
void Monte( Gauge dataOr,  cudaGaugeField& data, cuRNGState *rngstate, Float Beta, unsigned int nhb, unsigned int nover) {

  TimeProfile profileHBOVR("HeatBath_OR_Relax");
  MonteArg<Gauge, Float, NCOLORS> montearg(dataOr, data, Beta, rngstate);
  if (getVerbosity() >= QUDA_SUMMARIZE) profileHBOVR.Start(QUDA_PROFILE_COMPUTE);
  GaugeHB<Float, Gauge, NCOLORS, NElems, true> hb(montearg);
  for(int step=0; step<nhb; ++step){
    for(int parity=0; parity<2; ++parity){
      for(int mu=0; mu<4; ++mu){
        hb.SetParam(mu, parity);
        hb.apply(0);
        #ifdef MULTI_GPU
        PGaugeExchange( data, mu, parity);
        #endif
      }
    }
  }
  if (getVerbosity() >= QUDA_SUMMARIZE){
    cudaDeviceSynchronize();
    profileHBOVR.Stop(QUDA_PROFILE_COMPUTE);
    double secs = profileHBOVR.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (hb.flops() * 8 * nhb * 1e-9)/(secs);
    double gbytes = hb.bytes() * 8 * nhb /(secs*1e9);
    #ifdef MULTI_GPU
    printfQuda("HB: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops*comm_size(), gbytes*comm_size());
    #else
    printfQuda("HB: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops, gbytes);
    #endif
  }

  if (getVerbosity() >= QUDA_SUMMARIZE) profileHBOVR.Start(QUDA_PROFILE_COMPUTE);
  GaugeHB<Float, Gauge, NCOLORS, NElems, false> relax(montearg);
  for(int step=0; step<nover; ++step){
    for(int parity=0; parity<2; ++parity){
      for(int mu=0; mu<4; ++mu){
        relax.SetParam(mu, parity);
        relax.apply(0);
        #ifdef MULTI_GPU
        PGaugeExchange( data, mu, parity);
        #endif
      }
    }
  }
  if (getVerbosity() >= QUDA_SUMMARIZE){
    cudaDeviceSynchronize();
    profileHBOVR.Stop(QUDA_PROFILE_COMPUTE);
    double secs = profileHBOVR.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (relax.flops() * 8 * nover * 1e-9)/(secs);
    double gbytes = relax.bytes() * 8 * nover /(secs*1e9);
    #ifdef MULTI_GPU
    printfQuda("OVR: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops*comm_size(), gbytes*comm_size());
    #else
    printfQuda("OVR: Time = %6.6f s, Gflop/s = %6.1f, GB/s = %6.1f\n", secs, gflops, gbytes);
    #endif
  }
}



template<typename Float>
void Monte( cudaGaugeField& data, cuRNGState *rngstate, Float Beta, unsigned int nhb, unsigned int nover) {

  // Switching to FloatNOrder for the gauge field in order to support RECONSTRUCT_12
  // Need to fix this!!
  if(data.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
      Monte<Float, 18, 3>(FloatNOrder<Float, 18, 2, 18>(data), data, rngstate, Beta, nhb, nover);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      Monte<Float, 12, 3>(FloatNOrder<Float, 18, 2, 12>(data), data, rngstate, Beta, nhb, nover);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      Monte<Float, 8, 3>(FloatNOrder<Float, 18, 2,  8>(data), data, rngstate, Beta, nhb, nover);
    
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else if(data.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
    if(data.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    //printfQuda("QUDA_RECONSTRUCT_NO\n");
      Monte<Float, 18, 3>(FloatNOrder<Float, 18, 4, 18>(data), data, rngstate, Beta, nhb, nover);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_12){
    //printfQuda("QUDA_RECONSTRUCT_12\n");
      Monte<Float, 12, 3>(FloatNOrder<Float, 18, 4, 12>(data), data, rngstate, Beta, nhb, nover);
    } else if(data.Reconstruct() == QUDA_RECONSTRUCT_8){
    //printfQuda("QUDA_RECONSTRUCT_8\n");
      Monte<Float, 8, 3>(FloatNOrder<Float, 18, 4,  8>(data), data, rngstate, Beta, nhb, nover);
    } else {
      errorQuda("Reconstruction type %d of gauge field not supported", data.Reconstruct());
    }
  } else {
    errorQuda("Invalid Gauge Order\n");
  }
}


/** @brief Perform heatbath and overrelaxation. Performs nhb heatbath steps followed by nover overrelaxation steps.
* 
* @param[in,out] data Gauge field
* @param[in,out] rngstate state of the CURAND random number generator
* @param[in] Beta inverse of the gauge coupling, beta = 2 Nc / g_0^2 
* @param[in] nhb number of heatbath steps
* @param[in] nover number of overrelaxation steps
*/
void Monte( cudaGaugeField& data, cuRNGState *rngstate, double Beta, unsigned int nhb, unsigned int nover) {
  if(data.Precision() == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported\n");
  }
  if (data.Precision() == QUDA_SINGLE_PRECISION) {
    Monte<float> (data, rngstate, (float)Beta, nhb, nover);
  } else if(data.Precision() == QUDA_DOUBLE_PRECISION) {
    Monte<double>(data, rngstate, Beta, nhb, nover);
  } else {
    errorQuda("Precision %d not supported", data.Precision());
  }
}


}
