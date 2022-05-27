#include <quda_matrix.h>
#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <atomic_helper.h>
#include <random_helper.h>
#include <kernel.h>

namespace quda {

  /**
    @brief Calculate the SU(2) index block in the SU(Nc) matrix
    @param block number to calculate the index's, the total number of blocks is Nc * (Nc - 1) / 2.
    @return Returns two index's in int2 type, accessed by .x and .y.
 */
  template <int nColor>
  __host__ __device__ inline int2 IndexBlock(int block)
  {
    int2 id;
    int i1;
    int found = 0;
    int del_i = 0;
    int index = -1;
    while ( del_i < (nColor - 1) && found == 0 ) {
      del_i++;
      for ( i1 = 0; i1 < (nColor - del_i); i1++ ) {
        index++;
        if ( index == block ) {
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
    @param block number to calculate de index's, the total number of blocks is Nc * (Nc - 1) / 2.
    @param p store the first index
    @param q store the second index
 */
  template<int nColor>
  __host__ __device__ inline void IndexBlock(int block, int &p, int &q)
  {
    if ( nColor == 3 ) {
      if ( block == 0 ) { p = 0; q = 1; }
      else if ( block == 1 ) { p = 1; q = 2; }
      else { p = 0; q = 2; }
    } else if ( nColor > 3 ) {
      int i1;
      int found = 0;
      int del_i = 0;
      int index = -1;
      while ( del_i < (nColor - 1) && found == 0 ) {
        del_i++;
        for ( i1 = 0; i1 < (nColor - del_i); i1++ ) {
          index++;
          if ( index == block ) {
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
  __device__ inline Matrix<T,2> generate_su2_matrix_milc(T al, RNGState& localState)
  {
    T xr1 = uniform<T>::rand(localState);
    xr1 = (log((xr1 + static_cast<T>(1.e-10))));
    T xr2 = uniform<T>::rand(localState);
    xr2 = (log((xr2 + static_cast<T>(1.e-10))));
    T xr3 = uniform<T>::rand(localState);
    T xr4 = uniform<T>::rand(localState);
    xr3 = cospi(static_cast<T>(2.0) * xr3);
    T d = -(xr2 + xr1 * xr3 * xr3 ) / al;
    //now  beat each  site into submission
    int nacd = 0;
    if ((1.00 - 0.5 * d) > xr4 * xr4 ) nacd = 1;
    if (nacd == 0 && al > 2.0 ) { //k-p algorithm
#pragma unroll
      for (int k = 0; k < 20; k++) {
        //get four random numbers (add a small increment to prevent taking log(0.)
        xr1 = uniform<T>::rand(localState);
        xr1 = (log((xr1 + 1.e-10)));
        xr2 = uniform<T>::rand(localState);
        xr2 = (log((xr2 + 1.e-10)));
        xr3 = uniform<T>::rand(localState);
        xr4 = uniform<T>::rand(localState);
        xr3 = cospi(static_cast<T>(2.0) * xr3);
        d = -(xr2 + xr1 * xr3 * xr3) / al;
        if ((1.00 - 0.5 * d) > xr4 * xr4 ) break;
      }
    } //endif nacd
    Matrix<T,2> a;
    T r;
    if (nacd == 0 && al <= 2.0 ) { //creutz algorithm
      xr3 = exp(-2.0 * al);
      xr4 = 1.0 - xr3;
#pragma unroll
      for (int k = 0; k < 20; k++) {
        //get two random numbers
        xr1 = uniform<T>::rand(localState);
        xr2 = uniform<T>::rand(localState);
        r = xr3 + xr4 * xr1;
        a(0,0) = 1.00 + log(r) / al;
        if ((1.0 - a(0,0) * a(0,0)) > xr2 * xr2) break;
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
    a(1,1) = (2.0 * uniform<T>::rand(localState) - 1.0) * r;
    //compute a1 and a2
    xr1 = xr3 - a(1,1) * a(1,1);
    xr1 = abs(xr1);
    xr1 = sqrt(xr1);
    //xr2 is a random number between 0 and 2*pi
    xr2 = static_cast<T>(2.0) * uniform<T>::rand(localState);
    T tmp[2];
    sincospi(xr2, &tmp[1], &tmp[0]);
    a(0,1) = xr1 * tmp[0];
    a(1,0) = xr1 * tmp[1];
    return a;
  }

  /**
    @brief Return SU(2) subgroup (4 real numbers) from SU(3) matrix
    @param tmp1 input SU(3) matrix
    @param block to retrieve from 0 to 2.
    @return 4 real numbers
 */
  template < class T>
  __host__ __device__ inline Matrix<T,2> get_block_su2( Matrix<complex<T>,3> tmp1, int block )
  {
    Matrix<T,2> r;
    switch ( block ) {
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
  template <class T, int nColor>
  __host__ __device__ inline Matrix<T,2> get_block_su2( Matrix<complex<T>,nColor> tmp1, int2 id )
  {
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
  template <class T, int nColor>
  __host__ __device__ inline Matrix<complex<T>,nColor> block_su2_to_sun( Matrix<T,2> rr, int2 id )
  {
    Matrix<complex<T>,nColor> tmp1;
    setIdentity(&tmp1);
    tmp1(id.x,id.x) = complex<T>( rr(0,0), rr(1,1) );
    tmp1(id.x,id.y) = complex<T>( rr(1,0), rr(0,1) );
    tmp1(id.y,id.x) = complex<T>(-rr(1,0), rr(0,1) );
    tmp1(id.y,id.y) = complex<T>( rr(0,0),-rr(1,1) );
    return tmp1;
  }

  /**
    @brief Update the SU(Nc) link with the new SU(2) matrix, link <- u * link
    @param u SU(2) matrix represented by four real numbers
    @param link SU(Nc) matrix
    @param id indices
 */
  template <class T, int nColor>
  __host__ __device__ inline void mul_block_sun( Matrix<T,2> u, Matrix<complex<T>,nColor> &link, int2 id )
  {
#pragma unroll
    for (int j = 0; j < nColor; j++) {
      complex<T> tmp = complex<T>( u(0,0), u(1,1) ) * link(id.x, j) + complex<T>( u(1,0), u(0,1) ) * link(id.y, j);
      link(id.y, j) = complex<T>(-u(1,0), u(0,1) ) * link(id.x, j) + complex<T>( u(0,0),-u(1,1) ) * link(id.y, j);
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
  __host__ __device__ inline void block_su2_to_su3( Matrix<Cmplx,3> &U, Cmplx a00, Cmplx a01, Cmplx a10, Cmplx a11, int block )
  {
    Cmplx tmp;
    switch ( block ) {
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
  __host__ __device__ inline Matrix<Float,2> mulsu2UVDagger(Matrix<Float,2> v, Matrix<Float,2> u)
  {
    Matrix<Float,2> b;
    b(0,0) = v(0,0) * u(0,0) + v(0,1) * u(0,1) + v(1,0) * u(1,0) + v(1,1) * u(1,1);
    b(0,1) = v(0,1) * u(0,0) - v(0,0) * u(0,1) + v(1,0) * u(1,1) - v(1,1) * u(1,0);
    b(1,0) = v(1,0) * u(0,0) - v(0,0) * u(1,0) + v(1,1) * u(0,1) - v(0,1) * u(1,1);
    b(1,1) = v(1,1) * u(0,0) - v(0,0) * u(1,1) + v(0,1) * u(1,0) - v(1,0) * u(0,1);
    return b;
  }

  /**
    @brief Link update by pseudo-heatbath
    @param U link to be updated
    @param F staple
    @param localstate CURAND rng state
  */
  template <class Float, int nColor>
  __device__ inline void heatBathSUN( Matrix<complex<Float>,nColor>& U, Matrix<complex<Float>,nColor> F,
                                      RNGState& localState, Float BetaOverNc )
  {
    if (nColor == 3) {
      //////////////////////////////////////////////////////////////////
      /*
         for( int block = 0; block < nColor; block++ ) {
         Matrix<complex<T>,3> tmp1 = U * F;
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

#pragma unroll
      for (int block = 0; block < nColor; block++) {
        int p,q;
        IndexBlock<nColor>(block, p, q);
        complex<Float> a0((Float)0.0, (Float)0.0);
        complex<Float> a1 = a0;
        complex<Float> a2 = a0;
        complex<Float> a3 = a0;

#pragma unroll
        for (int j = 0; j < nColor; j++) {
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
        Float k = sqrt(r(0,0) * r(0,0) + r(0,1) * r(0,1) + r(1,0) * r(1,0) + r(1,1) * r(1,1));;
        Float ap = BetaOverNc * k;
        k = 1.0 / k;
        r *= k;
        Matrix<Float,2> a = generate_su2_matrix_milc<Float>(ap, localState);
        r = mulsu2UVDagger<Float>( a, r);
        ///////////////////////////////////////
        a0 = complex<Float>( r(0,0), r(1,1) );
        a1 = complex<Float>( r(1,0), r(0,1) );
        a2 = complex<Float>(-r(1,0), r(0,1) );
        a3 = complex<Float>( r(0,0),-r(1,1) );
        complex<Float> tmp0;

#pragma unroll
        for (int j = 0; j < nColor; j++) {
          tmp0 = a0 * U(p,j) + a1 * U(q,j);
          U(q,j) = a2 * U(p,j) + a3 * U(q,j);
          U(p,j) = tmp0;
        }
        //FLOP_min = (nColor * 64 + 19 + 28 + 28) * 3 = nColor * 192 + 225
      }
      //////////////////////////////////////////////////////////////////
    } else if ( nColor > 3 ) {
      //////////////////////////////////////////////////////////////////
      //TESTED IN SU(4) SP THIS IS WORST
      Matrix<complex<Float>,nColor> M = U * F;

#pragma unroll
      for (int block = 0; block < nColor * ( nColor - 1) / 2; block++) {
        int2 id = IndexBlock<nColor>( block );
        Matrix<Float,2> r = get_block_su2<Float>(M, id);
        Float k = sqrt(r(0,0) * r(0,0) + r(0,1) * r(0,1) + r(1,0) * r(1,0) + r(1,1) * r(1,1));
        Float ap = BetaOverNc * k;
        k = 1.0 / k;
        r *= k;
        Matrix<Float,2> a = generate_su2_matrix_milc<Float>(ap, localState);
        Matrix<Float,2> rr = mulsu2UVDagger<Float>( a, r);
        ///////////////////////////////////////
        mul_block_sun<Float, nColor>( rr, U, id);
        mul_block_sun<Float, nColor>( rr, M, id);
        ///////////////////////////////////////
      }
      /* / TESTED IN SU(4) SP THIS IS FASTER
         for ( int block = 0; block < nColor * ( nColor - 1) / 2; block++ ) {
         int2 id = IndexBlock<nColor>( block );
         complex a0 = complex::zero();
         complex a1 = complex::zero();
         complex a2 = complex::zero();
         complex a3 = complex::zero();

         for ( int j = 0; j < nColor; j++ ) {
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
         T k = sqrt(r(0,0) * r(0,0) + r(0,1) * r(0,1) + r(1,0) * r(1,0) + r(1,1) * r(1,1));
         T ap = BetaOverNc * k;
         k = (T)1.0 / k;
         r *= k;
         //Matrix<T,2> a = generate_su2_matrix<T4, T>(ap, localState);
         Matrix<T,2> a = generate_su2_matrix_milc<T>(ap, localState);
         r = mulsu2UVDagger<T>( a, r);
         mul_block_sun<T>( r, U, id); */
         /*
           a0 = complex( r(0,0), r(1,1) );
           a1 = complex( r(1,0), r(0,1) );
           a2 = complex(-r(1,0), r(0,1) );
           a3 = complex( r(0,0),-r(1,1) );
           complex tmp0;

           for ( int j = 0; j < nColor; j++ ) {
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
  template <class Float, int nColor>
  __device__ inline void overrelaxationSUN( Matrix<complex<Float>,nColor>& U, Matrix<complex<Float>,nColor> F )
  {
    if (nColor == 3) {
      //////////////////////////////////////////////////////////////////
      /*
         for( int block = 0; block < 3; block++ ) {
         Matrix<complex<T>,3> tmp1 = U * F;
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

#pragma unroll
      for (int block = 0; block < 3; block++) {
        int p,q;
        IndexBlock<nColor>(block, p, q);
        complex<Float> a0((Float)0., (Float)0.);
        complex<Float> a1 = a0;
        complex<Float> a2 = a0;
        complex<Float> a3 = a0;

#pragma unroll
        for ( int j = 0; j < nColor; j++ ) {
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
        Float norm = 1.0 / sqrt(r(0,0) * r(0,0) + r(0,1) * r(0,1) + r(1,0) * r(1,0) + r(1,1) * r(1,1));;
        r(0,0) *= norm;
        r(0,1) *= -norm;
        r(1,0) *= -norm;
        r(1,1) *= -norm;


        ///////////////////////////////////////
        a0 = complex<Float>( r(0,0), r(1,1) );
        a1 = complex<Float>( r(1,0), r(0,1) );
        a2 = complex<Float>(-r(1,0), r(0,1) );
        a3 = complex<Float>( r(0,0),-r(1,1) );
        complex<Float> tmp0, tmp1;

#pragma unroll
        for ( int j = 0; j < nColor; j++ ) {
          tmp0 = a0 * U(p,j) + a1 * U(q,j);
          tmp1 = a2 * U(p,j) + a3 * U(q,j);
          U(p,j) = a0 * tmp0 + a1 * tmp1;
          U(q,j) = a2 * tmp0 + a3 * tmp1;
        }
        //FLOP = (nColor * 88 + 17) * 3
      }
      ///////////////////////////////////////////////////////////////////
    }
    else if ( nColor > 3 ) {
      ///////////////////////////////////////////////////////////////////
      Matrix<complex<Float>,nColor> M = U * F;
#pragma unroll
      for ( int block = 0; block < nColor * ( nColor - 1) / 2; block++ ) {
        int2 id = IndexBlock<nColor>( block );
        Matrix<Float,2> r = get_block_su2<Float, nColor>(M, id);
        //normalize and conjugate
        Float norm = 1.0 / sqrt(r(0,0) * r(0,0) + r(0,1) * r(0,1) + r(1,0) * r(1,0) + r(1,1) * r(1,1));;
        r(0,0) *= norm;
        r(0,1) *= -norm;
        r(1,0) *= -norm;
        r(1,1) *= -norm;
        mul_block_sun<Float, nColor>( r, U, id);
        mul_block_sun<Float, nColor>( r, U, id);
        mul_block_sun<Float, nColor>( r, M, id);
        mul_block_sun<Float, nColor>( r, M, id);
        ///////////////////////////////////////
      }
      /*  //TESTED IN SU(4) SP THIS IS WORST
          for( int block = 0; block < nColor * ( nColor - 1) / 2; block++ ) {
         int2 id = IndexBlock<nColor>( block );
          complex a0 = complex::zero();
          complex a1 = complex::zero();
          complex a2 = complex::zero();
          complex a3 = complex::zero();

          for(int j = 0; j < nColor; j++){
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

          for(int j = 0; j < nColor; j++){
          tmp0 = a0 * U(id.x, j) + a1 * U(id.y, j);
          tmp1 = a2 * U(id.x, j) + a3 * U(id.y, j);
          U(id.x, j) = a0 * tmp0 + a1 * tmp1;
          U(id.y, j) = a2 * tmp0 + a3 * tmp1;
          }
          }
       */
    }
  }

  template <typename Float_, int nColor_, QudaReconstructType recon, bool heatbath_>
  struct MonteArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    using Gauge = typename gauge_mapper<Float, recon>::type;
    static constexpr bool heatbath = heatbath_;

    int X[4];       // grid dimensions
    int border[4];
    Gauge dataOr;
    Float BetaOverNc;
    RNGState *rng;
    int mu;
    int parity;
    MonteArg(GaugeField &data, Float Beta, RNGState *rng, int mu, int parity) :
      kernel_param(dim3(data.LocalVolumeCB(), 1, 1)),
      dataOr(data),
      rng(rng),
      mu(mu),
      parity(parity)
    {
      BetaOverNc = Beta / (Float)nColor;
      for (int dir = 0; dir < 4; dir++) {
        border[dir] = data.R()[dir];
        X[dir] = data.X()[dir] - border[dir] * 2;
      } 
    }
  };

  template <typename Arg> struct HB
  {
    const Arg &arg;
    constexpr HB(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb)
    {
      using Link = Matrix<complex<typename Arg::Float>, Arg::nColor>;
      auto mu = arg.mu;
      auto parity = arg.parity;

      int X[4];
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];

      int x[4];
      getCoords(x, x_cb, X, parity);
#pragma unroll
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }
      int e_cb = linkIndex(x, X);

      Link staple;
      setZero(&staple);

      Link U;
#pragma unroll
      for (int nu = 0; nu < 4; nu++) if (mu != nu) {
          int dx[4] = { 0, 0, 0, 0 };
          Link link = arg.dataOr(nu, e_cb, parity);
          dx[nu]++;
          U = arg.dataOr(mu, linkIndexShift(x,dx,X), 1 - parity);
          link *= U;
          dx[nu]--;
          dx[mu]++;
          U = arg.dataOr(nu, linkIndexShift(x,dx,X), 1 - parity);
          link *= conj(U);
          staple += link;
          dx[mu]--;
          dx[nu]--;
          link = arg.dataOr(nu, linkIndexShift(x,dx,X), 1 - parity);
          U = arg.dataOr(mu, linkIndexShift(x,dx,X), 1 - parity);
          link = conj(link) * U;
          dx[mu]++;
          U = arg.dataOr(nu, linkIndexShift(x,dx,X), parity);
          link *= U;
          staple += link;
        }
      U = arg.dataOr(mu, e_cb, parity);
      if (Arg::heatbath) {
        RNGState localState = arg.rng[x_cb];
        heatBathSUN( U, conj(staple), localState, arg.BetaOverNc );
        arg.rng[x_cb] = localState;
      } else {
        overrelaxationSUN( U, conj(staple) );
      }
      arg.dataOr(mu, e_cb, parity) = U;
    }
  };

}
