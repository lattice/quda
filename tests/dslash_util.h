#ifndef _DSLASH_UTIL_H
#define _DSLASH_UTIL_H

#include <test_util.h>

template <typename Float>
static inline void sum(Float *dst, Float *a, Float *b, int cnt) {
  for (int i = 0; i < cnt; i++)
    dst[i] = a[i] + b[i];
}

template <typename Float>
static inline void sub(Float *dst, Float *a, Float *b, int cnt) {
  for (int i = 0; i < cnt; i++)
    dst[i] = a[i] - b[i];
}

template <typename Float>
static inline void ax(Float *dst, Float a, Float *x, int cnt) {
  for (int i = 0; i < cnt; i++)
    dst[i] = a * x[i];
}

// performs the operation y[i] = x[i] + a*y[i]
template <typename Float>
static inline void xpay(Float *x, Float a, Float *y, int len) {
  for (int i=0; i<len; i++) y[i] = x[i] + a*y[i];
}
// performs the operation y[i] = a*x[i] - y[i]
template <typename Float>
static inline void axmy(Float *x, Float a, Float *y, int len) {
  for (int i=0; i<len; i++) y[i] = a*x[i] - y[i];
}

template <typename Float>
static double norm2(Float *v, int len) {
  double sum=0.0;
  for (int i=0; i<len; i++) sum += v[i]*v[i];
  return sum;
}

template <typename Float>
static inline void negx(Float *x, int len) {
  for (int i=0; i<len; i++) x[i] = -x[i];
}

template <typename sFloat, typename gFloat>
static inline void dot(sFloat* res, gFloat* a, sFloat* b) {
  res[0] = res[1] = 0;
  for (int m = 0; m < 3; m++) {
    sFloat a_re = a[2*m+0];
    sFloat a_im = a[2*m+1];
    sFloat b_re = b[2*m+0];
    sFloat b_im = b[2*m+1];
    res[0] += a_re * b_re - a_im * b_im;
    res[1] += a_re * b_im + a_im * b_re;
  }
}

template <typename Float>
static inline void su3Transpose(Float *res, Float *mat) {
  for (int m = 0; m < 3; m++) {
    for (int n = 0; n < 3; n++) {
      res[m*(3*2) + n*(2) + 0] = + mat[n*(3*2) + m*(2) + 0];
      res[m*(3*2) + n*(2) + 1] = - mat[n*(3*2) + m*(2) + 1];
    }
  }
}


template <typename sFloat, typename gFloat>
static inline void su3Mul(sFloat *res, gFloat *mat, sFloat *vec) {
  for (int n = 0; n < 3; n++) dot(&res[n*(2)], &mat[n*(3*2)], vec);
}

template <typename sFloat, typename gFloat>
static inline void su3Tmul(sFloat *res, gFloat *mat, sFloat *vec) {
  gFloat matT[3*3*2];
  su3Transpose(matT, mat);
  su3Mul(res, matT, vec);
}


// i represents a "half index" into an even or odd "half lattice".
// when oddBit={0,1} the half lattice is {even,odd}.
// 
// the displacements, such as dx, refer to the full lattice coordinates. 
//
// neighborIndex() takes a "half index", displaces it, and returns the
// new "half index", which can be an index into either the even or odd lattices.
// displacements of magnitude one always interchange odd and even lattices.
//


template <typename Float>
static inline Float *gaugeLink(int i, int dir, int oddBit, Float **gaugeEven, Float **gaugeOdd, int nbr_distance) {
  Float **gaugeField;
  int j;
  int d = nbr_distance;
  if (dir % 2 == 0) {
    j = i;
    gaugeField = (oddBit ? gaugeOdd : gaugeEven);
  }
  else {
    switch (dir) {
    case 1: j = neighborIndex(i, oddBit, 0, 0, 0, -d); break;
    case 3: j = neighborIndex(i, oddBit, 0, 0, -d, 0); break;
    case 5: j = neighborIndex(i, oddBit, 0, -d, 0, 0); break;
    case 7: j = neighborIndex(i, oddBit, -d, 0, 0, 0); break;
    default: j = -1; break;
    }
    gaugeField = (oddBit ? gaugeEven : gaugeOdd);
  }
  
  return &gaugeField[dir/2][j*(3*3*2)];
}

template <typename Float>
static inline Float *spinorNeighbor(int i, int dir, int oddBit, Float *spinorField, int neighbor_distance) 
{
  int j;
  int nb = neighbor_distance;
  switch (dir) {
  case 0: j = neighborIndex(i, oddBit, 0, 0, 0, +nb); break;
  case 1: j = neighborIndex(i, oddBit, 0, 0, 0, -nb); break;
  case 2: j = neighborIndex(i, oddBit, 0, 0, +nb, 0); break;
  case 3: j = neighborIndex(i, oddBit, 0, 0, -nb, 0); break;
  case 4: j = neighborIndex(i, oddBit, 0, +nb, 0, 0); break;
  case 5: j = neighborIndex(i, oddBit, 0, -nb, 0, 0); break;
  case 6: j = neighborIndex(i, oddBit, +nb, 0, 0, 0); break;
  case 7: j = neighborIndex(i, oddBit, -nb, 0, 0, 0); break;
  default: j = -1; break;
  }
    
  return &spinorField[j*(mySpinorSiteSize)];
}


#ifdef MULTI_GPU

static inline int
x4_mg(int i, int oddBit)
{
  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  return x4;
}

template <typename Float>
static inline Float *gaugeLink_mg4dir(int i, int dir, int oddBit, Float **gaugeEven, Float **gaugeOdd,
			Float** ghostGaugeEven, Float** ghostGaugeOdd, int n_ghost_faces, int nbr_distance) {
  Float **gaugeField;
  int j;
  int d = nbr_distance;
  if (dir % 2 == 0) {
    j = i;
    gaugeField = (oddBit ? gaugeOdd : gaugeEven);
  }
  else {

    int Y = fullLatticeIndex(i, oddBit);
    int x4 = Y/(Z[2]*Z[1]*Z[0]);
    int x3 = (Y/(Z[1]*Z[0])) % Z[2];
    int x2 = (Y/Z[0]) % Z[1];
    int x1 = Y % Z[0];
    int X1= Z[0];
    int X2= Z[1];
    int X3= Z[2];
    int X4= Z[3];
    Float* ghostGaugeField;

    switch (dir) {
    case 1:
      { //-X direction
        int new_x1 = (x1 - d + X1 )% X1;
        if (x1 -d < 0){
	  ghostGaugeField = (oddBit?ghostGaugeEven[0]: ghostGaugeOdd[0]);
	  int offset = (n_ghost_faces + x1 -d)*X4*X3*X2/2 + (x4*X3*X2 + x3*X2+x2)/2;
	  return &ghostGaugeField[offset*(3*3*2)];
        }
        j = (x4*X3*X2*X1 + x3*X2*X1 + x2*X1 + new_x1) / 2;
        break;
      }
    case 3:
      { //-Y direction
        int new_x2 = (x2 - d + X2 )% X2;
        if (x2 -d < 0){
          ghostGaugeField = (oddBit?ghostGaugeEven[1]: ghostGaugeOdd[1]);
          int offset = (n_ghost_faces + x2 -d)*X4*X3*X1/2 + (x4*X3*X1 + x3*X1+x1)/2;
          return &ghostGaugeField[offset*(3*3*2)];
        }
        j = (x4*X3*X2*X1 + x3*X2*X1 + new_x2*X1 + x1) / 2;
        break;

      }
    case 5:
      { //-Z direction
        int new_x3 = (x3 - d + X3 )% X3;
        if (x3 -d < 0){
          ghostGaugeField = (oddBit?ghostGaugeEven[2]: ghostGaugeOdd[2]);
          int offset = (n_ghost_faces + x3 -d)*X4*X2*X1/2 + (x4*X2*X1 + x2*X1+x1)/2;
          return &ghostGaugeField[offset*(3*3*2)];
        }
        j = (x4*X3*X2*X1 + new_x3*X2*X1 + x2*X1 + x1) / 2;
        break;
      }
    case 7:
      { //-T direction
        int new_x4 = (x4 - d + X4)% X4;
        if (x4 -d < 0){
          ghostGaugeField = (oddBit?ghostGaugeEven[3]: ghostGaugeOdd[3]);
          int offset = (n_ghost_faces + x4 -d)*X1*X2*X3/2 + (x3*X2*X1 + x2*X1+x1)/2;
          return &ghostGaugeField[offset*(3*3*2)];
        }
        j = (new_x4*(X3*X2*X1) + x3*(X2*X1) + x2*(X1) + x1) / 2;
        break;
      }//7

    default: j = -1; printf("ERROR: wrong dir \n"); exit(1);
    }
    gaugeField = (oddBit ? gaugeEven : gaugeOdd);

  }

  return &gaugeField[dir/2][j*(3*3*2)];
}

template <typename Float>
static inline Float *spinorNeighbor_mg4dir(int i, int dir, int oddBit, Float *spinorField, Float** fwd_nbr_spinor, 
					   Float** back_nbr_spinor, int neighbor_distance, int nFace)
{
  int j;
  int nb = neighbor_distance;
  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  int X1= Z[0];
  int X2= Z[1];
  int X3= Z[2];
  int X4= Z[3];

  switch (dir) {
  case 0://+X
    {
      int new_x1 = (x1 + nb)% X1;
      if(x1+nb >=X1){
        int offset = ( x1 + nb -X1)*X4*X3*X2/2+(x4*X3*X2 + x3*X2+x2)/2;
        return fwd_nbr_spinor[0] + offset*mySpinorSiteSize;
      }
      j = (x4*X3*X2*X1 + x3*X2*X1 + x2*X1 + new_x1) / 2;
      break;

    }
  case 1://-X
    {
      int new_x1 = (x1 - nb + X1)% X1;
      if(x1 - nb < 0){ 
        int offset = ( x1+nFace- nb)*X4*X3*X2/2+(x4*X3*X2 + x3*X2+x2)/2;
        return back_nbr_spinor[0] + offset*mySpinorSiteSize;
      } 
      j = (x4*X3*X2*X1 + x3*X2*X1 + x2*X1 + new_x1) / 2;
      break;
    }
  case 2://+Y
    {
      int new_x2 = (x2 + nb)% X2;
      if(x2+nb >=X2){
        int offset = ( x2 + nb -X2)*X4*X3*X1/2+(x4*X3*X1 + x3*X1+x1)/2;
        return fwd_nbr_spinor[1] + offset*mySpinorSiteSize;
      } 
      j = (x4*X3*X2*X1 + x3*X2*X1 + new_x2*X1 + x1) / 2;
      break;
    }
  case 3:// -Y
    {
      int new_x2 = (x2 - nb + X2)% X2;
      if(x2 - nb < 0){ 
        int offset = ( x2 + nFace -nb)*X4*X3*X1/2+(x4*X3*X1 + x3*X1+x1)/2;
        return back_nbr_spinor[1] + offset*mySpinorSiteSize;
      } 
      j = (x4*X3*X2*X1 + x3*X2*X1 + new_x2*X1 + x1) / 2;
      break;
    }
  case 4://+Z
    {
      int new_x3 = (x3 + nb)% X3;
      if(x3+nb >=X3){
        int offset = ( x3 + nb -X3)*X4*X2*X1/2+(x4*X2*X1 + x2*X1+x1)/2;
        return fwd_nbr_spinor[2] + offset*mySpinorSiteSize;
      } 
      j = (x4*X3*X2*X1 + new_x3*X2*X1 + x2*X1 + x1) / 2;
      break;
    }
  case 5://-Z
    {
      int new_x3 = (x3 - nb + X3)% X3;
      if(x3 - nb < 0){ 
        int offset = ( x3 + nFace -nb)*X4*X2*X1/2+(x4*X2*X1 + x2*X1+x1)/2;
        return back_nbr_spinor[2] + offset*mySpinorSiteSize;
      }
      j = (x4*X3*X2*X1 + new_x3*X2*X1 + x2*X1 + x1) / 2;
      break;
    }
  case 6://+T 
    {
      j = neighborIndex_mg(i, oddBit, +nb, 0, 0, 0);
      int x4 = x4_mg(i, oddBit);
      if ( (x4 + nb) >= Z[3]){
        int offset = (x4+nb - Z[3])*Vsh_t;
        return &fwd_nbr_spinor[3][(offset+j)*mySpinorSiteSize];
      }
      break;
    }
  case 7://-T 
    {
      j = neighborIndex_mg(i, oddBit, -nb, 0, 0, 0);
      int x4 = x4_mg(i, oddBit);
      if ( (x4 - nb) < 0){
        int offset = ( x4 - nb +nFace)*Vsh_t;
        return &back_nbr_spinor[3][(offset+j)*mySpinorSiteSize];
      }
      break;
    }
  default: j = -1; printf("ERROR: wrong dir\n"); exit(1);
  }

  return &spinorField[j*(mySpinorSiteSize)];
}

#endif // MULTI_GPU

#endif // _DSLASH_UTIL_H
