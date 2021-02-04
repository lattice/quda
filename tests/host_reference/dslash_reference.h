#pragma once

#include <host_utils.h>
#include <comm_quda.h>

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

// performs the operation y[i] = a*x[i] + y[i]
template <typename Float>
static inline void axpy(Float a, Float *x, Float *y, int len) {
  for (int i=0; i<len; i++) y[i] = a*x[i] + y[i];
}

// performs the operation y[i] = a*x[i] + b*y[i]
template <typename Float>
static inline void axpby(Float a, Float *x, Float b, Float *y, int len) {
  for (int i=0; i<len; i++) y[i] = a*x[i] + b*y[i];
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
void verifyInversion(void *spinorOut, void *spinorIn, void *spinorCheck, QudaGaugeParam &gauge_param,
                     QudaInvertParam &inv_param, void **gauge, void *clover, void *clover_inv);

void verifyInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn, void *spinorCheck,
                     QudaGaugeParam &gauge_param, QudaInvertParam &inv_param, void **gauge, void *clover,
                     void *clover_inv);

void verifyDomainWallTypeInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn, void *spinorCheck,
                                   QudaGaugeParam &gauge_param, QudaInvertParam &inv_param, void **gauge, void *clover,
                                   void *clover_inv);

void verifyWilsonTypeInversion(void *spinorOut, void **spinorOutMulti, void *spinorIn, void *spinorCheck,
                               QudaGaugeParam &gauge_param, QudaInvertParam &inv_param, void **gauge, void *clover,
                               void *clover_inv);

void verifyStaggeredInversion(quda::ColorSpinorField *tmp, quda::ColorSpinorField *ref, quda::ColorSpinorField *in,
                              quda::ColorSpinorField *out, double mass, void *qdp_fatlink[], void *qdp_longlink[],
                              void **ghost_fatlink, void **ghost_longlink, QudaGaugeParam &gauge_param,
                              QudaInvertParam &inv_param, int shift);

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

  return &spinorField[j * (my_spinor_site_size)];
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
//
template <QudaPCType type> int neighborIndex_5d(int i, int oddBit, int dxs, int dx4, int dx3, int dx2, int dx1)
{
  // fullLatticeIndex was modified for fullLatticeIndex_4d.  It is in util_quda.cpp.
  // This code bit may not properly perform 5dPC.
  int X = type == QUDA_5D_PC ? fullLatticeIndex_5d(i, oddBit) : fullLatticeIndex_5d_4dpc(i, oddBit);
  // Checked that this matches code in dslash_core_ante.h.
  int xs = X/(Z[3]*Z[2]*Z[1]*Z[0]);
  int x4 = (X/(Z[2]*Z[1]*Z[0])) % Z[3];
  int x3 = (X/(Z[1]*Z[0])) % Z[2];
  int x2 = (X/Z[0]) % Z[1];
  int x1 = X % Z[0];
  // Displace and project back into domain 0,...,Ls-1.
  // Note that we add Ls to avoid the negative problem
  // of the C % operator.
  xs = (xs+dxs+Ls) % Ls;
  // Etc.
  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];
  // Return linear half index.  Remember that integer division
  // rounds down.
  return (xs*(Z[3]*Z[2]*Z[1]*Z[0]) + x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
}

template <QudaPCType type, typename Float>
Float *spinorNeighbor_5d(int i, int dir, int oddBit, Float *spinorField, int neighbor_distance = 1, int siteSize = 24)
{
  int nb = neighbor_distance;
  int j;
  switch (dir) {
  case 0: j = neighborIndex_5d<type>(i, oddBit, 0, 0, 0, 0, +nb); break;
  case 1: j = neighborIndex_5d<type>(i, oddBit, 0, 0, 0, 0, -nb); break;
  case 2: j = neighborIndex_5d<type>(i, oddBit, 0, 0, 0, +nb, 0); break;
  case 3: j = neighborIndex_5d<type>(i, oddBit, 0, 0, 0, -nb, 0); break;
  case 4: j = neighborIndex_5d<type>(i, oddBit, 0, 0, +nb, 0, 0); break;
  case 5: j = neighborIndex_5d<type>(i, oddBit, 0, 0, -nb, 0, 0); break;
  case 6: j = neighborIndex_5d<type>(i, oddBit, 0, +nb, 0, 0, 0); break;
  case 7: j = neighborIndex_5d<type>(i, oddBit, 0, -nb, 0, 0, 0); break;
  case 8: j = neighborIndex_5d<type>(i, oddBit, +nb, 0, 0, 0, 0); break;
  case 9: j = neighborIndex_5d<type>(i, oddBit, -nb, 0, 0, 0, 0); break;
  default: j = -1; break;
  }
  return &spinorField[j*siteSize];
}

#ifdef MULTI_GPU
inline int x4_mg(int i, int oddBit)
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
        if (x1 -d < 0 && comm_dim_partitioned(0)){
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
        if (x2 -d < 0 && comm_dim_partitioned(1)){
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
        if (x3 -d < 0 && comm_dim_partitioned(2)){
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
        if (x4 -d < 0 && comm_dim_partitioned(3)){
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
      if(x1+nb >=X1 && comm_dim_partitioned(0) ){
        int offset = ( x1 + nb -X1)*X4*X3*X2/2+(x4*X3*X2 + x3*X2+x2)/2;
        return fwd_nbr_spinor[0] + offset * my_spinor_site_size;
      }
      j = (x4*X3*X2*X1 + x3*X2*X1 + x2*X1 + new_x1) / 2;
      break;
    }
  case 1://-X
    {
      int new_x1 = (x1 - nb + X1)% X1;
      if(x1 - nb < 0 && comm_dim_partitioned(0)){
        int offset = ( x1+nFace- nb)*X4*X3*X2/2+(x4*X3*X2 + x3*X2+x2)/2;
        return back_nbr_spinor[0] + offset * my_spinor_site_size;
      } 
      j = (x4*X3*X2*X1 + x3*X2*X1 + x2*X1 + new_x1) / 2;
      break;
    }
  case 2://+Y
    {
      int new_x2 = (x2 + nb)% X2;
      if(x2+nb >=X2 && comm_dim_partitioned(1)){
        int offset = ( x2 + nb -X2)*X4*X3*X1/2+(x4*X3*X1 + x3*X1+x1)/2;
        return fwd_nbr_spinor[1] + offset * my_spinor_site_size;
      } 
      j = (x4*X3*X2*X1 + x3*X2*X1 + new_x2*X1 + x1) / 2;
      break;
    }
  case 3:// -Y
    {
      int new_x2 = (x2 - nb + X2)% X2;
      if(x2 - nb < 0 && comm_dim_partitioned(1)){
        int offset = ( x2 + nFace -nb)*X4*X3*X1/2+(x4*X3*X1 + x3*X1+x1)/2;
        return back_nbr_spinor[1] + offset * my_spinor_site_size;
      } 
      j = (x4*X3*X2*X1 + x3*X2*X1 + new_x2*X1 + x1) / 2;
      break;
    }
  case 4://+Z
    {
      int new_x3 = (x3 + nb)% X3;
      if(x3+nb >=X3 && comm_dim_partitioned(2)){
        int offset = ( x3 + nb -X3)*X4*X2*X1/2+(x4*X2*X1 + x2*X1+x1)/2;
        return fwd_nbr_spinor[2] + offset * my_spinor_site_size;
      } 
      j = (x4*X3*X2*X1 + new_x3*X2*X1 + x2*X1 + x1) / 2;
      break;
    }
  case 5://-Z
    {
      int new_x3 = (x3 - nb + X3)% X3;
      if(x3 - nb < 0 && comm_dim_partitioned(2)){
        int offset = ( x3 + nFace -nb)*X4*X2*X1/2+(x4*X2*X1 + x2*X1+x1)/2;
        return back_nbr_spinor[2] + offset * my_spinor_site_size;
      }
      j = (x4*X3*X2*X1 + new_x3*X2*X1 + x2*X1 + x1) / 2;
      break;
    }
  case 6://+T
    {
      j = neighborIndex_mg(i, oddBit, +nb, 0, 0, 0);
      int x4 = x4_mg(i, oddBit);
      if ( (x4 + nb) >= Z[3]  && comm_dim_partitioned(3)){
        int offset = (x4+nb - Z[3])*Vsh_t;
        return &fwd_nbr_spinor[3][(offset + j) * my_spinor_site_size];
      }
      break;
    }
  case 7://-T
    {
      j = neighborIndex_mg(i, oddBit, -nb, 0, 0, 0);
      int x4 = x4_mg(i, oddBit);
      if ( (x4 - nb) < 0 && comm_dim_partitioned(3)){
        int offset = ( x4 - nb +nFace)*Vsh_t;
        return &back_nbr_spinor[3][(offset + j) * my_spinor_site_size];
      }
      break;
    }
  default: j = -1; printf("ERROR: wrong dir\n"); exit(1);
  }

  return &spinorField[j * (my_spinor_site_size)];
}

template <QudaPCType type> int neighborIndex_5d_mgpu(int i, int oddBit, int dxs, int dx4, int dx3, int dx2, int dx1)
{
  int ret;

  int Y = (type == QUDA_5D_PC) ? fullLatticeIndex_5d(i, oddBit) : fullLatticeIndex_5d_4dpc(i, oddBit);

  int xs = Y/(Z[3]*Z[2]*Z[1]*Z[0]);
  int x4 = (Y/(Z[2]*Z[1]*Z[0])) % Z[3];
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  int ghost_x4 = x4+ dx4;

  xs = (xs+dxs+Ls) % Ls;
  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];

  if ( (ghost_x4 >= 0 && ghost_x4) < Z[3] || !comm_dim_partitioned(3)){
    ret = (xs*Z[3]*Z[2]*Z[1]*Z[0] + x4*Z[2]*Z[1]*Z[0] + x3*Z[1]*Z[0] + x2*Z[0] + x1) >> 1;
  }else{
    ret = (xs*Z[2]*Z[1]*Z[0] + x3*Z[1]*Z[0] + x2*Z[0] + x1) >> 1;
  }

  return ret;
}

template <QudaPCType type> int x4_5d_mgpu(int i, int oddBit)
{
  int Y = (type == QUDA_5D_PC) ? fullLatticeIndex_5d(i, oddBit) : fullLatticeIndex_5d_4dpc(i, oddBit);
  return (Y/(Z[2]*Z[1]*Z[0])) % Z[3];
}

template <QudaPCType type, typename Float>
Float *spinorNeighbor_5d_mgpu(int i, int dir, int oddBit, Float *spinorField, Float **fwd_nbr_spinor,
    Float **back_nbr_spinor, int neighbor_distance, int nFace, int spinorSize = 24)
{
  int j;
  int nb = neighbor_distance;
  int Y = (type == QUDA_5D_PC) ? fullLatticeIndex_5d(i, oddBit) : fullLatticeIndex_5d_4dpc(i, oddBit);

  int xs = Y/(Z[3]*Z[2]*Z[1]*Z[0]);
  int x4 = (Y/(Z[2]*Z[1]*Z[0])) % Z[3];
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
      if(x1+nb >=X1 && comm_dim_partitioned(0)) {
        int offset = ((x1 + nb -X1)*Ls*X4*X3*X2+xs*X4*X3*X2+x4*X3*X2 + x3*X2+x2) >> 1;
        return fwd_nbr_spinor[0] + offset*spinorSize;
      }
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + x3*X2*X1 + x2*X1 + new_x1) >> 1;
      break;
    }
  case 1://-X
    {
      int new_x1 = (x1 - nb + X1)% X1;
      if(x1 - nb < 0 && comm_dim_partitioned(0)) {
        int offset = (( x1+nFace- nb)*Ls*X4*X3*X2 + xs*X4*X3*X2 + x4*X3*X2 + x3*X2 + x2) >> 1;
        return back_nbr_spinor[0] + offset*spinorSize;
      }
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + x3*X2*X1 + x2*X1 + new_x1) >> 1;
      break;
    }
  case 2://+Y
    {
      int new_x2 = (x2 + nb)% X2;
      if(x2+nb >=X2 && comm_dim_partitioned(1)) {
        int offset = (( x2 + nb -X2)*Ls*X4*X3*X1+xs*X4*X3*X1+x4*X3*X1 + x3*X1+x1) >> 1;
        return fwd_nbr_spinor[1] + offset*spinorSize;
      }
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + x3*X2*X1 + new_x2*X1 + x1) >> 1;
      break;
    }
  case 3:// -Y
    {
      int new_x2 = (x2 - nb + X2)% X2;
      if(x2 - nb < 0 && comm_dim_partitioned(1)) {
        int offset = (( x2 + nFace -nb)*Ls*X4*X3*X1+xs*X4*X3*X1+ x4*X3*X1 + x3*X1+x1) >> 1;
        return back_nbr_spinor[1] + offset*spinorSize;
      }
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + x3*X2*X1 + new_x2*X1 + x1) >> 1;
      break;
    }
  case 4://+Z
    {
      int new_x3 = (x3 + nb)% X3;
      if(x3+nb >=X3 && comm_dim_partitioned(2)) {
        int offset = (( x3 + nb -X3)*Ls*X4*X2*X1+xs*X4*X2*X1+x4*X2*X1 + x2*X1+x1) >> 1;
        return fwd_nbr_spinor[2] + offset*spinorSize;
      }
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + new_x3*X2*X1 + x2*X1 + x1) >> 1;
      break;
    }
  case 5://-Z
    {
      int new_x3 = (x3 - nb + X3)% X3;
      if(x3 - nb < 0 && comm_dim_partitioned(2)){
        int offset = (( x3 + nFace -nb)*Ls*X4*X2*X1+xs*X4*X2*X1+x4*X2*X1+x2*X1+x1) >> 1;
        return back_nbr_spinor[2] + offset*spinorSize;
      }
      j = (xs*X4*X3*X2*X1 + x4*X3*X2*X1 + new_x3*X2*X1 + x2*X1 + x1) >> 1;
      break;
    }
  case 6://+T
    {
      int x4 = x4_5d_mgpu<type>(i, oddBit);
      if ( (x4 + nb) >= Z[3] && comm_dim_partitioned(3)) {
        int offset = ((x4 + nb - Z[3])*Ls*X3*X2*X1+xs*X3*X2*X1+x3*X2*X1+x2*X1+x1) >> 1;
        return fwd_nbr_spinor[3] + offset*spinorSize;
      }
      j = neighborIndex_5d_mgpu<type>(i, oddBit, 0, +nb, 0, 0, 0);
      break;
    }
  case 7://-T
    {
      int x4 = x4_5d_mgpu<type>(i, oddBit);
      if ( (x4 - nb) < 0 && comm_dim_partitioned(3)) {
        int offset = (( x4 - nb +nFace)*Ls*X3*X2*X1+xs*X3*X2*X1+x3*X2*X1+x2*X1+x1) >> 1;
        return back_nbr_spinor[3] + offset*spinorSize;
      }
      j = neighborIndex_5d_mgpu<type>(i, oddBit, 0, -nb, 0, 0, 0);
      break;
    }
  default: j = -1; printf("ERROR: wrong dir\n"); exit(1);
  }

  return &spinorField[j*(spinorSize)];
}


#endif // MULTI_GPU
