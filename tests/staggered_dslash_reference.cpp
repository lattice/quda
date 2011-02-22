#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <test_util.h>
#include <quda_internal.h>
#include <quda.h>
#include <util_quda.h>
#include <staggered_dslash_reference.h>
#include "misc.h"
#include "exchange_face.h"

extern void *memset(void *s, int c, size_t n);

static int mySpinorSiteSize = 6;

int Z[4];
int V;
int Vh;
int Vs_t;
int Vsh_x, Vsh_y, Vsh_z, Vsh_t;

void setDims(int *X) {
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];
  }
  Vh = V/2;

  Vs_t = Z[0]*Z[1]*Z[2];
  Vsh_t = Vs_t/2;
}

template <typename Float>
void sum(Float *dst, Float *a, Float *b, int cnt) {
  for (int i = 0; i < cnt; i++)
    dst[i] = a[i] + b[i];
}
template <typename Float>
void sub(Float *dst, Float *a, Float *b, int cnt) {
  for (int i = 0; i < cnt; i++)
    dst[i] = a[i] - b[i];
}
// performs the operation y[i] = x[i] + a*y[i]
template <typename Float>
void xpay(Float *x, Float a, Float *y, int len) {
    for (int i=0; i<len; i++) y[i] = x[i] + a*y[i];
}
// performs the operation y[i] = a*x[i] - y[i]
template <typename Float>
void axmy(Float *x, Float a, Float *y, int len) {
    for (int i=0; i<len; i++) y[i] = a*x[i] - y[i];
}

template <typename Float>
void negx(Float *x, int len) {
    for (int i=0; i<len; i++) x[i] = -x[i];
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
Float *gaugeLink(int i, int dir, int oddBit, Float **gaugeEven, Float **gaugeOdd, int nbr_distance) {
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
Float *spinorNeighbor(int i, int dir, int oddBit, Float *spinorField, int neighbor_distance) 
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

template <typename sFloat, typename gFloat>
void dot(sFloat* res, gFloat* a, sFloat* b) {
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
void su3Transpose(Float *res, Float *mat) {
  for (int m = 0; m < 3; m++) {
    for (int n = 0; n < 3; n++) {
      res[m*(3*2) + n*(2) + 0] = + mat[n*(3*2) + m*(2) + 0];
      res[m*(3*2) + n*(2) + 1] = - mat[n*(3*2) + m*(2) + 1];
    }
  }
}


template <typename sFloat, typename gFloat>
void su3Mul(sFloat *res, gFloat *mat, sFloat *vec) {
  for (int n = 0; n < 3; n++) dot(&res[n*(2)], &mat[n*(3*2)], vec);
}

template <typename sFloat, typename gFloat>
void su3Tmul(sFloat *res, gFloat *mat, sFloat *vec) {
  gFloat matT[3*3*2];
  su3Transpose(matT, mat);
  su3Mul(res, matT, vec);
}


//
// dslashReference()
//
// if oddBit is zero: calculate even parity spinor elements (using odd parity spinor) 
// if oddBit is one:  calculate odd parity spinor elements 
//
// if daggerBit is zero: perform ordinary dslash operator
// if daggerBit is one:  perform hermitian conjugate of dslash
//
template<typename Float>
void display_link_internal(Float* link)
{
    int i, j;
    
    for (i = 0;i < 3; i++){
	for(j=0;j < 3; j++){
	    printf("(%10f,%10f) \t", link[i*3*2 + j*2], link[i*3*2 + j*2 + 1]);
	}
	printf("\n");
    }
    printf("\n");
    return;
}


template <typename sFloat, typename gFloat>
void dslashReference(sFloat *res, gFloat **fatlink, gFloat** longlink, sFloat *spinorField, int oddBit, int daggerBit) 
{
    for (int i=0; i<Vh*1*3*2; i++) res[i] = 0.0;
    
    gFloat *fatlinkEven[4], *fatlinkOdd[4];
    gFloat *longlinkEven[4], *longlinkOdd[4];
    
    for (int dir = 0; dir < 4; dir++) {  
	fatlinkEven[dir] = fatlink[dir];
	fatlinkOdd[dir] = fatlink[dir] + Vh*gaugeSiteSize;
	longlinkEven[dir] =longlink[dir];
	longlinkOdd[dir] = longlink[dir] + Vh*gaugeSiteSize;    
    }

    for (int i = 0; i < Vh; i++) {
	memset(res + i*mySpinorSiteSize, 0, mySpinorSiteSize*sizeof(sFloat));
	for (int dir = 0; dir < 8; dir++) {
	    gFloat* fatlnk = gaugeLink(i, dir, oddBit, fatlinkEven, fatlinkOdd, 1);
	    gFloat* longlnk = gaugeLink(i, dir, oddBit, longlinkEven, longlinkOdd, 3);

	    sFloat *first_neighbor_spinor = spinorNeighbor(i, dir, oddBit, spinorField, 1);
	    sFloat *third_neighbor_spinor = spinorNeighbor(i, dir, oddBit, spinorField, 3);


	    sFloat gaugedSpinor[mySpinorSiteSize];

	    if (dir % 2 == 0){
		su3Mul(gaugedSpinor, fatlnk, first_neighbor_spinor);
		sum(&res[i*mySpinorSiteSize], &res[i*mySpinorSiteSize], gaugedSpinor, mySpinorSiteSize);	    
		su3Mul(gaugedSpinor, longlnk, third_neighbor_spinor);
		sum(&res[i*mySpinorSiteSize], &res[i*mySpinorSiteSize], gaugedSpinor, mySpinorSiteSize);								
	    }
	    else{
		su3Tmul(gaugedSpinor, fatlnk, first_neighbor_spinor);
		sub(&res[i*mySpinorSiteSize], &res[i*mySpinorSiteSize], gaugedSpinor, mySpinorSiteSize);       	
		
		su3Tmul(gaugedSpinor, longlnk, third_neighbor_spinor);
		sub(&res[i*mySpinorSiteSize], &res[i*mySpinorSiteSize], gaugedSpinor, mySpinorSiteSize);       
		
	    }	    	    
	}
	if (daggerBit){
	    negx(&res[i*mySpinorSiteSize], mySpinorSiteSize);
	}
    }

}




void staggered_dslash(void *res, void **fatlink, void** longlink, void *spinorField, int oddBit, int daggerBit,
		      QudaPrecision sPrecision, QudaPrecision gPrecision) {
    
    if (sPrecision == QUDA_DOUBLE_PRECISION) {
	if (gPrecision == QUDA_DOUBLE_PRECISION){
	    dslashReference((double*)res, (double**)fatlink, (double**)longlink, (double*)spinorField, oddBit, daggerBit);
	}else{
	    dslashReference((double*)res, (float**)fatlink, (float**)longlink, (double*)spinorField, oddBit, daggerBit);
	}
    }
    else{
	if (gPrecision == QUDA_DOUBLE_PRECISION){
	    dslashReference((float*)res, (double**)fatlink, (double**)longlink, (float*)spinorField, oddBit, daggerBit);
	}else{
	    dslashReference((float*)res, (float**)fatlink, (float**)longlink, (float*)spinorField, oddBit, daggerBit);
	}
    }
}




template <typename sFloat, typename gFloat>
void Mat(sFloat *out, gFloat **fatlink, gFloat** longlink, sFloat *in, sFloat kappa, int daggerBit) 
{
    sFloat *inEven = in;
    sFloat *inOdd  = in + Vh*mySpinorSiteSize;
    sFloat *outEven = out;
    sFloat *outOdd = out + Vh*mySpinorSiteSize;
    
    // full dslash operator
    dslashReference(outOdd, fatlink, longlink, inEven, 1, daggerBit);
    dslashReference(outEven, fatlink, longlink, inOdd, 0, daggerBit);
    
    // lastly apply the kappa term
    xpay(in, -kappa, out, V*mySpinorSiteSize);
}


void 
mat(void *out, void **fatlink, void** longlink, void *in, double kappa, int dagger_bit,
       QudaPrecision sPrecision, QudaPrecision gPrecision) 
{
    
    if (sPrecision == QUDA_DOUBLE_PRECISION){
	if (gPrecision == QUDA_DOUBLE_PRECISION) {
	    Mat((double*)out, (double**)fatlink, (double**)longlink, (double*)in, (double)kappa, dagger_bit);
	}else {
	    Mat((double*)out, (float**)fatlink, (float**)longlink, (double*)in, (double)kappa, dagger_bit);
	}
    }else{
	if (gPrecision == QUDA_DOUBLE_PRECISION){ 
	    Mat((float*)out, (double**)fatlink, (double**)longlink, (float*)in, (float)kappa, dagger_bit);
	}else {
	    Mat((float*)out, (float**)fatlink, (float**)longlink, (float*)in, (float)kappa, dagger_bit);
	}
    }
}



template <typename sFloat, typename gFloat>
void
Matdagmat(sFloat *out, gFloat **fatlink, gFloat** longlink, sFloat *in, sFloat mass, int daggerBit, sFloat* tmp, QudaParity parity) 
{
    
    sFloat msq_x4 = mass*mass*4;

    switch(parity){
    case QUDA_EVEN_PARITY:
	{
	    sFloat *inEven = in;
	    sFloat *outEven = out;
	    dslashReference(tmp, fatlink, longlink, inEven, 1, daggerBit);
	    dslashReference(outEven, fatlink, longlink, tmp, 0, daggerBit);
	    
	    // lastly apply the mass term
	    axmy(inEven, msq_x4, outEven, Vh*mySpinorSiteSize);
	    break;
	}
    case QUDA_ODD_PARITY:
	{
	    sFloat *inOdd = in;
	    sFloat *outOdd = out;
	    dslashReference(tmp, fatlink, longlink, inOdd, 0, daggerBit);
	    dslashReference(outOdd, fatlink, longlink, tmp, 1, daggerBit);
	    
	    // lastly apply the mass term
	    axmy(inOdd, msq_x4, outOdd, Vh*mySpinorSiteSize);
	    break;	
	}
	
    default:
	fprintf(stderr, "ERROR: invalid parity in %s,line %d\n", __FUNCTION__, __LINE__);
	break;
    }
    
}



void 
matdagmat(void *out, void **fatlink, void** longlink, void *in, double mass, int dagger_bit,
	  QudaPrecision sPrecision, QudaPrecision gPrecision, void* tmp, QudaParity parity) 
{
  
  if (sPrecision == QUDA_DOUBLE_PRECISION){
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      Matdagmat((double*)out, (double**)fatlink, (double**)longlink, (double*)in, (double)mass, dagger_bit, (double*)tmp, parity);
    }else {
      Matdagmat((double*)out, (float**)fatlink, (float**)longlink, (double*)in, (double)mass, dagger_bit, (double*) tmp, parity);
    }
  }else{
    if (gPrecision == QUDA_DOUBLE_PRECISION){ 
      Matdagmat((float*)out, (double**)fatlink, (double**)longlink, (float*)in, (float)mass, dagger_bit, (float*)tmp, parity);
    }else {
      Matdagmat((float*)out, (float**)fatlink, (float**)longlink, (float*)in, (float)mass, dagger_bit, (float*)tmp, parity);
    }
  }
}





// Apply the even-odd preconditioned Dirac operator
template <typename sFloat, typename gFloat>
static void MatPC(sFloat *outEven, gFloat **fatlink, gFloat** longlink, sFloat *inEven, sFloat kappa, 
	      int daggerBit, MatPCType matpc_type) {
    
    sFloat *tmp = (sFloat*)malloc(Vh*mySpinorSiteSize*sizeof(sFloat));
    
    // full dslash operator
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
	dslashReference(tmp, fatlink, longlink, inEven, 1, daggerBit);
	dslashReference(outEven, fatlink, longlink, tmp, 0, daggerBit);

	//dslashReference(outEven, fatlink, longlink, inEven, 1, daggerBit);
    } else {
	dslashReference(tmp, fatlink, longlink, inEven, 0, daggerBit);
	dslashReference(outEven, fatlink, longlink, tmp, 1, daggerBit);
    }    
  
    // lastly apply the kappa term
    
    sFloat kappa2 = -kappa*kappa;
    xpay(inEven, kappa2, outEven, Vh*mySpinorSiteSize);
    
    free(tmp);
}


void
staggered_matpc(void *outEven, void **fatlink, void**longlink, void *inEven, double kappa, 
		MatPCType matpc_type, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision) 
{
    
    if (sPrecision == QUDA_DOUBLE_PRECISION)
	if (gPrecision == QUDA_DOUBLE_PRECISION) {
	    MatPC((double*)outEven, (double**)fatlink, (double**)longlink, (double*)inEven, (double)kappa, dagger_bit, matpc_type);
	}
	else{
	    MatPC((double*)outEven, (double**)fatlink, (double**)longlink, (double*)inEven, (double)kappa, dagger_bit, matpc_type);
	}
    else {
	if (gPrecision == QUDA_DOUBLE_PRECISION){ 
	    MatPC((float*)outEven, (double**)fatlink, (double**)longlink, (float*)inEven, (float)kappa, dagger_bit, matpc_type);
	}else{
	    MatPC((float*)outEven, (float**)fatlink, (float**)longlink, (float*)inEven, (float)kappa, dagger_bit, matpc_type);
	}
    }
}



#ifdef MULTI_GPU

int
x4_mg(int i, int oddBit)
{
  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  return x4;
}

 
template <typename Float>
Float *gaugeLink_mg4dir(int i, int dir, int oddBit, Float **gaugeEven, Float **gaugeOdd,
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
          int offset = (n_ghost_faces + x4 -d)*X1*X2*X3/2 + (x3*X2*X1 + x2*X2+x1)/2;
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
Float *spinorNeighbor_mg4dir(int i, int dir, int oddBit, Float *spinorField,
			     Float** fwd_nbr_spinor, Float** back_nbr_spinor, int neighbor_distance)
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
        int offset = ( x1+3- nb)*X4*X3*X2/2+(x4*X3*X2 + x3*X2+x2)/2;
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
        int offset = ( x2 + 3 -nb)*X4*X3*X1/2+(x4*X3*X1 + x3*X1+x1)/2;
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
        int offset = ( x3 + 3 -nb)*X4*X2*X1/2+(x4*X2*X1 + x2*X1+x1)/2;
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
        int offset = ( x4 - nb +3)*Vsh_t;
        return &back_nbr_spinor[3][(offset+j)*mySpinorSiteSize];
      }
      break;
  }
  default: j = -1; printf("ERROR: wrong dir\n"); exit(1);
  }

  return &spinorField[j*(mySpinorSiteSize)];
}




template <typename sFloat, typename gFloat>
void dslashReference_mg4dir(sFloat *res, gFloat **fatlink, gFloat** ghostFatlink, gFloat** longlink,  gFloat** ghostLonglink,
			    sFloat *spinorField, sFloat** fwd_nbr_spinor, sFloat** back_nbr_spinor, int oddBit, int daggerBit)
{

  

  for (int i=0; i<Vh*1*3*2; i++) res[i] = 0.0;

  int Vsh[4] = {Vsh_x, Vsh_y, Vsh_z, Vsh_t};
  gFloat *fatlinkEven[4], *fatlinkOdd[4];
  gFloat *longlinkEven[4], *longlinkOdd[4];
  gFloat *ghostFatlinkEven[4], *ghostFatlinkOdd[4];
  gFloat *ghostLonglinkEven[4], *ghostLonglinkOdd[4];

  for (int dir = 0; dir < 4; dir++) {
    fatlinkEven[dir] = fatlink[dir];
    fatlinkOdd[dir] = fatlink[dir] + Vh*gaugeSiteSize;
    longlinkEven[dir] =longlink[dir];
    longlinkOdd[dir] = longlink[dir] + Vh*gaugeSiteSize;

    ghostFatlinkEven[dir] = ghostFatlink[dir];
    ghostFatlinkOdd[dir] = ghostFatlink[dir] + Vsh[dir]*gaugeSiteSize;
    ghostLonglinkEven[dir] = ghostLonglink[dir];
    ghostLonglinkOdd[dir] = ghostLonglink[dir] + 3*Vsh[dir]*gaugeSiteSize;
  }



  for (int i = 0; i < Vh; i++) {
    memset(res + i*mySpinorSiteSize, 0, mySpinorSiteSize*sizeof(sFloat));
    for (int dir = 0; dir < 8; dir++) {
#if 1
      gFloat* fatlnk = gaugeLink_mg4dir(i, dir, oddBit, fatlinkEven, fatlinkOdd, ghostFatlinkEven, ghostFatlinkOdd, 1, 1);
      gFloat* longlnk = gaugeLink_mg4dir(i, dir, oddBit, longlinkEven, longlinkOdd, ghostLonglinkEven, ghostLonglinkOdd, 3, 3);

      sFloat *first_neighbor_spinor = spinorNeighbor_mg4dir(i, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 1);
      sFloat *third_neighbor_spinor = spinorNeighbor_mg4dir(i, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 3);

#else
      gFloat* fatlnk = gaugeLink(i, dir, oddBit, fatlinkEven, fatlinkOdd, 1);
      gFloat* longlnk = gaugeLink(i, dir, oddBit, longlinkEven, longlinkOdd, 3);

      sFloat *first_neighbor_spinor = spinorNeighbor(i, dir, oddBit, spinorField, 1);
      sFloat *third_neighbor_spinor = spinorNeighbor(i, dir, oddBit, spinorField, 3);
#endif

      sFloat gaugedSpinor[mySpinorSiteSize];


      if (dir % 2 == 0){
        su3Mul(gaugedSpinor, fatlnk, first_neighbor_spinor);
        sum(&res[i*mySpinorSiteSize], &res[i*mySpinorSiteSize], gaugedSpinor, mySpinorSiteSize);
        su3Mul(gaugedSpinor, longlnk, third_neighbor_spinor);
        sum(&res[i*mySpinorSiteSize], &res[i*mySpinorSiteSize], gaugedSpinor, mySpinorSiteSize);                                                        
      }
      else{
        su3Tmul(gaugedSpinor, fatlnk, first_neighbor_spinor);
        sub(&res[i*mySpinorSiteSize], &res[i*mySpinorSiteSize], gaugedSpinor, mySpinorSiteSize);

        su3Tmul(gaugedSpinor, longlnk, third_neighbor_spinor);
        sub(&res[i*mySpinorSiteSize], &res[i*mySpinorSiteSize], gaugedSpinor, mySpinorSiteSize);

      }
    }
    if (daggerBit){
      negx(&res[i*mySpinorSiteSize], mySpinorSiteSize);
    }
  }

}



void staggered_dslash_mg4dir(cpuColorSpinorField* out, void **fatlink, void** longlink, void** ghost_fatlink, void** ghost_longlink,
			     cpuColorSpinorField* in, 
			     int oddBit, int daggerBit,
			     QudaPrecision sPrecision, QudaPrecision gPrecision)
{

  QudaParity otherparity;

  if (oddBit == QUDA_EVEN_PARITY){
    otherparity = QUDA_ODD_PARITY;
  }else if (oddBit == QUDA_ODD_PARITY){
    otherparity = QUDA_EVEN_PARITY;
  }else{
    printf("ERROR: full parity not supported in function %s\n", __FUNCTION__);
    exit(1);
  }
  exchange_cpu_spinor4dir(in, Z, otherparity);
  
  void** fwd_nbr_spinor = in->fwdGhostFaceBuffer;
  void** back_nbr_spinor = in->backGhostFaceBuffer;

  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    if (gPrecision == QUDA_DOUBLE_PRECISION){
      dslashReference_mg4dir((double*)out->v, (double**)fatlink, (double**)ghost_fatlink,(double**)longlink,  (double**)ghost_longlink,
			     (double*)in->v, (double**)fwd_nbr_spinor, (double**)back_nbr_spinor, oddBit, daggerBit);
    }else{
      dslashReference_mg4dir((double*)out->v, (float**)fatlink, (float**)ghost_fatlink, (float**)longlink,  (float**)ghost_longlink,
			     (double*)in->v, (double**)fwd_nbr_spinor, (double**)back_nbr_spinor, oddBit, daggerBit);
    }
  }
  else{
    if (gPrecision == QUDA_DOUBLE_PRECISION){
      dslashReference_mg4dir((float*)out->v, (double**)fatlink, (double**)ghost_fatlink, (double**)longlink,  (double**)ghost_longlink,
			     (float*)in->v, (float**)fwd_nbr_spinor, (float**)back_nbr_spinor, oddBit, daggerBit);
    }else{
      dslashReference_mg4dir((float*)out->v, (float**)fatlink, (float**)ghost_fatlink,  (float**)longlink, (float**)ghost_longlink,
			     (float*)in->v, (float**)fwd_nbr_spinor, (float**)back_nbr_spinor, oddBit, daggerBit);
    }
  }
  
  
}


void 
matdagmat_mg4dir(cpuColorSpinorField* out, void **fatlink, void** ghost_fatlink, void** longlink, void** ghost_longlink, 
		 cpuColorSpinorField* in, double mass, int dagger_bit,
		 QudaPrecision sPrecision, QudaPrecision gPrecision, cpuColorSpinorField* tmp, QudaParity parity) 
{
  //assert sPrecision and gPrecision must be the same
  if (sPrecision != gPrecision){
    errorQuda("Spinor precision and gPrecison is not the same");
  }
  
  QudaParity otherparity;
  if (parity == QUDA_EVEN_PARITY){
    otherparity = QUDA_ODD_PARITY;
  }else if (parity == QUDA_ODD_PARITY){
    otherparity = QUDA_EVEN_PARITY;
  }else{
    printf("ERROR: full parity not supported in function %s\n", __FUNCTION__);
    exit(1);
  }
  
  staggered_dslash_mg4dir(tmp, fatlink, longlink, ghost_fatlink, ghost_longlink,
			  in, otherparity, dagger_bit, sPrecision, gPrecision);
  
  staggered_dslash_mg4dir(out, fatlink, longlink, ghost_fatlink, ghost_longlink,
			  tmp, parity, dagger_bit, sPrecision, gPrecision);
  
  double msq_x4 = mass*mass*4;
  if (sPrecision == QUDA_DOUBLE_PRECISION){
    axmy((double*)in->v, (double)msq_x4, (double*)out->v, Vh*mySpinorSiteSize);
  }else{
    axmy((float*)in->v, (float)msq_x4, (float*)out->v, Vh*mySpinorSiteSize);    
  }

}

#endif

