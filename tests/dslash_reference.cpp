#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <util_quda.h>

#include <test_util.h>
#include <blas_reference.h>
#include <dslash_reference.h>

int Z[4];
int V;
int Vh;

void setDims(int *X) {
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];
  }
  Vh = V/2;
}

template <typename Float>
void sum(Float *dst, Float *a, Float *b, int cnt) {
  for (int i = 0; i < cnt; i++)
    dst[i] = a[i] + b[i];
}

// performs the operation y[i] = x[i] + a*y[i]
template <typename Float>
void xpay(Float *x, Float a, Float *y, int len) {
    for (int i=0; i<len; i++) y[i] = x[i] + a*y[i];
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

int neighborIndex(int i, int oddBit, int dx4, int dx3, int dx2, int dx1) {
  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  
  // assert (oddBit == (x+y+z+t)%2);
  
  x4 = (x4+dx4+Z[3]) % Z[3];
  x3 = (x3+dx3+Z[2]) % Z[2];
  x2 = (x2+dx2+Z[1]) % Z[1];
  x1 = (x1+dx1+Z[0]) % Z[0];
  
  return (x4*(Z[2]*Z[1]*Z[0]) + x3*(Z[1]*Z[0]) + x2*(Z[0]) + x1) / 2;
}

template <typename Float>
Float *gaugeLink(int i, int dir, int oddBit, Float **gaugeEven, Float **gaugeOdd) {
  Float **gaugeField;
  int j;
  
  if (dir % 2 == 0) {
    j = i;
    gaugeField = (oddBit ? gaugeOdd : gaugeEven);
  }
  else {
    switch (dir) {
    case 1: j = neighborIndex(i, oddBit, 0, 0, 0, -1); break;
    case 3: j = neighborIndex(i, oddBit, 0, 0, -1, 0); break;
    case 5: j = neighborIndex(i, oddBit, 0, -1, 0, 0); break;
    case 7: j = neighborIndex(i, oddBit, -1, 0, 0, 0); break;
    default: j = -1; break;
    }
    gaugeField = (oddBit ? gaugeEven : gaugeOdd);
  }
  
  return &gaugeField[dir/2][j*(3*3*2)];
}

template <typename Float>
Float *spinorNeighbor(int i, int dir, int oddBit, Float *spinorField) {
  int j;
  switch (dir) {
  case 0: j = neighborIndex(i, oddBit, 0, 0, 0, +1); break;
  case 1: j = neighborIndex(i, oddBit, 0, 0, 0, -1); break;
  case 2: j = neighborIndex(i, oddBit, 0, 0, +1, 0); break;
  case 3: j = neighborIndex(i, oddBit, 0, 0, -1, 0); break;
  case 4: j = neighborIndex(i, oddBit, 0, +1, 0, 0); break;
  case 5: j = neighborIndex(i, oddBit, 0, -1, 0, 0); break;
  case 6: j = neighborIndex(i, oddBit, +1, 0, 0, 0); break;
  case 7: j = neighborIndex(i, oddBit, -1, 0, 0, 0); break;
  default: j = -1; break;
  }
  
  return &spinorField[j*(4*3*2)];
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

const double projector[8][4][4][2] = {
  {
    {{1,0}, {0,0}, {0,0}, {0,-1}},
    {{0,0}, {1,0}, {0,-1}, {0,0}},
    {{0,0}, {0,1}, {1,0}, {0,0}},
    {{0,1}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {0,1}},
    {{0,0}, {1,0}, {0,1}, {0,0}},
    {{0,0}, {0,-1}, {1,0}, {0,0}},
    {{0,-1}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {1,0}},
    {{0,0}, {1,0}, {-1,0}, {0,0}},
    {{0,0}, {-1,0}, {1,0}, {0,0}},
    {{1,0}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {-1,0}},
    {{0,0}, {1,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {1,0}, {0,0}},
    {{-1,0}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,-1}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {0,1}},
    {{0,1}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {0,-1}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,1}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {0,-1}},
    {{0,-1}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {0,1}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {-1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {-1,0}},
    {{-1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {-1,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {1,0}},
    {{1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {1,0}}
  }
};


// todo pass projector
template <typename Float>
void multiplySpinorByDiracProjector(Float *res, int projIdx, Float *spinorIn) {
  for (int i=0; i<4*3*2; i++) res[i] = 0.0;

  for (int s = 0; s < 4; s++) {
    for (int t = 0; t < 4; t++) {
      Float projRe = projector[projIdx][s][t][0];
      Float projIm = projector[projIdx][s][t][1];
      
      for (int m = 0; m < 3; m++) {
	Float spinorRe = spinorIn[t*(3*2) + m*(2) + 0];
	Float spinorIm = spinorIn[t*(3*2) + m*(2) + 1];
	res[s*(3*2) + m*(2) + 0] += projRe*spinorRe - projIm*spinorIm;
	res[s*(3*2) + m*(2) + 1] += projRe*spinorIm + projIm*spinorRe;
      }
    }
  }
}


//
// dslashReference()
//
// if oddBit is zero: calculate odd parity spinor elements (using even parity spinor)
// if oddBit is one:  calculate even parity spinor elements
//
// if daggerBit is zero: perform ordinary dslash operator
// if daggerBit is one:  perform hermitian conjugate of dslash
//
template <typename sFloat, typename gFloat>
void dslashReference(sFloat *res, gFloat **gaugeFull, sFloat *spinorField, int oddBit, int daggerBit) {
  for (int i=0; i<Vh*4*3*2; i++) res[i] = 0.0;
  
  gFloat *gaugeEven[4], *gaugeOdd[4];
  for (int dir = 0; dir < 4; dir++) {  
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir]  = gaugeFull[dir]+Vh*gaugeSiteSize;
  }
  
  for (int i = 0; i < Vh; i++) {
    for (int dir = 0; dir < 8; dir++) {
      gFloat *gauge = gaugeLink(i, dir, oddBit, gaugeEven, gaugeOdd);
      sFloat *spinor = spinorNeighbor(i, dir, oddBit, spinorField);
      
      sFloat projectedSpinor[4*3*2], gaugedSpinor[4*3*2];
      int projIdx = 2*(dir/2)+(dir+daggerBit)%2;
      multiplySpinorByDiracProjector(projectedSpinor, projIdx, spinor);
      
      for (int s = 0; s < 4; s++) {
	if (dir % 2 == 0)
	  su3Mul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
	else
	  su3Tmul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
      }
      
      sum(&res[i*(4*3*2)], &res[i*(4*3*2)], gaugedSpinor, 4*3*2);
    }
  }
}

void dslash(void *res, void **gaugeFull, void *spinorField, int oddBit, int daggerBit,
	    QudaPrecision sPrecision, QudaPrecision gPrecision) {
  
  if (sPrecision == QUDA_DOUBLE_PRECISION) 
    if (gPrecision == QUDA_DOUBLE_PRECISION)
      dslashReference((double*)res, (double**)gaugeFull, (double*)spinorField, oddBit, daggerBit);
    else
      dslashReference((double*)res, (float**)gaugeFull, (double*)spinorField, oddBit, daggerBit);
  else
    if (gPrecision == QUDA_DOUBLE_PRECISION)
      dslashReference((float*)res, (double**)gaugeFull, (float*)spinorField, oddBit, daggerBit);
    else
      dslashReference((float*)res, (float**)gaugeFull, (float*)spinorField, oddBit, daggerBit);

}

template <typename sFloat, typename gFloat>
void Mat(sFloat *out, gFloat **gauge, sFloat *in, sFloat kappa, int daggerBit) {
  sFloat *inEven = in;
  sFloat *inOdd  = in + Vh*spinorSiteSize;
  sFloat *outEven = out;
  sFloat *outOdd = out + Vh*spinorSiteSize;
  
  // full dslash operator
  dslashReference(outOdd, gauge, inEven, 1, daggerBit);
  dslashReference(outEven, gauge, inOdd, 0, daggerBit);
  
  // lastly apply the kappa term
  xpay(in, -kappa, out, V*spinorSiteSize);
}

void mat(void *out, void **gauge, void *in, double kappa, int dagger_bit,
	 QudaPrecision sPrecision, QudaPrecision gPrecision) {

  if (sPrecision == QUDA_DOUBLE_PRECISION)
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      Mat((double*)out, (double**)gauge, (double*)in, (double)kappa, dagger_bit);
    else 
      Mat((double*)out, (float**)gauge, (double*)in, (double)kappa, dagger_bit);
  else
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      Mat((float*)out, (double**)gauge, (float*)in, (float)kappa, dagger_bit);
    else 
      Mat((float*)out, (float**)gauge, (float*)in, (float)kappa, dagger_bit);
}

// Apply the even-odd preconditioned Dirac operator
template <typename sFloat, typename gFloat>
void MatPC(sFloat *outEven, gFloat **gauge, sFloat *inEven, sFloat kappa, 
	   int daggerBit, QudaMatPCType matpc_type) {
  
  sFloat *tmp = (sFloat*)malloc(Vh*spinorSiteSize*sizeof(sFloat));
    
  // full dslash operator
  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashReference(tmp, gauge, inEven, 1, daggerBit);
    dslashReference(outEven, gauge, tmp, 0, daggerBit);
  } else {
    dslashReference(tmp, gauge, inEven, 0, daggerBit);
    dslashReference(outEven, gauge, tmp, 1, daggerBit);
  }    
  
  // lastly apply the kappa term
  sFloat kappa2 = -kappa*kappa;
  xpay(inEven, kappa2, outEven, Vh*spinorSiteSize);
  free(tmp);
}

void matpc(void *outEven, void **gauge, void *inEven, double kappa, 
	   QudaMatPCType matpc_type, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision) {

  if (sPrecision == QUDA_DOUBLE_PRECISION)
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      MatPC((double*)outEven, (double**)gauge, (double*)inEven, (double)kappa, dagger_bit, matpc_type);
    else
      MatPC((double*)outEven, (float**)gauge, (double*)inEven, (double)kappa, dagger_bit, matpc_type);
  else
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      MatPC((float*)outEven, (double**)gauge, (float*)inEven, (float)kappa, dagger_bit, matpc_type);
    else
      MatPC((float*)outEven, (float**)gauge, (float*)inEven, (float)kappa, dagger_bit, matpc_type);
}
