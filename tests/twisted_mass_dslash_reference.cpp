#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <util_quda.h>

#include <test_util.h>
#include <blas_reference.h>
#include <twisted_mass_dslash_reference.h>

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

// applies b*(1 + i*a*gamma_5)
template <typename sFloat>
void twistGamma5(sFloat *out, sFloat *in, const sFloat kappa, const sFloat mu, 
		 const QudaTwistFlavorType flavor, const int V, QudaTwistGamma5Type twist) {

  sFloat a=0.0,b=0.0;
  if (twist == QUDA_TWIST_GAMMA5_DIRECT) { // applying the twist
    a = 2.0 * kappa * mu * flavor; // mu already includes the flavor
    b = 1.0;
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) { // applying the inverse twist
    a = -2.0 * kappa * mu * flavor;
    b = 1.0 / (1.0 + a*a);
  } else {
    printf("Twist type %d not defined\n", twist);
    exit(0);
  }

  for(int i = 0; i < V; i++) {
    sFloat tmp[24];
    for(int s = 0; s < 4; s++)
      for(int c = 0; c < 3; c++) {
	sFloat a5 = ((s / 2) ? -1.0 : +1.0) * a;	  
	tmp[s * 6 + c * 2 + 0] = b* (in[i * 24 + s * 6 + c * 2 + 0] - a5*in[i * 24 + s * 6 + c * 2 + 1]);
	tmp[s * 6 + c * 2 + 1] = b* (in[i * 24 + s * 6 + c * 2 + 1] + a5*in[i * 24 + s * 6 + c * 2 + 0]);
      }

    for (int j=0; j<24; j++) out[i*24+j] = tmp[j];
  }
  
}

// this actually applies the preconditioned dslash, e.g., D_ee^{-1} D_eo or D_oo^{-1} D_oe
void dslash(void *res, void **gaugeFull, void *spinorField, double kappa, double mu, 
	    QudaTwistFlavorType flavor, int oddBit, int daggerBit,
	    QudaPrecision sPrecision, QudaPrecision gPrecision) {

  if (!daggerBit) {
    if (sPrecision == QUDA_DOUBLE_PRECISION) {
      if (gPrecision == QUDA_DOUBLE_PRECISION) {
	dslashReference((double*)res, (double**)gaugeFull, (double*)spinorField, oddBit, daggerBit);
      } else {
	dslashReference((double*)res, (float**)gaugeFull, (double*)spinorField, oddBit, daggerBit);
      } 
      twistGamma5((double*)res, (double*)res, kappa, mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
    } else {
      if (gPrecision == QUDA_DOUBLE_PRECISION) {
	dslashReference((float*)res, (double**)gaugeFull, (float*)spinorField, oddBit, daggerBit);
      } else {
	dslashReference((float*)res, (float**)gaugeFull, (float*)spinorField, oddBit, daggerBit);
      }
      twistGamma5((float*)res, (float*)res, (float)kappa, (float)mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
    }
  } else {
    if (sPrecision == QUDA_DOUBLE_PRECISION) {
      twistGamma5((double*)spinorField, (double*)spinorField, kappa, mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      if (gPrecision == QUDA_DOUBLE_PRECISION) {
	dslashReference((double*)res, (double**)gaugeFull, (double*)spinorField, oddBit, daggerBit);
      } else {
	dslashReference((double*)res, (float**)gaugeFull, (double*)spinorField, oddBit, daggerBit);
      }
      twistGamma5((double*)spinorField, (double*)spinorField, kappa, mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT);
    } else {
      twistGamma5((float*)spinorField, (float*)spinorField, (float)kappa, (float)mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      if (gPrecision == QUDA_DOUBLE_PRECISION) {
	dslashReference((float*)res, (double**)gaugeFull, (float*)spinorField, oddBit, daggerBit);
      } else {
	dslashReference((float*)res, (float**)gaugeFull, (float*)spinorField, oddBit, daggerBit);
      }
      twistGamma5((float*)spinorField, (float*)spinorField, (float)kappa, (float)mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT);
    }
  }
}

template <typename sFloat, typename gFloat>
void Mat(sFloat *out, gFloat **gauge, sFloat *in, sFloat kappa, sFloat mu, 
	 QudaTwistFlavorType flavor, int daggerBit) {

  sFloat *inEven = in;
  sFloat *inOdd  = in + Vh*spinorSiteSize;
  sFloat *outEven = out;
  sFloat *outOdd = out + Vh*spinorSiteSize;
  
  sFloat *tmp = (sFloat*)malloc(V*spinorSiteSize*sizeof(sFloat));

  // full dslash operator
  dslashReference(outOdd, gauge, inEven, 1, daggerBit);
  dslashReference(outEven, gauge, inOdd, 0, daggerBit);
  // apply the twist term
  twistGamma5(tmp, in, kappa, mu, flavor, V, QUDA_TWIST_GAMMA5_DIRECT);

  // combine
  xpay(tmp, -kappa, out, V*spinorSiteSize);

  free(tmp);
}

void mat(void *out, void **gauge, void *in, double kappa, double mu, 
	 QudaTwistFlavorType flavor, int dagger_bit,
	 QudaPrecision sPrecision, QudaPrecision gPrecision) {

  if (sPrecision == QUDA_DOUBLE_PRECISION)
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      Mat((double*)out, (double**)gauge, (double*)in, (double)kappa, (double)mu, flavor, dagger_bit);
    else 
      Mat((double*)out, (float**)gauge, (double*)in, (double)kappa, (double)mu, flavor, dagger_bit);
  else
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      Mat((float*)out, (double**)gauge, (float*)in, (float)kappa, (float)mu, flavor, dagger_bit);
    else 
      Mat((float*)out, (float**)gauge, (float*)in, (float)kappa, (float)mu, flavor, dagger_bit);
}

template <typename Float>
double norm2(Float *v, int len) {
  double sum=0.0;
  for (int i=0; i<len; i++) sum += v[i]*v[i];
  return sum;
}

// Apply the even-odd preconditioned Dirac operator
template <typename sFloat, typename gFloat>
void MatPC(sFloat *outEven, gFloat **gauge, sFloat *inEven, sFloat kappa, sFloat mu, 
	   QudaTwistFlavorType flavor, int daggerBit, QudaMatPCType matpc_type) {
  
  sFloat *tmp = (sFloat*)malloc(Vh*spinorSiteSize*sizeof(sFloat));
    
  if (!daggerBit) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashReference(tmp, gauge, inEven, 1, daggerBit);
      twistGamma5(tmp, tmp, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven, gauge, tmp, 0, daggerBit);
      twistGamma5(outEven, outEven, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      dslashReference(tmp, gauge, inEven, 0, daggerBit);
      twistGamma5(tmp, tmp, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven, gauge, tmp, 1, daggerBit);
      twistGamma5(outEven, outEven, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
    }
  } else {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      twistGamma5(inEven, inEven, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(tmp, gauge, inEven, 1, daggerBit);
      twistGamma5(tmp, tmp, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven, gauge, tmp, 0, daggerBit);
      twistGamma5(inEven, inEven, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      twistGamma5(inEven, inEven, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(tmp, gauge, inEven, 0, daggerBit);
      twistGamma5(tmp, tmp, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven, gauge, tmp, 1, daggerBit);
      twistGamma5(inEven, inEven, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT); // undo
    }
  }
  // lastly apply the kappa term
  sFloat kappa2 = -kappa*kappa;
  xpay(inEven, kappa2, outEven, Vh*spinorSiteSize);
  free(tmp);

}

void matpc(void *outEven, void **gauge, void *inEven, double kappa, double mu, QudaTwistFlavorType flavor,
	   QudaMatPCType matpc_type, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision) {

  if (matpc_type != QUDA_MATPC_EVEN_EVEN && matpc_type != QUDA_MATPC_ODD_ODD) {
    printf("Only symmetric preconditioning is implemented in reference\n");
    exit(-1);
  }

  if (sPrecision == QUDA_DOUBLE_PRECISION)
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      MatPC((double*)outEven, (double**)gauge, (double*)inEven, (double)kappa, (double)mu, 
	    flavor, dagger_bit, matpc_type);
    else
      MatPC((double*)outEven, (float**)gauge, (double*)inEven, (double)kappa, (double)mu, 
	    flavor, dagger_bit, matpc_type);
  else
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      MatPC((float*)outEven, (double**)gauge, (float*)inEven, (float)kappa, (float)mu, 
	    flavor, dagger_bit, matpc_type);
    else
      MatPC((float*)outEven, (float**)gauge, (float*)inEven, (float)kappa, (float)mu,
	    flavor, dagger_bit, matpc_type);
}
