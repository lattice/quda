#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <util_quda.h>

#include <host_utils.h>
#include <wilson_dslash_reference.h>

#include <gauge_field.h>
#include <color_spinor_field.h>

#include <dslash_reference.h>
#include <string.h>

using namespace quda;

static const double projector[8][4][4][2] = {
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

#ifndef MULTI_GPU

template <typename sFloat, typename gFloat>
void dslashReference(sFloat *res, gFloat **gaugeFull, sFloat *spinorField, int oddBit, int daggerBit) {
  for (int i = 0; i < Vh * my_spinor_site_size; i++) res[i] = 0.0;

  gFloat *gaugeEven[4], *gaugeOdd[4];
  for (int dir = 0; dir < 4; dir++) {  
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir] = gaugeFull[dir] + Vh * gauge_site_size;
  }
  
  for (int i = 0; i < Vh; i++) {
    for (int dir = 0; dir < 8; dir++) {
      gFloat *gauge = gaugeLink(i, dir, oddBit, gaugeEven, gaugeOdd, 1);
      sFloat *spinor = spinorNeighbor(i, dir, oddBit, spinorField, 1);
      
      sFloat projectedSpinor[4*3*2], gaugedSpinor[4*3*2];
      int projIdx = 2*(dir/2)+(dir+daggerBit)%2;
      multiplySpinorByDiracProjector(projectedSpinor, projIdx, spinor);
      
      for (int s = 0; s < 4; s++) {
	if (dir % 2 == 0) su3Mul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
	else su3Tmul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
      }
      
      sum(&res[i*(4*3*2)], &res[i*(4*3*2)], gaugedSpinor, 4*3*2);
    }
  }
}

#else

template <typename sFloat, typename gFloat>
void dslashReference(sFloat *res, gFloat **gaugeFull,  gFloat **ghostGauge, sFloat *spinorField, 
		     sFloat **fwdSpinor, sFloat **backSpinor, int oddBit, int daggerBit) {
  for (int i = 0; i < Vh * my_spinor_site_size; i++) res[i] = 0.0;

  gFloat *gaugeEven[4], *gaugeOdd[4];
  gFloat *ghostGaugeEven[4], *ghostGaugeOdd[4];
  for (int dir = 0; dir < 4; dir++) {  
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir] = gaugeFull[dir] + Vh * gauge_site_size;

    ghostGaugeEven[dir] = ghostGauge[dir];
    ghostGaugeOdd[dir] = ghostGauge[dir] + (faceVolume[dir] / 2) * gauge_site_size;
  }
  
  for (int i = 0; i < Vh; i++) {

    for (int dir = 0; dir < 8; dir++) {
      gFloat *gauge = gaugeLink_mg4dir(i, dir, oddBit, gaugeEven, gaugeOdd, ghostGaugeEven, ghostGaugeOdd, 1, 1);
      sFloat *spinor = spinorNeighbor_mg4dir(i, dir, oddBit, spinorField, fwdSpinor, backSpinor, 1, 1);

      sFloat projectedSpinor[my_spinor_site_size], gaugedSpinor[my_spinor_site_size];
      int projIdx = 2*(dir/2)+(dir+daggerBit)%2;
      multiplySpinorByDiracProjector(projectedSpinor, projIdx, spinor);
      
      for (int s = 0; s < 4; s++) {
	if (dir % 2 == 0) su3Mul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
	else su3Tmul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
      }
      
      sum(&res[i*(4*3*2)], &res[i*(4*3*2)], gaugedSpinor, 4*3*2);
    }

  }
}

#endif

// this actually applies the preconditioned dslash, e.g., D_ee^{-1} D_eo or D_oo^{-1} D_oe
void wil_dslash(void *out, void **gauge, void *in, int oddBit, int daggerBit,
		QudaPrecision precision, QudaGaugeParam &gauge_param) {
  
#ifndef MULTI_GPU  
  if (precision == QUDA_DOUBLE_PRECISION)
    dslashReference((double*)out, (double**)gauge, (double*)in, oddBit, daggerBit);
  else
    dslashReference((float*)out, (float**)gauge, (float*)in, oddBit, daggerBit);
#else

  GaugeFieldParam gauge_field_param(gauge, gauge_param);
  gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuGaugeField cpu(gauge_field_param);
  void **ghostGauge = (void**)cpu.Ghost();

  // Get spinor ghost fields
  // First wrap the input spinor into a ColorSpinorField
  ColorSpinorParam csParam;
  csParam.v = in;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = Z[d];
  csParam.setPrecision(precision);
  csParam.pad = 0;
  csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  csParam.x[0] /= 2;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_REFERENCE_FIELD_CREATE;
  
  cpuColorSpinorField inField(csParam);

  {  // Now do the exchange
    QudaParity otherParity = QUDA_INVALID_PARITY;
    if (oddBit == QUDA_EVEN_PARITY) otherParity = QUDA_ODD_PARITY;
    else if (oddBit == QUDA_ODD_PARITY) otherParity = QUDA_EVEN_PARITY;
    else errorQuda("ERROR: full parity not supported in function %s", __FUNCTION__);
    const int nFace = 1;

    inField.exchangeGhost(otherParity, nFace, daggerBit);
  }
  void** fwd_nbr_spinor = inField.fwdGhostFaceBuffer;
  void** back_nbr_spinor = inField.backGhostFaceBuffer;

  if (precision == QUDA_DOUBLE_PRECISION) {
    dslashReference((double*)out, (double**)gauge, (double**)ghostGauge, (double*)in, 
		    (double**)fwd_nbr_spinor, (double**)back_nbr_spinor, oddBit, daggerBit);
  } else{
    dslashReference((float*)out, (float**)gauge, (float**)ghostGauge, (float*)in, 
		    (float**)fwd_nbr_spinor, (float**)back_nbr_spinor, oddBit, daggerBit);
  }

#endif

}

// applies b*(1 + i*a*gamma_5)
template <typename sFloat>
void twistGamma5(sFloat *out, sFloat *in, const int dagger, const sFloat kappa, const sFloat mu, 
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

  if (dagger) a *= -1.0;

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

void twist_gamma5(void *out, void *in,  int daggerBit, double kappa, double mu, QudaTwistFlavorType flavor, 
		 int V, QudaTwistGamma5Type twist, QudaPrecision precision) {

  if (precision == QUDA_DOUBLE_PRECISION) {
    twistGamma5((double*)out, (double*)in, daggerBit, kappa, mu, flavor, V, twist);
  } else {
    twistGamma5((float*)out, (float*)in, daggerBit, (float)kappa, (float)mu, flavor, V, twist);
  } 
}


void tm_dslash(void *res, void **gaugeFull, void *spinorField, double kappa, double mu, 
	       QudaTwistFlavorType flavor, int oddBit, QudaMatPCType matpc_type, int daggerBit, QudaPrecision precision,
	       QudaGaugeParam &gauge_param)
{

  if (daggerBit && (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD))
    twist_gamma5(spinorField, spinorField, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);

  wil_dslash(res, gaugeFull, spinorField, oddBit, daggerBit, precision, gauge_param);

  if (!daggerBit || (daggerBit && (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC))) {
    twist_gamma5(res, res, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
  } else {
    twist_gamma5(spinorField, spinorField, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
  }
}

void wil_mat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision,
	     QudaGaugeParam &gauge_param) {

  void *inEven = in;
  void *inOdd = (char *)in + Vh * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + Vh * spinor_site_size * precision;

  wil_dslash(outOdd, gauge, inEven, 1, dagger_bit, precision, gauge_param);
  wil_dslash(outEven, gauge, inOdd, 0, dagger_bit, precision, gauge_param);

  // lastly apply the kappa term
  xpay(in, -kappa, out, V * spinor_site_size, precision);
}

void tm_mat(void *out, void **gauge, void *in, double kappa, double mu, 
	    QudaTwistFlavorType flavor, int dagger_bit, QudaPrecision precision,
	    QudaGaugeParam &gauge_param) {

  void *inEven = in;
  void *inOdd = (char *)in + Vh * spinor_site_size * precision;
  void *outEven = out;
  void *outOdd = (char *)out + Vh * spinor_site_size * precision;
  void *tmp = malloc(V * spinor_site_size * precision);

  wil_dslash(outOdd, gauge, inEven, 1, dagger_bit, precision, gauge_param);
  wil_dslash(outEven, gauge, inOdd, 0, dagger_bit, precision, gauge_param);

  // apply the twist term to the full lattice
  twist_gamma5(tmp, in, dagger_bit, kappa, mu, flavor, V, QUDA_TWIST_GAMMA5_DIRECT, precision);

  // combine
  xpay(tmp, -kappa, (double *)out, V * spinor_site_size, precision);

  free(tmp);
}

// Apply the even-odd preconditioned Dirac operator
void wil_matpc(void *outEven, void **gauge, void *inEven, double kappa, 
	       QudaMatPCType matpc_type, int daggerBit, QudaPrecision precision,
	       QudaGaugeParam &gauge_param) {

  void *tmp = malloc(Vh * spinor_site_size * precision);

  // FIXME: remove once reference clover is finished
  // full dslash operator
  if (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    wil_dslash(tmp, gauge, inEven, 1, daggerBit, precision, gauge_param);
    wil_dslash(outEven, gauge, tmp, 0, daggerBit, precision, gauge_param);
  } else {
    wil_dslash(tmp, gauge, inEven, 0, daggerBit, precision, gauge_param);
    wil_dslash(outEven, gauge, tmp, 1, daggerBit, precision, gauge_param);
  }    
  
  // lastly apply the kappa term
  double kappa2 = -kappa*kappa;
  xpay(inEven, kappa2, outEven, Vh * spinor_site_size, precision);

  free(tmp);
}

// Apply the even-odd preconditioned Dirac operator
void tm_matpc(void *outEven, void **gauge, void *inEven, double kappa, double mu, QudaTwistFlavorType flavor,
	      QudaMatPCType matpc_type, int daggerBit, QudaPrecision precision, QudaGaugeParam &gauge_param) {

  void *tmp = malloc(Vh * spinor_site_size * precision);

  if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    wil_dslash(tmp, gauge, inEven, 1, daggerBit, precision, gauge_param);
    twist_gamma5(tmp, tmp, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(outEven, gauge, tmp, 0, daggerBit, precision, gauge_param);
    twist_gamma5(tmp, inEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
  } else if (matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    wil_dslash(tmp, gauge, inEven, 0, daggerBit, precision, gauge_param);
    twist_gamma5(tmp, tmp, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
    wil_dslash(outEven, gauge, tmp, 1, daggerBit, precision, gauge_param);
    twist_gamma5(tmp, inEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
  } else if (!daggerBit) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      wil_dslash(tmp, gauge, inEven, 1, daggerBit, precision, gauge_param);
      twist_gamma5(tmp, tmp, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven, gauge, tmp, 0, daggerBit, precision, gauge_param);
      twist_gamma5(outEven, outEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      wil_dslash(tmp, gauge, inEven, 0, daggerBit, precision, gauge_param);
      twist_gamma5(tmp, tmp, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven, gauge, tmp, 1, daggerBit, precision, gauge_param);
      twist_gamma5(outEven, outEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
    }
  } else {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      twist_gamma5(inEven, inEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp, gauge, inEven, 1, daggerBit, precision, gauge_param);
      twist_gamma5(tmp, tmp, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven, gauge, tmp, 0, daggerBit, precision, gauge_param);
      twist_gamma5(inEven, inEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      twist_gamma5(inEven, inEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(tmp, gauge, inEven, 0, daggerBit, precision, gauge_param);
      twist_gamma5(tmp, tmp, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven, gauge, tmp, 1, daggerBit, precision, gauge_param);
      twist_gamma5(inEven, inEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision); // undo
    }
  }
  // lastly apply the kappa term
  double kappa2 = -kappa*kappa;
  if (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD) {
    xpay(inEven, kappa2, outEven, Vh * spinor_site_size, precision);
  } else {
    xpay(tmp, kappa2, outEven, Vh * spinor_site_size, precision);
  }

  free(tmp);
}


//----- for non-degenerate dslash only----
template <typename sFloat>
void ndegTwistGamma5(sFloat *out1, sFloat *out2, sFloat *in1, sFloat *in2, const int dagger, const sFloat kappa, const sFloat mu, 
		 const sFloat epsilon, const int V, QudaTwistGamma5Type twist) {

  sFloat a=0.0, b=0.0, d=0.0;
  if (twist == QUDA_TWIST_GAMMA5_DIRECT) { // applying the twist
    a = 2.0 * kappa * mu;
    b = -2.0 * kappa * epsilon;
    d = 1.0;
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) { // applying the inverse twist
    a = -2.0 * kappa * mu;
    b = 2.0 * kappa * epsilon;
    d = 1.0 / (1.0 + a*a - b*b);
  } else {
    printf("Twist type %d not defined\n", twist);
    exit(0);
  }

  if (dagger) a *= -1.0;
  
  for(int i = 0; i < V; i++) {
    sFloat tmp1[24];
    sFloat tmp2[24];    
    for(int s = 0; s < 4; s++)
      for(int c = 0; c < 3; c++) {
	sFloat a5 = ((s / 2) ? -1.0 : +1.0) * a;
        tmp1[s * 6 + c * 2 + 0] = d
            * (in1[i * 24 + s * 6 + c * 2 + 0] - a5 * in1[i * 24 + s * 6 + c * 2 + 1]
                + b * in2[i * 24 + s * 6 + c * 2 + 0]);
        tmp1[s * 6 + c * 2 + 1] = d
            * (in1[i * 24 + s * 6 + c * 2 + 1] + a5 * in1[i * 24 + s * 6 + c * 2 + 0]
                + b * in2[i * 24 + s * 6 + c * 2 + 1]);
        tmp2[s * 6 + c * 2 + 0] = d
            * (in2[i * 24 + s * 6 + c * 2 + 0] + a5 * in2[i * 24 + s * 6 + c * 2 + 1]
                + b * in1[i * 24 + s * 6 + c * 2 + 0]);
        tmp2[s * 6 + c * 2 + 1] = d
            * (in2[i * 24 + s * 6 + c * 2 + 1] - a5 * in2[i * 24 + s * 6 + c * 2 + 0]
                + b * in1[i * 24 + s * 6 + c * 2 + 1]);
      }
    for (int j=0; j<24; j++) out1[i*24+j] = tmp1[j], out2[i*24+j] = tmp2[j];
  }
  
}

void ndeg_twist_gamma5(void *outf1, void *outf2, void *inf1, void *inf2, const int dagger, const double kappa, const double mu, 
		 const double epsilon, const int Vf, QudaTwistGamma5Type twist, QudaPrecision precision) 
{
  if (precision == QUDA_DOUBLE_PRECISION) 
  {
      ndegTwistGamma5((double*)outf1, (double*)outf2, (double*)inf1, (double*)inf2, dagger, kappa, mu, epsilon, Vf, twist);
  } 
  else //single precision dslash
  {
      ndegTwistGamma5((float*)outf1, (float*)outf2, (float*)inf1, (float*)inf2, dagger, (float)kappa, (float)mu, (float)epsilon, Vf, twist);
  }
}

void tm_ndeg_dslash(void *res1, void *res2, void **gauge, void *spinorField1, void *spinorField2, double kappa, double mu, 
	           double epsilon, int oddBit, int daggerBit, QudaMatPCType matpc_type, QudaPrecision precision, QudaGaugeParam &gauge_param) 
{
  if (daggerBit && (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD))
    ndeg_twist_gamma5(spinorField1, spinorField2, spinorField1, spinorField2, daggerBit, kappa, mu, epsilon, Vh,
        QUDA_TWIST_GAMMA5_INVERSE, precision);

  wil_dslash(res1, gauge, spinorField1, oddBit, daggerBit, precision, gauge_param);
  wil_dslash(res2, gauge, spinorField2, oddBit, daggerBit, precision, gauge_param);
  
  if (!daggerBit || (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)) {
    ndeg_twist_gamma5(res1, res2, res1, res2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
  }
}


void tm_ndeg_matpc(void *outEven1, void *outEven2, void **gauge, void *inEven1, void *inEven2, double kappa, double mu, double epsilon,
	   QudaMatPCType matpc_type, int daggerBit, QudaPrecision precision, QudaGaugeParam &gauge_param) {

  void *tmp1 = malloc(Vh * spinor_site_size * precision);
  void *tmp2 = malloc(Vh * spinor_site_size * precision);

  if (!daggerBit) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      wil_dslash(tmp1, gauge, inEven1, 1, daggerBit, precision, gauge_param);
      wil_dslash(tmp2, gauge, inEven2, 1, daggerBit, precision, gauge_param);
      ndeg_twist_gamma5(tmp1, tmp2,  tmp1, tmp2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven1, gauge, tmp1, 0, daggerBit, precision, gauge_param);
      wil_dslash(outEven2, gauge, tmp2, 0, daggerBit, precision, gauge_param);
      ndeg_twist_gamma5(outEven1, outEven2, outEven1, outEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      wil_dslash(tmp1, gauge, inEven1, 0, daggerBit, precision, gauge_param);
      wil_dslash(tmp2, gauge, inEven2, 0, daggerBit, precision, gauge_param);
      ndeg_twist_gamma5(tmp1, tmp2, tmp1, tmp2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven1, gauge, tmp1, 1, daggerBit, precision, gauge_param);
      wil_dslash(outEven2, gauge, tmp2, 1, daggerBit, precision, gauge_param);
      ndeg_twist_gamma5(outEven1, outEven2, outEven1, outEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
    }
  } else {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      ndeg_twist_gamma5(
          tmp1, tmp2, inEven1, inEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven1, gauge, tmp1, 1, daggerBit, precision, gauge_param);
      wil_dslash(outEven2, gauge, tmp2, 1, daggerBit, precision, gauge_param);
      ndeg_twist_gamma5(tmp1, tmp2, outEven1, outEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven1, gauge, tmp1, 0, daggerBit, precision, gauge_param);
      wil_dslash(outEven2, gauge, tmp2, 0, daggerBit, precision, gauge_param);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      ndeg_twist_gamma5(
          tmp1, tmp2, inEven1, inEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven1, gauge, tmp1, 0, daggerBit, precision, gauge_param);
      wil_dslash(outEven2, gauge, tmp2, 0, daggerBit, precision, gauge_param);
      ndeg_twist_gamma5(tmp1, tmp2, outEven1, outEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven1, gauge, tmp1, 1, daggerBit, precision, gauge_param);
      wil_dslash(outEven2, gauge, tmp2, 1, daggerBit, precision, gauge_param);
    }
  }
  
  if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      wil_dslash(tmp1, gauge, inEven1, 1, daggerBit, precision, gauge_param);
      wil_dslash(tmp2, gauge, inEven2, 1, daggerBit, precision, gauge_param);
      ndeg_twist_gamma5(tmp1, tmp2,  tmp1, tmp2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven1, gauge, tmp1, 0, daggerBit, precision, gauge_param);
      wil_dslash(outEven2, gauge, tmp2, 0, daggerBit, precision, gauge_param);
  } else if (matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      wil_dslash(tmp1, gauge, inEven1, 0, daggerBit, precision, gauge_param);
      wil_dslash(tmp2, gauge, inEven2, 0, daggerBit, precision, gauge_param);
      ndeg_twist_gamma5(tmp1, tmp2, tmp1, tmp2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE, precision);
      wil_dslash(outEven1, gauge, tmp1, 1, daggerBit, precision, gauge_param);
      wil_dslash(outEven2, gauge, tmp2, 1, daggerBit, precision, gauge_param);
  }  
  
  // lastly apply the kappa term
  double kappa2 = -kappa*kappa;
  if (matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    ndeg_twist_gamma5(inEven1, inEven2, inEven1, inEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
  }

  xpay(inEven1, kappa2, outEven1, Vh * spinor_site_size, precision);
  xpay(inEven2, kappa2, outEven2, Vh * spinor_site_size, precision);

  free(tmp1);
  free(tmp2);
}


void tm_ndeg_mat(void *evenOut, void* oddOut, void **gauge, void *evenIn, void *oddIn,  double kappa, double mu, double epsilon, int daggerBit, QudaPrecision precision, QudaGaugeParam &gauge_param) 
{
  //V-4d volume and Vh=V/2
  void *inEven1   = evenIn;
  void *inEven2 = (char *)evenIn + precision * Vh * spinor_site_size;

  void *inOdd1    = oddIn;
  void *inOdd2 = (char *)oddIn + precision * Vh * spinor_site_size;

  void *outEven1  = evenOut;
  void *outEven2 = (char *)evenOut + precision * Vh * spinor_site_size;

  void *outOdd1   = oddOut;
  void *outOdd2 = (char *)oddOut + precision * Vh * spinor_site_size;

  void *tmpEven1 = malloc(Vh * spinor_site_size * precision);
  void *tmpEven2 = malloc(Vh * spinor_site_size * precision);

  void *tmpOdd1 = malloc(Vh * spinor_site_size * precision);
  void *tmpOdd2 = malloc(Vh * spinor_site_size * precision);

  // full dslash operator:
  wil_dslash(outOdd1, gauge, inEven1, 1, daggerBit, precision, gauge_param);
  wil_dslash(outOdd2, gauge, inEven2, 1, daggerBit, precision, gauge_param);

  wil_dslash(outEven1, gauge, inOdd1, 0, daggerBit, precision, gauge_param);
  wil_dslash(outEven2, gauge, inOdd2, 0, daggerBit, precision, gauge_param);

  // apply the twist term
  ndeg_twist_gamma5(tmpEven1, tmpEven2, inEven1, inEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
  ndeg_twist_gamma5(tmpOdd1, tmpOdd2, inOdd1, inOdd2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_DIRECT, precision);
  // combine
  xpay(tmpOdd1, -kappa, outOdd1, Vh * spinor_site_size, precision);
  xpay(tmpOdd2, -kappa, outOdd2, Vh * spinor_site_size, precision);

  xpay(tmpEven1, -kappa, outEven1, Vh * spinor_site_size, precision);
  xpay(tmpEven2, -kappa, outEven2, Vh * spinor_site_size, precision);

  free(tmpOdd1);
  free(tmpOdd2);
  //
  free(tmpEven1);
  free(tmpEven2);
}

//End of nondeg TM
