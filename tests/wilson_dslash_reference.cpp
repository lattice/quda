#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <util_quda.h>


#include <test_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>

#include <gauge_field.h>
#include <color_spinor_field.h>
#include <face_quda.h>

#include <dslash_util.h>
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
  for (int i=0; i<Vh*mySpinorSiteSize; i++) res[i] = 0.0;
  
  gFloat *gaugeEven[4], *gaugeOdd[4];
  for (int dir = 0; dir < 4; dir++) {  
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir]  = gaugeFull[dir]+Vh*gaugeSiteSize;
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
  for (int i=0; i<Vh*mySpinorSiteSize; i++) res[i] = 0.0;
  
  gFloat *gaugeEven[4], *gaugeOdd[4];
  gFloat *ghostGaugeEven[4], *ghostGaugeOdd[4];
  for (int dir = 0; dir < 4; dir++) {  
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir]  = gaugeFull[dir]+Vh*gaugeSiteSize;

    ghostGaugeEven[dir] = ghostGauge[dir];
    ghostGaugeOdd[dir] = ghostGauge[dir] + (faceVolume[dir]/2)*gaugeSiteSize;
  }
  
  for (int i = 0; i < Vh; i++) {

    for (int dir = 0; dir < 8; dir++) {
      gFloat *gauge = gaugeLink_mg4dir(i, dir, oddBit, gaugeEven, gaugeOdd, ghostGaugeEven, ghostGaugeOdd, 1, 1);
      sFloat *spinor = spinorNeighbor_mg4dir(i, dir, oddBit, spinorField, fwdSpinor, backSpinor, 1, 1);
      
      sFloat projectedSpinor[mySpinorSiteSize], gaugedSpinor[mySpinorSiteSize];
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
  csParam.precision = precision;
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

    inField.exchangeGhost(otherParity, daggerBit);
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


void wil_mat(void *out, void **gauge, void *in, double kappa, int dagger_bit, QudaPrecision precision,
	     QudaGaugeParam &gauge_param) {

  void *inEven = in;
  void *inOdd  = (char*)in + Vh*spinorSiteSize*precision;
  void *outEven = out;
  void *outOdd = (char*)out + Vh*spinorSiteSize*precision;

  wil_dslash(outOdd, gauge, inEven, 1, dagger_bit, precision, gauge_param);
  wil_dslash(outEven, gauge, inOdd, 0, dagger_bit, precision, gauge_param);

  // lastly apply the kappa term
  xpay(in, -kappa, out, V*spinorSiteSize, precision);
}

// Apply the even-odd preconditioned Dirac operator
void wil_matpc(void *outEven, void **gauge, void *inEven, double kappa, 
	       QudaMatPCType matpc_type, int daggerBit, QudaPrecision precision,
	       QudaGaugeParam &gauge_param) {

  void *tmp = malloc(Vh*spinorSiteSize*precision);
    
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
  xpay(inEven, kappa2, outEven, Vh*spinorSiteSize, precision);

  free(tmp);
}


//End of nondeg TM
