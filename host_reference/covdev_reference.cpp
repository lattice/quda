#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <host_utils.h>
#include <misc.h>
#include <covdev_reference.h>
#include <dslash_reference.h>

#include <quda_internal.h>
#include <quda.h>
#include <util_quda.h>
#include <blas_quda.h>

extern void *memset(void *s, int c, size_t n);

// covdevReference()
//
// if oddBit is zero: calculate even parity spinor elements (using odd parity spinor) 
// if oddBit is one:  calculate odd parity spinor elements 
//
// if daggerBit is zero: perform ordinary covariant derivative operator
// if daggerBit is one:  perform hermitian covariant derivative operator
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
void covdevReference(sFloat *res, gFloat **link, sFloat *spinorField, 
		     int oddBit, int daggerBit, int mu) 
{
  for (int i = 0; i < Vh * my_spinor_site_size; i++) res[i] = 0.0;

  gFloat *linkEven[4], *linkOdd[4];
  
  for (int dir = 0; dir < 4; dir++) {  
    linkEven[dir] = link[dir];
    linkOdd[dir] = link[dir] + Vh * gauge_site_size;
  }

  for (int sid = 0; sid < Vh; sid++) {
    int offset = my_spinor_site_size * sid;

    sFloat gaugedSpinor[my_spinor_site_size];

    gFloat *lnk    = gaugeLink(sid, mu, oddBit, linkEven, linkOdd, 1);
    sFloat *spinor = spinorNeighbor(sid, mu, oddBit, spinorField, 1);

    if (daggerBit) {
      for (int s = 0; s < 4; s++)
        su3Tmul(&gaugedSpinor[s*6], lnk, &spinor[s*6]);
    } else {
      for (int s = 0; s < 4; s++)
        su3Mul (&gaugedSpinor[s*6], lnk, &spinor[s*6]);
    }

    sum(&res[offset], &res[offset], gaugedSpinor, my_spinor_site_size);
  } // 4-d volume
}

void covdev_dslash(void *res, void **link, void *spinorField, int oddBit, int daggerBit, int mu,
                   QudaPrecision sPrecision, QudaPrecision gPrecision)
{

  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    if (gPrecision == QUDA_DOUBLE_PRECISION){
      covdevReference((double*)res, (double**)link, (double*)spinorField, oddBit, daggerBit, mu);
    } else {
      covdevReference((double*)res, (float**) link, (double*)spinorField, oddBit, daggerBit, mu);
    }
  }
  else {
    if (gPrecision == QUDA_DOUBLE_PRECISION){
      covdevReference((float*)res, (double**)link, (float*)spinorField, oddBit, daggerBit, mu);
    } else {
      covdevReference((float*)res, (float**) link, (float*)spinorField, oddBit, daggerBit, mu);
    }
  }
}

template <typename sFloat, typename gFloat>
void Mat(sFloat *out, gFloat **link, sFloat *in, int daggerBit, int mu) 
{
  sFloat *inEven = in;
  sFloat *inOdd = in + Vh * my_spinor_site_size;
  sFloat *outEven = out;
  sFloat *outOdd = out + Vh * my_spinor_site_size;

  // full dslash operator
  covdevReference(outOdd,  link, inEven, 1, daggerBit, mu);
  covdevReference(outEven, link, inOdd,  0, daggerBit, mu);
}


void mat(void *out, void **link, void *in, int dagger_bit, int mu,
    QudaPrecision sPrecision, QudaPrecision gPrecision) 
{
    
  if (sPrecision == QUDA_DOUBLE_PRECISION){
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      Mat((double*)out, (double**)link, (double*)in, dagger_bit, mu);
    } else {
      Mat((double*)out, (float**) link, (double*)in, dagger_bit, mu);
    }
  } else {
    if (gPrecision == QUDA_DOUBLE_PRECISION){ 
      Mat((float*)out, (double**)link, (float*)in, dagger_bit, mu);
    } else {
      Mat((float*)out, (float**) link, (float*)in, dagger_bit, mu);
    }
  }
}



template <typename sFloat, typename gFloat>
void Matdagmat(sFloat *out, gFloat **link, sFloat *in, int daggerBit, int mu, sFloat* tmp, QudaParity parity) 
{
  switch(parity){
  case QUDA_EVEN_PARITY:
  {
      sFloat *inEven  = in;
      sFloat *outEven = out;
      covdevReference(tmp,     link, inEven, 1, daggerBit, mu);
      covdevReference(outEven, link, tmp,    0, daggerBit, mu);
      break;
  }
  case QUDA_ODD_PARITY:
    {
      sFloat *inOdd  = in;
      sFloat *outOdd = out;
      covdevReference(tmp,    link, inOdd, 0, daggerBit, mu);
      covdevReference(outOdd, link, tmp,   1, daggerBit, mu);
      break;	
    }
	
  default:
    fprintf(stderr, "ERROR: invalid parity in %s,line %d\n", __FUNCTION__, __LINE__);
    break;
  }
    
}



void matdagmat(void *out, void **link, void *in, int dagger_bit, int mu,
	  QudaPrecision sPrecision, QudaPrecision gPrecision, void *tmp, QudaParity parity) 
{
  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      Matdagmat((double*)out, (double**)link, (double*)in, dagger_bit, mu, (double*)tmp, parity);
    } else {
      Matdagmat((double*)out, (float**) link, (double*)in, dagger_bit, mu, (double*)tmp, parity);
    }
  } else {
    if (gPrecision == QUDA_DOUBLE_PRECISION){ 
      Matdagmat((float*)out, (double**)link, (float*)in, dagger_bit, mu, (float*)tmp, parity);
    } else {
      Matdagmat((float*)out, (float**) link, (float*)in, dagger_bit, mu, (float*)tmp, parity);
    }
  }
}

#ifdef MULTI_GPU

template <typename sFloat, typename gFloat>
void covdevReference_mg4dir(sFloat *res, gFloat **link, gFloat **ghostLink, sFloat *spinorField,
                            sFloat **fwd_nbr_spinor, sFloat **back_nbr_spinor, int oddBit, int daggerBit, int mu)
{
  for (int i = 0; i < Vh * my_spinor_site_size; i++) res[i] = 0.0;

  gFloat *linkEven[4], *linkOdd[4];
  gFloat *ghostLinkEven[4], *ghostLinkOdd[4];

  for (int dir = 0; dir < 4; dir++) {
    linkEven[dir] = link[dir];
    linkOdd[dir] = link[dir] + Vh * gauge_site_size;

    ghostLinkEven[dir] = ghostLink[dir];
    ghostLinkOdd[dir] = ghostLink[dir] + (faceVolume[dir] / 2) * gauge_site_size;
  }

  for (int sid = 0; sid < Vh; sid++) {
    int offset = my_spinor_site_size * sid;

    gFloat *lnk    = gaugeLink_mg4dir(sid, mu, oddBit, linkEven, linkOdd, ghostLinkEven, ghostLinkOdd, 1, 1);
    sFloat *spinor = spinorNeighbor_mg4dir(sid, mu, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 1, 1);

    sFloat gaugedSpinor[my_spinor_site_size];

    if (daggerBit) {
      for (int s = 0; s < 4; s++)
        su3Tmul(&gaugedSpinor[s*6], lnk, &spinor[s*6]);
    } else {
      for (int s = 0; s < 4; s++)
        su3Mul (&gaugedSpinor[s*6], lnk, &spinor[s*6]);
    }
    sum(&res[offset], &res[offset], gaugedSpinor, my_spinor_site_size);
  } // 4-d volume
}

void covdev_dslash_mg4dir(cpuColorSpinorField* out, void **link, void** ghostLink, 
			     cpuColorSpinorField* in, int oddBit, int daggerBit, int mu,
			     QudaPrecision sPrecision, QudaPrecision gPrecision)
{
  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (oddBit == QUDA_EVEN_PARITY) {
    otherparity = QUDA_ODD_PARITY;
  } else if (oddBit == QUDA_ODD_PARITY) {
    otherparity = QUDA_EVEN_PARITY;
  } else {
    errorQuda("ERROR: full parity not supported in function %s", __FUNCTION__);
  }
  const int nFace = 1;

  in->exchangeGhost(otherparity, nFace, daggerBit);

  void** fwd_nbr_spinor = in->fwdGhostFaceBuffer;
  void** back_nbr_spinor = in->backGhostFaceBuffer;

  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      covdevReference_mg4dir((double*)out->V(), (double**)link, (double**)ghostLink, (double*)in->V(),
			     (double**)fwd_nbr_spinor, (double**)back_nbr_spinor, oddBit, daggerBit, mu);
    } else {
      covdevReference_mg4dir((double*)out->V(), (float**) link, (float**) ghostLink, (double*)in->V(),
			     (double**)fwd_nbr_spinor, (double**)back_nbr_spinor, oddBit, daggerBit, mu);
      }
  } else {
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      covdevReference_mg4dir((float*)out->V(), (double**)link, (double**)ghostLink, (float*)in->V(),
			     (float**)fwd_nbr_spinor, (float**)back_nbr_spinor, oddBit, daggerBit, mu);
    } else {
      covdevReference_mg4dir((float*)out->V(), (float**)link, (float**)ghostLink, (float*)in->V(),
			     (float**)fwd_nbr_spinor, (float**)back_nbr_spinor, oddBit, daggerBit, mu);
    }
  }
  
}

template <typename sFloat, typename gFloat>
void Mat_mg4dir(cpuColorSpinorField *out, gFloat **link, gFloat **ghostLink, cpuColorSpinorField *in, int daggerBit, int mu) 
{
  const int nFace = 1;
  {
    cpuColorSpinorField &inEven = static_cast<cpuColorSpinorField&>(in->Even());
    cpuColorSpinorField &outOdd  = static_cast<cpuColorSpinorField&>(out->Odd());

    inEven.exchangeGhost(QUDA_EVEN_PARITY, nFace, daggerBit);

    covdevReference_mg4dir(reinterpret_cast<sFloat*>(outOdd.V()), link, ghostLink,
			   reinterpret_cast<sFloat*>(inEven.V()),
			   reinterpret_cast<sFloat**>(inEven.fwdGhostFaceBuffer),
			   reinterpret_cast<sFloat**>(inEven.backGhostFaceBuffer),
			   1, daggerBit, mu);
  }

  {
    cpuColorSpinorField &inOdd  = static_cast<cpuColorSpinorField&>(in->Odd());
    cpuColorSpinorField &outEven = static_cast<cpuColorSpinorField&>(out->Even());

    inOdd.exchangeGhost(QUDA_ODD_PARITY, nFace, daggerBit);

    covdevReference_mg4dir(reinterpret_cast<sFloat*>(outEven.V()), link, ghostLink,
			   reinterpret_cast<sFloat*>(inOdd.V()),
			   reinterpret_cast<sFloat**>(inOdd.fwdGhostFaceBuffer),
			   reinterpret_cast<sFloat**>(inOdd.backGhostFaceBuffer),
			   0, daggerBit, mu);
  }
}


void mat_mg4dir(cpuColorSpinorField *out, void **link, void **ghostLink, cpuColorSpinorField *in, int dagger_bit, int mu,
    QudaPrecision sPrecision, QudaPrecision gPrecision) 
{
    
  if (sPrecision == QUDA_DOUBLE_PRECISION){
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      Mat_mg4dir<double, double>(out, (double**)link, (double**) ghostLink, in, dagger_bit, mu);
    } else {
      Mat_mg4dir<double, float> (out, (float**) link, (float**)  ghostLink, in, dagger_bit, mu);
    }
  } else {
    if (gPrecision == QUDA_DOUBLE_PRECISION){ 
      Mat_mg4dir<float, double> (out, (double**)link, (double**) ghostLink, in, dagger_bit, mu);
    } else {
      Mat_mg4dir<float, float>  (out, (float**) link, (float**)  ghostLink, in, dagger_bit, mu);
    }
  }
}



void matdagmat_mg4dir(cpuColorSpinorField* out, void **link, void** ghostLink, cpuColorSpinorField* in,
		      int dagger_bit, int mu, QudaPrecision sPrecision, QudaPrecision gPrecision,
		      cpuColorSpinorField* tmp, QudaParity parity)
{
  //assert sPrecision and gPrecision must be the same
  if (sPrecision != gPrecision){
    errorQuda("Spinor precision and gPrecison is not the same");
  }
  
  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (parity == QUDA_EVEN_PARITY){
    otherparity = QUDA_ODD_PARITY;
  } else if (parity == QUDA_ODD_PARITY) {
    otherparity = QUDA_EVEN_PARITY;
  } else {
    errorQuda("ERROR: full parity not supported in function %s\n", __FUNCTION__);
  }
  
  covdev_dslash_mg4dir(tmp, link, ghostLink, in,  otherparity, dagger_bit, mu, sPrecision, gPrecision);

  covdev_dslash_mg4dir(out, link, ghostLink, tmp, parity,      dagger_bit, mu, sPrecision, gPrecision);
}

#endif

