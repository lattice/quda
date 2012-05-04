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
#include <blas_quda.h>

#include <face_quda.h>

extern void *memset(void *s, int c, size_t n);

#include <dslash_util.h>

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
void dslashReference(sFloat *res, gFloat **fatlink, gFloat** longlink, sFloat *spinorField, 
		     int oddBit, int daggerBit) 
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
      } else {
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

template <typename sFloat, typename gFloat>
void dslashReference_mg4dir(sFloat *res, gFloat **fatlink, gFloat** longlink, 
			    gFloat** ghostFatlink, gFloat** ghostLonglink,
			    sFloat *spinorField, sFloat** fwd_nbr_spinor, 
			    sFloat** back_nbr_spinor, int oddBit, int daggerBit)
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
      gFloat* fatlnk = gaugeLink_mg4dir(i, dir, oddBit, fatlinkEven, fatlinkOdd, ghostFatlinkEven, ghostFatlinkOdd, 1, 1);
      gFloat* longlnk = gaugeLink_mg4dir(i, dir, oddBit, longlinkEven, longlinkOdd, ghostLonglinkEven, ghostLonglinkOdd, 3, 3);

      sFloat *first_neighbor_spinor = spinorNeighbor_mg4dir(i, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 1, 3);
      sFloat *third_neighbor_spinor = spinorNeighbor_mg4dir(i, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 3, 3);

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



void staggered_dslash_mg4dir(cpuColorSpinorField* out, void **fatlink, void** longlink, void** ghost_fatlink, 
			     void** ghost_longlink, cpuColorSpinorField* in, int oddBit, int daggerBit, 
			     QudaPrecision sPrecision, QudaPrecision gPrecision)
{

  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (oddBit == QUDA_EVEN_PARITY){
    otherparity = QUDA_ODD_PARITY;
  }else if (oddBit == QUDA_ODD_PARITY){
    otherparity = QUDA_EVEN_PARITY;
  }else{
    errorQuda("ERROR: full parity not supported in function %s", __FUNCTION__);
  }

  int Nc = 3;
  int nFace = 3;
  FaceBuffer faceBuf(Z, 4, 2*Nc, nFace, sPrecision);
  faceBuf.exchangeCpuSpinor(*in, otherparity, daggerBit); 
  
  void** fwd_nbr_spinor = in->fwdGhostFaceBuffer;
  void** back_nbr_spinor = in->backGhostFaceBuffer;

  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    if (gPrecision == QUDA_DOUBLE_PRECISION){
      dslashReference_mg4dir((double*)out->V(), (double**)fatlink, (double**)longlink,  
			     (double**)ghost_fatlink, (double**)ghost_longlink, (double*)in->V(), 
			     (double**)fwd_nbr_spinor, (double**)back_nbr_spinor, oddBit, daggerBit);
    } else {
      dslashReference_mg4dir((double*)out->V(), (float**)fatlink, (float**)longlink, (float**)ghost_fatlink, (float**)ghost_longlink,
			     (double*)in->V(), (double**)fwd_nbr_spinor, (double**)back_nbr_spinor, oddBit, daggerBit);
    }
  }
  else{
    if (gPrecision == QUDA_DOUBLE_PRECISION){
      dslashReference_mg4dir((float*)out->V(), (double**)fatlink, (double**)longlink, (double**)ghost_fatlink, (double**)ghost_longlink,
			     (float*)in->V(), (float**)fwd_nbr_spinor, (float**)back_nbr_spinor, oddBit, daggerBit);
    }else{
      dslashReference_mg4dir((float*)out->V(), (float**)fatlink, (float**)longlink, (float**)ghost_fatlink, (float**)ghost_longlink,
			     (float*)in->V(), (float**)fwd_nbr_spinor, (float**)back_nbr_spinor, oddBit, daggerBit);
    }
  }
  
  
}

void 
matdagmat_mg4dir(cpuColorSpinorField* out, void **fatlink, void** longlink, void** ghost_fatlink, void** ghost_longlink, 
		 cpuColorSpinorField* in, double mass, int dagger_bit,
		 QudaPrecision sPrecision, QudaPrecision gPrecision, cpuColorSpinorField* tmp, QudaParity parity) 
{
  //assert sPrecision and gPrecision must be the same
  if (sPrecision != gPrecision){
    errorQuda("Spinor precision and gPrecison is not the same");
  }
  
  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (parity == QUDA_EVEN_PARITY){
    otherparity = QUDA_ODD_PARITY;
  }else if (parity == QUDA_ODD_PARITY){
    otherparity = QUDA_EVEN_PARITY;
  }else{
    errorQuda("ERROR: full parity not supported in function %s\n", __FUNCTION__);
  }
  
  staggered_dslash_mg4dir(tmp, fatlink, longlink, ghost_fatlink, ghost_longlink,
			  in, otherparity, dagger_bit, sPrecision, gPrecision);

  staggered_dslash_mg4dir(out, fatlink, longlink, ghost_fatlink, ghost_longlink,
			  tmp, parity, dagger_bit, sPrecision, gPrecision);
  
  double msq_x4 = mass*mass*4;
  if (sPrecision == QUDA_DOUBLE_PRECISION){
    axmy((double*)in->V(), (double)msq_x4, (double*)out->V(), Vh*mySpinorSiteSize);
  }else{
    axmy((float*)in->V(), (float)msq_x4, (float*)out->V(), Vh*mySpinorSiteSize);    
  }

}

#endif

