#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <host_utils.h>
#include <quda_internal.h>
#include <quda.h>
#include <util_quda.h>
#include <staggered_dslash_reference.h>
#include <command_line_params.h>
#include "misc.h"
#include <blas_quda.h>

#include <blas_reference.h>

extern void *memset(void *s, int c, size_t n);

#include <dslash_util.h>

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

// dslashReference()
//
// if oddBit is zero: calculate even parity spinor elements (using odd parity spinor)
// if oddBit is one:  calculate odd parity spinor elements
// if daggerBit is zero: perform ordinary dslash operator
// if daggerBit is one:  perform hermitian conjugate of dslash
template <typename sFloat, typename gFloat>
void dslashReference(sFloat *res, gFloat **fatlink, gFloat **longlink, gFloat **ghostFatlink, gFloat **ghostLonglink,
    sFloat *spinorField, sFloat **fwd_nbr_spinor, sFloat **back_nbr_spinor, int oddBit, int daggerBit, int nSrc,
    QudaDslashType dslash_type)
{
  for (int i=0; i<Vh*my_spinor_site_size*nSrc; i++) res[i] = 0.0;

  gFloat *fatlinkEven[4], *fatlinkOdd[4];
  gFloat *longlinkEven[4], *longlinkOdd[4];

#ifdef MULTI_GPU
  gFloat *ghostFatlinkEven[4], *ghostFatlinkOdd[4];
  gFloat *ghostLonglinkEven[4], *ghostLonglinkOdd[4];
#endif

  for (int dir = 0; dir < 4; dir++) {
    fatlinkEven[dir] = fatlink[dir];
    fatlinkOdd[dir] = fatlink[dir] + Vh*gauge_site_size;
    longlinkEven[dir] =longlink[dir];
    longlinkOdd[dir] = longlink[dir] + Vh*gauge_site_size;

#ifdef MULTI_GPU
    ghostFatlinkEven[dir] = ghostFatlink[dir];
    ghostFatlinkOdd[dir] = ghostFatlink[dir] + (faceVolume[dir]/2)*gauge_site_size;
    ghostLonglinkEven[dir] = ghostLonglink[dir];
    ghostLonglinkOdd[dir] = ghostLonglink[dir] + 3*(faceVolume[dir]/2)*gauge_site_size;
#endif
  }

  for (int xs=0; xs<nSrc; xs++) {

    for (int i = 0; i < Vh; i++) {
      int sid = i + xs*Vh;
      int offset = my_spinor_site_size*sid;

      for (int dir = 0; dir < 8; dir++) {
#ifdef MULTI_GPU
        const int nFace = dslash_type == QUDA_ASQTAD_DSLASH ? 3 : 1;
        gFloat* fatlnk = gaugeLink_mg4dir(i, dir, oddBit, fatlinkEven, fatlinkOdd, ghostFatlinkEven, ghostFatlinkOdd, 1, 1);
        gFloat *longlnk = dslash_type == QUDA_ASQTAD_DSLASH ?
            gaugeLink_mg4dir(i, dir, oddBit, longlinkEven, longlinkOdd, ghostLonglinkEven, ghostLonglinkOdd, 3, 3) :
            nullptr;
        sFloat *first_neighbor_spinor = spinorNeighbor_5d_mgpu<QUDA_4D_PC>(
            sid, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 1, nFace, my_spinor_site_size);
        sFloat *third_neighbor_spinor = dslash_type == QUDA_ASQTAD_DSLASH ?
            spinorNeighbor_5d_mgpu<QUDA_4D_PC>(
                sid, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 3, nFace, my_spinor_site_size) :
            nullptr;
#else
        gFloat *fatlnk = gaugeLink(i, dir, oddBit, fatlinkEven, fatlinkOdd, 1);
        gFloat *longlnk
            = dslash_type == QUDA_ASQTAD_DSLASH ? gaugeLink(i, dir, oddBit, longlinkEven, longlinkOdd, 3) : nullptr;
        sFloat *first_neighbor_spinor = spinorNeighbor_5d<QUDA_4D_PC>(sid, dir, oddBit, spinorField, 1, my_spinor_site_size);
        sFloat *third_neighbor_spinor = dslash_type == QUDA_ASQTAD_DSLASH ?
            spinorNeighbor_5d<QUDA_4D_PC>(sid, dir, oddBit, spinorField, 3, my_spinor_site_size) :
            nullptr;
#endif
        sFloat gaugedSpinor[my_spinor_site_size];

        if (dir % 2 == 0){
          su3Mul(gaugedSpinor, fatlnk, first_neighbor_spinor);
          sum(&res[offset], &res[offset], gaugedSpinor, my_spinor_site_size);

          if (dslash_type == QUDA_ASQTAD_DSLASH) {
            su3Mul(gaugedSpinor, longlnk, third_neighbor_spinor);
            sum(&res[offset], &res[offset], gaugedSpinor, my_spinor_site_size);
          }
        } else {
          su3Tmul(gaugedSpinor, fatlnk, first_neighbor_spinor);
          if (dslash_type == QUDA_LAPLACE_DSLASH) {
            sum(&res[offset], &res[offset], gaugedSpinor, my_spinor_site_size);
          } else {
            sub(&res[offset], &res[offset], gaugedSpinor, my_spinor_site_size);
          }

          if (dslash_type == QUDA_ASQTAD_DSLASH) {
            su3Tmul(gaugedSpinor, longlnk, third_neighbor_spinor);
            sub(&res[offset], &res[offset], gaugedSpinor, my_spinor_site_size);
          }
        }
      }

      if (daggerBit) negx(&res[offset], my_spinor_site_size);
    } // 4-d volume
  } // right-hand-side  
}

void staggered_dslash(ColorSpinorField *out, void **fatlink, void **longlink, void **ghost_fatlink,
    void **ghost_longlink, ColorSpinorField *in, int oddBit, int daggerBit, QudaPrecision sPrecision,
    QudaPrecision gPrecision, QudaDslashType dslash_type)
{
  const int nSrc = in->X(4);
  
  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (oddBit == QUDA_EVEN_PARITY) {
    otherparity = QUDA_ODD_PARITY;
  } else if (oddBit == QUDA_ODD_PARITY) {
    otherparity = QUDA_EVEN_PARITY;
  } else {
    errorQuda("ERROR: full parity not supported in function %s", __FUNCTION__);
  }
  const int nFace = dslash_type == QUDA_ASQTAD_DSLASH ? 3 : 1;

  in->exchangeGhost(otherparity, nFace, daggerBit);

  void** fwd_nbr_spinor = ((cpuColorSpinorField*)in)->fwdGhostFaceBuffer;
  void** back_nbr_spinor = ((cpuColorSpinorField*)in)->backGhostFaceBuffer;

  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      dslashReference((double *)out->V(), (double **)fatlink, (double **)longlink, (double **)ghost_fatlink,
          (double **)ghost_longlink, (double *)in->V(), (double **)fwd_nbr_spinor, (double **)back_nbr_spinor, oddBit,
          daggerBit, nSrc, dslash_type);
    } else {
      dslashReference((double *)out->V(), (float **)fatlink, (float **)longlink, (float **)ghost_fatlink,
          (float **)ghost_longlink, (double *)in->V(), (double **)fwd_nbr_spinor, (double **)back_nbr_spinor, oddBit,
          daggerBit, nSrc, dslash_type);
      }
  } else {
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      dslashReference((float *)out->V(), (double **)fatlink, (double **)longlink, (double **)ghost_fatlink,
          (double **)ghost_longlink, (float *)in->V(), (float **)fwd_nbr_spinor, (float **)back_nbr_spinor, oddBit,
          daggerBit, nSrc, dslash_type);
    } else {
      dslashReference((float *)out->V(), (float **)fatlink, (float **)longlink, (float **)ghost_fatlink,
          (float **)ghost_longlink, (float *)in->V(), (float **)fwd_nbr_spinor, (float **)back_nbr_spinor, oddBit,
          daggerBit, nSrc, dslash_type);
    }
  }
}

void matdagmat(ColorSpinorField *out, void **fatlink, void **longlink, void **ghost_fatlink, void **ghost_longlink,
    ColorSpinorField *in, double mass, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision,
    ColorSpinorField *tmp, QudaParity parity, QudaDslashType dslash_type)
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

  staggered_dslash(tmp, fatlink, longlink, ghost_fatlink, ghost_longlink, in, otherparity, dagger_bit, sPrecision,
      gPrecision, dslash_type);

  staggered_dslash(out, fatlink, longlink, ghost_fatlink, ghost_longlink, tmp, parity, dagger_bit, sPrecision,
      gPrecision, dslash_type);

  double msq_x4 = mass*mass*4;
  if (sPrecision == QUDA_DOUBLE_PRECISION){
    axmy((double*)in->V(), (double)msq_x4, (double*)out->V(), out->X(4)*Vh*my_spinor_site_size);
  }else{
    axmy((float*)in->V(), (float)msq_x4, (float*)out->V(), out->X(4)*Vh*my_spinor_site_size);
  }
}

void verifyStaggeredInversion(ColorSpinorField *tmp, ColorSpinorField *ref, ColorSpinorField *in, ColorSpinorField *out, 
			      void *qdp_fatlink[], void *qdp_longlink[], void **ghost_fatlink, void **ghost_longlink, 
			      QudaGaugeParam &gauge_param, QudaInvertParam &inv_param) {
  switch (test_type) {
  case 0: // full parity solution
  case 1: // solving prec system, reconstructing
  case 2:
    
    // In QUDA, the full staggered operator has the sign convention
    // {{m, -D_eo},{-D_oe,m}}, while the CPU verify function does not
    // have the minus sign. Passing in QUDA_DAG_YES solves this
    // discrepancy.
    staggered_dslash(reinterpret_cast<quda::cpuColorSpinorField *>(&ref->Even()), qdp_fatlink, qdp_longlink, ghost_fatlink,
		     ghost_longlink, reinterpret_cast<quda::cpuColorSpinorField *>(&out->Odd()), QUDA_EVEN_PARITY,
		     QUDA_DAG_YES, inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);
    staggered_dslash(reinterpret_cast<quda::cpuColorSpinorField *>(&ref->Odd()), qdp_fatlink, qdp_longlink, ghost_fatlink,
		     ghost_longlink, reinterpret_cast<quda::cpuColorSpinorField *>(&out->Even()), QUDA_ODD_PARITY,
		     QUDA_DAG_YES, inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);
    
    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      xpay(out->V(), kappa, ref->V(), ref->Length(), gauge_param.cpu_prec);
      ax(0.5 / kappa, ref->V(), ref->Length(), gauge_param.cpu_prec);
    } else {
      axpy(2 * mass, out->V(), ref->V(), ref->Length(), gauge_param.cpu_prec);
    }
    break;
    
  case 3: // even
  case 4:
    
    matdagmat(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, out, mass, 0, inv_param.cpu_prec,
	      gauge_param.cpu_prec, tmp, test_type == 3 ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY, dslash_type);
    break;
    
  }
  
  int len = 0;
  if (solution_type == QUDA_MAT_SOLUTION || solution_type == QUDA_MATDAG_MAT_SOLUTION) {
    len = V;
  } else {
    len = Vh;
  }

  mxpy(in->V(), ref->V(), len * my_spinor_site_size, inv_param.cpu_prec);
  double nrm2 = norm_2(ref->V(), len * my_spinor_site_size, inv_param.cpu_prec);
  double src2 = norm_2(in->V(), len * my_spinor_site_size, inv_param.cpu_prec);
  
  double hqr = sqrt(blas::HeavyQuarkResidualNorm(*out, *ref).z);
  double l2r = sqrt(nrm2 / src2);
  
  printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, host = %g\n", 
	     inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq, hqr);
  
}
