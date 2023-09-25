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

#include <dslash_reference.h>

template <typename Float> void display_link_internal(Float *link)
{
  int i, j;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) { printf("(%10f,%10f) \t", link[i * 3 * 2 + j * 2], link[i * 3 * 2 + j * 2 + 1]); }
    printf("\n");
  }
  printf("\n");
  return;
}

// staggeredDslashReferenece()
//
// if oddBit is zero: calculate even parity spinor elements (using odd parity spinor)
// if oddBit is one:  calculate odd parity spinor elements
// if daggerBit is zero: perform ordinary dslash operator
// if daggerBit is one:  perform hermitian conjugate of dslash
template <typename sFloat, typename gFloat>
#ifdef MULTI_GPU
void staggeredDslashReference(sFloat *res, gFloat **fatlink, gFloat **longlink, gFloat **ghostFatlink,
                              gFloat **ghostLonglink, sFloat *spinorField, sFloat **fwd_nbr_spinor,
                              sFloat **back_nbr_spinor, int oddBit, int daggerBit, QudaDslashType dslash_type)
#else
void staggeredDslashReference(sFloat *res, gFloat **fatlink, gFloat **longlink, gFloat **, gFloat **, sFloat *spinorField,
                              sFloat **, sFloat **, int oddBit, int daggerBit, QudaDslashType dslash_type)
#endif
{
  for (auto i = 0lu; i < Vh * stag_spinor_site_size; i++) res[i] = 0.0;

  gFloat *fatlinkEven[4], *fatlinkOdd[4];
  gFloat *longlinkEven[4], *longlinkOdd[4];

#ifdef MULTI_GPU
  gFloat *ghostFatlinkEven[4], *ghostFatlinkOdd[4];
  gFloat *ghostLonglinkEven[4], *ghostLonglinkOdd[4];
#endif

  for (int dir = 0; dir < 4; dir++) {
    fatlinkEven[dir] = fatlink[dir];
    fatlinkOdd[dir] = fatlink[dir] + Vh * gauge_site_size;
    longlinkEven[dir] = longlink[dir];
    longlinkOdd[dir] = longlink[dir] + Vh * gauge_site_size;

#ifdef MULTI_GPU
    ghostFatlinkEven[dir] = ghostFatlink[dir];
    ghostFatlinkOdd[dir] = ghostFatlink[dir] + (faceVolume[dir] / 2) * gauge_site_size;
    ghostLonglinkEven[dir] = ghostLonglink ? ghostLonglink[dir] : nullptr;
    ghostLonglinkOdd[dir] = ghostLonglink ? ghostLonglink[dir] + 3 * (faceVolume[dir] / 2) * gauge_site_size : nullptr;
#endif
  }

  for (int sid = 0; sid < Vh; sid++) {
    int offset = stag_spinor_site_size * sid;

    for (int dir = 0; dir < 8; dir++) {
#ifdef MULTI_GPU
      const int nFace = dslash_type == QUDA_ASQTAD_DSLASH ? 3 : 1;
      gFloat *fatlnk
        = gaugeLink_mg4dir(sid, dir, oddBit, fatlinkEven, fatlinkOdd, ghostFatlinkEven, ghostFatlinkOdd, 1, 1);
      gFloat *longlnk = dslash_type == QUDA_ASQTAD_DSLASH ?
        gaugeLink_mg4dir(sid, dir, oddBit, longlinkEven, longlinkOdd, ghostLonglinkEven, ghostLonglinkOdd, 3, 3) :
        nullptr;
      sFloat *first_neighbor_spinor = spinorNeighbor_5d_mgpu<QUDA_4D_PC>(
        sid, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 1, nFace, stag_spinor_site_size);
      sFloat *third_neighbor_spinor = dslash_type == QUDA_ASQTAD_DSLASH ?
        spinorNeighbor_5d_mgpu<QUDA_4D_PC>(sid, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 3, nFace,
                                           stag_spinor_site_size) :
        nullptr;
#else
      gFloat *fatlnk = gaugeLink(sid, dir, oddBit, fatlinkEven, fatlinkOdd, 1);
      gFloat *longlnk
        = dslash_type == QUDA_ASQTAD_DSLASH ? gaugeLink(sid, dir, oddBit, longlinkEven, longlinkOdd, 3) : nullptr;
      sFloat *first_neighbor_spinor
        = spinorNeighbor_5d<QUDA_4D_PC>(sid, dir, oddBit, spinorField, 1, stag_spinor_site_size);
      sFloat *third_neighbor_spinor = dslash_type == QUDA_ASQTAD_DSLASH ?
        spinorNeighbor_5d<QUDA_4D_PC>(sid, dir, oddBit, spinorField, 3, stag_spinor_site_size) :
        nullptr;
#endif
      sFloat gaugedSpinor[stag_spinor_site_size];

      if (dir % 2 == 0) {
        su3Mul(gaugedSpinor, fatlnk, first_neighbor_spinor);
        sum(&res[offset], &res[offset], gaugedSpinor, stag_spinor_site_size);

        if (dslash_type == QUDA_ASQTAD_DSLASH) {
          su3Mul(gaugedSpinor, longlnk, third_neighbor_spinor);
          sum(&res[offset], &res[offset], gaugedSpinor, stag_spinor_site_size);
        }
      } else {
        su3Tmul(gaugedSpinor, fatlnk, first_neighbor_spinor);
        if (dslash_type == QUDA_LAPLACE_DSLASH) {
          sum(&res[offset], &res[offset], gaugedSpinor, stag_spinor_site_size);
        } else {
          sub(&res[offset], &res[offset], gaugedSpinor, stag_spinor_site_size);
        }

        if (dslash_type == QUDA_ASQTAD_DSLASH) {
          su3Tmul(gaugedSpinor, longlnk, third_neighbor_spinor);
          sub(&res[offset], &res[offset], gaugedSpinor, stag_spinor_site_size);
        }
      }
    } // forward/backward in all four directions

    if (daggerBit) negx(&res[offset], stag_spinor_site_size);
  } // 4-d volume
}

void staggeredDslash(ColorSpinorField &out, void **fatlink, void **longlink, void **ghost_fatlink,
                     void **ghost_longlink, const ColorSpinorField &in, int oddBit, int daggerBit,
                     QudaPrecision sPrecision, QudaPrecision gPrecision, QudaDslashType dslash_type)
{
  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (oddBit == QUDA_EVEN_PARITY) {
    otherparity = QUDA_ODD_PARITY;
  } else if (oddBit == QUDA_ODD_PARITY) {
    otherparity = QUDA_EVEN_PARITY;
  } else {
    errorQuda("ERROR: full parity not supported");
  }
  const int nFace = dslash_type == QUDA_ASQTAD_DSLASH ? 3 : 1;

  in.exchangeGhost(otherparity, nFace, daggerBit);

  void **fwd_nbr_spinor = in.fwdGhostFaceBuffer;
  void **back_nbr_spinor = in.backGhostFaceBuffer;

  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      staggeredDslashReference((double *)out.V(), (double **)fatlink, (double **)longlink, (double **)ghost_fatlink,
                               (double **)ghost_longlink, (double *)in.V(), (double **)fwd_nbr_spinor,
                               (double **)back_nbr_spinor, oddBit, daggerBit, dslash_type);
    } else {
      staggeredDslashReference((double *)out.V(), (float **)fatlink, (float **)longlink, (float **)ghost_fatlink,
                               (float **)ghost_longlink, (double *)in.V(), (double **)fwd_nbr_spinor,
                               (double **)back_nbr_spinor, oddBit, daggerBit, dslash_type);
    }
  } else {
    if (gPrecision == QUDA_DOUBLE_PRECISION) {
      staggeredDslashReference((float *)out.V(), (double **)fatlink, (double **)longlink, (double **)ghost_fatlink,
                               (double **)ghost_longlink, (float *)in.V(), (float **)fwd_nbr_spinor,
                               (float **)back_nbr_spinor, oddBit, daggerBit, dslash_type);
    } else {
      staggeredDslashReference((float *)out.V(), (float **)fatlink, (float **)longlink, (float **)ghost_fatlink,
                               (float **)ghost_longlink, (float *)in.V(), (float **)fwd_nbr_spinor,
                               (float **)back_nbr_spinor, oddBit, daggerBit, dslash_type);
    }
  }
}

void staggeredMatDagMat(ColorSpinorField &out, void **fatlink, void **longlink, void **ghost_fatlink,
                        void **ghost_longlink, const ColorSpinorField &in, double mass, int dagger_bit,
                        QudaPrecision sPrecision, QudaPrecision gPrecision, ColorSpinorField &tmp, QudaParity parity,
                        QudaDslashType dslash_type)
{
  // assert sPrecision and gPrecision must be the same
  if (sPrecision != gPrecision) { errorQuda("Spinor precision and gPrecison is not the same"); }

  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (parity == QUDA_EVEN_PARITY) {
    otherparity = QUDA_ODD_PARITY;
  } else if (parity == QUDA_ODD_PARITY) {
    otherparity = QUDA_EVEN_PARITY;
  } else {
    errorQuda("full parity not supported in function");
  }

  staggeredDslash(tmp, fatlink, longlink, ghost_fatlink, ghost_longlink, in, otherparity, dagger_bit, sPrecision,
                  gPrecision, dslash_type);

  staggeredDslash(out, fatlink, longlink, ghost_fatlink, ghost_longlink, tmp, parity, dagger_bit, sPrecision,
                  gPrecision, dslash_type);

  double msq_x4 = mass * mass * 4;
  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    axmy((double *)in.V(), (double)msq_x4, (double *)out.V(), Vh * stag_spinor_site_size);
  } else {
    axmy((float *)in.V(), (float)msq_x4, (float *)out.V(), Vh * stag_spinor_site_size);
  }
}
