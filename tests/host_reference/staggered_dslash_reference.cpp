#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <host_utils.h>
#include <quda_internal.h>
#include <quda.h>
#include <gauge_field.h>
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
template <typename sFloat, typename gFloat = sFloat>
void staggeredDslashReference(void *res_, void **fatlink_, void **longlink_, void **ghostFatlink_,
                              void **ghostLonglink_, void *spinorField_, void **fwd_nbr_spinor_,
                              void **back_nbr_spinor_, int oddBit, int daggerBit, QudaDslashType dslash_type,
                              bool use_ghost)
{
  auto res = reinterpret_cast<sFloat*>(res_);
  auto fatlink = reinterpret_cast<gFloat**>(fatlink_);
  auto longlink = reinterpret_cast<gFloat**>(longlink_);
  auto spinorField = reinterpret_cast<sFloat*>(spinorField_);
  auto ghostFatlink = reinterpret_cast<gFloat**>(ghostFatlink_);
  auto ghostLonglink = reinterpret_cast<gFloat**>(ghostLonglink_);
  auto fwd_nbr_spinor = reinterpret_cast<sFloat**>(fwd_nbr_spinor_);
  auto back_nbr_spinor = reinterpret_cast<sFloat**>(back_nbr_spinor_);

  for (auto i = 0lu; i < Vh * stag_spinor_site_size; i++) res[i] = 0.0;

  gFloat *fatlinkEven[4] = { nullptr, nullptr, nullptr, nullptr };
  gFloat *fatlinkOdd[4] = { nullptr, nullptr, nullptr, nullptr };
  gFloat *longlinkEven[4] = { nullptr, nullptr, nullptr, nullptr };
  gFloat *longlinkOdd[4] = { nullptr, nullptr, nullptr, nullptr };

  gFloat *ghostFatlinkEven[4] = { nullptr, nullptr, nullptr, nullptr };
  gFloat *ghostFatlinkOdd[4] = { nullptr, nullptr, nullptr, nullptr };
  gFloat *ghostLonglinkEven[4] = { nullptr, nullptr, nullptr, nullptr };
  gFloat *ghostLonglinkOdd[4] = { nullptr, nullptr, nullptr, nullptr };

  for (int dir = 0; dir < 4; dir++) {
    fatlinkEven[dir] = fatlink[dir];
    fatlinkOdd[dir] = fatlink[dir] + Vh * gauge_site_size;

    if (use_ghost && ghostFatlink != nullptr) {
      ghostFatlinkEven[dir] = ghostFatlink[dir];
      ghostFatlinkOdd[dir] = ghostFatlink[dir] + (faceVolume[dir] / 2) * gauge_site_size;
    } else {
      errorQuda("Expected a ghost buffer for fat links");
    }

    if (dslash_type == QUDA_ASQTAD_DSLASH) {
      longlinkEven[dir] = longlink[dir];
      longlinkOdd[dir] = longlink[dir] + Vh * gauge_site_size;

      if (use_ghost && ghostLonglink != nullptr) {
        ghostLonglinkEven[dir] = ghostLonglink[dir];
        ghostLonglinkOdd[dir] = ghostLonglink[dir] + 3 * (faceVolume[dir] / 2) * gauge_site_size;
      } else {
        errorQuda("Expected a ghost buffer for long links");
      }
    }
  }

  for (int sid = 0; sid < Vh; sid++) {
    int offset = stag_spinor_site_size * sid;

    for (int dir = 0; dir < 8; dir++) {
      gFloat *fatlnk = nullptr;
      gFloat *longlnk = nullptr;
      sFloat *first_neighbor_spinor = nullptr;
      sFloat *third_neighbor_spinor = nullptr;
      // true for multi-GPU runs except when testing the MatPCLocal operator
      if (use_ghost) {
        const int nFace = dslash_type == QUDA_ASQTAD_DSLASH ? 3 : 1;
        fatlnk = gaugeLink_mg4dir(sid, dir, oddBit, fatlinkEven, fatlinkOdd,
                                  ghostFatlinkEven, ghostFatlinkOdd, 1, 1);
        first_neighbor_spinor = spinorNeighbor_5d_mgpu<QUDA_4D_PC>(
          sid, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 1, nFace, stag_spinor_site_size);
        if (dslash_type == QUDA_ASQTAD_DSLASH) {
          longlnk =  gaugeLink_mg4dir(sid, dir, oddBit, longlinkEven, longlinkOdd,
                           ghostLonglinkEven, ghostLonglinkOdd, 3, 3);
          third_neighbor_spinor = spinorNeighbor_5d_mgpu<QUDA_4D_PC>(sid, dir, oddBit, spinorField,
                                  fwd_nbr_spinor, back_nbr_spinor, 3, nFace, stag_spinor_site_size);
        }
      } else {
        fatlnk = gaugeLink(sid, dir, oddBit, fatlinkEven, fatlinkOdd, 1);
        first_neighbor_spinor
          = spinorNeighbor_5d<QUDA_4D_PC>(sid, dir, oddBit, spinorField, 1, stag_spinor_site_size);
        if (dslash_type == QUDA_ASQTAD_DSLASH) {
          longlnk = gaugeLink(sid, dir, oddBit, longlinkEven, longlinkOdd, 3);
          third_neighbor_spinor = spinorNeighbor_5d<QUDA_4D_PC>(sid, dir, oddBit, spinorField,
                                  3, stag_spinor_site_size);
        }
      }
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
                     QudaPrecision sPrecision, QudaPrecision gPrecision, QudaDslashType dslash_type,
                     bool use_ghost)
{
  // assert sPrecision and gPrecision must be the same
  if (sPrecision != gPrecision) { errorQuda("Spinor precision and gPrecison is not the same"); }

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

  void *in_spinor = const_cast<void*>(in.V());
  void **fwd_nbr_spinor = in.fwdGhostFaceBuffer;
  void **back_nbr_spinor = in.backGhostFaceBuffer;

  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    staggeredDslashReference<double>(out.V(), fatlink, longlink, ghost_fatlink, ghost_longlink,
      in_spinor, fwd_nbr_spinor, back_nbr_spinor, oddBit, daggerBit, dslash_type, use_ghost);
  } else {
    staggeredDslashReference<float>(out.V(), fatlink, longlink, ghost_fatlink, ghost_longlink,
      in_spinor, fwd_nbr_spinor, back_nbr_spinor, oddBit, daggerBit, dslash_type, use_ghost);
  }
}

void staggeredMatDagMat(ColorSpinorField &out, void **fatlink, void **longlink, void **ghost_fatlink,
                        void **ghost_longlink, const ColorSpinorField &in, double mass, int dagger_bit,
                        QudaPrecision sPrecision, QudaPrecision gPrecision, ColorSpinorField &tmp, QudaParity parity,
                        QudaDslashType dslash_type, bool use_ghost)
{
  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (parity == QUDA_EVEN_PARITY) {
    otherparity = QUDA_ODD_PARITY;
  } else if (parity == QUDA_ODD_PARITY) {
    otherparity = QUDA_EVEN_PARITY;
  } else {
    errorQuda("full parity not supported in function");
  }

  staggeredDslash(tmp, fatlink, longlink, ghost_fatlink, ghost_longlink, in, otherparity, dagger_bit, sPrecision,
                  gPrecision, dslash_type, use_ghost);

  staggeredDslash(out, fatlink, longlink, ghost_fatlink, ghost_longlink, tmp, parity, dagger_bit, sPrecision,
                  gPrecision, dslash_type, use_ghost);

  double msq_x4 = mass * mass * 4;
  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    axmy((double *)in.V(), (double)msq_x4, (double *)out.V(), Vh * stag_spinor_site_size);
  } else {
    axmy((float *)in.V(), (float)msq_x4, (float *)out.V(), Vh * stag_spinor_site_size);
  }
}

// Versions of the above functions that take in cpuGaugeField references
void staggeredDslash(ColorSpinorField &out, cpuGaugeField *fatlink, cpuGaugeField *longlink,
                     const ColorSpinorField &in, int oddBit, int daggerBit, QudaDslashType dslash_type,
                     bool use_ghost) {

  if (dslash_type == QUDA_ASQTAD_DSLASH)
    staggeredDslash(out, (void**)fatlink->Gauge_p(), (void**)longlink->Gauge_p(), fatlink->Ghost(),
                    longlink->Ghost(), in, oddBit, daggerBit, out.Precision(), fatlink->Precision(),
                    dslash_type, use_ghost);
 else
    staggeredDslash(out, (void**)fatlink->Gauge_p(), nullptr, fatlink->Ghost(), nullptr, in, oddBit,
                    daggerBit, out.Precision(), fatlink->Precision(), dslash_type, use_ghost);
}

void staggeredMatDagMat(ColorSpinorField &out, cpuGaugeField *fatlink, cpuGaugeField *longlink,
                        const ColorSpinorField &in, double mass, int dagger_bit, ColorSpinorField &tmp,
                        QudaParity parity, QudaDslashType dslash_type, bool use_ghost) {
  if (dslash_type == QUDA_ASQTAD_DSLASH) 
    staggeredMatDagMat(out, (void**)fatlink->Gauge_p(), (void**)longlink->Gauge_p(), fatlink->Ghost(),
                       longlink->Ghost(), in, mass, dagger_bit, out.Precision(), fatlink->Precision(),
                       tmp, parity, dslash_type, use_ghost);
  else
    staggeredMatDagMat(out, (void**)fatlink->Gauge_p(), nullptr, fatlink->Ghost(), nullptr, in,
                       mass, dagger_bit, out.Precision(), fatlink->Precision(), tmp, parity,
                       dslash_type, use_ghost);
}

