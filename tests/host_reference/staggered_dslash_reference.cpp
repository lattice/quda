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
#include <gauge_field.h>

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
template <typename real_t>
#ifdef MULTI_GPU
void staggeredDslashReference(real_t *res, real_t **fatlink, real_t **longlink, real_t **ghostFatlink,
                              real_t **ghostLonglink, real_t *spinorField, real_t **fwd_nbr_spinor,
                              real_t **back_nbr_spinor, int oddBit, int daggerBit, QudaDslashType dslash_type)
#else
void staggeredDslashReference(real_t *res, real_t **fatlink, real_t **longlink, real_t **, real_t **, real_t *spinorField,
                              real_t **, real_t **, int oddBit, int daggerBit, QudaDslashType dslash_type)
#endif
{
#pragma omp parallel for
  for (auto i = 0lu; i < Vh * stag_spinor_site_size; i++) res[i] = 0.0;

  real_t *fatlinkEven[4], *fatlinkOdd[4];
  real_t *longlinkEven[4], *longlinkOdd[4];

#ifdef MULTI_GPU
  real_t *ghostFatlinkEven[4], *ghostFatlinkOdd[4];
  real_t *ghostLonglinkEven[4], *ghostLonglinkOdd[4];
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

#pragma omp parallel for
  for (int sid = 0; sid < Vh; sid++) {
    int offset = stag_spinor_site_size * sid;

    for (int dir = 0; dir < 8; dir++) {
#ifdef MULTI_GPU
      const int nFace = dslash_type == QUDA_ASQTAD_DSLASH ? 3 : 1;
      real_t *fatlnk
        = gaugeLink_mg4dir(sid, dir, oddBit, fatlinkEven, fatlinkOdd, ghostFatlinkEven, ghostFatlinkOdd, 1, 1);
      real_t *longlnk = dslash_type == QUDA_ASQTAD_DSLASH ?
        gaugeLink_mg4dir(sid, dir, oddBit, longlinkEven, longlinkOdd, ghostLonglinkEven, ghostLonglinkOdd, 3, 3) :
        nullptr;
      real_t *first_neighbor_spinor = spinorNeighbor_5d_mgpu<QUDA_4D_PC>(
        sid, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 1, nFace, stag_spinor_site_size);
      real_t *third_neighbor_spinor = dslash_type == QUDA_ASQTAD_DSLASH ?
        spinorNeighbor_5d_mgpu<QUDA_4D_PC>(sid, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 3, nFace,
                                           stag_spinor_site_size) :
        nullptr;
#else
      real_t *fatlnk = gaugeLink(sid, dir, oddBit, fatlinkEven, fatlinkOdd, 1);
      real_t *longlnk
        = dslash_type == QUDA_ASQTAD_DSLASH ? gaugeLink(sid, dir, oddBit, longlinkEven, longlinkOdd, 3) : nullptr;
      real_t *first_neighbor_spinor
        = spinorNeighbor_5d<QUDA_4D_PC>(sid, dir, oddBit, spinorField, 1, stag_spinor_site_size);
      real_t *third_neighbor_spinor = dslash_type == QUDA_ASQTAD_DSLASH ?
        spinorNeighbor_5d<QUDA_4D_PC>(sid, dir, oddBit, spinorField, 3, stag_spinor_site_size) :
        nullptr;
#endif
      real_t gaugedSpinor[stag_spinor_site_size];

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

void stag_dslash(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
                 const ColorSpinorField &in, int oddBit, int daggerBit, QudaDslashType dslash_type)
{
  // assert sPrecision and gPrecision must be the same
  if (in.Precision() != fat_link.Precision()) { errorQuda("The spinor precision and gauge precision are not the same"); }

  // assert we have single-parity spinors
  if (out.SiteSubset() != QUDA_PARITY_SITE_SUBSET || in.SiteSubset() != QUDA_PARITY_SITE_SUBSET)
    errorQuda("Unexpected site subsets for stag_dslash, out %d in %d", out.SiteSubset(), in.SiteSubset());

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

  void *qdp_fatlink[] = {fat_link.data(0), fat_link.data(1), fat_link.data(2), fat_link.data(3)};
  void *qdp_longlink[] = {long_link.data(0), long_link.data(1), long_link.data(2), long_link.data(3)};
  void *ghost_fatlink[]
    = {fat_link.Ghost()[0].data(), fat_link.Ghost()[1].data(), fat_link.Ghost()[2].data(), fat_link.Ghost()[3].data()};
  void *ghost_longlink[]
    = {long_link.Ghost()[0].data(), long_link.Ghost()[1].data(), long_link.Ghost()[2].data(), long_link.Ghost()[3].data()};

  if (in.Precision() == QUDA_DOUBLE_PRECISION) {
    staggeredDslashReference(static_cast<double*>(out.data()),
                             reinterpret_cast<double**>(qdp_fatlink),
                             reinterpret_cast<double**>(qdp_longlink),
                             reinterpret_cast<double**>(ghost_fatlink),
                             reinterpret_cast<double**>(ghost_longlink),
                             static_cast<double*>(in.data()),
                             reinterpret_cast<double**>(in.fwdGhostFaceBuffer),
                             reinterpret_cast<double**>(in.backGhostFaceBuffer), oddBit, daggerBit, dslash_type);
  } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
    staggeredDslashReference(static_cast<float*>(out.data()),
                             reinterpret_cast<float**>(qdp_fatlink),
                             reinterpret_cast<float**>(qdp_longlink),
                             reinterpret_cast<float**>(ghost_fatlink),
                             reinterpret_cast<float**>(ghost_longlink),
                             static_cast<float*>(in.data()),
                             reinterpret_cast<float**>(in.fwdGhostFaceBuffer),
                             reinterpret_cast<float**>(in.backGhostFaceBuffer), oddBit, daggerBit, dslash_type);
  }
}

void stag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
              const ColorSpinorField &in, double mass, int daggerBit, QudaDslashType dslash_type)
{
  // assert sPrecision and gPrecision must be the same
  if (in.Precision() != fat_link.Precision()) { errorQuda("The spinor precision and gauge precision are not the same"); }

  // assert we have full-parity spinors
  if (out.SiteSubset() != QUDA_FULL_SITE_SUBSET || in.SiteSubset() != QUDA_FULL_SITE_SUBSET)
    errorQuda("Unexpected site subsets for stag_mat, out %d in %d", out.SiteSubset(), in.SiteSubset());

  // In QUDA, the full staggered operator has the sign convention
  // {{m, -D_eo},{-D_oe,m}}, while the CPU verify function does not
  // have the minus sign. Inverting the expected dagger convention
  // solves this discrepancy.
  stag_dslash(out.Even(), fat_link, long_link, in.Odd(), QUDA_EVEN_PARITY, 1 - daggerBit, dslash_type);
  stag_dslash(out.Odd(), fat_link, long_link, in.Even(), QUDA_ODD_PARITY, 1 - daggerBit, dslash_type);

  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    double kappa = 1.0 / (8 + mass);
    xpay(in.data(), kappa, out.data(), out.Length(), out.Precision());
  } else {
    axpy(2 * mass, in.data(), out.data(), out.Length(), out.Precision());
  }
}

void stag_matdag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
              const ColorSpinorField &in, double mass, int daggerBit, QudaDslashType dslash_type)
{
  // assert sPrecision and gPrecision must be the same
  if (in.Precision() != fat_link.Precision()) { errorQuda("The spinor precision and gauge precision are not the same"); }

  // assert we have full-parity spinors
  if (out.SiteSubset() != QUDA_FULL_SITE_SUBSET || in.SiteSubset() != QUDA_FULL_SITE_SUBSET)
    errorQuda("Unexpected site subsets for stag_matdagmat, out %d in %d", out.SiteSubset(), in.SiteSubset());

  // Create temporary spinors
  quda::ColorSpinorParam csParam(in);
  quda::ColorSpinorField tmp(csParam);

  // Apply mat in sequence
  stag_mat(tmp, fat_link, long_link, in, mass, daggerBit, dslash_type);
  stag_mat(out, fat_link, long_link, tmp, mass, 1 - daggerBit, dslash_type);
}

void stag_matpc(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in, double mass, int,
                QudaParity parity, QudaDslashType dslash_type)
{
  // assert sPrecision and gPrecision must be the same
  if (in.Precision() != fat_link.Precision()) { errorQuda("The spinor precision and gauge precison are not the same"); }

  // assert we have single-parity spinors
  if (out.SiteSubset() != QUDA_PARITY_SITE_SUBSET || in.SiteSubset() != QUDA_PARITY_SITE_SUBSET)
    errorQuda("Unexpected site subsets for stag_matpc, out %d in %d", out.SiteSubset(), in.SiteSubset());

  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (parity == QUDA_EVEN_PARITY) {
    otherparity = QUDA_ODD_PARITY;
  } else if (parity == QUDA_ODD_PARITY) {
    otherparity = QUDA_EVEN_PARITY;
  } else {
    errorQuda("full parity not supported in function");
  }

  // Create temporary spinors
  quda::ColorSpinorParam csParam(in);
  quda::ColorSpinorField tmp(csParam);

  // dagger bit does not matter
  stag_dslash(tmp, fat_link, long_link, in, otherparity, 0, dslash_type);
  stag_dslash(out, fat_link, long_link, tmp, parity, 0, dslash_type);

  double msq_x4 = mass * mass * 4;
  if (in.Precision() == QUDA_DOUBLE_PRECISION) {
    axmy(static_cast<double*>(in.data()), msq_x4, static_cast<double*>(out.data()), Vh * stag_spinor_site_size);
  } else {
    axmy(static_cast<float*>(in.data()), static_cast<float>(msq_x4), static_cast<float*>(out.data()), Vh * stag_spinor_site_size);
  }
}
