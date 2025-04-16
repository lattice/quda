#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gauge_field.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include "instantiate_host.hpp"
#include "host_utils.h"
#include "index_utils.hpp"
#include "util_quda.h"
#include "staggered_dslash_reference.h"
#include "dslash_reference.h"
//#include "command_line_params.h"
//#include "misc.h"

/**
 * @brief Perform a staggered Dslash operation on a spinor field
 * @tparam real_t The data type of the fields (e.g., float or double)
 * @param[out] res The result spinor field
 * @param[in] fatlink The fat gauge links
 * @param[in] longlink The long gauge links (only used for ASQTAD Dslash)
 * @param[in] ghostFatlink The ghost fat gauge links (only used in multi-GPU mode)
 * @param[in] ghostLonglink The ghost long gauge links (only used in multi-GPU mode and for ASQTAD Dslash)
 * @param[in] spinorField The input spinor field
 * @param[in] fwd_nbr_spinor The forward neighbor spinor fields (only used in multi-GPU mode)
 * @param[in] back_nbr_spinor The backward neighbor spinor fields (only used in multi-GPU mode)
 * @param[in] oddBit The odd/even bit for the site index
 * @param[in] daggerBit Perform the ordinary dslash (0) or Hermitian conjugate (1)
 * @param[in] dslash_type The type of Dslash operation
 * @param[in] laplace3D Whether we applying the 3-d laplace operator
 * (in the case of dslash_type being QUDA_LAPLACE_DSLASH)
 */
template <typename real_t> struct StaggeredDslashReference {
  void operator()(void *res_, const void *const *fatlink_, const void *const *longlink_,
                  const void *const *ghostFatlink_, const void *const *ghostLonglink_, const void *spinorField_,
                  const void *const *fwd_nbr_spinor_, const void *const *back_nbr_spinor_, int oddBit, int daggerBit,
                  QudaDslashType dslash_type, int laplace3D)
  {
    if (laplace3D < 4 && dslash_type != QUDA_LAPLACE_DSLASH)
      errorQuda("laplace3D = %d only supported for Laplace dslash (%d requested)", laplace3D, dslash_type);

    auto res = reinterpret_cast<real_t *>(res_);
    auto fatlink = reinterpret_cast<const real_t *const *>(fatlink_);
    auto longlink = reinterpret_cast<const real_t *const *>(longlink_);
    auto ghostFatlink = reinterpret_cast<const real_t *const *>(ghostFatlink_);
    auto ghostLonglink = reinterpret_cast<const real_t *const *>(ghostLonglink_);
    auto spinorField = reinterpret_cast<const real_t *>(spinorField_);
    auto fwd_nbr_spinor = reinterpret_cast<const real_t *const *>(fwd_nbr_spinor_);
    auto back_nbr_spinor = reinterpret_cast<const real_t *const *>(back_nbr_spinor_);

#pragma omp parallel for
    for (auto i = 0lu; i < Vh * stag_spinor_site_size; i++) res[i] = 0.0;

    const real_t *fatlinkEven[4], *fatlinkOdd[4];
    const real_t *longlinkEven[4], *longlinkOdd[4];

    const real_t *ghostFatlinkEven[4] = {nullptr, nullptr, nullptr, nullptr};
    const real_t *ghostFatlinkOdd[4] = {nullptr, nullptr, nullptr, nullptr};
    const real_t *ghostLonglinkEven[4] = {nullptr, nullptr, nullptr, nullptr};
    const real_t *ghostLonglinkOdd[4] = {nullptr, nullptr, nullptr, nullptr};

    for (int dir = 0; dir < 4; dir++) {
      fatlinkEven[dir] = fatlink[dir];
      fatlinkOdd[dir] = fatlink[dir] + Vh * gauge_site_size;
      longlinkEven[dir] = longlink[dir];
      longlinkOdd[dir] = longlink[dir] + Vh * gauge_site_size;

      if (is_multi_gpu()) {
        ghostFatlinkEven[dir] = ghostFatlink[dir];
        ghostFatlinkOdd[dir] = ghostFatlink[dir] + (faceVolume[dir] / 2) * gauge_site_size;
        ghostLonglinkEven[dir] = ghostLonglink ? ghostLonglink[dir] : nullptr;
        ghostLonglinkOdd[dir]
          = ghostLonglink ? ghostLonglink[dir] + 3 * (faceVolume[dir] / 2) * gauge_site_size : nullptr;
      }
    }

#pragma omp parallel for
    for (int sid = 0; sid < Vh; sid++) {
      int offset = stag_spinor_site_size * sid;

      for (int dir = 0; dir < 8; dir++) {
        if (laplace3D == dir / 2) continue; // skip dimensions if needed
        const int nFace = dslash_type == QUDA_ASQTAD_DSLASH ? 3 : 1;
        const real_t *fatlnk
          = gaugeLink(sid, dir, oddBit, fatlinkEven, fatlinkOdd, ghostFatlinkEven, ghostFatlinkOdd, 1, 1);
        const real_t *longlnk = dslash_type == QUDA_ASQTAD_DSLASH ?
          gaugeLink(sid, dir, oddBit, longlinkEven, longlinkOdd, ghostLonglinkEven, ghostLonglinkOdd, 3, 3) :
          nullptr;
        const real_t *first_neighbor_spinor = spinorNeighbor(sid, dir, oddBit, spinorField, fwd_nbr_spinor,
                                                             back_nbr_spinor, 1, nFace, stag_spinor_site_size);
        const real_t *third_neighbor_spinor = dslash_type == QUDA_ASQTAD_DSLASH ?
          spinorNeighbor(sid, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 3, nFace, stag_spinor_site_size) :
          nullptr;

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
};

void stag_dslash(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
                 const ColorSpinorField &in, int oddBit, int daggerBit, QudaDslashType dslash_type, int laplace3D)
{
  // assert sPrecision and gPrecision must be the same
  if (in.Precision() != fat_link.Precision()) {
    errorQuda("The spinor precision and gauge precision are not the same");
  }

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

  const void *const ghost_fatlink[]
    = {fat_link.Ghost()[0].data(), fat_link.Ghost()[1].data(), fat_link.Ghost()[2].data(), fat_link.Ghost()[3].data()};
  const void *const ghost_longlink[] = {long_link.Ghost()[0].data(), long_link.Ghost()[1].data(),
                                        long_link.Ghost()[2].data(), long_link.Ghost()[3].data()};

  instantiate_host<StaggeredDslashReference>(in.Precision(), out.data<void *>(), fat_link.data_array<const void *>().data,
                                             long_link.data_array<const void *>().data, ghost_fatlink, ghost_longlink,
                                             in.data<void *>(), in.fwdGhostFaceBuffer, in.backGhostFaceBuffer, oddBit,
                                             daggerBit, dslash_type, laplace3D);
}

void stag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
              const ColorSpinorField &in, double mass, int daggerBit, QudaDslashType dslash_type, int laplace3D)
{
  checkPrecision(in, fat_link);

  // assert we have full-parity spinors
  if (out.SiteSubset() != QUDA_FULL_SITE_SUBSET || in.SiteSubset() != QUDA_FULL_SITE_SUBSET)
    errorQuda("Unexpected site subsets for stag_mat, out %d in %d", out.SiteSubset(), in.SiteSubset());

  // In QUDA, the full staggered operator has the sign convention
  // {{m, -D_eo},{-D_oe,m}}, while the CPU verify function does not
  // have the minus sign. Inverting the expected dagger convention
  // solves this discrepancy.
  stag_dslash(out.Even(), fat_link, long_link, in.Odd(), QUDA_EVEN_PARITY, 1 - daggerBit, dslash_type, laplace3D);
  stag_dslash(out.Odd(), fat_link, long_link, in.Even(), QUDA_ODD_PARITY, 1 - daggerBit, dslash_type, laplace3D);

  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    int dimension = laplace3D < 4 ? 3 : 4;
    double kappa = 1.0 / (2 * dimension + mass);
    xpay(in.data(), kappa, out.data(), out.Length(), out.Precision());
  } else {
    axpy(2 * mass, in.data(), out.data(), out.Length(), out.Precision());
  }
}

void stag_matdag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
                     const ColorSpinorField &in, double mass, int daggerBit, QudaDslashType dslash_type, int laplace3D)
{
  checkPrecision(in, fat_link);

  // assert we have full-parity spinors
  if (out.SiteSubset() != QUDA_FULL_SITE_SUBSET || in.SiteSubset() != QUDA_FULL_SITE_SUBSET)
    errorQuda("Unexpected site subsets for stag_matdagmat, out %d in %d", out.SiteSubset(), in.SiteSubset());

  // Create temporary spinors
  quda::ColorSpinorParam csParam(in);
  quda::ColorSpinorField tmp(csParam);

  // Apply mat in sequence
  stag_mat(tmp, fat_link, long_link, in, mass, daggerBit, dslash_type, laplace3D);
  stag_mat(out, fat_link, long_link, tmp, mass, 1 - daggerBit, dslash_type, laplace3D);
}

void stag_matpc(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
                const ColorSpinorField &in, double mass, int, QudaParity parity, QudaDslashType dslash_type,
                int laplace3D)
{
  if (laplace3D < 4) errorQuda("Cannot use 3-d operator with e/o preconditioning");
  checkPrecision(in, fat_link);

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
  stag_dslash(tmp, fat_link, long_link, in, otherparity, 0, dslash_type, laplace3D);
  stag_dslash(out, fat_link, long_link, tmp, parity, 0, dslash_type, laplace3D);

  double msq_x4 = mass * mass * 4;
  if (in.Precision() == QUDA_DOUBLE_PRECISION) {
    axmy(static_cast<double *>(in.data()), msq_x4, static_cast<double *>(out.data()), Vh * stag_spinor_site_size);
  } else {
    axmy(static_cast<float *>(in.data()), static_cast<float>(msq_x4), static_cast<float *>(out.data()),
         Vh * stag_spinor_site_size);
  }
}
