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

/**
  * @brief Base host routine to apply the even-odd or odd-even component of a staggered-type dslash
  *
  * @tparam real_t Datatype used in the host dslash
  * @param res Host output result
  * @param fatlink Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
  * @param longlink Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
  * @param ghostFatlink Ghost zones for the host fat links
  * @param ghostLonglink Ghost zones for the host long links
  * @param spinorField Host input spinor
  * @param fwd_nbr_spinor Forward ghost zones for the host input spinor
  * @param back_nbr_spinor Backwards ghost zones for the host input spinor
  * @param oddBit 0 for D_eo [calculate even parity spinor elements (using odd parity spinor)], 1 for D_oe [calculate odd parity spinor elements]
  * @param daggerBit 0 for the regular operator, 1 for the Hermitian conjugate of dslash
  * @param dslash_type Dslash type
  * @param comm_override Override array used for local operators, elements are "1" for comms as usual, "0" to use PBC instead, default all 1
  */
template <typename real_t>
void staggeredDslashReference(real_t *res, real_t **fatlink, real_t **longlink, real_t **ghostFatlink,
                              real_t **ghostLonglink, real_t *spinorField, real_t **fwd_nbr_spinor,
                              real_t **back_nbr_spinor, int oddBit, int daggerBit, QudaDslashType dslash_type,
                              std::array<int, 4> comm_override = { 1, 1, 1, 1 })
{
#pragma omp parallel for
  for (auto i = 0lu; i < Vh * stag_spinor_site_size; i++) res[i] = 0.0;

  real_t *fatlinkEven[4] = { nullptr, nullptr, nullptr, nullptr };
  real_t *fatlinkOdd[4] = { nullptr, nullptr, nullptr, nullptr };
  real_t *longlinkEven[4] = { nullptr, nullptr, nullptr, nullptr };
  real_t *longlinkOdd[4] = { nullptr, nullptr, nullptr, nullptr };

  real_t *ghostFatlinkEven[4] = { nullptr, nullptr, nullptr, nullptr };
  real_t *ghostFatlinkOdd[4] = { nullptr, nullptr, nullptr, nullptr };
  real_t *ghostLonglinkEven[4] = { nullptr, nullptr, nullptr, nullptr };
  real_t *ghostLonglinkOdd[4] = { nullptr, nullptr, nullptr, nullptr };

  for (int dir = 0; dir < 4; dir++) {
    fatlinkEven[dir] = fatlink[dir];
    fatlinkOdd[dir] = fatlink[dir] + Vh * gauge_site_size;
    longlinkEven[dir] = longlink[dir];
    longlinkOdd[dir] = longlink[dir] + Vh * gauge_site_size;

    if (is_multi_gpu_build()) {
      ghostFatlinkEven[dir] = ghostFatlink[dir];
      ghostFatlinkOdd[dir] = ghostFatlink[dir] + (faceVolume[dir] / 2) * gauge_site_size;
      ghostLonglinkEven[dir] = ghostLonglink ? ghostLonglink[dir] : nullptr;
      ghostLonglinkOdd[dir] = ghostLonglink ? ghostLonglink[dir] + 3 * (faceVolume[dir] / 2) * gauge_site_size : nullptr;
    }
  }

#pragma omp parallel for
  for (int sid = 0; sid < Vh; sid++) {
    int offset = stag_spinor_site_size * sid;

    for (int dir = 0; dir < 8; dir++) {
      real_t fatlnk[gauge_site_size];
      real_t longlnk[gauge_site_size];
      real_t first_neighbor_spinor[stag_spinor_site_size];
      real_t third_neighbor_spinor[stag_spinor_site_size];
      if constexpr (is_multi_gpu_build())
      {
        const int nFace = dslash_type == QUDA_ASQTAD_DSLASH ? 3 : 1;
        memcpy(fatlnk, gaugeLink_mg4dir(sid, dir, oddBit, fatlinkEven, fatlinkOdd, ghostFatlinkEven, ghostFatlinkOdd, 1, 1, comm_override), sizeof(real_t) * gauge_site_size);
        memcpy(first_neighbor_spinor,
               spinorNeighbor_5d_mgpu<QUDA_4D_PC>(sid, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 1, nFace, stag_spinor_site_size, comm_override),
               sizeof(real_t) * stag_spinor_site_size);
        if (dslash_type == QUDA_ASQTAD_DSLASH) {
          memcpy(longlnk, gaugeLink_mg4dir(sid, dir, oddBit, longlinkEven, longlinkOdd, ghostLonglinkEven, ghostLonglinkOdd, 3, 3, comm_override), sizeof(real_t) * gauge_site_size);
          memcpy(third_neighbor_spinor,
                 spinorNeighbor_5d_mgpu<QUDA_4D_PC>(sid, dir, oddBit, spinorField, fwd_nbr_spinor, back_nbr_spinor, 3, nFace, stag_spinor_site_size, comm_override), sizeof(real_t) * stag_spinor_site_size);
        } else {
          memset(longlnk, 0, sizeof(real_t) * gauge_site_size);
          memset(third_neighbor_spinor, 0, sizeof(real_t) * stag_spinor_site_size);
        }  
      } else {
        memcpy(fatlnk, gaugeLink(sid, dir, oddBit, fatlinkEven, fatlinkOdd, 1), sizeof(real_t) * gauge_site_size);
        memcpy(first_neighbor_spinor, spinorNeighbor_5d<QUDA_4D_PC>(sid, dir, oddBit, spinorField, 1, stag_spinor_site_size), sizeof(real_t) * stag_spinor_site_size);
        if (dslash_type == QUDA_ASQTAD_DSLASH) {
          memcpy(longlnk, gaugeLink(sid, dir, oddBit, longlinkEven, longlinkOdd, 3), sizeof(real_t) * gauge_site_size);
          memcpy(third_neighbor_spinor, spinorNeighbor_5d<QUDA_4D_PC>(sid, dir, oddBit, spinorField, 3, stag_spinor_site_size), sizeof(real_t) * stag_spinor_site_size);
        } else {
          memset(longlnk, 0, sizeof(real_t) * gauge_site_size);
          memset(third_neighbor_spinor, 0, sizeof(real_t) * stag_spinor_site_size);
        }
      }
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
                 const ColorSpinorField &in, int oddBit, int daggerBit, QudaDslashType dslash_type, std::array<int, 4> comm_override)
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

  auto fwd_nbr_spinor = in.fwdGhostFaceBuffer;
  auto back_nbr_spinor = in.backGhostFaceBuffer;

  void *qdp_fatlink[4] = {fat_link.data(0), fat_link.data(1), fat_link.data(2), fat_link.data(3)};
  void *qdp_longlink[4] = {long_link.data(0), long_link.data(1), long_link.data(2), long_link.data(3)};

  void *ghost_fatlink[4] = { nullptr, nullptr, nullptr, nullptr };
  void *ghost_longlink[4] = { nullptr, nullptr, nullptr, nullptr };

  for (int d = 0; d < 4; d++) {
    if (quda::comm_dim_partitioned(d) && comm_override[d]) {
      ghost_fatlink[d] = fat_link.Ghost()[d].data();
      if (dslash_type == QUDA_ASQTAD_DSLASH)
        ghost_longlink[d] = long_link.Ghost()[d].data();
    }
  }

  if (in.Precision() == QUDA_DOUBLE_PRECISION) {
    // note: qdp_fatlink and qdp_longlink, etc, can be replaced with feature/openmp's raw_pointer
    staggeredDslashReference((double *)out.data(), (double **)qdp_fatlink, (double **)qdp_longlink,
                             (double**)ghost_fatlink, (double**)ghost_longlink,
                             (double *)in.data(), (double **)fwd_nbr_spinor,
                             (double **)back_nbr_spinor, oddBit, daggerBit, dslash_type, comm_override);
  } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
    staggeredDslashReference((float *)out.data(), (float **)qdp_fatlink, (float **)qdp_longlink,
                             (float**)ghost_fatlink, (float**)ghost_longlink,
                             (float *)in.data(), (float **)fwd_nbr_spinor,
                             (float **)back_nbr_spinor, oddBit, daggerBit, dslash_type, comm_override);
  }
}

void stag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
              const ColorSpinorField &in, double mass, int daggerBit, QudaDslashType dslash_type, std::array<int, 4> comm_override)
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
  stag_dslash(out.Even(), fat_link, long_link, in.Odd(), QUDA_EVEN_PARITY, 1 - daggerBit, dslash_type, comm_override);
  stag_dslash(out.Odd(), fat_link, long_link, in.Even(), QUDA_ODD_PARITY, 1 - daggerBit, dslash_type, comm_override);

  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    double kappa = 1.0 / (8 + mass);
    xpay((void*)in.data(), kappa, out.data(), out.Length(), out.Precision());
  } else {
    axpy(2 * mass, (void*)in.data(), out.data(), out.Length(), out.Precision());
  }
}

void stag_matdag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
              const ColorSpinorField &in, double mass, int daggerBit, QudaDslashType dslash_type, std::array<int, 4> comm_override)
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
  stag_mat(tmp, fat_link, long_link, in, mass, daggerBit, dslash_type, comm_override);
  stag_mat(out, fat_link, long_link, tmp, mass, 1 - daggerBit, dslash_type, comm_override);
}

void stag_matpc(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in, double mass, int,
                QudaParity parity, QudaDslashType dslash_type, std::array<int, 4> comm_override)
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
  stag_dslash(tmp, fat_link, long_link, in, otherparity, 0, dslash_type, comm_override);
  stag_dslash(out, fat_link, long_link, tmp, parity, 0, dslash_type, comm_override);

  double msq_x4 = mass * mass * 4;
  if (in.Precision() == QUDA_DOUBLE_PRECISION) {
    axmy((double *)in.data(), (double)msq_x4, (double *)out.data(), Vh * stag_spinor_site_size);
  } else {
    axmy((float *)in.data(), (float)msq_x4, (float *)out.data(), Vh * stag_spinor_site_size);
  }
}

void stag_mat_local(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
              double mass, int daggerBit, QudaDslashType dslash_type) {
  // Construct R
  // We round up to the nearest even number to avoid odd_bit woes
  int face_depth = (dslash_type == QUDA_ASQTAD_DSLASH) ? 4 : 2;
  int R[4] = { face_depth * comm_dim_partitioned(0), face_depth * comm_dim_partitioned(1),
               face_depth * comm_dim_partitioned(2), face_depth * comm_dim_partitioned(3) };

  // We need to "hack" the proper values for Vh, etc, in
  int padded_V = 1;
  int W[4];
  for (int d = 0; d < 4; d++) {
    W[d] = Z[d] + 2 * R[d];
    padded_V *= Z[d] + 2 * R[d];
  }
  int padded_Vh = padded_V / 2;

  auto in_alias = static_cast<char*>(in.data());
  auto out_alias = static_cast<char*>(out.data());

  // Allocate an extended spinor, zero it out
  ColorSpinorParam csParam(in);
  for (int d = 0; d < 4; d++)
    csParam.x[d] = W[d];
  csParam.create = QUDA_ZERO_FIELD_CREATE;

  int precision = in.Precision();

  ColorSpinorField padded_in(csParam);
  ColorSpinorField padded_out(csParam);
  ColorSpinorField padded_tmp(csParam);

  auto padded_in_alias = static_cast<char*>(padded_in.data());
  auto padded_out_alias = static_cast<char*>(padded_out.data());

  // Inject the input spinor into the padded spinor
#pragma omp parallel for
  for (int x_cb = 0; x_cb < Vh; x_cb++) {
    for (int parity = 0; parity < 2; parity++) {
      int x_padded_full = get_padded_coord(x_cb, parity, Z, W, R);
      memcpy(&padded_in_alias[stag_spinor_site_size * precision * x_padded_full],
             &in_alias[stag_spinor_site_size * precision * (x_cb + Vh * parity)], stag_spinor_site_size * precision);
    }
  }

  // Backup V, etc, variables; restore them later
  int V_old = V;   V = padded_V;
  int Vh_old = Vh; Vh = padded_Vh;
  int Z_old[4];
  for (int d = 0; d < 4; d++) {
    Z_old[d] = Z[d];
    Z[d] = W[d];
  }

  // Apply the staggered operator
  stag_mat(padded_out, fat_link, long_link, padded_in, mass, daggerBit, dslash_type, {0, 0, 0, 0});

  // Restore V, etc
  V = V_old; Vh = Vh_old;
  for (int d = 0; d < 4; d++) { Z[d] = Z_old[d]; }

  // Extract the padded output spinor, place it into the proper output spinor
#pragma omp parallel for
  for (int x_cb = 0; x_cb < Vh; x_cb++) {
    for (int parity = 0; parity < 2; parity++) {
      int x_padded_full = get_padded_coord(x_cb, parity, Z, W, R);
      memcpy(&out_alias[stag_spinor_site_size * precision * (x_cb + Vh * parity)],
             &padded_out_alias[stag_spinor_site_size * precision * x_padded_full], stag_spinor_site_size * precision);
    }
  }
}


void stag_matdag_mat_local(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
              double mass, int daggerBit, QudaDslashType dslash_type) {
  // Construct R
  // We round up to the nearest even number to avoid odd_bit woes
  int face_depth = (dslash_type == QUDA_ASQTAD_DSLASH) ? 4 : 2;
  int R[4] = { face_depth * comm_dim_partitioned(0), face_depth * comm_dim_partitioned(1),
               face_depth * comm_dim_partitioned(2), face_depth * comm_dim_partitioned(3) };

  // We need to "hack" the proper values for Vh, etc, in
  int padded_V = 1;
  int W[4];
  for (int d = 0; d < 4; d++) {
    W[d] = Z[d] + 2 * R[d];
    padded_V *= Z[d] + 2 * R[d];
  }
  int padded_Vh = padded_V / 2;

  auto in_alias = static_cast<char*>(in.data());
  auto out_alias = static_cast<char*>(out.data());

  // Allocate an extended spinor, zero it out
  ColorSpinorParam csParam(in);
  for (int d = 0; d < 4; d++)
    csParam.x[d] = W[d];
  csParam.create = QUDA_ZERO_FIELD_CREATE;

  int precision = in.Precision();

  ColorSpinorField padded_in(csParam);
  ColorSpinorField padded_out(csParam);
  ColorSpinorField padded_tmp(csParam);

  auto padded_in_alias = static_cast<char*>(padded_in.data());
  auto padded_out_alias = static_cast<char*>(padded_out.data());

  // Inject the input spinor into the padded spinor
#pragma omp parallel for
  for (int x_cb = 0; x_cb < Vh; x_cb++) {
    for (int parity = 0; parity < 2; parity++) {
      int x_padded_full = get_padded_coord(x_cb, parity, Z, W, R);
      memcpy(&padded_in_alias[stag_spinor_site_size * precision * x_padded_full],
             &in_alias[stag_spinor_site_size * precision * (x_cb + Vh * parity)], stag_spinor_site_size * precision);
    }
  }

  // Backup V, etc, variables; restore them later
  int V_old = V;   V = padded_V;
  int Vh_old = Vh; Vh = padded_Vh;
  int Z_old[4];
  for (int d = 0; d < 4; d++) {
    Z_old[d] = Z[d];
    Z[d] = W[d];
  }

  // Apply the staggered operator
  stag_matdag_mat(padded_out, fat_link, long_link, padded_in, mass, daggerBit, dslash_type, {0, 0, 0, 0});

  // Restore V, etc
  V = V_old; Vh = Vh_old;
  for (int d = 0; d < 4; d++) { Z[d] = Z_old[d]; }

  // Extract the padded output spinor, place it into the proper output spinor
#pragma omp parallel for
  for (int x_cb = 0; x_cb < Vh; x_cb++) {
    for (int parity = 0; parity < 2; parity++) {
      int x_padded_full = get_padded_coord(x_cb, parity, Z, W, R);
      memcpy(&out_alias[stag_spinor_site_size * precision * (x_cb + Vh * parity)],
             &padded_out_alias[stag_spinor_site_size * precision * x_padded_full], stag_spinor_site_size * precision);
    }
  }
}

void stag_matpc_local(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
                double mass, int dagger_bit, QudaParity parity, QudaDslashType dslash_type) {
  // Construct R
  // We round up to the nearest even number to avoid odd_bit woes
  int face_depth = (dslash_type == QUDA_ASQTAD_DSLASH) ? 4 : 2;
  int R[4] = { face_depth * comm_dim_partitioned(0), face_depth * comm_dim_partitioned(1),
               face_depth * comm_dim_partitioned(2), face_depth * comm_dim_partitioned(3) };

  // We need to "hack" the proper values for Vh, etc, in
  int padded_V = 1;
  int W[4];
  for (int d = 0; d < 4; d++) {
    W[d] = Z[d] + 2 * R[d];
    padded_V *= Z[d] + 2 * R[d];
  }
  int padded_Vh = padded_V / 2;

  auto in_alias = static_cast<char*>(in.data());
  auto out_alias = static_cast<char*>(out.data());

  // Allocate an extended spinor, zero it out
  ColorSpinorParam csParam(in);
  for (int d = 0; d < 4; d++)
    csParam.x[d] = W[d];
  csParam.x[0] /= 2;
  csParam.create = QUDA_ZERO_FIELD_CREATE;

  int precision = in.Precision();

  ColorSpinorField padded_in(csParam);
  ColorSpinorField padded_out(csParam);
  ColorSpinorField padded_tmp(csParam);

  auto padded_in_alias = static_cast<char*>(padded_in.data());
  auto padded_out_alias = static_cast<char*>(padded_out.data());

  // Inject the input spinor into the padded spinor
#pragma omp parallel for
  for (int x_cb = 0; x_cb < Vh; x_cb++) {
    int x_padded_cb = get_padded_x_cb(x_cb, parity, Z, W, R);
    memcpy(&padded_in_alias[stag_spinor_site_size * precision * x_padded_cb],
            &in_alias[stag_spinor_site_size * precision * x_cb], stag_spinor_site_size * precision);
  }

  // Backup V, etc, variables; restore them later
  int V_old = V;   V = padded_V;
  int Vh_old = Vh; Vh = padded_Vh;
  int Z_old[4];
  for (int d = 0; d < 4; d++) {
    Z_old[d] = Z[d];
    Z[d] = W[d];
  }

  // Apply the staggered operator
  stag_matpc(padded_out, fat_link, long_link, padded_in, mass, dagger_bit, parity, dslash_type, {0, 0, 0, 0});

  // Restore V, etc
  V = V_old; Vh = Vh_old;
  for (int d = 0; d < 4; d++) { Z[d] = Z_old[d]; }

  // Extract the padded output spinor, place it into the proper output spinor
#pragma omp parallel for
  for (int x_cb = 0; x_cb < Vh; x_cb++) {
    int x_padded_cb = get_padded_x_cb(x_cb, parity, Z, W, R);
    memcpy(&out_alias[stag_spinor_site_size * precision * x_cb],
                 &padded_out_alias[stag_spinor_site_size * precision * x_padded_cb], stag_spinor_site_size * precision);
  }
}
