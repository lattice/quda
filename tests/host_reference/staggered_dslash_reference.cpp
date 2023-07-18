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

    if (use_ghost) {
      if (ghostFatlink != nullptr) {
        ghostFatlinkEven[dir] = ghostFatlink[dir];
        ghostFatlinkOdd[dir] = ghostFatlink[dir] + (faceVolume[dir] / 2) * gauge_site_size;
      } else {
        errorQuda("Expected a ghost buffer for fat links");
      }
    }

    if (dslash_type == QUDA_ASQTAD_DSLASH) {
      longlinkEven[dir] = longlink[dir];
      longlinkOdd[dir] = longlink[dir] + Vh * gauge_site_size;

      if (use_ghost) {
        if (ghostLonglink != nullptr) {
          ghostLonglinkEven[dir] = ghostLonglink[dir];
          ghostLonglinkOdd[dir] = ghostLonglink[dir] + 3 * (faceVolume[dir] / 2) * gauge_site_size;
        } else {
          errorQuda("Expected a ghost buffer for long links");
        }
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

void staggeredMatDagMatLocal(ColorSpinorField &out, cpuGaugeField *fatlink, cpuGaugeField *longlink,
                        const ColorSpinorField &in, double mass, int dagger_bit,
                        QudaParity parity, QudaDslashType dslash_type) {
  // Construct R
  //int face_depth = (dslash_type == QUDA_ASQTAD_DSLASH) ? 3 : 1;
  // extended fields may be broken in the context of an odd number of directions being padded an odd distance?
  int face_depth = (dslash_type == QUDA_ASQTAD_DSLASH) ? 4 : 2;
  int R[4] = { face_depth * comm_dim_partitioned(0), face_depth * comm_dim_partitioned(1),
               face_depth * comm_dim_partitioned(2), face_depth * comm_dim_partitioned(3) };

  // compute a parity switch bit (only for odd R values...)
  //int odd_bit = (comm_dim_partitioned(0) + comm_dim_partitioned(1) + comm_dim_partitioned(2) + comm_dim_partitioned(3)) & 1;
  int odd_bit = 0;
  int parity_bit = (parity == QUDA_EVEN_PARITY) ? 0 : 1;
  int extended_parity_bit = parity_bit ^ odd_bit;

  // We need to "hack" the proper values for Vh, etc, in
  int padded_V = 1;
  int W[4];
  int Z_int[4];
  for (int d = 0; d < 4; d++) {
    W[d] = Z[d] + 2 * R[d];
    Z_int[d] = Z[d];
    padded_V *= Z[d] + 2 * R[d];
  }
  int padded_Vh = padded_V / 2;

  char *in_alias = (char*)in.V();
  char *out_alias = (char*)out.V();

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

  char *padded_in_alias = (char*)padded_in.V();
  char *padded_out_alias = (char*)padded_out.V();

  // Inject the input spinor into the padded spinor
  for (int t_c = 0; t_c < Z[3]; t_c++) {
    for (int z_c = 0; z_c < Z[2]; z_c++) {
      for (int y_c = 0; y_c < Z[1]; y_c++) {
        for (int xh_c = 0; xh_c < (Z[0] / 2); xh_c++) {
          // get the flat interior coordinate
          int x_c = 2 * xh_c;
          x_c += (parity_bit + y_c + z_c + t_c) & 1;
          int index_cb_4d = (((t_c * Z[2] + z_c) * Z[1] + y_c) * Z[0] + x_c) >> 1;

          // get the flat padded coordinate
          int x_padded_c = x_c + R[0];
          int y_padded_c = y_c + R[1];
          int z_padded_c = z_c + R[2];
          int t_padded_c = t_c + R[3];
          int index_padded_cb_4d = (((t_padded_c * W[2] + z_padded_c) * W[1] + y_padded_c) * W[0] + x_padded_c) >> 1;

          // copy data
          memcpy(&padded_in_alias[stag_spinor_site_size * precision * index_padded_cb_4d],
                 &in_alias[stag_spinor_site_size * precision * index_cb_4d], stag_spinor_site_size * precision);
        }
      }
    }
  }

  /*for (int index_cb_4d = 0; index_cb_4d < Vh; index_cb_4d++) {
    // calculate padded_index_cb_4d
    int x[4];
    coordinate_from_shrinked_index(x, index_cb_4d, Z_int, R, parity);
    int padded_index_cb_4d = index_4d_cb_from_coordinate_4d(x, W);
    // copy data
    memcpy(&padded_in_alias[stag_spinor_site_size * precision * padded_index_cb_4d],
           &in_alias[stag_spinor_site_size * precision * index_cb_4d], stag_spinor_site_size * precision);
  }*/

  // Backup V, etc, variables; restore them later
  int V_old = V;   V = padded_V;
  int Vh_old = Vh; Vh = padded_Vh;
  int Z_old[4];
  for (int d = 0; d < 4; d++) {
    Z_old[d] = Z[d];
    Z[d] = W[d];
  }

  // Apply the staggered operator
  staggeredMatDagMat(padded_out, fatlink, longlink, padded_in, mass, dagger_bit, padded_tmp,
                     extended_parity_bit ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY, dslash_type, false);

  // Restore V, etc
  V = V_old; Vh = Vh_old;
  for (int d = 0; d < 4; d++) { Z[d] = Z_old[d]; }

  // Extract the padded output spinor, place it into the proper output spinor
  for (int t_c = 0; t_c < Z[3]; t_c++) {
    for (int z_c = 0; z_c < Z[2]; z_c++) {
      for (int y_c = 0; y_c < Z[1]; y_c++) {
        for (int xh_c = 0; xh_c < (Z[0] / 2); xh_c++) {
          // get the flat interior coordinate
          int x_c = 2 * xh_c;
          x_c += (parity_bit + y_c + z_c + t_c) & 1;
          int index_cb_4d = (((t_c * Z[2] + z_c) * Z[1] + y_c) * Z[0] + x_c) >> 1;

          // get the flat padded coordinate
          int x_padded_c = x_c + R[0];
          int y_padded_c = y_c + R[1];
          int z_padded_c = z_c + R[2];
          int t_padded_c = t_c + R[3];
          int index_padded_cb_4d = (((t_padded_c * W[2] + z_padded_c) * W[1] + y_padded_c) * W[0] + x_padded_c) >> 1;

          // copy data
          memcpy(&out_alias[stag_spinor_site_size * precision * index_cb_4d],
                 &padded_out_alias[stag_spinor_site_size * precision * index_padded_cb_4d], stag_spinor_site_size * precision);
        }
      }
    }
  }

  /*for (int index_cb_4d = 0; index_cb_4d < Vh; index_cb_4d++) {
    // calculate padded_index_cb_4d
    int x[4];
    coordinate_from_shrinked_index(x, index_cb_4d, Z_int, R, parity);
    int padded_index_cb_4d = index_4d_cb_from_coordinate_4d(x, W);
    // copy data
    memcpy(&out_alias[stag_spinor_site_size * precision * index_cb_4d],
           &padded_out_alias[stag_spinor_site_size * precision * padded_index_cb_4d],
           stag_spinor_site_size * precision);
  }*/

}

