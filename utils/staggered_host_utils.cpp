#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <short.h>

// QUDA headers
#include <unitarization_links.h>

// External headers
#include <llfat_utils.h>
#include <staggered_gauge_utils.h>
#include <host_utils.h>
#include <command_line_params.h>

#include <qio_field.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3

using namespace std;

// Staggered gauge field utils
//------------------------------------------------------
void constructStaggeredHostGhostGaugeField(quda::GaugeField *cpuFat, quda::GaugeField *cpuLong, void *milc_fatlink,
                                           void *milc_longlink, QudaGaugeParam &gauge_param)
{

  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  GaugeFieldParam cpuFatParam(milc_fatlink, gauge_param);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuFat = GaugeField::Create(cpuFatParam);

  gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(milc_longlink, gauge_param);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuLong = GaugeField::Create(cpuLongParam);
}

void constructStaggeredHostDeviceGaugeField(void **qdp_inlink, void **qdp_longlink_cpu, void **qdp_longlink_gpu,
                                            void **qdp_fatlink_cpu, void **qdp_fatlink_gpu, QudaGaugeParam &gauge_param,
                                            int argc, char **argv, bool &gauge_loaded)
{
  // load a field WITHOUT PHASES
  if (strcmp(latfile, "")) {
    if (!gauge_loaded) {
      read_gauge_field(latfile, qdp_inlink, gauge_param.cpu_prec, gauge_param.X, argc, argv);
      if (dslash_type != QUDA_LAPLACE_DSLASH) {
        applyGaugeFieldScaling_long(qdp_inlink, Vh, &gauge_param, QUDA_STAGGERED_DSLASH, gauge_param.cpu_prec);
      }
      gauge_loaded = true;
    } // else it's already been loaded
  } else {
    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      constructQudaGaugeField(qdp_inlink, 1, gauge_param.cpu_prec, &gauge_param);
    } else {
      constructFatLongGaugeField(qdp_inlink, qdp_longlink_cpu, 1, gauge_param.cpu_prec, &gauge_param,
                                 compute_fatlong ? QUDA_STAGGERED_DSLASH : dslash_type);
    }
  }

  // QUDA_STAGGERED_DSLASH follows the same codepath whether or not you
  // "compute" the fat/long links or not.
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink_gpu[dir], qdp_inlink[dir], V * gauge_site_size * host_gauge_data_type_size);
      memcpy(qdp_fatlink_cpu[dir], qdp_inlink[dir], V * gauge_site_size * host_gauge_data_type_size);
      memset(qdp_longlink_gpu[dir], 0, V * gauge_site_size * host_gauge_data_type_size);
      memset(qdp_longlink_cpu[dir], 0, V * gauge_site_size * host_gauge_data_type_size);
    }
  } else {
    // QUDA_ASQTAD_DSLASH
    if (compute_fatlong) {
      computeFatLongGPUandCPU(qdp_fatlink_gpu, qdp_longlink_gpu, qdp_fatlink_cpu, qdp_longlink_cpu, qdp_inlink,
                              gauge_param, host_gauge_data_type_size, n_naiks, eps_naik);
    } else {
      // Not computing FatLong
      for (int dir = 0; dir < 4; dir++) {
        memcpy(qdp_fatlink_gpu[dir], qdp_inlink[dir], V * gauge_site_size * host_gauge_data_type_size);
        memcpy(qdp_fatlink_cpu[dir], qdp_inlink[dir], V * gauge_site_size * host_gauge_data_type_size);
        memcpy(qdp_longlink_gpu[dir], qdp_longlink_cpu[dir], V * gauge_site_size * host_gauge_data_type_size);
      }
    }
  }
}

void constructStaggeredHostGaugeField(void **qdp_inlink, void **qdp_longlink, void **qdp_fatlink,
                                      QudaGaugeParam &gauge_param, int argc, char **argv)
{
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;

  if (strcmp(latfile, "")) {
    // load in the command line supplied gauge field using QIO and LIME
    read_gauge_field(latfile, qdp_inlink, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    if (dslash_type != QUDA_LAPLACE_DSLASH) {
      applyGaugeFieldScaling_long(qdp_inlink, Vh, &gauge_param, QUDA_STAGGERED_DSLASH, gauge_param.cpu_prec);
    }
  } else {
    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      constructQudaGaugeField(qdp_inlink, 1, gauge_param.cpu_prec, &gauge_param);
    } else {
      constructFatLongGaugeField(qdp_inlink, qdp_longlink, 1, gauge_param.cpu_prec, &gauge_param,
                                 compute_fatlong ? QUDA_STAGGERED_DSLASH : dslash_type);
    }
  }

  // QUDA_STAGGERED_DSLASH follows the same codepath whether or not you
  // "compute" the fat/long links or not.
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink[dir], qdp_inlink[dir], V * gauge_site_size * host_gauge_data_type_size);
      memset(qdp_longlink[dir], 0, V * gauge_site_size * host_gauge_data_type_size);
    }
  } else {
    // QUDA_ASQTAD_DSLASH
    if (compute_fatlong) {
      computeFatLongGPU(qdp_fatlink, qdp_longlink, qdp_inlink, gauge_param, host_gauge_data_type_size, n_naiks, eps_naik);
    } else {
      for (int dir = 0; dir < 4; dir++) {
        memcpy(qdp_fatlink[dir], qdp_inlink[dir], V * gauge_site_size * host_gauge_data_type_size);
      }
    }
  }
}

void constructFatLongGaugeField(void **fatlink, void **longlink, int type, QudaPrecision precision,
                                QudaGaugeParam *param, QudaDslashType dslash_type)
{
  if (type == 0) {
    if (precision == QUDA_DOUBLE_PRECISION) {
      constructUnitGaugeField((double **)fatlink, param);
      constructUnitGaugeField((double **)longlink, param);
    } else {
      constructUnitGaugeField((float **)fatlink, param);
      constructUnitGaugeField((float **)longlink, param);
    }
  } else {
    if (precision == QUDA_DOUBLE_PRECISION) {
      // if doing naive staggered then set to long links so that the staggered phase is applied
      param->type = dslash_type == QUDA_ASQTAD_DSLASH ? QUDA_ASQTAD_FAT_LINKS : QUDA_ASQTAD_LONG_LINKS;
      if (type != 3)
        constructRandomGaugeField((double **)fatlink, param, dslash_type);
      else
        applyStaggeredScaling((double **)fatlink, param, type);
      param->type = QUDA_ASQTAD_LONG_LINKS;
      if (dslash_type == QUDA_ASQTAD_DSLASH) {
        if (type != 3)
          constructRandomGaugeField((double **)longlink, param, dslash_type);
        else
          applyStaggeredScaling((double **)longlink, param, type);
      }
    } else {
      param->type = dslash_type == QUDA_ASQTAD_DSLASH ? QUDA_ASQTAD_FAT_LINKS : QUDA_ASQTAD_LONG_LINKS;
      if (type != 3)
        constructRandomGaugeField((float **)fatlink, param, dslash_type);
      else
        applyStaggeredScaling((float **)fatlink, param, type);

      param->type = QUDA_ASQTAD_LONG_LINKS;
      if (dslash_type == QUDA_ASQTAD_DSLASH) {
        if (type != 3)
          constructRandomGaugeField((float **)longlink, param, dslash_type);
        else
          applyStaggeredScaling((float **)longlink, param, type);
      }
    }

    if (dslash_type == QUDA_ASQTAD_DSLASH) {
      // incorporate non-trivial phase into long links
      const double phase = (M_PI * rand()) / RAND_MAX;
      const complex<double> z = polar(1.0, phase);
      for (int dir = 0; dir < 4; ++dir) {
        for (int i = 0; i < V; ++i) {
          for (int j = 0; j < gauge_site_size; j += 2) {
            if (precision == QUDA_DOUBLE_PRECISION) {
              complex<double> *l = (complex<double> *)(&(((double *)longlink[dir])[i * gauge_site_size + j]));
              *l *= z;
            } else {
              complex<float> *l = (complex<float> *)(&(((float *)longlink[dir])[i * gauge_site_size + j]));
              *l *= z;
            }
          }
        }
      }
    }

    if (type == 3) return;

    // set all links to zero to emulate the 1-link operator (needed for host comparison)
    if (dslash_type == QUDA_STAGGERED_DSLASH) {
      for (int dir = 0; dir < 4; ++dir) {
        for (int i = 0; i < V; ++i) {
          for (int j = 0; j < gauge_site_size; j += 2) {
            if (precision == QUDA_DOUBLE_PRECISION) {
              ((double *)longlink[dir])[i * gauge_site_size + j] = 0.0;
              ((double *)longlink[dir])[i * gauge_site_size + j + 1] = 0.0;
            } else {
              ((float *)longlink[dir])[i * gauge_site_size + j] = 0.0;
              ((float *)longlink[dir])[i * gauge_site_size + j + 1] = 0.0;
            }
          }
        }
      }
    }
  }
}

void loadFatLongGaugeQuda(void *milc_fatlink, void *milc_longlink, QudaGaugeParam &gauge_param)
{
  // Specific gauge parameters for MILC
  int pad_size = 0;
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2;
  int t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2;
  pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
#endif

  int fat_pad = pad_size;
  int link_pad = 3 * pad_size;

  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;

  gauge_param.ga_pad = fat_pad;
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
    gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
  } else {
    gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;
  }
  gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;

  loadGaugeQuda(milc_fatlink, &gauge_param);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
    gauge_param.ga_pad = link_pad;
    gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
    gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
    gauge_param.reconstruct_precondition = link_recon_precondition;
    loadGaugeQuda(milc_longlink, &gauge_param);
  }
}

#ifndef MULTI_GPU
template <typename su3_matrix, typename Float>
void computeLongLinkCPU(void **longlink, su3_matrix **sitelink, Float *act_path_coeff)
{

  su3_matrix temp;
  for (int dir = XUP; dir <= TUP; ++dir) {
    int dx[4] = {0, 0, 0, 0};
    for (int i = 0; i < V; ++i) {
      // Initialize the longlinks
      su3_matrix *llink = ((su3_matrix *)longlink[dir]) + i;
      llfat_scalar_mult_su3_matrix(sitelink[dir] + i, act_path_coeff[1], llink);
      dx[dir] = 1;
      int nbr_idx = neighborIndexFullLattice(Z, i, dx);
      llfat_mult_su3_nn(llink, sitelink[dir] + nbr_idx, &temp);
      dx[dir] = 2;
      nbr_idx = neighborIndexFullLattice(Z, i, dx);
      llfat_mult_su3_nn(&temp, sitelink[dir] + nbr_idx, llink);
    }
  }
  return;
}
#else

template <typename su3_matrix, typename Float>
void computeLongLinkCPU(void **longlink, su3_matrix **sitelinkEx, Float *act_path_coeff)
{
  int E[4];
  for (int dir = 0; dir < 4; ++dir) E[dir] = Z[dir] + 4;
  const int extended_volume = E[3] * E[2] * E[1] * E[0];

  su3_matrix temp;
  for (int t = 0; t < Z[3]; ++t) {
    for (int z = 0; z < Z[2]; ++z) {
      for (int y = 0; y < Z[1]; ++y) {
        for (int x = 0; x < Z[0]; ++x) {
          const int oddBit = (x + y + z + t) & 1;
          int little_index = ((((t * Z[2] + z) * Z[1] + y) * Z[0] + x) / 2) + oddBit * Vh;
          int large_index
            = (((((t + 2) * E[2] + (z + 2)) * E[1] + (y + 2)) * E[0] + x + 2) / 2) + oddBit * (extended_volume / 2);

          for (int dir = XUP; dir <= TUP; ++dir) {
            int dx[4] = {0, 0, 0, 0};
            su3_matrix *llink = ((su3_matrix *)longlink[dir]) + little_index;
            llfat_scalar_mult_su3_matrix(sitelinkEx[dir] + large_index, act_path_coeff[1], llink);
            dx[dir] = 1;
            int nbr_index = neighborIndexFullLattice(E, large_index, dx);
            llfat_mult_su3_nn(llink, sitelinkEx[dir] + nbr_index, &temp);
            dx[dir] = 2;
            nbr_index = neighborIndexFullLattice(E, large_index, dx);
            llfat_mult_su3_nn(&temp, sitelinkEx[dir] + nbr_index, llink);
          }
        } // x
      }   // y
    }     // z
  }       // t
  return;
}
#endif

void computeLongLinkCPU(void **longlink, void **sitelink, QudaPrecision prec, void *act_path_coeff)
{
  if (longlink) {
    switch (prec) {
    case QUDA_DOUBLE_PRECISION:
      computeLongLinkCPU((void **)longlink, (su3_matrix<double> **)sitelink, (double *)act_path_coeff);
      break;

    case QUDA_SINGLE_PRECISION:
      computeLongLinkCPU((void **)longlink, (su3_matrix<float> **)sitelink, (float *)act_path_coeff);
      break;
    default:
      fprintf(stderr, "ERROR: unsupported precision(%d)\n", prec);
      exit(1);
      break;
    }
  } // if(longlink)
}

// Compute the full HISQ stencil on the CPU.
// If "eps_naik" is 0, there's no naik correction,
// and this routine skips building the paths in "act_path_coeffs[2]"
void computeHISQLinksCPU(void **fatlink, void **longlink, void **fatlink_eps, void **longlink_eps, void **sitelink,
                         void *qudaGaugeParamPtr, double **act_path_coeffs, double eps_naik)
{
  // Prepare various things
  QudaGaugeParam &qudaGaugeParam = *((QudaGaugeParam *)qudaGaugeParamPtr);
  // Needed for unitarization, following "unitarize_link_test.cpp"
  quda::GaugeFieldParam gParam(0, qudaGaugeParam);
  gParam.pad = 0;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.order = QUDA_MILC_GAUGE_ORDER; // must be true!

  const QudaPrecision prec = qudaGaugeParam.cpu_prec;
  const size_t gSize = prec;

  // Compute n_naiks
  const int n_naiks = (eps_naik == 0.0 ? 1 : 2);

  ///////////////////////////////
  // Create extended CPU field //
  ///////////////////////////////

  void *sitelink_ex[4];
  for (int i = 0; i < 4; i++) sitelink_ex[i] = pinned_malloc(V_ex * gauge_site_size * gSize);

#ifdef MULTI_GPU
  void *ghost_sitelink[4];
  void *ghost_sitelink_diag[16];
#endif

  int X1 = Z[0];
  int X2 = Z[1];
  int X3 = Z[2];
  int X4 = Z[3];

  for (int i = 0; i < V_ex; i++) {
    int sid = i;
    int oddBit = 0;
    if (i >= Vh_ex) {
      sid = i - Vh_ex;
      oddBit = 1;
    }

    int za = sid / E1h;
    int x1h = sid - za * E1h;
    int zb = za / E2;
    int x2 = za - zb * E2;
    int x4 = zb / E3;
    int x3 = zb - x4 * E3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2 * x1h + x1odd;

    if (x1 < 2 || x1 >= X1 + 2 || x2 < 2 || x2 >= X2 + 2 || x3 < 2 || x3 >= X3 + 2 || x4 < 2 || x4 >= X4 + 2) {
#ifdef MULTI_GPU
      continue;
#endif
    }

    x1 = (x1 - 2 + X1) % X1;
    x2 = (x2 - 2 + X2) % X2;
    x3 = (x3 - 2 + X3) % X3;
    x4 = (x4 - 2 + X4) % X4;

    int idx = (x4 * X3 * X2 * X1 + x3 * X2 * X1 + x2 * X1 + x1) >> 1;
    if (oddBit) { idx += Vh; }
    for (int dir = 0; dir < 4; dir++) {
      char *src = (char *)sitelink[dir];
      char *dst = (char *)sitelink_ex[dir];
      memcpy(dst + i * gauge_site_size * gSize, src + idx * gauge_site_size * gSize, gauge_site_size * gSize);
    } // dir
  }   // i

  /////////////////////////////////////
  // Allocate all CPU intermediaries //
  /////////////////////////////////////

  void *v_reflink[4];    // V link -- fat7 smeared link
  void *w_reflink[4];    // unitarized V link
  void *w_reflink_ex[4]; // extended W link
  for (int i = 0; i < 4; i++) {
    v_reflink[i] = safe_malloc(V * gauge_site_size * gSize);
    w_reflink[i] = safe_malloc(V * gauge_site_size * gSize);
    w_reflink_ex[i] = safe_malloc(V_ex * gauge_site_size * gSize);
  }

#ifdef MULTI_GPU
  void *ghost_wlink[4];
  void *ghost_wlink_diag[16];
#endif

  // Copy of V link needed for CPU unitarization routines
  void *v_sitelink = pinned_malloc(4 * V * gauge_site_size * gSize);

  // FIXME: we have this complication because references takes coeff as float/double
  //        depending on the precision while the GPU code aways take coeff as double
  void *coeff;
  double coeff_dp[6];
  float coeff_sp[6];

  /////////////////////////////////////////////////////
  // Create V links (fat7 links), 1st path table set //
  /////////////////////////////////////////////////////

  for (int i = 0; i < 6; i++) coeff_sp[i] = coeff_dp[i] = act_path_coeffs[0][i];
  coeff = (prec == QUDA_DOUBLE_PRECISION) ? (void *)coeff_dp : (void *)coeff_sp;

  // Only need fat links.
#ifdef MULTI_GPU
  int optflag = 0;
  // we need x,y,z site links in the back and forward T slice
  // so it is 3*2*Vs_t
  int Vs[4] = {Vs_x, Vs_y, Vs_z, Vs_t};
  for (int i = 0; i < 4; i++) ghost_sitelink[i] = safe_malloc(8 * Vs[i] * gauge_site_size * gSize);

  // nu |     |
  //   |_____|
  //     mu

  for (int nu = 0; nu < 4; nu++) {
    for (int mu = 0; mu < 4; mu++) {
      if (nu == mu) {
        ghost_sitelink_diag[nu * 4 + mu] = NULL;
      } else {
        // the other directions
        int dir1, dir2;
        for (dir1 = 0; dir1 < 4; dir1++) {
          if (dir1 != nu && dir1 != mu) { break; }
        }
        for (dir2 = 0; dir2 < 4; dir2++) {
          if (dir2 != nu && dir2 != mu && dir2 != dir1) { break; }
        }
        ghost_sitelink_diag[nu * 4 + mu] = safe_malloc(Z[dir1] * Z[dir2] * gauge_site_size * gSize);
        memset(ghost_sitelink_diag[nu * 4 + mu], 0, Z[dir1] * Z[dir2] * gauge_site_size * gSize);
      }
    }
  }
  exchange_cpu_sitelink(gParam.x, sitelink, ghost_sitelink, ghost_sitelink_diag, prec, &qudaGaugeParam, optflag);
  llfat_reference_mg(v_reflink, sitelink, ghost_sitelink, ghost_sitelink_diag, prec, coeff);
#else
  llfat_reference(v_reflink, sitelink, prec, coeff);
#endif

  /////////////////////////////////////////
  // Create W links (unitarized V links) //
  /////////////////////////////////////////

  // This is based on "unitarize_link_test.cpp"

  // Format change
  reorderQDPtoMILC(v_sitelink, v_reflink, V, gauge_site_size, prec, prec);

  // Prepare cpuGaugeFields for unitarization
  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  gParam.gauge = v_sitelink;
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  quda::GaugeField *cpuVLink = quda::GaugeField::Create(gParam);

  gParam.create = QUDA_ZERO_FIELD_CREATE;
  quda::GaugeField *cpuWLink = quda::GaugeField::Create(gParam);

  // unitarize
  unitarizeLinksCPU(*cpuWLink, *cpuVLink);

  // Copy back into "w_reflink"
  reorderMILCtoQDP(w_reflink, cpuWLink->Gauge_p(), V, gauge_site_size, prec, prec);

  // Clean up cpuGaugeFields, we don't need them anymore.
  delete cpuVLink;
  delete cpuWLink;

  ///////////////////////////////////
  // Prepare for extended W fields //
  ///////////////////////////////////

  for (int i = 0; i < V_ex; i++) {
    int sid = i;
    int oddBit = 0;
    if (i >= Vh_ex) {
      sid = i - Vh_ex;
      oddBit = 1;
    }

    int za = sid / E1h;
    int x1h = sid - za * E1h;
    int zb = za / E2;
    int x2 = za - zb * E2;
    int x4 = zb / E3;
    int x3 = zb - x4 * E3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2 * x1h + x1odd;

    if (x1 < 2 || x1 >= X1 + 2 || x2 < 2 || x2 >= X2 + 2 || x3 < 2 || x3 >= X3 + 2 || x4 < 2 || x4 >= X4 + 2) {
#ifdef MULTI_GPU
      continue;
#endif
    }

    x1 = (x1 - 2 + X1) % X1;
    x2 = (x2 - 2 + X2) % X2;
    x3 = (x3 - 2 + X3) % X3;
    x4 = (x4 - 2 + X4) % X4;

    int idx = (x4 * X3 * X2 * X1 + x3 * X2 * X1 + x2 * X1 + x1) >> 1;
    if (oddBit) { idx += Vh; }
    for (int dir = 0; dir < 4; dir++) {
      char *src = (char *)w_reflink[dir];
      char *dst = (char *)w_reflink_ex[dir];
      memcpy(dst + i * gauge_site_size * gSize, src + idx * gauge_site_size * gSize, gauge_site_size * gSize);
    } // dir
  }   // i

  //////////////////////////////
  // Create extended W fields //
  //////////////////////////////

#ifdef MULTI_GPU
  optflag = 0;
  // we need x,y,z site links in the back and forward T slice
  // so it is 3*2*Vs_t
  for (int i = 0; i < 4; i++) ghost_wlink[i] = safe_malloc(8 * Vs[i] * gauge_site_size * gSize);

  // nu |     |
  //   |_____|
  //     mu

  for (int nu = 0; nu < 4; nu++) {
    for (int mu = 0; mu < 4; mu++) {
      if (nu == mu) {
        ghost_wlink_diag[nu * 4 + mu] = NULL;
      } else {
        // the other directions
        int dir1, dir2;
        for (dir1 = 0; dir1 < 4; dir1++) {
          if (dir1 != nu && dir1 != mu) { break; }
        }
        for (dir2 = 0; dir2 < 4; dir2++) {
          if (dir2 != nu && dir2 != mu && dir2 != dir1) { break; }
        }
        ghost_wlink_diag[nu * 4 + mu] = safe_malloc(Z[dir1] * Z[dir2] * gauge_site_size * gSize);
        memset(ghost_wlink_diag[nu * 4 + mu], 0, Z[dir1] * Z[dir2] * gauge_site_size * gSize);
      }
    }
  }
#endif

  ////////////////////////////////////////////
  // Prepare to create Naiks, 3rd table set //
  ////////////////////////////////////////////

  if (n_naiks > 1) {

    for (int i = 0; i < 6; i++) coeff_sp[i] = coeff_dp[i] = act_path_coeffs[2][i];
    coeff = (prec == QUDA_DOUBLE_PRECISION) ? (void *)coeff_dp : (void *)coeff_sp;

#ifdef MULTI_GPU

    exchange_cpu_sitelink(qudaGaugeParam.X, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec,
                          &qudaGaugeParam, optflag);
    llfat_reference_mg(fatlink, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec, coeff);

    {
      int R[4] = {2, 2, 2, 2};
      exchange_cpu_sitelink_ex(qudaGaugeParam.X, R, w_reflink_ex, QUDA_QDP_GAUGE_ORDER, qudaGaugeParam.cpu_prec, 0, 4);
      computeLongLinkCPU(longlink, w_reflink_ex, qudaGaugeParam.cpu_prec, coeff);
    }
#else
    llfat_reference(fatlink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
    computeLongLinkCPU(longlink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
#endif

    // Rescale fat and long links into eps links
    for (int i = 0; i < 4; i++) {
      cpu_axy(prec, eps_naik, fatlink[i], fatlink_eps[i], V * gauge_site_size);
      cpu_axy(prec, eps_naik, longlink[i], longlink_eps[i], V * gauge_site_size);
    }
  }

  /////////////////////////////////////////////////////////////
  // Prepare to create X links and long links, 2nd table set //
  /////////////////////////////////////////////////////////////

  for (int i = 0; i < 6; i++) coeff_sp[i] = coeff_dp[i] = act_path_coeffs[1][i];
  coeff = (prec == QUDA_DOUBLE_PRECISION) ? (void *)coeff_dp : (void *)coeff_sp;

#ifdef MULTI_GPU
  optflag = 0;

  // We've already built the extended W fields.

  exchange_cpu_sitelink(qudaGaugeParam.X, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec,
                        &qudaGaugeParam, optflag);
  llfat_reference_mg(fatlink, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec, coeff);

  {
    int R[4] = {2, 2, 2, 2};
    exchange_cpu_sitelink_ex(qudaGaugeParam.X, R, w_reflink_ex, QUDA_QDP_GAUGE_ORDER, qudaGaugeParam.cpu_prec, 0, 4);
    computeLongLinkCPU(longlink, w_reflink_ex, qudaGaugeParam.cpu_prec, coeff);
  }
#else
  llfat_reference(fatlink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
  computeLongLinkCPU(longlink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
#endif

  if (n_naiks > 1) {
    // Accumulate into eps links.
    for (int i = 0; i < 4; i++) {
      cpu_xpy(prec, fatlink[i], fatlink_eps[i], V * gauge_site_size);
      cpu_xpy(prec, longlink[i], longlink_eps[i], V * gauge_site_size);
    }
  }

  //////////////
  // Clean up //
  //////////////

  for (int i = 0; i < 4; i++) {
    host_free(sitelink_ex[i]);
    host_free(v_reflink[i]);
    host_free(w_reflink[i]);
    host_free(w_reflink_ex[i]);
  }
  host_free(v_sitelink);

#ifdef MULTI_GPU
  for (int i = 0; i < 4; i++) {
    host_free(ghost_sitelink[i]);
    host_free(ghost_wlink[i]);
    for (int j = 0; j < 4; j++) {
      if (i == j) continue;
      host_free(ghost_sitelink_diag[i * 4 + j]);
      host_free(ghost_wlink_diag[i * 4 + j]);
    }
  }
#endif
}

void constructStaggeredTestSpinorParam(quda::ColorSpinorParam *cs_param, const QudaInvertParam *inv_param,
                                       const QudaGaugeParam *gauge_param)
{
  // Lattice vector spacetime/colour/spin/parity properties
  cs_param->nColor = 3;
  cs_param->nSpin = 1;
  cs_param->nDim = 5;
  for (int d = 0; d < 4; d++) cs_param->x[d] = gauge_param->X[d];
  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION || inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  if (pc) cs_param->x[0] /= 2;
  cs_param->x[4] = 1;
  cs_param->siteSubset = pc ? QUDA_PARITY_SITE_SUBSET : QUDA_FULL_SITE_SUBSET;

  // Lattice vector data properties
  cs_param->setPrecision(inv_param->cpu_prec);
  cs_param->pad = 0;
  cs_param->siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  cs_param->fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  cs_param->gammaBasis = inv_param->gamma_basis;
  cs_param->create = QUDA_ZERO_FIELD_CREATE;
  cs_param->location = QUDA_CPU_FIELD_LOCATION;
}

// data reordering routines
template <typename Out, typename In> void reorderQDPtoMILC(Out *milc_out, In **qdp_in, int V, int siteSize)
{
  for (int i = 0; i < V; i++) {
    for (int dir = 0; dir < 4; dir++) {
      for (int j = 0; j < siteSize; j++) {
        milc_out[(i * 4 + dir) * siteSize + j] = static_cast<Out>(qdp_in[dir][i * siteSize + j]);
      }
    }
  }
}

void reorderQDPtoMILC(void *milc_out, void **qdp_in, int V, int siteSize, QudaPrecision out_precision,
                      QudaPrecision in_precision)
{
  if (out_precision == QUDA_SINGLE_PRECISION) {
    if (in_precision == QUDA_SINGLE_PRECISION) {
      reorderQDPtoMILC<float, float>((float *)milc_out, (float **)qdp_in, V, siteSize);
    } else if (in_precision == QUDA_DOUBLE_PRECISION) {
      reorderQDPtoMILC<float, double>((float *)milc_out, (double **)qdp_in, V, siteSize);
    }
  } else if (out_precision == QUDA_DOUBLE_PRECISION) {
    if (in_precision == QUDA_SINGLE_PRECISION) {
      reorderQDPtoMILC<double, float>((double *)milc_out, (float **)qdp_in, V, siteSize);
    } else if (in_precision == QUDA_DOUBLE_PRECISION) {
      reorderQDPtoMILC<double, double>((double *)milc_out, (double **)qdp_in, V, siteSize);
    }
  }
}

template <typename Out, typename In> void reorderMILCtoQDP(Out **qdp_out, In *milc_in, int V, int siteSize)
{
  for (int i = 0; i < V; i++) {
    for (int dir = 0; dir < 4; dir++) {
      for (int j = 0; j < siteSize; j++) {
        qdp_out[dir][i * siteSize + j] = static_cast<Out>(milc_in[(i * 4 + dir) * siteSize + j]);
      }
    }
  }
}

void reorderMILCtoQDP(void **qdp_out, void *milc_in, int V, int siteSize, QudaPrecision out_precision,
                      QudaPrecision in_precision)
{
  if (out_precision == QUDA_SINGLE_PRECISION) {
    if (in_precision == QUDA_SINGLE_PRECISION) {
      reorderMILCtoQDP<float, float>((float **)qdp_out, (float *)milc_in, V, siteSize);
    } else if (in_precision == QUDA_DOUBLE_PRECISION) {
      reorderMILCtoQDP<float, double>((float **)qdp_out, (double *)milc_in, V, siteSize);
    }
  } else if (out_precision == QUDA_DOUBLE_PRECISION) {
    if (in_precision == QUDA_SINGLE_PRECISION) {
      reorderMILCtoQDP<double, float>((double **)qdp_out, (float *)milc_in, V, siteSize);
    } else if (in_precision == QUDA_DOUBLE_PRECISION) {
      reorderMILCtoQDP<double, double>((double **)qdp_out, (double *)milc_in, V, siteSize);
    }
  }
}

template <typename Float> void applyStaggeredScaling(Float **res, QudaGaugeParam *param, int type)
{

  if (type == 3) applyGaugeFieldScaling_long((Float **)res, Vh, param, QUDA_STAGGERED_DSLASH);

  return;
}

template <typename Float>
void applyGaugeFieldScaling_long(Float **gauge, int Vh, QudaGaugeParam *param, QudaDslashType dslash_type)
{
  int X1h = param->X[0] / 2;
  int X1 = param->X[0];
  int X2 = param->X[1];
  int X3 = param->X[2];
  int X4 = param->X[3];

  // rescale long links by the appropriate coefficient
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    for (int d = 0; d < 4; d++) {
      for (int i = 0; i < V * gauge_site_size; i++) {
        gauge[d][i] /= (-24 * param->tadpole_coeff * param->tadpole_coeff);
      }
    }
  }

  // apply the staggered phases
  for (int d = 0; d < 3; d++) {

    // even
    for (int i = 0; i < Vh; i++) {

      int index = fullLatticeIndex(i, 0);
      int i4 = index / (X3 * X2 * X1);
      int i3 = (index - i4 * (X3 * X2 * X1)) / (X2 * X1);
      int i2 = (index - i4 * (X3 * X2 * X1) - i3 * (X2 * X1)) / X1;
      int i1 = index - i4 * (X3 * X2 * X1) - i3 * (X2 * X1) - i2 * X1;
      int sign = 1;

      if (d == 0) {
        if (i4 % 2 == 1) { sign = -1; }
      }

      if (d == 1) {
        if ((i4 + i1) % 2 == 1) { sign = -1; }
      }
      if (d == 2) {
        if ((i4 + i1 + i2) % 2 == 1) { sign = -1; }
      }

      for (int j = 0; j < 18; j++) { gauge[d][i * gauge_site_size + j] *= sign; }
    }
    // odd
    for (int i = 0; i < Vh; i++) {
      int index = fullLatticeIndex(i, 1);
      int i4 = index / (X3 * X2 * X1);
      int i3 = (index - i4 * (X3 * X2 * X1)) / (X2 * X1);
      int i2 = (index - i4 * (X3 * X2 * X1) - i3 * (X2 * X1)) / X1;
      int i1 = index - i4 * (X3 * X2 * X1) - i3 * (X2 * X1) - i2 * X1;
      int sign = 1;

      if (d == 0) {
        if (i4 % 2 == 1) { sign = -1; }
      }

      if (d == 1) {
        if ((i4 + i1) % 2 == 1) { sign = -1; }
      }
      if (d == 2) {
        if ((i4 + i1 + i2) % 2 == 1) { sign = -1; }
      }

      for (int j = 0; j < 18; j++) { gauge[d][(Vh + i) * gauge_site_size + j] *= sign; }
    }
  }

  // Apply boundary conditions to temporal links
  if (param->t_boundary == QUDA_ANTI_PERIODIC_T && last_node_in_t()) {
    for (int j = 0; j < Vh; j++) {
      int sign = 1;
      if (dslash_type == QUDA_ASQTAD_DSLASH) {
        if (j >= (X4 - 3) * X1h * X2 * X3) { sign = -1; }
      } else {
        if (j >= (X4 - 1) * X1h * X2 * X3) { sign = -1; }
      }

      for (int i = 0; i < 18; i++) {
        gauge[3][j * gauge_site_size + i] *= sign;
        gauge[3][(Vh + j) * gauge_site_size + i] *= sign;
      }
    }
  }
}

void applyGaugeFieldScaling_long(void **gauge, int Vh, QudaGaugeParam *param, QudaDslashType dslash_type,
                                 QudaPrecision local_prec)
{
  if (local_prec == QUDA_DOUBLE_PRECISION) {
    applyGaugeFieldScaling_long((double **)gauge, Vh, param, dslash_type);
  } else if (local_prec == QUDA_SINGLE_PRECISION) {
    applyGaugeFieldScaling_long((float **)gauge, Vh, param, dslash_type);
  } else {
    errorQuda("Invalid type %d for applyGaugeFieldScaling_long\n", local_prec);
  }
}
