#include <quda.h>
#include <quda_fortran.h>
#include <quda_api.h>
#include <malloc_quda.h>
#include <util_quda.h>
#include <gauge_field.h>

/*
  @file fortran_interface.cpp

  @brief The following functions are for the Fortran interface, which
  is utilized by the TIFR and BQCD applications.
*/

// local function declarations from interface_quda.cpp that are not exported
namespace quda
{
  GaugeField *getResidentGauge();
}
void freeSloppyGaugeQuda();

void init_quda_(int *dev) { initQuda(*dev); }
void init_quda_device_(int *dev) { initQudaDevice(*dev); }
void init_quda_memory_() { initQudaMemory(); }
void end_quda_() { endQuda(); }
void load_gauge_quda_(void *h_gauge, QudaGaugeParam *param) { loadGaugeQuda(h_gauge, param); }
void free_gauge_quda_() { freeGaugeQuda(); }
void free_sloppy_gauge_quda_() { freeSloppyGaugeQuda(); }
void load_clover_quda_(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param)
{
  loadCloverQuda(h_clover, h_clovinv, inv_param);
}
void free_clover_quda_(void) { freeCloverQuda(); }
void dslash_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity *parity)
{
  dslashQuda(h_out, h_in, inv_param, *parity);
}
void clover_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity *parity, int *inverse)
{
  cloverQuda(h_out, h_in, inv_param, *parity, *inverse);
}
void mat_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param) { MatQuda(h_out, h_in, inv_param); }
void mat_dag_mat_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param) { MatDagMatQuda(h_out, h_in, inv_param); }
void invert_quda_(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  fflush(stdout);
  // ensure that fifth dimension is set to 1
  if (param->dslash_type == QUDA_ASQTAD_DSLASH || param->dslash_type == QUDA_STAGGERED_DSLASH) param->Ls = 1;
  invertQuda(hp_x, hp_b, param);
  fflush(stdout);
}

void invert_multishift_quda_(void *h_x, void *hp_b, QudaInvertParam *param)
{
  if (!quda::getResidentGauge()) errorQuda("Resident gauge field not allocated");

  auto &U = *quda::getResidentGauge();

  // ensure that fifth dimension is set to 1
  if (param->dslash_type == QUDA_ASQTAD_DSLASH || param->dslash_type == QUDA_STAGGERED_DSLASH) param->Ls = 1;

  // get data into array of pointers
  int nSpin = (param->dslash_type == QUDA_STAGGERED_DSLASH || param->dslash_type == QUDA_ASQTAD_DSLASH) ? 1 : 4;

  // compute offset assuming TIFR padded ordering (FIXME)
  if (param->dirac_order != QUDA_TIFR_PADDED_DIRAC_ORDER)
    errorQuda("Fortran multi-shift solver presently only supports QUDA_TIFR_PADDED_DIRAC_ORDER and not %d",
              param->dirac_order);
  auto X = U.X();
  size_t cb_offset = (X[0] / 2) * X[1] * (X[2] + 4) * X[3] * U.Ncolor() * nSpin * 2 * param->cpu_prec;
  void *hp_x[QUDA_MAX_MULTI_SHIFT];
  for (int i = 0; i < param->num_offset; i++) hp_x[i] = static_cast<char *>(h_x) + i * cb_offset;

  invertMultiShiftQuda(hp_x, hp_b, param);
}

void flush_chrono_quda_(int *index) { flushChronoQuda(*index); }

void register_pinned_quda_(void *ptr, size_t *bytes) { register_pinned(ptr, *bytes); }

void unregister_pinned_quda_(void *ptr) { unregister_pinned(ptr); }

void new_quda_gauge_param_(QudaGaugeParam *param) { *param = newQudaGaugeParam(); }
void new_quda_invert_param_(QudaInvertParam *param) { *param = newQudaInvertParam(); }

void update_gauge_field_quda_(void *gauge, void *momentum, double *dt, bool *conj_mom, bool *exact, QudaGaugeParam *param)
{
  updateGaugeFieldQuda(gauge, momentum, *dt, (int)*conj_mom, (int)*exact, param);
}

static inline int opp(int dir) { return 7 - dir; }

static void createGaugeForcePaths(int **paths, int dir, int num_loop_types)
{

  int index = 0;
  // Plaquette paths
  if (num_loop_types >= 1)
    for (int i = 0; i < 4; ++i) {
      if (i == dir) continue;
      paths[index][0] = i;
      paths[index][1] = opp(dir);
      paths[index++][2] = opp(i);
      paths[index][0] = opp(i);
      paths[index][1] = opp(dir);
      paths[index++][2] = i;
    }

  // Rectangle Paths
  if (num_loop_types >= 2)
    for (int i = 0; i < 4; ++i) {
      if (i == dir) continue;
      paths[index][0] = paths[index][1] = i;
      paths[index][2] = opp(dir);
      paths[index][3] = paths[index][4] = opp(i);
      index++;
      paths[index][0] = paths[index][1] = opp(i);
      paths[index][2] = opp(dir);
      paths[index][3] = paths[index][4] = i;
      index++;
      paths[index][0] = dir;
      paths[index][1] = i;
      paths[index][2] = paths[index][3] = opp(dir);
      paths[index][4] = opp(i);
      index++;
      paths[index][0] = dir;
      paths[index][1] = opp(i);
      paths[index][2] = paths[index][3] = opp(dir);
      paths[index][4] = i;
      index++;
      paths[index][0] = i;
      paths[index][1] = paths[index][2] = opp(dir);
      paths[index][3] = opp(i);
      paths[index][4] = dir;
      index++;
      paths[index][0] = opp(i);
      paths[index][1] = paths[index][2] = opp(dir);
      paths[index][3] = i;
      paths[index][4] = dir;
      index++;
    }

  if (num_loop_types >= 3) {
    // Staple paths
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        if (i == dir || j == dir || i == j) continue;
        paths[index][0] = i;
        paths[index][1] = j;
        paths[index][2] = opp(dir);
        paths[index][3] = opp(i), paths[index][4] = opp(j);
        index++;
        paths[index][0] = i;
        paths[index][1] = opp(j);
        paths[index][2] = opp(dir);
        paths[index][3] = opp(i), paths[index][4] = j;
        index++;
        paths[index][0] = opp(i);
        paths[index][1] = j;
        paths[index][2] = opp(dir);
        paths[index][3] = i, paths[index][4] = opp(j);
        index++;
        paths[index][0] = opp(i);
        paths[index][1] = opp(j);
        paths[index][2] = opp(dir);
        paths[index][3] = i, paths[index][4] = j;
        index++;
      }
    }
  }
}

void compute_gauge_force_quda_(void *mom, void *gauge, int *num_loop_types, double *coeff, double *dt,
                               QudaGaugeParam *param)
{

  int numPaths = 0;
  switch (*num_loop_types) {
  case 1: numPaths = 6; break;
  case 2: numPaths = 24; break;
  case 3: numPaths = 48; break;
  default: errorQuda("Invalid num_loop_types = %d\n", *num_loop_types);
  }

  auto *loop_coeff = static_cast<double *>(safe_malloc(numPaths * sizeof(double)));
  int *path_length = static_cast<int *>(safe_malloc(numPaths * sizeof(int)));

  if (*num_loop_types >= 1)
    for (int i = 0; i < 6; ++i) {
      loop_coeff[i] = coeff[0];
      path_length[i] = 3;
    }
  if (*num_loop_types >= 2)
    for (int i = 6; i < 24; ++i) {
      loop_coeff[i] = coeff[1];
      path_length[i] = 5;
    }
  if (*num_loop_types >= 3)
    for (int i = 24; i < 48; ++i) {
      loop_coeff[i] = coeff[2];
      path_length[i] = 5;
    }

  int **input_path_buf[4];
  for (int dir = 0; dir < 4; ++dir) {
    input_path_buf[dir] = static_cast<int **>(safe_malloc(numPaths * sizeof(int *)));
    for (int i = 0; i < numPaths; ++i) {
      input_path_buf[dir][i] = static_cast<int *>(safe_malloc(path_length[i] * sizeof(int)));
    }
    createGaugeForcePaths(input_path_buf[dir], dir, *num_loop_types);
  }

  int max_length = 6;

  computeGaugeForceQuda(mom, gauge, input_path_buf, path_length, loop_coeff, numPaths, max_length, *dt, param);

  for (auto &dir : input_path_buf) {
    for (int i = 0; i < numPaths; ++i) host_free(dir[i]);
    host_free(dir);
  }

  host_free(path_length);
  host_free(loop_coeff);
}

void compute_staggered_force_quda_(void *h_mom, double *dt, double *delta, void *gauge, void *x,
                                   QudaGaugeParam *gauge_param, QudaInvertParam *inv_param)
{
  computeStaggeredForceQuda(h_mom, *dt, *delta, gauge, (void **)x, gauge_param, inv_param);
}

// apply the staggered phases
void apply_staggered_phase_quda_()
{
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("applying staggered phase\n");
  if (quda::getResidentGauge()) {
    quda::getResidentGauge()->applyStaggeredPhase();
  } else {
    errorQuda("No persistent gauge field");
  }
}

// remove the staggered phases
void remove_staggered_phase_quda_()
{
  if (getVerbosity() >= QUDA_VERBOSE) printfQuda("removing staggered phase\n");
  if (quda::getResidentGauge()) {
    quda::getResidentGauge()->removeStaggeredPhase();
  } else {
    errorQuda("No persistent gauge field");
  }
  qudaDeviceSynchronize();
}

// evaluate the kinetic term
void kinetic_quda_(double *kin, void *momentum, QudaGaugeParam *param) { *kin = momActionQuda(momentum, param); }

/**
 * BQCD wants a node mapping with x varying fastest.
 */
static int bqcd_rank_from_coords(const int *coords, void *fdata)
{
  int *dims = static_cast<int *>(fdata);

  int rank = coords[3];
  for (int i = 2; i >= 0; i--) { rank = dims[i] * rank + coords[i]; }
  return rank;
}

void comm_set_gridsize_(int *grid) { initCommsGridQuda(4, grid, bqcd_rank_from_coords, static_cast<void *>(grid)); }

void plaq_quda_(double plaq[3]) { plaqQuda(plaq); }
