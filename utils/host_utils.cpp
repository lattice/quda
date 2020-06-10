#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <short.h>

#include <comm_quda.h>

// This contains the appropriate ifdef guards already
#include <mpi_comm_handle.h>

// QUDA headers
#include <color_spinor_field.h>
#include <unitarization_links.h>
#include <dslash_quda.h>

// External headers
#include <llfat_utils.h>
#include <staggered_gauge_utils.h>
#include <host_utils.h>
#include <command_line_params.h>

#include <misc.h>
#include <qio_field.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

using namespace std;

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3

int Z[4];
int V;
int Vh;
int Vs_x, Vs_y, Vs_z, Vs_t;
int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
int faceVolume[4];

//extended volume, +4
int E1, E1h, E2, E3, E4;
int E[4];
int V_ex, Vh_ex;

int Ls;
int V5;
int V5h;
double kappa5;

int my_spinor_site_size;

extern float fat_link_max;

// Set some local QUDA precision variables
QudaPrecision local_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision &cpu_prec = local_prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_refinement_sloppy = prec_refinement_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;
QudaPrecision &cuda_prec_ritz = prec_ritz;

size_t host_gauge_data_type_size = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
size_t host_spinor_data_type_size = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
size_t host_clover_data_type_size = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

void setQudaPrecisions()
{
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
  if (smoother_halo_prec == QUDA_INVALID_PRECISION) smoother_halo_prec = prec_null;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;
}

void setQudaMgSolveTypes()
{
  for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
    if (coarse_solve_type[i] == QUDA_INVALID_SOLVE) coarse_solve_type[i] = solve_type;
    if (smoother_solve_type[i] == QUDA_INVALID_SOLVE) smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE;
  }
}

void setQudaDefaultMgTestParams()
{
  // We give here some default values
  for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
    mg_verbosity[i] = QUDA_SUMMARIZE;
    setup_inv[i] = QUDA_BICGSTAB_INVERTER;
    num_setup_iter[i] = 1;
    setup_tol[i] = 5e-6;
    setup_maxiter[i] = 500;
    setup_maxiter_refresh[i] = 20;
    mu_factor[i] = 1.;
    coarse_solve_type[i] = QUDA_INVALID_SOLVE;
    smoother_solve_type[i] = QUDA_INVALID_SOLVE;
    mg_schwarz_type[i] = QUDA_INVALID_SCHWARZ;
    mg_schwarz_cycle[i] = 1;
    smoother_type[i] = QUDA_GCR_INVERTER;
    smoother_tol[i] = 0.25;
    coarse_solver[i] = QUDA_GCR_INVERTER;
    coarse_solver_tol[i] = 0.25;
    coarse_solver_maxiter[i] = 100;
    solver_location[i] = QUDA_CUDA_FIELD_LOCATION;
    setup_location[i] = QUDA_CUDA_FIELD_LOCATION;
    nu_pre[i] = 2;
    nu_post[i] = 2;
    n_block_ortho[i] = 1;

    // Default eigensolver params
    mg_eig[i] = false;
    mg_eig_tol[i] = 1e-3;
    mg_eig_nEv[i] = nvec[i];
    mg_eig_nKr[i] = 3 * nvec[i];
    mg_eig_require_convergence[i] = QUDA_BOOLEAN_TRUE;
    mg_eig_type[i] = QUDA_EIG_TR_LANCZOS;
    mg_eig_spectrum[i] = QUDA_SPECTRUM_SR_EIG;
    mg_eig_check_interval[i] = 5;
    mg_eig_max_restarts[i] = 100;
    mg_eig_use_normop[i] = QUDA_BOOLEAN_FALSE;
    mg_eig_use_dagger[i] = QUDA_BOOLEAN_FALSE;
    mg_eig_use_poly_acc[i] = QUDA_BOOLEAN_TRUE;
    mg_eig_poly_deg[i] = 100;
    mg_eig_amin[i] = 1.0;
    mg_eig_amax[i] = -1.0; // use power iterations

    setup_ca_basis[i] = QUDA_POWER_BASIS;
    setup_ca_basis_size[i] = 4;
    setup_ca_lambda_min[i] = 0.0;
    setup_ca_lambda_max[i] = -1.0; // use power iterations

    coarse_solver_ca_basis[i] = QUDA_POWER_BASIS;
    coarse_solver_ca_basis_size[i] = 4;
    coarse_solver_ca_lambda_min[i] = 0.0;
    coarse_solver_ca_lambda_max[i] = -1.0;

    strcpy(mg_vec_infile[i], "");
    strcpy(mg_vec_outfile[i], "");
  }
}

void constructQudaGaugeField(void **gauge, int type, QudaPrecision precision, QudaGaugeParam *param)
{
  if (type == 0) {
    if (precision == QUDA_DOUBLE_PRECISION)
      constructUnitGaugeField((double **)gauge, param);
    else
      constructUnitGaugeField((float **)gauge, param);
  } else if (type == 1) {
    if (precision == QUDA_DOUBLE_PRECISION)
      constructRandomGaugeField((double **)gauge, param);
    else
      constructRandomGaugeField((float **)gauge, param);
  } else {
    if (precision == QUDA_DOUBLE_PRECISION)
      applyGaugeFieldScaling((double **)gauge, Vh, param);
    else
      applyGaugeFieldScaling((float **)gauge, Vh, param);
  }
}

void constructHostGaugeField(void **gauge, QudaGaugeParam &gauge_param, int argc, char **argv)
{

  // 0 = unit gauge
  // 1 = random SU(3)
  // 2 = supplied field
  int construct_type = 0;
  if (strcmp(latfile, "")) {
    // load in the command line supplied gauge field using QIO and LIME
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_type = 2;
  } else {
    if (unit_gauge)
      construct_type = 0;
    else
      construct_type = 1;
  }
  constructQudaGaugeField(gauge, construct_type, gauge_param.cpu_prec, &gauge_param);
}

void constructHostCloverField(void *clover, void *clover_inv, QudaInvertParam &inv_param)
{
  double norm = 0.01; // clover components are random numbers in the range (-norm, norm)
  double diag = 1.0;  // constant added to the diagonal

  if (!compute_clover) constructQudaCloverField(clover, norm, diag, inv_param.clover_cpu_prec);

  inv_param.compute_clover = compute_clover;
  if (compute_clover) inv_param.return_clover = 1;
  inv_param.compute_clover_inverse = 1;
  inv_param.return_clover_inverse = 1;
}

void constructQudaCloverField(void *clover, double norm, double diag, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    constructCloverField((double *)clover, norm, diag);
  else
    constructCloverField((float *)clover, norm, diag);
}

void constructWilsonTestSpinorParam(quda::ColorSpinorParam *cs_param, const QudaInvertParam *inv_param,
                                    const QudaGaugeParam *gauge_param)
{
  // Lattice vector spacetime/colour/spin/parity properties
  cs_param->nColor = 3;
  cs_param->nSpin = 4;
  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH || inv_param->dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || inv_param->dslash_type == QUDA_MOBIUS_DWF_DSLASH || inv_param->dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
    cs_param->nDim = 5;
    cs_param->x[4] = inv_param->Ls;
  } else {
    cs_param->nDim = 4;
  }
  for (int d = 0; d < 4; d++) cs_param->x[d] = gauge_param->X[d];
  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION || inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  if (pc) cs_param->x[0] /= 2;
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

void constructRandomSpinorSource(void *v, int nSpin, int nColor, QudaPrecision precision, const int *const x,
                                 quda::RNG &rng)
{
  quda::ColorSpinorParam param;
  param.v = v;
  param.nColor = nColor;
  param.nSpin = nSpin;
  param.setPrecision(precision);
  param.create = QUDA_REFERENCE_FIELD_CREATE;
  param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param.nDim = 4;
  param.siteSubset = QUDA_FULL_SITE_SUBSET;
  param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  param.location = QUDA_CPU_FIELD_LOCATION; // DMH FIXME so one can construct device noise
  for (int d = 0; d < 4; d++) param.x[d] = x[d];
  quda::cpuColorSpinorField spinor_in(param);
  quda::spinorNoise(spinor_in, rng, QUDA_NOISE_UNIFORM);
}

void initComms(int argc, char **argv, std::array<int, 4> &commDims) { initComms(argc, argv, commDims.data()); }

void initComms(int argc, char **argv, int *const commDims)
{
  if (getenv("QUDA_TEST_GRID_SIZE")) get_gridsize_from_env(commDims);

#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);

  // make sure the QMP logical ordering matches QUDA's
  if (rank_order == 0) {
    int map[] = {3, 2, 1, 0};
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  } else {
    int map[] = { 0, 1, 2, 3 };
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  }
#elif defined(MPI_COMMS)
  MPI_Init(&argc, &argv);
#endif

  QudaCommsMap func = rank_order == 0 ? lex_rank_from_coords_t : lex_rank_from_coords_x;

  initCommsGridQuda(4, commDims, func, NULL);
  initRand();

  printfQuda("Rank order is %s major (%s running fastest)\n",
	     rank_order == 0 ? "column" : "row", rank_order == 0 ? "t" : "x");

}

void finalizeComms()
{
#if defined(QMP_COMMS)
  QMP_finalize_msg_passing();
#elif defined(MPI_COMMS)
  MPI_Finalize();
#endif
}

void initRand()
{
  int rank = 0;

#if defined(QMP_COMMS)
  rank = QMP_get_node_number();
#elif defined(MPI_COMMS)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  srand(17*rank + 137);
}

void setDims(int *X) {
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];

    faceVolume[d] = 1;
    for (int i=0; i<4; i++) {
      if (i==d) continue;
      faceVolume[d] *= X[i];
    }
  }
  Vh = V/2;

  Vs_x = X[1]*X[2]*X[3];
  Vs_y = X[0]*X[2]*X[3];
  Vs_z = X[0]*X[1]*X[3];
  Vs_t = X[0]*X[1]*X[2];

  Vsh_x = Vs_x/2;
  Vsh_y = Vs_y/2;
  Vsh_z = Vs_z/2;
  Vsh_t = Vs_t/2;


  E1=X[0]+4; E2=X[1]+4; E3=X[2]+4; E4=X[3]+4;
  E1h=E1/2;
  E[0] = E1;
  E[1] = E2;
  E[2] = E3;
  E[3] = E4;
  V_ex = E1*E2*E3*E4;
  Vh_ex = V_ex/2;

}

void dw_setDims(int *X, const int L5)
{
  V = 1;
  for (int d = 0; d < 4; d++) {
    V *= X[d];
    Z[d] = X[d];

    faceVolume[d] = 1;
    for (int i=0; i<4; i++) {
      if (i==d) continue;
      faceVolume[d] *= X[i];
    }
  }
  Vh = V/2;

  Ls = L5;
  V5 = V*Ls;
  V5h = Vh*Ls;

  Vs_t = Z[0]*Z[1]*Z[2]*Ls;//?
  Vsh_t = Vs_t/2;  //?
}

void setSpinorSiteSize(int n) { my_spinor_site_size = n; }

int dimPartitioned(int dim) { return ((gridsize_from_cmdline[dim] > 1) || dim_partitioned[dim]); }

bool last_node_in_t()
{
  // only apply T-boundary at edge nodes
#ifdef MULTI_GPU
  return commCoords(3) == commDim(3) - 1;
#else
  return true;
#endif
}

int index_4d_cb_from_coordinate_4d(const int coordinate[4], const int dim[4])
{
  return (((coordinate[3] * dim[2] + coordinate[2]) * dim[1] + coordinate[1]) * dim[0] + coordinate[0]) >> 1;
}

void coordinate_from_shrinked_index(int coordinate[4], int shrinked_index, const int shrinked_dim[4],
                                    const int shift[4], int parity)
{
  int aux[4];
  aux[0] = shrinked_index * 2;

  for (int i = 0; i < 3; i++) { aux[i + 1] = aux[i] / shrinked_dim[i]; }

  coordinate[0] = aux[0] - aux[1] * shrinked_dim[0];
  coordinate[1] = aux[1] - aux[2] * shrinked_dim[1];
  coordinate[2] = aux[2] - aux[3] * shrinked_dim[2];
  coordinate[3] = aux[3];

  // Find the full coordinate in the shrinked volume.
  coordinate[0] += (parity + coordinate[3] + coordinate[2] + coordinate[1]) & 1;

  // if(shrinked_index == 3691) printfQuda("coordinate[0] = %d\n", coordinate[0]);
  for (int d = 0; d < 4; d++) { coordinate[d] += shift[d]; }
}

// i represents a "half index" into an even or odd "half lattice".
// when oddBit={0,1} the half lattice is {even,odd}.
//
// the displacements, such as dx, refer to the full lattice coordinates.
//
// neighborIndex() takes a "half index", displaces it, and returns the
// new "half index", which can be an index into either the even or odd lattices.
// displacements of magnitude one always interchange odd and even lattices.
//

int neighborIndex(int i, int oddBit, int dx4, int dx3, int dx2, int dx1)
{
  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y / (Z[2] * Z[1] * Z[0]);
  int x3 = (Y / (Z[1] * Z[0])) % Z[2];
  int x2 = (Y / Z[0]) % Z[1];
  int x1 = Y % Z[0];

  // assert (oddBit == (x+y+z+t)%2);

  x4 = (x4 + dx4 + Z[3]) % Z[3];
  x3 = (x3 + dx3 + Z[2]) % Z[2];
  x2 = (x2 + dx2 + Z[1]) % Z[1];
  x1 = (x1 + dx1 + Z[0]) % Z[0];

  return (x4 * (Z[2] * Z[1] * Z[0]) + x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
}

int neighborIndex(int dim[4], int index, int oddBit, int dx[4])
{

  const int fullIndex = fullLatticeIndex(dim, index, oddBit);

  int x[4];
  x[3] = fullIndex / (dim[2] * dim[1] * dim[0]);
  x[2] = (fullIndex / (dim[1] * dim[0])) % dim[2];
  x[1] = (fullIndex / dim[0]) % dim[1];
  x[0] = fullIndex % dim[0];

  for (int dir = 0; dir < 4; ++dir) x[dir] = (x[dir] + dx[dir] + dim[dir]) % dim[dir];

  return (((x[3] * dim[2] + x[2]) * dim[1] + x[1]) * dim[0] + x[0]) / 2;
}

int neighborIndex_mg(int i, int oddBit, int dx4, int dx3, int dx2, int dx1)
{
  int ret;

  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];

  int ghost_x4 = x4 + dx4;

  // assert (oddBit == (x+y+z+t)%2);

  x4 = (x4 + dx4 + Z[3]) % Z[3];
  x3 = (x3 + dx3 + Z[2]) % Z[2];
  x2 = (x2 + dx2 + Z[1]) % Z[1];
  x1 = (x1 + dx1 + Z[0]) % Z[0];

  if ((ghost_x4 >= 0 && ghost_x4 < Z[3]) || !comm_dim_partitioned(3)) {
    ret = (x4 * (Z[2] * Z[1] * Z[0]) + x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
  } else {
    ret = (x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
  }

  return ret;
}

/*
 * This is a computation of neighbor using the full index and the displacement in each direction
 *
 */

int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1)
{
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh) {
    oddBit = 1;
    half_idx = i - Vh;
  }

  int nbr_half_idx = neighborIndex(half_idx, oddBit, dx4, dx3, dx2, dx1);
  int oddBitChanged = (dx4 + dx3 + dx2 + dx1) % 2;
  if (oddBitChanged) { oddBit = 1 - oddBit; }
  int ret = nbr_half_idx;
  if (oddBit) { ret = Vh + nbr_half_idx; }

  return ret;
}

int neighborIndexFullLattice(int dim[4], int index, int dx[4])
{
  const int volume = dim[0] * dim[1] * dim[2] * dim[3];
  const int halfVolume = volume / 2;
  int oddBit = 0;
  int halfIndex = index;

  if (index >= halfVolume) {
    oddBit = 1;
    halfIndex = index - halfVolume;
  }

  int neighborHalfIndex = neighborIndex(dim, halfIndex, oddBit, dx);

  int oddBitChanged = (dx[0] + dx[1] + dx[2] + dx[3]) % 2;
  if (oddBitChanged) { oddBit = 1 - oddBit; }

  return neighborHalfIndex + oddBit * halfVolume;
}

int neighborIndexFullLattice_mg(int i, int dx4, int dx3, int dx2, int dx1)
{
  int ret;
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh) {
    oddBit = 1;
    half_idx = i - Vh;
  }

  int Y = fullLatticeIndex(half_idx, oddBit);
  int x4 = Y / (Z[2] * Z[1] * Z[0]);
  int x3 = (Y / (Z[1] * Z[0])) % Z[2];
  int x2 = (Y / Z[0]) % Z[1];
  int x1 = Y % Z[0];
  int ghost_x4 = x4 + dx4;

  x4 = (x4 + dx4 + Z[3]) % Z[3];
  x3 = (x3 + dx3 + Z[2]) % Z[2];
  x2 = (x2 + dx2 + Z[1]) % Z[1];
  x1 = (x1 + dx1 + Z[0]) % Z[0];

  if (ghost_x4 >= 0 && ghost_x4 < Z[3]) {
    ret = (x4 * (Z[2] * Z[1] * Z[0]) + x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
  } else {
    ret = (x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
    return ret;
  }

  int oddBitChanged = (dx4 + dx3 + dx2 + dx1) % 2;
  if (oddBitChanged) { oddBit = 1 - oddBit; }

  if (oddBit) { ret += Vh; }

  return ret;
}

// X indexes the lattice site
void printSpinorElement(void *spinor, int X, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    for (int s = 0; s < 4; s++) printVector((double *)spinor + X * 24 + s * 6);
  else
    for (int s = 0; s < 4; s++) printVector((float *)spinor + X * 24 + s * 6);
}

// X indexes the full lattice
void printGaugeElement(void *gauge, int X, QudaPrecision precision)
{
  if (getOddBit(X) == 0) {
    if (precision == QUDA_DOUBLE_PRECISION)
      for (int m = 0; m < 3; m++) printVector((double *)gauge + (X / 2) * gauge_site_size + m * 3 * 2);
    else
      for (int m = 0; m < 3; m++) printVector((float *)gauge + (X / 2) * gauge_site_size + m * 3 * 2);

  } else {
    if (precision == QUDA_DOUBLE_PRECISION)
      for (int m = 0; m < 3; m++) printVector((double *)gauge + (X / 2 + Vh) * gauge_site_size + m * 3 * 2);
    else
      for (int m = 0; m < 3; m++) printVector((float *)gauge + (X / 2 + Vh) * gauge_site_size + m * 3 * 2);
  }
}

int fullLatticeIndex(int dim[4], int index, int oddBit){

  int za = index/(dim[0]>>1);
  int zb = za/dim[1];
  int x2 = za - zb*dim[1];
  int x4 = zb/dim[2];
  int x3 = zb - x4*dim[2];

  return  2*index + ((x2 + x3 + x4 + oddBit) & 1);
}

// given a "half index" i into either an even or odd half lattice (corresponding
// to oddBit = {0, 1}), returns the corresponding full lattice index.
int fullLatticeIndex(int i, int oddBit) {
  /*
    int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
    return 2*i + (boundaryCrossings + oddBit) % 2;
  */

  int X1 = Z[0];
  int X2 = Z[1];
  int X3 = Z[2];
  //int X4 = Z[3];
  int X1h =X1/2;

  int sid =i;
  int za = sid/X1h;
  //int x1h = sid - za*X1h;
  int zb = za/X2;
  int x2 = za - zb*X2;
  int x4 = zb/X3;
  int x3 = zb - x4*X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  //int x1 = 2*x1h + x1odd;
  int X = 2 * sid + x1odd;

  return X;
}

extern "C" {
/**
   @brief Set the default ASAN options.  This ensures that QUDA just
   works when SANITIZE is enabled without requiring ASAN_OPTIONS to
   be set.
 */
const char *__asan_default_options() { return "protect_shadow_gap=0"; }
}

/**
 * For MPI, the default node mapping is lexicographical with t varying fastest.
 */

void get_gridsize_from_env(int *const dims)
{
  char *grid_size_env = getenv("QUDA_TEST_GRID_SIZE");
  if (grid_size_env) {
    std::stringstream grid_list(grid_size_env);

    int dim;
    int i = 0;
    while (grid_list >> dim) {
      if (i >= 4) errorQuda("Unexpected grid size array length");
      dims[i] = dim;
      if (grid_list.peek() == ',') grid_list.ignore();
      i++;
    }
  }
}

int lex_rank_from_coords_t(const int *coords, void *fdata)
{
  int rank = coords[0];
  for (int i = 1; i < 4; i++) { rank = gridsize_from_cmdline[i] * rank + coords[i]; }
  return rank;
}

int lex_rank_from_coords_x(const int *coords, void *fdata)
{
  int rank = coords[3];
  for (int i = 2; i >= 0; i--) { rank = gridsize_from_cmdline[i] * rank + coords[i]; }
  return rank;
}

template <typename Float> void printVector(Float *v)
{
  printfQuda("{(%f %f) (%f %f) (%f %f)}\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}

// returns 0 or 1 if the full lattice index X is even or odd
int getOddBit(int Y)
{
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  return (x4 + x3 + x2 + x1) % 2;
}

// a+=b
template <typename Float> inline void complexAddTo(Float *a, Float *b)
{
  a[0] += b[0];
  a[1] += b[1];
}

// a = b*c
template <typename Float> inline void complexProduct(Float *a, Float *b, Float *c)
{
  a[0] = b[0] * c[0] - b[1] * c[1];
  a[1] = b[0] * c[1] + b[1] * c[0];
}

// a = conj(b)*conj(c)
template <typename Float> inline void complexConjugateProduct(Float *a, Float *b, Float *c)
{
  a[0] = b[0] * c[0] - b[1] * c[1];
  a[1] = -b[0] * c[1] - b[1] * c[0];
}

// a = conj(b)*c
template <typename Float> inline void complexDotProduct(Float *a, Float *b, Float *c)
{
  a[0] = b[0] * c[0] + b[1] * c[1];
  a[1] = b[0] * c[1] - b[1] * c[0];
}

// a += b*c
template <typename Float> inline void accumulateComplexProduct(Float *a, Float *b, Float *c, Float sign)
{
  a[0] += sign * (b[0] * c[0] - b[1] * c[1]);
  a[1] += sign * (b[0] * c[1] + b[1] * c[0]);
}

// a += conj(b)*c)
template <typename Float> inline void accumulateComplexDotProduct(Float *a, Float *b, Float *c)
{
  a[0] += b[0] * c[0] + b[1] * c[1];
  a[1] += b[0] * c[1] - b[1] * c[0];
}

template <typename Float> inline void accumulateConjugateProduct(Float *a, Float *b, Float *c, int sign)
{
  a[0] += sign * (b[0] * c[0] - b[1] * c[1]);
  a[1] -= sign * (b[0] * c[1] + b[1] * c[0]);
}

template <typename Float> inline void su3Construct12(Float *mat)
{
  Float *w = mat + 12;
  w[0] = 0.0;
  w[1] = 0.0;
  w[2] = 0.0;
  w[3] = 0.0;
  w[4] = 0.0;
  w[5] = 0.0;
}

// Stabilized Bunk and Sommer
template <typename Float> inline void su3Construct8(Float *mat)
{
  mat[0] = atan2(mat[1], mat[0]);
  mat[1] = atan2(mat[13], mat[12]);
  for (int i = 8; i < 18; i++) mat[i] = 0.0;
}

void su3_construct(void *mat, QudaReconstructType reconstruct, QudaPrecision precision)
{
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    if (precision == QUDA_DOUBLE_PRECISION)
      su3Construct12((double *)mat);
    else
      su3Construct12((float *)mat);
  } else {
    if (precision == QUDA_DOUBLE_PRECISION)
      su3Construct8((double *)mat);
    else
      su3Construct8((float *)mat);
  }
}

// given first two rows (u,v) of SU(3) matrix mat, reconstruct the third row
// as the cross product of the conjugate vectors: w = u* x v*
//
// 48 flops
template <typename Float> static void su3Reconstruct12(Float *mat, int dir, int ga_idx, QudaGaugeParam *param)
{
  Float *u = &mat[0 * (3 * 2)];
  Float *v = &mat[1 * (3 * 2)];
  Float *w = &mat[2 * (3 * 2)];
  w[0] = 0.0;
  w[1] = 0.0;
  w[2] = 0.0;
  w[3] = 0.0;
  w[4] = 0.0;
  w[5] = 0.0;
  accumulateConjugateProduct(w + 0 * (2), u + 1 * (2), v + 2 * (2), +1);
  accumulateConjugateProduct(w + 0 * (2), u + 2 * (2), v + 1 * (2), -1);
  accumulateConjugateProduct(w + 1 * (2), u + 2 * (2), v + 0 * (2), +1);
  accumulateConjugateProduct(w + 1 * (2), u + 0 * (2), v + 2 * (2), -1);
  accumulateConjugateProduct(w + 2 * (2), u + 0 * (2), v + 1 * (2), +1);
  accumulateConjugateProduct(w + 2 * (2), u + 1 * (2), v + 0 * (2), -1);
  Float u0 = (dir < 3 ? param->anisotropy : (ga_idx >= (Z[3] - 1) * Z[0] * Z[1] * Z[2] / 2 ? param->t_boundary : 1));
  w[0] *= u0;
  w[1] *= u0;
  w[2] *= u0;
  w[3] *= u0;
  w[4] *= u0;
  w[5] *= u0;
}

template <typename Float> static void su3Reconstruct8(Float *mat, int dir, int ga_idx, QudaGaugeParam *param)
{
  // First reconstruct first row
  Float row_sum = 0.0;
  row_sum += mat[2] * mat[2];
  row_sum += mat[3] * mat[3];
  row_sum += mat[4] * mat[4];
  row_sum += mat[5] * mat[5];
  Float u0 = (dir < 3 ? param->anisotropy : (ga_idx >= (Z[3] - 1) * Z[0] * Z[1] * Z[2] / 2 ? param->t_boundary : 1));
  Float U00_mag = sqrt(1.f / (u0 * u0) - row_sum);

  mat[14] = mat[0];
  mat[15] = mat[1];

  mat[0] = U00_mag * cos(mat[14]);
  mat[1] = U00_mag * sin(mat[14]);

  Float column_sum = 0.0;
  for (int i = 0; i < 2; i++) column_sum += mat[i] * mat[i];
  for (int i = 6; i < 8; i++) column_sum += mat[i] * mat[i];
  Float U20_mag = sqrt(1.f / (u0 * u0) - column_sum);

  mat[12] = U20_mag * cos(mat[15]);
  mat[13] = U20_mag * sin(mat[15]);

  // First column now restored

  // finally reconstruct last elements from SU(2) rotation
  Float r_inv2 = 1.0 / (u0 * row_sum);

  // U11
  Float A[2];
  complexDotProduct(A, mat + 0, mat + 6);
  complexConjugateProduct(mat + 8, mat + 12, mat + 4);
  accumulateComplexProduct(mat + 8, A, mat + 2, u0);
  mat[8] *= -r_inv2;
  mat[9] *= -r_inv2;

  // U12
  complexConjugateProduct(mat + 10, mat + 12, mat + 2);
  accumulateComplexProduct(mat + 10, A, mat + 4, -u0);
  mat[10] *= r_inv2;
  mat[11] *= r_inv2;

  // U21
  complexDotProduct(A, mat + 0, mat + 12);
  complexConjugateProduct(mat + 14, mat + 6, mat + 4);
  accumulateComplexProduct(mat + 14, A, mat + 2, -u0);
  mat[14] *= r_inv2;
  mat[15] *= r_inv2;

  // U12
  complexConjugateProduct(mat + 16, mat + 6, mat + 2);
  accumulateComplexProduct(mat + 16, A, mat + 4, u0);
  mat[16] *= -r_inv2;
  mat[17] *= -r_inv2;
}

void su3_reconstruct(void *mat, int dir, int ga_idx, QudaReconstructType reconstruct, QudaPrecision precision,
                     QudaGaugeParam *param)
{
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    if (precision == QUDA_DOUBLE_PRECISION)
      su3Reconstruct12((double *)mat, dir, ga_idx, param);
    else
      su3Reconstruct12((float *)mat, dir, ga_idx, param);
  } else {
    if (precision == QUDA_DOUBLE_PRECISION)
      su3Reconstruct8((double *)mat, dir, ga_idx, param);
    else
      su3Reconstruct8((float *)mat, dir, ga_idx, param);
  }
}

template <typename Float> static int compareFloats(Float *a, Float *b, int len, double epsilon)
{
  for (int i = 0; i < len; i++) {
    double diff = fabs(a[i] - b[i]);
    if (diff > epsilon) {
      printfQuda("ERROR: i=%d, a[%d]=%f, b[%d]=%f\n", i, i, a[i], i, b[i]);
      return 0;
    }
  }
  return 1;
}

int compare_floats(void *a, void *b, int len, double epsilon, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    return compareFloats((double *)a, (double *)b, len, epsilon);
  else
    return compareFloats((float *)a, (float *)b, len, epsilon);
}

// 4d checkerboard.
// given a "half index" i into either an even or odd half lattice (corresponding
// to oddBit = {0, 1}), returns the corresponding full lattice index.
// Cf. GPGPU code in dslash_core_ante.h.
// There, i is the thread index.
int fullLatticeIndex_4d(int i, int oddBit) {
  if (i >= Vh || i < 0) {printf("i out of range in fullLatticeIndex_4d"); exit(-1);}
  /*
    int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
    return 2*i + (boundaryCrossings + oddBit) % 2;
  */

  int X1 = Z[0];
  int X2 = Z[1];
  int X3 = Z[2];
  //int X4 = Z[3];
  int X1h =X1/2;

  int sid =i;
  int za = sid/X1h;
  //int x1h = sid - za*X1h;
  int zb = za/X2;
  int x2 = za - zb*X2;
  int x4 = zb/X3;
  int x3 = zb - x4*X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  //int x1 = 2*x1h + x1odd;
  int X = 2 * sid + x1odd;

  return X;
}

// 5d checkerboard.
// given a "half index" i into either an even or odd half lattice (corresponding
// to oddBit = {0, 1}), returns the corresponding full lattice index.
// Cf. GPGPU code in dslash_core_ante.h.
// There, i is the thread index sid.
// This function is used by neighborIndex_5d in dslash_reference.cpp.
//ok
int fullLatticeIndex_5d(int i, int oddBit) {
  int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2) + i/(Z[3]*Z[2]*Z[1]*Z[0]/2);
  return 2*i + (boundaryCrossings + oddBit) % 2;
}

int fullLatticeIndex_5d_4dpc(int i, int oddBit) {
  int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
  return 2*i + (boundaryCrossings + oddBit) % 2;
}

int x4_from_full_index(int i)
{
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh){
    oddBit =1;
    half_idx = i - Vh;
  }

  int Y = fullLatticeIndex(half_idx, oddBit);
  int x4 = Y/(Z[2]*Z[1]*Z[0]);

  return x4;
}

template <typename Float> void applyGaugeFieldScaling(Float **gauge, int Vh, QudaGaugeParam *param)
{
  // Apply spatial scaling factor (u0) to spatial links
  for (int d = 0; d < 3; d++) {
    for (int i = 0; i < gauge_site_size * Vh * 2; i++) { gauge[d][i] /= param->anisotropy; }
  }

  // Apply boundary conditions to temporal links
  if (param->t_boundary == QUDA_ANTI_PERIODIC_T && last_node_in_t()) {
    for (int j = (Z[0]/2)*Z[1]*Z[2]*(Z[3]-1); j < Vh; j++) {
      for (int i = 0; i < gauge_site_size; i++) {
        gauge[3][j * gauge_site_size + i] *= -1.0;
        gauge[3][(Vh + j) * gauge_site_size + i] *= -1.0;
      }
    }
  }

  if (param->gauge_fix) {
    // set all gauge links (except for the last Z[0]*Z[1]*Z[2]/2) to the identity,
    // to simulate fixing to the temporal gauge.
    int iMax = (last_node_in_t() ? (Z[0] / 2) * Z[1] * Z[2] * (Z[3] - 1) : Vh);
    int dir = 3; // time direction only
    Float *even = gauge[dir];
    Float *odd = gauge[dir] + Vh * gauge_site_size;
    for (int i = 0; i< iMax; i++) {
      for (int m = 0; m < 3; m++) {
	for (int n = 0; n < 3; n++) {
	  even[i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  even[i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	  odd [i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  odd [i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	}
      }
    }
  }
}

// static void constructUnitGaugeField(Float **res, QudaGaugeParam *param) {
template <typename Float> void constructUnitGaugeField(Float **res, QudaGaugeParam *param)
{
  Float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {
    resEven[dir] = res[dir];
    resOdd[dir] = res[dir] + Vh * gauge_site_size;
  }

  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Vh; i++) {
      for (int m = 0; m < 3; m++) {
	for (int n = 0; n < 3; n++) {
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	}
      }
    }
  }

  applyGaugeFieldScaling(res, Vh, param);
}

// normalize the vector a
template <typename Float>
static void normalize(complex<Float> *a, int len) {
  double sum = 0.0;
  for (int i=0; i<len; i++) sum += norm(a[i]);
  for (int i=0; i<len; i++) a[i] /= sqrt(sum);
}

// orthogonalize vector b to vector a
template <typename Float>
static void orthogonalize(complex<Float> *a, complex<Float> *b, int len) {
  complex<double> dot = 0.0;
  for (int i=0; i<len; i++) dot += conj(a[i])*b[i];
  for (int i=0; i<len; i++) b[i] -= (complex<Float>)dot*a[i];
}

template <typename Float> void constructRandomGaugeField(Float **res, QudaGaugeParam *param, QudaDslashType dslash_type)
{
  Float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {
    resEven[dir] = res[dir];
    resOdd[dir] = res[dir] + Vh * gauge_site_size;
  }

  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Vh; i++) {
      for (int m = 1; m < 3; m++) { // last 2 rows
	for (int n = 0; n < 3; n++) { // 3 columns
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
          resOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = rand() / (Float)RAND_MAX;
        }
      }
      normalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), (complex<Float>*)(resEven[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resEven[dir] + (i*3 + 2)*3*2), 3);

      normalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), (complex<Float>*)(resOdd[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resOdd[dir] + (i*3 + 2)*3*2), 3);

      {
	Float *w = resEven[dir]+(i*3+0)*3*2;
	Float *u = resEven[dir]+(i*3+1)*3*2;
	Float *v = resEven[dir]+(i*3+2)*3*2;

        for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }

      {
	Float *w = resOdd[dir]+(i*3+0)*3*2;
	Float *u = resOdd[dir]+(i*3+1)*3*2;
	Float *v = resOdd[dir]+(i*3+2)*3*2;

        for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }

    }
  }

  if (param->type == QUDA_WILSON_LINKS) {
    applyGaugeFieldScaling(res, Vh, param);
  } else if (param->type == QUDA_ASQTAD_LONG_LINKS) {
    applyGaugeFieldScaling_long(res, Vh, param, dslash_type);
  } else if (param->type == QUDA_ASQTAD_FAT_LINKS) {
    for (int dir = 0; dir < 4; dir++) {
      for (int i = 0; i < Vh; i++) {
	for (int m = 0; m < 3; m++) { // last 2 rows
	  for (int n = 0; n < 3; n++) { // 3 columns
	    resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] =1.0* rand() / (Float)RAND_MAX;
	    resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 2.0* rand() / (Float)RAND_MAX;
	    resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = 3.0*rand() / (Float)RAND_MAX;
	    resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 4.0*rand() / (Float)RAND_MAX;
	  }
	}
      }
    }
  }
}

template <typename Float> void constructUnitaryGaugeField(Float **res)
{
  Float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {
    resEven[dir] = res[dir];
    resOdd[dir] = res[dir] + Vh * gauge_site_size;
  }

  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Vh; i++) {
      for (int m = 1; m < 3; m++) { // last 2 rows
	for (int n = 0; n < 3; n++) { // 3 columns
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
          resOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = rand() / (Float)RAND_MAX;
        }
      }
      normalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), (complex<Float>*)(resEven[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resEven[dir] + (i*3 + 2)*3*2), 3);

      normalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), (complex<Float>*)(resOdd[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resOdd[dir] + (i*3 + 2)*3*2), 3);

      {
	Float *w = resEven[dir]+(i*3+0)*3*2;
	Float *u = resEven[dir]+(i*3+1)*3*2;
	Float *v = resEven[dir]+(i*3+2)*3*2;

        for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }

      {
	Float *w = resOdd[dir]+(i*3+0)*3*2;
	Float *u = resOdd[dir]+(i*3+1)*3*2;
	Float *v = resOdd[dir]+(i*3+2)*3*2;

        for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }
    }
  }
}

template <typename Float> void constructCloverField(Float *res, double norm, double diag)
{

  Float c = 2.0 * norm / RAND_MAX;

  for(int i = 0; i < V; i++) {
    for (int j = 0; j < 72; j++) {
      res[i*72 + j] = c*rand() - norm;
    }

    //impose clover symmetry on each chiral block
    for (int ch=0; ch<2; ch++) {
      res[i*72 + 3 + 36*ch] = -res[i*72 + 0 + 36*ch];
      res[i*72 + 4 + 36*ch] = -res[i*72 + 1 + 36*ch];
      res[i*72 + 5 + 36*ch] = -res[i*72 + 2 + 36*ch];
      res[i*72 + 30 + 36*ch] = -res[i*72 + 6 + 36*ch];
      res[i*72 + 31 + 36*ch] = -res[i*72 + 7 + 36*ch];
      res[i*72 + 32 + 36*ch] = -res[i*72 + 8 + 36*ch];
      res[i*72 + 33 + 36*ch] = -res[i*72 + 9 + 36*ch];
      res[i*72 + 34 + 36*ch] = -res[i*72 + 16 + 36*ch];
      res[i*72 + 35 + 36*ch] = -res[i*72 + 17 + 36*ch];
    }

    for (int j = 0; j<6; j++) {
      res[i*72 + j] += diag;
      res[i*72 + j+36] += diag;
    }
  }
}

template <typename Float>
static void checkGauge(Float **oldG, Float **newG, double epsilon) {

  const int fail_check = 17;
  int fail[4][fail_check];
  int iter[4][18];
  for (int d=0; d<4; d++) for (int i=0; i<fail_check; i++) fail[d][i] = 0;
  for (int d=0; d<4; d++) for (int i=0; i<18; i++) iter[d][i] = 0;

  for (int d=0; d<4; d++) {
    for (int eo=0; eo<2; eo++) {
      for (int i=0; i<Vh; i++) {
	int ga_idx = (eo*Vh+i);
	for (int j=0; j<18; j++) {
	  double diff = fabs(newG[d][ga_idx*18+j] - oldG[d][ga_idx*18+j]);/// fabs(oldG[d][ga_idx*18+j]);

	  for (int f=0; f<fail_check; f++) if (diff > pow(10.0,-(f+1))) fail[d][f]++;
	  if (diff > epsilon) iter[d][j]++;
	}
      }
    }
  }

  printf("Component fails (X, Y, Z, T)\n");
  for (int i=0; i<18; i++) printf("%d fails = (%8d, %8d, %8d, %8d)\n", i, iter[0][i], iter[1][i], iter[2][i], iter[3][i]);

  printf("\nDeviation Failures = (X, Y, Z, T)\n");
  for (int f=0; f<fail_check; f++) {
    printf("%e Failures = (%9d, %9d, %9d, %9d) = (%6.5f, %6.5f, %6.5f, %6.5f)\n", pow(10.0, -(f + 1)), fail[0][f],
           fail[1][f], fail[2][f], fail[3][f], fail[0][f] / (double)(V * 18), fail[1][f] / (double)(V * 18),
           fail[2][f] / (double)(V * 18), fail[3][f] / (double)(V * 18));
  }

}

void check_gauge(void **oldG, void **newG, double epsilon, QudaPrecision precision) {
  if (precision == QUDA_DOUBLE_PRECISION)
    checkGauge((double**)oldG, (double**)newG, epsilon);
  else
    checkGauge((float**)oldG, (float**)newG, epsilon);
}

void createSiteLinkCPU(void **link, QudaPrecision precision, int phase)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    constructUnitaryGaugeField((double**)link);
  }else {
    constructUnitaryGaugeField((float**)link);
  }

  if(phase){

    for(int i=0;i < V;i++){
      for(int dir =XUP; dir <= TUP; dir++){
	int idx = i;
	int oddBit =0;
	if (i >= Vh) {
	  idx = i - Vh;
	  oddBit = 1;
	}

	int X1 = Z[0];
	int X2 = Z[1];
	int X3 = Z[2];
	int X4 = Z[3];

	int full_idx = fullLatticeIndex(idx, oddBit);
	int i4 = full_idx /(X3*X2*X1);
	int i3 = (full_idx - i4*(X3*X2*X1))/(X2*X1);
	int i2 = (full_idx - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
        int i1 = full_idx - i4 * (X3 * X2 * X1) - i3 * (X2 * X1) - i2 * X1;

        double coeff= 1.0;
	switch(dir){
	case XUP:
	  if ( (i4 & 1) != 0){
	    coeff *= -1;
	  }
	  break;

	case YUP:
	  if ( ((i4+i1) & 1) != 0){
	    coeff *= -1;
	  }
	  break;

	case ZUP:
	  if ( ((i4+i1+i2) & 1) != 0){
	    coeff *= -1;
	  }
	  break;

        case TUP:
          if (last_node_in_t() && i4 == (X4 - 1)) { coeff *= -1; }
          break;

	default:
	  printf("ERROR: wrong dir(%d)\n", dir);
	  exit(1);
	}

        if (precision == QUDA_DOUBLE_PRECISION){
          // double* mylink = (double*)link;
          // mylink = mylink + (4*i + dir)*gauge_site_size;
          double *mylink = (double *)link[dir];
          mylink = mylink + i * gauge_site_size;

          mylink[12] *= coeff;
          mylink[13] *= coeff;
          mylink[14] *= coeff;
          mylink[15] *= coeff;
          mylink[16] *= coeff;
          mylink[17] *= coeff;

        }else{
          // float* mylink = (float*)link;
          // mylink = mylink + (4*i + dir)*gauge_site_size;
          float *mylink = (float *)link[dir];
          mylink = mylink + i * gauge_site_size;

          mylink[12] *= coeff;
	  mylink[13] *= coeff;
	  mylink[14] *= coeff;
	  mylink[15] *= coeff;
	  mylink[16] *= coeff;
	  mylink[17] *= coeff;
        }
      }
    }
  }

#if 1
  for(int dir= 0;dir < 4;dir++){
    for (int i = 0; i < V * gauge_site_size; i++) {
      if (precision ==QUDA_SINGLE_PRECISION){
	float* f = (float*)link[dir];
	if (f[i] != f[i] || (fabsf(f[i]) > 1.e+3) ){
	  fprintf(stderr, "ERROR:  %dth: bad number(%f) in function %s \n",i, f[i], __FUNCTION__);
	  exit(1);
	}
      }else{
	double* f = (double*)link[dir];
	if (f[i] != f[i] || (fabs(f[i]) > 1.e+3)){
	  fprintf(stderr, "ERROR:  %dth: bad number(%f) in function %s \n",i, f[i], __FUNCTION__);
	  exit(1);
	}
      }
    }
  }
#endif

  return;
}


template <typename Float>
int compareLink(Float **linkA, Float **linkB, int len) {
  const int fail_check = 16;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int iter[18];
  for (int i=0; i<18; i++) iter[i] = 0;

  for(int dir=0;dir < 4; dir++){
    for (int i=0; i<len; i++) {
      for (int j=0; j<18; j++) {
	int is = i*18+j;
	double diff = fabs(linkA[dir][is]-linkB[dir][is]);
	for (int f=0; f<fail_check; f++)
	  if (diff > pow(10.0,-(f+1))) fail[f]++;
	//if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
	if (diff > 1e-3) iter[j]++;
      }
    }
  }

  for (int i=0; i<18; i++) printfQuda("%d fails = %d\n", i, iter[i]);

  int accuracy_level = 0;
  for(int f =0; f < fail_check; f++){
    if(fail[f] == 0){
      accuracy_level =f;
    }
  }

  for (int f=0; f<fail_check; f++) {
    printfQuda("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)), fail[f], 4*len*18, fail[f] / (double)(4*len*18));
  }

  return accuracy_level;
}

static int
compare_link(void **linkA, void **linkB, int len, QudaPrecision precision)
{
  int ret;

  if (precision == QUDA_DOUBLE_PRECISION) {
    ret = compareLink((double**)linkA, (double**)linkB, len);
  } else {
    ret = compareLink((float**)linkA, (float**)linkB, len);
  }

  return ret;
}


// X indexes the lattice site
static void printLinkElement(void *link, int X, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION){
    for (int i = 0; i < 3; i++) { printVector((double *)link + X * gauge_site_size + i * 6); }

  }
  else{
    for (int i = 0; i < 3; i++) { printVector((float *)link + X * gauge_site_size + i * 6); }
  }
}

int strong_check_link(void **linkA, const char *msgA, void **linkB, const char *msgB, int len, QudaPrecision prec)
{
  printfQuda("%s\n", msgA);
  printLinkElement(linkA[0], 0, prec);
  printfQuda("\n");
  printLinkElement(linkA[0], 1, prec);
  printfQuda("...\n");
  printLinkElement(linkA[3], len - 1, prec);
  printfQuda("\n");

  printfQuda("\n%s\n", msgB);
  printLinkElement(linkB[0], 0, prec);
  printfQuda("\n");
  printLinkElement(linkB[0], 1, prec);
  printfQuda("...\n");
  printLinkElement(linkB[3], len - 1, prec);
  printfQuda("\n");

  int ret = compare_link(linkA, linkB, len, prec);
  return ret;
}

void createMomCPU(void *mom, QudaPrecision precision)
{
  void* temp;

  size_t gSize = (precision == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  temp = malloc(4 * V * gauge_site_size * gSize);
  if (temp == NULL){
    fprintf(stderr, "Error: malloc failed for temp in function %s\n", __FUNCTION__);
    exit(1);
  }

  for(int i=0;i < V;i++){
    if (precision == QUDA_DOUBLE_PRECISION){
      for(int dir=0;dir < 4;dir++){
        double *thismom = (double *)mom;
        for (int k = 0; k < mom_site_size; k++) {
          thismom[(4 * i + dir) * mom_site_size + k] = 1.0 * rand() / RAND_MAX;
          if (k == mom_site_size - 1) thismom[(4 * i + dir) * mom_site_size + k] = 0.0;
        }
      }
    }else{
      for(int dir=0;dir < 4;dir++){
	float* thismom=(float*)mom;
        for (int k = 0; k < mom_site_size; k++) {
          thismom[(4 * i + dir) * mom_site_size + k] = 1.0 * rand() / RAND_MAX;
          if (k == mom_site_size - 1) thismom[(4 * i + dir) * mom_site_size + k] = 0.0;
        }
      }
    }
  }

  free(temp);
  return;
}

void createHwCPU(void *hw, QudaPrecision precision)
{
  for(int i=0;i < V;i++){
    if (precision == QUDA_DOUBLE_PRECISION){
      for(int dir=0;dir < 4;dir++){
	double* thishw = (double*)hw;
        for (int k = 0; k < hw_site_size; k++) { thishw[(4 * i + dir) * hw_site_size + k] = 1.0 * rand() / RAND_MAX; }
      }
    }else{
      for(int dir=0;dir < 4;dir++){
	float* thishw=(float*)hw;
        for (int k = 0; k < hw_site_size; k++) { thishw[(4 * i + dir) * hw_site_size + k] = 1.0 * rand() / RAND_MAX; }
      }
    }
  }

  return;
}


template <typename Float>
int compare_mom(Float *momA, Float *momB, int len) {
  const int fail_check = 16;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int iter[mom_site_size];
  for (int i = 0; i < mom_site_size; i++) iter[i] = 0;

  for (int i=0; i<len; i++) {
    for (int j = 0; j < mom_site_size - 1; j++) {
      int is = i * mom_site_size + j;
      double diff = fabs(momA[is]-momB[is]);
      for (int f=0; f<fail_check; f++)
	if (diff > pow(10.0,-(f+1))) fail[f]++;
      //if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
      if (diff > 1e-3) iter[j]++;
    }
  }

  int accuracy_level = 0;
  for(int f =0; f < fail_check; f++){
    if(fail[f] == 0){
      accuracy_level =f+1;
    }
  }

  for (int i = 0; i < mom_site_size; i++) printfQuda("%d fails = %d\n", i, iter[i]);

  for (int f=0; f<fail_check; f++) {
    printfQuda("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)), fail[f], len*9, fail[f]/(double)(len*9));
  }

  return accuracy_level;
}

static void printMomElement(void *mom, int X, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION){
    double *thismom = ((double *)mom) + X * mom_site_size;
    printVector(thismom);
    printfQuda("(%9f,%9f) (%9f,%9f)\n", thismom[6], thismom[7], thismom[8], thismom[9]);
  }else{
    float *thismom = ((float *)mom) + X * mom_site_size;
    printVector(thismom);
    printfQuda("(%9f,%9f) (%9f,%9f)\n", thismom[6], thismom[7], thismom[8], thismom[9]);
  }
}

int strong_check_mom(void *momA, void *momB, int len, QudaPrecision prec)
{
  printfQuda("mom:\n");
  printMomElement(momA, 0, prec);
  printfQuda("\n");
  printMomElement(momA, 1, prec);
  printfQuda("\n");
  printMomElement(momA, 2, prec);
  printfQuda("\n");
  printMomElement(momA, 3, prec);
  printfQuda("...\n");

  printfQuda("\nreference mom:\n");
  printMomElement(momB, 0, prec);
  printfQuda("\n");
  printMomElement(momB, 1, prec);
  printfQuda("\n");
  printMomElement(momB, 2, prec);
  printfQuda("\n");
  printMomElement(momB, 3, prec);
  printfQuda("\n");

  int ret;
  if (prec == QUDA_DOUBLE_PRECISION){
    ret = compare_mom((double*)momA, (double*)momB, len);
  }else{
    ret = compare_mom((float*)momA, (float*)momB, len);
  }

  return ret;
}

// compute the magnitude squared anti-Hermitian matrix, including the
// MILC convention of subtracting 4 from each site norm to improve
// stability
template <typename real> double mom_action(real *mom_, int len)
{
  double action = 0.0;
  for (int i = 0; i < len; i++) {
    real *mom = mom_ + i * mom_site_size;
    double local = 0.0;
    for (int j = 0; j < 6; j++) local += mom[j] * mom[j];
    for (int j = 6; j < 9; j++) local += 0.5 * mom[j] * mom[j];
    local -= 4.0;
    action += local;
  }

  return action;
}

double mom_action(void *mom, QudaPrecision prec, int len)
{
  double action = 0.0;
  if (prec == QUDA_DOUBLE_PRECISION) {
    action = mom_action<double>((double *)mom, len);
  } else if (prec == QUDA_SINGLE_PRECISION) {
    action = mom_action<float>((float *)mom, len);
  }
  comm_allreduce(&action);
  return action;
}

static struct timeval startTime;

void stopwatchStart() { gettimeofday(&startTime, NULL); }

double stopwatchReadSeconds()
{
  struct timeval endTime;
  gettimeofday(&endTime, NULL);

  long ds = endTime.tv_sec - startTime.tv_sec;
  long dus = endTime.tv_usec - startTime.tv_usec;
  return ds + 0.000001*dus;
}

void performanceStats(double *time, double *gflops)
{
  auto mean_time = 0.0;
  auto mean_time2 = 0.0;
  auto mean_gflops = 0.0;
  auto mean_gflops2 = 0.0;
  // skip first solve due to allocations, potential UVM swapping overhead
  for (int i = 1; i < Nsrc; i++) {
    mean_time += time[i];
    mean_time2 += time[i] * time[i];
    mean_gflops += gflops[i];
    mean_gflops2 += gflops[i] * gflops[i];
  }

  auto NsrcM1 = Nsrc - 1;

  mean_time /= NsrcM1;
  mean_time2 /= NsrcM1;
  auto stddev_time = NsrcM1 > 1 ? sqrt((NsrcM1 / ((double)NsrcM1 - 1.0)) * (mean_time2 - mean_time * mean_time)) :
                                  std::numeric_limits<double>::infinity();
  mean_gflops /= NsrcM1;
  mean_gflops2 /= NsrcM1;
  auto stddev_gflops = NsrcM1 > 1 ? sqrt((NsrcM1 / ((double)NsrcM1 - 1.0)) * (mean_gflops2 - mean_gflops * mean_gflops)) :
                                    std::numeric_limits<double>::infinity();
  printfQuda("%d solves, with mean solve time %g (stddev = %g), mean GFLOPS %g (stddev = %g) [excluding first solve]\n",
             Nsrc, mean_time, stddev_time, mean_gflops, stddev_gflops);
}
