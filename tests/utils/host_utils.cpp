#include <limits>
#include <complex>
#include <vector>
#include <random>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <short.h>

#include <comm_quda.h>

// This contains the appropriate ifdef guards already
#include <mpi_comm_handle.h>

// QUDA headers
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <unitarization_links.h>
#include <dirac_quda.h>
#include <qio_field.h>

// External headers
#include "llfat_utils.h"
#include "gauge_utils.h"
#include "staggered_gauge_utils.h"
#include "host_utils.h"
#include "instantiate_host.hpp"
#include "rng_utils.hpp"
#include "index_utils.hpp"
#include "command_line_params.h"
#include "misc.h"

template <typename T> using complex = std::complex<T>;

int Z[4];
int V;
int Vh;
int Vs_x, Vs_y, Vs_z, Vs_t;
int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
int faceVolume[4];

// extended volume, +4
int E1, E1h, E2, E3, E4;
int E[4];
int V_ex, Vh_ex;

int Ls;
int V5;
int V5h;
double kappa5;

extern float fat_link_max;

// Set some local QUDA precision variables
QudaPrecision local_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision &cpu_prec = local_prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_refinement_sloppy = prec_refinement_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;
QudaPrecision &cuda_prec_eigensolver = prec_eigensolver;
QudaPrecision &cuda_prec_ritz = prec_ritz;

// Host hypercubic RNG
std::vector<std::mt19937_64> host_rand;

size_t host_gauge_data_type_size = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
size_t host_spinor_data_type_size = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
size_t host_clover_data_type_size = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

void setQudaPrecisions()
{
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_eigensolver == QUDA_INVALID_PRECISION) prec_eigensolver = prec_sloppy;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
  if (smoother_halo_prec == QUDA_INVALID_PRECISION) smoother_halo_prec = prec_null;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;
  if (link_recon_eigensolver == QUDA_RECONSTRUCT_INVALID) link_recon_eigensolver = link_recon_sloppy;
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
#ifdef QUDA_MMA_AVAILABLE
    mg_setup_use_mma[i] = true;
#else
    mg_setup_use_mma[i] = false;
#endif
    mg_dslash_use_mma[i] = false;
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
    smoother_type[i] = QUDA_MR_INVERTER;
    smoother_tol[i] = 0.25;
    coarse_solver[i] = QUDA_GCR_INVERTER;
    coarse_solver_tol[i] = 0.25;
    coarse_solver_maxiter[i] = 100;
    solver_location[i] = QUDA_CUDA_FIELD_LOCATION;
    setup_location[i] = QUDA_CUDA_FIELD_LOCATION;
    nu_pre[i] = 2;
    nu_post[i] = 2;
    n_block_ortho[i] = 1;
    block_ortho_two_pass[i] = true;

    // Default eigensolver params
    mg_eig[i] = false;
    mg_eig_tol[i] = 1e-3;
    mg_eig_n_ev[i] = nvec[i];
    mg_eig_n_kr[i] = 3 * nvec[i];
    mg_eig_require_convergence[i] = QUDA_BOOLEAN_TRUE;
    mg_eig_type[i] = QUDA_EIG_TR_LANCZOS;
    mg_eig_spectrum[i] = QUDA_SPECTRUM_SR_EIG;
    mg_eig_check_interval[i] = 5;
    mg_eig_max_restarts[i] = 100;
    mg_eig_max_ortho_attempts[i] = 10;
    mg_eig_use_normop[i] = QUDA_BOOLEAN_FALSE;
    mg_eig_use_dagger[i] = QUDA_BOOLEAN_FALSE;
    mg_eig_use_poly_acc[i] = QUDA_BOOLEAN_TRUE;
    mg_eig_poly_deg[i] = 100;
    mg_eig_amin[i] = 1.0;
    mg_eig_amax[i] = -1.0; // use power iterations
    mg_eig_save_prec[i] = QUDA_DOUBLE_PRECISION;

    setup_ca_basis[i] = QUDA_POWER_BASIS;
    setup_ca_basis_size[i] = 4;
    setup_ca_lambda_min[i] = 0.0;
    setup_ca_lambda_max[i] = -1.0; // use power iterations

    coarse_solver_ca_basis[i] = QUDA_POWER_BASIS;
    coarse_solver_ca_basis_size[i] = 4;
    coarse_solver_ca_lambda_min[i] = 0.0;
    coarse_solver_ca_lambda_max[i] = -1.0;

    smoother_solver_ca_basis[i] = QUDA_POWER_BASIS;
    smoother_solver_ca_lambda_min[i] = 0.0;
    smoother_solver_ca_lambda_max[i] = -1.0; // use power iterations
  }
}

void constructHostCloverField(void *clover, void *, QudaInvertParam &inv_param)
{
  double norm = 0.01; // clover components are random numbers in the range (-norm, norm)
  double diag = 1.0;  // constant added to the diagonal

  if (!compute_clover) constructQudaCloverField(clover, norm, diag, inv_param.clover_cpu_prec);

  inv_param.compute_clover = compute_clover;
  if (compute_clover) inv_param.return_clover = 1;
  inv_param.compute_clover_inverse = 1;
  inv_param.return_clover_inverse = 1;
}

/**
 * @brief Construct a random (but reasonable) clover field
 *
 * @tparam real_t Floating point type
 * @param[out] clover The clover field
 * @param[in] norm Scale factor for clover field elements
 * @param[in] diag Diagonal addition to the clover field
 */
template <typename real_t> struct ConstructCloverField {
  void operator()(void *res, double norm, double diag)
  {
#pragma omp parallel for
    for (auto i = 0lu; i < static_cast<size_t>(Vh); i++) {
      for (auto parity = 0lu; parity < 2lu; parity++) {
        auto clover_matrix = reinterpret_cast<real_t *>(res) + 72 * (parity * Vh + i);
        for (int j = 0; j < 72; j++) { clover_matrix[j] = random_uniform_host<real_t>(i, parity, -norm, norm); }

        // impose clover symmetry on each chiral block
        for (int ch = 0; ch < 2; ch++) {
          clover_matrix[3 + 36 * ch] = -clover_matrix[0 + 36 * ch];
          clover_matrix[4 + 36 * ch] = -clover_matrix[1 + 36 * ch];
          clover_matrix[5 + 36 * ch] = -clover_matrix[2 + 36 * ch];
          clover_matrix[30 + 36 * ch] = -clover_matrix[6 + 36 * ch];
          clover_matrix[31 + 36 * ch] = -clover_matrix[7 + 36 * ch];
          clover_matrix[32 + 36 * ch] = -clover_matrix[8 + 36 * ch];
          clover_matrix[33 + 36 * ch] = -clover_matrix[9 + 36 * ch];
          clover_matrix[34 + 36 * ch] = -clover_matrix[16 + 36 * ch];
          clover_matrix[35 + 36 * ch] = -clover_matrix[17 + 36 * ch];
        }

        for (int j = 0; j < 6; j++) {
          clover_matrix[j] += diag;
          clover_matrix[j + 36] += diag;
        }
      }
    }
  }
};

void constructQudaCloverField(void *clover, double norm, double diag, QudaPrecision precision)
{
  instantiate_host<ConstructCloverField>(precision, clover, norm, diag);
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
  } else if ((inv_param->dslash_type == QUDA_TWISTED_MASS_DSLASH || inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
             && (inv_param->twist_flavor == QUDA_TWIST_NONDEG_DOUBLET)) {
    cs_param->nDim = 5;
    cs_param->x[4] = 2;
  } else {
    cs_param->nDim = 4;
  }
  cs_param->twistFlavor = inv_param->twist_flavor;
  cs_param->pc_type = inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH ? QUDA_5D_PC : QUDA_4D_PC;
  for (int d = 0; d < 4; d++) cs_param->x[d] = gauge_param->X[d];
  bool pc = is_pc_solution(inv_param->solution_type);
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

void constructRandomSpinorSource(void *v, int nSpin, int nColor, QudaPrecision precision, QudaSolutionType sol_type,
                                 const int *const x, int nDim, quda::RNG &rng)
{
  quda::ColorSpinorParam param;
  param.v = v;
  param.nColor = nColor;
  param.nSpin = nSpin;
  param.setPrecision(precision);
  param.create = QUDA_REFERENCE_FIELD_CREATE;
  param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param.nDim = nDim;
  param.pc_type = QUDA_4D_PC;
  param.siteSubset = is_pc_solution(sol_type) ? QUDA_PARITY_SITE_SUBSET : QUDA_FULL_SITE_SUBSET;
  param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  param.location = QUDA_CPU_FIELD_LOCATION; // DMH FIXME so one can construct device noise
  for (int d = 0; d < nDim; d++) param.x[d] = x[d];
  if (is_pc_solution(sol_type)) param.x[0] /= 2;
  quda::ColorSpinorField spinor_in(param);
  quda::spinorNoise(spinor_in, rng, QUDA_NOISE_UNIFORM);
}

// Helper functions
bool is_pc_solution(QudaSolutionType type)
{
  switch (type) {
  case QUDA_MATPC_SOLUTION:
  case QUDA_MATPC_DAG_SOLUTION:
  case QUDA_MATPCDAG_MATPC_SOLUTION:
  case QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION: return true;
  default: return false;
  }
}

bool is_full_solution(QudaSolutionType type)
{
  switch (type) {
  case QUDA_MAT_SOLUTION:
  case QUDA_MATDAG_MAT_SOLUTION: return true;
  default: return false;
  }
}

bool is_full_solve(QudaSolveType type)
{
  switch (type) {
  case QUDA_DIRECT_SOLVE:
  case QUDA_NORMOP_SOLVE:
  case QUDA_NORMERR_SOLVE: return true;
  default: return false;
  }
}

bool is_preconditioned_solve(QudaSolveType type)
{
  switch (type) {
  case QUDA_DIRECT_PC_SOLVE:
  case QUDA_NORMOP_PC_SOLVE:
  case QUDA_NORMERR_PC_SOLVE: return true;
  default: return false;
  }
}

bool is_normal_solve(QudaInverterType inv_type, QudaSolveType solve_type)
{
  switch (solve_type) {
  case QUDA_NORMOP_SOLVE:
  case QUDA_NORMOP_PC_SOLVE: return true;
  default:
    switch (inv_type) {
    case QUDA_CGNR_INVERTER:
    case QUDA_CGNE_INVERTER:
    case QUDA_CA_CGNR_INVERTER:
    case QUDA_CA_CGNE_INVERTER: return true;
    default: return false;
    }
  }
}

bool is_hermitian_solver(QudaInverterType type)
{
  switch (type) {
  case QUDA_CG_INVERTER:
  case QUDA_CA_CG_INVERTER: return true;
  default: return false;
  }
}

bool support_solution_accumulator_pipeline(QudaInverterType type)
{
  switch (type) {
  case QUDA_CG_INVERTER:
  case QUDA_CA_CG_INVERTER:
  case QUDA_CGNR_INVERTER:
  case QUDA_CGNE_INVERTER:
  case QUDA_PCG_INVERTER: return true;
  default: return false;
  }
}

bool is_normal_residual(QudaInverterType type)
{
  switch (type) {
  case QUDA_CGNR_INVERTER:
  case QUDA_CG3NR_INVERTER:
  case QUDA_CA_CGNR_INVERTER: return true;
  default: return false;
  }
}

bool is_staggered(QudaDslashType type) { return quda::Dirac::is_staggered_type(type); }

bool is_chiral(QudaDslashType type) { return quda::Dirac::is_dwf(type); }

bool is_laplace(QudaDslashType type)
{
  switch (type) {
  case QUDA_LAPLACE_DSLASH: return true;
  default: return false;
  }
}

void initComms(int argc, char **argv, std::array<int, 4> &commDims) { initComms(argc, argv, commDims.data()); }

#if defined(QMP_COMMS) || defined(MPI_COMMS)
void initComms(int argc, char **argv, int *const commDims)
#else
void initComms(int, char **, int *const commDims)
#endif
{
  if (getenv("QUDA_TEST_GRID_SIZE")) { get_size_from_env(commDims, "QUDA_TEST_GRID_SIZE"); }
  if (getenv("QUDA_TEST_GRID_PARTITION")) { get_size_from_env(grid_partition.data(), "QUDA_TEST_GRID_PARTITION"); }

#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_FUNNELED, &tl);

  // make sure the QMP logical ordering matches QUDA's
  if (rank_order == 0) {
    int map[] = {3, 2, 1, 0};
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  } else {
    int map[] = {0, 1, 2, 3};
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  }
#elif defined(MPI_COMMS)
  int provided = 0;
  int required = MPI_THREAD_FUNNELED;
  int flag = MPI_Init_thread(&argc, &argv, required, &provided);

  if (provided != required) {
    printf("%s: required thread-safety level %d can't be provided %d\n", __func__, required, provided);
    fflush(stdout);
    exit(flag);
  }
#endif

  QudaCommsMap func = rank_order == 0 ? lex_rank_from_coords_t : lex_rank_from_coords_x;

  initCommsGridQuda(4, commDims, func, NULL);

  for (int d = 0; d < 4; d++) {
    if (dim_partitioned[d]) { quda::commDimPartitionedSet(d); }
  }

  initRand();

  printfQuda("Rank order is %s major (%s running fastest)\n", rank_order == 0 ? "column" : "row",
             rank_order == 0 ? "t" : "x");
}

void finalizeComms()
{
  quda::comm_finalize();
#if defined(QMP_COMMS)
  QMP_finalize_msg_passing();
#elif defined(MPI_COMMS)
  MPI_Finalize();
#endif
}

void initRand()
{
  using quda::comm_coord;
  using quda::comm_dim;

  int rank = 0;

#if defined(QMP_COMMS)
  rank = QMP_get_node_number();
#elif defined(MPI_COMMS)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  srand(17 * rank + 137);

  // initialize the hypercubic RNG
  std::array<int, 4> X = {xdim, ydim, zdim, tdim};
  int volume = X[0] * X[1] * X[2] * X[3];
  int volume_h = volume / 2;

  host_rand.resize(volume);
  std::array<uint64_t, 4> X_global;
  for (int d = 0; d < 4; d++) X_global[d] = static_cast<uint64_t>(X[d] * comm_dim(d));

  for (int parity = 0; parity < 2; parity++)
    for (int i = 0; i < volume_h; i++) {
      // get the local coordinate
      std::array<uint64_t, 4> x;
      getCoords(x, i, X, parity);
      for (int d = 0; d < 4; d++) x[d] += X[d] * comm_coord(d);
      uint64_t global_idx = (((x[3] * X_global[2] + x[2]) * X_global[1]) + x[1]) * X_global[0] + x[0];
      host_rand[parity * volume_h + i] = std::mt19937_64(17ul * global_idx + 137);
    }
}

void setDims(int *X)
{
  V = 1;
  for (int d = 0; d < 4; d++) {
    V *= X[d];
    Z[d] = X[d];

    faceVolume[d] = 1;
    for (int i = 0; i < 4; i++) {
      if (i == d) continue;
      faceVolume[d] *= X[i];
    }
  }
  Vh = V / 2;

  Vs_x = X[1] * X[2] * X[3];
  Vs_y = X[0] * X[2] * X[3];
  Vs_z = X[0] * X[1] * X[3];
  Vs_t = X[0] * X[1] * X[2];

  Vsh_x = Vs_x / 2;
  Vsh_y = Vs_y / 2;
  Vsh_z = Vs_z / 2;
  Vsh_t = Vs_t / 2;

  E1 = X[0] + 4;
  E2 = X[1] + 4;
  E3 = X[2] + 4;
  E4 = X[3] + 4;
  E1h = E1 / 2;
  E[0] = E1;
  E[1] = E2;
  E[2] = E3;
  E[3] = E4;
  V_ex = E1 * E2 * E3 * E4;
  Vh_ex = V_ex / 2;
}

void dw_setDims(int *X, const int L5)
{
  V = 1;
  for (int d = 0; d < 4; d++) {
    V *= X[d];
    Z[d] = X[d];

    faceVolume[d] = 1;
    for (int i = 0; i < 4; i++) {
      if (i == d) continue;
      faceVolume[d] *= X[i];
    }
  }
  Vh = V / 2;

  Ls = L5;
  V5 = V * Ls;
  V5h = Vh * Ls;

  Vs_t = Z[0] * Z[1] * Z[2] * Ls; //?
  Vsh_t = Vs_t / 2;               //?
}

int dimPartitioned(int dim) { return ((gridsize_from_cmdline[dim] > 1) || dim_partitioned[dim]); }

bool last_node_in_t()
{
  // only apply T-boundary at edge nodes
#ifdef MULTI_GPU
  return quda::commCoords(3) == quda::commDim(3) - 1;
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

int neighborIndex_mg(int i, int oddBit, int dx4, int dx3, int dx2, int dx1)
{
  int ret;

  int Y = fullLatticeIndex(i, oddBit);
  int x4 = Y / (Z[2] * Z[1] * Z[0]);
  int x3 = (Y / (Z[1] * Z[0])) % Z[2];
  int x2 = (Y / Z[0]) % Z[1];
  int x1 = Y % Z[0];

  int ghost_x4 = x4 + dx4;

  // assert (oddBit == (x+y+z+t)%2);

  x4 = (x4 + dx4 + Z[3]) % Z[3];
  x3 = (x3 + dx3 + Z[2]) % Z[2];
  x2 = (x2 + dx2 + Z[1]) % Z[1];
  x1 = (x1 + dx1 + Z[0]) % Z[0];

  if ((ghost_x4 >= 0 && ghost_x4 < Z[3]) || !quda::comm_dim_partitioned(3)) {
    ret = (x4 * (Z[2] * Z[1] * Z[0]) + x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
  } else {
    ret = (x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
  }

  return ret;
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

/**
 * For MPI, the default node mapping is lexicographical with t varying fastest.
 */

void get_size_from_env(int *const dims, const char env[])
{
  char *grid_size_env = getenv(env);
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

int lex_rank_from_coords_t(const int *coords, void *)
{
  int rank = coords[0];
  for (int i = 1; i < 4; i++) { rank = gridsize_from_cmdline[i] * rank + coords[i]; }
  return rank;
}

int lex_rank_from_coords_x(const int *coords, void *)
{
  int rank = coords[3];
  for (int i = 2; i >= 0; i--) { rank = gridsize_from_cmdline[i] * rank + coords[i]; }
  return rank;
}

// returns 0 or 1 if the full lattice index X is even or odd
int getOddBit(int Y)
{
  int x4 = Y / (Z[2] * Z[1] * Z[0]);
  int x3 = (Y / (Z[1] * Z[0])) % Z[2];
  int x2 = (Y / Z[0]) % Z[1];
  int x1 = Y % Z[0];
  return (x4 + x3 + x2 + x1) % 2;
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
    if (diff > epsilon || std::isnan(diff)) {
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

template <typename Float> static double compareFloats_v2(Float *a, Float *b, int len, double epsilon)
{
  double global_diff = 0.0;
  for (int i = 0; i < len; i++) {
    double diff = fabs(a[i] - b[i]);
    if (diff > epsilon || std::isnan(diff)) {
      //printfQuda("ERROR: i=%d, a[%d]=%f, b[%d]=%f\n", i, i, a[i], i, b[i]);
      return diff;
    }
    global_diff = std::max(global_diff, diff);
  }
  return global_diff;
}

// returns deviation instead of failure flag
double compare_floats_v2(void *a, void *b, int len, double epsilon, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    return compareFloats_v2((double *)a, (double *)b, len, epsilon);
  else
    return compareFloats_v2((float *)a, (float *)b, len, epsilon);
}

template <typename Float> static void checkGauge(Float **oldG, Float **newG, double epsilon)
{

  const int fail_check = 17;
  int fail[4][fail_check];
  int iter[4][18];
  for (int d = 0; d < 4; d++)
    for (int i = 0; i < fail_check; i++) fail[d][i] = 0;
  for (int d = 0; d < 4; d++)
    for (int i = 0; i < 18; i++) iter[d][i] = 0;

  for (int d = 0; d < 4; d++) {
    for (int eo = 0; eo < 2; eo++) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        int ga_idx = (eo * Vh + i);
        for (int j = 0; j < 18; j++) {
          double diff = fabs(newG[d][ga_idx * 18 + j] - oldG[d][ga_idx * 18 + j]); /// fabs(oldG[d][ga_idx*18+j]);

          for (int f = 0; f < fail_check; f++)
            if (diff > pow(10.0, -(f + 1)) || std::isnan(diff)) {
#pragma omp atomic
              fail[d][f]++;
            }
          if (diff > epsilon || std::isnan(diff)) {
#pragma omp atomic
            iter[d][j]++;
          }
        }
      }
    }
  }

  printf("Component fails (X, Y, Z, T)\n");
  for (int i = 0; i < 18; i++)
    printf("%d fails = (%8d, %8d, %8d, %8d)\n", i, iter[0][i], iter[1][i], iter[2][i], iter[3][i]);

  printf("\nDeviation Failures = (X, Y, Z, T)\n");
  for (int f = 0; f < fail_check; f++) {
    printf("%e Failures = (%9d, %9d, %9d, %9d) = (%6.5f, %6.5f, %6.5f, %6.5f)\n", pow(10.0, -(f + 1)), fail[0][f],
           fail[1][f], fail[2][f], fail[3][f], fail[0][f] / (double)(V * 18), fail[1][f] / (double)(V * 18),
           fail[2][f] / (double)(V * 18), fail[3][f] / (double)(V * 18));
  }
}

void check_gauge(void **oldG, void **newG, double epsilon, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    checkGauge((double **)oldG, (double **)newG, epsilon);
  else
    checkGauge((float **)oldG, (float **)newG, epsilon);
}

std::complex<double> twoColorSpinorContract(std::complex<double> *spinor1, std::complex<double> *spinor2)
{
  int col_inc = 3;

  std::vector<int> col_st {0, 1, 2};
  std::vector<int> row_st {0, 3, 6};

  std::vector<complex<double>> test_contract(9 * V);
  complex<double> trace = {0., 0.};
  double trace_re, trace_im;
  for (int i = 0; i < V; i++) {

    for (int ii = 0; ii < 9; ii++) {
      int which_col_idx = (ii % 3), which_row_idx = (ii - (ii % 3)) / 3;

      std::complex<double> dot = {0., 0.};

      for (int i_s = 0; i_s < 4; i_s++) {

        int s_row_idx = i * 12 + col_st[which_row_idx] + i_s * col_inc;
        int s_col_idx = i * 12 + col_st[which_col_idx] + i_s * col_inc;

        auto m1 = std::conj(spinor1[s_row_idx]);
        auto m2 = spinor2[s_col_idx];

        dot += m1 * m2;
      }
      test_contract[i * 9 + ii] = dot;
    }
    trace += (test_contract[i * 9] + test_contract[i * 9 + 4] + test_contract[i * 9 + 8]);
  }
  trace_re = trace.real();
  trace_im = trace.imag();
  quda::comm_allreduce_sum(trace_re);
  quda::comm_allreduce_sum(trace_im);

  std::complex<double> trace_fin = {trace_re, trace_im};
  return trace_fin;
}

void createSiteLinkCPU(void *const *gauge, QudaPrecision precision, SiteLinkType phase)
{
  if (phase == SiteLinkType::SITELINK_PHASE_NO) {
    constructRandomSU3GaugeField(gauge, precision);
  } else if (phase == SiteLinkType::SITELINK_PHASE_MILC) {
    constructRandomSU3GaugeField(gauge, precision);
    applyGaugeStaggeredPhase(gauge, Vh, Z, precision, QUDA_ANTI_PERIODIC_T, QUDA_STAGGERED_PHASE_MILC);
  } else if (phase == SiteLinkType::SITELINK_PHASE_U1) {
    constructRandomSU3GaugeField(gauge, precision);
    applyRandomU1Phase(gauge, precision);
  } else if (phase == SiteLinkType::SITELINK_RANDOM) {
    constructRandomMatrixGaugeField(gauge, precision);
  } else if (phase == SiteLinkType::SITELINK_NOISY) {
    constructRandomSU3GaugeField(gauge, precision);

    // this 1/40 is relatively arbitrary, but it's made to add a bit of perturbative
    // noise that can be re-unitarized away
    addNoiseToGaugeField(gauge, 1.0 / 40.0, precision);
  }
}

void createSiteLinkCPU(quda::GaugeField &gauge, QudaPrecision precision, SiteLinkType phase)
{
  if (gauge.Order() == QUDA_QDP_GAUGE_ORDER) {
    createSiteLinkCPU(static_cast<void *const *>(gauge.raw_pointer()), precision, phase);
  } else {
    quda::GaugeFieldParam param(gauge);
    param.order = QUDA_QDP_GAUGE_ORDER;
    param.create = QUDA_NULL_FIELD_CREATE;
    quda::GaugeField u(param);
    createSiteLinkCPU(static_cast<void *const *>(u.raw_pointer()), precision, phase);
    gauge = u;
  }
}

template <typename Float> int compareLink(Float **linkA, Float **linkB, int len)
{
  const int fail_check = 16;
  int fail[fail_check];
  for (int f = 0; f < fail_check; f++) fail[f] = 0;

  int iter[18];
  for (int i = 0; i < 18; i++) iter[i] = 0;

  for (int dir = 0; dir < 4; dir++) {
#pragma omp parallel for
    for (int i = 0; i < len; i++) {
      for (int j = 0; j < 18; j++) {
        int is = i * 18 + j;
        double diff = fabs(linkA[dir][is] - linkB[dir][is]);
        for (int f = 0; f < fail_check; f++)
          if (diff > pow(10.0, -(f + 1)) || std::isnan(diff)) {
#pragma omp atomic
            fail[f]++;
          }
        // if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
        if (diff > 1e-3 || std::isnan(diff)) {
#pragma omp atomic
          iter[j]++;
        }
      }
    }
  }

  for (int i = 0; i < 18; i++) printfQuda("%d fails = %d\n", i, iter[i]);

  int accuracy_level = 0;
  for (int f = 0; f < fail_check; f++) {
    if (fail[f] == 0) { accuracy_level = f; }
  }

  for (int f = 0; f < fail_check; f++) {
    printfQuda("%e Failures: %d / %d  = %e\n", pow(10.0, -(f + 1)), fail[f], 4 * len * 18,
               fail[f] / (double)(4 * len * 18));
  }

  return accuracy_level;
}

static int compare_link(void **linkA, void **linkB, int len, QudaPrecision precision)
{
  int ret;

  if (precision == QUDA_DOUBLE_PRECISION) {
    ret = compareLink((double **)linkA, (double **)linkB, len);
  } else {
    ret = compareLink((float **)linkA, (float **)linkB, len);
  }

  return ret;
}

static int compare_link(const GaugeField &a, const GaugeField &b)
{
  if (a.Order() != QUDA_QDP_GAUGE_ORDER) errorQuda("Unsupported gauge order %d", a.Order());
  int ret;
  if (checkPrecision(a, b) == QUDA_DOUBLE_PRECISION) {
    ret = compareLink(reinterpret_cast<double **>(a.raw_pointer()), reinterpret_cast<double **>(b.raw_pointer()),
                      a.Volume());
  } else {
    ret = compareLink(reinterpret_cast<float **>(a.raw_pointer()), reinterpret_cast<float **>(b.raw_pointer()),
                      a.Volume());
  }

  return ret;
}

// X indexes the lattice site
static void printLinkElement(void *link, int X, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    for (int i = 0; i < 3; i++) { printVector((double *)link + X * gauge_site_size + i * 6); }

  } else {
    for (int i = 0; i < 3; i++) { printVector((float *)link + X * gauge_site_size + i * 6); }
  }
}

int strong_check_link(void **linkA, const char *msgA, void **linkB, const char *msgB, int len, QudaPrecision prec)
{
  if (verbosity >= QUDA_VERBOSE) {
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
  }

  return compare_link(linkA, linkB, len, prec);
}

int strong_check_link(const GaugeField &linkA, const std::string &msgA, const GaugeField &linkB, const std::string &msgB)
{
  if (linkA.Order() != QUDA_QDP_GAUGE_ORDER) errorQuda("Unsupported gauge order %d", linkA.Order());
  if (verbosity >= QUDA_VERBOSE) {
    printfQuda("%s\n", msgA.c_str());
    printLinkElement(linkA.data(0), 0, prec);
    printfQuda("\n");
    printLinkElement(linkA.data(0), 1, prec);
    printfQuda("...\n");
    printLinkElement(linkA.data(3), linkA.Volume() - 1, prec);
    printfQuda("\n");

    printfQuda("\n%s\n", msgB.c_str());
    printLinkElement(linkB.data(0), 0, prec);
    printfQuda("\n");
    printLinkElement(linkB.data(0), 1, prec);
    printfQuda("...\n");
    printLinkElement(linkB.data(3), linkB.Volume() - 1, prec);
    printfQuda("\n");
  }

  return compare_link(linkA, linkB);
}

void createStagForOprodCPU(void *stag_for_oprod, QudaPrecision precision, const int *const x, quda::RNG &rng)
{
  unsigned long shift = x[0] * x[1] * x[2] * x[3] * stag_spinor_site_size;
  if (precision == QUDA_DOUBLE_PRECISION) {
    double *dstag = (double *)stag_for_oprod;
    // matpc: compute a full-volume spinor
    for (int d = 0; d < 4; d++)
      constructRandomSpinorSource(dstag + d * shift, 1, 3, QUDA_DOUBLE_PRECISION, QUDA_MAT_SOLUTION, x, 4, rng);
  } else {
    float *fstag = (float *)stag_for_oprod;
    for (int d = 0; d < 4; d++)
      constructRandomSpinorSource(fstag + d * shift, 1, 3, QUDA_SINGLE_PRECISION, QUDA_MAT_SOLUTION, x, 4, rng);
  }
}

void performanceStats(std::vector<double> &time, std::vector<double> &gflops, std::vector<int> &iter)
{
  auto mean_time = 0.0;
  auto mean_time2 = 0.0;
  auto mean_gflops = 0.0;
  auto mean_gflops2 = 0.0;
  auto mean_iter = 0.0;
  auto mean_iter2 = 0.0;
  // skip first solve due to allocations, potential UVM swapping overhead
  for (int i = 1; i < Nsrc; i++) {
    mean_time += time[i];
    mean_time2 += time[i] * time[i];
    mean_gflops += gflops[i];
    mean_gflops2 += gflops[i] * gflops[i];
    mean_iter += iter[i];
    mean_iter2 += iter[i] * iter[i];
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

  mean_iter /= NsrcM1;
  mean_iter2 /= NsrcM1;
  auto stddev_iter = NsrcM1 > 1 ? sqrt((NsrcM1 / ((double)NsrcM1 - 1.0)) * (mean_iter2 - mean_iter * mean_iter)) :
                                  std::numeric_limits<double>::infinity();

  printfQuda("%d solves, mean iteration count %g (stddev = %g), with mean solve time %g (stddev = %g), mean GFLOPS %g "
             "(stddev = %g) [excluding first solve]\n",
             Nsrc, mean_iter, stddev_iter, mean_time, stddev_time, mean_gflops, stddev_gflops);
}
