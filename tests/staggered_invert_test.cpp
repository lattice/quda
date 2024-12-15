#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <color_spinor_field.h>
#include <gauge_field.h>

// External headers
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <llfat_utils.h>
#include "test.h"

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;
QudaMultigridParam mg_param;
QudaInvertParam mg_inv_param;
QudaEigParam mg_eig_param[QUDA_MAX_MG_LEVEL];
QudaEigParam eig_param;
bool use_split_grid = false;
bool use_multi_src = false;

// print instructions on how to run the old tests
bool print_legacy_info = false;

// if --enable-testing true is passed, we run the tests defined in here
#include <staggered_invert_test_gtest.hpp>

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("prec    prec_sloppy   multishift  matpc_type  recon  recon_sloppy solve_type S_dimension T_dimension "
             "Ls_dimension   dslash_type  normalization\n");
  printfQuda(
    "%6s   %6s          %d     %12s     %2s     %2s         %10s %3d/%3d/%3d     %3d         %2d       %14s  %8s\n",
    get_prec_str(prec), get_prec_str(prec_sloppy), multishift, get_matpc_str(matpc_type), get_recon_str(link_recon),
    get_recon_str(link_recon_sloppy), get_solve_str(solve_type), xdim, ydim, zdim, tdim, Lsdim,
    get_dslash_str(dslash_type), get_mass_normalization_str(normalization));

  if (inv_multigrid) {
    printfQuda("MG parameters\n");
    printfQuda(" - number of levels %d\n", mg_levels);
    for (int i = 0; i < mg_levels - 1; i++) {
      printfQuda(" - level %d number of null-space vectors %d\n", i + 1, nvec[i]);
      printfQuda(" - level %d null-space vector batch size %d\n", i + 1, nvec_batch[i]);
      printfQuda(" - level %d number of pre-smoother applications %d\n", i + 1, nu_pre[i]);
      printfQuda(" - level %d number of post-smoother applications %d\n", i + 1, nu_post[i]);
    }

    printfQuda("MG Eigensolver parameters\n");
    for (int i = 0; i < mg_levels; i++) {
      if (low_mode_check || mg_eig[i]) {
        printfQuda(" - level %d solver mode %s\n", i + 1, get_eig_type_str(mg_eig_type[i]));
        printfQuda(" - level %d spectrum requested %s\n", i + 1, get_eig_spectrum_str(mg_eig_spectrum[i]));
        if (mg_eig_type[i] == QUDA_EIG_BLK_TR_LANCZOS)
          printfQuda(" - eigenvector block size %d\n", mg_eig_block_size[i]);
        printfQuda(" - level %d number of eigenvectors requested n_conv %d\n", i + 1, nvec[i]);
        printfQuda(" - level %d size of eigenvector search space %d\n", i + 1, mg_eig_n_ev[i]);
        printfQuda(" - level %d size of Krylov space %d\n", i + 1, mg_eig_n_kr[i]);
        printfQuda(" - level %d solver tolerance %e\n", i + 1, mg_eig_tol[i]);
        printfQuda(" - level %d convergence required (%s)\n", i + 1, mg_eig_require_convergence[i] ? "true" : "false");
        printfQuda(" - level %d Operator: daggered (%s) , norm-op (%s)\n", i + 1,
                   mg_eig_use_dagger[i] ? "true" : "false", mg_eig_use_normop[i] ? "true" : "false");
        if (mg_eig_use_poly_acc[i]) {
          printfQuda(" - level %d Chebyshev polynomial degree %d\n", i + 1, mg_eig_poly_deg[i]);
          printfQuda(" - level %d Chebyshev polynomial minumum %e\n", i + 1, mg_eig_amin[i]);
          if (mg_eig_amax[i] <= 0)
            printfQuda(" - level %d Chebyshev polynomial maximum will be computed\n", i + 1);
          else
            printfQuda(" - level %d Chebyshev polynomial maximum %e\n", i + 1, mg_eig_amax[i]);
        }
        printfQuda("\n");
      }
    }
  }

  if (inv_deflate) {
    printfQuda("\n   Eigensolver parameters\n");
    printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
    printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
    if (eig_type == QUDA_EIG_BLK_TR_LANCZOS) printfQuda(" - eigenvector block size %d\n", eig_block_size);
    printfQuda(" - number of eigenvectors requested %d\n", eig_n_conv);
    printfQuda(" - size of eigenvector search space %d\n", eig_n_ev);
    printfQuda(" - size of Krylov space %d\n", eig_n_kr);
    printfQuda(" - solver tolerance %e\n", eig_tol);
    printfQuda(" - convergence required (%s)\n", eig_require_convergence ? "true" : "false");
    if (eig_compute_svd) {
      printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
      printfQuda(" - ***********************************************************\n");
      printfQuda(" - **** Overriding any previous choices of operator type. ****\n");
      printfQuda(" - ****    SVD demands normal operator, will use MdagM    ****\n");
      printfQuda(" - ***********************************************************\n");
    } else {
      printfQuda(" - Operator: daggered (%s) , norm-op (%s)\n", eig_use_dagger ? "true" : "false",
                 eig_use_normop ? "true" : "false");
    }
    if (eig_use_poly_acc) {
      printfQuda(" - Chebyshev polynomial degree %d\n", eig_poly_deg);
      printfQuda(" - Chebyshev polynomial minumum %e\n", eig_amin);
      if (eig_amax <= 0)
        printfQuda(" - Chebyshev polynomial maximum will be computed\n");
      else
        printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
    }
  }

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

void display_legacy_info()
{
  printfQuda("Instructions for running legacy tests:\n");
  printfQuda("--test 0 -> --solve-type direct    --solution-type mat    --inv-type bicgstab\n");
  printfQuda("--test 1 -> --solve-type direct-pc --solution-type mat    --inv-type cg --matpc even-even\n");
  printfQuda("--test 2 -> --solve-type direct-pc --solution-type mat    --inv-type cg --matpc odd-odd\n");
  printfQuda("--test 3 -> --solve-type direct-pc --solution-type mat-pc --inv-type cg --matpc even-even\n");
  printfQuda("--test 4 -> --solve-type direct-pc --solution-type mat-pc --inv-type cg --matpc odd-odd\n");
  printfQuda(
    "--test 5 -> --solve-type direct-pc --solution-type mat-pc --inv-type cg --matpc even-even --multishift 8\n");
  printfQuda(
    "--test 6 -> --solve-type direct-pc --solution-type mat-pc --inv-type cg --matpc odd-odd   --multishift 8\n");
}

GaugeField cpuFatQDP = {};
GaugeField cpuLongQDP = {};
GaugeField cpuFatMILC = {};
GaugeField cpuLongMILC = {};

void init()
{
  // Set QUDA internal parameters
  gauge_param = newQudaGaugeParam();
  setStaggeredGaugeParam(gauge_param);
  QudaGaugeSmearParam smear_param;
  if (gauge_smear) {
    smear_param = newQudaGaugeSmearParam();
    setGaugeSmearParam(smear_param);
  }

  inv_param = newQudaInvertParam();
  mg_inv_param = newQudaInvertParam();
  mg_param = newQudaMultigridParam();
  eig_param = newQudaEigParam();

  if (inv_multigrid) {
    // Set some default values for MG solve types
    setQudaMgSolveTypes();
    setStaggeredMGInvertParam(inv_param);
    // Set sub structures
    mg_param.invert_param = &mg_inv_param;
    for (int i = 0; i < mg_levels; i++) {
      if (mg_eig[i]) {
        mg_eig_param[i] = newQudaEigParam();
        setMultigridEigParam(mg_eig_param[i], i);
        mg_param.eig_param[i] = &mg_eig_param[i];
      } else {
        mg_param.eig_param[i] = nullptr;
      }
    }
    // Set MG
    setStaggeredMultigridParam(mg_param);
  } else {
    setStaggeredInvertParam(inv_param);
  }

  if (inv_deflate) {
    setEigParam(eig_param);
    inv_param.eig_param = &eig_param;
    if (use_split_grid) { errorQuda("Split grid does not work with deflation yet.\n"); }
  } else {
    inv_param.eig_param = nullptr;
  }

  setDims(gauge_param.X);
  dw_setDims(gauge_param.X, 1);

  // Staggered Gauge construct START
  //-----------------------------------------------------------------------------------
  // Allocate host staggered gauge fields
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  GaugeFieldParam cpuParam(gauge_param);
  cpuParam.order = QUDA_QDP_GAUGE_ORDER;
  cpuParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField cpuIn = GaugeField(cpuParam);
  cpuFatQDP = GaugeField(cpuParam);
  cpuParam.order = QUDA_MILC_GAUGE_ORDER;
  cpuFatMILC = GaugeField(cpuParam);

  cpuParam.link_type = QUDA_ASQTAD_LONG_LINKS;
  cpuParam.nFace = 3;
  cpuParam.order = QUDA_QDP_GAUGE_ORDER;
  cpuLongQDP = GaugeField(cpuParam);
  cpuParam.order = QUDA_MILC_GAUGE_ORDER;
  cpuLongMILC = GaugeField(cpuParam);

  void *qdp_inlink[4] = {cpuIn.data(0), cpuIn.data(1), cpuIn.data(2), cpuIn.data(3)};
  void *qdp_fatlink[4] = {cpuFatQDP.data(0), cpuFatQDP.data(1), cpuFatQDP.data(2), cpuFatQDP.data(3)};
  void *qdp_longlink[4] = {cpuLongQDP.data(0), cpuLongQDP.data(1), cpuLongQDP.data(2), cpuLongQDP.data(3)};
  constructStaggeredHostGaugeField(qdp_inlink, qdp_longlink, qdp_fatlink, gauge_param, 0, nullptr, true);

  // Reorder gauge fields to MILC order
  cpuFatMILC = cpuFatQDP;
  cpuLongMILC = cpuLongQDP;

  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  // This needs to be called before `loadFatLongGaugeQuda` because this routine also loads the
  // gauge fields with different parameters.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    // Compute fat link plaquette
    computeStaggeredPlaquetteQDPOrder(qdp_fatlink, plaq, gauge_param, dslash_type);
    printfQuda("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }

  freeGaugeQuda();

  loadFatLongGaugeQuda(cpuFatMILC.data(), cpuLongMILC.data(), gauge_param);

  // now copy back to QDP aliases, since these are used for the reference dslash
  cpuFatQDP = cpuFatMILC;
  cpuLongQDP = cpuLongMILC;
  // ensure QDP alias has exchanged ghosts
  cpuFatQDP.exchangeGhost();
  cpuLongQDP.exchangeGhost();

  // Staggered Gauge construct END
  //-----------------------------------------------------------------------------------
}

std::vector<std::array<double, 2>> solve(test_t param)
{
  inv_param.inv_type = ::testing::get<0>(param);
  inv_param.solution_type = ::testing::get<1>(param);
  inv_param.solve_type = ::testing::get<2>(param);
  inv_param.cuda_prec_sloppy = ::testing::get<3>(param);
  multishift = ::testing::get<4>(param);
  inv_param.solution_accumulator_pipeline = ::testing::get<5>(param);

  // schwarz parameters
  auto schwarz_param = ::testing::get<6>(param);
  inv_param.schwarz_type = ::testing::get<0>(schwarz_param);
  inv_param.inv_type_precondition = ::testing::get<1>(schwarz_param);
  inv_param.cuda_prec_precondition = ::testing::get<2>(schwarz_param);

  inv_param.residual_type = ::testing::get<7>(param);

  // reset lambda_max if we're doing a testing loop to ensure correct lambma_max
  if (enable_testing) inv_param.ca_lambda_max = -1.0;

  logQuda(QUDA_SUMMARIZE, "Solution = %s, Solve = %s, Solver = %s, Sloppy precision = %s\n",
          get_solution_str(inv_param.solution_type), get_solve_str(inv_param.solve_type),
          get_solver_str(inv_param.inv_type), get_prec_str(inv_param.cuda_prec_sloppy));

  // params related to split grid.
  for (int i = 0; i < 4; i++) inv_param.split_grid[i] = grid_partition[i];
  int num_sub_partition = grid_partition[0] * grid_partition[1] * grid_partition[2] * grid_partition[3];
  use_split_grid = num_sub_partition > 1;
  use_multi_src = use_split_grid || (Nsrc_tile > 1);

  // Setup the multigrid preconditioner
  void *mg_preconditioner = nullptr;
  if (inv_multigrid) {
    if (use_split_grid) { errorQuda("Split grid does not work with MG yet."); }
    mg_preconditioner = newMultigridQuda(&mg_param);
    inv_param.preconditioner = mg_preconditioner;

    printfQuda("MG Setup Done: %g secs, %g Gflops\n", mg_param.invert_param->secs,
               mg_param.invert_param->gflops / mg_param.invert_param->secs);
    if (mg_param.invert_param->energy > 0) {
      printfQuda("Energy = %g J, Mean power = %g W, mean temp = %g C, mean clock = %f\n", mg_param.invert_param->energy,
                 mg_param.invert_param->power, mg_param.invert_param->temp, mg_param.invert_param->clock);
    }
  }

  // Staggered vector construct START
  //-----------------------------------------------------------------------------------
  if (Nsrc > QUDA_MAX_MULTI_SRC)
    errorQuda("Nsrc = %d which is great than QUDA_MAX_MULTI_SRC = %d\n", Nsrc, QUDA_MAX_MULTI_SRC);
  std::vector<quda::ColorSpinorField> in(Nsrc);
  std::vector<quda::ColorSpinorField> out(Nsrc);
  std::vector<quda::ColorSpinorField> out_multishift(Nsrc * multishift);
  quda::ColorSpinorParam cs_param;
  constructStaggeredTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  std::vector<std::vector<void *>> _hp_multi_x(Nsrc, std::vector<void *>(multishift));

  // Staggered vector construct END
  //-----------------------------------------------------------------------------------

  // Setup multishift parameters (if needed)
  //---------------------------------------------------------------------------

  // Masses
  std::vector<double> masses(multishift);

  if (multishift > 1) {
    if (use_split_grid)
      errorQuda("Multishift currently doesn't support split grid.\n");

    inv_param.num_offset = multishift;

    // Consistency check for masses, tols, tols_hq size if we're setting custom values
    if (multishift_shifts.size() != 0)
      errorQuda("Multishift shifts are not supported for Wilson-type fermions");
    if (multishift_masses.size() != 0 && multishift_masses.size() != static_cast<unsigned long>(multishift))
      errorQuda("Multishift mass count %d does not agree with number of masses passed in %lu\n", multishift, multishift_masses.size());
    if (multishift_tols.size() != 0 && multishift_tols.size() != static_cast<unsigned long>(multishift))
      errorQuda("Multishift tolerance count %d does not agree with number of masses passed in %lu\n", multishift, multishift_tols.size());
    if (multishift_tols_hq.size() != 0 && multishift_tols_hq.size() != static_cast<unsigned long>(multishift))
      errorQuda("Multishift hq tolerance count %d does not agree with number of masses passed in %lu\n", multishift, multishift_tols_hq.size());

    // Copy offsets and tolerances into inv_param; allocate and copy data pointers
    for (int i = 0; i < multishift; i++) {
      masses[i] = (multishift_masses.size() == 0 ? (mass + i * i * 0.01) : multishift_masses[i]);
      inv_param.offset[i] = 4 * masses[i] * masses[i];
      inv_param.tol_offset[i] = (multishift_tols.size() == 0 ? inv_param.tol : multishift_tols[i]);
      inv_param.tol_hq_offset[i] = (multishift_tols_hq.size() == 0 ? inv_param.tol_hq : multishift_tols_hq[i]);

      // Allocate memory and set pointers
      for (int n = 0; n < Nsrc; n++) {
        out_multishift[n * multishift + i] = quda::ColorSpinorField(cs_param);
        _hp_multi_x[n][i] = out_multishift[n * multishift + i].data();
      }

      logQuda(QUDA_VERBOSE, "Multishift mass %d = %e ; tolerance %e ; hq tolerance %e\n", i, masses[i], inv_param.tol_offset[i], inv_param.tol_hq_offset[i]);
    }
  }

  // Setup multishift parameters END
  //-----------------------------------------------------------------------------------

  // Prepare rng, fill host spinors with random numbers
  //-----------------------------------------------------------------------------------

  std::vector<double> time(Nsrc);
  std::vector<double> gflops(Nsrc);
  std::vector<int> iter(Nsrc);

  // Create a temporary spinor just to seed the rng
  quda::ColorSpinorField tmp(cs_param);
  quda::RNG rng(tmp, 1234);
  tmp = quda::ColorSpinorField();

  for (int n = 0; n < Nsrc; n++) {
    // Populate the host spinor with random numbers.
    in[n] = quda::ColorSpinorField(cs_param);
    quda::spinorNoise(in[n], rng, QUDA_NOISE_UNIFORM);
    out[n] = quda::ColorSpinorField(cs_param);
  }

  // Prepare rng, fill host spinors with random numbers END
  //-----------------------------------------------------------------------------------

  // QUDA invert test
  //----------------------------------------------------------------------------

  if (!use_multi_src || multishift > 1) {

    for (int n = 0; n < Nsrc; n++) {
      // If deflating, preserve the deflation space between solves
      if (inv_deflate) eig_param.preserve_deflation = n < Nsrc - 1 ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      // Perform QUDA inversions
      if (multishift > 1) {
        invertMultiShiftQuda(_hp_multi_x[n].data(), in[n].data(), &inv_param);
      } else {
        invertQuda(out[n].data(), in[n].data(), &inv_param);
      }

      // move residuals to n^th location for verification after solves have finished
      inv_param.true_res[n] = inv_param.true_res[0];
      inv_param.true_res_hq[n] = inv_param.true_res_hq[0];

      time[n] = inv_param.secs;
      gflops[n] = inv_param.gflops / inv_param.secs;
      iter[n] = inv_param.iter;
      printfQuda("Done: %i iter / %g secs = %g Gflops\n", inv_param.iter, inv_param.secs,
                 inv_param.gflops / inv_param.secs);
      if (inv_param.energy > 0) {
        printfQuda("Energy = %g J, Mean power = %g W, mean temp = %g C, mean clock = %f\n\n", inv_param.energy,
                   inv_param.power, inv_param.temp, inv_param.clock);
      }
    }
  } else {

    inv_param.num_src = Nsrc_tile;
    inv_param.num_src_per_sub_partition = Nsrc_tile / num_sub_partition;
    // Host arrays for solutions, sources, and check
    std::vector<void *> _hp_x(Nsrc_tile);
    std::vector<void *> _hp_b(Nsrc_tile);

    for (int j = 0; j < Nsrc; j += Nsrc_tile) {
      for (int i = 0; i < Nsrc_tile; i++) {
        _hp_x[i] = out[j + i].data();
        _hp_b[i] = in[j + i].data();
      }

      if (inv_deflate) eig_param.preserve_deflation = j < Nsrc - Nsrc_tile ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      invertMultiSrcQuda(_hp_x.data(), _hp_b.data(), &inv_param);

      // move residuals to (i+j)^th location for verification after solves have finished
      for (int i = 0; i < Nsrc_tile; i++) {
        inv_param.true_res[j + i] = inv_param.true_res[i];
        inv_param.true_res_hq[j + i] = inv_param.true_res_hq[i];
      }

      quda::comm_allreduce_int(inv_param.iter);
      inv_param.iter /= comm_size() / num_sub_partition;
      quda::comm_allreduce_sum(inv_param.gflops);
      inv_param.gflops /= comm_size() / num_sub_partition;
      quda::comm_allreduce_max(inv_param.secs);
      printfQuda("Done: %d sub-partitions - %i iter / %g secs = %g Gflops, %g secs per source\n", num_sub_partition,
                 inv_param.iter, inv_param.secs, inv_param.gflops / inv_param.secs, inv_param.secs / Nsrc_tile);
      if (inv_param.energy > 0) {
        printfQuda("Energy = %g J (%g J per source), Mean power = %g W, mean temp = %g C, mean clock = %f\n\n",
                   inv_param.energy, inv_param.energy / Nsrc_tile, inv_param.power, inv_param.temp, inv_param.clock);
      }
    }
  }

  // Free the multigrid solver
  if (inv_multigrid) destroyMultigridQuda(mg_preconditioner);

  // Compute timings
  if (!use_multi_src) performanceStats(time, gflops, iter);

  std::vector<std::array<double, 2>> res(Nsrc);
  // Perform host side verification of inversion if requested
  if (verify_results) {
    for (int n = 0; n < Nsrc; n++) {
      if (multishift > 1) {
        printfQuda("\nSource %d:\n", n);
        // Create an appropriate subset of the full out_multishift vector
        std::vector<quda::ColorSpinorField> out_subset
          = {out_multishift.begin() + n * multishift, out_multishift.begin() + (n + 1) * multishift};
        res[n] = verifyStaggeredInversion(in[n], out_subset, cpuFatQDP, cpuLongQDP, inv_param, laplace3D);
      } else {
        res[n] = verifyStaggeredInversion(in[n], out[n], cpuFatQDP, cpuLongQDP, inv_param, laplace3D, n);
      }
    }
  }

  return res;
}

void cleanup()
{
  cpuFatQDP = {};
  cpuLongQDP = {};
  cpuFatMILC = {};
  cpuLongMILC = {};
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  setQudaStaggeredDefaultInvTestParams();
  setQudaDefaultMgTestParams();
  // Parse command line options
  auto app = make_app();
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  add_multigrid_option_group(app);
  add_comms_option_group(app);
  add_testing_option_group(app);
  app->add_option("--legacy-test-info", print_legacy_info,
                  "Print info on how to reproduce the old '--test #' behavior with flags, then exit");
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  setVerbosity(verbosity);

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  if (print_legacy_info) {
    display_legacy_info();
    errorQuda("Exiting...");
  }

  if (inv_deflate && inv_multigrid)
    errorQuda("Error: Cannot use both deflation and multigrid preconditioners on top level solve");

  initRand();

  // Only these fermions are supported in this file
  if constexpr (is_enabled_laplace()) {
    if (!is_staggered(dslash_type) && !is_laplace(dslash_type))
      errorQuda("dslash_type %s not supported", get_dslash_str(dslash_type));
  } else {
    if (is_laplace(dslash_type))
      errorQuda("The Laplace dslash is not enabled, cmake configure with -DQUDA_DIRAC_LAPLACE=ON");
    if (!is_staggered(dslash_type)) errorQuda("dslash_type %s not supported", get_dslash_str(dslash_type));
  }

  // Need to add support for LAPLACE MG?
  if (inv_multigrid) {
    if (!is_staggered(dslash_type)) {
      errorQuda("dslash_type %s not supported for multigrid preconditioner", get_dslash_str(dslash_type));
    }
  }

  display_test_info();

  initQuda(device_ordinal);

  if (enable_testing) {
    // We need to force a well-behaved operator + reasonable convergence, otherwise
    // the staggered tests will fail. These checks are designed to be consistent
    // with what's in [src]/tests/CMakeFiles.txt, which have been "sanity checked"
    bool changes = false;
    if (!compute_fatlong) {
      compute_fatlong = true;
      changes = true;
    }

    double expected_tol = (prec == QUDA_SINGLE_PRECISION) ? 1e-5 : 1e-6;
    if (tol != expected_tol) {
      tol = expected_tol;
      changes = true;
    }
    if (tol_hq != expected_tol) {
      tol_hq = expected_tol;
      changes = true;
    }
    if (niter != 1000) {
      niter = 1000;
      changes = true;
    }

    if (changes) {
      printfQuda("For gtest, various defaults are changed:\n");
      printfQuda("  --compute-fat-long true\n");
      printfQuda("  --tol (1e-6 for double, 1e-5 for single)\n");
      printfQuda("  --tol-hq (1e-6 for double, 1e-5 for single)\n");
      printfQuda("  --niter 1000\n");
    }
  }

  init();

  int result = 0;
  if (enable_testing) { // tests are defined in staggered_invert_test_gtest.hpp
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (quda::comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
    result = RUN_ALL_TESTS();
  } else {
    solve(test_t {inv_type, solution_type, solve_type, prec_sloppy, multishift, solution_accumulator_pipeline,
                  schwarz_t {precon_schwarz_type, inv_multigrid ? QUDA_MG_INVERTER : precon_type, prec_precondition},
                  inv_param.residual_type});
  }

  cleanup();

  // Finalize the QUDA library
  freeGaugeQuda();
  endQuda();
  finalizeComms();

  return result;
}
