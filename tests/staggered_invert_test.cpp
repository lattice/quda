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

#define MAX(a, b) ((a) > (b) ? (a) : (b))

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

void test(int argc, char **argv)
{
  // Set QUDA internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
  setStaggeredGaugeParam(gauge_param);
  if (!inv_multigrid) setStaggeredInvertParam(inv_param);

  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  QudaEigParam mg_eig_param[mg_levels];

  // params related to split grid.
  for (int i = 0; i < 4; i++) inv_param.split_grid[i] = grid_partition[i];
  int num_sub_partition = grid_partition[0] * grid_partition[1] * grid_partition[2] * grid_partition[3];
  bool use_split_grid = num_sub_partition > 1;

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
    setStaggeredMultigridParam(mg_param);
  }

  QudaEigParam eig_param = newQudaEigParam();
  if (inv_deflate) {
    setEigParam(eig_param);
    inv_param.eig_param = &eig_param;
    if (use_split_grid) { errorQuda("Split grid does not work with deflation yet.\n"); }
  } else {
    inv_param.eig_param = nullptr;
  }

  setDims(gauge_param.X);
  // Hack: use the domain wall dimensions so we may use the 5th dim for multi indexing
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
  cpuParam.create = QUDA_NULL_FIELD_CREATE;
  cpuParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuParam.order = QUDA_QDP_GAUGE_ORDER;
  GaugeField cpuIn = GaugeField(cpuParam);
  GaugeField cpuFatQDP = GaugeField(cpuParam);
  cpuParam.order = QUDA_MILC_GAUGE_ORDER;
  GaugeField cpuFatMILC = GaugeField(cpuParam);

  cpuParam.link_type = QUDA_ASQTAD_LONG_LINKS;
  cpuParam.nFace = 3;
  cpuParam.order = QUDA_QDP_GAUGE_ORDER;
  GaugeField cpuLongQDP = GaugeField(cpuParam);
  cpuParam.order = QUDA_MILC_GAUGE_ORDER;
  GaugeField cpuLongMILC = GaugeField(cpuParam);

  void *qdp_inlink[4] = {cpuIn.data(0), cpuIn.data(1), cpuIn.data(2), cpuIn.data(3)};
  void *qdp_fatlink[4] = {cpuFatQDP.data(0), cpuFatQDP.data(1), cpuFatQDP.data(2), cpuFatQDP.data(3)};
  void *qdp_longlink[4] = {cpuLongQDP.data(0), cpuLongQDP.data(1), cpuLongQDP.data(2), cpuLongQDP.data(3)};
  constructStaggeredHostGaugeField(qdp_inlink, qdp_longlink, qdp_fatlink, gauge_param, argc, argv, true);

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

  loadFatLongGaugeQuda(cpuFatMILC.data(), cpuLongMILC.data(), gauge_param);

  // now copy back to QDP aliases, since these are used for the reference dslash
  cpuFatQDP = cpuFatMILC;
  cpuLongQDP = cpuLongMILC;
  // ensure QDP alias has exchanged ghosts
  cpuFatQDP.exchangeGhost();
  cpuLongQDP.exchangeGhost();

  // Staggered Gauge construct END
  //-----------------------------------------------------------------------------------

  // Setup the multigrid preconditioner
  void *mg_preconditioner = nullptr;
  if (inv_multigrid) {
    if (use_split_grid) { errorQuda("Split grid does not work with MG yet."); }
    mg_preconditioner = newMultigridQuda(&mg_param);
    inv_param.preconditioner = mg_preconditioner;

    printfQuda("MG Setup Done: %g secs, %g Gflops\n", mg_param.secs, mg_param.gflops / mg_param.secs);
  }

  // Staggered vector construct START
  //-----------------------------------------------------------------------------------
  std::vector<quda::ColorSpinorField> in(Nsrc);
  std::vector<quda::ColorSpinorField> out(Nsrc);
  quda::ColorSpinorParam cs_param;
  constructStaggeredTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  for (int k = 0; k < Nsrc; k++) {
    in[k] = quda::ColorSpinorField(cs_param);
    out[k] = quda::ColorSpinorField(cs_param);
  }
  ColorSpinorField ref(cs_param);
  ColorSpinorField tmp(cs_param);
  // Staggered vector construct END
  //-----------------------------------------------------------------------------------

  // Prepare rng
  quda::RNG rng(ref, 1234);

  // Performance measuring
  std::vector<double> time(Nsrc);
  std::vector<double> gflops(Nsrc);
  std::vector<int> iter(Nsrc);

  // QUDA invert test
  //----------------------------------------------------------------------------

  if (test_type >= 0 && test_type <= 4) {
    // case 0: // full parity solution, full parity system
    // case 1: // full parity solution, solving EVEN EVEN prec system
    // case 2: // full parity solution, solving ODD ODD prec system
    // case 3: // even parity solution, solving EVEN system
    // case 4: // odd parity solution, solving ODD system

    if (multishift != 1) errorQuda("Multishift not supported for test %d\n", test_type);

    for (int k = 0; k < Nsrc; k++) { quda::spinorNoise(in[k], rng, QUDA_NOISE_UNIFORM); }

    if (!use_split_grid) {
      for (int k = 0; k < Nsrc; k++) {
        if (inv_deflate) eig_param.preserve_deflation = k < Nsrc - 1 ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
        invertQuda(out[k].data(), in[k].data(), &inv_param);
        time[k] = inv_param.secs;
        gflops[k] = inv_param.gflops / inv_param.secs;
        iter[k] = inv_param.iter;
        printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs,
                   inv_param.gflops / inv_param.secs);
      }
    } else {
      std::vector<void *> _hp_x(Nsrc);
      std::vector<void *> _hp_b(Nsrc);
      for (int k = 0; k < Nsrc; k++) {
        _hp_x[k] = out[k].data();
        _hp_b[k] = in[k].data();
      }
      inv_param.num_src = Nsrc;
      inv_param.num_src_per_sub_partition = Nsrc / num_sub_partition;
      invertMultiSrcStaggeredQuda(_hp_x.data(), _hp_b.data(), &inv_param, cpuFatMILC.data(), cpuLongMILC.data(),
                                  &gauge_param);
      quda::comm_allreduce_int(inv_param.iter);
      inv_param.iter /= comm_size() / num_sub_partition;
      quda::comm_allreduce_sum(inv_param.gflops);
      inv_param.gflops /= comm_size() / num_sub_partition;
      quda::comm_allreduce_max(inv_param.secs);
      printfQuda("Done: %d sub-partitions - %i iter / %g secs = %g Gflops\n\n", num_sub_partition, inv_param.iter,
                 inv_param.secs, inv_param.gflops / inv_param.secs);
    }

    for (int k = 0; k < Nsrc; k++) {
      if (verify_results)
        verifyStaggeredInversion(tmp, ref, in[k], out[k], mass, cpuFatQDP, cpuLongQDP, gauge_param, inv_param, 0);
    }
  } else if (test_type == 5 || test_type == 6) {
    // case 5: // multi mass CG, even parity solution, solving EVEN system
    // case 6: // multi mass CG, odd parity solution, solving ODD system

    if (use_split_grid)
      errorQuda("Multishift currently doesn't support split grid.\n");

    if (multishift < 2)
      errorQuda("Multishift inverter requires more than one shift, multishift = %d\n", multishift);

    inv_param.num_offset = multishift;

    // Prepare vectors for masses
    std::vector<double> masses(multishift);

    // Consistency check for masses, tols, tols_hq size if we're setting custom values
    if (multishift_shifts.size() != 0)
      errorQuda("Multishift shifts are not supported for Wilson-type fermions");
    if (multishift_masses.size() != 0 && multishift_masses.size() != static_cast<unsigned long>(multishift))
      errorQuda("Multishift mass count %d does not agree with number of masses passed in %lu\n", multishift, multishift_masses.size());
    if (multishift_tols.size() != 0 && multishift_tols.size() != static_cast<unsigned long>(multishift))
      errorQuda("Multishift tolerance count %d does not agree with number of masses passed in %lu\n", multishift, multishift_tols.size());
    if (multishift_tols_hq.size() != 0 && multishift_tols_hq.size() != static_cast<unsigned long>(multishift))
      errorQuda("Multishift hq tolerance count %d does not agree with number of masses passed in %lu\n", multishift, multishift_tols_hq.size());

    // Allocate storage of output arrays
    std::vector<void*> outArray(multishift);
    std::vector<ColorSpinorField> qudaOutArray(multishift, cs_param);

    // Copy offsets and tolerances into inv_param; copy data pointers into outArray
    for (int i = 0; i < multishift; i++) {
      masses[i] = (multishift_masses.size() == 0 ? (mass + i * i * 0.01) : multishift_masses[i]);
      inv_param.offset[i] = 4 * masses[i] * masses[i];
      inv_param.tol_offset[i] = (multishift_tols.size() == 0 ? inv_param.tol : multishift_tols[i]);
      inv_param.tol_hq_offset[i] = (multishift_tols_hq.size() == 0 ? inv_param.tol_hq : multishift_tols_hq[i]);

      outArray[i] = qudaOutArray[i].data();

      logQuda(QUDA_VERBOSE, "Multishift mass %d = %e ; tolerance %e ; hq tolerance %e\n", i, masses[i], inv_param.tol_offset[i], inv_param.tol_hq_offset[i]);
    }

    for (int k = 0; k < Nsrc; k++) {
      quda::spinorNoise(in[k], rng, QUDA_NOISE_UNIFORM);
      invertMultiShiftQuda((void **)outArray.data(), in[k].data(), &inv_param);

      time[k] = inv_param.secs;
      gflops[k] = inv_param.gflops / inv_param.secs;
      iter[k] = inv_param.iter;
      printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs,
                 inv_param.gflops / inv_param.secs);

      for (int i = 0; i < multishift; i++) {
        printfQuda("%dth solution: mass=%f, ", i, masses[i]);
        verifyStaggeredInversion(tmp, ref, in[k], qudaOutArray[i], masses[i], cpuFatQDP, cpuLongQDP, gauge_param,
                                 inv_param, i);
      }
    }
  } else {
    errorQuda("Unsupported test type");
  } // switch

  // Compute timings
  if (Nsrc > 1 && !use_split_grid) performanceStats(time, gflops, iter);

  // Free the multigrid solver
  if (inv_multigrid) destroyMultigridQuda(mg_preconditioner);
}

int main(int argc, char **argv)
{
  setQudaDefaultMgTestParams();
  // Parse command line options
  auto app = make_app();
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  add_multigrid_option_group(app);
  add_comms_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"full", 0}, {"full_ee_prec", 1}, {"full_oo_prec", 2}, {"even", 3},
                                          {"odd", 4},  {"mcg_even", 5},     {"mcg_odd", 6}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  setVerbosity(verbosity);
  if (!inv_multigrid) solve_type = QUDA_INVALID_SOLVE;

  if (inv_deflate && inv_multigrid) {
    errorQuda("Error: Cannot use both deflation and multigrid preconditioners on top level solve");
  }

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  initRand();

  // Only these fermions are supported in this file. Ensure a reasonable default,
  // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH) {
    printfQuda("dslash_type %s not supported, defaulting to %s\n", get_dslash_str(dslash_type),
               get_dslash_str(QUDA_ASQTAD_DSLASH));
    dslash_type = QUDA_ASQTAD_DSLASH;
  }

  // Need to add support for LAPLACE MG?
  if (inv_multigrid) {
    if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH) {
      errorQuda("dslash_type %s not supported for multigrid preconditioner", get_dslash_str(dslash_type));
    }
  }

  // Deduce operator, solution, and operator preconditioning types
  if (!inv_multigrid) setQudaStaggeredInvTestParams();

  display_test_info();

  initQuda(device_ordinal);

  test(argc, argv);

  // Finalize the QUDA library
  endQuda();

  // Finalize the communications layer
  finalizeComms();

  return 0;
}
