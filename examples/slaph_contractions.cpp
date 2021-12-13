// QUDA headers
#include <quda.h>
#include <host_utils.h>
#include <command_line_params.h>

// C++
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

int main(int argc, char **argv)
{
  setQudaDefaultMgTestParams();
  auto app = make_app();   // Parameter class that reads cmdline arguments. It modifies global variables.
  add_multigrid_option_group(app);
  add_propagator_option_group(app);
  add_contraction_option_group(app);
  add_su3_option_group(app);
  add_slaph_option_group(app);
  
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  setQudaPrecisions();

  // init QUDA
  initComms(argc, argv, gridsize_from_cmdline);

  // Run-time parameter checks
  {
    if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH) {
      if (comm_rank() == 0) printf("dslash_type %d not supported\n", dslash_type);
      exit(0);
    }
    if (inv_multigrid) {
      // Only these fermions are supported with MG
      if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH) {
        if (comm_rank() == 0) printf("dslash_type %d not supported for MG\n", dslash_type);
        exit(0);
      }
      // Only these solve types are supported with MG
      if (solve_type != QUDA_DIRECT_SOLVE && solve_type != QUDA_DIRECT_PC_SOLVE) {
        if (comm_rank() == 0) printf("Solve_type %d not supported with MG. Please use QUDA_DIRECT_SOLVE or QUDA_DIRECT_PC_SOLVE\n\n",
                                     solve_type);
        exit(0);
      }
    }
  }

  initQuda(device_ordinal);

  // Gauge Parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam(); // create an instance of a class that can hold parameters
  setWilsonGaugeParam(gauge_param); // set the content of this instance to the currently set global values
  setDims(gauge_param.X);

  // Invert Parameters
  QudaInvertParam inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaEigParam mg_eig_param[mg_levels];
  //QudaEigParam eig_param = newQudaEigParam();
  if (inv_multigrid) {
    setQudaMgSolveTypes();
    setMultigridInvertParam(inv_param);
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
    setMultigridParam(mg_param);
  } else {
    setInvertParam(inv_param);
  }
  inv_param.eig_param = nullptr;
  if (inv_multigrid) {
    if (open_flavor) {
      if (comm_rank() == 0) printf("all the MG settings will be shared for qq, ql and qs propagator\n");
      for (int i = 0; i < mg_levels; i++) {
         if (strcmp(mg_param.vec_infile[i], "") != 0 || strcmp(mg_param.vec_outfile[i], "") != 0){
           if (comm_rank() == 0) printf("Save or write vec not possible! As when open flavor turned on inverter will be called "
                                        "3 times thus vec will be over written\n");
           exit(0);
         }
      }
    }
  }

  // allocate and load gaugefield on host
  void *gauge[4];
  for (auto &dir : gauge) dir = malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  loadGaugeQuda((void *)gauge, &gauge_param); // copy gaugefield to GPU

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    // If you pass nullptr to QUDA, it will automatically compute
    // the clover and clover inverse terms. If you need QUDA to return
    // clover fields to you, pass valid pointers to the function
    // and set:
    // inv_param.compute_clover = 1;
    // inv_param.compute_clover_inverse = 1;
    // inv_param.return_clover = 1;
    // inv_param.return_clover_inverse = 1;
    loadCloverQuda(nullptr, nullptr, &inv_param);
  }
  if (comm_rank() == 0) printf("-----------------------------------------------------------------------------------\n");

  // compute plaquette
  double plaq[3];
  plaqQuda(plaq);
  if (comm_rank() == 0) printf("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  
  // Now QUDA is initialised and the fields are loaded, we may setup the preconditioner
  void *mg_preconditioner = nullptr;

  // Wilson ColorSpinorParams
  quda::ColorSpinorParam cs_param;
  constructWilsonSpinorParam(&cs_param, &inv_param, &gauge_param);
  int spinor_dim = cs_param.nColor * cs_param.nSpin;

  // Allocate memory on host for one source for each of the 12x12 color+spinor combinations
  size_t bytes_per_float = sizeof(double);
  auto *source_array = (double *)malloc(spinor_dim * spinor_dim * V * 2 * bytes_per_float);
  auto *prop_array = (double *)malloc(spinor_dim * spinor_dim * V * 2 * bytes_per_float);
  // C array of pointers to the memory allocated above of the colorspinorfields (later ColorSpinorField->V())
  void *source_array_ptr[spinor_dim];
  void *prop_array_ptr[spinor_dim];
  for (int i = 0; i < spinor_dim; i++) {
    int offset = i * V * spinor_dim * 2;
    source_array_ptr[i] = source_array + offset;
    prop_array_ptr[i] = prop_array + offset;
  }

  //clean up
  if (inv_multigrid) destroyMultigridQuda(mg_preconditioner);
  freeGaugeQuda();
  for (auto &dir : gauge) free(dir);
  
  free(source_array);
  free(prop_array);
  if (comm_rank() == 0) printf("----------------------------------------------------------------------------------\n");
  endQuda();
  finalizeComms();

  return 0;
}
