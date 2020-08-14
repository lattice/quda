// QUDA headers
#include <quda.h>

// External headers
#include <host_utils.h>
#include <command_line_params.h>

//
// Created by luis on 03.08.20.
//
int main(int argc, char **argv)
{
  // Parameter class that reads line arguments. It modifies global parameter variables
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  setQudaPrecisions();

  // init QUDA
  initComms(argc, argv, gridsize_from_cmdline);
  initQuda(device);

  // gauge params
  QudaGaugeParam gauge_param = newQudaGaugeParam(); // create an instance of a class that can hold the gauge parameters
  setWilsonGaugeParam(gauge_param); // set the content of this instance to the globally set values (default or from
                                    // command line)
  setDims(gauge_param.X);

  // allocate gaugefield on host
  void *gauge[4];
  for (auto &dir : gauge) dir = malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  loadGaugeQuda((void *)gauge, &gauge_param); // copy gaugefield to GPU

  printfQuda("-----------------------------------------------------------------------------------\n");
  // compute plaquette
  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // invert params
  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  // Allocate host side memory for clover terms if needed.
  void *clover = nullptr;
  void *clover_inv = nullptr;
  // Allocate space on the host
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    if  (!compute_clover){
      errorQuda("Specified clover dslash-type but did not specify compute-clover!");
    }
    clover = malloc(V * clover_site_size * host_clover_data_type_size);
    clover_inv = malloc(V * clover_site_size * host_spinor_data_type_size);
    constructHostCloverField(clover, clover_inv, inv_param);
    loadCloverQuda(clover, clover_inv, &inv_param);
  }
  // ColorSpinors (Wilson)
  //FIXME what about this parameter class? should it be part of quda.h like invertparam etc?
  quda::ColorSpinorParam cs_param;
  quda::ColorSpinorParam* cs_param_ptr = &cs_param;
  constructWilsonTestSpinorParam(&cs_param, &inv_param, &gauge_param); //FIXME remove Test from this function name?
  int spinor_dim = cs_param.nColor * cs_param.nSpin;
  setSpinorSiteSize(spinor_dim * 2); // this sets the global variable my_spinor_site_size

  size_t bytes_per_float = sizeof(double);
  // Allocate memory on host for one source (0,0,0,0) for each of the 12x12 color+spinor combinations
  auto *source_array = (double *)malloc(spinor_dim * spinor_dim * V * 2 * bytes_per_float);
  auto *prop_array = (double *)malloc(spinor_dim * spinor_dim * V * 2 * bytes_per_float);

  //this will be a C array of pointers to the memory (CSF->V()) of the spinor_dim colorspinorfields. functions declared in quda.h
  // can only accept C code for backwards compatibility reasons
  void *CSF_V_ptr_arr_source[spinor_dim];
  void *CSF_V_ptr_arr_prop[spinor_dim];
  
  // Actually create ColorSpinorField objects and tell them to use the memory from above
  for (int i = 0; i < spinor_dim; i++) {
    CSF_V_ptr_arr_source[i] = malloc(sizeof(void *));
    CSF_V_ptr_arr_prop[i] = malloc(sizeof(void *));
    int offset = i * V * spinor_dim * 2;
    CSF_V_ptr_arr_source[i] = source_array + offset;
    CSF_V_ptr_arr_prop[i] = prop_array + offset;
  }

  //this is where the result will be stored
  void *correlation_function_sum = nullptr;

  // Loop over the number of sources to use. Default is prop_n_sources=1. Default source position = 0 0 0 0
  for (int n = 0; n < prop_n_sources; n++) {
    printfQuda("Source position: %d %d %d %d\n", prop_source_position[n][0], prop_source_position[n][1],
               prop_source_position[n][2], prop_source_position[n][3]);
    for (int i = 0; i < spinor_dim; i++) {
      const int source[4] = {prop_source_position[n][0], prop_source_position[n][1], prop_source_position[n][2],
                             prop_source_position[n][3]};
      constructPointSpinorSource(CSF_V_ptr_arr_source[i], cs_param.nSpin, cs_param.nColor, inv_param.cpu_prec,
                                 gauge_param.X, i, source);
      inv_param.solver_normalization = QUDA_SOURCE_NORMALIZATION; // Make explicit for now.

      invertQuda(CSF_V_ptr_arr_prop[i], CSF_V_ptr_arr_source[i], &inv_param);

    }

    contractSpatialQuda(CSF_V_ptr_arr_prop, CSF_V_ptr_arr_prop, &correlation_function_sum, contract_type, &inv_param, (void*)cs_param_ptr, gauge_param.X);
  }

  // print correlators
  size_t corr_dim;
  contract_type == QUDA_CONTRACT_TYPE_DR_SUM_SPATIAL ? corr_dim = 2 : corr_dim = 3;
  size_t global_corr_length = gauge_param.X[corr_dim] * comm_dim(corr_dim);
  size_t n_numbers_per_slice = 2 * 16;
  for (int G_idx = 0; G_idx < 16; G_idx++) {
    for (size_t t = 0; t < global_corr_length; t++) {
      printfQuda("sum: g=%d t=%lu %e %e\n", G_idx, t, ((double*)correlation_function_sum)[n_numbers_per_slice * t + 2 * G_idx],
                 ((double*)correlation_function_sum)[n_numbers_per_slice * t + 2 * G_idx + 1]);
    }
  }

  // free memory
  freeGaugeQuda();
  for (auto &dir : gauge) free(dir);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    freeCloverQuda();
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }

  free(source_array);
  free(prop_array);
  free(CSF_V_ptr_arr_source);
  free(CSF_V_ptr_arr_prop);
  free(correlation_function_sum);

  printfQuda("----------------------------------------------------------------------------------\n");
  endQuda();
  finalizeComms();

  return 0;
}
