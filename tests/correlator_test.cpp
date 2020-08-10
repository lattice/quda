//QUDA headers
#include <quda.h>

//External headers
#include <host_utils.h>
#include <command_line_params.h>

//
// Created by luis on 03.08.20.
//
int main(int argc, char **argv)
{
  //Parameter class that reads line arguments. It modifies global parameter variables
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  setQudaPrecisions();

  //init QUDA
  initComms(argc, argv, gridsize_from_cmdline);
  initQuda(device);

  // gauge params
  QudaGaugeParam gauge_param = newQudaGaugeParam(); //create an instance of a class that can hold the gauge parameters
  setWilsonGaugeParam(gauge_param); //set the content of this instance to the globally set values (default or from
                                       // command line)
  setDims(gauge_param.X);

  //allocate gaugefield on host
  void *gauge[4];
  for (auto & dir : gauge) dir = malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  loadGaugeQuda((void *)gauge, &gauge_param); //copy gaugefield to GPU

  printfQuda("-----------------------------------------------------------------------------------\n");
  //compute plaquette
  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  //invert params
  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  // Allocate host side memory for clover terms if needed.
  void *clover = nullptr;
  void *clover_inv = nullptr;
  // Allocate space on the host
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    clover = malloc(V * clover_site_size * host_clover_data_type_size);
    clover_inv = malloc(V * clover_site_size * host_spinor_data_type_size);
    constructHostCloverField(clover, clover_inv, inv_param);
    loadCloverQuda(clover, clover_inv, &inv_param);
  }
  //ColorSpinors (Wilson)
  quda::ColorSpinorParam cs_param;
  constructWilsonTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  int spinor_dim = cs_param.nColor*cs_param.nSpin;
  setSpinorSiteSize(spinor_dim *2); //this sets the global variable my_spinor_site_size

  std::vector<quda::ColorSpinorField *> qudaProp4D(
    spinor_dim); // qudaProp4D is a vector that contains 12 pointers to colorspinorfields.
  std::vector<quda::ColorSpinorField *> qudaSource4D(spinor_dim);

  size_t bytes_per_float = sizeof(double);
  // Allocate memory on host for one source (0,0,0,0) for each of the 12x12 color+spinor combinations
  auto *out_array = (double*)malloc(12*12*V*2* bytes_per_float);
  auto *in_array = (double*)malloc(12*12*V*2* bytes_per_float);

  //Actually create ColorSpinorField objects and tell them to use the memory from above
  cs_param.create = QUDA_REFERENCE_FIELD_CREATE;
  for (int dil = 0; dil < 12; dil++) {
    int offset = dil*V* spinor_dim *2;
    cs_param.v = in_array + offset;
    qudaSource4D[dil] = quda::ColorSpinorField::Create(cs_param);
    cs_param.v = out_array + offset;
    qudaProp4D[dil] = quda::ColorSpinorField::Create(cs_param);
  }

  // temporal or spatial correlator?
  size_t corr_dim = 3;
  if (contract_type == QUDA_CONTRACT_TYPE_DR_SUM_SPATIAL) {
    corr_dim = 2;
  }
  size_t local_corr_length = gauge_param.X[corr_dim];
  size_t local_corr_offset = local_corr_length * comm_coord(corr_dim);
  size_t global_corr_length = local_corr_length * comm_dim(corr_dim);

  //host memory for correlator results. comm_dim(corr_dim)*array_length is global Ntau or Nz
  size_t n_numbers_per_slice = 2 * cs_param.nSpin * cs_param.nSpin;
  size_t corr_size_in_bytes = n_numbers_per_slice * global_corr_length * bytes_per_float;
  auto *correlation_function = (double*)malloc(corr_size_in_bytes);
  auto *correlation_function_sum = (double*)malloc(corr_size_in_bytes);
  memset(correlation_function_sum, 0, corr_size_in_bytes);

  // Loop over the number of sources to use. Default is prop_n_sources=1. Default source position = 0 0 0 0
  for(int n=0; n<prop_n_sources; n++) {
      printfQuda("Source position: %d %d %d %d\n", prop_source_position[n][0], prop_source_position[n][1], prop_source_position[n][2], prop_source_position[n][3]);
    for (int dil = 0; dil < 12; dil++) {
      const int source[4] = {prop_source_position[n][0],
                               prop_source_position[n][1],
                               prop_source_position[n][2],
                               prop_source_position[n][3]};
      constructPointSpinorSource(qudaSource4D[dil]->V(), cs_param.nSpin, cs_param.nColor, inv_param.cpu_prec, gauge_param.X, dil, source);
      inv_param.solver_normalization = QUDA_SOURCE_NORMALIZATION; // Make explicit for now.

      invertQuda(qudaProp4D[dil]->V(), qudaSource4D[dil]->V(), &inv_param);
      contractQuda(qudaProp4D[dil]->V(), qudaProp4D[dil]->V(),
                   correlation_function+ n_numbers_per_slice * local_corr_offset, contract_type, &inv_param, gauge_param.X);

      if(comm_dim(corr_dim) > 1) comm_gather_array(correlation_function, n_numbers_per_slice * local_corr_length);

      for(int G_idx =0; G_idx <16; G_idx++) {
        for(size_t t=0; t< global_corr_length; t++) {
          correlation_function_sum[n_numbers_per_slice*t + 2*G_idx  ] += correlation_function[n_numbers_per_slice*t + 2*G_idx  ];
          correlation_function_sum[n_numbers_per_slice*t + 2*G_idx+1] += correlation_function[n_numbers_per_slice*t + 2*G_idx+1];
        }
      }
    }

    //print correlators
    for(int G_idx=0; G_idx <16; G_idx++) {
      for(size_t t=0; t<global_corr_length; t++) {
        printfQuda("sum: g=%d t=%lu %e %e\n", G_idx, t, correlation_function_sum[n_numbers_per_slice*t + 2*G_idx], correlation_function_sum[n_numbers_per_slice*t + 2*G_idx + 1]);
      }
    }
  }

  //free memory
  for (int i = 0; i < spinor_dim; i++) {
    delete qudaProp4D[i];
    delete qudaSource4D[i];
  }
  free(correlation_function);
  free(correlation_function_sum);
  free(in_array);
  free(out_array);
  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    freeCloverQuda();
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }
  for (auto & dir : gauge) free(dir);

  printfQuda("----------------------------------------------------------------------------------\n");
  endQuda();
  finalizeComms();

  return 0;
}
