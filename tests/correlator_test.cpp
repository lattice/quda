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

  //ColorSpinors (Wilson)
  quda::ColorSpinorParam cs_param;
  constructWilsonTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  int spinor_length = cs_param.nColor*cs_param.nSpin;
  setSpinorSiteSize(spinor_length*2); //this sets the global variable my_spinor_site_size

  std::vector<quda::ColorSpinorField *> qudaProp4D(spinor_length); // qudaProp4D is a vector that contains 12 pointers to colorspinorfields.
  std::vector<quda::ColorSpinorField *> qudaSource4D(spinor_length);

  size_t data_size = sizeof(double);
  // Allocate memory on host for one source (0,0,0,0) for each of the 12x12 color+spinor combinations
  auto *out_array = (double*)malloc(12*12*V*2*data_size);
  auto *in_array = (double*)malloc(12*12*V*2*data_size);

  //Actually create ColorSpinorField objects and tell them to use the memory from above
  cs_param.create = QUDA_REFERENCE_FIELD_CREATE;
  for (int dil = 0; dil < 12; dil++) {
    int offset = dil*V*spinor_length*2;
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
  size_t array_length = gauge_param.X[corr_dim];


  //host memory for correlator results
  size_t array_size_in_bytes = 2 * cs_param.nSpin * cs_param.nSpin * comm_dim(corr_dim) * array_length * data_size;
  auto *correlation_function = (double*)malloc(array_size_in_bytes);
  auto *correlation_function_sum = (double*)malloc(array_size_in_bytes);
  memset(correlation_function_sum, 0, array_size_in_bytes);

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
                   correlation_function+2*16*array_length*comm_coord(corr_dim), contract_type, &inv_param, gauge_param.X);

      if(comm_dim(corr_dim) > 1) comm_gather_array(correlation_function, 2*16*array_length);

      for(int gamma_mat=0; gamma_mat<16; gamma_mat++) {
        for(size_t t=0; t<comm_dim(corr_dim) * array_length; t++) {
          correlation_function_sum[2*(16*t + gamma_mat)  ] += correlation_function[2*(16*t + gamma_mat)  ];
          correlation_function_sum[2*(16*t + gamma_mat)+1] += correlation_function[2*(16*t + gamma_mat)+1];
        }
      }
    }

    for(int gamma_mat=0; gamma_mat<16; gamma_mat++) {
      for(size_t t=0; t<comm_dim(corr_dim) * array_length; t++) {
        printfQuda("sum: g=%d t=%lu %e %e\n", gamma_mat, t, correlation_function_sum[2*(16*t + gamma_mat)], correlation_function_sum[2*(16*t + gamma_mat) + 1]);
      }
    }
  }

  //free memory
  for (int i = 0; i < 12; i++) {
    delete qudaProp4D[i];
    delete qudaSource4D[i];
  }
  free(correlation_function);
  free(correlation_function_sum);
  free(in_array);
  free(out_array);
  freeGaugeQuda();
  for (auto & dir : gauge) free(dir);

  printfQuda("----------------------------------------------------------------------------------\n");
  endQuda();
  finalizeComms();

  return 0;
}
