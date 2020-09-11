// QUDA headers
#include <quda.h>

// External headers
#include <host_utils.h>
#include <command_line_params.h>

#include <iostream>

const std::vector<std::string> QudaCorrelatorChannels = {
  "G1", "G2", "G3", "G4", "G5G1", "G5G2", "G5G3", "G5G4", "1", "G5", "S12", "S13", "S14", "S23", "S24", "S34"
};

void print_correlators(const void* correlation_function_sum, const QudaCorrelatorParam corr_param, const int n){
  printfQuda("#src_x src_y src_z src_t    px    py    pz    pt     G   z/t          real          imag  channel\n");
  for (int px = 0; px <= momentum[0]; px++) {
    for (int py = 0; py <= momentum[1]; py++) {
      for (int pz = 0; pz <= momentum[2]; pz++) {
        for (int pt = 0; pt <= momentum[3]; pt++) {
          for (int G_idx = 0; G_idx < 16; G_idx++) {
            for (size_t t = 0; t < corr_param.global_corr_length; t++) {
              int index_real = (px + py * (momentum[0] + 1) + pz * (momentum[0] + 1) * (momentum[1] + 1)
                                + pt * (momentum[0] + 1) * (momentum[1] + 1) * (momentum[2] + 1))
                               * corr_param.n_numbers_per_slice * corr_param.global_corr_length
                               + corr_param.n_numbers_per_slice * ((t + corr_param.overall_shift_dim) % corr_param.global_corr_length) + 2 * G_idx;
              int index_imag = index_real + 1;
              double sign = G_idx < 8 ? -1. : 1.; // the minus sign from g5gm -> gmg5
              printfQuda(" %5d %5d %5d %5d %5d %5d %5d %5d %5d %5lu % e % e #%s",
                         prop_source_position[n][0], prop_source_position[n][1], prop_source_position[n][2], prop_source_position[n][3], px, py, pz, pt, G_idx, t,
                         ((double *)correlation_function_sum)[index_real] * sign,
                         ((double *)correlation_function_sum)[index_imag] * sign, QudaCorrelatorChannels[G_idx].c_str());
              printfQuda("\n");
            }
          }
        }
      }
    }
  }
}

void save_correlators_to_file(const void* correlation_function_sum, const QudaCorrelatorParam &corr_param, const int n){
  std::ofstream corr_file;
  std::stringstream filepath;
  filepath << correlator_save_dir << "/";
  filepath << "mcorr";
  switch(corr_param.corr_flavors){
  case QUDA_CORRELATOR_QQ: filepath << "_qq"; break;
  case QUDA_CORRELATOR_QS: filepath << "_qs"; break;
  case QUDA_CORRELATOR_QL: filepath << "_ql"; break;
  default:break;
  }
  switch(contract_type){
  case QUDA_CONTRACT_TYPE_DR_FT_Z:
  case QUDA_CONTRACT_TYPE_OPEN_FT_Z:
  case QUDA_CONTRACT_TYPE_OPEN_SUM_Z:
    filepath << "_s"; //spatial
    break;
  default: filepath << "_t"; //temporal
  }
  filepath << "_s" << dim[0] << "t" << dim[3];
  if (correlator_file_affix[0] != '\0') {
    filepath << "_" << correlator_file_affix ;
    }
  filepath << "_" << "k" << std::setprecision(5) << std::fixed << kappa;
  filepath << ".dat";
  printfQuda("Saving correlator in %s \n", filepath.str().c_str());

  corr_file.open(filepath.str());

  const int src_width = 6, mom_width = 3, precision = 8;
  const int float_width = precision+9; //for scientific notation
  corr_file << "#"
            << std::setw(src_width) << "src_x"
            << std::setw(src_width) << "src_y"
            << std::setw(src_width) << "src_z"
            << std::setw(src_width) << "src_t"
            << std::setw(mom_width) << "px"
            << std::setw(mom_width) << "py"
            << std::setw(mom_width) << "pz"
            << std::setw(mom_width) << "pt"
            << std::setw(src_width) << "G"
            << std::setw(src_width) << "z/t"
            << std::setw(float_width) << "real"
            << std::setw(float_width) << "imag"
            << std::endl;
  for (int px = 0; px <= momentum[0]; px++) {
    for (int py = 0; py <= momentum[1]; py++) {
      for (int pz = 0; pz <= momentum[2]; pz++) {
        for (int pt = 0; pt <= momentum[3]; pt++) {
          for (int G_idx = 0; G_idx < 16; G_idx++) {
            for (size_t t = 0; t < corr_param.global_corr_length; t++) {
              size_t index_real = (px + py * (momentum[0] + 1) + pz * (momentum[0] + 1) * (momentum[1] + 1)
                                + pt * (momentum[0] + 1) * (momentum[1] + 1) * (momentum[2] + 1))
                               * corr_param.n_numbers_per_slice * corr_param.global_corr_length
                               + corr_param.n_numbers_per_slice * ((t + corr_param.overall_shift_dim) % corr_param.global_corr_length) + 2 * G_idx;
              size_t index_imag = index_real + 1;
              double sign = G_idx < 8 ? -1. : 1.; // the minus sign from g5gm -> gmg5
              corr_file << " "
                        << std::setw(src_width) << prop_source_position[n][0]
                        << std::setw(src_width) << prop_source_position[n][1]
                        << std::setw(src_width) << prop_source_position[n][2]
                        << std::setw(src_width) << prop_source_position[n][3]
                        << std::setw(mom_width) << px
                        << std::setw(mom_width) << py
                        << std::setw(mom_width) << pz
                        << std::setw(mom_width) << pt
                        << std::setw(src_width) << G_idx
                        << std::setw(src_width) << t
                        << std::setw(float_width) << std::setprecision(precision)
                        << std::scientific << ((double *)correlation_function_sum)[index_real] * sign
                        << std::setw(float_width) << std::setprecision(precision)
                        << std::scientific << ((double *)correlation_function_sum)[index_imag] * sign
                        << std::endl;
            }
          }
        }
      }
    }
  }
  corr_file.close();
}


int main(int argc, char **argv)
{
  setQudaDefaultMgTestParams();
  auto app = make_app();   // Parameter class that reads cmdline arguments. It modifies global variables.
  add_multigrid_option_group(app);
  add_eigen_option_group(app);
  add_propagator_option_group(app);
  add_contraction_option_group(app);
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
      printfQuda("dslash_type %d not supported\n", dslash_type);
      exit(0);
    }
    if (inv_multigrid) {
      // Only these fermions are supported with MG
      if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH) {
        printfQuda("dslash_type %d not supported for MG\n", dslash_type);
        exit(0);
      }
      // Only these solve types are supported with MG
      if (solve_type != QUDA_DIRECT_SOLVE && solve_type != QUDA_DIRECT_PC_SOLVE) {
        printfQuda("Solve_type %d not supported with MG. Please use QUDA_DIRECT_SOLVE or QUDA_DIRECT_PC_SOLVE\n\n",
                   solve_type);
        exit(0);
      }
    }
  }

  initQuda(device);

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
      printfQuda("all the MG settings will be shared for qq, ql and qs propagator\n");
      for (int i = 0; i < mg_levels; i++) {
         if (strcmp(mg_param.vec_infile[i], "") != 0 || strcmp(mg_param.vec_outfile[i], "") != 0){
          printfQuda("Save or write vec not possible! As when open flavor turned on inverter will be called "
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

  printfQuda("-----------------------------------------------------------------------------------\n");

  // compute plaquette
  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // Allocate host side memory for clover terms if needed.
  void *clover = nullptr;
  void *clover_inv = nullptr;
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    if (!compute_clover) { errorQuda("Specified clover dslash-type but did not specify compute-clover!"); }
    clover = malloc(V * clover_site_size * host_clover_data_type_size);
    clover_inv = malloc(V * clover_site_size * host_spinor_data_type_size);
    constructHostCloverField(clover, clover_inv, inv_param);
    if (inv_multigrid) {
      // This line ensures that if we need to construct the clover inverse (in either the smoother or the solver) we do so
      if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE) {
        inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
      }
    }
    loadCloverQuda(clover, clover_inv, &inv_param);
    if (inv_multigrid) {
      // Restore actual solve_type we want to do
      inv_param.solve_type = solve_type;
    }
  }

  // Now QUDA is initialised and the fields are loaded, we may setup the preconditioner
  void *mg_preconditioner = nullptr;
  if (inv_multigrid) {
    mg_preconditioner = newMultigridQuda(&mg_param);
    inv_param.preconditioner = mg_preconditioner;
  }

  // Wilson ColorSpinorParams
  quda::ColorSpinorParam cs_param;
  quda::ColorSpinorParam *cs_param_ptr = &cs_param;
  constructWilsonSpinorParam(&cs_param, &inv_param, &gauge_param);
  int spinor_dim = cs_param.nColor * cs_param.nSpin;
  setSpinorSiteSize(spinor_dim * 2); // this sets the global variable my_spinor_site_size

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

  // Decide between contract types (open or with gammas, summed or non-summed, spatial or temporal, finite momentum)
  size_t corr_dim = 0, local_corr_length = 0;
  int Nmom = (momentum[0]+1)*(momentum[1]+1);
  if (contract_type == QUDA_CONTRACT_TYPE_DR_FT_Z) {
    corr_dim = 2;
    momentum[2]=0;
    Nmom *= (momentum[3]+1);
  } else if (contract_type == QUDA_CONTRACT_TYPE_DR_FT_T) {
    corr_dim = 3;
    momentum[3]=0;
    Nmom *= (momentum[2]+1);
  } else {
    errorQuda("Unsupported contraction type %d given", contract_type);
  }
  local_corr_length = gauge_param.X[corr_dim];

  // some lengths and sizes
  QudaCorrelatorParam corr_param;
  corr_param.global_corr_length = local_corr_length * comm_dim(corr_dim);
  corr_param.n_numbers_per_slice = 2 * cs_param.nSpin * cs_param.nSpin;
  corr_param.corr_size_in_bytes = Nmom*corr_param.n_numbers_per_slice * corr_param.global_corr_length * sizeof(double);
  corr_param.corr_flavors = QUDA_CORRELATOR_QQ;

  void* correlation_function_sum = malloc(corr_param.corr_size_in_bytes); // This is where the result will be stored

  // Loop over the number of sources to use. Default is prop_n_sources=1 and source position = 0 0 0 0
  for (int n = 0; n < prop_n_sources; n++) {
    printfQuda("Source position: %d %d %d %d\n", prop_source_position[n][0], prop_source_position[n][1],
               prop_source_position[n][2], prop_source_position[n][3]);
    const int source[4]
      = {prop_source_position[n][0], prop_source_position[n][1], prop_source_position[n][2], prop_source_position[n][3]};

    // The overall shift of the position of the corr. need this when the source is not at origin.
    corr_param.overall_shift_dim = source[corr_dim];

    // Loop over spin X color dilution positions, construct the sources and invert
    for (int i = 0; i < spinor_dim; i++) {
      // FIXME add the smearing
      constructPointSpinorSource(source_array_ptr[i], cs_param.nSpin, cs_param.nColor, inv_param.cpu_prec,
                                 gauge_param.X, i, source);
      inv_param.solver_normalization = QUDA_SOURCE_NORMALIZATION; // Make explicit for now.
      invertQuda(prop_array_ptr[i], source_array_ptr[i], &inv_param);
    }
    // Coming soon....
    // propagatorQuda(prop_array_ptr, source_array_ptr, &inv_param, &correlation_function_sum, contract_type, (void *)cs_param_ptr, gauge_param.X);

    memset(correlation_function_sum, 0, corr_param.corr_size_in_bytes); // zero out the result array
    contractFTQuda(prop_array_ptr, prop_array_ptr, &correlation_function_sum, contract_type, &inv_param,
                   (void *)cs_param_ptr, gauge_param.X, source, momentum.begin());

    // Print correlators for this source
    print_correlators(correlation_function_sum, corr_param, n);
    save_correlators_to_file(correlation_function_sum, corr_param, n);
  }

  //FIXME abstract the whole flavor selection into one function!
  if(open_flavor){
    //first we calculate heavy-light correlators
    kappa=kappa_light;
    if (inv_multigrid) {
      setMultigridInvertParam(inv_param);
      mg_param.invert_param = &mg_inv_param;
      setMultigridParam(mg_param);
    } else {
      setInvertParam(inv_param);
    }
    inv_param.eig_param = nullptr;

    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      constructHostCloverField(clover, clover_inv, inv_param);
      if (inv_multigrid) {
        if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE) {
          inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
        }
      }
      loadCloverQuda(clover, clover_inv, &inv_param);
      if (inv_multigrid) {
        inv_param.solve_type = solve_type;
      }
    }
    if (inv_multigrid) {
      inv_param.preconditioner = mg_preconditioner;
    }

    constructWilsonSpinorParam(&cs_param, &inv_param, &gauge_param);
    auto *prop_array_light = (double *)malloc(spinor_dim * spinor_dim * V * 2 * bytes_per_float);
    void *prop_array_ptr_light[spinor_dim];
    for (int i = 0; i < spinor_dim; i++) {
      int offset = i * V * spinor_dim * 2;
      prop_array_ptr_light[i] = prop_array_light + offset;
    }

    for (int n = 0; n < prop_n_sources; n++) {
      printfQuda("Source position: %d %d %d %d\n", prop_source_position[n][0], prop_source_position[n][1],
                 prop_source_position[n][2], prop_source_position[n][3]);
      const int source[4] = {prop_source_position[n][0], prop_source_position[n][1], prop_source_position[n][2],
                             prop_source_position[n][3]};

      corr_param.overall_shift_dim = source[corr_dim];

      // FIXME add the smearing too
      for (int i = 0; i < spinor_dim; i++) {
        constructPointSpinorSource(source_array_ptr[i], cs_param.nSpin, cs_param.nColor, inv_param.cpu_prec,
                                   gauge_param.X, i, source);
        inv_param.solver_normalization = QUDA_SOURCE_NORMALIZATION; // Make explicit for now.
        invertQuda(prop_array_ptr_light[i], source_array_ptr[i], &inv_param);
      }
      memset(correlation_function_sum, 0, corr_param.corr_size_in_bytes);
      contractFTQuda(prop_array_ptr, prop_array_ptr_light, &correlation_function_sum, contract_type, &inv_param,
                     (void *)cs_param_ptr, gauge_param.X, source, momentum.begin());
      printfQuda("printing heavy-light correlators\n");
      corr_param.corr_flavors = QUDA_CORRELATOR_QL;
      print_correlators(correlation_function_sum, corr_param, n);
      save_correlators_to_file(correlation_function_sum, corr_param, n);
    }
    free(prop_array_light);

    //then we calculate heavy-strange correlators
    kappa=kappa_strange;
    if (inv_multigrid) {
      setMultigridInvertParam(inv_param);
      mg_param.invert_param = &mg_inv_param;
      setMultigridParam(mg_param);
    } else {
      setInvertParam(inv_param);
    }
    inv_param.eig_param = nullptr;

    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      constructHostCloverField(clover, clover_inv, inv_param);
      if (inv_multigrid) {
        if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE) {
          inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
        }
      }
      loadCloverQuda(clover, clover_inv, &inv_param);
      if (inv_multigrid) {
        inv_param.solve_type = solve_type;
      }
    }
    if (inv_multigrid) {
      inv_param.preconditioner = mg_preconditioner;
    }

    constructWilsonSpinorParam(&cs_param, &inv_param, &gauge_param);
    auto *prop_array_strange = (double *)malloc(spinor_dim * spinor_dim * V * 2 * bytes_per_float);
    void *prop_array_ptr_strange[spinor_dim];
    for (int i = 0; i < spinor_dim; i++) {
      int offset = i * V * spinor_dim * 2;
      prop_array_ptr_strange[i] = prop_array_strange + offset;
    }

    for (int n = 0; n < prop_n_sources; n++) {
      printfQuda("Source position: %d %d %d %d\n", prop_source_position[n][0], prop_source_position[n][1],
                 prop_source_position[n][2], prop_source_position[n][3]);
      const int source[4] = {prop_source_position[n][0], prop_source_position[n][1], prop_source_position[n][2],
                             prop_source_position[n][3]};

      corr_param.overall_shift_dim = source[corr_dim];

      // FIXME add the smearing too
      for (int i = 0; i < spinor_dim; i++) {
        constructPointSpinorSource(source_array_ptr[i], cs_param.nSpin, cs_param.nColor, inv_param.cpu_prec,
                                   gauge_param.X, i, source);
        inv_param.solver_normalization = QUDA_SOURCE_NORMALIZATION; // Make explicit for now.
        invertQuda(prop_array_ptr_strange[i], source_array_ptr[i], &inv_param);
      }
      memset(correlation_function_sum, 0, corr_param.corr_size_in_bytes);
      contractFTQuda(prop_array_ptr, prop_array_ptr_strange, &correlation_function_sum, contract_type, &inv_param,
                     (void *)cs_param_ptr, gauge_param.X, source, momentum.begin());
      printfQuda("printing heavy-strange correlators\n");
      corr_param.corr_flavors = QUDA_CORRELATOR_QS;
      print_correlators(correlation_function_sum, corr_param, n);
      save_correlators_to_file(correlation_function_sum, corr_param, n);
    }
    free(prop_array_strange);
  }

  if (inv_multigrid) destroyMultigridQuda(mg_preconditioner);

  // free memory
  freeGaugeQuda();
  for (auto &dir : gauge) free(dir);
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    freeCloverQuda();
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }
  free(source_array);
  free(prop_array);
  free(correlation_function_sum);

  printfQuda("----------------------------------------------------------------------------------\n");
  endQuda();
  finalizeComms();

  return 0;
}
