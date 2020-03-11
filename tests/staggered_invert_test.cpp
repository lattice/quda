#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>
#include <gauge_field.h>
#include <unitarization_links.h>
#include <random_quda.h>

// External headers
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <llfat_utils.h>
#include <qio_field.h>

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon test_type  S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %s         %d/%d/%d          %d \n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy),
             get_staggered_test_type(test_type), xdim, ydim, zdim, tdim);

  printfQuda("\n   Eigensolver parameters\n");
  printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
  printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
  printfQuda(" - number of eigenvectors requested %d\n", eig_nConv);
  printfQuda(" - size of eigenvector search space %d\n", eig_nEv);
  printfQuda(" - size of Krylov space %d\n", eig_nKr);
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
    if (eig_amax < 0)
      printfQuda(" - Chebyshev polynomial maximum will be computed\n");
    else
      printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
  }

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));

  return;
}

int main(int argc, char **argv)
{
  solve_type = QUDA_INVALID_SOLVE;
  // Parse command line options
  auto app = make_app();
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"full", 0}, {"full_ee_prec", 1}, {"full_oo_prec", 2}, {"even", 3},
                                          {"odd", 4},  {"mcg_even", 5},     {"mcg_odd", 6}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (test_type < 0 || test_type > 6) {
    errorQuda("Test type %d is outside the valid range.\n", test_type);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);
  setQudaDefaultPrecs();

  // Only these fermions are supported in this file
  if (dslash_type != QUDA_STAGGERED_DSLASH && 
      dslash_type != QUDA_ASQTAD_DSLASH && 
      dslash_type != QUDA_LAPLACE_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  // Deduce operator, solution, and operator preconditioning types
  setQudaStaggeredInvTestParams();

  display_test_info();
  
  // Set QUDA internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setStaggeredQDPGaugeParam(gauge_param);
  QudaInvertParam inv_param = newQudaInvertParam();
  setStaggeredInvertParam(inv_param);
  QudaEigParam eig_param = newQudaEigParam();
  setEigParam(eig_param);
  // We encapsulate the eigensolver parameters inside the invert parameter structure
  inv_param.eig_param = inv_deflate ? &eig_param : nullptr;

  // This must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gauge_param.X);
  // Hack: use the domain wall dimensions so we may use the 5th dim for multi indexing
  dw_setDims(gauge_param.X, 1); 
  setSpinorSiteSize(6);

  // Staggered Gauge construct START
  //-----------------------------------------------------------------------------------
  // Allocate host staggered gauge fields
  void* qdp_inlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_fatlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_longlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* milc_fatlink = nullptr;
  void* milc_longlink = nullptr;
  void** ghost_fatlink = nullptr;
  void **ghost_longlink = nullptr;
  GaugeField *cpuFat;
  GaugeField *cpuLong;
  
  for (int dir = 0; dir < 4; dir++) {
    qdp_inlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_fatlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_longlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  }
  milc_fatlink = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
  milc_longlink = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
  
  constructStaggeredHostGaugeField(qdp_inlink, qdp_longlink, qdp_fatlink, gauge_param, argc, argv);
  
  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  if(dslash_type == QUDA_ASQTAD_DSLASH) {
    // Compute fat link plaquette
    computeStaggeredPlaquetteQDPOrder(qdp_fatlink, plaq, gauge_param, dslash_type);
    printfQuda("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }

  // Reorder gauge fields to MILC order
  reorderQDPtoMILC(milc_fatlink, qdp_fatlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
  reorderQDPtoMILC(milc_longlink, qdp_longlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
  
  // Staggered vector construct START
  //-----------------------------------------------------------------------------------
  ColorSpinorField *in;
  ColorSpinorField *out;
  ColorSpinorField *ref;
  ColorSpinorField *tmp;
  ColorSpinorParam cs_param;
  constructStaggeredTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  in = quda::ColorSpinorField::Create(cs_param);
  out = quda::ColorSpinorField::Create(cs_param);
  ref = quda::ColorSpinorField::Create(cs_param);
  tmp = quda::ColorSpinorField::Create(cs_param);
  // Staggered vector construct END
  //-----------------------------------------------------------------------------------
    
  // Staggered Gauge construct START
  //-----------------------------------------------------------------------------------
  // FIXME: currently assume staggered is SU(3)
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ? QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  // Create ghost gauge fields in case of multi GPU builds.
#ifdef MULTI_GPU
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  GaugeFieldParam cpuFatParam(milc_fatlink, gauge_param);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuFat = GaugeField::Create(cpuFatParam);
  ghost_fatlink = (void**)cpuFat->Ghost();

  gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(milc_longlink, gauge_param);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuLong = GaugeField::Create(cpuLongParam);
  ghost_longlink = (void**)cpuLong->Ghost();
#endif

  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ? QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  
  // Set MILC specific params and load the gauge fields
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    setStaggeredMILCGaugeParam(gauge_param);
  }
  loadGaugeQuda(milc_fatlink, &gauge_param);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {  
    setStaggeredMILCGaugeParam(gauge_param);
    loadGaugeQuda(milc_longlink, &gauge_param);
  }
  // Staggered Gauge construct END
  //-----------------------------------------------------------------------------------
  
  // Prepare rng
  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 1234);
  rng->Init();

  // Performance measuring
  double *time = new double[Nsrc];
  double *gflops = new double[Nsrc];

  // Pointers for tests 5 and 6
  // Quark masses
  std::vector<double> masses(multishift);
  // Host array for solutions
  void **outArray = (void**)malloc(multishift*sizeof(void*));      
  // QUDA host array for internal checks and malloc
  std::vector<ColorSpinorField*> qudaOutArray(multishift);
  
  switch (test_type) {
  case 0: // full parity solution, full parity system
  case 1: // full parity solution, solving EVEN EVEN prec system
  case 2: // full parity solution, solving ODD ODD prec system
  case 3: // even parity solution, solving EVEN system
  case 4: // odd parity solution, solving ODD system

    if(multishift != 1) {
      printfQuda("Multishift not supported for test %d\n", test_type);
      exit(0);
    }
    
    for (int k = 0; k < Nsrc; k++) {
      constructRandomSpinorSource(in->V(), 1, 3, inv_param.cpu_prec, cs_param.x, *rng);
      //quda::spinorNoise(*in, *rng, QUDA_NOISE_UNIFORM);
      invertQuda(out->V(), in->V(), &inv_param);
      
      time[k] = inv_param.secs;
      gflops[k] = inv_param.gflops / inv_param.secs;
      printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs,
                 inv_param.gflops / inv_param.secs);
      verifyStaggeredInversion(tmp, ref, in, out, mass, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, gauge_param, inv_param, 0);      
    }

    // Compute timings
    if (Nsrc > 1) performanceStats(time, gflops);
    break;
    
  case 5: // multi mass CG, even parity solution, solving EVEN system
  case 6: // multi mass CG, odd parity solution, solving ODD system
    
    if(multishift < 2) {
      printfQuda("Multishift inverter requires more than one shift, multishift = %d\n", multishift);
      exit(0);
    }
    
    inv_param.num_offset = multishift;    
    for(int i=0; i<multishift; i++) {
      // Set masses and offsets
      masses[i] = 0.06 + i*i*0.01;	
      inv_param.offset[i] = 4 * masses[i] * masses[i];
      // Set tolerances for the heavy quarks, these can be set independently
      // (functions of i) if desired
      inv_param.tol_offset[i] = inv_param.tol;
      inv_param.tol_hq_offset[i] = inv_param.tol_hq;
      // Allocate memory and set pointers
      qudaOutArray[i] = ColorSpinorField::Create(cs_param);
      outArray[i] = qudaOutArray[i]->V();
    }
    
    constructRandomSpinorSource(in->V(), 1, 3, inv_param.cpu_prec, cs_param.x, *rng);
    invertMultiShiftQuda((void**)outArray, in->V(), &inv_param);      
    cudaDeviceSynchronize();
    
    printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs, inv_param.gflops / inv_param.secs);
    
    for (int i = 0; i < multishift; i++) {
      printfQuda("%dth solution: mass=%f, ", i, masses[i]);
      verifyStaggeredInversion(tmp, ref, in, qudaOutArray[i], masses[i], qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, gauge_param, inv_param, i);	
    }
    
    for (int i = 0; i < multishift; i++) delete qudaOutArray[i];
    free(outArray);
    break;    
    
  default: errorQuda("Unsupported test type");    
    
  } // switch
  
  delete[] time;
  delete[] gflops;
  
  // Free RNG
  rng->Release();
  delete rng;
  
  // Clean up gauge fields, at least
  for (int dir = 0; dir < 4; dir++) {
    if (qdp_inlink[dir] != nullptr) {
      free(qdp_inlink[dir]);
      qdp_inlink[dir] = nullptr;
    }
    if (qdp_fatlink[dir] != nullptr) {
      free(qdp_fatlink[dir]);
      qdp_fatlink[dir] = nullptr;
    }
    if (qdp_longlink[dir] != nullptr) {
      free(qdp_longlink[dir]);
      qdp_longlink[dir] = nullptr;
    }
  }
  if (milc_fatlink != nullptr) { free(milc_fatlink); milc_fatlink = nullptr; }
  if (milc_longlink != nullptr) { free(milc_longlink); milc_longlink = nullptr; }

#ifdef MULTI_GPU
  if (cpuFat != nullptr) { delete cpuFat; cpuFat = nullptr; }
  if (cpuLong != nullptr) { delete cpuLong; cpuLong = nullptr; }
#endif

  delete in;
  delete out;
  delete ref;
  delete tmp;

  endQuda();
  finalizeComms();
  return 0;
}
