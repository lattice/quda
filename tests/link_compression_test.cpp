#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <misc.h>
#include <qio_field.h>

//#include <zfp.h>

#include <comm_quda.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim);
  
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

int main(int argc, char **argv)
{
  auto app = make_app();
  add_su3_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  if (prec_sloppy == QUDA_INVALID_PRECISION) 
    prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) 
    link_recon_sloppy = link_recon;

  setWilsonGaugeParam(gauge_param);
  setDims(gauge_param.X);
  
  initQuda(device_ordinal);

  setVerbosity(verbosity);

  // call srand() with a rank-dependent seed
  initRand();  

  // All user inputs now defined
  display_test_info();
  
  // Construct gauge arrays
  void *gauge[4], *gauge_fund[4], *gauge_new[4];
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    gauge_fund[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    gauge_new[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    for(int i = 0; i < V * gauge_site_size; i++) ((double*)gauge_new[dir])[i] = 0.0;
  }
   
  // Construct a host gauge field
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  
  // Load the original gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  // Copy the gauge field
  for (int dir = 0; dir < 4; dir++) 
    for(int i = 0; i < V * gauge_site_size; i++) 
      ((double*)gauge_fund[dir])[i] =  ((double*)gauge[dir])[i];
  
  // Do a fundamental docomposition on the gauge field copy for compression tests.
  fundamentalHostGaugeField(gauge_fund, prec);

  // Plaquette
  //----------
  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette gauge precise is %.16e (spatial = %.16e, temporal = %.16e)\n", plaq[0], plaq[1],
             plaq[2]);

#ifdef GPU_GAUGE_TOOLS

  // Topological charge and gauge energy
  //------------------------------------
  // Size of floating point data
  size_t data_size = prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float);
  size_t array_size = V * data_size;
  void *qDensity = malloc(array_size);
  // start the timer
  double time0 = -((double)clock());
  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  param.compute_qcharge = QUDA_BOOLEAN_TRUE;
  param.compute_qcharge_density = QUDA_BOOLEAN_TRUE;
  param.qcharge_density = qDensity;

  gaugeObservablesQuda(&param);

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("Computed Etot, Es, Et, Q is\n%.16e %.16e, %.16e %.16e\nDone in %g secs\n", param.energy[0],
             param.energy[1], param.energy[2], param.qcharge, time0);

  // Use ZFP to compress then decpompress a gauge field in the fundamental rep
  //--------------------------------------------------------------------------
  time0 = -((double)clock()); 
  double comp_ratio = zfp_compress_decompress_link(gauge_fund, gauge_new)/(4 * gauge_site_size);
  
  // Exponentiate the decompressed links
  exponentiateHostGaugeField(gauge_new, su3_taylor_N, prec);
  
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("Total time for compression and decompression = %g (%g per link)\n", time0, time0/(4 * xdim*ydim*zdim*tdim));
  printfQuda("Average compression ratio = %f (%.2fx)\n", comp_ratio, 1.0/comp_ratio);
  
  // Load the reconstrcuted gauge to the device
  loadGaugeQuda((void *)gauge_new, &gauge_param);
  
  // Compute differences of gauge obserables using the reconstructed field
  double plaq_recon[3];
  plaqQuda(plaq_recon);  
  QudaGaugeObservableParam param_recon = newQudaGaugeObservableParam();
  param_recon.compute_qcharge = QUDA_BOOLEAN_TRUE;
  param_recon.compute_qcharge_density = QUDA_BOOLEAN_TRUE;
  param_recon.qcharge_density = qDensity;
  gaugeObservablesQuda(&param_recon);

  printfQuda("\nComputed gauge obvervables: compressed->decompressed:\n");
  printfQuda("plaquette %.16e (spatial = %.16e, temporal = %.16e)\n", plaq_recon[0], plaq_recon[1],
	     plaq_recon[2]);
  printfQuda("Etot, Es, Et, Q decompressed is\n%.16e %.16e, %.16e %.16e\n", param_recon.energy[0],
	     param_recon.energy[1], param_recon.energy[2], param_recon.qcharge);
  
  printfQuda("\nComputed gauge obvervables: original - compressed->decompressed at tol %e:\n", su3_comp_tol);
  printfQuda("plaquette %.16e (spatial = %.16e, temporal = %.16e)\n", plaq[0] - plaq_recon[0], plaq[1] - plaq_recon[1],
             plaq[2] - plaq_recon[2]);
  printfQuda("Etot, Es, Et, Q diff is\n%.16e %.16e, %.16e %.16e\n", param.energy[0] - param_recon.energy[0],
             param.energy[1] - param_recon.energy[1], param.energy[2] - param_recon.energy[2], param.qcharge - param_recon.qcharge);

  // Save if output string is specified
  if (strcmp(gauge_outfile,"")) {
    
    printfQuda("Saving the gauge field to file %s\n", gauge_outfile);
    
    void *cpu_gauge[4];
    for (int dir = 0; dir < 4; dir++) { cpu_gauge[dir] = malloc(V * gauge_site_size * gauge_param.cpu_prec); }
    
    // Copy device field to CPU field
    saveGaugeQuda(cpu_gauge, &gauge_param);    
    
    // Write to disk
    write_gauge_field(gauge_outfile, cpu_gauge, gauge_param.cpu_prec, gauge_param.X, 0, (char**)0);
    
    for (int dir = 0; dir<4; dir++) free(cpu_gauge[dir]);
  } else {
    printfQuda("No output file specified.\n");
  }
  
#else
  printfQuda("Skipping other gauge tests since gauge tools have not been compiled\n");
#endif

  if (verify_results) check_gauge(gauge, gauge_new, 1e-3, gauge_param.cpu_prec);
  
  freeGaugeQuda();
  endQuda();

  // release memory
  for (int dir = 0; dir < 4; dir++) {
    free(gauge[dir]);
    free(gauge_fund[dir]);
    free(gauge_new[dir]);
  }

  finalizeComms();
  return 0;
}
