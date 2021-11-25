//C++ headers 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//QUDA headers
#include <quda.h>
#include <gauge_field.h>
#include <comm_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <gauge_tools.h>
#include <misc.h>

#include <pgauge_monte.h>
#include <random_quda.h>
#include <unitarization_links.h>

// Local helper functions
//------------------------------------------------------------------------------------
void saveGaugeField(int step, cudaGaugeField *gaugeEx, cudaGaugeField *gauge)
{
  // Construct a filename
  char param_str[256];
  char this_file_name[256];
  char base_file_name[256];
  strcpy(base_file_name, gauge_outfile);
  strcpy(this_file_name, gauge_outfile);
  sprintf(param_str, "_heatbath_Lx%dLy%dLz%dLt%d_beta%f_step%d.lime", xdim, ydim, zdim, tdim, heatbath_beta_value, step);
  strcat(this_file_name, param_str);
  printfQuda("Saving the gauge field to file %s\n", this_file_name);
  strcpy(gauge_outfile, this_file_name);

  // Save the device gauge field
  saveDeviceGaugeField(gaugeEx, gauge);

  // Restore outfile base
  strcpy(gauge_outfile, base_file_name);
}

void constructGaugeField(QudaGaugeParam &gauge_param, cudaGaugeField *gaugeEx,
			 cudaGaugeField *gauge, RNG *randstates) {
  
  if (strcmp(latfile, "")) { // We loaded in a gauge field
    // copy internal extended field to gaugeEx
    copyExtendedResidentGaugeQuda((void*)gaugeEx);
  } else {
    if (heatbath_coldstart) InitGaugeField(*gaugeEx);
    else InitGaugeField(*gaugeEx, *randstates);
    
    // copy into regular field
    copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);
    
    // load the gauge field from gauge
    gauge_param.gauge_order = gauge->Order();
    gauge_param.location = QUDA_CUDA_FIELD_LOCATION;
    
    loadGaugeQuda(gauge->Gauge_p(), &gauge_param);
  }
}

void display_info()
{
  printfQuda("running the following simulation:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim, Lsdim);

  printfQuda("Heatbath with %d overrelaxation and %d heatbath hits per step\n", heatbath_num_overrelax_per_step, heatbath_num_heatbath_per_step);
  printfQuda(" - Start type %s\n", strcmp(latfile,"") ? "loaded" : (heatbath_coldstart ? "cold" : "hot"));
  printfQuda(" - Warm up steps %d\n", heatbath_warmup_steps);
  printfQuda(" - Measurement Steps %d\n", heatbath_num_steps);
  printfQuda(" - Step start %d\n", heatbath_step_start);
  printfQuda(" - Checkpoint Steps %d\n", heatbath_checkpoint);
  printfQuda(" - Beta %f\n", heatbath_beta_value);
  
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

int main(int argc, char **argv)
{
  // command line options
  auto app = make_app();
  add_heatbath_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Initialise QUDA
  //----------------------------------------------------------------------------
  // Set values for precisions via the command line.
  setQudaPrecisions();
  
  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // initialize the QUDA library
  initQuda(device_ordinal);
  
  // call srand() with a rank-dependent seed
  initRand();

  // Set verbosity
  setVerbosity(verbosity);

  // Set QUDA's internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  // Set the dimensions
  setDims(gauge_param.X);  

  // Allocate space on the host
  void *load_gauge[4];
  for (int dir = 0; dir < 4; dir++) { load_gauge[dir] = malloc(V * gauge_site_size * gauge_param.cpu_prec); }
  constructHostGaugeField(load_gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)load_gauge, &gauge_param);
  
  // All user inputs now defined
  display_info();
  //----------------------------------------------------------------------------
  

  // Construct an extended device gauge field
  //--------------------------------------------------------------------------
  //using namespace quda;
  GaugeFieldParam gParam(0, gauge_param);
  gParam.pad = 0;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.create      = QUDA_NULL_FIELD_CREATE;
  gParam.link_type   = gauge_param.type;
  gParam.reconstruct = gauge_param.reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  cudaGaugeField *gauge = new cudaGaugeField(gParam);
  
  int pad = 0;
  int y[4];
  int R[4] = {0,0,0,0};
  for(int dir=0; dir<4; ++dir) if(comm_dim_partitioned(dir)) R[dir] = 2;
  for(int dir=0; dir<4; ++dir) y[dir] = gauge_param.X[dir] + 2 * R[dir];
  GaugeFieldParam gParamEx(y, prec, link_recon,
			   pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
  gParamEx.create = QUDA_ZERO_FIELD_CREATE;
  gParamEx.order = gParam.order;
  gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParamEx.t_boundary = gParam.t_boundary;
  gParamEx.nFace = 1;
  for(int dir=0; dir<4; ++dir) gParamEx.r[dir] = R[dir];
  cudaGaugeField *gaugeEx = new cudaGaugeField(gParamEx);
  
  // CURAND random generator initialization
  RNG *randstates = new RNG(*gauge, 1234);
  
  constructGaugeField(gauge_param, gaugeEx, gauge, randstates);
  //--------------------------------------------------------------------------
  
  
  // Plaquette and Q charge measurement
  //--------------------------------------------------------------------------
  // start the timer
  double time0 = -((double)clock());
  QudaGaugeObservableParam param = newQudaGaugeObservableParam();
  param.compute_plaquette = QUDA_BOOLEAN_TRUE;
  param.compute_qcharge = QUDA_BOOLEAN_TRUE;
  
  // Run the QUDA computation
  gaugeObservablesQuda(&param);
  
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  
  printfQuda("Computed plaquette is %.16e (spatial = %.16e, temporal = %.16e)\n", param.plaquette[0], param.plaquette[1], param.plaquette[2]);
  printfQuda("Computed q charge = %.16e\n", param.qcharge);
  //--------------------------------------------------------------------------
  
  
  // Begin 
  //--------------------------------------------------------------------------
  // Start the timer
  time0 = -((double)clock());
    
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  printfQuda("Heatbath complete, total time = %g secs\n", time0);
   
  //Release all temporary memory used for data exchange between GPUs in multi-GPU mode
  PGaugeExchangeFree();
  
  delete gauge;
  delete gaugeEx;
  delete randstates;
  for (int dir = 0; dir<4; dir++) free(load_gauge[dir]);
  //--------------------------------------------------------------------------
  
  // Finalize the QUDA library
  endQuda();
  finalizeComms();
  
  return 0;
}
