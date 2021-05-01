#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <gauge_field.h>

#include <comm_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <misc.h>
#include <gauge_tools.h>

#include <pgauge_monte.h>
#include <random_quda.h>
#include <unitarization_links.h>

#include <qio_field.h>

#include <gtest/gtest.h>

using namespace quda;

void display_test_info()
{
  printfQuda("running the following test:\n");
  
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim, Lsdim);

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}


void SetReunitarizationConsts(){
  const double unitarize_eps = 1e-14;
  const double max_error = 1e-10;
  const int reunit_allow_svd = 1;
  const int reunit_svd_only  = 0;
  const double svd_rel_error = 1e-6;
  const double svd_abs_error = 1e-6;
  setUnitarizeLinksConstants(unitarize_eps, max_error,
			     reunit_allow_svd, reunit_svd_only,
			     svd_rel_error, svd_abs_error);
  
}

bool checkDimsPartitioned()
{
  if (comm_dim_partitioned(0) || comm_dim_partitioned(1) || comm_dim_partitioned(2) || comm_dim_partitioned(3))
    return true;
  return false;
}

bool comparePlaquette(double3 a, double3 b)
{
  printfQuda("Plaq:    %.16e, %.16e, %.16e\n", a.x, a.y, a.z);
  printfQuda("Plaq_gf: %.16e, %.16e, %.16e\n", b.x, b.y, b.z);   
  double a0,a1,a2;
  a0 = std::abs(a.x - b.x);
  a1 = std::abs(a.y - b.y);
  a2 = std::abs(a.z - b.z);
  double prec_val = 1.0e-5;
  if (prec == QUDA_DOUBLE_PRECISION) prec_val = 1.0e-15;
  return ((a0 < prec_val) && (a1 < prec_val) && (a2 < prec_val));
}

bool checkDeterminant(double2 detu)
{
  printfQuda("Det: %.16e: %.16e\n", detu.x, detu.y);
  double prec_val = 5e-8;
  if (prec == QUDA_DOUBLE_PRECISION) prec_val = 1.0e-15;
  return std::abs(1.0 - detu.x) < prec_val && std::abs(detu.y) < prec_val;
}

int main(int argc, char **argv)
{
  // command line options  
  auto app = make_app();
  add_gaugefix_option_group(app);
  add_heatbath_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"OVR", 0}, {"FFT", 1}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // initialize the QUDA library
  initQuda(device_ordinal);

  // *** QUDA parameters begin here.
  setVerbosity(QUDA_VERBOSE);
  QudaGaugeParam param = newQudaGaugeParam();

  double3 plaq;
  cudaGaugeField *U;
  int nsteps = heatbath_num_steps;
  int nhbsteps = heatbath_num_heatbath_per_step;
  int novrsteps = heatbath_num_overrelax_per_step;
  bool coldstart = heatbath_coldstart;
  double beta_value = heatbath_beta_value;
  
  RNG * randstates;
  
  // Setup gauge container.
  param.cpu_prec = prec;
  param.cpu_prec = prec;
  param.cuda_prec = prec;
  param.reconstruct = link_recon;
  param.cuda_prec_sloppy = prec;
  param.reconstruct_sloppy = link_recon;
  
  param.type = QUDA_WILSON_LINKS;
  param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  
  param.X[0] = xdim;
  param.X[1] = ydim;
  param.X[2] = zdim;
  param.X[3] = tdim;
  setDims(param.X);
  
  param.anisotropy = 1.0;  //don't support anisotropy for now!!!!!!
  param.t_boundary = QUDA_PERIODIC_T;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  param.ga_pad = 0;
  
  GaugeFieldParam gParam(0, param);
  gParam.pad = 0;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.create      = QUDA_NULL_FIELD_CREATE;
  gParam.link_type   = param.type;
  gParam.reconstruct = param.reconstruct;
  gParam.setPrecision(gParam.Precision(), true);
  
  int y[4];
  int R[4] = {0,0,0,0};
  for(int dir=0; dir<4; ++dir) if(comm_dim_partitioned(dir)) R[dir] = 2;
  for(int dir=0; dir<4; ++dir) y[dir] = param.X[dir] + 2 * R[dir];
  int pad = 0;
  GaugeFieldParam gParamEx(y, prec, link_recon,
			   pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
  gParamEx.create = QUDA_ZERO_FIELD_CREATE;
  gParamEx.order = gParam.order;
  gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParamEx.t_boundary = gParam.t_boundary;
  gParamEx.nFace = 1;
  for(int dir=0; dir<4; ++dir) gParamEx.r[dir] = R[dir];
  U = new cudaGaugeField(gParamEx);

  // CURAND random generator initialization
  randstates = new RNG(gParam, 1234);
  randstates->Init();
    
  int *num_failures_h = (int *)mapped_malloc(sizeof(int));
  int *num_failures_d = (int *)get_mapped_device_pointer(num_failures_h);
  
  if (link_recon != QUDA_RECONSTRUCT_8 && coldstart)
    InitGaugeField(*U);
  else
    InitGaugeField(*U, *randstates);
  
  // Reunitarization setup
  SetReunitarizationConsts();
  plaquette(*U);
  
  for(int step=1; step<=nsteps; ++step){
    printfQuda("Step %d\n",step);
    Monte(*U, *randstates, beta_value, nhbsteps, novrsteps);
    
    //Reunitarize gauge links...
    *num_failures_h = 0;
    unitarizeLinks(*U, num_failures_d);
    qudaDeviceSynchronize();
    if (*num_failures_h > 0) errorQuda("Error in the unitarization\n");
    
    plaquette(*U);
  }
  
  plaq = plaquette(*U);
  printfQuda("Plaq: %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
  
  host_free(num_failures_h);

  // Gauge Fixing Routines
  //---------------------------------------------------------------------------
  switch (test_type) {
  case 0:  
    printfQuda("%s gauge fixing with overrelaxation\n", gf_gauge_dir == 4 ? "Landau" : "Coulomb");
    gaugeFixingOVR(*U, gf_gauge_dir, gf_maxiter, gf_verbosity_interval, gf_ovr_relaxation_boost, gf_tolerance, gf_reunit_interval, gf_theta_condition);
    comparePlaquette(plaq, plaquette(*U));
    break;
    
  case 1:
    if (!checkDimsPartitioned()) {
      printfQuda("%s gauge fixing with steepest descent method with FFTs\n", gf_gauge_dir == 4 ? "Landau" : "Coulomb");
      gaugeFixingFFT(*U, gf_gauge_dir, gf_maxiter, gf_verbosity_interval, gf_fft_alpha, gf_fft_autotune, gf_tolerance, gf_theta_condition);
      comparePlaquette(plaq, plaquette(*U));
    } else {
      errorQuda("FFT gauge fixing not supported for multi GPU geometry");
    }
    break;
    
  default:
    errorQuda("Unknown test type %d", test_type);
  }

  double2 link_trace = getLinkTrace(*U);
  printfQuda("Tr: %.16e:%.16e\n", link_trace.x/3.0, link_trace.y/3.0);

  // Save if output string is specified
  if (strcmp(gauge_outfile,"")) {
    
    printfQuda("Saving the gauge field to file %s\n", gauge_outfile);

    QudaGaugeParam gauge_param = newQudaGaugeParam();
    setWilsonGaugeParam(gauge_param);
    
    void *cpu_gauge[4];
    for (int dir = 0; dir < 4; dir++) { cpu_gauge[dir] = malloc(V * gauge_site_size * gauge_param.cpu_prec); }
    
    cudaGaugeField *gauge;
    gauge = new cudaGaugeField(gParam);
    
    // copy into regular field
    copyExtendedGauge(*gauge, *U, QUDA_CUDA_FIELD_LOCATION);    
    saveGaugeFieldQuda((void*)cpu_gauge, (void*)gauge, &gauge_param);
    
    // Write to disk
    write_gauge_field(gauge_outfile, cpu_gauge, gauge_param.cpu_prec, gauge_param.X, 0, (char**)0);
    
    for (int dir = 0; dir<4; dir++) free(cpu_gauge[dir]);
    delete gauge;
  } else {
    printfQuda("No output file specified.\n");
  }  
  
  delete U;
  
  //Release all temporary memory used for data exchange between GPUs in multi-GPU mode
  PGaugeExchangeFree();
  
  randstates->Release();
  delete randstates;
  
  freeGaugeQuda();    
  endQuda();
  finalizeComms();
  
  return 0;
}
