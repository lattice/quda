#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <gauge_field.h>

#include <comm_quda.h>
#include <test_util.h>
#include <test_params.h>
#include <gauge_tools.h>
#include "misc.h"

#include <pgauge_monte.h>
#include <random_quda.h>
#include <unitarization_links.h>

#include <qio_field.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif



#define MAX(a,b) ((a)>(b)?(a):(b))
#define DABS(a) ((a)<(0.)?(-(a)):(a))


namespace quda {
  extern void setTransferGPU(bool);
}

void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim, Lsdim);     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  
  return ;
  
}

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;

void setGaugeParam(QudaGaugeParam &gauge_param) {
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.tadpole_coeff = 1.0;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;

  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif
}

 void setReunitarizationConsts(){
   using namespace quda;
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

void CallUnitarizeLinks(quda::cudaGaugeField *cudaInGauge){
   using namespace quda;
   int *num_failures_dev = (int*)device_malloc(sizeof(int));
   int num_failures;
   cudaMemset(num_failures_dev, 0, sizeof(int));
   unitarizeLinks(*cudaInGauge, num_failures_dev);

   cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
   if(num_failures>0) errorQuda("Error in the unitarization\n");
   device_free(num_failures_dev);
  }

int main(int argc, char **argv)
{

  // command line options
  auto app = make_app();
  // app->get_formatter()->column_width(40);
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** QUDA parameters begin here.

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  setDims(gauge_param.X);

  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *load_gauge[4];

  for (int dir = 0; dir < 4; dir++) {
    load_gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, load_gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(load_gauge, 2, gauge_param.cpu_prec, &gauge_param);
  }

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  {
    using namespace quda;
    GaugeFieldParam gParam(0, gauge_param);
    gParam.pad = 0;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.create      = QUDA_NULL_FIELD_CREATE;
    gParam.link_type   = gauge_param.type;
    gParam.reconstruct = gauge_param.reconstruct;
    gParam.order       = (gauge_param.cuda_prec == QUDA_DOUBLE_PRECISION || gauge_param.reconstruct == QUDA_RECONSTRUCT_NO )
      ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
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
    randstates->Init();

    int nsteps = heatbath_num_steps;
    int nwarm = heatbath_warmup_steps;
    int nhbsteps = heatbath_num_heatbath_per_step;
    int novrsteps = heatbath_num_overrelax_per_step;
    bool  coldstart = heatbath_coldstart;
    double beta_value = heatbath_beta_value;

    printfQuda("Starting heatbath for beta = %f from a %s start\n", beta_value, strcmp(latfile,"") ? "loaded" : (coldstart ? "cold" : "hot"));
    printfQuda("  %d Heatbath hits and %d overrelaxation hits per step\n", nhbsteps, novrsteps);
    printfQuda("  %d Warmup steps\n", nwarm);
    printfQuda("  %d Measurement steps\n", nsteps);

    if (strcmp(latfile,"")) { // Copy in loaded gauge field 

      printfQuda("Loading the gauge field in %s\n", latfile);

      loadGaugeQuda(load_gauge, &gauge_param);
      // Get pointer to internal resident gauge field
      cudaGaugeField* extendedGaugeResident = new cudaGaugeField(gParamEx);
      copyExtendedResidentGaugeQuda((void*)extendedGaugeResident, QUDA_CUDA_FIELD_LOCATION);
      InitGaugeField(*gaugeEx);
      copyExtendedGauge(*gaugeEx, *extendedGaugeResident, QUDA_CUDA_FIELD_LOCATION);
      delete extendedGaugeResident;

    } else if (link_recon != QUDA_RECONSTRUCT_8 && coldstart) {
      InitGaugeField( *gaugeEx);
    } else {
      InitGaugeField( *gaugeEx, *randstates );
    }

    // copy into regular field
    copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);

    // load the gauge field from gauge
    gauge_param.gauge_order = (gauge_param.cuda_prec == QUDA_DOUBLE_PRECISION || gauge_param.reconstruct == QUDA_RECONSTRUCT_NO )
      ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
    gauge_param.location = QUDA_CUDA_FIELD_LOCATION;

    loadGaugeQuda(gauge->Gauge_p(), &gauge_param);
    double3 plaq = plaquette(*gaugeEx);
    double charge = qChargeQuda();
    printfQuda("Initial gauge field plaquette = %e topological charge = %e\n", plaq.x, charge);

    // Reunitarization setup
    setReunitarizationConsts();
    plaquette(*gaugeEx);

    // Do a warmup if requested
    if (nwarm > 0) {
      for (int step = 1; step <= nwarm; ++step) {
        Monte( *gaugeEx, *randstates, beta_value, nhbsteps, novrsteps);  
        CallUnitarizeLinks(gaugeEx);
      }
    }
      

    // copy into regular field
    copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);

    // load the gauge field from gauge
    gauge_param.gauge_order = (gauge_param.cuda_prec == QUDA_DOUBLE_PRECISION || gauge_param.reconstruct == QUDA_RECONSTRUCT_NO )
      ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
    gauge_param.location = QUDA_CUDA_FIELD_LOCATION;

    loadGaugeQuda(gauge->Gauge_p(), &gauge_param);
    plaq = plaquette(*gaugeEx);
    charge = qChargeQuda();
    printfQuda("step=0 plaquette = %e topological charge = %e\n", plaq.x, charge);


    freeGaugeQuda();

    for(int step=1; step<=nsteps; ++step){
      printfQuda("Step %d\n", step);
      Monte( *gaugeEx, *randstates, beta_value, nhbsteps, novrsteps);

      //Reunitarize gauge links...
      CallUnitarizeLinks(gaugeEx);

      // copy into regular field
      copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);

      loadGaugeQuda(gauge->Gauge_p(), &gauge_param);
      plaq = plaquette(*gaugeEx);
      charge = qChargeQuda();
      printfQuda("step=%d plaquette = %e topological charge = %e\n", step, plaq.x, charge);

      freeGaugeQuda();
    }

    // Save if output string is specified
    if (strcmp(gauge_outfile,"")) {

      printfQuda("Saving the gauge field to file %s\n", gauge_outfile);

      QudaGaugeParam gauge_param = newQudaGaugeParam();
      setGaugeParam(gauge_param);

      size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
      void *cpu_gauge[4];
      for (int dir = 0; dir < 4; dir++) {
        cpu_gauge[dir] = malloc(V*gaugeSiteSize*gSize);
      }

      // copy into regular field
      copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);

      saveGaugeFieldQuda((void*)cpu_gauge, (void*)gauge, &gauge_param);

      write_gauge_field(gauge_outfile, cpu_gauge, gauge_param.cpu_prec, gauge_param.X, 0, (char**)0);


      for (int dir = 0; dir<4; dir++) free(cpu_gauge[dir]);
    } else {
      printfQuda("No output file specified.\n");
    }

    delete gauge;
    delete gaugeEx;
    //Release all temporary memory used for data exchange between GPUs in multi-GPU mode
    PGaugeExchangeFree();
 
    randstates->Release();
    delete randstates;
  }

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
    
  //printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
  //inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);
  printfQuda("\nDone, total time = %g secs\n", 
	 time0);

  freeGaugeQuda();
  
  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  for (int dir = 0; dir<4; dir++) free(load_gauge[dir]);

  return 0;
}
