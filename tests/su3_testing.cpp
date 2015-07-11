#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <gauge_field.h>

#include <comm_quda.h>
#include <test_util.h>
#include <gauge_tools.h>

#include <pgauge_monte.h>
#include <random.h>
#include <hisq_links_quda.h>



using   namespace quda;
QudaGaugeParam param;

extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern char latfile[];

int num_failures=0;
int *num_failures_dev;

#define MAX(a,b) ((a)>(b)?(a):(b))
#define DABS(a) ((a)<(0.)?(-(a)):(a))


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
  cudaMalloc((void**)&num_failures_dev, sizeof(int));
  cudaMemset(num_failures_dev, 0, sizeof(int));
  if(num_failures_dev == NULL) errorQuda("cudaMalloc failed for dev_pointer\n");
}

bool compareDoube2(double2 a, double2 b){
  double a0,a1;
  a0 = DABS(a.x - b.x);
  a1=DABS(a.y - b.y);
  double prec_val = 1.0e-5;
  if(prec == QUDA_DOUBLE_PRECISION) prec_val = 1.0e-15;
  if( (a0 < prec_val) && (a1  < prec_val) ) return true;
  return false;
}

bool CheckDeterminant(double2 detu){
  double prec_val = 1.0e-8;
  if(prec == QUDA_DOUBLE_PRECISION) prec_val = 1.0e-15;
  if(DABS(1.0 - detu.x) < prec_val && DABS(detu.y) < prec_val) return true;
  return false;
}


void CallUnitarizeLinks(cudaGaugeField *cudaInGauge){
    unitarizeLinksQuda(*cudaInGauge, num_failures_dev);
    cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
    if(num_failures>0){
      cudaFree(num_failures_dev); 
      errorQuda("Error in the unitarization\n"); 
      exit(1);
    }
    cudaMemset(num_failures_dev, 0, sizeof(int));
}
void RunTest(int argc, char **argv) {


  setVerbosity(QUDA_VERBOSE);
  if (true) {
    printfQuda("Tuning...\n");
    setTuning(QUDA_TUNE_YES);
  }

  param = newQudaGaugeParam();

  //Setup Gauge container!!!!!!
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
  gParam.order       = (param.cuda_prec == QUDA_DOUBLE_PRECISION || param.reconstruct == QUDA_RECONSTRUCT_NO ) ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;


#ifdef MULTI_GPU
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
  cudaGaugeField *cudaInGauge = new cudaGaugeField(gParamEx); 
#else
  cudaGaugeField *cudaInGauge = new cudaGaugeField(gParam);
#endif


  int nsteps = 10;
  int nhbsteps = 4;
  int novrsteps = 4;
  bool coldstart = false;
  double beta_value = 6.2;


  Timer a0,a1;
  a0.Start();
  a1.Start();

  int halfvolume = xdim*ydim*zdim*tdim >> 1;
  printfQuda("xdim=%d\tydim=%d\tzdim=%d\ttdim=%d\trng_size=%d\n",xdim,ydim,zdim,tdim,halfvolume);
  // CURAND random generator initialization
  RNG randstates(halfvolume, 1234, param.X);
  randstates.Init();
  // Reunitarization setup
  SetReunitarizationConsts();


  if(link_recon != QUDA_RECONSTRUCT_8 && coldstart) InitGaugeField( *cudaInGauge);
  else{
    InitGaugeField( *cudaInGauge, randstates.State());
  }
  Plaquette( *cudaInGauge) ;

  for(int step=1; step<=nsteps; ++step){
    printfQuda("Step %d\n",step);
    Monte( *cudaInGauge, randstates.State(), beta_value, nhbsteps, novrsteps);
    //Reunitarize gauge links...
    CallUnitarizeLinks(cudaInGauge);
    Plaquette( *cudaInGauge) ;
  }
  a1.Stop();
  printfQuda("Time Monte -> %.6f s\n", a1.Last());

  double2 detu = getLinkDeterminant(*cudaInGauge);
  double2 plaq = Plaquette( *cudaInGauge) ;
  bool testgen = false;
  //check plaquette value for beta = 6.2
  if(plaq.x < 0.614 && plaq.x > 0.611 && plaq.y < 0.614 && plaq.y > 0.611) testgen = true;
  printf("-----------------------------------------------------------------------\n");
  printf("Test for gauge generation: %s\n", (( testgen && CheckDeterminant(detu)) ? "PASSED" : "FAILED") );
  printf("-----------------------------------------------------------------------\n");

  int reunit_interval = 10;

  printfQuda("Landau gauge fixing with overrelaxation\n");
  plaq = Plaquette( *cudaInGauge) ;
  gaugefixingOVR(*cudaInGauge, 4, 100, 10, 1.5, 0, reunit_interval, 1);
  printf("-----------------------------------------------------------------------\n");
  printf("Test for Landau gauge fixing with overrrelaxation: %s\n", \
    compareDoube2(plaq, Plaquette( *cudaInGauge)) ? "PASSED" : "FAILED");
  printf("-----------------------------------------------------------------------\n");

  printfQuda("Coulomb gauge fixing with overrelaxation\n");
  plaq =  Plaquette( *cudaInGauge);
  gaugefixingOVR(*cudaInGauge, 3, 100, 10, 1.5, 0, reunit_interval, 1);
  printf("-----------------------------------------------------------------------\n");
  printf("Test for Coulomb gauge fixing with overrrelaxation: %s\n", \
    compareDoube2(plaq, Plaquette( *cudaInGauge)) ? "PASSED" : "FAILED");
  printf("-----------------------------------------------------------------------\n");
  
  if(comm_size() == 1){
    printfQuda("Landau gauge fixing with steepest descent method with FFTs\n");
    plaq = Plaquette( *cudaInGauge) ;
    gaugefixingFFT(*cudaInGauge, 4, 100, 10, 0.08, 0, 0, 1);
    printf("-----------------------------------------------------------------------\n");
    printf("Test for Landau gauge fixing with SDM with FFT: %s\n", \
      compareDoube2(plaq, Plaquette( *cudaInGauge)) ? "PASSED" : "FAILED");
    printf("-----------------------------------------------------------------------\n");

    printfQuda("Coulomb gauge fixing with steepest descent method with FFTs\n");
    plaq = Plaquette( *cudaInGauge) ;
    gaugefixingFFT(*cudaInGauge, 3, 100, 10, 0.08, 0, 0, 1);
    printf("-----------------------------------------------------------------------\n");
    printf("Test for Coulomb gauge fixing with SDM with FFT: %s\n", \
      compareDoube2(plaq, Plaquette( *cudaInGauge)) ? "PASSED" : "FAILED");
    printf("-----------------------------------------------------------------------\n");
  }

  detu = getLinkDeterminant(*cudaInGauge);
  double2 tru = getLinkTrace(*cudaInGauge);
  printf("Det: %.16e:%.16e\n", detu.x, detu.y);
  printf("Tr: %.16e:%.16e\n", tru.x/3.0, tru.y/3.0);

  randstates.Release();
  delete cudaInGauge;
  cudaFree(num_failures_dev);
  //Release all temporary memory used for data exchange between GPUs in multi-GPU mode
  PGaugeExchangeFree();
  a0.Stop();
  printfQuda("Time -> %.6f s\n", a0.Last());
}



void SU3GaugeFixTest(int argc, char **argv) {

  initQuda(-1);
  prec = QUDA_DOUBLE_PRECISION;
  link_recon = QUDA_RECONSTRUCT_NO;
  RunTest(argc, argv);
  endQuda();

}

int main(int argc, char **argv){

  xdim=ydim=zdim=tdim=32;
  int i;
  for (i=1; i<argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }

    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  SU3GaugeFixTest(argc, argv);

  finalizeComms();

  return EXIT_SUCCESS;
}
