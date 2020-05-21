#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <quda.h>
#include "host_utils.h"
#include <command_line_params.h>
#include "gauge_field.h"
#include "misc.h"
#include "hisq_force_reference.h"
#include "ks_improved_force.h"
#include "momentum.h"
#include <dslash_quda.h> 
#include <sys/time.h>

#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

using namespace quda;

cudaGaugeField *cudaGauge = NULL;
cpuGaugeField  *cpuGauge  = NULL;

cudaGaugeField *cudaForce = NULL;
cpuGaugeField  *cpuForce = NULL;

cudaGaugeField *cudaMom = NULL;
cpuGaugeField *cpuMom  = NULL;
cpuGaugeField *refMom  = NULL;

static QudaGaugeParam qudaGaugeParam;
static QudaGaugeParam qudaGaugeParam_ex;
static void* hw; // the array of half_wilson_vector

QudaGaugeFieldOrder gauge_order = QUDA_QDP_GAUGE_ORDER;

cpuGaugeField *cpuOprod = NULL;
cudaGaugeField *cudaOprod = NULL;
cpuGaugeField *cpuLongLinkOprod = NULL;
cudaGaugeField *cudaLongLinkOprod = NULL;

int ODD_BIT = 1;

QudaPrecision link_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision hw_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cpu_hw_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision mom_prec = QUDA_DOUBLE_PRECISION;

cudaGaugeField *cudaGauge_ex = NULL;
cpuGaugeField  *cpuGauge_ex  = NULL;
cudaGaugeField *cudaForce_ex = NULL;
cpuGaugeField  *cpuForce_ex = NULL;
cpuGaugeField *cpuOprod_ex = NULL;
cudaGaugeField *cudaOprod_ex = NULL;
cpuGaugeField *cpuLongLinkOprod_ex = NULL;
cudaGaugeField *cudaLongLinkOprod_ex = NULL;

GaugeFieldParam gParam_ex;
GaugeFieldParam gParam;

static void setPrecision(QudaPrecision precision)
{
  link_prec = precision;
  hw_prec = precision;
  cpu_hw_prec = precision;
  mom_prec = precision;
  return;
}

void total_staple_io_flops(QudaPrecision prec, QudaReconstructType recon, double* io, double* flops)
{
  //total IO counting for the middle/side/all link kernels
  //Explanation about these numbers can be founed in the corresnponding kernel functions in
  //the hisq kernel core file
  int linksize = prec*recon;
  int cmsize = prec*18;

  int matrix_mul_flops = 198;
  int matrix_add_flops = 18;

  int num_calls_middle_link[6] = {24, 24, 96, 96, 24, 24};
  int middle_link_data_io[6][2] = {
    {3,6},
    {3,4},
    {3,7},
    {3,5},
    {3,5},
    {3,2}
  };
  int middle_link_data_flops[6][2] = {
    {3,1},
    {2,0},
    {4,1},
    {3,0},
    {4,1},
    {2,0}
  };

  int num_calls_side_link[2]= {192, 48};
  int side_link_data_io[2][2] = {
    {1, 6},
    {0, 3}
  };
  int side_link_data_flops[2][2] = {
    {2, 2},
    {0, 1}
  };

  int num_calls_all_link[2] ={192, 192};
  int all_link_data_io[2][2] = {
    {3, 8},
    {3, 6}
  };
  int all_link_data_flops[2][2] = {
    {6, 3},
    {4, 2}
  };

  double total_io = 0;
  for(int i = 0;i < 6; i++){
    total_io += num_calls_middle_link[i]
      *(middle_link_data_io[i][0]*linksize + middle_link_data_io[i][1]*cmsize);
  }

  for(int i = 0;i < 2; i++){
    total_io += num_calls_side_link[i]
      *(side_link_data_io[i][0]*linksize + side_link_data_io[i][1]*cmsize);
  }
  for(int i = 0;i < 2; i++){
    total_io += num_calls_all_link[i]
      *(all_link_data_io[i][0]*linksize + all_link_data_io[i][1]*cmsize);
  }	
  total_io *= V;


  double total_flops = 0;
  for(int i = 0;i < 6; i++){
    total_flops += num_calls_middle_link[i]
      *(middle_link_data_flops[i][0]*matrix_mul_flops + middle_link_data_flops[i][1]*matrix_add_flops);
  }

  for(int i = 0;i < 2; i++){
    total_flops += num_calls_side_link[i]
      *(side_link_data_flops[i][0]*matrix_mul_flops + side_link_data_flops[i][1]*matrix_add_flops);
  }
  for(int i = 0;i < 2; i++){
    total_flops += num_calls_all_link[i]
      *(all_link_data_flops[i][0]*matrix_mul_flops + all_link_data_flops[i][1]*matrix_add_flops);
  }	
  total_flops *= V;

  *io=total_io;
  *flops = total_flops;

  printfQuda("flop/byte =%.1f\n", total_flops/total_io);
  return ;  
}

#ifdef MULTI_GPU
static int R[4] = {2, 2, 2, 2};
#else
static int R[4] = {0, 0, 0, 0};
#endif

// allocate memory
// set the layout, etc.
static void hisq_force_init()
{
  initQuda(device);

  qudaGaugeParam.X[0] = xdim;
  qudaGaugeParam.X[1] = ydim;
  qudaGaugeParam.X[2] = zdim;
  qudaGaugeParam.X[3] = tdim;

  setDims(qudaGaugeParam.X);

  qudaGaugeParam.cpu_prec = link_prec;
  qudaGaugeParam.cuda_prec = link_prec;
  qudaGaugeParam.reconstruct = link_recon;
  qudaGaugeParam.anisotropy = 1.0;

  memcpy(&qudaGaugeParam_ex, &qudaGaugeParam, sizeof(QudaGaugeParam));

  gParam = GaugeFieldParam(0, qudaGaugeParam);
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;

  gParam.order = gauge_order;
  cpuGauge = new cpuGaugeField(gParam);

  gParam_ex = GaugeFieldParam(0, qudaGaugeParam_ex);
  gParam_ex.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
  gParam_ex.create = QUDA_NULL_FIELD_CREATE;
  gParam_ex.link_type = QUDA_GENERAL_LINKS;
  gParam_ex.order = gauge_order;
  for (int d=0; d<4; d++) { gParam_ex.r[d] = R[d]; gParam_ex.x[d] = gParam.x[d] + 2*gParam_ex.r[d]; }  // set halo region for CPU
  cpuGauge_ex = new cpuGaugeField(gParam_ex);

  if (gauge_order == QUDA_QDP_GAUGE_ORDER) {
    createSiteLinkCPU((void**)cpuGauge->Gauge_p(), qudaGaugeParam.cpu_prec, 1);
  } else {
    errorQuda("Unsupported gauge order %d", gauge_order);
  }

  copyExtendedGauge(*cpuGauge_ex, *cpuGauge, QUDA_CPU_FIELD_LOCATION);

  gParam_ex.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam_ex.setPrecision(prec);
  gParam_ex.reconstruct = link_recon;
  gParam_ex.pad = 0;
  gParam_ex.order = QUDA_FLOAT2_GAUGE_ORDER;
  for (int d=0; d<4; d++) { gParam_ex.r[d] = (comm_dim_partitioned(d)) ? 2 : 0; gParam_ex.x[d] = gParam.x[d] + 2*gParam_ex.r[d]; }  // set halo region
  cudaGauge_ex = new cudaGaugeField(gParam_ex);
  qudaGaugeParam.site_ga_pad = gParam_ex.pad;
  //record gauge pad size  

  gParam_ex.pad = 0;
  gParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam_ex.create = QUDA_ZERO_FIELD_CREATE;
  gParam_ex.order = gauge_order;
  for (int d=0; d<4; d++) { gParam_ex.r[d] = R[d]; gParam_ex.x[d] = gParam.x[d] + 2*gParam_ex.r[d]; }  // set halo region for CPU
  cpuForce_ex = new cpuGaugeField(gParam_ex); 


  gParam_ex.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  for (int d=0; d<4; d++) { gParam_ex.r[d] = (comm_dim_partitioned(d)) ? 2 : 0; gParam_ex.x[d] = gParam.x[d] + 2*gParam_ex.r[d]; }  // set halo region
  cudaForce_ex = new cudaGaugeField(gParam_ex); 

  // create the momentum matrix
  gParam.pad = 0;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.order = QUDA_MILC_GAUGE_ORDER;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  cpuMom = new cpuGaugeField(gParam);
  refMom = new cpuGaugeField(gParam);  

  //createMomCPU(cpuMom->Gauge_p(), mom_prec);

  hw = malloc(4 * cpuGauge->Volume() * hw_site_size * qudaGaugeParam.cpu_prec);
  if (hw == NULL){
    fprintf(stderr, "ERROR: malloc failed for hw\n");
    exit(1);
  }

  createHwCPU(hw, hw_prec);

  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.order = gauge_order;
  gParam.pad = 0;
  cpuOprod = new cpuGaugeField(gParam);
  computeLinkOrderedOuterProduct(hw, cpuOprod->Gauge_p(), hw_prec, 1, gauge_order);
  cpuLongLinkOprod = new cpuGaugeField(gParam);
  computeLinkOrderedOuterProduct(hw, cpuLongLinkOprod->Gauge_p(), hw_prec, 3, gauge_order);

  gParam_ex.link_type = QUDA_GENERAL_LINKS;
  gParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam_ex.order = gauge_order;
  for (int d=0; d<4; d++) { gParam_ex.r[d] = R[d]; gParam_ex.x[d] = gParam.x[d] + 2*gParam_ex.r[d]; }  // set halo region for CPU
  cpuOprod_ex = new cpuGaugeField(gParam_ex);
  cpuLongLinkOprod_ex = new cpuGaugeField(gParam_ex);

  copyExtendedGauge(*cpuOprod_ex, *cpuOprod, QUDA_CPU_FIELD_LOCATION);

  copyExtendedGauge(*cpuLongLinkOprod_ex, *cpuLongLinkOprod, QUDA_CPU_FIELD_LOCATION);

  gParam_ex.order = QUDA_FLOAT2_GAUGE_ORDER;
  for (int d=0; d<4; d++) { gParam_ex.r[d] = (comm_dim_partitioned(d)) ? 2 : 0; gParam_ex.x[d] = gParam.x[d] + 2*gParam_ex.r[d]; }  // set halo region
  cudaOprod_ex = new cudaGaugeField(gParam_ex);

}

static void hisq_force_end()
{
  delete cudaMom;
  delete cudaGauge;
  delete cudaForce_ex;
  delete cudaGauge_ex; 
  //delete cudaOprod_ex; // already deleted
  delete cudaLongLinkOprod_ex;

  delete cpuGauge;
  delete cpuMom;
  delete refMom;
  delete cpuOprod;  
  delete cpuLongLinkOprod;

  delete cpuGauge_ex;
  delete cpuForce_ex;
  delete cpuOprod_ex;  
  delete cpuLongLinkOprod_ex;

  free(hw);

  endQuda();
}

static int hisq_force_test(void)
{
  setVerbosity(QUDA_VERBOSE);

  hisq_force_init();

  //float weight = 1.0;
  float act_path_coeff[6];

  act_path_coeff[0] = 0.625000;
  act_path_coeff[1] = -0.058479;
  act_path_coeff[2] = -0.087719;
  act_path_coeff[3] = 0.030778;
  act_path_coeff[4] = -0.007200;
  act_path_coeff[5] = -0.123113;

  //double d_weight = 1.0;
  double d_act_path_coeff[6];
  for(int i=0; i<6; ++i){
    d_act_path_coeff[i] = act_path_coeff[i];
  }

  cpuGauge_ex->exchangeExtendedGhost(R,true);
  cudaGauge_ex->loadCPUField(*cpuGauge);
  cudaGauge_ex->exchangeExtendedGhost(cudaGauge_ex->R());

  cpuOprod_ex->exchangeExtendedGhost(R,true);
  cudaOprod_ex->loadCPUField(*cpuOprod);
  cudaOprod_ex->exchangeExtendedGhost(cudaOprod_ex->R());

  cpuLongLinkOprod_ex->exchangeExtendedGhost(R,true);

  struct timeval ht0, ht1;
  gettimeofday(&ht0, NULL);
  if (verify_results){
    hisqStaplesForceCPU(d_act_path_coeff, qudaGaugeParam, *cpuOprod_ex, *cpuGauge_ex, cpuForce_ex);
    hisqLongLinkForceCPU(d_act_path_coeff[1], qudaGaugeParam, *cpuLongLinkOprod_ex, *cpuGauge_ex, cpuForce_ex);
    hisqCompleteForceCPU(qudaGaugeParam, *cpuForce_ex, *cpuGauge_ex, refMom);
  }
  gettimeofday(&ht1, NULL);

  struct timeval t0, t1, t2, t3;

  gettimeofday(&t0, NULL);

  fermion_force::hisqStaplesForce(*cudaForce_ex, *cudaOprod_ex, *cudaGauge_ex, d_act_path_coeff);
  cudaDeviceSynchronize(); 
  gettimeofday(&t1, NULL);

  delete cudaOprod_ex; //doing this to lower the peak memory usage
  gParam_ex.order = QUDA_FLOAT2_GAUGE_ORDER;
  for (int d=0; d<4; d++) { gParam_ex.r[d] = (comm_dim_partitioned(d)) ? 2 : 0; gParam_ex.x[d] = gParam.x[d] + 2*gParam_ex.r[d]; }  // set halo region
  cudaLongLinkOprod_ex = new cudaGaugeField(gParam_ex);
  cudaLongLinkOprod_ex->loadCPUField(*cpuLongLinkOprod);
  cudaLongLinkOprod_ex->exchangeExtendedGhost(cudaLongLinkOprod_ex->R());
  fermion_force::hisqLongLinkForce(*cudaForce_ex, *cudaLongLinkOprod_ex, *cudaGauge_ex, d_act_path_coeff[1]);
  cudaDeviceSynchronize(); 

  gettimeofday(&t2, NULL);

  gParam.create = QUDA_ZERO_FIELD_CREATE; // initialize to zero
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.pad = 0; //X1*X2*X3/2;
  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  cudaMom = new cudaGaugeField(gParam);

  //record the mom pad
  qudaGaugeParam.mom_ga_pad = gParam.pad;

  fermion_force::hisqCompleteForce(*cudaForce_ex, *cudaGauge_ex);
  updateMomentum(*cudaMom, 1.0, *cudaForce_ex, __func__);

  cudaDeviceSynchronize();
  gettimeofday(&t3, NULL);

  checkCudaError();

  cudaMom->saveCPUField(*cpuMom);

  int accuracy_level = 3;
  if(verify_results){
    int res;
    res = compare_floats(cpuMom->Gauge_p(), refMom->Gauge_p(), 4 * cpuMom->Volume() * mom_site_size, 1e-5,
                         qudaGaugeParam.cpu_prec);

    accuracy_level = strong_check_mom(cpuMom->Gauge_p(), refMom->Gauge_p(), 4*cpuMom->Volume(), qudaGaugeParam.cpu_prec);
    printfQuda("Test %s\n",(1 == res) ? "PASSED" : "FAILED");
  }
  double total_io;
  double total_flops;
  total_staple_io_flops(link_prec, link_recon, &total_io, &total_flops);

  float perf_flops = total_flops / (TDIFF(t0, t1)) *1e-9;
  float perf = total_io / (TDIFF(t0, t1)) *1e-9;
  printfQuda("Staples time: %.2f ms, perf = %.2f GFLOPS, achieved bandwidth= %.2f GB/s\n", TDIFF(t0,t1)*1000, perf_flops, perf);
  printfQuda("Staples time : %g ms\t LongLink time : %g ms\t Completion time : %g ms\n", TDIFF(t0,t1)*1000, TDIFF(t1,t2)*1000, TDIFF(t2,t3)*1000);
  printfQuda("Host time (half-wilson fermion force) : %g ms\n", TDIFF(ht0, ht1)*1000);

  hisq_force_end();

  return accuracy_level;
}


static void display_test_info()
{
  printfQuda("running the following fermion force computation test:\n");

  printfQuda("link_precision           link_reconstruct           space_dim(x/y/z)         T_dimension       Gauge_order\n");
  printfQuda("%s                       %s                         %d/%d/%d                  %d                %s\n", 
      get_prec_str(link_prec),
      get_recon_str(link_recon), 
      xdim, ydim, zdim, tdim,
      get_gauge_order_str(gauge_order));
  return ;

}

int main(int argc, char **argv)
{
  auto app = make_app();
  // app->get_formatter()->column_width(40);
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);

  CLI::TransformPairs<QudaGaugeFieldOrder> gauge_order_map {{"milc", QUDA_MILC_GAUGE_ORDER},
                                                            {"qdp", QUDA_QDP_GAUGE_ORDER}};
  app->add_option("--gauge-order", gauge_order, "")->transform(CLI::QUDACheckedTransformer(gauge_order_map));

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if(gauge_order == QUDA_MILC_GAUGE_ORDER){
    errorQuda("Multi-gpu for milc order is not supported\n");
  }

  initComms(argc, argv, gridsize_from_cmdline);

  setPrecision(prec);

  display_test_info();

  int accuracy_level = hisq_force_test();

  finalizeComms();

  if(accuracy_level >=3 ){
    return EXIT_SUCCESS;
  }else{
    return -1;
  }
}
