#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <quda.h>
#include "test_util.h"
#include "gauge_field.h"
#include "fat_force_quda.h"
#include "misc.h"
#include "hisq_force_reference.h"
#include "hisq_force_quda.h"
#include "hisq_force_utils.h"
#include "hw_quda.h"
#include <fat_force_quda.h>
#include <face_quda.h>

#include <sys/time.h>

#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

#define GPU_DIRECT

#include "fermion_force_reference.h"
using namespace hisq::fermion_force;

extern void usage(char** argv);
extern int device;
cudaGaugeField *cudaGauge = NULL;
cpuGaugeField  *cpuGauge  = NULL;

cudaGaugeField *cudaForce = NULL;
cpuGaugeField  *cpuForce = NULL;

cudaGaugeField *cudaMom = NULL;
cpuGaugeField *cpuMom  = NULL;
cpuGaugeField *refMom  = NULL;

static FullHw cudaHw;
static QudaGaugeParam qudaGaugeParam;
static QudaGaugeParam qudaGaugeParam_ex;
static void* hw; // the array of half_wilson_vector

QudaGaugeFieldOrder gauge_order = QUDA_QDP_GAUGE_ORDER;

cpuGaugeField *cpuOprod = NULL;
cudaGaugeField *cudaOprod = NULL;
cpuGaugeField *cpuLongLinkOprod = NULL;
cudaGaugeField *cudaLongLinkOprod = NULL;

int verify_results = 0;
int ODD_BIT = 1;
extern int xdim, ydim, zdim, tdim;
extern int gridsize_from_cmdline[];


extern QudaPrecision prec;
extern QudaReconstructType link_recon;
QudaPrecision link_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision hw_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cpu_hw_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision mom_prec = QUDA_DOUBLE_PRECISION;

void* siteLink_1d;
void* siteLink_2d[4];
void* siteLink_ex_2d[4];

cudaGaugeField *cudaGauge_ex = NULL;
cpuGaugeField  *cpuGauge_ex  = NULL;
cudaGaugeField *cudaForce_ex = NULL;
cpuGaugeField  *cpuForce_ex = NULL;
cpuGaugeField *cpuOprod_ex = NULL;
cudaGaugeField *cudaOprod_ex = NULL;
cpuGaugeField *cpuLongLinkOprod_ex = NULL;
cudaGaugeField *cudaLongLinkOprod_ex = NULL;



static void setPrecision(QudaPrecision precision)
{
  link_prec = precision;
  hw_prec = precision;
  cpu_hw_prec = precision;
  mom_prec = precision;
  return;
}

int Z[4];
int V;
int Vh;
int V_ex;
int Vh_ex;

static int X1, X1h, X2, X3, X4;
static int E1, E1h, E2, E3, E4;
int E[4];


void
setDims(int *X)
{
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];
  }
  Vh = V/2;

  V_ex = 1;
  for (int d=0; d< 4; d++) {
    V_ex *= X[d]+4;
  }
  Vh_ex = V_ex/2;

  X1=X[0]; X2 = X[1]; X3=X[2]; X4=X[3];
  X1h=X1/2;
  E1=X1+4; E2=X2+4; E3=X3+4; E4=X4+4;
  E1h=E1/2;

  E[0] = E1;
  E[1] = E2;
  E[2] = E3;
  E[3] = E4;

}


void
total_staple_io_flops(QudaPrecision prec, QudaReconstructType recon, double* io, double* flops)
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


void initDslashConstants(const cudaGaugeField &gauge, const int sp_stride);


// allocate memory
// set the layout, etc.
static void
hisq_force_init()
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

  //  qudaGaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;
  qudaGaugeParam.gauge_order = gauge_order;
  qudaGaugeParam.anisotropy = 1.0;

  
  memcpy(&qudaGaugeParam_ex, &qudaGaugeParam, sizeof(QudaGaugeParam));
  qudaGaugeParam_ex.X[0] = qudaGaugeParam.X[0] + 4;
  qudaGaugeParam_ex.X[1] = qudaGaugeParam.X[1] + 4;
  qudaGaugeParam_ex.X[2] = qudaGaugeParam.X[2] + 4;
  qudaGaugeParam_ex.X[3] = qudaGaugeParam.X[3] + 4;


  
  GaugeFieldParam gParam(0, qudaGaugeParam);
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.link_type = QUDA_ASQTAD_GENERAL_LINKS;
  cpuGauge = new cpuGaugeField(gParam);
  
#ifdef MULTI_GPU
  GaugeFieldParam gParam_ex(0, qudaGaugeParam_ex);
  gParam_ex.create = QUDA_NULL_FIELD_CREATE;
  gParam_ex.link_type = QUDA_ASQTAD_GENERAL_LINKS;
  cpuGauge_ex = new cpuGaugeField(gParam_ex);
#endif

  int gSize = qudaGaugeParam.cpu_prec;
  // this is a hack to get the gauge field to appear as a void** rather than void*
  for(int i=0;i < 4;i++){
#ifdef GPU_DIRECT
    cudaMallocHost(&siteLink_2d[i], V*gaugeSiteSize* qudaGaugeParam.cpu_prec);
    cudaMallocHost((void**)&siteLink_ex_2d[i], V_ex*gaugeSiteSize*qudaGaugeParam.cpu_prec);
#else
    siteLink_2d[i] = malloc(V*gaugeSiteSize* qudaGaugeParam.cpu_prec);
    siteLink_ex_2d[i] = malloc(V_ex*gaugeSiteSize*qudaGaugeParam.cpu_prec);
#endif
    if(siteLink_2d[i] == NULL || siteLink_ex_2d[i] == NULL){
      errorQuda("malloc failed for siteLink_2d/siteLink_ex_2d\n");
    }
    memset(siteLink_2d[i], 0, V*gaugeSiteSize* qudaGaugeParam.cpu_prec);
    memset(siteLink_ex_2d[i], 0, V_ex*gaugeSiteSize*qudaGaugeParam.cpu_prec);
  }
  //siteLink_1d is only used in fermion reference computation
  siteLink_1d = malloc(4*V*gaugeSiteSize* qudaGaugeParam.cpu_prec);
  
  
  // fills the gauge field with random numbers
  createSiteLinkCPU(siteLink_2d, qudaGaugeParam.cpu_prec, 1);
  for(int i=0; i < V_ex; i++){
    int sid = i;
    int oddBit=0;
    if(i >= Vh_ex){
      sid = i - Vh_ex;
      oddBit = 1;
    }

    int za = sid/E1h;
    int x1h = sid - za*E1h;
    int zb = za/E2;
    int x2 = za - zb*E2;
    int x4 = zb/E3;
    int x3 = zb - x4*E3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;
    
    
    if( x1< 2 || x1 >= X1 +2
        || x2< 2 || x2 >= X2 +2
        || x3< 2 || x3 >= X3 +2
        || x4< 2 || x4 >= X4 +2){
      continue;
    }
    
    
    
    x1 = (x1 - 2 + X1) % X1;
    x2 = (x2 - 2 + X2) % X2;
    x3 = (x3 - 2 + X3) % X3;
    x4 = (x4 - 2 + X4) % X4;
    
    int idx = (x4*X3*X2*X1+x3*X2*X1+x2*X1+x1)>>1;
    if(oddBit){
      idx += Vh;
    }
    for(int dir= 0; dir < 4; dir++){
      char* src = (char*)siteLink_2d[dir];
      char* dst = (char*)siteLink_ex_2d[dir];
      memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
    }//dir

    /*
    if(x1 == 0 && x2 == 0&& x3 == 0&& x4 == 0 && i == 1885)
    {
      float* data = ((float*)siteLink_ex_2d[0]) + i*18;
      printf("cpu matrix\n");
      for(int j=0; j<3; j++){
	printf("(%f %f) (%f %f) (%f %f)\n", data[0], data[1], data[2], data[3], data[4], data[5]);
	data += 6;
      }
      
    }
    */


  }//i

  


  for(int dir = 0; dir < 4; dir++){
    for(int i = 0;i < V; i++){
      char* src = (char*)siteLink_2d[dir];
      char* dst = (char*)siteLink_1d;
      memcpy(dst + (4*i+dir)*gaugeSiteSize*link_prec, src + i*gaugeSiteSize*link_prec, gaugeSiteSize \
	     *link_prec);
    }
  }

  if(gauge_order == QUDA_MILC_GAUGE_ORDER){
    for(int dir = 0; dir < 4; dir++){
      for(int i = 0;i < V; i++){
	char* src = (char*)siteLink_2d[dir];
	char* dst = (char*)cpuGauge->Gauge_p();
	memcpy(dst + (4*i+dir)*gaugeSiteSize*link_prec, src + i*gaugeSiteSize*link_prec, gaugeSiteSize*link_prec);   
      }
    }
  }else{
    for(int dir=0;dir < 4; dir++){
      char* src = (char*)siteLink_2d[dir];
      char* dst = ((char**)cpuGauge->Gauge_p())[dir];
      memcpy(dst, src, V*gaugeSiteSize*link_prec);
    }
  }
#ifdef MULTI_GPU
  //for multi-gpu we have to use qdp format now
  if(gauge_order == QUDA_MILC_GAUGE_ORDER){
    errorQuda("multi_gpu milc is not supported\n");    
  }
  for(int dir=0;dir < 4; dir++){
    char* src = (char*)siteLink_ex_2d[dir];
    char* dst = ((char**)cpuGauge_ex->Gauge_p())[dir];
    memcpy(dst, src, V_ex*gaugeSiteSize*link_prec);
  }  
  
#endif

  gParam.precision = qudaGaugeParam.cuda_prec;
  gParam.reconstruct = link_recon;
  cudaGauge = new cudaGaugeField(gParam);

#ifdef MULTI_GPU
  gParam_ex.precision = prec;
  gParam_ex.reconstruct = link_recon;
  gParam_ex.pad = 0;
  cudaGauge_ex = new cudaGaugeField(gParam_ex);
#endif

#ifdef MULTI_GPU
  gParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam_ex.create = QUDA_ZERO_FIELD_CREATE;
  cpuForce_ex = new cpuGaugeField(gParam_ex); 
  
  gParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaForce_ex = new cudaGaugeField(gParam_ex); 
#endif
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  cpuForce = new cpuGaugeField(gParam); 
  
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaForce = new cudaGaugeField(gParam); 


  // create the momentum matrix
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.order = QUDA_MILC_GAUGE_ORDER;
  cpuMom = new cpuGaugeField(gParam);
  refMom = new cpuGaugeField(gParam);  
  
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  cudaMom = new cudaGaugeField(gParam); // Are the elements initialised to zero? - No!

  createMomCPU(cpuMom->Gauge_p(), mom_prec);
  hw = malloc(4*cpuGauge->Volume()*hwSiteSize*qudaGaugeParam.cpu_prec);
  if (hw == NULL){
    fprintf(stderr, "ERROR: malloc failed for hw\n");
    exit(1);
  }

  createHwCPU(hw, hw_prec);
  cudaHw = createHwQuda(qudaGaugeParam.X, hw_prec);


  gParam.link_type = QUDA_ASQTAD_GENERAL_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.order = gauge_order;
  cpuOprod = new cpuGaugeField(gParam);
  computeLinkOrderedOuterProduct(hw, cpuOprod->Gauge_p(), hw_prec, 1, gauge_order);
  cpuLongLinkOprod = new cpuGaugeField(gParam);
  computeLinkOrderedOuterProduct(hw, cpuLongLinkOprod->Gauge_p(), hw_prec, 3, gauge_order);

#ifdef MULTI_GPU
  gParam_ex.link_type = QUDA_ASQTAD_GENERAL_LINKS;
  gParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam_ex.order = gauge_order;
  cpuOprod_ex = new cpuGaugeField(gParam_ex);
  //computeLinkOrderedOuterProduct(hw, cpuOprod_ex->Gauge_p(), hw_prec, 1, gauge_order);
  
  cpuLongLinkOprod_ex = new cpuGaugeField(gParam_ex);
  //computeLinkOrderedOuterProduct(hw, cpuLongLinkOprod_ex->Gauge_p(), hw_prec, 3, gauge_order);
  
  for(int i=0; i < V_ex; i++){
    int sid = i;
    int oddBit=0;
    if(i >= Vh_ex){
      sid = i - Vh_ex;
      oddBit = 1;
    }
    
    int za = sid/E1h;
    int x1h = sid - za*E1h;
    int zb = za/E2;
    int x2 = za - zb*E2;
    int x4 = zb/E3;
    int x3 = zb - x4*E3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;
    
    
    if( x1< 2 || x1 >= X1 +2
        || x2< 2 || x2 >= X2 +2
        || x3< 2 || x3 >= X3 +2
        || x4< 2 || x4 >= X4 +2){
      continue;
    }
    
    
    
    x1 = (x1 - 2 + X1) % X1;
    x2 = (x2 - 2 + X2) % X2;
    x3 = (x3 - 2 + X3) % X3;
    x4 = (x4 - 2 + X4) % X4;
    
    int idx = (x4*X3*X2*X1+x3*X2*X1+x2*X1+x1)>>1;
    if(oddBit){
      idx += Vh;
    }
    for(int dir= 0; dir < 4; dir++){
      char* src = ((char**)cpuOprod->Gauge_p())[dir];
      char* dst = ((char**)cpuOprod_ex->Gauge_p())[dir];
      memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);

      src = ((char**)cpuLongLinkOprod->Gauge_p())[dir];
      dst = ((char**)cpuLongLinkOprod_ex->Gauge_p())[dir];
      memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
      
    }//dir
  }//i



  cudaOprod_ex = new cudaGaugeField(gParam_ex);
  cudaLongLinkOprod_ex = new cudaGaugeField(gParam_ex);
#endif



  cudaOprod = new cudaGaugeField(gParam);
  cudaLongLinkOprod = new cudaGaugeField(gParam);


  return;
}


static void 
hisq_force_end()
{
  for(int i = 0;i < 4; i++){
#ifdef GPU_DIRECT
    cudaFreeHost(siteLink_2d[i]);
    cudaFreeHost(siteLink_ex_2d[i]);
#else
    free(siteLink_2d[i]);
    free(siteLink_ex_2d[i]);
#endif
  }
  free(siteLink_1d);

  delete cudaMom;
  delete cudaForce;
  delete cudaGauge;
  delete cudaOprod;
  delete cudaLongLinkOprod;
  freeHwQuda(cudaHw);
  
  delete cpuGauge;
  delete cpuForce;
  delete cpuMom;
  delete refMom;
  delete cpuOprod;  
  delete cpuLongLinkOprod;
  free(hw);

  endQuda();

  return;
}

static int 
hisq_force_test(void)
{
  hisq_force_init();

  initDslashConstants(*cudaGauge, cudaGauge->VolumeCB());
  hisqForceInitCuda(&qudaGaugeParam);


   
  float weight = 1.0;
  float act_path_coeff[6];

  act_path_coeff[0] = 0.625000;
  act_path_coeff[1] = -0.058479;
  act_path_coeff[2] = -0.087719;
  act_path_coeff[3] = 0.030778;
  act_path_coeff[4] = -0.007200;
  act_path_coeff[5] = -0.123113;


  double d_weight = 1.0;
  double d_act_path_coeff[6];
  for(int i=0; i<6; ++i){
    d_act_path_coeff[i] = act_path_coeff[i];
  }
  
  
  cudaMom->loadCPUField(*refMom, QUDA_CPU_FIELD_LOCATION);


#ifdef MULTI_GPU
  int optflag = 0;
  exchange_cpu_sitelink_ex(qudaGaugeParam.X, (void**)cpuGauge_ex->Gauge_p(), cpuGauge->Order(), qudaGaugeParam.cpu_prec, optflag);
  loadLinkToGPU_ex(cudaGauge_ex, cpuGauge_ex);  
#else
  loadLinkToGPU(cudaGauge, cpuGauge, &qudaGaugeParam);  
#endif


#ifdef MULTI_GPU
  exchange_cpu_sitelink_ex(qudaGaugeParam.X, (void**)cpuOprod_ex->Gauge_p(), cpuOprod_ex->Order(), qudaGaugeParam.cpu_prec, optflag);
  loadLinkToGPU_ex(cudaOprod_ex, cpuOprod_ex); 
#else
  loadLinkToGPU(cudaOprod, cpuOprod, &qudaGaugeParam);
#endif
  
  
#ifdef MULTI_GPU
  exchange_cpu_sitelink_ex(qudaGaugeParam.X, (void**)cpuLongLinkOprod_ex->Gauge_p(), cpuLongLinkOprod_ex->Order(), qudaGaugeParam.cpu_prec, optflag);
  loadLinkToGPU_ex(cudaLongLinkOprod_ex, cpuLongLinkOprod_ex);
#else
  loadLinkToGPU(cudaLongLinkOprod, cpuLongLinkOprod, &qudaGaugeParam);
#endif

#ifdef MULTI_GPU  
#else
  loadHwToGPU(cudaHw, hw, cpu_hw_prec);
#endif

  
  struct timeval ht0, ht1;
  gettimeofday(&ht0, NULL);
  if (verify_results){
#if 0    
    if(cpu_hw_prec == QUDA_SINGLE_PRECISION){
      const float eps = 0.5;
      fermion_force_reference(eps, weight, 0, act_path_coeff, hw, siteLink_1d, refMom->Gauge_p());
    }else if(cpu_hw_prec == QUDA_DOUBLE_PRECISION){
      const double eps = 0.5;
      fermion_force_reference(eps, d_weight, 0, d_act_path_coeff, hw, siteLink_1d, refMom->Gauge_p());
    }
#else
    
    void* coeff;
    void* naik_coeff;
    if(cpu_hw_prec == QUDA_SINGLE_PRECISION){
      coeff = act_path_coeff;
      naik_coeff = &act_path_coeff[1];
    }else{
      coeff = d_act_path_coeff;
      naik_coeff = &d_act_path_coeff[1];
    }
    
#define TEST_ONLY
#ifdef TEST_ONLY
    printfQuda("Testing only .................................................\n");
#ifdef MULTI_GPU
    hisqLongLinkForceCPU(d_act_path_coeff[1], qudaGaugeParam, *cpuLongLinkOprod_ex, *cpuGauge_ex, cpuForce_ex);
    hisqCompleteForceCPU(qudaGaugeParam, *cpuForce_ex, *cpuGauge_ex, refMom);
    //hisqCompleteForceCPU(qudaGaugeParam, *cpuOprod_ex, *cpuGauge_ex, refMom);
#else
    hisqLongLinkForceCPU(d_act_path_coeff[1], qudaGaugeParam, *cpuLongLinkOprod, *cpuGauge, cpuForce);
    hisqCompleteForceCPU(qudaGaugeParam, *cpuForce, *cpuGauge, refMom);
#endif

#else
    printfQuda("Not Testing only .................................................\n");
    hisqStaplesForceCPU(d_act_path_coeff, qudaGaugeParam, *cpuOprod, *cpuGauge, cpuForce);
    hisqLongLinkForceCPU(d_act_path_coeff[1], qudaGaugeParam, *cpuLongLinkOprod, *cpuGauge, cpuForce);
    hisqCompleteForceCPU(qudaGaugeParam, *cpuForce, *cpuGauge, refMom);
#endif //TEST_ONLY
#endif //0 or 1
  }
  gettimeofday(&ht1, NULL);

  struct timeval t0, t1, t2, t3, t4, t5;
#ifdef TEST_ONLY

#ifdef MULTI_GPU
  hisqLongLinkForceCuda(d_act_path_coeff[1], qudaGaugeParam, *cudaLongLinkOprod_ex, *cudaGauge_ex, cudaForce_ex);
  hisqCompleteForceCuda(qudaGaugeParam, *cudaForce_ex, *cudaGauge_ex, cudaMom);
  //hisqCompleteForceCuda(qudaGaugeParam, *cudaOprod_ex, *cudaGauge_ex, cudaMom);
#else
  hisqLongLinkForceCuda(d_act_path_coeff[1], qudaGaugeParam, *cudaLongLinkOprod, *cudaGauge, cudaForce);
  hisqCompleteForceCuda(qudaGaugeParam, *cudaForce, *cudaGauge, cudaMom);
#endif

#else
#ifdef MULTI_GPU
  hisqStaplesForceCuda(d_act_path_coeff, qudaGaugeParam, *cudaOprod_ex, *cudaGauge_ex, cudaForce_ex);
  hisqLongLinkForceCuda(d_act_path_coeff[1], qudaGaugeParam, *cudaLongLinkOprod_ex, *cudaGauge_ex, cudaForce_ex);
  hisqCompleteForceCuda(qudaGaugeParam, *cudaForce_ex, *cudaGauge_ex, cudaMom);
#else
  hisqStaplesForceCuda(d_act_path_coeff, qudaGaugeParam, *cudaOprod, *cudaGauge, cudaForce);
  hisqLongLinkForceCuda(d_act_path_coeff[1], qudaGaugeParam, *cudaLongLinkOprod, *cudaGauge, cudaForce);
  hisqCompleteForceCuda(qudaGaugeParam, *cudaForce, *cudaGauge, cudaMom);
#endif
#endif

  cudaThreadSynchronize();
  checkCudaError();



  cudaMom->saveCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);

  int res;
  res = compare_floats(cpuMom->Gauge_p(), refMom->Gauge_p(), 4*cpuMom->Volume()*momSiteSize, 1e-5, qudaGaugeParam.cpu_prec);

  int accuracy_level = strong_check_mom(cpuMom->Gauge_p(), refMom->Gauge_p(), 4*cpuMom->Volume(), qudaGaugeParam.cpu_prec);
  printfQuda("Test %s\n",(1 == res) ? "PASSED" : "FAILED");

  double total_io;
  double total_flops;
  total_staple_io_flops(link_prec, link_recon, &total_io, &total_flops);
  
  float perf_flops = total_flops / (TDIFF(t0, t1)) *1e-9;
  float perf = total_io / (TDIFF(t0, t1)) *1e-9;
  printfQuda("Staples time: %.2f ms, perf =%.2f GFLOPS, achieved bandwidth= %.2f GB/s\n", TDIFF(t0,t1)*1000, perf_flops, perf);
  printfQuda("Staples time : %g ms\t LongLink time : %g ms\t Completion time : %g ms\n", TDIFF(t0,t1)*1000, TDIFF(t2,t3)*1000, TDIFF(t4,t5)*1000);
  printfQuda("Host time (half-wilson fermion force) : %g ms\n", TDIFF(ht0, ht1)*1000);

  hisq_force_end();

  return accuracy_level;
}


static void
display_test_info()
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

void
usage_extra(char** argv )
{
  printfQuda("Extra options: \n");
  printfQuda("    --verify                                  # Verify the GPU results using CPU results\n");
  return ;
}
int 
main(int argc, char **argv) 
{
  int i;
  for (i =1;i < argc; i++){
	
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }    

    if( strcmp(argv[i], "--gauge-order") == 0){
      if(i+1 >= argc){
        usage(argv);
      }

      if(strcmp(argv[i+1], "milc") == 0){
        gauge_order = QUDA_MILC_GAUGE_ORDER;
      }else if(strcmp(argv[i+1], "qdp") == 0){
        gauge_order = QUDA_QDP_GAUGE_ORDER;
      }else{
        fprintf(stderr, "Error: unsupported gauge-field order\n");
        exit(1);
      }
      i++;
      continue;
    }

    if( strcmp(argv[i], "--verify") == 0){
      verify_results=1;
      continue;	    
    }	
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

#ifdef MULTI_GPU
  if(gauge_order == QUDA_MILC_GAUGE_ORDER){
    errorQuda("Multi-gpu for milc order is not supported\n");
  }

    initCommsQuda(argc, argv, gridsize_from_cmdline, 4);
#endif

  setPrecision(prec);

  display_test_info();
    
  int accuracy_level = hisq_force_test();


#ifdef MULTI_GPU
  endCommsQuda();
#endif

  if(accuracy_level >=3 ){
    return EXIT_SUCCESS;
  }else{
    return -1;
  }
  
}


