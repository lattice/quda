#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>

#include <quda_internal.h>

#include "quda.h"
#include "dslash_quda.h"
#include "llfat_quda.h"

#include "test_util.h"
#include "llfat_reference.h"
#include "misc.h"
#include "gauge_field.h"
#include "fat_force_quda.h"

#ifdef MULTI_GPU
#include "face_quda.h"
#include "mpicomm.h"
#include <mpi.h>
#endif

#define MAX(a,b) ((a)>(b)? (a):(b))

static cudaGaugeField *cudaSiteLink, *cudaSiteLink_ex;
static cudaGaugeField *cudaFatLink;
static cudaGaugeField* cudaStapleField, *cudaStapleField1;
static cudaGaugeField* cudaStapleField_ex, *cudaStapleField1_ex;

static QudaGaugeParam qudaGaugeParam;
static QudaGaugeParam qudaGaugeParam_ex;
static cpuGaugeField *fatlink, *reflink;
static cpuGaugeField *sitelink, *sitelink_ex;

#ifdef MULTI_GPU
static void* ghost_sitelink[4];
static void* ghost_sitelink_diag[16];
#endif

static int verify_results = 0;


#define DIM 24

extern int device;
int Z[4];
int V;
int Vh;
int Vs[4];
int Vsh[4];
static int Vs_x, Vs_y, Vs_z, Vs_t;
static int Vsh_x, Vsh_y, Vsh_z, Vsh_t;

static int V_ex;
static int Vh_ex;

static int X1, X1h, X2, X3, X4;
static int E1, E1h, E2, E3, E4;


extern int xdim, ydim, zdim, tdim;
extern int gridsize_from_cmdline[];

extern QudaReconstructType link_recon;
extern QudaPrecision  prec;
static QudaPrecision  cpu_prec = QUDA_DOUBLE_PRECISION;
static size_t gSize;

void
setDims(int *X) {
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];
  }
  Vh = V/2;

  Vs[0] = Vs_x = X[1]*X[2]*X[3];
  Vs[1] = Vs_y = X[0]*X[2]*X[3];
  Vs[2] = Vs_z = X[0]*X[1]*X[3];
  Vs[3] = Vs_t = X[0]*X[1]*X[2];

  Vsh[0] = Vsh_x = Vs_x/2;
  Vsh[1] = Vsh_y = Vs_y/2;
  Vsh[2] = Vsh_z = Vs_z/2;
  Vsh[3] = Vsh_t = Vs_t/2;

  V_ex = 1;
  for (int d=0; d< 4; d++) {
    V_ex *= X[d]+4;
  }
  Vh_ex = V_ex/2; 

  X1=X[0]; X2 = X[1]; X3=X[2]; X4=X[3];
  X1h=X1/2;
  E1=X1+4; E2=X2+4; E3=X3+4; E4=X4+4;
  E1h=E1/2;
  
  
}

static void
llfat_init(int test)
{ 
  initQuda(device);

  gSize = cpu_prec;
  
  qudaGaugeParam = newQudaGaugeParam();
  qudaGaugeParam_ex = newQudaGaugeParam();
  
  qudaGaugeParam.anisotropy = 1.0;

  qudaGaugeParam.X[0] = xdim;
  qudaGaugeParam.X[1] = ydim;
  qudaGaugeParam.X[2] = zdim;
  qudaGaugeParam.X[3] = tdim;

  setDims(qudaGaugeParam.X);
    
  qudaGaugeParam.cpu_prec = cpu_prec;
  qudaGaugeParam.cuda_prec = prec;
  qudaGaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  qudaGaugeParam.type=QUDA_WILSON_LINKS;

  GaugeFieldParam gParam(0, qudaGaugeParam);
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.order = QUDA_MILC_GAUGE_ORDER;
  gParam.pinned = 1;
  gParam.link_type = QUDA_ASQTAD_FAT_LINKS;
  fatlink = new cpuGaugeField(gParam);
  if(fatlink == NULL){
    printfQuda("ERROR: Creating fatlink failed\n");
  }
   
  gParam.order = QUDA_QDP_GAUGE_ORDER;
  gParam.pinned = 1;
  gParam.link_type = QUDA_WILSON_LINKS;
  sitelink = new cpuGaugeField(gParam);
  if(sitelink == NULL){
    errorQuda("ERROR: Creating sitelink failed\n");
  }
  
  //reset pinned
  gParam.pinned = 0;

  memcpy(&qudaGaugeParam_ex, &qudaGaugeParam, sizeof(QudaGaugeParam)); 
  qudaGaugeParam_ex.X[0] = xdim+4;
  qudaGaugeParam_ex.X[1] = ydim+4;
  qudaGaugeParam_ex.X[2] = zdim+4;
  qudaGaugeParam_ex.X[3] = tdim+4;
  GaugeFieldParam gParam_ex(0, qudaGaugeParam_ex);
  gParam_ex.create = QUDA_NULL_FIELD_CREATE;
  gParam_ex.order = QUDA_QDP_GAUGE_ORDER;
  gParam_ex.pinned = 0;
  sitelink_ex = new cpuGaugeField(gParam_ex);
  if(sitelink_ex == NULL){
    errorQuda("ERROR: Creating sitelink_ex failed\n");
  }
  
#ifdef MULTI_GPU
  //we need x,y,z site links in the back and forward T slice
  // so it is 3*2*Vs_t
  for(int i=0;i < 4; i++){
    ghost_sitelink[i] = malloc(8*Vs[i]*gaugeSiteSize*gSize);
    if (ghost_sitelink[i] == NULL){
      printf("ERROR: malloc failed for ghost_sitelink[%d] \n",i);
      exit(1);
    }
  }

  /*
    nu |     |
       |_____|
          mu     
  */
  
  for(int nu=0;nu < 4;nu++){
    for(int mu=0; mu < 4;mu++){
      if(nu == mu){
	ghost_sitelink_diag[nu*4+mu] = NULL;
      }else{
	//the other directions
	int dir1, dir2;
	for(dir1= 0; dir1 < 4; dir1++){
	  if(dir1 !=nu && dir1 != mu){
	    break;
	  }
	}
	for(dir2=0; dir2 < 4; dir2++){
	  if(dir2 != nu && dir2 != mu && dir2 != dir1){
	    break;
	  }
	}
	ghost_sitelink_diag[nu*4+mu] = malloc(Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
	if(ghost_sitelink_diag[nu*4+mu] == NULL){
	  errorQuda("malloc failed for ghost_sitelink_diag\n");
	}
	
	memset(ghost_sitelink_diag[nu*4+mu], 0, Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
      }

    }
  }

#endif

  
  reflink = new cpuGaugeField(gParam);
  if(reflink == NULL){
    printfQuda("ERROR: Creating reflink failed\n");
  }
  
  createSiteLinkCPU((void**)sitelink->Gauge_p(), qudaGaugeParam.cpu_prec, 1);


  //FIXME:
  //assuming all dimension size is even
  //fill in the extended sitelink 
  int i;
  for(i=0; i < V_ex; i++){
    
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

      /*
      if( x1< 2 || x1 >= X1 +2
          || x2< 2 || x2 >= X2 +2
          || x3< 2 || x3 >= X3 +2
          || x4< 2 || x4 >= X4 +2){
        continue;
      }
      */


      x1 = (x1 - 2 + X1) % X1;
      x2 = (x2 - 2 + X2) % X2;
      x3 = (x3 - 2 + X3) % X3;
      x4 = (x4 - 2 + X4) % X4;

      int idx = (x4*X3*X2*X1+x3*X2*X1+x2*X1+x1)>>1;
      if(oddBit){
        idx += Vh;
      }
      for(int dir= 0; dir < 4; dir++){
        char* src = ((char**)sitelink->Gauge_p())[dir];
        char* dst = ((char**)sitelink_ex->Gauge_p())[dir];
        memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
      }//dir
  }//i
  
  
  qudaGaugeParam.llfat_ga_pad = gParam.pad = Vsh_t;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.link_type = QUDA_ASQTAD_FAT_LINKS;
  cudaFatLink = new cudaGaugeField(gParam);

  switch(test){
  case 0:
    {
#ifdef MULTI_GPU
      int Vh_2d_max = MAX(xdim*ydim/2, xdim*zdim/2);
      Vh_2d_max = MAX(Vh_2d_max, xdim*tdim/2);
      Vh_2d_max = MAX(Vh_2d_max, ydim*zdim/2);  
      Vh_2d_max = MAX(Vh_2d_max, ydim*tdim/2);  
      Vh_2d_max = MAX(Vh_2d_max, zdim*tdim/2);  
      
      qudaGaugeParam.site_ga_pad = gParam.pad = 3*(Vsh_x+Vsh_y+Vsh_z+Vsh_t) + 4*Vh_2d_max;
      gParam.reconstruct = link_recon;
      gParam.link_type = QUDA_WILSON_LINKS;
      cudaSiteLink = new cudaGaugeField(gParam);  
 

      GaugeFieldParam gStapleParam(0, qudaGaugeParam);
      gStapleParam.create = QUDA_NULL_FIELD_CREATE;  
      gStapleParam.reconstruct = QUDA_RECONSTRUCT_NO;
      gStapleParam.is_staple = 1; //these two condition means it is a staple instead of a normal gauge field
      gStapleParam.pad = 3*(Vsh_x + Vsh_y + Vsh_z+ Vsh_t);
      cudaStapleField = new cudaGaugeField(gStapleParam);
      cudaStapleField1 = new cudaGaugeField(gStapleParam);
      

      qudaGaugeParam.staple_pad = 3*(Vsh_x + Vsh_y + Vsh_z+ Vsh_t);

#else
      qudaGaugeParam.site_ga_pad = gParam.pad = Vsh_t;
      gParam.reconstruct = link_recon;
      cudaSiteLink = new cudaGaugeField(gParam);
      
      GaugeFieldParam gStapleParam(0, qudaGaugeParam);
      gStapleParam.create = QUDA_NULL_FIELD_CREATE;  
      gStapleParam.reconstruct = QUDA_RECONSTRUCT_NO;
      gStapleParam.is_staple = 1; //these two condition means it is a staple instead of a normal gauge field
      gStapleParam.pad = 3*Vsh_t;
      cudaStapleField = new cudaGaugeField(gStapleParam);
      cudaStapleField1 = new cudaGaugeField(gStapleParam);
      
      qudaGaugeParam.staple_pad = Vsh_t;

#endif
      break;
    }      
  case 1:
    {
      qudaGaugeParam_ex.site_ga_pad = gParam_ex.pad = E1*E2*E3/2*3;
      gParam_ex.reconstruct = link_recon;
      //createLinkQuda(&cudaSiteLink_ex, &qudaGaugeParam_ex);
      cudaSiteLink_ex = new cudaGaugeField(gParam_ex);
      
      GaugeFieldParam gStapleParam_ex(0, qudaGaugeParam_ex);
      gStapleParam_ex.create = QUDA_NULL_FIELD_CREATE;  
      gStapleParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
      gStapleParam_ex.is_staple = 1; //these two condition means it is a staple instead of a normal gauge field
      gStapleParam_ex.pad = 3*(Vsh_x + Vsh_y + Vsh_z+ Vsh_t);
      cudaStapleField_ex = new cudaGaugeField(gStapleParam_ex);
      cudaStapleField1_ex = new cudaGaugeField(gStapleParam_ex);
      

      qudaGaugeParam_ex.staple_pad =  E1*E2*E2/2*3;


      //set llfat_ga_gad in qudaGaugeParam.ex as well
      qudaGaugeParam_ex.llfat_ga_pad = qudaGaugeParam.llfat_ga_pad;
      break;
      
    }
  default:
    errorQuda("Test type (%d) not supported\n", test);
  }

  initDslashConstants(*cudaFatLink, 0);
  
  return;
}

void 
llfat_end(int test)  
{   

  delete fatlink;
  delete sitelink;
  delete sitelink_ex;
  
#ifdef MULTI_GPU  
  int i;
  for(i=0;i < 4;i++){
    free(ghost_sitelink[i]);
  }
  for(i=0;i <4; i++){
    for(int j=0;j <4; j++){
      if (i==j){
	continue;
      }
      free(ghost_sitelink_diag[i*4+j]);
    }    
  }
#endif

  delete reflink;
  
  switch(test){
  case 0:
    delete cudaSiteLink;
    delete cudaStapleField;
    delete cudaStapleField1;

  case 1:
    delete cudaSiteLink_ex;
    delete cudaStapleField_ex;
    delete cudaStapleField1_ex;
    break;
  default:
    errorQuda("Error: invalid test type(%d)\n", test);
  }

  delete cudaFatLink;
  
#ifdef MULTI_GPU
  exchange_llfat_cleanup();
#endif
  
  endQuda();
}



static int
llfat_test(int test) 
{
  llfat_init(test);


  float act_path_coeff_1[6];
  double act_path_coeff_2[6];
  
  for(int i=0;i < 6;i++){
    act_path_coeff_1[i]= 0.1*i;
    act_path_coeff_2[i]= 0.1*i;
  }
  

 

  void* act_path_coeff;    
  if(qudaGaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION){
    act_path_coeff = act_path_coeff_2;
  }else{
    act_path_coeff = act_path_coeff_1;	
  }
  if (verify_results){

#ifdef MULTI_GPU
    int optflag = 0;
    exchange_cpu_sitelink(qudaGaugeParam.X, (void**)sitelink->Gauge_p(), ghost_sitelink, ghost_sitelink_diag, qudaGaugeParam.cpu_prec, optflag);
    llfat_reference_mg((void**)reflink->Gauge_p(), (void**)sitelink->Gauge_p(), ghost_sitelink, ghost_sitelink_diag, qudaGaugeParam.cpu_prec, act_path_coeff);
#else
    llfat_reference((void**)reflink->Gauge_p(), (void**)sitelink->Gauge_p(), qudaGaugeParam.cpu_prec, act_path_coeff);
#endif
  }

  struct timeval t0, t1, t2, t3;
  int flops= 61632; 

  switch(test){
  case 0:
    {
      llfat_init_cuda(&qudaGaugeParam);
      //The number comes from CPU implementation in MILC, fermion_links_helpers.c    
      
      gettimeofday(&t0, NULL);
#ifdef MULTI_GPU
      qudaGaugeParam.ga_pad = qudaGaugeParam.site_ga_pad;
      qudaGaugeParam.reconstruct = link_recon;
      
      loadLinkToGPU(cudaSiteLink, sitelink, &qudaGaugeParam);
      //cudaSiteLink->loadCPUField(*sitelink, QUDA_CPU_FIELD_LOCATION);
      
#else
      qudaGaugeParam.ga_pad = qudaGaugeParam.site_ga_pad;
      qudaGaugeParam.reconstruct = link_recon;

      loadLinkToGPU(cudaSiteLink, sitelink, &qudaGaugeParam);
      //cudaSiteLink->loadCPUField(*sitelink, QUDA_CPU_FIELD_LOCATION); 

#endif
      
      gettimeofday(&t1, NULL);  
      
      
      llfat_cuda(*cudaFatLink, *cudaSiteLink, *cudaStapleField, *cudaStapleField1, 
		 &qudaGaugeParam, act_path_coeff_2);
      break;

    }
  case 1:    
    {
      llfat_init_cuda_ex(&qudaGaugeParam_ex);
#ifdef MULTI_GPU
      gettimeofday(&t0, NULL);
      exchange_cpu_sitelink_ex(qudaGaugeParam.X, (void**)sitelink_ex->Gauge_p(), qudaGaugeParam.cpu_prec, 1);    
      qudaGaugeParam_ex.ga_pad = qudaGaugeParam_ex.site_ga_pad;
      qudaGaugeParam_ex.reconstruct = link_recon;
      loadLinkToGPU_ex(cudaSiteLink_ex, sitelink_ex, &qudaGaugeParam_ex);
      gettimeofday(&t1, NULL);
      llfat_cuda_ex(*cudaFatLink, *cudaSiteLink_ex, *cudaStapleField_ex, *cudaStapleField1_ex, &qudaGaugeParam, act_path_coeff_2);
#else
      gettimeofday(&t0, NULL);
      //exchange_cpu_sitelink_ex(qudaGaugeParam.X, (void**)sitelink_ex->Gauge_p(), qudaGaugeParam.cpu_prec, 1);    
      qudaGaugeParam_ex.ga_pad = qudaGaugeParam_ex.site_ga_pad;
      qudaGaugeParam_ex.reconstruct = link_recon;
      loadLinkToGPU_ex(cudaSiteLink_ex, sitelink_ex, &qudaGaugeParam_ex);
      gettimeofday(&t1, NULL);
      llfat_cuda_ex(*cudaFatLink, *cudaSiteLink_ex, *cudaStapleField_ex, *cudaStapleField1_ex, &qudaGaugeParam, act_path_coeff_2);
#endif

      break;
    }

  default:
    errorQuda("ERROR: test type(%d) not supported\n", test);
  }

  gettimeofday(&t2, NULL);
  storeLinkToCPU(fatlink, cudaFatLink, &qudaGaugeParam);
  
  //cudaFatLink->saveCPUField(*fatlink, QUDA_CPU_FIELD_LOCATION);
  gettimeofday(&t3, NULL);
  
#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))
  double secs = TDIFF(t0,t3);
  int i;
  void* myfatlink[4];  
  for(i=0;i < 4;i++){
    myfatlink[i] = malloc(V*gaugeSiteSize*gSize);
    if(myfatlink[i] == NULL){
      printf("Error: malloc failed for myfatlink[%d]\n", i);
      exit(1);
    }
    memset(myfatlink[i], 0, V*gaugeSiteSize*gSize);
  }

 for(i=0;i < V; i++){
   for(int dir=0; dir< 4; dir++){
     char* src = ((char*)fatlink->Gauge_p())+ (4*i+dir)*gaugeSiteSize*gSize;
     char* dst = ((char*)myfatlink[dir]) + i*gaugeSiteSize*gSize;
     memcpy(dst, src, gaugeSiteSize*gSize);
   }
 }  


  int res=1;
  for(int i=0;i < 4;i++){
    res &= compare_floats(((void**)reflink->Gauge_p())[i], myfatlink[i], V*gaugeSiteSize, 1e-3, qudaGaugeParam.cpu_prec);
  }
  int accuracy_level;
  
  accuracy_level = strong_check_link((void**)reflink->Gauge_p(), myfatlink, V, qudaGaugeParam.cpu_prec);  
  
  printfQuda("Test %s\n",(1 == res) ? "PASSED" : "FAILED");	    
  int volume = qudaGaugeParam.X[0]*qudaGaugeParam.X[1]*qudaGaugeParam.X[2]*qudaGaugeParam.X[3];
  double perf = 1.0* flops*volume/(secs*1024*1024*1024);
  printfQuda("gpu time =%.2f ms, flops= %.2f Gflops\n", secs*1000, perf);


  for(i=0;i < 4;i++){
	free(myfatlink[i]);
  }
  llfat_end(test);
    
  if (res == 0){//failed
    printfQuda("\n");
    printfQuda("Warning: your test failed. \n");
    printfQuda("	Did you use --verify?\n");
    printfQuda("	Did you check the GPU health by running cuda memtest?\n");
  }
  
  printfQuda("h2d=%f s, computation in gpu=%f s, d2h=%f s, total time=%f s\n",
             TDIFF(t0, t1), TDIFF(t1, t2), TDIFF(t2, t3), TDIFF(t0, t3));
  
  
  return accuracy_level;
}            


static void
display_test_info(int test)
{
  printfQuda("running the following test:\n");
    
  printfQuda("link_precision           link_reconstruct           space_dimension        T_dimension       Test\n");
  printfQuda("%s                       %s                         %d/%d/%d/                  %d            %d\n", 
	     get_prec_str(prec),
	     get_recon_str(link_recon), 
	     xdim, ydim, zdim, tdim, test);

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n",
             commDimPartitioned(0),
             commDimPartitioned(1),
             commDimPartitioned(2),
             commDimPartitioned(3));

  return ;
  
}

static void
usage(char** argv )
{
  printfQuda("Usage: %s <args>\n", argv[0]);
  printfQuda("  --device <dev_id>               Set which device to run on\n");
  printfQuda("  --gprec <double/single/half>    Link precision\n"); 
  printfQuda("  --recon <8/12>                  Link reconstruction type\n"); 
  printfQuda("  --sdim <n>                      Set spacial dimention\n");
  printfQuda("  --tdim <n>                      Set T dimention size(default 24)\n"); 
  printfQuda("  --verify                        Verify the GPU results using CPU results\n");
  printfQuda("  --help                          Print out this message\n"); 
  exit(1);
  return ;
}

int 
main(int argc, char **argv) 
{

  int test = 0;
  
  //default to 18 reconstruct, 8^3 x 8 
  link_recon = QUDA_RECONSTRUCT_NO;
  xdim=ydim=zdim=tdim=8;
  cpu_prec = prec = QUDA_DOUBLE_PRECISION;

  int i;
  for (i =1;i < argc; i++){

    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }

    if( strcmp(argv[i], "--cpu_prec") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      cpu_prec =  get_prec(argv[i+1]);
      i++;
      continue;	    
    }	 
    
    if( strcmp(argv[i], "--test") == 0){
      if (i+1 >= argc){
	usage(argv);
      }
      test =  atoi(argv[i+1]);
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

  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);


  display_test_info(test);

    
  int accuracy_level = llfat_test(test);
    
  printfQuda("accuracy_level=%d\n", accuracy_level);

  endCommsQuda();

  int ret;
  if(accuracy_level >=3 ){
    ret = 0; 
  }else{
    ret = 1; //we delclare the test failed
  }

  return ret;
}


