#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "quda.h"
#include "test_util.h"
#include "llfat_reference.h"
#include "misc.h"
#include "util_quda.h"

#ifdef MULTI_GPU
#include "face_quda.h"
#include "mpicomm.h"
#endif

#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

extern void usage(char** argv);
static int verify_results = 0;

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
extern QudaPrecision prec;
static QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
static QudaGaugeFieldOrder gauge_order = QUDA_QDP_GAUGE_ORDER;

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

static int
llfat_test(int test)
{

  QudaGaugeParam qudaGaugeParam;
#ifdef MULTI_GPU
  void* ghost_sitelink[4];
  void* ghost_sitelink_diag[16];
#endif
  

  initQuda(device);

  cpu_prec = prec;
  gSize = cpu_prec;  
  qudaGaugeParam = newQudaGaugeParam();
  
  qudaGaugeParam.anisotropy = 1.0;
  
  qudaGaugeParam.X[0] = xdim;
  qudaGaugeParam.X[1] = ydim;
  qudaGaugeParam.X[2] = zdim;
  qudaGaugeParam.X[3] = tdim;

  setDims(qudaGaugeParam.X);
   
  qudaGaugeParam.cpu_prec = cpu_prec;
  qudaGaugeParam.cuda_prec = prec;
  qudaGaugeParam.gauge_order = gauge_order;
  qudaGaugeParam.type=QUDA_WILSON_LINKS;
  qudaGaugeParam.reconstruct = link_recon;
  qudaGaugeParam.flag = QUDA_FAT_PRESERVE_CPU_GAUGE
    | QUDA_FAT_PRESERVE_GPU_GAUGE
    | QUDA_FAT_PRESERVE_COMM_MEM;
  
  void* fatlink;
  cudaMallocHost((void**)&fatlink, 4*V*gaugeSiteSize*gSize);
  if(fatlink == NULL){
    errorQuda("ERROR: allocating fatlink failed\n");
  }
  
  void* sitelink[4];
  for(int i=0;i < 4;i++){
    cudaMallocHost((void**)&sitelink[i], V*gaugeSiteSize*gSize);
    if(sitelink[i] == NULL){
      errorQuda("ERROR; allocate sitelink[%d] failed\n", i);
    }
  }
  
  void* sitelink_ex[4];
  for(int i=0;i < 4;i++){
    cudaMallocHost((void**)&sitelink_ex[i], V_ex*gaugeSiteSize*gSize);
    if(sitelink_ex[i] == NULL){
      errorQuda("ERROR; allocate sitelink_ex[%d] failed\n", i);
    }
  }


  void* milc_sitelink;
  milc_sitelink = (void*)malloc(4*V*gaugeSiteSize*gSize);
  if(milc_sitelink == NULL){
    errorQuda("ERROR: allocating milc_sitelink failed\n");
  }

  void* milc_sitelink_ex;
  milc_sitelink_ex = (void*)malloc(4*V_ex*gaugeSiteSize*gSize);
  if(milc_sitelink_ex == NULL){
    errorQuda("Error: allocating milc_sitelink failed\n");
  }



  createSiteLinkCPU(sitelink, qudaGaugeParam.cpu_prec, 1);

  if(gauge_order == QUDA_MILC_GAUGE_ORDER){
	  for(int i=0; i<V; ++i){
		  for(int dir=0; dir<4; ++dir){
			  char* src = (char*)sitelink[dir];
			  memcpy((char*)milc_sitelink + (i*4 + dir)*gaugeSiteSize*gSize, src+i*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
		  }	
	  }
  }



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
      char* src = (char*)sitelink[dir];
      char* dst = (char*)sitelink_ex[dir];
      memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);

      // milc ordering 
      memcpy((char*)milc_sitelink_ex + (i*4 + dir)*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
    }//dir
  }//i
  

  double act_path_coeff[6];
  for(int i=0;i < 6;i++){
    act_path_coeff[i]= 0.1*i;
  }

  
  //only record the last call's performance
  //the first one is for creating the cpu/cuda data structures
  struct timeval t0, t1;
  
  for(int i=0;i < 2;i++){
    gettimeofday(&t0, NULL);
    if(gauge_order == QUDA_QDP_GAUGE_ORDER){
      if(test == 0){
	computeFatLinkQuda(fatlink, sitelink, act_path_coeff, &qudaGaugeParam,
			   QUDA_COMPUTE_FAT_STANDARD);
      }else{
	computeFatLinkQuda(fatlink, sitelink_ex, act_path_coeff, &qudaGaugeParam,
			   QUDA_COMPUTE_FAT_EXTENDED_VOLUME);
      }
    }else if(gauge_order == QUDA_MILC_GAUGE_ORDER){
      if(test == 0){
	computeFatLinkQuda(fatlink, (void**)milc_sitelink, act_path_coeff, &qudaGaugeParam,
			   QUDA_COMPUTE_FAT_STANDARD);
      }else{
	computeFatLinkQuda(fatlink, (void**)milc_sitelink_ex, act_path_coeff, &qudaGaugeParam,
			   QUDA_COMPUTE_FAT_EXTENDED_VOLUME);
      }
    }
    gettimeofday(&t1, NULL);
  }
  
  double secs = TDIFF(t0,t1);
  
  void* reflink[4];
  for(int i=0;i < 4;i++){
    reflink[i] = malloc(V*gaugeSiteSize*gSize);
    if(reflink[i] == NULL){
      errorQuda("ERROR; allocate reflink[%d] failed\n", i);
    }
  }
  
  if (verify_results){
    
    //FIXME: we have this compplication because references takes coeff as float/double 
    //        depending on the precision while the GPU code aways take coeff as double
    void* coeff;
    double coeff_dp[6];
    float  coeff_sp[6];
    for(int i=0;i < 6;i++){
      coeff_sp[i] = coeff_dp[i] = act_path_coeff[i];
    }
    if(prec == QUDA_DOUBLE_PRECISION){
      coeff = coeff_dp;
    }else{
      coeff = coeff_sp;
    }
#ifdef MULTI_GPU
    int optflag = 0;
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
    
    exchange_cpu_sitelink(qudaGaugeParam.X, sitelink, ghost_sitelink, ghost_sitelink_diag, qudaGaugeParam.cpu_prec, &qudaGaugeParam, optflag);
    llfat_reference_mg(reflink, sitelink, ghost_sitelink, ghost_sitelink_diag, qudaGaugeParam.cpu_prec, coeff);
#else
    llfat_reference(reflink, sitelink, qudaGaugeParam.cpu_prec, coeff);
#endif
    
  }//verify_results
  
  //format change for fatlink
  void* myfatlink[4];
  for(int i=0;i < 4;i++){
    myfatlink[i] = malloc(V*gaugeSiteSize*gSize);
    if(myfatlink[i] == NULL){
      printf("Error: malloc failed for myfatlink[%d]\n", i);
      exit(1);
    }
    memset(myfatlink[i], 0, V*gaugeSiteSize*gSize);
  }
  
  for(int i=0;i < V; i++){
    for(int dir=0; dir< 4; dir++){
      char* src = ((char*)fatlink)+ (4*i+dir)*gaugeSiteSize*gSize;
      char* dst = ((char*)myfatlink[dir]) + i*gaugeSiteSize*gSize;
      memcpy(dst, src, gaugeSiteSize*gSize);
    }
  }
  
    int res=1;
    for(int i=0;i < 4;i++){
      res &= compare_floats(reflink[i], myfatlink[i], V*gaugeSiteSize, 1e-3, qudaGaugeParam.cpu_prec);
    }
    int accuracy_level;
    
    accuracy_level = strong_check_link(myfatlink, "GPU results: ",
				       reflink, "CPU reference results:",
				       V, qudaGaugeParam.cpu_prec);
    
    printfQuda("Test %s\n",(1 == res) ? "PASSED" : "FAILED");
    int volume = qudaGaugeParam.X[0]*qudaGaugeParam.X[1]*qudaGaugeParam.X[2]*qudaGaugeParam.X[3];
    int flops= 61632;
    double perf = 1.0* flops*volume/(secs*1024*1024*1024);
    printfQuda("fatlink computation time =%.2f ms, flops= %.2f Gflops\n", secs*1000, perf);
    
    
    for(int i=0;i < 4;i++){
      free(myfatlink[i]);
    }
    
    if (res == 0){//failed
      printfQuda("\n");
      printfQuda("Warning: your test failed. \n");
      printfQuda(" Did you use --verify?\n");
      printfQuda(" Did you check the GPU health by running cuda memtest?\n");
    }
    
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
    
    for(int i=0;i < 4; i++){
      cudaFreeHost(sitelink[i]);
      cudaFreeHost(sitelink_ex[i]);
      free(reflink[i]);
    }
    cudaFreeHost(fatlink);
    if(milc_sitelink) free(milc_sitelink);
    if(milc_sitelink_ex) free(milc_sitelink_ex);
#ifdef MULTI_GPU
    exchange_llfat_cleanup();
#endif
    endQuda();
    
    return accuracy_level;
    
}

static void
display_test_info(int test)
{
  printfQuda("running the following test:\n");
    
  printfQuda("link_precision           link_reconstruct           space_dimension        T_dimension       Test          Ordering\n");
  printfQuda("%s                       %s                         %d/%d/%d/                  %d             %d              %s \n", 
	     get_prec_str(prec),
	     get_recon_str(link_recon), 
	     xdim, ydim, zdim, tdim, test, 
	     get_gauge_order_str(gauge_order));

#ifdef MULTI_GPU
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n",
             commDimPartitioned(0),
             commDimPartitioned(1),
             commDimPartitioned(2),
             commDimPartitioned(3));
#endif

  return ;
  
}

void
usage_extra(char** argv )
{
  printfQuda("Extra options:\n");
  printfQuda("    --test <0/1>                             # Test method\n");
  printfQuda("                                                0: standard method\n");
  printfQuda("                                                1: extended volume method\n");
  printfQuda("    --verify                                 # Verify the GPU results using CPU results\n");
  printfQuda("    --gauge-order <qdp/milc>		   # ordering of the input gauge-field\n");
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

    
    if( strcmp(argv[i], "--test") == 0){
      if (i+1 >= argc){
	usage(argv);
      }
      test = atoi(argv[i+1]);
      i++;
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


