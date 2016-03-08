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
#include "malloc_quda.h"

#ifdef MULTI_GPU
#include "face_quda.h"
#include "comm_quda.h"
#endif

#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

extern void usage(char** argv);
extern bool verify_results;

extern int device;
extern int xdim, ydim, zdim, tdim;
extern int gridsize_from_cmdline[];

extern QudaReconstructType link_recon;
extern QudaPrecision prec;
static QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
//static QudaGaugeFieldOrder gauge_order = QUDA_QDP_GAUGE_ORDER;
static QudaGaugeFieldOrder gauge_order = QUDA_MILC_GAUGE_ORDER;

static size_t gSize;

  static void
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
  /*
     qudaGaugeParam.flag = QUDA_FAT_PRESERVE_CPU_GAUGE
     | QUDA_FAT_PRESERVE_GPU_GAUGE
     | QUDA_FAT_PRESERVE_COMM_MEM;
     */
  qudaGaugeParam.preserve_gauge =0;
  void* fatlink = pinned_malloc(4*V*gaugeSiteSize*gSize);
  void* longlink = pinned_malloc(4*V*gaugeSiteSize*gSize);

  void* sitelink[4];
  for(int i=0;i < 4;i++) sitelink[i] = pinned_malloc(V*gaugeSiteSize*gSize);

  void* sitelink_ex[4];
  for(int i=0;i < 4;i++) sitelink_ex[i] = pinned_malloc(V_ex*gaugeSiteSize*gSize);

  void* milc_sitelink;
  milc_sitelink = (void*)safe_malloc(4*V*gaugeSiteSize*gSize);

  void* milc_sitelink_ex;
  milc_sitelink_ex = (void*)safe_malloc(4*V_ex*gaugeSiteSize*gSize);

  createSiteLinkCPU(sitelink, qudaGaugeParam.cpu_prec, 1);

  if(gauge_order == QUDA_MILC_GAUGE_ORDER){
    for(int i=0; i<V; ++i){
      for(int dir=0; dir<4; ++dir){
        char* src = (char*)sitelink[dir];
        memcpy((char*)milc_sitelink + (i*4 + dir)*gaugeSiteSize*gSize, src+i*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
      }	
    }
  }

  int X1=Z[0];
  int X2=Z[1];
  int X3=Z[2];
  int X4=Z[3];

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
#ifdef MULTI_GPU
      continue;
#endif
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

  QudaComputeFatMethod method = (test) ? QUDA_COMPUTE_FAT_EXTENDED_VOLUME : QUDA_COMPUTE_FAT_STANDARD;
  void* longlink_ptr = longlink;
#ifdef MULTI_GPU
  if(!test) longlink_ptr = NULL; // Have to have an extended volume for the long-link calculation
#endif

  gettimeofday(&t0, NULL);
  computeKSLinkQuda(fatlink, longlink_ptr, NULL, milc_sitelink, act_path_coeff, &qudaGaugeParam, method);
  gettimeofday(&t1, NULL);

  double secs = TDIFF(t0,t1);

  void* fat_reflink[4];
  void* long_reflink[4];
  for(int i=0;i < 4;i++){
    fat_reflink[i] = safe_malloc(V*gaugeSiteSize*gSize);
    long_reflink[i] = safe_malloc(V*gaugeSiteSize*gSize);
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
    int Vs[4] = {Vs_x, Vs_y, Vs_z, Vs_t};
    for(int i=0;i < 4; i++){
      ghost_sitelink[i] = safe_malloc(8*Vs[i]*gaugeSiteSize*gSize);
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
          ghost_sitelink_diag[nu*4+mu] = safe_malloc(Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
          memset(ghost_sitelink_diag[nu*4+mu], 0, Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
        }

      }
    }

    exchange_cpu_sitelink(qudaGaugeParam.X, sitelink, ghost_sitelink, ghost_sitelink_diag, qudaGaugeParam.cpu_prec, &qudaGaugeParam, optflag);
    llfat_reference_mg(fat_reflink, sitelink, ghost_sitelink, ghost_sitelink_diag, qudaGaugeParam.cpu_prec, coeff);
  
    {
      int R[4] = {2,2,2,2};
      exchange_cpu_sitelink_ex(qudaGaugeParam.X, R, sitelink_ex, QUDA_QDP_GAUGE_ORDER, qudaGaugeParam.cpu_prec, 0, 4);
      computeLongLinkCPU(long_reflink, sitelink_ex, qudaGaugeParam.cpu_prec, coeff);
    }
#else
    llfat_reference(fat_reflink, sitelink, qudaGaugeParam.cpu_prec, coeff);
    computeLongLinkCPU(long_reflink, sitelink, qudaGaugeParam.cpu_prec, coeff);
#endif

  }//verify_results

  //format change for fatlink and longlink
  void* myfatlink[4];
  void* mylonglink[4];
  for(int i=0;i < 4;i++){
    myfatlink[i] = safe_malloc(V*gaugeSiteSize*gSize);
    mylonglink[i] = safe_malloc(V*gaugeSiteSize*gSize);
    memset(myfatlink[i], 0, V*gaugeSiteSize*gSize);
    memset(mylonglink[i], 0, V*gaugeSiteSize*gSize);
  }

  for(int i=0;i < V; i++){
    for(int dir=0; dir< 4; dir++){
      char* src = ((char*)fatlink)+ (4*i+dir)*gaugeSiteSize*gSize;
      char* dst = ((char*)myfatlink[dir]) + i*gaugeSiteSize*gSize;
      memcpy(dst, src, gaugeSiteSize*gSize);

      src = ((char*)longlink)+ (4*i+dir)*gaugeSiteSize*gSize;
      dst = ((char*)mylonglink[dir]) + i*gaugeSiteSize*gSize;
      memcpy(dst, src, gaugeSiteSize*gSize);
    }
  }

  if (verify_results) {
    printfQuda("Checking fat links...\n");
    int res=1;
    for(int dir=0; dir<4; dir++){
      res &= compare_floats(fat_reflink[dir], myfatlink[dir], V*gaugeSiteSize, 1e-3, qudaGaugeParam.cpu_prec);
    }
    
    strong_check_link(myfatlink, "GPU results: ",
		      fat_reflink, "CPU reference results:",
		      V, qudaGaugeParam.cpu_prec);
    
    printfQuda("Fat-link test %s\n\n",(1 == res) ? "PASSED" : "FAILED");
#ifdef MULTI_GPU
    if(test){
#endif
      printfQuda("Checking long links...\n");
      res = 1;
      for(int dir=0; dir<4; ++dir){
	res &= compare_floats(long_reflink[dir], mylonglink[dir], V*gaugeSiteSize, 1e-3, qudaGaugeParam.cpu_prec);
      }
      
      strong_check_link(mylonglink, "GPU results: ",
			long_reflink, "CPU reference results:",
			V, qudaGaugeParam.cpu_prec);
      
      printfQuda("Long-link test %s\n\n",(1 == res) ? "PASSED" : "FAILED");
      
#ifdef MULTI_GPU
    }else{ // !test
      printfQuda("Extended volume is required for multi-GPU long-link construction\n");
    }
#endif
  }

  int volume = qudaGaugeParam.X[0]*qudaGaugeParam.X[1]*qudaGaugeParam.X[2]*qudaGaugeParam.X[3];
  int flops= 61632;
#ifdef MULTI_GPU
  if(test) flops += (252*4); // long-link contribution 
#else
  flops += (252*4); // 2*117 + 18 (two matrix-matrix multiplications and a matrix rescale)
#endif

  double perf = 1.0* flops*volume/(secs*1024*1024*1024);
  printfQuda("link computation time =%.2f ms, flops= %.2f Gflops\n", secs*1000, perf);


  for(int i=0;i < 4;i++){
    host_free(myfatlink[i]);
    host_free(mylonglink[i]);
  }

#ifdef MULTI_GPU
  if (verify_results){
    for(int i=0; i<4; i++){
      host_free(ghost_sitelink[i]);
      for(int j=0;j <4; j++){
        if (i==j) continue;
        host_free(ghost_sitelink_diag[i*4+j]);
      }
    }
  }
#endif

  for(int i=0;i < 4; i++){
    host_free(sitelink[i]);
    host_free(sitelink_ex[i]);
    host_free(fat_reflink[i]);
    host_free(long_reflink[i]);
  }
  host_free(fatlink);
  host_free(longlink);
  if(milc_sitelink) host_free(milc_sitelink);
  if(milc_sitelink_ex) host_free(milc_sitelink_ex);
#ifdef MULTI_GPU
  exchange_llfat_cleanup();
#endif
  endQuda();
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
      dimPartitioned(0),
      dimPartitioned(1),
      dimPartitioned(2),
      dimPartitioned(3));
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
  printfQuda("    --gauge-order <qdp/milc>		   # ordering of the input gauge-field\n");
  return ;
}

int
main(int argc, char **argv)
{

#ifdef MULTI_GPU
  int test = 1;
#else
  int test = 0;
#endif
  //default to 18 reconstruct, 8^3 x 8
  link_recon = QUDA_RECONSTRUCT_NO;
  xdim=ydim=zdim=tdim=8;
  cpu_prec = prec = QUDA_DOUBLE_PRECISION;

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

    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }


#ifdef MULTI_GPU
  if(gauge_order == QUDA_MILC_GAUGE_ORDER && test == 0){
    errorQuda("ERROR: milc format for multi-gpu with test0 is not supported yet!\n");
  }
#endif

  initComms(argc, argv, gridsize_from_cmdline);
  display_test_info(test);
  llfat_test(test);
  finalizeComms();
}


