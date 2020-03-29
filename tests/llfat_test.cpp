#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "quda.h"
#include "host_utils.h"
#include "llfat_utils.h"
#include <command_line_params.h>
#include "misc.h"
#include "util_quda.h"
#include "malloc_quda.h"

#ifdef MULTI_GPU
#include "comm_quda.h"
#endif

#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

static QudaGaugeFieldOrder gauge_order = QUDA_MILC_GAUGE_ORDER;

static void llfat_test()
{
  QudaGaugeParam qudaGaugeParam;
#ifdef MULTI_GPU
  void* ghost_sitelink[4];
  void* ghost_sitelink_diag[16];
#endif

  initQuda(device);

  cpu_prec = prec;
  host_gauge_data_type_size = cpu_prec;
  qudaGaugeParam = newQudaGaugeParam();

  qudaGaugeParam.anisotropy = 1.0;

  qudaGaugeParam.X[0] = xdim;
  qudaGaugeParam.X[1] = ydim;
  qudaGaugeParam.X[2] = zdim;
  qudaGaugeParam.X[3] = tdim;

  setDims(qudaGaugeParam.X);

  qudaGaugeParam.cpu_prec = cpu_prec;
  qudaGaugeParam.cuda_prec = qudaGaugeParam.cuda_prec_sloppy = prec;
  qudaGaugeParam.gauge_order = gauge_order;
  qudaGaugeParam.type = QUDA_WILSON_LINKS;
  qudaGaugeParam.reconstruct = qudaGaugeParam.reconstruct_sloppy = link_recon;
  qudaGaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  qudaGaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  qudaGaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  qudaGaugeParam.ga_pad = 0;

  void *fatlink = pinned_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
  void *longlink = pinned_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

  void* sitelink[4];
  for (int i = 0; i < 4; i++) sitelink[i] = pinned_malloc(V * gauge_site_size * host_gauge_data_type_size);

  void* sitelink_ex[4];
  for (int i = 0; i < 4; i++) sitelink_ex[i] = pinned_malloc(V_ex * gauge_site_size * host_gauge_data_type_size);

  void* milc_sitelink;
  milc_sitelink = (void *)safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

  void* milc_sitelink_ex;
  milc_sitelink_ex = (void *)safe_malloc(4 * V_ex * gauge_site_size * host_gauge_data_type_size);

  createSiteLinkCPU(sitelink, qudaGaugeParam.cpu_prec, 1);

  if(gauge_order == QUDA_MILC_GAUGE_ORDER){
    for(int i=0; i<V; ++i){
      for(int dir=0; dir<4; ++dir){
        char* src = (char*)sitelink[dir];
        memcpy((char *)milc_sitelink + (i * 4 + dir) * gauge_site_size * host_gauge_data_type_size,
               src + i * gauge_site_size * host_gauge_data_type_size, gauge_site_size * host_gauge_data_type_size);
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
      memcpy(dst + i * gauge_site_size * host_gauge_data_type_size,
             src + idx * gauge_site_size * host_gauge_data_type_size, gauge_site_size * host_gauge_data_type_size);

      // milc ordering
      memcpy((char *)milc_sitelink_ex + (i * 4 + dir) * gauge_site_size * host_gauge_data_type_size,
             src + idx * gauge_site_size * host_gauge_data_type_size, gauge_site_size * host_gauge_data_type_size);
    }//dir
  }//i


  double act_path_coeff[6];
  for(int i=0;i < 6;i++){
    act_path_coeff[i]= 0.1*i;
  }


  //only record the last call's performance
  //the first one is for creating the cpu/cuda data structures
  struct timeval t0, t1;

  void* longlink_ptr = longlink;
  {
    printfQuda("Tuning...\n");
    computeKSLinkQuda(fatlink, longlink_ptr, NULL, milc_sitelink, act_path_coeff, &qudaGaugeParam);
  }

  printfQuda("Running %d iterations of computation\n", niter);
  gettimeofday(&t0, NULL);
  for (int i=0; i<niter; i++)
    computeKSLinkQuda(fatlink, longlink_ptr, NULL, milc_sitelink, act_path_coeff, &qudaGaugeParam);
  gettimeofday(&t1, NULL);

  double secs = TDIFF(t0,t1);

  void* fat_reflink[4];
  void* long_reflink[4];
  for(int i=0;i < 4;i++){
    fat_reflink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    long_reflink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  }

  if (verify_results){

    //FIXME: we have this complication because references takes coeff as float/double
    //        depending on the precision while the GPU code aways take coeff as double
    void* coeff;
    double coeff_dp[6];
    float  coeff_sp[6];
    for (int i=0; i < 6;i++) coeff_sp[i] = coeff_dp[i] = act_path_coeff[i];
    coeff = (prec == QUDA_DOUBLE_PRECISION) ? (void*)coeff_dp : (void*)coeff_sp;

#ifdef MULTI_GPU
    int optflag = 0;
    //we need x,y,z site links in the back and forward T slice
    // so it is 3*2*Vs_t
    int Vs[4] = {Vs_x, Vs_y, Vs_z, Vs_t};
    for (int i = 0; i < 4; i++)
      ghost_sitelink[i] = safe_malloc(8 * Vs[i] * gauge_site_size * host_gauge_data_type_size);

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
          ghost_sitelink_diag[nu * 4 + mu] = safe_malloc(Z[dir1] * Z[dir2] * gauge_site_size * host_gauge_data_type_size);
          memset(ghost_sitelink_diag[nu * 4 + mu], 0, Z[dir1] * Z[dir2] * gauge_site_size * host_gauge_data_type_size);
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
  for(int i=0; i < 4; i++){
    myfatlink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    mylonglink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    memset(myfatlink[i], 0, V * gauge_site_size * host_gauge_data_type_size);
    memset(mylonglink[i], 0, V * gauge_site_size * host_gauge_data_type_size);
  }

  for(int i=0; i < V; i++){
    for(int dir=0; dir< 4; dir++){
      char *src = ((char *)fatlink) + (4 * i + dir) * gauge_site_size * host_gauge_data_type_size;
      char *dst = ((char *)myfatlink[dir]) + i * gauge_site_size * host_gauge_data_type_size;
      memcpy(dst, src, gauge_site_size * host_gauge_data_type_size);

      src = ((char *)longlink) + (4 * i + dir) * gauge_site_size * host_gauge_data_type_size;
      dst = ((char *)mylonglink[dir]) + i * gauge_site_size * host_gauge_data_type_size;
      memcpy(dst, src, gauge_site_size * host_gauge_data_type_size);
    }
  }

  if (verify_results) {
    printfQuda("Checking fat links...\n");
    int res=1;
    for(int dir=0; dir<4; dir++){
      res &= compare_floats(fat_reflink[dir], myfatlink[dir], V * gauge_site_size, 1e-3, qudaGaugeParam.cpu_prec);
    }
    
    strong_check_link(myfatlink, "GPU results: ",
		      fat_reflink, "CPU reference results:",
		      V, qudaGaugeParam.cpu_prec);
    
    printfQuda("Fat-link test %s\n\n",(1 == res) ? "PASSED" : "FAILED");

    printfQuda("Checking long links...\n");
    res = 1;
    for(int dir=0; dir<4; ++dir){
      res &= compare_floats(long_reflink[dir], mylonglink[dir], V * gauge_site_size, 1e-3, qudaGaugeParam.cpu_prec);
    }
      
    strong_check_link(mylonglink, "GPU results: ",
		      long_reflink, "CPU reference results:",
		      V, qudaGaugeParam.cpu_prec);
      
    printfQuda("Long-link test %s\n\n",(1 == res) ? "PASSED" : "FAILED");
  }

  int volume = qudaGaugeParam.X[0]*qudaGaugeParam.X[1]*qudaGaugeParam.X[2]*qudaGaugeParam.X[3];
  long long flops= 61632 * (long long)niter;
  flops += (252*4)*(long long)niter; // long-link contribution

  double perf = flops*volume/(secs*1024*1024*1024);
  printfQuda("link computation time =%.2f ms, flops= %.2f Gflops\n", (secs*1000)/niter, perf);

  for (int i=0; i < 4; i++) {
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

  for(int i=0; i < 4; i++){
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

static void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("link_precision           link_reconstruct           space_dimension        T_dimension       Ordering\n");
  printfQuda("%s                       %s                         %d/%d/%d/                  %d             %s \n", 
      get_prec_str(prec),
      get_recon_str(link_recon), 
      xdim, ydim, zdim, tdim,
      get_gauge_order_str(gauge_order));

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n",
      dimPartitioned(0),
      dimPartitioned(1),
      dimPartitioned(2),
      dimPartitioned(3));

  return ;

}


int main(int argc, char **argv)
{

  //default to 18 reconstruct, 8^3 x 8
  link_recon = QUDA_RECONSTRUCT_NO;
  xdim=ydim=zdim=tdim=8;
  cpu_prec = prec = QUDA_DOUBLE_PRECISION;

  // command line options
  auto app = make_app();
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

  initComms(argc, argv, gridsize_from_cmdline);
  display_test_info();
  llfat_test();
  finalizeComms();
}


