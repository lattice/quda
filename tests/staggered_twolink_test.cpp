#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "quda.h"
#include "host_utils.h"
#include "llfat_utils.h"
#include <command_line_params.h>
#include "misc.h"
#include "util_quda.h"
#include "malloc_quda.h"
#include "comm_quda.h"


#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

int getLinkPadding_(const int dim[4])
{
  int padding = std::max(dim[1]*dim[2]*dim[3]/2, dim[0]*dim[2]*dim[3]/2);
  padding = std::max(padding, dim[0]*dim[1]*dim[3]/2);
  padding = std::max(padding, dim[0]*dim[1]*dim[2]/2);
  return padding;
}

void initExtendedField(void* sitelink_ex[4], void* sitelink[4]) 
{	
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
      char* src = (char*)sitelink[dir];
      char* dst = (char*)sitelink_ex[dir];
      memcpy(dst + i * gauge_site_size * host_gauge_data_type_size,
             src + idx * gauge_site_size * host_gauge_data_type_size, gauge_site_size * host_gauge_data_type_size);
    }//dir
  }//i
  return;
}

static QudaGaugeFieldOrder gauge_order = QUDA_MILC_GAUGE_ORDER;

static void twolink_test(int argc, char **argv)
{
  QudaGaugeParam qudaGaugeParam;

  initQuda(device_ordinal);

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
  qudaGaugeParam.ga_pad = getLinkPadding_(qudaGaugeParam.X);

  void *twolink = pinned_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

  void* sitelink[4];
  for (int i = 0; i < 4; i++) sitelink[i]   = pinned_malloc(V * gauge_site_size * host_gauge_data_type_size);

  void* sitelink_ex[4];
  for (int i = 0; i < 4; i++) sitelink_ex[i] = pinned_malloc(V_ex * gauge_site_size * host_gauge_data_type_size);

  void* milc_sitelink;
  milc_sitelink = (void *)safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

  constructHostGaugeField(sitelink, qudaGaugeParam, argc, argv);
  // Reorder gauge fields to MILC order
  reorderQDPtoMILC(milc_sitelink, sitelink, V, gauge_site_size, qudaGaugeParam.cpu_prec, qudaGaugeParam.cpu_prec);
  //
  initExtendedField(sitelink_ex, sitelink);  

  //only record the last call's performance
  //the first one is for creating the cpu/cuda data structures
  struct timeval t0, t1;

  loadGaugeQuda(milc_sitelink, &qudaGaugeParam);

  {
    printfQuda("Tuning...\n");
    computeTwoLinkQuda((void*)twolink, nullptr, &qudaGaugeParam);
  }

  printfQuda("Running %d iterations of computation\n", niter);
  gettimeofday(&t0, NULL);
  for (int i=0; i<niter; i++)
    computeTwoLinkQuda((void*)twolink, nullptr, &qudaGaugeParam);
  gettimeofday(&t1, NULL);

  double secs = TDIFF(t0,t1);

  if (verify_results) {  

    void* two_reflink[4];
    for(int i=0;i < 4;i++){
      two_reflink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    }

    computeTwoLinkCPU(two_reflink, sitelink_ex, &qudaGaugeParam);
 
    void* mytwolink[4];
    for(int i=0; i < 4; i++){
      mytwolink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      memset(mytwolink[i], 0, V * gauge_site_size * host_gauge_data_type_size);
    }

    for(int i=0; i < V; i++){
      for(int dir=0; dir< 4; dir++){
        char *src = ((char *)twolink) + (4 * i + dir) * gauge_site_size * host_gauge_data_type_size;
        char *dst = ((char *)mytwolink[dir]) + i * gauge_site_size * host_gauge_data_type_size;
        memcpy(dst, src, gauge_site_size * host_gauge_data_type_size);
      }
    }

    printfQuda("Checking two links...\n");
    int res = 1;
    for(int dir=0; dir<4; ++dir){
      res &= compare_floats(two_reflink[dir], mytwolink[dir], V * gauge_site_size, 1e-3, qudaGaugeParam.cpu_prec);
    }
      
    strong_check_link(mytwolink, "GPU results: ",
		      two_reflink, "CPU reference results:",
		      V, qudaGaugeParam.cpu_prec);
      
    printfQuda("Two-link test %s\n\n",(1 == res) ? "PASSED" : "FAILED");

    for (int i=0; i < 4; i++) {
      host_free(mytwolink[i]);
      host_free(two_reflink[i]);
    }
  }

  int volume = qudaGaugeParam.X[0]*qudaGaugeParam.X[1]*qudaGaugeParam.X[2]*qudaGaugeParam.X[3];
  long long flops= 61632 * (long long)niter;
  flops += (252*4)*(long long)niter; // long-link contribution

  double perf = flops*volume/(secs*1024*1024*1024);
  printfQuda("link computation time =%.2f ms, flops= %.2f Gflops\n", (secs*1000)/niter, perf);

  for(int i=0; i < 4; i++){
    host_free(sitelink[i]);
    host_free(sitelink_ex[i]);
  }
  host_free(twolink);
  if(milc_sitelink) host_free(milc_sitelink);

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
  twolink_test(argc, argv);
  finalizeComms();

  return 0;
}


