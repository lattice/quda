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
#include <sys/time.h>
#include <gtest/gtest.h>

using namespace quda;

cudaGaugeField *cudaFatLink = NULL;
cpuGaugeField  *cpuFatLink  = NULL;

cudaGaugeField *cudaOprod = NULL;
cpuGaugeField  *cpuOprod = NULL;

cudaGaugeField *cudaResult = NULL;
cpuGaugeField *cpuResult = NULL;

cpuGaugeField *cpuReference = NULL;

static QudaGaugeParam gaugeParam;

// Create a field of links that are not su3_matrices
void createNoisyLinkCPU(void** field, QudaPrecision prec, int seed)
{
  createSiteLinkCPU(field, prec, 0);

  srand(seed);
  for(int dir=0; dir<4; ++dir){
    for(int i=0; i<V*18; ++i){
      if(prec == QUDA_DOUBLE_PRECISION){
       double* ptr = ((double**)field)[dir] + i; 
       *ptr += (rand() - RAND_MAX/2.0)/(20.0*RAND_MAX);
      }else if(prec == QUDA_SINGLE_PRECISION){
     	  float* ptr = ((float**)field)[dir]+i;
        *ptr += (rand() - RAND_MAX/2.0)/(20.0*RAND_MAX);
      }  
    }
  }
}

// allocate memory
// set the layout, etc.
static void hisq_force_init()
{
  gaugeParam.X[0] = xdim;
  gaugeParam.X[1] = ydim;
  gaugeParam.X[2] = zdim;
  gaugeParam.X[3] = tdim;

  setDims(gaugeParam.X);

  gaugeParam.cpu_prec = QUDA_DOUBLE_PRECISION;
  gaugeParam.cuda_prec = prec;
  gaugeParam.reconstruct = link_recon;
  gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  GaugeFieldParam gParam(gaugeParam);
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.anisotropy = 1;
  
  cpuFatLink   = new cpuGaugeField(gParam);
  cpuOprod     = new cpuGaugeField(gParam);
  cpuResult    = new cpuGaugeField(gParam); 
  cpuReference = new cpuGaugeField(gParam);
 
  // create "gauge fields"
  int seed=0;
#ifdef MULTI_GPU
  seed += comm_rank();
#endif

  createNoisyLinkCPU((void**)cpuFatLink->Gauge_p(), gaugeParam.cpu_prec, seed);
  createNoisyLinkCPU((void**)cpuOprod->Gauge_p(), gaugeParam.cpu_prec, seed+1);

  gParam.setPrecision(gaugeParam.cuda_prec, true);

  cudaFatLink = new cudaGaugeField(gParam);
  cudaOprod   = new cudaGaugeField(gParam); 
  cudaResult  = new cudaGaugeField(gParam);

  gParam.order = QUDA_QDP_GAUGE_ORDER;

  cudaFatLink->loadCPUField(*cpuFatLink);
  cudaOprod->loadCPUField(*cpuOprod);
}

static void hisq_force_end()
{
  delete cpuFatLink;
  delete cpuOprod;
  delete cpuResult;

  delete cudaFatLink;
  delete cudaOprod;
  delete cudaResult;

  delete cpuReference;
}

TEST(hisq_force_unitarize, verify)
{
  hisq_force_init();

  double unitarize_eps = 1e-5;
  const double hisq_force_filter = 5e-5;
  const double max_det_error = 1e-12;
  const bool allow_svd = true;
  const bool svd_only = false;
  const double svd_rel_err = 1e-8;
  const double svd_abs_err = 1e-8;

  fermion_force::setUnitarizeForceConstants(unitarize_eps, hisq_force_filter, max_det_error, allow_svd, svd_only, svd_rel_err, svd_abs_err);

  int *num_failures_dev = (int *)device_malloc(sizeof(int));
  qudaMemset(num_failures_dev, 0, sizeof(int));

  printfQuda("Calling unitarizeForce\n");
  fermion_force::unitarizeForce(*cudaResult, *cudaOprod, *cudaFatLink, num_failures_dev);

  device_free(num_failures_dev);

  if (verify_results) {
    printfQuda("Calling unitarizeForceCPU\n");
    fermion_force::unitarizeForceCPU(*cpuResult, *cpuOprod, *cpuFatLink);
  }

  cudaResult->saveCPUField(*cpuReference);
  
  printfQuda("Comparing CPU and GPU results\n");
  int res[4];

  double accuracy = prec == QUDA_DOUBLE_PRECISION ? 1e-10 : 1e-5;
  for (int dir = 0; dir < 4; ++dir) {
    res[dir] = compare_floats(((char **)cpuReference->Gauge_p())[dir], ((char **)cpuResult->Gauge_p())[dir],
                              cpuReference->Volume() * gauge_site_size, accuracy, gaugeParam.cpu_prec);

    comm_allreduce_int(&res[dir]);
    res[dir] /= comm_size();
  }

  hisq_force_end();

  for (int dir = 0; dir < 4; ++dir) { ASSERT_EQ(res[dir], 1) << "Dir:" << dir; }
}

static void display_test_info()
{
  printfQuda("running the following fermion force computation test:\n");
    
  printfQuda("link_precision           link_reconstruct           space_dim(x/y/z)         T_dimension\n");
  printfQuda("%s                       %s                         %d/%d/%d                  %d \n", get_prec_str(prec),
             get_recon_str(link_recon), xdim, ydim, zdim, tdim);
}

int main(int argc, char **argv)
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);

  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);
  initQuda(device_ordinal);

  display_test_info();

  int test_rc = RUN_ALL_TESTS();

  endQuda();
  finalizeComms();

  return test_rc;
}

