#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <quda.h>
#include <instantiate.h>
#include <gauge_field.h>

#include "host_utils.h"
#include "command_line_params.h"
#include "misc.h"
#include "test.h"
#include "hisq_force_reference.h"
#include "ks_improved_force.h"

using test_t = ::testing::tuple<QudaPrecision>;

class HisqUnitarizeTest : public ::testing::TestWithParam<test_t>
{
protected:
  QudaPrecision precision;

public:
  HisqUnitarizeTest() : precision(::testing::get<0>(GetParam())) { }
};

void hisq_unitarize(QudaPrecision prec)
{
  setVerbosity(verbosity);
  QudaGaugeParam gaugeParam;

  gaugeParam.X[0] = xdim;
  gaugeParam.X[1] = ydim;
  gaugeParam.X[2] = zdim;
  gaugeParam.X[3] = tdim;

  setDims(gaugeParam.X);

  gaugeParam.location = QUDA_CPU_FIELD_LOCATION;
  gaugeParam.cpu_prec = QUDA_DOUBLE_PRECISION;
  gaugeParam.cuda_prec = prec;
  gaugeParam.reconstruct = link_recon;
  gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  quda::GaugeFieldParam gParam(gaugeParam);
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.anisotropy = 1;

  quda::GaugeField cpuFatLink = quda::GaugeField(gParam);

  auto cpuOprod = quda::GaugeField(gParam);
  auto cpuResult = quda::GaugeField(gParam);
  auto cpuReference = quda::GaugeField(gParam);

  // create "gauge fields"
  createSiteLinkCPU(cpuFatLink, gaugeParam.cpu_prec, SiteLinkType::SITELINK_NOISY);
  createSiteLinkCPU(cpuOprod, gaugeParam.cpu_prec, SiteLinkType::SITELINK_NOISY);

  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.setPrecision(gaugeParam.cuda_prec, true);

  auto cudaFatLink = quda::GaugeField(gParam);
  auto cudaOprod = quda::GaugeField(gParam);
  auto cudaResult = quda::GaugeField(gParam);

  gParam.order = QUDA_QDP_GAUGE_ORDER;

  cudaFatLink.copy(cpuFatLink);
  cudaOprod.copy(cpuOprod);

  double unitarize_eps = 1e-5;
  const double hisq_force_filter = 5e-5;
  const double max_det_error = 1e-12;
  const bool allow_svd = true;
  const bool svd_only = false;
  const double svd_rel_err = 1e-8;
  const double svd_abs_err = 1e-8;

  quda::fermion_force::setUnitarizeForceConstants(unitarize_eps, hisq_force_filter, max_det_error, allow_svd, svd_only,
                                                  svd_rel_err, svd_abs_err);

  int *num_failures_dev = (int *)device_malloc(sizeof(int));
  qudaMemset(num_failures_dev, 0, sizeof(int));

  printfQuda("Calling unitarizeForce\n");
  quda::fermion_force::unitarizeForce(cudaResult, cudaOprod, cudaFatLink, num_failures_dev);

  device_free(num_failures_dev);

  if (verify_results) {
    printfQuda("Calling unitarizeForceCPU\n");
    quda::fermion_force::unitarizeForceCPU(cpuResult, cpuOprod, cpuFatLink);
  }

  cpuReference.copy(cudaResult);

  printfQuda("Comparing CPU and GPU results\n");
  int res[4];

  double accuracy = prec == QUDA_DOUBLE_PRECISION ? 1e-10 : 1e-5;
  for (int dir = 0; dir < 4; ++dir) {
    res[dir] = compare_floats(cpuReference.data<void *>(dir), cpuResult.data<void *>(dir),
                              cpuReference.Volume() * gauge_site_size, accuracy, gaugeParam.cpu_prec);

    quda::comm_allreduce_int(res[dir]);
    res[dir] /= quda::comm_size();
  }

  for (int dir = 0; dir < 4; ++dir) { ASSERT_EQ(res[dir], 1) << "Dir:" << dir; }
}

TEST_P(HisqUnitarizeTest, verify)
{
  prec = ::testing::get<0>(GetParam());
  if (!quda::is_enabled(prec)) GTEST_SKIP();
  hisq_unitarize(prec);
}

auto test_str
  = [](testing::TestParamInfo<test_t> param) { return std::string(get_prec_str(::testing::get<0>(param.param))); };

INSTANTIATE_TEST_SUITE_P(, HisqUnitarizeTest, ::testing::Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION), test_str);

struct hisq_unitarize_test : public quda_test {
  void display_info() const override
  {
    printfQuda("link_precision           link_reconstruct           space_dim(x/y/z)         T_dimension\n");
    printfQuda("%s                       %s                         %d/%d/%d                  %d \n",
               get_prec_str(prec), get_recon_str(link_recon), xdim, ydim, zdim, tdim);
  }

  hisq_unitarize_test(int argc, char **argv) : quda_test("hisq_unitarize_test", argc, argv) { }
};

int main(int argc, char **argv)
{
  hisq_unitarize_test test(argc, argv);
  test.init();
  int test_rc = 0;

  if (!enable_testing) {
    hisq_unitarize(prec);
  } else {
    test_rc = test.execute();
  }
  return test_rc;
}
