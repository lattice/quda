#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <quda.h>
#include <gauge_field.h>
#include <unitarization_links.h>

#include <gtest/gtest.h>

#include "host_utils.h"
#include "command_line_params.h"
#include "misc.h"
#include "test.h"

using namespace quda;

static std::array<std::array<double, 6>, 3> act_paths;

// Input sitelinks (QDP order, host precision)
static void *qdp_sitelink[4] = {};

// CPU reference links (QDP order, host precision)
static void *v_reflink[4] = {};
static void *w_reflink[4] = {};
static void *fat_reflink[4] = {};
static void *long_reflink[4] = {};
static void *fat_reflink_eps[4] = {};
static void *long_reflink_eps[4] = {};

// GPU results in MILC order (GPU precision) -- per parameter combo
static void *milc_sitelink = nullptr;
static void *vlink = nullptr;
static void *wlink = nullptr;
static void *fatlink = nullptr;
static void *longlink = nullptr;
static void *fatlink_eps = nullptr;
static void *longlink_eps = nullptr;

// GPU results reordered to QDP (host precision) -- per parameter combo
static void *qdp_vlink[4] = {};
static void *qdp_wlink[4] = {};
static void *qdp_fatlink[4] = {};
static void *qdp_longlink[4] = {};
static void *qdp_fatlink_eps[4] = {};
static void *qdp_longlink_eps[4] = {};

// Current parameter combo tracking
static QudaGaugeParam gauge_param;
static QudaPrecision cur_prec = QUDA_INVALID_PRECISION;
static bool cur_has_naik = false;
static bool host_initialized = false;
static bool links_valid = false;
static int cur_n_naiks = 1;

static void init_host()
{
  if (host_initialized) return;

  gauge_param = newQudaGaugeParam();
  setStaggeredGaugeParam(gauge_param);

  setDims(gauge_param.X);
  dw_setDims(gauge_param.X, 1);

  setUnitarizeLinksConstants(1e-6, 1e-11, true, false, 1e-4, 1e-4);

  double u1 = 1.0 / tadpole_factor;
  double u2 = u1 * u1;
  double u4 = u2 * u2;
  double u6 = u4 * u2;

  act_paths[0] = {(1.0 / 8.0), u2 * (0.0), u2 * (-1.0 / 8.0) * 0.5, u4 * (1.0 / 8.0) * 0.25 * 0.5,
                  u6 * (-1.0 / 8.0) * 0.125 * (1.0 / 6.0), u4 * (0.0)};

  act_paths[1] = {((1.0 / 8.0) + (2.0 * 6.0 / 16.0) + (1.0 / 8.0)), (-1.0 / 24.0), (-1.0 / 8.0) * 0.5,
                  (1.0 / 8.0) * 0.25 * 0.5, (-1.0 / 8.0) * 0.125 * (1.0 / 6.0), (-2.0 / 16.0)};

  act_paths[2] = {(1.0 / 8.0), (-1.0 / 24.0), 0.0, 0.0, 0.0, 0.0};

  for (int i = 0; i < 4; i++) qdp_sitelink[i] = pinned_malloc(V * gauge_site_size * host_gauge_data_type_size);
  createSiteLinkCPU(qdp_sitelink, gauge_param.cpu_prec, SiteLinkType::SITELINK_PHASE_NO);

  for (int i = 0; i < 4; i++) {
    v_reflink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    w_reflink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    fat_reflink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    long_reflink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    fat_reflink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    long_reflink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  }

  // Build all CPU references with Naik enabled so we get eps links too.
  // When testing without Naik, we just compare against the non-eps references.
  double save_eps = eps_naik;
  int save_n = n_naiks;
  eps_naik = -0.03;
  n_naiks = 2;
  computeHISQLinksCPU(fat_reflink, long_reflink, fat_reflink_eps, long_reflink_eps, v_reflink, w_reflink, qdp_sitelink,
                      &gauge_param, act_paths, eps_naik);
  eps_naik = save_eps;
  n_naiks = save_n;

  host_initialized = true;
}

static void teardown_links()
{
  if (!links_valid) return;

  if (milc_sitelink) { host_free(milc_sitelink); milc_sitelink = nullptr; }
  if (vlink) { host_free(vlink); vlink = nullptr; }
  if (wlink) { host_free(wlink); wlink = nullptr; }
  if (fatlink) { host_free(fatlink); fatlink = nullptr; }
  if (longlink) { host_free(longlink); longlink = nullptr; }
  if (fatlink_eps) { host_free(fatlink_eps); fatlink_eps = nullptr; }
  if (longlink_eps) { host_free(longlink_eps); longlink_eps = nullptr; }

  for (int i = 0; i < 4; i++) {
    if (qdp_vlink[i]) { host_free(qdp_vlink[i]); qdp_vlink[i] = nullptr; }
    if (qdp_wlink[i]) { host_free(qdp_wlink[i]); qdp_wlink[i] = nullptr; }
    if (qdp_fatlink[i]) { host_free(qdp_fatlink[i]); qdp_fatlink[i] = nullptr; }
    if (qdp_longlink[i]) { host_free(qdp_longlink[i]); qdp_longlink[i] = nullptr; }
    if (qdp_fatlink_eps[i]) { host_free(qdp_fatlink_eps[i]); qdp_fatlink_eps[i] = nullptr; }
    if (qdp_longlink_eps[i]) { host_free(qdp_longlink_eps[i]); qdp_longlink_eps[i] = nullptr; }
  }

  freeGaugeQuda();
  links_valid = false;
}

static void setup_links(QudaPrecision test_prec, QudaReconstructType /*recon*/, bool has_naik)
{
  if (links_valid && cur_prec == test_prec && cur_has_naik == has_naik) return;

  teardown_links();

  cur_prec = test_prec;
  cur_has_naik = has_naik;
  cur_n_naiks = has_naik ? 2 : 1;

  gauge_param.cuda_prec = test_prec;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;

  auto cpu_prec_backup = gauge_param.cpu_prec;
  gauge_param.cpu_prec = gauge_param.cuda_prec;

  size_t field_bytes = 4 * V * gauge_site_size * test_prec;

  milc_sitelink = safe_malloc(field_bytes);
  reorderQDPtoMILC(milc_sitelink, qdp_sitelink, V, gauge_site_size, test_prec, cpu_prec_backup);

  vlink = pinned_malloc(field_bytes);
  wlink = pinned_malloc(field_bytes);
  fatlink = pinned_malloc(field_bytes);
  longlink = pinned_malloc(field_bytes);

  if (cur_n_naiks > 1) {
    fatlink_eps = pinned_malloc(field_bytes);
    longlink_eps = pinned_malloc(field_bytes);
  }

  for (int i = 0; i < 4; i++) {
    qdp_vlink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_wlink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_fatlink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_longlink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    if (cur_n_naiks > 1) {
      qdp_fatlink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      qdp_longlink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    }
  }

  // GPU link build: V, W
  computeKSLinkQuda(vlink, nullptr, wlink, milc_sitelink, act_paths[0].data(), &gauge_param);

  // GPU link build: Naik correction
  if (cur_n_naiks > 1) {
    computeKSLinkQuda(fatlink, longlink, nullptr, wlink, act_paths[2].data(), &gauge_param);
    cpu_axy(test_prec, -0.03, fatlink, fatlink_eps, V * 4 * gauge_site_size);
    cpu_axy(test_prec, -0.03, longlink, longlink_eps, V * 4 * gauge_site_size);
  } else {
    memset(fatlink, 0, field_bytes);
    memset(longlink, 0, field_bytes);
  }

  // GPU link build: X + long
  computeKSLinkQuda(fatlink, longlink, nullptr, wlink, act_paths[1].data(), &gauge_param);

  if (cur_n_naiks > 1) {
    cpu_xpy(test_prec, fatlink, fatlink_eps, V * 4 * gauge_site_size);
    cpu_xpy(test_prec, longlink, longlink_eps, V * 4 * gauge_site_size);
  }

  gauge_param.cpu_prec = cpu_prec_backup;

  // Reorder GPU results from MILC to QDP (with precision conversion)
  reorderMILCtoQDP(qdp_vlink, vlink, V, gauge_site_size, gauge_param.cpu_prec, test_prec);
  reorderMILCtoQDP(qdp_wlink, wlink, V, gauge_site_size, gauge_param.cpu_prec, test_prec);
  reorderMILCtoQDP(qdp_fatlink, fatlink, V, gauge_site_size, gauge_param.cpu_prec, test_prec);
  reorderMILCtoQDP(qdp_longlink, longlink, V, gauge_site_size, gauge_param.cpu_prec, test_prec);

  if (cur_n_naiks > 1) {
    reorderMILCtoQDP(qdp_fatlink_eps, fatlink_eps, V, gauge_site_size, gauge_param.cpu_prec, test_prec);
    reorderMILCtoQDP(qdp_longlink_eps, longlink_eps, V, gauge_site_size, gauge_param.cpu_prec, test_prec);
  }

#ifdef MULTI_GPU
  exchange_llfat_cleanup();
#endif

  links_valid = true;
}

static void destroy_host()
{
  teardown_links();

  for (int i = 0; i < 4; i++) {
    if (qdp_sitelink[i]) { host_free(qdp_sitelink[i]); qdp_sitelink[i] = nullptr; }
    if (v_reflink[i]) { host_free(v_reflink[i]); v_reflink[i] = nullptr; }
    if (w_reflink[i]) { host_free(w_reflink[i]); w_reflink[i] = nullptr; }
    if (fat_reflink[i]) { host_free(fat_reflink[i]); fat_reflink[i] = nullptr; }
    if (long_reflink[i]) { host_free(long_reflink[i]); long_reflink[i] = nullptr; }
    if (fat_reflink_eps[i]) { host_free(fat_reflink_eps[i]); fat_reflink_eps[i] = nullptr; }
    if (long_reflink_eps[i]) { host_free(long_reflink_eps[i]); long_reflink_eps[i] = nullptr; }
  }

  host_initialized = false;
}

using test_t = ::testing::tuple<QudaPrecision, QudaReconstructType, bool>;

class HisqEndToEndTest : public ::testing::TestWithParam<test_t>
{
protected:
  QudaPrecision test_prec;
  QudaReconstructType test_recon;
  bool has_naik;

  bool skip()
  {
    if ((QUDA_PRECISION & test_prec) == 0 || (QUDA_RECONSTRUCT & getReconstructNibble(test_recon)) == 0) return true;
    return false;
  }

public:
  void SetUp() override
  {
    test_prec = ::testing::get<0>(GetParam());
    test_recon = ::testing::get<1>(GetParam());
    has_naik = ::testing::get<2>(GetParam());
    if (skip()) GTEST_SKIP();
    init_host();
    setup_links(test_prec, test_recon, has_naik);
  }

  void TearDown() override { }
};

TEST_P(HisqEndToEndTest, link_fat7)
{
  double max_dev = 10.0 * getTolerance(test_prec);

  for (int dir = 0; dir < 4; dir++) {
    double dev = compare_floats_v2(v_reflink[dir], qdp_vlink[dir], V * gauge_site_size, max_dev, gauge_param.cpu_prec);
    EXPECT_LE(dev, max_dev) << "V link (fat7) mismatch in dir " << dir;
  }

  strong_check_link(qdp_vlink, "V link GPU results:", v_reflink, "CPU reference results:", V, gauge_param.cpu_prec);
}

TEST_P(HisqEndToEndTest, link_unitarize)
{
  double max_dev = 10.0 * getTolerance(test_prec);

  for (int dir = 0; dir < 4; dir++) {
    double dev = compare_floats_v2(w_reflink[dir], qdp_wlink[dir], V * gauge_site_size, max_dev, gauge_param.cpu_prec);
    EXPECT_LE(dev, max_dev) << "W link (unitarized) mismatch in dir " << dir;
  }

  strong_check_link(qdp_wlink, "W link GPU results:", w_reflink, "CPU reference results:", V, gauge_param.cpu_prec);
}

TEST_P(HisqEndToEndTest, link_fat_long)
{
  double max_dev = 10.0 * getTolerance(test_prec);

  for (int dir = 0; dir < 4; dir++) {
    double dev
      = compare_floats_v2(fat_reflink[dir], qdp_fatlink[dir], V * gauge_site_size, max_dev, gauge_param.cpu_prec);
    EXPECT_LE(dev, max_dev) << "X link (fat) mismatch in dir " << dir;
  }

  strong_check_link(qdp_fatlink, "X link GPU results:", fat_reflink, "CPU reference results:", V, gauge_param.cpu_prec);

  for (int dir = 0; dir < 4; dir++) {
    double dev
      = compare_floats_v2(long_reflink[dir], qdp_longlink[dir], V * gauge_site_size, max_dev, gauge_param.cpu_prec);
    EXPECT_LE(dev, max_dev) << "Long link mismatch in dir " << dir;
  }

  strong_check_link(qdp_longlink, "Long link GPU results:", long_reflink, "CPU reference results:", V,
                    gauge_param.cpu_prec);
}

TEST_P(HisqEndToEndTest, link_naik)
{
  if (!has_naik) GTEST_SKIP() << "Naik correction not applicable without eps_naik";

  double max_dev = 10.0 * getTolerance(test_prec);

  for (int dir = 0; dir < 4; dir++) {
    double dev = compare_floats_v2(fat_reflink_eps[dir], qdp_fatlink_eps[dir], V * gauge_site_size, max_dev,
                                   gauge_param.cpu_prec);
    EXPECT_LE(dev, max_dev) << "Naik fat link mismatch in dir " << dir;
  }

  strong_check_link(qdp_fatlink_eps, "Naik fat GPU results:", fat_reflink_eps, "CPU reference results:", V,
                    gauge_param.cpu_prec);

  for (int dir = 0; dir < 4; dir++) {
    double dev = compare_floats_v2(long_reflink_eps[dir], qdp_longlink_eps[dir], V * gauge_site_size, max_dev,
                                   gauge_param.cpu_prec);
    EXPECT_LE(dev, max_dev) << "Naik long link mismatch in dir " << dir;
  }

  strong_check_link(qdp_longlink_eps, "Naik long GPU results:", long_reflink_eps, "CPU reference results:", V,
                    gauge_param.cpu_prec);
}

std::string getHisqEndToEndTestName(testing::TestParamInfo<test_t> param)
{
  const QudaPrecision prec = ::testing::get<0>(param.param);
  const QudaReconstructType recon = ::testing::get<1>(param.param);
  const bool has_naik = ::testing::get<2>(param.param);
  std::stringstream ss;
  ss << get_prec_str(prec);
  ss << "_r" << recon;
  if (has_naik) ss << "_naik";
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(QUDA, HisqEndToEndTest,
                         ::testing::Combine(::testing::Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                            ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_13),
                                            ::testing::Bool()),
                         getHisqEndToEndTestName);

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  cpu_prec = prec = QUDA_DOUBLE_PRECISION;

  setVerbosity(verbosity);
  initComms(argc, argv, gridsize_from_cmdline);
  initQuda(device_ordinal);

  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  int test_rc = RUN_ALL_TESTS();
  if (test_rc != 0) warningQuda("Tests failed");

  destroy_host();
  endQuda();
  finalizeComms();

  return test_rc;
}
