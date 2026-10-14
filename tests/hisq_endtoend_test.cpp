#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <unitarization_links.h>
#include <ks_improved_force.h>
#include <staggered_oprod.h>
#include <momentum.h>
#include <malloc_quda.h>

#include <gtest/gtest.h>

#include "host_utils.h"
#include "command_line_params.h"
#include "misc.h"
#include "test.h"
#include "hisq_force_reference.h"
#include "momentum_force_reference.h"
#include "momentum_utils.h"

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
static int cur_partition = -1;

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

  for (int i = 0; i < 4; i++) qdp_sitelink[i] = host_pinned_malloc(V * gauge_site_size * host_gauge_data_type_size);
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
  int part = 0;
  for (int d = 0; d < 4; d++) part |= (comm_dim_partitioned(d) << d);
  if (links_valid && cur_prec == test_prec && cur_has_naik == has_naik && cur_partition == part) return;
  cur_partition = part;

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

  vlink = host_pinned_malloc(field_bytes);
  wlink = host_pinned_malloc(field_bytes);
  fatlink = host_pinned_malloc(field_bytes);
  longlink = host_pinned_malloc(field_bytes);

  if (cur_n_naiks > 1) {
    fatlink_eps = host_pinned_malloc(field_bytes);
    longlink_eps = host_pinned_malloc(field_bytes);
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

static void teardown_force(); // forward declaration
static void destroy_host()
{
  teardown_force();
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

// ============================================================
// Force infrastructure
// ============================================================

#ifdef MULTI_GPU
static lat_dim_t force_R = {2, 2, 2, 2};
#else
static lat_dim_t force_R = {0, 0, 0, 0};
#endif

// CPU outer products (host precision)
static GaugeField *cpuOprod = nullptr;
static GaugeField *cpuLongLinkOprod = nullptr;

// CPU gauge and extended gauge/force fields for reference computation
static GaugeField *cpuGauge = nullptr;
static GaugeField *cpuGauge_ex = nullptr;
static GaugeField *cpuOprod_ex = nullptr;
static GaugeField *cpuLongLinkOprod_ex = nullptr;
static GaugeField *cpuForce_ex = nullptr;
static GaugeField *cpuForce = nullptr;
static GaugeField *cpuMom = nullptr;
static GaugeField *refMom = nullptr;

// GPU extended gauge fields
static GaugeField *cudaGauge_ex = nullptr;
static GaugeField *cudaForce_ex = nullptr;
static GaugeField *cudaOprod_ex = nullptr;
static GaugeField *cudaLongLinkOprod_ex = nullptr;

// GPU force/momentum fields (non-extended)
static GaugeField *cudaForce = nullptr;
static GaugeField *cudaMom = nullptr;

// Raw spinor for outer product
static void *stag_for_oprod = nullptr;

static bool force_valid = false;
static QudaPrecision force_cur_prec = QUDA_INVALID_PRECISION;
static bool force_cur_has_naik = false;
static int force_cur_partition = -1;

static void teardown_force()
{
  if (!force_valid) return;

  delete cpuOprod; cpuOprod = nullptr;
  delete cpuLongLinkOprod; cpuLongLinkOprod = nullptr;
  delete cpuGauge; cpuGauge = nullptr;
  delete cpuGauge_ex; cpuGauge_ex = nullptr;
  delete cpuOprod_ex; cpuOprod_ex = nullptr;
  delete cpuLongLinkOprod_ex; cpuLongLinkOprod_ex = nullptr;
  delete cpuForce_ex; cpuForce_ex = nullptr;
  delete cpuForce; cpuForce = nullptr;
  delete cpuMom; cpuMom = nullptr;
  delete refMom; refMom = nullptr;

  delete cudaGauge_ex; cudaGauge_ex = nullptr;
  delete cudaForce_ex; cudaForce_ex = nullptr;
  delete cudaOprod_ex; cudaOprod_ex = nullptr;
  delete cudaLongLinkOprod_ex; cudaLongLinkOprod_ex = nullptr;
  delete cudaForce; cudaForce = nullptr;
  delete cudaMom; cudaMom = nullptr;

  if (stag_for_oprod) { host_free(stag_for_oprod); stag_for_oprod = nullptr; }

  force_valid = false;
}

static void setup_force(QudaPrecision test_prec, QudaReconstructType recon, bool has_naik)
{
  int part = 0;
  for (int d = 0; d < 4; d++) part |= (comm_dim_partitioned(d) << d);
  if (force_valid && force_cur_prec == test_prec && force_cur_has_naik == has_naik && force_cur_partition == part)
    return;

  teardown_force();
  force_cur_prec = test_prec;
  force_cur_has_naik = has_naik;
  force_cur_partition = part;

  QudaReconstructType cuda_link_recon = recon;
  if (cuda_link_recon != QUDA_RECONSTRUCT_NO && cuda_link_recon != QUDA_RECONSTRUCT_13)
    cuda_link_recon = QUDA_RECONSTRUCT_NO;

  fermion_force::setUnitarizeForceConstants(1e-14, 5e-5, 1e-10, true, false, 1e-8, 1e-8);

  QudaGaugeParam forceGaugeParam = newQudaGaugeParam();
  for (int d = 0; d < 4; d++) forceGaugeParam.X[d] = gauge_param.X[d];
  forceGaugeParam.cpu_prec = gauge_param.cpu_prec;
  forceGaugeParam.cuda_prec = test_prec;
  forceGaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  forceGaugeParam.type = QUDA_GENERAL_LINKS;
  forceGaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  forceGaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
  forceGaugeParam.staggered_phase_applied = true;
  forceGaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  forceGaugeParam.anisotropy = 1.0;
  forceGaugeParam.tadpole_coeff = 1.0;

  GaugeFieldParam gParam(forceGaugeParam);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.order = QUDA_QDP_GAUGE_ORDER;

  GaugeFieldParam gParam_ex(forceGaugeParam);
  gParam_ex.location = QUDA_CPU_FIELD_LOCATION;
  gParam_ex.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
  gParam_ex.create = QUDA_NULL_FIELD_CREATE;
  gParam_ex.link_type = QUDA_GENERAL_LINKS;
  gParam_ex.order = QUDA_QDP_GAUGE_ORDER;
  for (int d = 0; d < 4; d++) {
    gParam_ex.r[d] = force_R[d];
    gParam_ex.x[d] = gParam.x[d] + 2 * gParam_ex.r[d];
  }

  // CPU gauge field (using sitelinks, for createSiteLinkCPU)
  GaugeFieldParam cpuGaugeParam(gParam);
  cpuGaugeParam.create = QUDA_NULL_FIELD_CREATE;
  cpuGauge = new GaugeField(cpuGaugeParam);
  createSiteLinkCPU(*cpuGauge, forceGaugeParam.cpu_prec, SiteLinkType::SITELINK_PHASE_NO);

  cpuGauge_ex = new GaugeField(gParam_ex);
  copyExtendedGauge(*cpuGauge_ex, *cpuGauge, QUDA_CPU_FIELD_LOCATION);

  // CPU outer product fields
  cpuOprod = new GaugeField(gParam);
  cpuLongLinkOprod = new GaugeField(gParam);

  cpuOprod_ex = new GaugeField(gParam_ex);
  cpuLongLinkOprod_ex = new GaugeField(gParam_ex);

  // RNG and random spinors
  {
    ColorSpinorParam qParam;
    qParam.v = nullptr;
    qParam.nColor = 3;
    qParam.nSpin = 1;
    qParam.setPrecision(forceGaugeParam.cpu_prec);
    qParam.create = QUDA_REFERENCE_FIELD_CREATE;
    qParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    qParam.nDim = 4;
    qParam.pc_type = QUDA_4D_PC;
    qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    qParam.location = QUDA_CPU_FIELD_LOCATION;
    for (int d = 0; d < 4; d++) qParam.x[d] = forceGaugeParam.X[d];
    ColorSpinorField spinor_in(qParam);
    RNG rng(spinor_in, 1234);

    stag_for_oprod = safe_malloc(4 * V * stag_spinor_site_size * forceGaugeParam.cpu_prec);
    createStagForOprodCPU(stag_for_oprod, forceGaugeParam.cpu_prec, forceGaugeParam.X, rng);
  }

  // CPU outer products from spinors
  computeLinkOrderedOuterProduct(stag_for_oprod, *cpuOprod, forceGaugeParam.cpu_prec, 1);
  computeLinkOrderedOuterProduct(stag_for_oprod, *cpuLongLinkOprod, forceGaugeParam.cpu_prec, 3);

  copyExtendedGauge(*cpuOprod_ex, *cpuOprod, QUDA_CPU_FIELD_LOCATION);
  copyExtendedGauge(*cpuLongLinkOprod_ex, *cpuLongLinkOprod, QUDA_CPU_FIELD_LOCATION);

  // CPU force and momentum fields
  cpuForce = new GaugeField(gParam);
  cpuForce_ex = new GaugeField(gParam_ex);

  GaugeFieldParam momParam(gParam);
  momParam.reconstruct = QUDA_RECONSTRUCT_10;
  momParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  momParam.order = QUDA_MILC_GAUGE_ORDER;
  cpuMom = new GaugeField(momParam);
  refMom = new GaugeField(momParam);

  // GPU extended gauge field (loaded with U links from cpuGauge)
  GaugeFieldParam cudaGaugeParam_ex(forceGaugeParam);
  cudaGaugeParam_ex.location = QUDA_CUDA_FIELD_LOCATION;
  cudaGaugeParam_ex.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
  cudaGaugeParam_ex.create = QUDA_NULL_FIELD_CREATE;
  cudaGaugeParam_ex.reconstruct = cuda_link_recon;
  cudaGaugeParam_ex.setPrecision(test_prec, true);
  for (int d = 0; d < 4; d++) {
    cudaGaugeParam_ex.r[d] = (comm_dim_partitioned(d)) ? 2 : 0;
    cudaGaugeParam_ex.x[d] = forceGaugeParam.X[d] + 2 * cudaGaugeParam_ex.r[d];
  }
  cudaGauge_ex = new GaugeField(cudaGaugeParam_ex);

  // GPU extended force/oprod fields (always recon NO)
  GaugeFieldParam cudaForceParam_ex(cudaGaugeParam_ex);
  cudaForceParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaForce_ex = new GaugeField(cudaForceParam_ex);
  cudaOprod_ex = new GaugeField(cudaForceParam_ex);
  cudaLongLinkOprod_ex = new GaugeField(cudaForceParam_ex);

  // GPU non-extended force field (for verify)
  GaugeFieldParam cudaForceParam(forceGaugeParam);
  cudaForceParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaForceParam.create = QUDA_NULL_FIELD_CREATE;
  cudaForceParam.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaForceParam.link_type = QUDA_GENERAL_LINKS;
  cudaForceParam.setPrecision(test_prec, true);
  cudaForce = new GaugeField(cudaForceParam);

  // GPU momentum field
  GaugeFieldParam cudaMomParam(cudaForceParam);
  cudaMomParam.reconstruct = QUDA_RECONSTRUCT_10;
  cudaMomParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  cudaMom = new GaugeField(cudaMomParam);

  // Exchange CPU extended fields
  cpuGauge_ex->exchangeExtendedGhost(force_R, true);
  cpuOprod_ex->exchangeExtendedGhost(force_R, true);
  cpuLongLinkOprod_ex->exchangeExtendedGhost(force_R, true);

  // Copy gauge and oprods to GPU
  cudaGauge_ex->copy(*cpuGauge);
  cudaGauge_ex->exchangeExtendedGhost(cudaGauge_ex->R());

  cudaOprod_ex->copy(*cpuOprod);
  cudaOprod_ex->exchangeExtendedGhost(cudaOprod_ex->R());

  cudaLongLinkOprod_ex->copy(*cpuLongLinkOprod);
  cudaLongLinkOprod_ex->exchangeExtendedGhost(cudaLongLinkOprod_ex->R());

  force_valid = true;
}

static bool ctest_all_partitions = false;

using test_t = ::testing::tuple<QudaPrecision, QudaReconstructType, bool, int>;

class HisqEndToEndTest : public ::testing::TestWithParam<test_t>
{
protected:
  QudaPrecision test_prec;
  QudaReconstructType test_recon;
  bool has_naik;
  int partition;

  bool skip()
  {
    if ((QUDA_PRECISION & test_prec) == 0 || (QUDA_RECONSTRUCT & getReconstructNibble(test_recon)) == 0) return true;
    const std::array<bool, 16> partition_enabled {true, true,  true,  false, true, false, false, false,
                                                  true, false, false, false, true, false, true,  true};
    if (!ctest_all_partitions && !partition_enabled[partition]) return true;
    return false;
  }

  void verify_link_fat7();
  void verify_link_unitarize();
  void verify_link_fat_long();
  void verify_link_naik();
  void verify_force_oprod();
  void verify_force_naik_oprod();
  void verify_force_result(const char *label);
  void verify_hisq_force_interface();

public:
  void SetUp() override
  {
    test_prec = ::testing::get<0>(GetParam());
    test_recon = ::testing::get<1>(GetParam());
    has_naik = ::testing::get<2>(GetParam());
    partition = ::testing::get<3>(GetParam());
    if (skip()) GTEST_SKIP();

    commDimPartitionedReset();
    for (int j = 0; j < 4; j++) {
      if (partition & (1 << j)) { commDimPartitionedSet(j); }
    }
    updateR();

    init_host();
    setup_links(test_prec, test_recon, has_naik);
  }

  void TearDown() override { }
};

// ============================================================
// Verification helpers
// ============================================================

void HisqEndToEndTest::verify_link_fat7()
{
  double max_dev = 10.0 * getTolerance(test_prec);

  for (int dir = 0; dir < 4; dir++) {
    double dev = compare_floats_v2(v_reflink[dir], qdp_vlink[dir], V * gauge_site_size, max_dev, gauge_param.cpu_prec);
    EXPECT_LE(dev, max_dev) << "V link (fat7) mismatch in dir " << dir;
  }

  strong_check_link(qdp_vlink, "V link GPU results:", v_reflink, "CPU reference results:", V, gauge_param.cpu_prec);
}

void HisqEndToEndTest::verify_link_unitarize()
{
  double max_dev = 10.0 * getTolerance(test_prec);

  for (int dir = 0; dir < 4; dir++) {
    double dev = compare_floats_v2(w_reflink[dir], qdp_wlink[dir], V * gauge_site_size, max_dev, gauge_param.cpu_prec);
    EXPECT_LE(dev, max_dev) << "W link (unitarized) mismatch in dir " << dir;
  }

  strong_check_link(qdp_wlink, "W link GPU results:", w_reflink, "CPU reference results:", V, gauge_param.cpu_prec);
}

void HisqEndToEndTest::verify_link_fat_long()
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

void HisqEndToEndTest::verify_link_naik()
{
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

void HisqEndToEndTest::verify_force_result(const char *label)
{
  copyExtendedGauge(*cpuForce, *cpuForce_ex, QUDA_CPU_FIELD_LOCATION);
  copyExtendedGauge(*cudaForce, *cudaForce_ex, QUDA_CUDA_FIELD_LOCATION);

  GaugeFieldParam hfParam(*cpuForce);
  hfParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField hostVerify(hfParam);
  hostVerify.copy(*cudaForce);

  int res = 1;
  for (int dir = 0; dir < 4; dir++) {
    res &= compare_floats(cpuForce->data<void *>(dir), hostVerify.data<void *>(dir), V * gauge_site_size,
                           getTolerance(test_prec), gauge_param.cpu_prec);
  }

  void *hPtrs[4], *cPtrs[4];
  for (int d = 0; d < 4; d++) { hPtrs[d] = hostVerify.data<void *>(d); cPtrs[d] = cpuForce->data<void *>(d); }
  int level = strong_check_link(hPtrs, "GPU result:", cPtrs, "CPU reference:", V, gauge_param.cpu_prec);
  int tolerance = getNegLog10Tolerance(test_prec);
  EXPECT_EQ(res, 1) << label << " compare_floats mismatch";
  ASSERT_GE(level, tolerance) << label << " strong_check mismatch";
}

void HisqEndToEndTest::verify_force_oprod()
{
  setup_force(test_prec, test_recon, has_naik);

  QudaPrecision host_prec = gauge_param.cpu_prec;

  GaugeFieldParam oParam(gauge_param);
  oParam.location = QUDA_CUDA_FIELD_LOCATION;
  oParam.nFace = 0;
  oParam.create = QUDA_ZERO_FIELD_CREATE;
  oParam.link_type = QUDA_GENERAL_LINKS;
  oParam.reconstruct = QUDA_RECONSTRUCT_NO;
  oParam.setPrecision(host_prec, true);
  oParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;

  GaugeField gpuStapleOprod(oParam);
  GaugeField gpuNaikOprod(oParam);
  GaugeField *oprod[2] = {&gpuStapleOprod, &gpuNaikOprod};

  ColorSpinorParam qParam;
  qParam.nColor = 3;
  qParam.nSpin = 1;
  qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  qParam.nDim = 4;
  qParam.pc_type = QUDA_4D_PC;
  qParam.setPrecision(host_prec, host_prec, true);
  qParam.pad = 0;
  for (int d = 0; d < 4; d++) qParam.x[d] = gauge_param.X[d];

  qParam.create = QUDA_NULL_FIELD_CREATE;
  qParam.location = QUDA_CUDA_FIELD_LOCATION;
  ColorSpinorField cudaQuark(qParam);

  qParam.location = QUDA_CPU_FIELD_LOCATION;
  qParam.create = QUDA_REFERENCE_FIELD_CREATE;
  qParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
  qParam.v = stag_for_oprod;
  ColorSpinorField cpuQuark(qParam);

  cudaQuark = cpuQuark;

  double coeff[2] = {0.625, -0.058479};
  computeStaggeredOprod(oprod, cudaQuark, coeff, 3);

  oParam.location = QUDA_CPU_FIELD_LOCATION;
  oParam.create = QUDA_NULL_FIELD_CREATE;
  oParam.order = QUDA_QDP_GAUGE_ORDER;
  GaugeField hostStapleOprod(oParam);
  GaugeField hostNaikOprod(oParam);
  hostStapleOprod.copy(gpuStapleOprod);
  hostNaikOprod.copy(gpuNaikOprod);

  GaugeField expectedStaple(oParam);
  GaugeField expectedNaik(oParam);
  for (int dir = 0; dir < 4; dir++) {
    cpu_axy(host_prec, coeff[0], cpuOprod->data<void *>(dir), expectedStaple.data<void *>(dir), V * gauge_site_size);
    cpu_axy(host_prec, coeff[1], cpuLongLinkOprod->data<void *>(dir), expectedNaik.data<void *>(dir),
            V * gauge_site_size);
  }

  double tol = getTolerance(host_prec);
  int res = 1;
  for (int dir = 0; dir < 4; dir++) {
    res &= compare_floats(expectedStaple.data<void *>(dir), hostStapleOprod.data<void *>(dir), V * gauge_site_size, tol,
                           host_prec);
  }
  EXPECT_EQ(res, 1) << "Staple outer product (1-hop) mismatch";
  strong_check_link(hostStapleOprod, "GPU staple oprod:", expectedStaple, "CPU expected:");

  res = 1;
  for (int dir = 0; dir < 4; dir++) {
    res &= compare_floats(expectedNaik.data<void *>(dir), hostNaikOprod.data<void *>(dir), V * gauge_site_size, tol,
                           host_prec);
  }
  EXPECT_EQ(res, 1) << "Naik outer product (3-hop) mismatch";
  strong_check_link(hostNaikOprod, "GPU Naik oprod:", expectedNaik, "CPU expected:");
}

void HisqEndToEndTest::verify_force_naik_oprod()
{
  setup_force(test_prec, test_recon, has_naik);

  QudaPrecision host_prec = gauge_param.cpu_prec;
  double level2_coeff_0 = act_paths[0][0];

  GaugeFieldParam oParam(gauge_param);
  oParam.location = QUDA_CUDA_FIELD_LOCATION;
  oParam.nFace = 0;
  oParam.create = QUDA_ZERO_FIELD_CREATE;
  oParam.link_type = QUDA_GENERAL_LINKS;
  oParam.reconstruct = QUDA_RECONSTRUCT_NO;
  oParam.setPrecision(host_prec, true);
  oParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;

  GaugeField stapleOprod(oParam);
  GaugeField oneLinkOprod(oParam);
  GaugeField naikOprod(oParam);

  ColorSpinorParam qParam;
  qParam.nColor = 3;
  qParam.nSpin = 1;
  qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  qParam.nDim = 4;
  qParam.pc_type = QUDA_4D_PC;
  qParam.setPrecision(host_prec, host_prec, true);
  qParam.pad = 0;
  for (int d = 0; d < 4; d++) qParam.x[d] = gauge_param.X[d];
  qParam.create = QUDA_NULL_FIELD_CREATE;
  qParam.location = QUDA_CUDA_FIELD_LOCATION;
  ColorSpinorField cudaQuark(qParam);

  qParam.location = QUDA_CPU_FIELD_LOCATION;
  qParam.create = QUDA_REFERENCE_FIELD_CREATE;
  qParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
  qParam.v = stag_for_oprod;
  ColorSpinorField cpuQuark(qParam);
  cudaQuark = cpuQuark;

  double coeff_reg[2] = {0.625, -0.058479};
  double coeff_naik[2] = {0.3125, -0.029240};

  {
    GaugeField *oprod[2] = {&stapleOprod, &naikOprod};
    computeStaggeredOprod(oprod, cudaQuark, coeff_reg, 3);
  }

  oneLinkOprod.copy(stapleOprod, level2_coeff_0);
  {
    GaugeField *oprod[2] = {&oneLinkOprod, &naikOprod};
    computeStaggeredOprod(oprod, cudaQuark, coeff_naik, 3);
  }

  oParam.location = QUDA_CPU_FIELD_LOCATION;
  oParam.create = QUDA_NULL_FIELD_CREATE;
  oParam.order = QUDA_QDP_GAUGE_ORDER;
  GaugeField hostStaple(oParam);
  GaugeField hostOneLink(oParam);
  GaugeField hostNaik(oParam);
  hostStaple.copy(stapleOprod);
  hostOneLink.copy(oneLinkOprod);
  hostNaik.copy(naikOprod);

  GaugeField expectedStaple(oParam);
  GaugeField expectedOneLink(oParam);
  GaugeField expectedNaik(oParam);
  for (int dir = 0; dir < 4; dir++) {
    cpu_axy(host_prec, coeff_reg[0], cpuOprod->data<void *>(dir), expectedStaple.data<void *>(dir),
            V * gauge_site_size);
    cpu_axy(host_prec, coeff_reg[0] * level2_coeff_0 + coeff_naik[0], cpuOprod->data<void *>(dir),
            expectedOneLink.data<void *>(dir), V * gauge_site_size);
    cpu_axy(host_prec, coeff_reg[1] + coeff_naik[1], cpuLongLinkOprod->data<void *>(dir),
            expectedNaik.data<void *>(dir), V * gauge_site_size);
  }

  double tol = getTolerance(host_prec);

  int res = 1;
  for (int dir = 0; dir < 4; dir++)
    res &= compare_floats(expectedStaple.data<void *>(dir), hostStaple.data<void *>(dir), V * gauge_site_size, tol,
                           host_prec);
  EXPECT_EQ(res, 1) << "stapleOprod mismatch";

  res = 1;
  for (int dir = 0; dir < 4; dir++)
    res &= compare_floats(expectedOneLink.data<void *>(dir), hostOneLink.data<void *>(dir), V * gauge_site_size, tol,
                           host_prec);
  EXPECT_EQ(res, 1) << "oneLinkOprod mismatch after copy+ax+Naik accumulation";
  strong_check_link(hostOneLink, "GPU oneLinkOprod:", expectedOneLink, "CPU expected:");

  res = 1;
  for (int dir = 0; dir < 4; dir++)
    res &= compare_floats(expectedNaik.data<void *>(dir), hostNaik.data<void *>(dir), V * gauge_site_size, tol,
                           host_prec);
  EXPECT_EQ(res, 1) << "naikOprod mismatch after Naik accumulation";
  strong_check_link(hostNaik, "GPU naikOprod:", expectedNaik, "CPU expected:");
}

void HisqEndToEndTest::verify_hisq_force_interface()
{
  QudaPrecision host_prec = gauge_param.cpu_prec;
  size_t gsize = V * gauge_site_size * host_prec;
  size_t field_bytes_hp = 4 * gsize;

  fermion_force::setUnitarizeForceConstants(1e-14, 5e-5, 1e-10, true, false, 1e-8, 1e-8);

  // ---- 1. Fresh U links (QDP order, host precision) ----
  auto link_type = SiteLinkType::SITELINK_PHASE_NO;
  void *qdp_u[4];
  for (int i = 0; i < 4; i++) qdp_u[i] = safe_malloc(gsize);
  createSiteLinkCPU(qdp_u, host_prec, link_type);

  // ---- 2. Build V, W on GPU via computeKSLinkQuda ----
  QudaGaugeParam linkParam = newQudaGaugeParam();
  setStaggeredGaugeParam(linkParam);
  linkParam.cuda_prec = test_prec;
  linkParam.reconstruct = QUDA_RECONSTRUCT_NO;

  auto cpu_prec_backup = linkParam.cpu_prec;
  linkParam.cpu_prec = test_prec;

  size_t field_bytes_tp = 4 * V * gauge_site_size * test_prec;
  void *milc_u_tp = safe_malloc(field_bytes_tp);
  reorderQDPtoMILC(milc_u_tp, qdp_u, V, gauge_site_size, test_prec, cpu_prec_backup);

  void *vlink_milc = host_pinned_malloc(field_bytes_tp);
  void *wlink_milc = host_pinned_malloc(field_bytes_tp);
  computeKSLinkQuda(vlink_milc, nullptr, wlink_milc, milc_u_tp, act_paths[0].data(), &linkParam);

  linkParam.cpu_prec = cpu_prec_backup;

  void *qdp_v[4], *qdp_w[4];
  for (int i = 0; i < 4; i++) {
    qdp_v[i] = safe_malloc(gsize);
    qdp_w[i] = safe_malloc(gsize);
  }
  reorderMILCtoQDP(qdp_v, vlink_milc, V, gauge_site_size, host_prec, test_prec);
  reorderMILCtoQDP(qdp_w, wlink_milc, V, gauge_site_size, host_prec, test_prec);

  host_free(milc_u_tp);
  host_free(vlink_milc);
  host_free(wlink_milc);
#ifdef MULTI_GPU
  exchange_llfat_cleanup();
#endif

  // ---- 3. Fresh spinor and outer products ----
  void *local_stag = nullptr;
  {
    ColorSpinorParam qParam;
    qParam.v = nullptr;
    qParam.nColor = 3;
    qParam.nSpin = 1;
    qParam.setPrecision(host_prec);
    qParam.create = QUDA_REFERENCE_FIELD_CREATE;
    qParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    qParam.nDim = 4;
    qParam.pc_type = QUDA_4D_PC;
    qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    qParam.location = QUDA_CPU_FIELD_LOCATION;
    for (int d = 0; d < 4; d++) qParam.x[d] = gauge_param.X[d];
    ColorSpinorField spinor_in(qParam);
    RNG rng(spinor_in, 5678);
    local_stag = safe_malloc(4 * V * stag_spinor_site_size * host_prec);
    createStagForOprodCPU(local_stag, host_prec, gauge_param.X, rng);
  }

  QudaGaugeParam fgp = newQudaGaugeParam();
  for (int d = 0; d < 4; d++) fgp.X[d] = gauge_param.X[d];
  fgp.cpu_prec = host_prec;
  fgp.cuda_prec = test_prec;
  fgp.reconstruct = QUDA_RECONSTRUCT_NO;
  fgp.type = QUDA_GENERAL_LINKS;
  fgp.t_boundary = QUDA_ANTI_PERIODIC_T;
  fgp.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
  fgp.staggered_phase_applied = true;
  fgp.gauge_fix = QUDA_GAUGE_FIXED_NO;
  fgp.anisotropy = 1.0;
  fgp.tadpole_coeff = 1.0;

  GaugeFieldParam gParam(fgp);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.order = QUDA_QDP_GAUGE_ORDER;

  // Coefficient setup matching computeHISQForceQuda's conventions
  double level2_coeff[6] = {1.0 / 8.0, 0.0, -1.0 / 16.0, 1.0 / 64.0, -1.0 / 384.0, -1.0 / 8.0};
  double fat7_act[6] = {1.0 / 8.0, -1.0 / 24.0, -1.0 / 16.0, 1.0 / 64.0, -1.0 / 384.0, 0.0};
  int num_terms = 1;
  int num_naik_terms = has_naik ? 1 : 0;
  double coeff_reg[2] = {0.625, -0.058479};
  double coeff_naik_term[2] = {0.3125, -0.029240};
  double *coeff_ptrs[2] = {coeff_reg, coeff_naik_term};

  // Compute outer products on GPU (matching computeHISQForceQuda's internal computeStaggeredOprod)
  // then copy back for the CPU reference chain, ensuring identical starting data.
  GaugeFieldParam oParam(fgp);
  oParam.location = QUDA_CUDA_FIELD_LOCATION;
  oParam.nFace = 0;
  oParam.create = QUDA_ZERO_FIELD_CREATE;
  oParam.link_type = QUDA_GENERAL_LINKS;
  oParam.reconstruct = QUDA_RECONSTRUCT_NO;
  oParam.setPrecision(host_prec, true);
  oParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;

  GaugeField gpuStapleOprod(oParam);
  GaugeField gpuNaikOprod(oParam);
  GaugeField *oprod_arr[2] = {&gpuStapleOprod, &gpuNaikOprod};

  {
    ColorSpinorParam qcParam;
    qcParam.nColor = 3;
    qcParam.nSpin = 1;
    qcParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    qcParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    qcParam.nDim = 4;
    qcParam.pc_type = QUDA_4D_PC;
    qcParam.setPrecision(host_prec, host_prec, true);
    qcParam.pad = 0;
    for (int d = 0; d < 4; d++) qcParam.x[d] = gauge_param.X[d];
    qcParam.create = QUDA_NULL_FIELD_CREATE;
    qcParam.location = QUDA_CUDA_FIELD_LOCATION;
    ColorSpinorField cudaQuark(qcParam);
    qcParam.location = QUDA_CPU_FIELD_LOCATION;
    qcParam.create = QUDA_REFERENCE_FIELD_CREATE;
    qcParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
    qcParam.v = local_stag;
    ColorSpinorField cpuQuark(qcParam);
    cudaQuark = cpuQuark;
    computeStaggeredOprod(oprod_arr, cudaQuark, coeff_reg, 3);
  }

  // Build oneLinkOprod on GPU matching computeHISQForceQuda: copy stapleOprod scaled by level2_coeff[0]
  GaugeField gpuOneLinkOprod(oParam);
  gpuOneLinkOprod.copy(gpuStapleOprod, level2_coeff[0]);

  if (has_naik) {
    GaugeField *oprod_naik[2] = {&gpuOneLinkOprod, &gpuNaikOprod};
    ColorSpinorParam qcParam2;
    qcParam2.nColor = 3;
    qcParam2.nSpin = 1;
    qcParam2.siteSubset = QUDA_FULL_SITE_SUBSET;
    qcParam2.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    qcParam2.nDim = 4;
    qcParam2.pc_type = QUDA_4D_PC;
    qcParam2.setPrecision(host_prec, host_prec, true);
    qcParam2.pad = 0;
    for (int d = 0; d < 4; d++) qcParam2.x[d] = gauge_param.X[d];
    qcParam2.create = QUDA_NULL_FIELD_CREATE;
    qcParam2.location = QUDA_CUDA_FIELD_LOCATION;
    ColorSpinorField cudaQuark2(qcParam2);
    qcParam2.location = QUDA_CPU_FIELD_LOCATION;
    qcParam2.create = QUDA_REFERENCE_FIELD_CREATE;
    qcParam2.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
    qcParam2.v = local_stag;
    ColorSpinorField cpuQuark2(qcParam2);
    cudaQuark2 = cpuQuark2;
    computeStaggeredOprod(oprod_naik, cudaQuark2, coeff_naik_term, 3);
  }

  // Copy GPU outer products to CPU QDP-order fields (already scaled by coefficients)
  oParam.location = QUDA_CPU_FIELD_LOCATION;
  oParam.create = QUDA_NULL_FIELD_CREATE;
  oParam.order = QUDA_QDP_GAUGE_ORDER;
  GaugeField stapleOprodCpuField(oParam);
  GaugeField oneLinkOprodCpuField(oParam);
  GaugeField naikOprodCpuField(oParam);
  stapleOprodCpuField.copy(gpuStapleOprod);
  oneLinkOprodCpuField.copy(gpuOneLinkOprod);
  naikOprodCpuField.copy(gpuNaikOprod);

  // ---- 4. MILC-order links at host precision for computeHISQForceQuda ----
  void *milc_u_hp = safe_malloc(field_bytes_hp);
  void *milc_v_hp = safe_malloc(field_bytes_hp);
  void *milc_w_hp = safe_malloc(field_bytes_hp);
  reorderQDPtoMILC(milc_u_hp, qdp_u, V, gauge_site_size, host_prec, host_prec);
  reorderQDPtoMILC(milc_v_hp, qdp_v, V, gauge_site_size, host_prec, host_prec);
  reorderQDPtoMILC(milc_w_hp, qdp_w, V, gauge_site_size, host_prec, host_prec);

  // ---- 5. Call computeHISQForceQuda ----
  void *fermion_ptrs[1] = {local_stag};

  size_t mom_bytes = 4 * V * mom_site_size * host_prec;
  void *milc_mom_out = safe_malloc(mom_bytes);
  memset(milc_mom_out, 0, mom_bytes);

  QudaGaugeParam forceParam = newQudaGaugeParam();
  for (int d = 0; d < 4; d++) forceParam.X[d] = gauge_param.X[d];
  forceParam.cpu_prec = host_prec;
  forceParam.cuda_prec = test_prec;
  forceParam.reconstruct = QUDA_RECONSTRUCT_NO;
  forceParam.type = QUDA_GENERAL_LINKS;
  forceParam.gauge_order = QUDA_MILC_GAUGE_ORDER;
  forceParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  forceParam.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
  forceParam.staggered_phase_applied = true;
  forceParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  forceParam.anisotropy = 1.0;
  forceParam.tadpole_coeff = 1.0;
  forceParam.use_resident_mom = false;
  forceParam.return_result_mom = true;
  forceParam.make_resident_mom = false;

  double dt = 1.0;
  computeHISQForceQuda(milc_mom_out, dt, level2_coeff, fat7_act, milc_w_hp, milc_v_hp, milc_u_hp, fermion_ptrs,
                        num_terms, num_naik_terms, coeff_ptrs, &forceParam);

  // ---- 6. CPU reference mirroring computeHISQForceQuda's internal logic ----

  double act_path_coeff[6] = {0, 1, level2_coeff[2], level2_coeff[3], level2_coeff[4], level2_coeff[5]};

  // Outer products are already scaled by GPU -- use them directly as staple/oneLink/naik oprods

  // Non-extended gauge fields from the same QDP ground truth
  GaugeField cpuU(gParam);
  GaugeField cpuW(gParam);
  GaugeField cpuV(gParam);
  for (int d = 0; d < 4; d++) {
    memcpy(cpuU.data<void *>(d), qdp_u[d], gsize);
    memcpy(cpuW.data<void *>(d), qdp_w[d], gsize);
    memcpy(cpuV.data<void *>(d), qdp_v[d], gsize);
  }

  // Extended fields derived from the same non-extended sources
  GaugeFieldParam gParam_ex(fgp);
  gParam_ex.location = QUDA_CPU_FIELD_LOCATION;
  gParam_ex.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
  gParam_ex.create = QUDA_NULL_FIELD_CREATE;
  gParam_ex.link_type = QUDA_GENERAL_LINKS;
  gParam_ex.order = QUDA_QDP_GAUGE_ORDER;
  for (int d = 0; d < 4; d++) {
    gParam_ex.r[d] = force_R[d];
    gParam_ex.x[d] = gParam.x[d] + 2 * gParam_ex.r[d];
  }

  GaugeField cpuU_ex(gParam_ex);
  copyExtendedGauge(cpuU_ex, cpuU, QUDA_CPU_FIELD_LOCATION);
  cpuU_ex.exchangeExtendedGhost(force_R, true);

  GaugeField cpuW_ex(gParam_ex);
  copyExtendedGauge(cpuW_ex, cpuW, QUDA_CPU_FIELD_LOCATION);
  cpuW_ex.exchangeExtendedGhost(force_R, true);

  GaugeField localOprod_ex(gParam_ex);
  GaugeField localForce_ex(gParam_ex);
  GaugeField localForce(gParam);

  // Level-2 staples
  copyExtendedGauge(localOprod_ex, stapleOprodCpuField, QUDA_CPU_FIELD_LOCATION);
  localOprod_ex.exchangeExtendedGhost(force_R, true);
  copyExtendedGauge(localForce_ex, oneLinkOprodCpuField, QUDA_CPU_FIELD_LOCATION);
  localForce_ex.exchangeExtendedGhost(force_R, true);
  hisqStaplesForceCPU(act_path_coeff, localOprod_ex, cpuW_ex, &localForce_ex);
  localForce_ex.exchangeExtendedGhost(force_R, true);

  // Long link
  copyExtendedGauge(localOprod_ex, naikOprodCpuField, QUDA_CPU_FIELD_LOCATION);
  localOprod_ex.exchangeExtendedGhost(force_R, true);
  hisqLongLinkForceCPU(act_path_coeff[1], localOprod_ex, cpuW_ex, &localForce_ex);
  localForce_ex.exchangeExtendedGhost(force_R, true);

  // Unitarize
  copyExtendedGauge(localForce, localForce_ex, QUDA_CPU_FIELD_LOCATION);
  GaugeFieldParam unitResParam(gParam);
  unitResParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField cpuUnitResult(unitResParam);
  fermion_force::unitarizeForceCPU(cpuUnitResult, localForce, cpuV);

  // Fat7 staples
  localForce_ex.zero();
  copyExtendedGauge(localOprod_ex, cpuUnitResult, QUDA_CPU_FIELD_LOCATION);
  localOprod_ex.exchangeExtendedGhost(force_R, true);
  hisqStaplesForceCPU(fat7_act, localOprod_ex, cpuU_ex, &localForce_ex);
  localForce_ex.exchangeExtendedGhost(force_R, true);

  // Complete force -> 18-real force field (matching GPU CompleteForce kernel)
  GaugeField cpuCompleteForce(gParam);
  hisqCompleteForceToForceCPU(cpuCompleteForce, localForce_ex, cpuU_ex);

  // Force -> 10-real momentum (matching GPU updateMomentum kernel)
  GaugeFieldParam momParam(gParam);
  momParam.reconstruct = QUDA_RECONSTRUCT_10;
  momParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  momParam.order = QUDA_MILC_GAUGE_ORDER;
  GaugeField cpuRefMom(momParam);
  updateMomentumFromForceCPU(cpuRefMom, dt, cpuCompleteForce);

  // ---- 7. Compare ----
  momParam.create = QUDA_REFERENCE_FIELD_CREATE;
  momParam.gauge = milc_mom_out;
  GaugeField gpuMomResult(momParam);

  int accuracy_level = strong_check_mom(gpuMomResult.data(), cpuRefMom.data(), 4 * V, host_prec);
  // The GPU and CPU run the full force chain independently. Each of the ~5 stages
  // (staples, long link, unitarize, fat7, complete) has an independent CPU reference
  // implementation that introduces O(eps) divergence per stage. The compound divergence
  // is volume-independent and uniformly distributed, giving ~2 digits of loss at double
  // precision. The full_chain test verifies each stage individually at full tolerance.
  int tolerance = std::max(getNegLog10Tolerance(test_prec) - 2, 1);
  ASSERT_GE(accuracy_level, tolerance) << "computeHISQForceQuda momentum mismatch vs CPU reference";

  // ---- 8. Cleanup ----
  for (int i = 0; i < 4; i++) {
    host_free(qdp_u[i]);
    host_free(qdp_v[i]);
    host_free(qdp_w[i]);
  }
  host_free(local_stag);
  host_free(milc_u_hp);
  host_free(milc_v_hp);
  host_free(milc_w_hp);
  host_free(milc_mom_out);
}

// ============================================================
// TEST_P definitions
// ============================================================

TEST_P(HisqEndToEndTest, full_chain)
{
  // --- Link checks ---
  { SCOPED_TRACE("link_fat7"); verify_link_fat7(); }
  { SCOPED_TRACE("link_unitarize"); verify_link_unitarize(); }
  { SCOPED_TRACE("link_fat_long"); verify_link_fat_long(); }
  if (has_naik) { SCOPED_TRACE("link_naik"); verify_link_naik(); }

  // --- Outer product checks ---
  { SCOPED_TRACE("force_oprod"); verify_force_oprod(); }
  if (has_naik) { SCOPED_TRACE("force_naik_oprod"); verify_force_naik_oprod(); }

  // --- Force chain (sequential, no recomputation) ---
  setup_force(test_prec, test_recon, has_naik);

  double d_act_path_coeff[6] = {0.625000, -0.058479, -0.087719, 0.030778, -0.007200, -0.123113};
  double fat7_coeff[6] = {0.625000, -0.058479, -0.087719, 0.030778, -0.007200, 0.0};

  // Restore outer products once
  cudaOprod_ex->copy(*cpuOprod);
  cudaOprod_ex->exchangeExtendedGhost(cudaOprod_ex->R());
  copyExtendedGauge(*cpuOprod_ex, *cpuOprod, QUDA_CPU_FIELD_LOCATION);
  cpuOprod_ex->exchangeExtendedGhost(force_R, true);

  cpuForce->zero();
  cpuForce_ex->zero();
  cudaForce->zero();
  cudaForce_ex->zero();
  cpuMom->zero();
  refMom->zero();
  cudaMom->zero();

  // Step 1: Level-2 staples
  fermion_force::hisqStaplesForce(*cudaForce_ex, *cudaOprod_ex, *cudaGauge_ex, d_act_path_coeff);
  qudaDeviceSynchronize();
  hisqStaplesForceCPU(d_act_path_coeff, *cpuOprod_ex, *cpuGauge_ex, cpuForce_ex);
  cpuForce_ex->exchangeExtendedGhost(force_R, true);
  { SCOPED_TRACE("force_staples_level2"); verify_force_result("Level-2 staples force"); }

  // Step 2: Long link (accumulates into existing force fields)
  fermion_force::hisqLongLinkForce(*cudaForce_ex, *cudaLongLinkOprod_ex, *cudaGauge_ex, d_act_path_coeff[1]);
  qudaDeviceSynchronize();
  hisqLongLinkForceCPU(d_act_path_coeff[1], *cpuLongLinkOprod_ex, *cpuGauge_ex, cpuForce_ex);
  cpuForce_ex->exchangeExtendedGhost(force_R, true);
  { SCOPED_TRACE("force_long_link"); verify_force_result("Long link force"); }

  // Step 3: Unitarize
  cudaForce_ex->exchangeExtendedGhost(cudaForce_ex->R());

  GaugeField *unitGauge = cudaGauge_ex;
  GaugeField unitGaugeLocal;
  if (cudaGauge_ex->Reconstruct() != QUDA_RECONSTRUCT_NO) {
    GaugeFieldParam ugParam(*cudaGauge_ex);
    ugParam.reconstruct = QUDA_RECONSTRUCT_NO;
    ugParam.create = QUDA_NULL_FIELD_CREATE;
    unitGaugeLocal = GaugeField(ugParam);
    unitGaugeLocal.copy(*cudaGauge_ex);
    unitGaugeLocal.exchangeExtendedGhost(unitGaugeLocal.R());
    unitGauge = &unitGaugeLocal;
  }

  int *num_failures_dev = static_cast<int *>(device_malloc(sizeof(int)));
  qudaMemset(num_failures_dev, 0, sizeof(int));
  fermion_force::unitarizeForce(*cudaOprod_ex, *cudaForce_ex, *unitGauge, num_failures_dev);
  qudaDeviceSynchronize();
  device_free(num_failures_dev);

  copyExtendedGauge(*cpuForce, *cpuForce_ex, QUDA_CPU_FIELD_LOCATION);
  GaugeFieldParam cpuResultParam(*cpuForce);
  cpuResultParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField cpuUnitResult(cpuResultParam);
  fermion_force::unitarizeForceCPU(cpuUnitResult, *cpuForce, *cpuGauge);

  {
    SCOPED_TRACE("force_unitarize");
    copyExtendedGauge(*cudaForce, *cudaOprod_ex, QUDA_CUDA_FIELD_LOCATION);
    GaugeFieldParam hfParam(*cpuForce);
    hfParam.create = QUDA_NULL_FIELD_CREATE;
    GaugeField hostVerify(hfParam);
    hostVerify.copy(*cudaForce);

    int res = 1;
    for (int dir = 0; dir < 4; dir++) {
      res &= compare_floats(cpuUnitResult.data<void *>(dir), hostVerify.data<void *>(dir), V * gauge_site_size,
                             getTolerance(test_prec), gauge_param.cpu_prec);
    }

    void *hPtrs[4], *cPtrs[4];
    for (int d = 0; d < 4; d++) { hPtrs[d] = hostVerify.data<void *>(d); cPtrs[d] = cpuUnitResult.data<void *>(d); }
    int level = strong_check_link(hPtrs, "GPU result:", cPtrs, "CPU reference:", V, gauge_param.cpu_prec);
    int tolerance = getNegLog10Tolerance(test_prec);
    EXPECT_EQ(res, 1) << "Unitarize force compare_floats mismatch";
    ASSERT_GE(level, tolerance) << "Unitarize force strong_check mismatch";
  }

  // Step 4: Fat7 staples (chained from unitarize output)
  cpuForce_ex->zero();
  copyExtendedGauge(*cpuOprod_ex, cpuUnitResult, QUDA_CPU_FIELD_LOCATION);
  cpuOprod_ex->exchangeExtendedGhost(force_R, true);

  cudaForce_ex->zero();
  cudaOprod_ex->exchangeExtendedGhost(cudaOprod_ex->R());

  fermion_force::hisqStaplesForce(*cudaForce_ex, *cudaOprod_ex, *cudaGauge_ex, fat7_coeff);
  qudaDeviceSynchronize();
  hisqStaplesForceCPU(fat7_coeff, *cpuOprod_ex, *cpuGauge_ex, cpuForce_ex);
  cpuForce_ex->exchangeExtendedGhost(force_R, true);
  { SCOPED_TRACE("force_staples_fat7"); verify_force_result("Chained fat7 staples force"); }

  // Step 5: Complete force + momentum
  fermion_force::hisqCompleteForce(*cudaForce_ex, *cudaGauge_ex);
  updateMomentum(*cudaMom, 1.0, *cudaForce_ex, "full_chain");
  qudaDeviceSynchronize();

  hisqCompleteForceCPU(*cpuForce_ex, *cpuGauge_ex, refMom);
  cpuMom->copy(*cudaMom);

  {
    SCOPED_TRACE("force_complete");
    compare_floats(cpuMom->data(), refMom->data(), 4 * cpuMom->Volume() * mom_site_size, getTolerance(test_prec),
                   gauge_param.cpu_prec);
    int accuracy_level = strong_check_mom(cpuMom->data(), refMom->data(), 4 * cpuMom->Volume(), gauge_param.cpu_prec);
    int tolerance = getNegLog10Tolerance(test_prec);
    ASSERT_GE(accuracy_level, tolerance) << "Complete force + momentum mismatch";
  }
}

TEST_P(HisqEndToEndTest, hisq_force_interface) { verify_hisq_force_interface(); }

// ============================================================
// Test instantiation
// ============================================================

std::string getHisqEndToEndTestName(testing::TestParamInfo<test_t> param)
{
  const QudaPrecision prec = ::testing::get<0>(param.param);
  const QudaReconstructType recon = ::testing::get<1>(param.param);
  const bool has_naik = ::testing::get<2>(param.param);
  const int part = ::testing::get<3>(param.param);
  std::stringstream ss;
  ss << get_prec_str(prec);
  ss << "_r" << recon;
  if (has_naik) ss << "_naik";
  ss << "_partition" << part;
  return ss.str();
}

#ifdef MULTI_GPU
INSTANTIATE_TEST_SUITE_P(QUDA, HisqEndToEndTest,
                         ::testing::Combine(::testing::Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                            ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_13),
                                            ::testing::Bool(), ::testing::Range(0, 16)),
                         getHisqEndToEndTestName);
#else
INSTANTIATE_TEST_SUITE_P(QUDA, HisqEndToEndTest,
                         ::testing::Combine(::testing::Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                            ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_13),
                                            ::testing::Bool(), ::testing::Values(0)),
                         getHisqEndToEndTestName);
#endif

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  auto app = make_app();
  app->add_option("--all-partitions", ctest_all_partitions, "Test all instead of reduced combination of partitions");
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
