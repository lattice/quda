#pragma once

#include <array>
#include <utility>
#include <vector>

#include <quda.h>
#include <comm_quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <staggered_oprod.h>
#include <timer.h>

#include <gtest/gtest.h>

#include "host_utils.h"
#include "command_line_params.h"
#include "misc.h"
#include "hisq_force_reference.h"

using namespace quda;

/**
 * @brief Coefficients for the outer product of the k-th quark field: distinct and sign-alternating across fields, with
 *        a smaller three-hop (Naik-like) term, which is only used for nFace = 3
 */
inline std::array<double, 2> oprod_coeff(int k)
{
  const double s = (k % 2 == 0) ? 1.0 : -1.0;
  return {s * (0.5 + 0.25 * k), -s * 0.02 * (k + 1)};
}

struct StaggeredOprodTestWrapper {

  QudaGaugeParam gauge_param;
  QudaPrecision test_prec = QUDA_INVALID_PRECISION;
  int nFace = 3;

  // host quark fields (host precision) and their device copies (test precision)
  std::vector<ColorSpinorField> cpuQuark;
  std::vector<ColorSpinorField> cudaQuark;

  // entry 1 (three-hop) stays empty for nFace = 1
  std::array<GaugeField, 2> gpuOprod;

  int n_out() const { return nFace == 3 ? 2 : 1; }

  void init_ctest(QudaPrecision prec_, int nFace_, int nsrc)
  {
    test_prec = prec_;
    nFace = nFace_;
    Nsrc = nsrc;
    init();
  }

  void init_test()
  {
    test_prec = prec;
    switch (dslash_type) {
    case QUDA_STAGGERED_DSLASH: nFace = 1; break;
    case QUDA_ASQTAD_DSLASH: nFace = 3; break;
    default: errorQuda("Unsupported dslash type %d, expected staggered or asqtad", dslash_type);
    }
    init();
  }

  void init()
  {
    if (Nsrc < 1) errorQuda("Nsrc = %d must be at least 1", Nsrc);

    gauge_param = newQudaGaugeParam();
    setStaggeredGaugeParam(gauge_param);
    setDims(gauge_param.X);

    ColorSpinorParam qParam;
    qParam.nColor = 3;
    qParam.nSpin = 1;
    qParam.setPrecision(gauge_param.cpu_prec);
    qParam.create = QUDA_NULL_FIELD_CREATE;
    qParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    qParam.nDim = 4;
    qParam.pc_type = QUDA_4D_PC;
    qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    qParam.location = QUDA_CPU_FIELD_LOCATION;
    for (int d = 0; d < 4; d++) qParam.x[d] = gauge_param.X[d];

    cpuQuark.reserve(Nsrc);
    for (int i = 0; i < Nsrc; i++) cpuQuark.emplace_back(qParam);
    RNG rng(cpuQuark[0], 1234);
    for (auto &q : cpuQuark) createStagForOprodCPU(q, rng);

    qParam.location = QUDA_CUDA_FIELD_LOCATION;
    qParam.setPrecision(test_prec, test_prec, true);
    cudaQuark.reserve(Nsrc);
    for (int i = 0; i < Nsrc; i++) {
      cudaQuark.emplace_back(qParam);
      cudaQuark[i] = cpuQuark[i];
    }

    GaugeFieldParam oParam(gauge_param);
    oParam.location = QUDA_CUDA_FIELD_LOCATION;
    oParam.nFace = 0;
    oParam.create = QUDA_ZERO_FIELD_CREATE;
    oParam.link_type = QUDA_GENERAL_LINKS;
    oParam.reconstruct = QUDA_RECONSTRUCT_NO;
    oParam.setPrecision(test_prec, true);
    oParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    for (int i = 0; i < n_out(); i++) gpuOprod[i] = GaugeField(oParam);
  }

  void end()
  {
    cpuQuark.clear();
    cudaQuark.clear();
    gpuOprod = {};
  }

  /**
   * @brief Accumulate the outer products of all device quark fields niter times
   * @return Wall-clock time in seconds
   */
  double oprodCUDA(int niter)
  {
    GaugeField *out[2] = {&gpuOprod[0], &gpuOprod[1]};
    host_timer_t host_timer;

    comm_barrier();
    host_timer.start();
    for (int i = 0; i < niter; i++) {
      for (int k = 0; k < Nsrc; k++) {
        const auto coeff = oprod_coeff(k);
        computeStaggeredOprod(out, cudaQuark[k], coeff.data(), nFace);
      }
    }
    qudaDeviceSynchronize();
    host_timer.stop();

    return host_timer.last();
  }

  void run_test(int niter, bool print_metrics = false)
  {
    printfQuda("Tuning...\n");
    oprodCUDA(1);

    printfQuda("Running %d iterations of computation\n", niter);
    double secs = oprodCUDA(niter);

    if (print_metrics) printfQuda("%fus per accumulation of %d outer products\n", 1e6 * secs / niter, Nsrc);
  }

  /**
   * @brief Accumulate one pass of outer products into zeroed outputs, reassigning a single device quark field for each
   *        term as the force interfaces do, and compare against the CPU reference
   * @return Per output (one-hop, then three-hop for nFace = 3): whether every element agrees, and the accuracy level
   */
  std::array<std::pair<bool, int>, 2> verify()
  {
    for (int i = 0; i < n_out(); i++) gpuOprod[i].zero();
    GaugeField *out[2] = {&gpuOprod[0], &gpuOprod[1]};

    GaugeFieldParam oParam(gpuOprod[0]);
    oParam.location = QUDA_CPU_FIELD_LOCATION;
    oParam.setPrecision(gauge_param.cpu_prec);
    oParam.order = QUDA_QDP_GAUGE_ORDER;
    oParam.create = QUDA_ZERO_FIELD_CREATE;
    std::array<GaugeField, 2> expectedOprod;
    for (int i = 0; i < n_out(); i++) expectedOprod[i] = GaugeField(oParam);

    ColorSpinorField quark(cudaQuark[0]);
    for (int k = 0; k < Nsrc; k++) {
      const auto coeff = oprod_coeff(k);
      quark = cpuQuark[k];
      computeStaggeredOprod(out, quark, coeff.data(), nFace);
      computeStaggeredOprodCPU(expectedOprod, cpuQuark[k], coeff.data(), nFace);
    }

    std::array<std::pair<bool, int>, 2> results {};
    oParam.create = QUDA_NULL_FIELD_CREATE;
    for (int i = 0; i < n_out(); i++) {
      GaugeField result(oParam);
      result.copy(gpuOprod[i]);
      results[i]
        = compare_link_fields(result, "GPU result:", expectedOprod[i], "CPU reference:", getTolerance(test_prec));
    }
    return results;
  }
};

/**
 * @brief Check the results of StaggeredOprodTestWrapper::verify with gtest
 */
inline void expect_oprod_results(const std::array<std::pair<bool, int>, 2> &results, int n_out, QudaPrecision test_prec)
{
  for (int i = 0; i < n_out; i++) {
    const char *label = i == 0 ? "One-hop" : "Three-hop";
    EXPECT_TRUE(results[i].first) << label << " outer product compare_floats mismatch";
    EXPECT_GE(results[i].second, getNegLog10Tolerance(test_prec)) << label << " outer product strong_check mismatch";
  }
}
