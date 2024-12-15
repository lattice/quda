#include <cstdlib>
#include <algorithm>
#include <vector>

#include "instantiate.h"
#include "test.h"
#include "misc.h"

/*
  This test will perhaps eventually evolve into a full test for 3-d
  Laplace eigenvector generation and sink projection.  For now the
  focus is eigen-vector projection
 */

using test_t = std::tuple<QudaPrecision, int, int, int, int>;

using ::testing::Combine;
using ::testing::get;
using ::testing::Values;

struct LaphTest : ::testing::TestWithParam<test_t> {
  test_t param;
  LaphTest() : param(GetParam()) { }
};

auto laph_test(test_t param)
{
  using namespace quda;

  QudaPrecision precision = get<0>(param);
  int nSink = get<1>(param);
  int nEv = get<2>(param);
  int tileSink = get<3>(param);
  int tileEv = get<4>(param);

  constexpr int nSpin = 4;
  constexpr int nColor = 3;

  QudaInvertParam invParam = newQudaInvertParam();
  invParam.cpu_prec = QUDA_DOUBLE_PRECISION;
  invParam.cuda_prec = precision;
  invParam.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invParam.dirac_order = QUDA_DIRAC_ORDER;

  ColorSpinorParam cs_param;
  cs_param.nColor = nColor;
  cs_param.nSpin = nSpin;
  cs_param.x = {xdim, ydim, zdim, tdim};
  cs_param.siteSubset = QUDA_FULL_SITE_SUBSET;
  cs_param.setPrecision(invParam.cpu_prec);
  cs_param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  cs_param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  cs_param.gammaBasis = invParam.gamma_basis;
  cs_param.pc_type = QUDA_4D_PC;
  cs_param.location = QUDA_CPU_FIELD_LOCATION;
  cs_param.create = QUDA_NULL_FIELD_CREATE;

  ColorSpinorField rngDummy(cs_param);
  RNG rng(rngDummy, 1234);

  // initialize quark sinks
  std::vector<ColorSpinorField> sinkList(nSink);
  for (auto &s : sinkList) {
    s = ColorSpinorField(cs_param);
    spinorNoise(s, rng, QUDA_NOISE_GAUSS);
  }

  // initialize EVs
  cs_param.nSpin = 1;
  std::vector<ColorSpinorField> evList(nEv);
  for (auto &e : evList) {
    e = ColorSpinorField(cs_param);
    spinorNoise(e, rng, QUDA_NOISE_GAUSS);
  }

  // host reference [nSink][nEv][Lt][nSpin][complexity]
  auto Lt = tdim * comm_dim(3);
  std::vector<Complex> hostRes(nSink * nEv * Lt * nSpin, 0.);

#pragma omp parallel for collapse(4)
  for (int iEv = 0; iEv < nEv; ++iEv) {
    for (int iSink = 0; iSink < nSink; ++iSink) {
      for (int iSpin = 0; iSpin < nSpin; ++iSpin) {
        for (int iT = 0; iT < tdim; ++iT) {
          int globT = comm_coord(3) * tdim + iT;
          for (int iZ = 0; iZ < zdim; ++iZ) {
            for (int iY = 0; iY < ydim; ++iY) {
              for (int iX = 0; iX < xdim; ++iX) {
                int coord[4] = {iX, iY, iZ, iT};
                int linInd;
                evList[iEv].OffsetIndex(linInd, coord);

                for (int iCol = 0; iCol < nColor; ++iCol)
                  hostRes[iSpin + nSpin * (globT + Lt * (iEv + nEv * iSink))]
                    += conj(evList[iEv].data<Complex *>()[iCol + nColor * linInd])
                    * sinkList[iSink].data<Complex *>()[iCol + nColor * (iSpin + nSpin * linInd)];
              }
            }
          }
        } // volume loops
      }   // spin loop
    }     // sink loop
  }       // ev loop

  comm_allreduce_sum(hostRes);

  // QUDA proper
  std::vector<void *> snkPtr(nSink);
  for (int iSink = 0; iSink < nSink; ++iSink) snkPtr[iSink] = sinkList[iSink].data();

  std::vector<void *> evPtr(nEv);
  for (int iEv = 0; iEv < nEv; ++iEv) evPtr[iEv] = evList[iEv].data();

  std::vector<Complex> qudaRes(nSink * nEv * Lt * nSpin, 0.);

  int X[4] = {xdim, ydim, zdim, tdim};
  laphSinkProject((__complex__ double *)qudaRes.data(), snkPtr.data(), nSink, tileSink, evPtr.data(), nEv, tileEv,
                  &invParam, X);
  printfQuda("laphSinkProject Done: %g secs, %g Gflops\n", invParam.secs, invParam.gflops / invParam.secs);

  auto tol = getTolerance(cuda_prec);
  int rtn = 0;
  for (unsigned int i = 0; i < qudaRes.size(); i++) {
    auto deviation = abs(qudaRes[i] - hostRes[i]) / abs(hostRes[i]);
    if (deviation > tol) {
      printfQuda("EV projection test failed at iEl=%d: (%f,%f) [QUDA], (%f,%f) [host]\n", i, qudaRes[i].real(),
                 qudaRes[i].imag(), hostRes[i].real(), hostRes[i].imag());
      EXPECT_LE(deviation, tol);
      rtn = 1;
    }
  }
  return rtn;
}

TEST_P(LaphTest, verify)
{
  if (!quda::is_enabled(get<0>(GetParam()))) GTEST_SKIP();
  laph_test(GetParam());
}

INSTANTIATE_TEST_SUITE_P(LaphTest, LaphTest,
                         Combine(Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION), Values(1, 4, 7), Values(2),
                                 Values(4), Values(2)),
                         [](testing::TestParamInfo<test_t> param) {
                           return std::string(get_prec_str(get<0>(param.param))) + "_"
                             + std::to_string(get<1>(param.param)) + "_" + std::to_string(get<2>(param.param)) + "_"
                             + std::to_string(get<3>(param.param)) + "_" + std::to_string(get<4>(param.param));
                         });

int main(int argc, char **argv)
{
  quda_test test("laph_test", argc, argv);
  test.init();

  int result = 0;
  if (enable_testing) {
    result = test.execute();
  } else {
    result = laph_test({cuda_prec, Msrc, Nsrc, Msrc_tile, Nsrc_tile});
  }

  return result;
}
