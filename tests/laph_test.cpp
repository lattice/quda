#include <cstdlib>
#include <algorithm>
#include <vector>

#include "test.h"

/*
  This test will eventually evolve into a full test for 3-d Laplace
  eigenvector generation and sink projection.  For now it's just a pile of hacks.
 */

using test_t = std::tuple<int, int>;

struct LaphTest : ::testing::TestWithParam<test_t> {
  int nSink;
  int nEv;

  LaphTest() : nSink(std::get<0>(GetParam())), nEv(std::get<1>(GetParam())) { }
};

TEST_P(LaphTest, verify)
{
  using namespace quda;

  constexpr int nSpin = 4;
  constexpr int nColor = 3;

  QudaInvertParam invParam = newQudaInvertParam();
  invParam.cpu_prec = QUDA_DOUBLE_PRECISION;
  invParam.cuda_prec = cuda_prec;
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

  // could OMP parallelize outer four loops without running afoul of
  // reductions over time slice
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

  // TODO MPI reduction

  // QUDA proper
  void *snkPtr[nSink];
  for (int iSink = 0; iSink < nSink; ++iSink) snkPtr[iSink] = sinkList[iSink].data();

  void *evPtr[nEv];
  for (int iEv = 0; iEv < nEv; ++iEv) evPtr[iEv] = evList[iEv].data();

  std::vector<Complex> qudaRes(nSink * nEv * Lt * nSpin, 0.);

  int X[4] = {xdim, ydim, zdim, tdim};
  laphSinkProject((__complex__ double *)qudaRes.data(), (void **)snkPtr, nSink, (void **)evPtr, nEv, &invParam, X);

  // check solutions
  auto tol = getTolerance(cuda_prec);
  for (unsigned int iEl = 0; iEl < qudaRes.size(); ++iEl) {
    auto deviation = abs(qudaRes[iEl] - hostRes[iEl]);
    if (deviation > tol) {
      printfQuda("EV projection test failed at iEl=%d: (%f,%f) [QUDA], (%f,%f) [host]\n", iEl, qudaRes[iEl].real(),
                 qudaRes[iEl].imag(), hostRes[iEl].real(), hostRes[iEl].imag());
    }
    EXPECT_LE(deviation, tol);
  }
}

INSTANTIATE_TEST_SUITE_P(LaphTest, LaphTest, ::testing::Combine(::testing::Values(1), ::testing::Values(1)));

int main(int argc, char **argv)
{
  quda_test test("laph_test", argc, argv);
  test.init();
  test.execute();
}
