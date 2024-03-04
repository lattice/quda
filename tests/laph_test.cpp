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
  constexpr int nSpin = 4;
  constexpr int nColor = 3;

  const int X[4] = {xdim, ydim, zdim, tdim};
  int nSites = X[0] * X[1] * X[2] * X[3];

  std::vector<std::vector<__complex__ double>> sink(nSink);
  for (auto &s : sink) s = std::vector<__complex__ double>(nSites * nSpin * nColor);

  std::vector<std::vector<__complex__ double>> ev(nEv);
  for (auto &e : ev) e = std::vector<__complex__ double>(nSites * nColor);

  std::vector<__complex__ double> result(nSink * nEv * X[3] * nSpin);

  QudaInvertParam invParam = newQudaInvertParam();
  invParam.cpu_prec = QUDA_DOUBLE_PRECISION;
  invParam.cuda_prec = QUDA_DOUBLE_PRECISION;
  invParam.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invParam.dirac_order = QUDA_DIRAC_ORDER;

  laphSinkProject(result.data(), reinterpret_cast<void **>(sink.data()), nSink, reinterpret_cast<void **>(ev.data()),
                  nEv, &invParam, X);
}

INSTANTIATE_TEST_SUITE_P(LaphTest, LaphTest, ::testing::Combine(::testing::Values(1), ::testing::Values(1)));

int main(int argc, char **argv)
{
  quda_test test("laph_test", argc, argv);
  test.init();
  test.execute();
}
