#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <algorithm>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>
#include <gauge_field.h>
#include <instantiate.h>
#include <tune_quda.h>

#include "misc.h"
#include "host_utils.h"
#include "gauge_utils.h"
#include "command_line_params.h"
#include "dslash_reference.h"
#include "covdev_reference.h"
#include "test.h"

using namespace quda;

const int nColor = 3;

QudaGaugeParam gauge_param;
GaugeField cpuLink;
void *links[4];

#include "covdev_test_gtest.hpp"

void init(int argc, char **argv)
{
  if (test_type != 0 and test_type != 1) errorQuda("Test type %d is not supported", test_type);

  gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  setDims(gauge_param.X);
  Ls = 1;

  if (Nsrc != 1) warningQuda("The covariant derivative doesn't support 5-d indexing, only source 0 will be tested");

  // Allocate host side memory for the gauge field.
  for (int dir = 0; dir < 4; dir++) { links[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size); }
  constructHostGaugeField(links, gauge_param, argc, argv);

  // cpuLink is only used for ghost allocation
  GaugeFieldParam cpuParam(gauge_param, links);
  cpuParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuLink = {cpuParam};
}

void end(void)
{
  for (int dir = 0; dir < 4; dir++) { host_free(links[dir]); }
  cpuLink = {};
}

double dslashCUDA(GaugeCovDev &dirac, ColorSpinorField &out, const ColorSpinorField &in, int niter, int mu)
{
  device_timer_t timer;
  timer.start();

  for (int i = 0; i < niter; i++) dirac.MCD(out, in, mu);

  timer.stop();
  return timer.last();
}

void covdevRef(ColorSpinorField &out, const ColorSpinorField &in, bool dagger, int mu)
{
  // compare to dslash reference implementation
  printfQuda("Calculating reference implementation...");
  mat(out, cpuLink, in, dagger, mu);
  printfQuda("done.\n");
}

std::array<double, 2> covdev_test(test_t param)
{
  QudaPrecision test_prec = ::testing::get<0>(param);
  QudaDagType test_dagger = ::testing::get<1>(param);
  int mu = ::testing::get<2>(param);

  printfQuda("Links sending...");
  gauge_param.cuda_prec = test_prec;
  gauge_param.cuda_prec_sloppy = test_prec;
  gauge_param.cuda_prec_precondition = test_prec;
  gauge_param.cuda_prec_refinement_sloppy = test_prec;
  gauge_param.cuda_prec_eigensolver = test_prec;
  loadGaugeQuda(links, &gauge_param);
  printfQuda("Links sent\n");

  auto inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  inv_param.cuda_prec = test_prec;
  inv_param.dslash_type = QUDA_COVDEV_DSLASH; // ensure we use the correct dslash
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  ColorSpinorParam csParam;
  csParam.nColor = nColor;
  csParam.nSpin = test_type == 0 ? 4 : 1; // use --test 1 for staggered case
  csParam.nDim = 4;
  for (int d = 0; d < 4; d++) { csParam.x[d] = gauge_param.X[d]; }

  csParam.setPrecision(cpu_prec);
  csParam.pad = 0;
  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.pc_type = QUDA_4D_PC;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis; // this parameter is meaningless for staggered
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  csParam.location = QUDA_CPU_FIELD_LOCATION;

  ColorSpinorField spinor(csParam);
  ColorSpinorField spinorOut(csParam);
  ColorSpinorField spinorRef(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gauge_param.X[0];

  printfQuda("Randomizing fields ...\n");
  spinor.Source(QUDA_RANDOM_SOURCE);

  printfQuda("Sending fields to GPU...");
  csParam.setPrecision(test_prec, test_prec, true);
  csParam.location = QUDA_CUDA_FIELD_LOCATION;

  ColorSpinorField cudaSpinor(csParam);
  ColorSpinorField cudaSpinorOut(csParam);

  printfQuda("Sending spinor field to GPU\n");
  cudaSpinor = spinor;

  double spinor_norm2 = blas::norm2(spinor);
  double cuda_spinor_norm2 = blas::norm2(cudaSpinor);
  printfQuda("Source CPU = %f, CUDA=%f\n", spinor_norm2, cuda_spinor_norm2);

  DiracParam diracParam;
  setDiracParam(diracParam, &inv_param, false);
  GaugeCovDev dirac(diracParam);

  int muQuda = mu + (test_dagger ? 4 : 0);

  // Reference computation
  covdevRef(spinorRef, spinor, test_dagger, mu);
  printfQuda("\n\nChecking muQuda = %d\n", muQuda);

  { // warm-up run
    printfQuda("Tuning...\n");
    dslashCUDA(dirac, cudaSpinorOut, cudaSpinor, 1, muQuda);
  }
  printfQuda("Executing %d kernel loop(s)...", niter);

  auto flops0 = quda::Tunable::flops_global();
  double secs = dslashCUDA(dirac, cudaSpinorOut, cudaSpinor, niter, muQuda);
  auto flops = (quda::Tunable::flops_global() - flops0);

  spinorOut = cudaSpinorOut;

  printfQuda("\n%fms per loop\n", 1000 * secs);
  printfQuda("GFLOPS = %f\n", 1.0e-9 * flops / secs);

  auto spinor_ref_norm2 = blas::norm2(spinorRef);
  auto spinor_out_norm2 = blas::norm2(spinorOut);
  auto cuda_spinor_out_norm2 = blas::norm2(cudaSpinorOut);
  printfQuda("Results mu = %d: CPU=%f, CUDA=%f, CPU-CUDA=%f\n", muQuda, spinor_ref_norm2, cuda_spinor_out_norm2,
             spinor_out_norm2);

  auto deviation = pow(10, -(double)(ColorSpinorField::Compare(spinorRef, spinorOut)));
  double tol = getTolerance(test_prec);

  return std::array<double, 2> {deviation, tol};
}

struct covdev_tester : quda_test {
  void display_info() const override
  {
    quda_test::display_info();
    printfQuda("S_dimension T_dimension Ls_dimension\n");
    printfQuda("%3d/%3d/%3d     %3d         %2d\n", xdim, ydim, zdim, tdim, Lsdim);
  }

  covdev_tester(int argc, char **argv) : quda_test("CovDev Test", argc, argv) { }

  void add_command_line_group(std::shared_ptr<QUDAApp> app) const override
  {
    quda_test::add_command_line_group(app);
    add_covdev_option_group(app);
  }
};

int main(int argc, char **argv)
{
  covdev_tester test(argc, argv);
  test.init();

  init(argc, argv);
  int result = 0;

  if (enable_testing) { // tests are defined in invert_test_gtest.hpp
    result = test.execute();
  } else { //
    covdev_test(test_t {prec, dagger ? QUDA_DAG_YES : QUDA_DAG_NO, covdev_mu});
  }

  end();
  return result;
}