#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <covdev_reference.h>
#include <gauge_field.h>

#include <assert.h>
#include <gtest/gtest.h>

using namespace quda;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

cpuGaugeField *cpuLink = nullptr;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut;

cudaColorSpinorField* tmp;

void *links[4];

#ifdef MULTI_GPU
void **ghostLink;
#endif

QudaParity parity = QUDA_EVEN_PARITY;

GaugeCovDev* dirac;

const int nColor = 3;

void init(int argc, char **argv)
{
  initQuda(device);

  setVerbosity(QUDA_VERBOSE);

  gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  setDims(gauge_param.X);
  Ls = 1;

  if (Nsrc != 1) warningQuda("The covariant derivative doesn't support 5-d indexing, only source 0 will be tested");

  setSpinorSiteSize(24);

  inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  inv_param.dslash_type = QUDA_COVDEV_DSLASH; // ensure we use the correct dslash

  ColorSpinorParam csParam;
  csParam.nColor=nColor;
  csParam.nSpin=4;
  csParam.nDim=4;
  for (int d = 0; d < 4; d++) { csParam.x[d] = gauge_param.X[d]; }
  //  csParam.x[4] = Nsrc; // number of sources becomes the fifth dimension

  csParam.setPrecision(inv_param.cpu_prec);
  csParam.pad = 0;
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.pc_type = QUDA_4D_PC;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder  = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis; // this parameter is meaningless for staggered
  csParam.create = QUDA_ZERO_FIELD_CREATE;    

  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gauge_param.X[0];

  printfQuda("Randomizing fields ...\n");
  spinor->Source(QUDA_RANDOM_SOURCE);

  // Allocate host side memory for the gauge field.
  //----------------------------------------------------------------------------
  for (int dir = 0; dir < 4; dir++) {
    links[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    if (links[dir] == NULL) {
      errorQuda("ERROR: malloc failed for gauge links");
    }  
  }
  constructHostGaugeField(links, gauge_param, argc, argv);

  // cpuLink is only used for ghost allocation
  GaugeFieldParam cpuParam(links, gauge_param);
  cpuParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuLink   = new cpuGaugeField(cpuParam);
  ghostLink = cpuLink->Ghost();

  printfQuda("Links sending...");
  loadGaugeQuda(links, &gauge_param);
  printfQuda("Links sent\n");

  printfQuda("Sending fields to GPU...");

  csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
  csParam.pad = inv_param.sp_pad;
  csParam.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);

  printfQuda("Creating cudaSpinor\n");
  cudaSpinor = new cudaColorSpinorField(csParam);

  printfQuda("Creating cudaSpinorOut\n");
  cudaSpinorOut = new cudaColorSpinorField(csParam);

  printfQuda("Sending spinor field to GPU\n");
  *cudaSpinor = *spinor;

  cudaDeviceSynchronize();
  checkCudaError();

  double spinor_norm2 = blas::norm2(*spinor);
  double cuda_spinor_norm2 = blas::norm2(*cudaSpinor);
  printfQuda("Source CPU = %f, CUDA=%f\n", spinor_norm2, cuda_spinor_norm2);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  tmp = new cudaColorSpinorField(csParam);

  DiracParam diracParam;
  setDiracParam(diracParam, &inv_param, false);

  diracParam.tmp1 = tmp;

  dirac = new GaugeCovDev(diracParam);
}

void end(void) 
{
  for (int dir = 0; dir < 4; dir++) {
    free(links[dir]);
  }

  delete dirac;
  delete cudaSpinor;
  delete cudaSpinorOut;
  delete tmp;
  delete spinor;
  delete spinorOut;
  delete spinorRef;

  if (cpuLink) delete cpuLink;

  endQuda();
}

double dslashCUDA(int niter, int mu)
{
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  for (int i = 0; i < niter; i++) dirac->MCD(*cudaSpinorOut, *cudaSpinor, mu);

  cudaEventCreate(&end);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float runTime;
  cudaEventElapsedTime(&runTime, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  double secs = runTime / 1000; //stopwatchReadSeconds();

  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    errorQuda("with ERROR: %s\n", cudaGetErrorString(stat));

  return secs;
}

void covdevRef(int mu)
{
  // compare to dslash reference implementation
  printfQuda("Calculating reference implementation...");
#ifdef MULTI_GPU
  mat_mg4dir(spinorRef, links, ghostLink, spinor, dagger, mu, inv_param.cpu_prec, gauge_param.cpu_prec);
#else
  mat(spinorRef->V(), links, spinor->V(), dagger, mu, inv_param.cpu_prec, gauge_param.cpu_prec);
#endif    
  printfQuda("done.\n");
}

TEST(dslash, verify)
{
  double deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*spinorRef, *spinorOut)));
  double tol = (inv_param.cuda_prec == QUDA_DOUBLE_PRECISION ? 1e-12 :
		(inv_param.cuda_prec == QUDA_SINGLE_PRECISION ? 1e-3 : 1e-1));
  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec recon   test_type     dagger   S_dim         T_dimension\n");
  printfQuda("%s   %s       %d           %d       %d/%d/%d        %d \n", 
      get_prec_str(prec), get_recon_str(link_recon), 
      test_type, dagger, xdim, ydim, zdim, tdim);
  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
      dimPartitioned(0),
      dimPartitioned(1),
      dimPartitioned(2),
      dimPartitioned(3));
}

int main(int argc, char **argv) 
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);
  // return code for google test
  int test_rc = 0;
  // command line options
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  display_test_info();

  init(argc, argv);

  int attempts = 1;
  for (int i = 0; i < attempts; i++) {

    // Test forward directions, then backward
    for (int dag = 0; dag < 2; dag++) {
      dag == 0 ? dagger = QUDA_DAG_NO : dagger = QUDA_DAG_YES;

      for (int mu = 0; mu < 4; mu++) { // We test all directions in one go
        int muCuda = mu + (dagger ? 4 : 0);
        int muCpu = mu * 2 + (dagger ? 1 : 0);

        // Reference computation
        covdevRef(muCpu);
        printfQuda("\n\nChecking muQuda = %d\n", muCuda);

        { // warm-up run
          printfQuda("Tuning...\n");
          dslashCUDA(1, muCuda);
        }

        printfQuda("Executing %d kernel loop(s)...", niter);

        double secs = dslashCUDA(niter, muCuda);
        *spinorOut = *cudaSpinorOut;
        printfQuda("\n%fms per loop\n", 1000 * secs);

        unsigned long long flops
          = niter * cudaSpinor->Nspin() * (8 * nColor - 2) * nColor * (long long)cudaSpinor->Volume();
        printfQuda("GFLOPS = %f\n", 1.0e-9 * flops / secs);

        double spinor_ref_norm2 = blas::norm2(*spinorRef);
        double spinor_out_norm2 = blas::norm2(*spinorOut);

        double cuda_spinor_out_norm2 = blas::norm2(*cudaSpinorOut);
        printfQuda("Results mu = %d: CPU=%f, CUDA=%f, CPU-CUDA=%f\n", muCuda, spinor_ref_norm2, cuda_spinor_out_norm2,
                   spinor_out_norm2);

        if (verify_results) {
          ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
          if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

          test_rc = RUN_ALL_TESTS();
          if (test_rc != 0) warningQuda("Tests failed");
        }
      } // Directions
    }   // Dagger
  }

  end();

  finalizeComms();
  return test_rc;
}

