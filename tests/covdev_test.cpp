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
#include <test_util.h>
#include <test_params.h>
#include <dslash_util.h>
#include <covdev_reference.h>
#include <gauge_field.h>

#include <assert.h>
#include <gtest/gtest.h>

using namespace quda;

#define MAX(a,b) ((a)>(b)?(a):(b))

QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cuda_prec;

QudaGaugeParam gaugeParam;
QudaInvertParam inv_param;

cpuGaugeField *cpuLink = NULL;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut;

cudaColorSpinorField* tmp;

void *hostGauge[4];
void *links[4];

#ifdef MULTI_GPU
void **ghostLink;
#endif

QudaParity parity = QUDA_EVEN_PARITY;
int transfer = 0; // include transfer time in the benchmark?

int X[4];

GaugeCovDev* dirac;

const int nColor = 3;

void init(int argc, char **argv)
{

  initQuda(device);

  setVerbosity(QUDA_VERBOSE);

  gaugeParam = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  cuda_prec = prec;

  gaugeParam.X[0] = X[0] = xdim;
  gaugeParam.X[1] = X[1] = ydim;
  gaugeParam.X[2] = X[2] = zdim;
  gaugeParam.X[3] = X[3] = tdim;

  setDims(gaugeParam.X);
  Ls = 1;

  if (Nsrc != 1)
    printfQuda ("The covariant derivative doesn't support 5-d indexing, only source 0 will be tested.\n");

  setSpinorSiteSize(24);

  gaugeParam.cpu_prec = cpu_prec;
  gaugeParam.cuda_prec = cuda_prec;
  gaugeParam.reconstruct = link_recon;
  gaugeParam.reconstruct_sloppy = gaugeParam.reconstruct;
  gaugeParam.cuda_prec_sloppy = gaugeParam.cuda_prec;

  // ensure we use the right dslash
  dslash_type = QUDA_COVDEV_DSLASH;

  gaugeParam.anisotropy = 1.0;
  gaugeParam.tadpole_coeff = 0.8;
  gaugeParam.scale = 1.0;
  gaugeParam.type = QUDA_WILSON_LINKS;
  gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;
  inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dslash_type = dslash_type;
  inv_param.mass = mass;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  int tmpint = MAX(X[1]*X[2]*X[3], X[0]*X[2]*X[3]);
  tmpint = MAX(tmpint, X[0]*X[1]*X[3]);
  tmpint = MAX(tmpint, X[0]*X[1]*X[2]);


  gaugeParam.ga_pad = tmpint;
  inv_param.sp_pad = tmpint;

  ColorSpinorParam csParam;
  csParam.nColor=nColor;
  csParam.nSpin=4;
  csParam.nDim=4;
  for(int d = 0; d < 4; d++) {
    csParam.x[d] = gaugeParam.X[d];
  }
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
  csParam.x[0] = gaugeParam.X[0];

  printfQuda("Randomizing fields ...\n");

  spinor->Source(QUDA_RANDOM_SOURCE);

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  for (int dir = 0; dir < 4; dir++) {
    links[dir] = malloc(V*gaugeSiteSize*gSize);

    if (links[dir] == NULL) {
      errorQuda("ERROR: malloc failed for gauge links");
    }  
  }

  construct_gauge_field(links, 1, gaugeParam.cpu_prec, &gaugeParam);

#ifdef MULTI_GPU
  gaugeParam.type = QUDA_SU3_LINKS;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuParam(links, gaugeParam);
  cpuParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuLink   = new cpuGaugeField(cpuParam);
  ghostLink = cpuLink->Ghost();

  int x_face_size = X[1]*X[2]*X[3]/2;
  int y_face_size = X[0]*X[2]*X[3]/2;
  int z_face_size = X[0]*X[1]*X[3]/2;
  int t_face_size = X[0]*X[1]*X[2]/2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gaugeParam.ga_pad = pad_size;    
#endif

  gaugeParam.type = QUDA_SU3_LINKS;
  gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = link_recon;
  
  printfQuda("Links sending..."); 
  loadGaugeQuda(links, &gaugeParam);
  printfQuda("Links sent\n"); 

  printfQuda("Sending fields to GPU..."); 

  if (!transfer) {
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.setPrecision(inv_param.cuda_prec);
    if (csParam.Precision() == QUDA_DOUBLE_PRECISION ) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    } else {
      /* Single and half */
      csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    }

    printfQuda("Creating cudaSpinor\n");
    cudaSpinor = new cudaColorSpinorField(csParam);

    printfQuda("Creating cudaSpinorOut\n");
    cudaSpinorOut = new cudaColorSpinorField(csParam);

    printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;

    cudaDeviceSynchronize();
    checkCudaError();
	
    double spinor_norm2 = blas::norm2(*spinor);
    double cuda_spinor_norm2=  blas::norm2(*cudaSpinor);
    printfQuda("Source CPU = %f, CUDA=%f\n", spinor_norm2, cuda_spinor_norm2);

    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    tmp = new cudaColorSpinorField(csParam);

    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, false);

    diracParam.tmp1=tmp;

    dirac = new GaugeCovDev(diracParam);

  } else {
    errorQuda("Error not suppported");
  }

  return;
}

void end(void) 
{
  for (int dir = 0; dir < 4; dir++) {
    free(links[dir]);
  }

  if (!transfer){
    delete dirac;
    delete cudaSpinor;
    delete cudaSpinorOut;
    delete tmp;
  }

  delete spinor;
  delete spinorOut;
  delete spinorRef;

  if (cpuLink) delete cpuLink;

  endQuda();
}

double dslashCUDA(int niter, int mu) {

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  for (int i = 0; i < niter; i++) {
    if (transfer){
      //MatQuda(spinorGPU, spinor, &inv_param);
    } else {
      dirac->MCD(*cudaSpinorOut, *cudaSpinor, mu);
    }
  }

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
  fflush(stdout);
#ifdef MULTI_GPU
  mat_mg4dir(spinorRef, links, ghostLink, spinor, dagger, mu, inv_param.cpu_prec, gaugeParam.cpu_prec);
#else
  mat(spinorRef->V(), links, spinor->V(), dagger, mu, inv_param.cpu_prec, gaugeParam.cpu_prec);
#endif    
  printfQuda("done.\n");

}

TEST(dslash, verify) {
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

  return ;

}

int main(int argc, char **argv) 
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);
  // return code for google test
  int test_rc = 0;
  // command line options
  auto app = make_app();
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

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
        if (!transfer) *spinorOut = *cudaSpinorOut;
        printfQuda("\n%fms per loop\n", 1000 * secs);

        unsigned long long flops
          = niter * cudaSpinor->Nspin() * (8 * nColor - 2) * nColor * (long long)cudaSpinor->Volume();
        printfQuda("GFLOPS = %f\n", 1.0e-9 * flops / secs);

        double spinor_ref_norm2 = blas::norm2(*spinorRef);
        double spinor_out_norm2 = blas::norm2(*spinorOut);

        if (!transfer) {
          double cuda_spinor_out_norm2 = blas::norm2(*cudaSpinorOut);
          printfQuda("Results mu = %d: CPU=%f, CUDA=%f, CPU-CUDA=%f\n", muCuda, spinor_ref_norm2, cuda_spinor_out_norm2,
                     spinor_out_norm2);
        } else {
          printfQuda("Result mu = %d: CPU=%f , CPU-CUDA=%f", mu, spinor_ref_norm2, spinor_out_norm2);
        }

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

