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
#include <qio_field.h>

#include <test_util.h>
#include <dslash_util.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include <misc.h>
#include <gtest.h>

using std::endl;
using std::cout;
using std::vector;

using namespace quda;

double kappa5;

QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cuda_prec;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

ColorSpinorParam csParam;

cudaColorSpinorField *cudaSpinor, *tmp1=nullptr, *tmp2=nullptr;

void *hostGauge[4], *hostClover, *hostCloverInv;

Dirac *dirac = nullptr;

// Dirac operator type
extern QudaDslashType dslash_type;

// Twisted mass flavor type
extern QudaTwistFlavorType twist_flavor;
extern QudaMatPCType matpc_type;

extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaDagType dagger;
QudaDagType not_dagger;

extern bool compute_clover;
extern double clover_coeff;

extern bool verify_results;
extern int niter;
extern char latfile[];

extern bool kernel_pack_t;

extern double mass; // mass of Dirac operator
extern double mu;

extern double tol; // tolerance for inverter
extern int    niter; // max solver iterations
extern int    Nsrc; // number of spinors to apply to simultaneously
extern int    Msrc; // block size used by trilinos solver

//extern QudaVerbosity verbosity = QUDA_VERBOSE;

namespace prototypequda{

void display_test_info()
{
  printfQuda("running the following test:\n");
 
  printfQuda("prec    recon   matpc_type   dagger   S_dim         T_dimension   Ls_dimension dslash_type    niter\n");
  printfQuda("%6s   %2s   %12s    %d    %3d/%3d/%3d        %3d             %2d   %14s   %d\n", 
	     get_prec_str(prec), get_recon_str(link_recon), 
	     get_matpc_str(matpc_type), dagger, xdim, ydim, zdim, tdim, Lsdim,
	     get_dslash_str(dslash_type), niter);
  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3));

  return ;
    
}


void initialize(int argc, char **argv) {

  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  cuda_prec = prec;

  gauge_param = newQudaGaugeParam();
  inv_param   = newQudaInvertParam();

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH ) {
    errorQuda("Operator type %d is not supported.", dslash_type);
  } else {   
    setDims(gauge_param.X);
    //setKernelPackT(kernel_pack_t);
    Ls = 1;
  }

  setSpinorSiteSize(24);

  gauge_param.anisotropy = 1.0;

  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.reconstruct_sloppy = link_recon;
  gauge_param.cuda_prec_sloppy = cuda_prec;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.mu = mu;
  inv_param.mass = mass;
  inv_param.Ls = Ls;
  inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));
  
  inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  inv_param.matpc_type = matpc_type;
  inv_param.dagger = dagger;
  not_dagger = (QudaDagType)((dagger + 1)%2);

  inv_param.cpu_prec = cpu_prec;
  if (inv_param.cpu_prec != gauge_param.cpu_prec) {
    errorQuda("Gauge and spinor CPU precisions must match");
  }
  inv_param.cuda_prec = cuda_prec;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

#ifndef MULTI_GPU // free parameter for single GPU
  gauge_param.ga_pad = 0;
#else // must be this one c/b face for multi gpu
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =std::max(x_face_size, y_face_size);
  pad_size = std::max(pad_size, z_face_size);
  pad_size = std::max(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  //inv_param.sp_pad = xdim*ydim*zdim/2;
  //inv_param.cl_pad = 24*24*24;

  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // test code only supports DeGrand-Rossi Basis
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = inv_param.clover_cuda_prec;
    inv_param.clover_cuda_prec_precondition = inv_param.clover_cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = clover_coeff;
    hostClover = malloc((size_t)V*cloverSiteSize*inv_param.clover_cpu_prec);
    hostCloverInv = malloc((size_t)V*cloverSiteSize*inv_param.clover_cpu_prec);
  }

  inv_param.tol     = tol;
  inv_param.maxiter = niter;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc((size_t)V*gaugeSiteSize*gauge_param.cpu_prec);
  
  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, hostGauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(hostGauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate a random SU(3) field
    construct_gauge_field(hostGauge, 1, gauge_param.cpu_prec, &gauge_param);
  }


  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    double norm = 0.1; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal
    construct_clover_field(hostClover, norm, diag, inv_param.clover_cpu_prec);
    memcpy(hostCloverInv, hostClover, (size_t)V*cloverSiteSize*inv_param.clover_cpu_prec);
  }

  printfQuda("done.\n"); fflush(stdout);
  
  initQuda(device);

  // set verbosity prior to loadGaugeQuda
  //setVerbosity(verbosity);
  inv_param.verbosity = QUDA_VERBOSE;// verbosity;

  printfQuda("Sending gauge field to GPU\n");
  loadGaugeQuda(hostGauge, &gauge_param);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    if (compute_clover) printfQuda("Computing clover field on GPU\n");
    else printfQuda("Sending clover field to GPU\n");
    inv_param.compute_clover = compute_clover;
    inv_param.return_clover = compute_clover;
    inv_param.compute_clover_inverse = compute_clover;
    inv_param.return_clover_inverse = compute_clover;

    loadCloverQuda(hostClover, hostCloverInv, &inv_param);
  }

  //Create a reference spinor object 
  {
    csParam.nColor = 3;
    csParam.nSpin  = 4;
    csParam.nDim   = 4;
    for (int d=0; d<4; d++) csParam.x[d] = gauge_param.X[d];
    csParam.pc_type = QUDA_5D_PC;

    csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    csParam.create    = QUDA_ZERO_FIELD_CREATE;

    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.setPrecision(inv_param.cuda_prec);
    if (csParam.Precision() == QUDA_DOUBLE_PRECISION ) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    } else {
      /* Single and half */
      csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    }
 
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;

    printfQuda("Creating reference cudaSpinor\n");
    cudaSpinor = new cudaColorSpinorField(csParam);

    tmp1 = new cudaColorSpinorField(csParam);
    tmp2 = new cudaColorSpinorField(csParam);

    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, /*pc=*/true);
    diracParam.tmp1 = tmp1;
    diracParam.tmp2 = tmp2;
   
    dirac = Dirac::create(diracParam);

  } 

  return;  
}

void end() {

   printfQuda("Cleaning resource...\n");
  
  {
    if(dirac != nullptr){
      delete dirac;
      dirac = nullptr;
    }
    delete cudaSpinor;
    delete tmp1;
    delete tmp2;
  }

  // release memory

  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    free(hostClover);
    free(hostCloverInv);
  }
//Finalize QUDA:
  endQuda();

  return;
}

}//end trilquda

extern void usage(char**);

int main(int argc, char *argv[]) {

  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);

  for (int i =1;i < argc; i++) {
    if(process_command_line_option(argc, argv, &i) == 0) continue;
       
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  prototypequda::initialize(argc, argv);

////////////////////////////////////////
  const int vol   = csParam.x[0]*csParam.x[1]*csParam.x[2]*csParam.x[3];
  if( dslash_type != QUDA_STAGGERED_DSLASH ){
    setSpinorSiteSize(24);
  } else {
    setSpinorSiteSize(6 );
  }

  inv_param.Ls = 32;

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float); 

  void *vec1    = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
 
  // perform the inversion
  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
    //((float*)spinorIn)[0] = 1.0;
    for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((float*)vec1)[i] = rand() / (float)RAND_MAX;
  } else {
    //((double*)spinorIn)[0] = 1.0;
    for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((double*)vec1)[i] = rand() / (double)RAND_MAX;
  }

  ColorSpinorParam cpuParam(vec1, inv_param, gauge_param.X, /*pc=*/ true, inv_param.input_location);
  ColorSpinorField *v1_h = ColorSpinorField::Create(cpuParam);
////////////////////////////////////////

  prototypequda::end();

  return 0;
} // end prototype_test.cpp
