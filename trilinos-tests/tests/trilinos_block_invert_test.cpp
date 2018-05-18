//@ORIGINAL HEADER
// ************************************************************************
//
//                 Belos: Block Linear Solvers Package
//                  Copyright 2004 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@ORIGINAL HEADER
//
// Modified for QUDA tests by A. Strelchenko (astrel@fnal.gov)
// This driver solves normal equations for the QUDA Wilson(-Clover) fermion operator
// The right-hand-side corresponds to a randomly generated solution.
// The initial guesses are all set to zero.
//
// NOTE: No preconditioner is used in this case.
//

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

#include <test_util.h>
#include <dslash_util.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include <qio_field.h>
// google test frame work
#include <gtest.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosBlockCGSolMgr.hpp"

// I/O for Harwell-Boeing files
#define HIDE_TPETRA_INOUT_IMPLEMENTATIONS
#include <Tpetra_MatrixIO.hpp>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Kokkos_DefaultNode.hpp>

using namespace Teuchos;
using Tpetra::Operator;
using Tpetra::CrsMatrix;
using Tpetra::MultiVector;
using std::endl;
using std::cout;
using std::vector;
using Teuchos::tuple;

using namespace quda;

static int MyPID = 0;

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

QudaVerbosity verbosity = QUDA_VERBOSE;

namespace trilquda{

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
    setKernelPackT(kernel_pack_t);
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
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
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
  setVerbosity(verbosity);
  inv_param.verbosity = verbosity;

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
    csParam.PCtype = QUDA_5D_PC;

    csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    csParam.create    = QUDA_ZERO_FIELD_CREATE;

    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.precision = inv_param.cuda_prec;
    if (csParam.precision == QUDA_DOUBLE_PRECISION ) {
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

template<class SC = Tpetra::Operator<>::scalar_type,
         class LO = typename Tpetra::Operator<SC>::local_ordinal_type,
         class GO = typename Tpetra::Operator<SC, LO>::global_ordinal_type,
         class NT = typename Tpetra::Operator<SC, LO, GO>::node_type>
class QUDA_WilsonClover_PC_Operator : public Tpetra::Operator<SC, LO, GO, NT> {
private:
  typedef typename NT::device_type device_type;

public:
  typedef typename Teuchos::ScalarTraits<SC>::magnitudeType mag_type;
  typedef Tpetra::Map<LO, GO, NT> map_type;

  QUDA_WilsonClover_PC_Operator (const Teuchos::RCP<const map_type>& map) :
    map_ (map), verbose_(true) { }

  Teuchos::RCP<const map_type> getDomainMap () const { return map_; }
  Teuchos::RCP<const map_type> getRangeMap  () const { return map_; }

  void SetVerbose(bool verbose) {verbose_ = verbose;}

  void apply (const Tpetra::MultiVector<SC, LO, GO, NT>& X,
              Tpetra::MultiVector<SC, LO, GO, NT>& Y,
              Teuchos::ETransp mode = Teuchos::NO_TRANS,
              SC alpha = Teuchos::ScalarTraits<SC>::one (),
              SC beta = Teuchos::ScalarTraits<SC>::zero ()) const
  {
    using Teuchos::RCP;
    typedef Tpetra::MultiVector<SC, LO, GO, NT> MV;

    TEUCHOS_TEST_FOR_EXCEPTION
      (mode == Teuchos::CONJ_TRANS and Teuchos::ScalarTraits<SC>::isComplex,
       std::logic_error, "Conjugate transpose case not implemented.");

    TEUCHOS_TEST_FOR_EXCEPTION
      (X.getNumVectors () != Y.getNumVectors (), std::logic_error, "Block sizes of input and output vectors do not match.");

    TEUCHOS_TEST_FOR_EXCEPTION
      ((inv_param.solution_type != QUDA_MATPC_SOLUTION) and (inv_param.solution_type != QUDA_MATPCDAG_MATPC_SOLUTION),
      std::logic_error, "This operator should be used for the preconditioned system.");
      
    auto numvecs_ = X.getNumVectors ();

////Print input norms:
    Teuchos::Array<mag_type> X_norms (numvecs_);
    Teuchos::Array<mag_type> Y_norms (numvecs_);

    X.norm2 (X_norms ());

    if(verbose_){
      if (MyPID == 0) {
        for (size_t j = 0; j < numvecs_; ++j) {
          std::cout << "Input array for column " << (j+1) << " of " << numvecs_
             << ": X norm: " << X_norms[j]
             << std::endl;
        }
      }
    }

////
    double quda_output_norm[numvecs_];

    for (size_t j = 0; j < numvecs_; ++j) {
      RCP<const MV> X_j = X.getVector (j);
      RCP<MV> Y_j = Y.getVectorNonConst (j);

      auto X_j_lcl_2d = X_j->template getLocalView<device_type> ();
      auto X_j_lcl = Kokkos::subview (X_j_lcl_2d, Kokkos::ALL (), 0);

      auto Y_j_lcl_2d = Y_j->template getLocalView<device_type> ();
      auto Y_j_lcl = Kokkos::subview (Y_j_lcl_2d, Kokkos::ALL (), 0);
/////////
      auto X_j_lcl_ptr = X_j_lcl.ptr_on_device ();
      auto Y_j_lcl_ptr = Y_j_lcl.ptr_on_device ();

      if(verbose_) {
        cudaPointerAttributes ptr_attr;
        if(cudaPointerGetAttributes(&ptr_attr, X_j_lcl_ptr) == cudaErrorInvalidValue) errorQuda("For X_j_lcl_ptr, a pointer was not allocated in, mapped by or registered with current CUDA context.\n");

        if ( ptr_attr.memoryType == cudaMemoryTypeDevice )
          warningQuda("Device allocation. (%d, is managed : %d)\n", ptr_attr.memoryType, ptr_attr.isManaged);
        else 
          warningQuda("Host allocation.(%d)\n", ptr_attr.memoryType);
      }

      ColorSpinorParam cpuParam(X_j_lcl_ptr, inv_param, gauge_param.X, /*pc=*/ true, inv_param.input_location);
      ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

      ColorSpinorParam cudaParam(cpuParam, inv_param);
      cudaColorSpinorField in(*in_h, cudaParam);

      auto trilinos_input_norm = X_norms[j];
      auto quda_input_norm     = sqrt(blas::norm2(in));
      auto input_norm_diff     = fabs(quda_input_norm - trilinos_input_norm); 

      if (verbose_) printfQuda("Converted input QUDA (%d) %e (trilinos/quda %e)\n", j, quda_input_norm, input_norm_diff);

      cudaParam.create = QUDA_NULL_FIELD_CREATE;
      cudaColorSpinorField out(in, cudaParam);

      //DiracMdagM m(dirac);
      //m(out, in);
      //
      dirac->MdagM(out, in); // apply the operator

      double kappa = inv_param.kappa;

      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        blas::ax(1.0/std::pow(2.0*kappa,4), out);
      } else if (inv_param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
        blas::ax(0.25/(kappa*kappa), out);
      }

      cpuParam.v = Y_j_lcl_ptr;
      cpuParam.location = inv_param.output_location;
      ColorSpinorField *out_h = ColorSpinorField::Create(cpuParam);
      *out_h = out;

      //if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
      quda_output_norm[j] = sqrt(blas::norm2(out));

      delete out_h;
      delete in_h;

    }

    Y.norm2 (Y_norms ());
////Print input norms:
    if(verbose_){
      if (MyPID == 0) {
        for (size_t j = 0; j < numvecs_; ++j) {
          std::cout << "Output norm for column " << (j+1) << " of " << numvecs_
               << ", Y norm: " << Y_norms[j]
               << std::endl;
        }
      }

      for (size_t j = 0; j < numvecs_; ++j) {
        auto trilinos_output_norm = Y_norms[j];
        auto output_norm_diff     = fabs(quda_output_norm[j] - trilinos_output_norm); 
        printfQuda("Output QUDA %e (trilinos/quda %e)\n", j, quda_output_norm, output_norm_diff);
      }
    }

    return;
  }//end apply


  void apply_cpu_wilson_clover_pc_operator(const Tpetra::MultiVector<SC, LO, GO, NT>& X,
              Tpetra::MultiVector<SC, LO, GO, NT>& Y) {
    using Teuchos::RCP;
    typedef Tpetra::MultiVector<SC, LO, GO, NT> MV;

    auto numvecs_ = X.getNumVectors ();

    MV R (X.getMap (), numvecs_);

    for (size_t j = 0; j < numvecs_; ++j) {
      RCP<const MV> X_j = X.getVector (j);
      RCP<MV> Y_j = Y.getVectorNonConst (j);
      RCP<MV> R_j = R.getVectorNonConst (j);

      auto X_j_lcl_2d = X_j->template getLocalView<device_type> ();
      auto X_j_lcl = Kokkos::subview (X_j_lcl_2d, Kokkos::ALL (), 0);

      auto Y_j_lcl_2d = Y_j->template getLocalView<device_type> ();
      auto Y_j_lcl = Kokkos::subview (Y_j_lcl_2d, Kokkos::ALL (), 0);

      auto R_j_lcl_2d = R_j->template getLocalView<device_type> ();
      auto R_j_lcl = Kokkos::subview (R_j_lcl_2d, Kokkos::ALL (), 0);
/////////
      auto X_j_lcl_ptr = X_j_lcl.ptr_on_device ();
      auto Y_j_lcl_ptr = Y_j_lcl.ptr_on_device ();
      auto R_j_lcl_ptr = R_j_lcl.ptr_on_device ();

      QudaMatPCType matpc_type = QUDA_MATPC_EVEN_EVEN;

      wil_matpc(R_j_lcl_ptr, hostGauge, X_j_lcl_ptr, inv_param.kappa, matpc_type, 0, cpu_prec, gauge_param);
      wil_matpc(Y_j_lcl_ptr, hostGauge, R_j_lcl_ptr, inv_param.kappa, matpc_type, 1, cpu_prec, gauge_param);
    }
  } //end cpu dslash..
 
private:
  Teuchos::RCP<const map_type> map_;
  bool verbose_;
};


extern void usage(char**);

typedef Tpetra::DefaultPlatform::DefaultPlatformType           Platform;
//typedef Tpetra::DefaultPlatform::DefaultPlatformType::NodeType cuda_Node;
typedef Kokkos::Compat::KokkosCudaWrapperNode                  cuda_Node;

int main(int argc, char *argv[]) {

//  typedef double ST;
  typedef Tpetra::Vector<>::scalar_type                 ST;//double
  typedef typename Tpetra::Map<>::local_ordinal_type    LO;
  typedef typename Tpetra::Map<>::global_ordinal_type   GO;
  typedef ScalarTraits<ST>                              SCT;
  typedef SCT::magnitudeType               MT;
  typedef Tpetra::Operator<ST,LO,GO,cuda_Node>      OP;
  typedef Tpetra::MultiVector<ST,LO,GO,cuda_Node>   MV;
  typedef Belos::OperatorTraits<ST,MV,OP> OPT;
  typedef Belos::MultiVecTraits<ST,MV>    MVT;
  typedef Tpetra::Map<LO,GO,cuda_Node>                   map_type;
  typedef Tpetra::MultiVector<ST,LO,GO,cuda_Node>        mv_type;

  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);

  for (int i =1;i < argc; i++) {
    if(process_command_line_option(argc, argv, &i) == 0) continue;
       
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  trilquda::initialize(argc, argv);

//  GlobalMPISession mpisess(&argc,&argv,&cout);

  bool success = false;
  bool verbose = true;
  try {
    const ST one  = SCT::one();

    Platform &platform = Tpetra::DefaultPlatform::getDefaultPlatform();
    RCP<const Comm<int> > comm = platform.getComm();
    RCP<cuda_Node>        node = platform.getNode();

    //
    // Get test parameters from command-line processor
    //
    bool proc_verbose = false;
    bool debug        = true;
    int frequency = 2;  // how often residuals are printed by solver
    int numrhs    = Nsrc; // total number of right-hand sides to solve for
    int blocksize = Msrc; // blocksize used by solver
    int maxiters  = inv_param.maxiter;   // maximum number of iterations for solver to use
    MT tol        = inv_param.tol;     // relative residual tolerance

    MyPID = rank(*comm);
    proc_verbose = ( verbose && (MyPID==0) );

    if (proc_verbose) {
      std::cout << Belos::Belos_Version() << std::endl << std::endl;
    }

    const GO gblNumRows = cudaSpinor->Length();
    const GO indexBase  = 0;

    RCP<const map_type> map (new map_type (gblNumRows, indexBase, comm));

    RCP<QUDA_WilsonClover_PC_Operator<ST, LO, GO, cuda_Node> > normalMdagM = rcp (new QUDA_WilsonClover_PC_Operator<ST, LO, GO, cuda_Node> (map));

    normalMdagM->SetVerbose(false);

    // Create initial vectors
    RCP<mv_type > B, X;
    X = rcp( new mv_type(map,numrhs) );
    MVT::MvRandom( *X );
    B = rcp( new mv_type(map,numrhs) );
    OPT::Apply( *normalMdagM, *X, *B );
    MVT::MvInit( *X, 0.0 );

    //
    // ********Other information used by block solver***********
    // *****************(can be user specified)******************
    //
    const int NumGlobalElements = B->getGlobalLength();
    if (maxiters == -1) {
      maxiters = NumGlobalElements/blocksize - 1; // maximum number of iterations to run
    }
    //
    ParameterList belosList;
    belosList.set( "Block Size", blocksize );              // Blocksize to be used by iterative solver
    belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
    belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
    int verbLevel = Belos::Errors + Belos::Warnings;
    if (debug) {
      verbLevel += Belos::Debug;
    }
    if (verbose) {
      verbLevel += Belos::TimingDetails + Belos::FinalSummary + Belos::StatusTestDetails;
    }
    belosList.set( "Verbosity", verbLevel );
    if (verbose) {
      if (frequency > 0) {
        belosList.set( "Output Frequency", frequency );
      }
    }
    //
    // Construct an unpreconditioned linear problem instance.
    //
    Belos::LinearProblem<ST,MV,OP> problem( normalMdagM, X, B );
    bool set = problem.setProblem();
    if (set == false) {
      if (proc_verbose)
        std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
      return -1;
    }
    //
    // *******************************************************************
    // *************Start the block CG iteration***********************
    // *******************************************************************
    //
    Belos::BlockCGSolMgr<ST,MV,OP> solver( rcp(&problem,false), rcp(&belosList,false) );

    //
    // **********Print out information about problem*******************
    //
    if (proc_verbose) {
      std::cout << std::endl << std::endl;
      std::cout << "Dimension of matrix: " << NumGlobalElements << std::endl;
      std::cout << "Number of right-hand sides: " << numrhs << std::endl;
      std::cout << "Block size used by solver: " << blocksize << std::endl;
      std::cout << "Max number of CG iterations: " << maxiters << std::endl;
      std::cout << "Relative residual tolerance: " << tol << std::endl;
      std::cout << std::endl;
    }
    //
    // Perform solve
    //
    Belos::ReturnType ret = solver.solve();
    //
    // Compute actual residuals.
    //

    bool badRes = false;
    std::vector<MT> actual_resids( numrhs );
    std::vector<MT> rhs_norm( numrhs );
    mv_type resid(map, numrhs);
    //OPT::Apply( *normalMdagM, *X, resid );
    normalMdagM->apply_cpu_wilson_clover_pc_operator(*X, resid);
    MVT::MvAddMv( -one, resid, one, *B, resid );
    MVT::MvNorm( resid, actual_resids );
    MVT::MvNorm( *B, rhs_norm );
    if (proc_verbose) {
      std::cout<< "---------- Actual Residuals (normalized) ----------"<<std::endl<<std::endl;
    }
    for ( int i=0; i<numrhs; i++) {
      MT actRes = actual_resids[i]/rhs_norm[i];
      if (proc_verbose) {
        std::cout<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
      }
      if (actRes > tol) badRes = true;
    }

    success = (ret==Belos::Converged && !badRes);

    if (success) {
      if (proc_verbose)
        std::cout << "\nEnd Result: TEST PASSED" << std::endl;
    } else {
      if (proc_verbose)
        std::cout << "\nEnd Result: TEST FAILED" << std::endl;
    }

  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  trilquda::end();

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
} // end trilinos_block_invert_test.cpp
