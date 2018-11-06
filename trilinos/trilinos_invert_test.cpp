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

#include <../test_util.h>
#include <../dslash_util.h>
#include <../wilson_dslash_reference.h>
#include <../domain_wall_dslash_reference.h>
#include <../misc.h>
#include <../gtest.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

//Trilinos(and Kokkos) headers

#include "BelosSolverManager.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosPseudoBlockCGSolMgr.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Vector.hpp"
#include "KokkosBlas1_mult.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_ParameterList.hpp"
#include <cstdlib> // EXIT_SUCCESS

/// \file wrapTpetraSolver.cpp
/// \brief Example of how to wrap a "native" solver as a Belos solver
///
/// By a "native" solver, I mean a solver written for a particular
/// linear algebra implementation.  This corresponds to a particular
/// (ScalarType, MV, and OP) Belos template parameter combination.
/// "Wrap as a Belos solver" means to make available as a
/// Belos::SolverManager subclass.
///
/// This example includes a "stub" generic implementation of the
/// Belos::SolverManager subclass, as well as the actual non-generic
/// implementation specifically for our linear algebra of choice
/// (Tpetra in this case).  In order to make this example shorter,
/// I've chosen a linear algebra implementation for which Belos has
/// MultiVecTraits and OperatorTraits specializations already
/// implemented.

static int myRank;

template<class SC, class LO, class GO, class NT>
class SolverInput {
private:
  typedef typename Teuchos::ScalarTraits<SC>::magnitudeType mag_type;

public:
  mag_type r_norm_orig;
  mag_type tol;
  int maxNumIters;
  bool needToScale;
};


/// \brief Result of a linear solve.
///
/// "A linear solve" refers either to a solve Ax=b with a single
/// right-hand side b, or to an aggregation of results of two such
/// solves with the same matrix, but different right-hand sides.
/// "Aggregation" here just means reporting a single group of metrics.
/// For example, for two solves, we report the max of the two
/// iteration counts, and the max of the two residual norms.
template<class SC, class LO, class GO, class NT>
class SolverOutput {
public:
  typedef typename Teuchos::ScalarTraits<SC>::magnitudeType mag_type;

private:
  typedef Teuchos::ScalarTraits<mag_type> STM;

public:
  //! Absolute residual norm.
  mag_type absResid;
  //! Relative residual norm (if applicable).
  mag_type relResid;
  //! Number of iterations executed.
  int numIters;
  //! Whether the solve converged.
  bool converged;

  /// \brief Default constructor.
  ///
  /// The default constructor creates output corresponding to "solving
  /// a linear system with zero right-hand sides."  This means that
  /// the solve succeeded trivially (converged == true), in zero
  /// iterations, with zero residual norm.
  SolverOutput () :
    absResid (STM::zero ()),
    relResid (STM::zero ()),
    numIters (0),
    converged (true)
  {}

  /// \brief Combine two solver outputs.
  ///
  /// "Combining" two solver outputs means aggregating the results of
  /// solving A x_1 = b_1 and A x_2 = b_2, that is, two solves with
  /// the same matrix, but different right-hand sides.  Combining is
  /// associative and commutative.
  void
  combine (const SolverOutput<SC, LO, GO, NT>& src)
  {
    // max of the residuals and iteration counts
    relResid = relResid > src.relResid ? relResid : src.relResid;
    absResid = absResid > src.absResid ? absResid : src.absResid;
    numIters = numIters > src.numIters ? numIters : src.numIters;
    // "converged" if all converged
    converged = converged && src.converged;
  }
};

template<class SC, class LO, class GO, class NT>
std::ostream&
operator<< (std::ostream& out,
            const SolverOutput<SC, LO, GO, NT>& so)
{
  using std::endl;

  out << "Solver output:" << endl
      << "  Absolute residual norm: " << so.absResid << endl
      << "  Relative residual norm: " << so.relResid << endl
      << "  Number of iterations: " << so.numIters << endl
      << "  Converged: " << (so.converged ? "true" : "false");
  return out;
}

/***********************************************QUDA STUFF*********************************************************/

using namespace quda;

//const QudaParity parity = QUDA_EVEN_PARITY; // even or odd?
const int transfer = 0; // include transfer time in the benchmark?

double kappa5;

QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cuda_prec;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

ColorSpinorParam csParam;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef, *spinorTmp;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut, *tmp1=0, *tmp2=0;

void *hostGauge[4], *hostClover, *hostCloverInv;

Dirac *dirac = nullptr;

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat, 3 = MatPCDagMatPC, 4 = MatDagMat)
// README FIRST:
// We want to solve (MdagM)^{\dagger} MdagM x = b where x and b are parity spinor. This corresponds to case 3 but 
// matrix should by generated for noramal MatDagMat

extern int test_type;

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

QudaVerbosity verbosity = QUDA_VERBOSE;

namespace trilquda{

void display_test_info()
{
  printfQuda("running the following test:\n");
 
  printfQuda("prec    recon   test_type     matpc_type   dagger   S_dim         T_dimension   Ls_dimension dslash_type    niter\n");
  printfQuda("%6s   %2s       %d           %12s    %d    %3d/%3d/%3d        %3d             %2d   %14s   %d\n", 
	     get_prec_str(prec), get_recon_str(link_recon), 
	     test_type, get_matpc_str(matpc_type), dagger, xdim, ydim, zdim, tdim, Lsdim,
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

  inv_param.mass = mass;
  inv_param.mu = mu;
  inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));
  
  inv_param.solve_type = (test_type == 2 || test_type == 4) ? QUDA_DIRECT_SOLVE : QUDA_DIRECT_PC_SOLVE;
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

  {
    switch(test_type) {
      case 0:
      case 1:
        //inv_param.solution_type = QUDA_MATPC_SOLUTION;
        //break;
      case 2:
        //inv_param.solution_type = QUDA_MAT_SOLUTION;
        //break;
        errorQuda("Test type %d not supported\n", test_type);
      case 3:
        inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
        break;
      case 4:
        //inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION;
        //break;
      default:
        errorQuda("Test type %d not defined\n", test_type);
    }
  }

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

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc((size_t)V*gaugeSiteSize*gauge_param.cpu_prec);
  
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = gauge_param.X[d];
  csParam.PCtype = QUDA_5D_PC;

  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;

  {
    if (test_type < 2 || test_type == 3) {//we use type 3
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
    } else {
      csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    }
  }

  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;

///Create a reference fields?

  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);
  spinorTmp = new cpuColorSpinorField(csParam);

  csParam.x[0] = gauge_param.X[0];
  
  printfQuda("Randomizing fields... ");

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, hostGauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(hostGauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate a random SU(3) field
    construct_gauge_field(hostGauge, 1, gauge_param.cpu_prec, &gauge_param);
  }

  spinor->Source(QUDA_RANDOM_SOURCE, 0);

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

  if (!transfer) {
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.precision = inv_param.cuda_prec;
    if (csParam.precision == QUDA_DOUBLE_PRECISION ) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    } else {
      /* Single and half */
      csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    }
 
    if (test_type < 2 || test_type == 3) {//we use type 3
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
    }

    printfQuda("Creating cudaSpinor\n");
    cudaSpinor = new cudaColorSpinorField(csParam);
    printfQuda("Creating cudaSpinorOut\n");
    cudaSpinorOut = new cudaColorSpinorField(csParam);

    tmp1 = new cudaColorSpinorField(csParam);

    if (test_type == 2 || test_type == 4) csParam.x[0] /= 2;// nop here

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp2 = new cudaColorSpinorField(csParam);

    printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;
    
    double cpu_norm = blas::norm2(*spinor);
    double cuda_norm = blas::norm2(*cudaSpinor);
    printfQuda("Source: CPU = %e, CUDA = %e\n", cpu_norm, cuda_norm);

    //WARNING : we need always normal (preconditioned) system, for a parity linear system, test_type =3
    bool pc = true;//(test_type != 2 && test_type != 4);

    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    diracParam.tmp1 = tmp1;
    diracParam.tmp2 = tmp2;
   
    dirac = Dirac::create(diracParam);

  } else {
    double cpu_norm = blas::norm2(*spinor);
    printfQuda("Source: CPU = %e\n", cpu_norm);
  }
//Now we can initialize Tpetra:
  Tpetra::initialize (&argc, &argv);
    
}

void end() {
  
//First, finalize Tpetra:
  Tpetra::finalize ();
//Now clean local resources:
  if (!transfer) {
    if(dirac != NULL)
    {
      delete dirac;
      dirac = NULL;
    }
    delete cudaSpinor;
    delete cudaSpinorOut;
    delete tmp1;
    delete tmp2;
  }

  // release memory
  delete spinor;
  delete spinorOut;
  delete spinorRef;
  delete spinorTmp;

  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    free(hostClover);
    free(hostCloverInv);
  }
//Finalize QUDA:
  endQuda();

}

/// \brief Tpetra implementation of CG
///
/// This CG implementation can solve linear systems with multiple
/// right-hand sides, but it solves them one right-hand side at a
/// time.  The reported convergence results in the case of multiple
/// right-hand sides is the max of the residuals and iteration counts,
/// and the AND of the "did it converge" Boolean values.


template<class SC = Tpetra::Operator<>::scalar_type,
         class LO = typename Tpetra::Operator<SC>::local_ordinal_type,
         class GO = typename Tpetra::Operator<SC, LO>::global_ordinal_type,
         class NT = typename Tpetra::Operator<SC, LO, GO>::node_type>
class CG {
public:
  typedef Tpetra::Operator<SC, LO, GO, NT> op_type;
  typedef Tpetra::MultiVector<SC, LO, GO, NT> mv_type;

private:
  typedef Tpetra::Vector<SC, LO, GO, NT> vec_type;
  typedef Teuchos::ScalarTraits<SC> STS;
  typedef typename STS::magnitudeType mag_type;
  typedef Teuchos::ScalarTraits<mag_type> STM;
  typedef typename NT::device_type device_type;

public:
  CG () :
    tol_ (STM::squareroot (STS::eps ())),
    maxNumIters_ (100),
    verbosity_ (0),
    needToScale_ (true),
    relResid_ (STM::zero ()),
    absResid_ (STM::zero ()),
    numIters_ (0),
    converged_ (false)
  {}

  CG (const Teuchos::RCP<const op_type>& A) :
    A_ (A),
    tol_ (STM::squareroot (STS::eps ())),
    maxNumIters_ (100),
    verbosity_ (0),
    needToScale_ (true),
    relResid_ (STM::zero ()),
    absResid_ (STM::zero ()),
    numIters_ (0),
    converged_ (false)
  {}

  //! Set the matrix A in the linear system to solve.
  void setMatrix (const Teuchos::RCP<const op_type>& A) {
    if (A_.getRawPtr () != A.getRawPtr ()) {
      A_ = A;
    }
  }

  Teuchos::RCP<const op_type> getMatrix () const {
    return A_;
  }

  /// \brief Fill \c params with all parameters this solver
  ///   understands, and either their current values, or their default
  ///   values.
  ///
  /// \param params [out] To be filled with this solver's parameters.
  /// \param defaultValues [in] Whether to use default values (true)
  ///   or current values (false).
  void
  getParameters (Teuchos::ParameterList& params,
                 const bool defaultValues) const
  {
    // Yes, the inner STS is supposed to be STS.  STS::eps() returns
    // mag_type.  It's SC's machine epsilon.
    const mag_type tol =
      defaultValues ? STM::squareroot (STS::eps ()) : tol_;
    const int maxNumIters = defaultValues ? 100 : maxNumIters_;
    const int verbosity = defaultValues ? 0 : verbosity_;
    const bool needToScale = defaultValues ? true : needToScale_;
    const std::string implResScal = needToScale ?
      "Norm of Preconditioned Initial Residual" :
      "None";

    params.set ("Convergence Tolerance", tol);
    params.set ("Implicit Residual Scaling", implResScal);
    params.set ("Maximum Iterations", maxNumIters);
    params.set ("Verbosity", verbosity);
  }

  /// \brief Set the solver's parameters.
  ///
  /// This solver takes a subset of the parameters that
  /// Belos::PseudoBlockCGSolMgr (Belos' CG implementation) takes, and
  /// ignores the rest.  If it takes a parameter but doesn't implement
  /// all that parameter's options, it throws an exception if it
  /// encounters an option that it does not implement.  The point is
  /// compatibility with Belos.
  void setParameters (Teuchos::ParameterList& params) {
    mag_type tol = tol_;
    bool needToScale = needToScale_;
    if (params.isParameter ("Convergence Tolerance")) {
      tol = params.get<mag_type> ("Convergence Tolerance");
      TEUCHOS_TEST_FOR_EXCEPTION
        (tol < STM::zero (), std::invalid_argument,
         "\"Convergence tolerance\" = " << tol << " < 0.");
    }
    if (params.isParameter ("Implicit Residual Scaling")) {
      const std::string implScal =
        params.get<std::string> ("Implicit Residual Scaling");
      if (implScal == "Norm of Initial Residual") {
        // FIXME (mfh 26 Oct 2016) Once we implement left
        // preconditioning, we'll have to keep separate preconditioned
        // and unpreconditioned absolute residuals.
        needToScale = true;
      }
      else if (implScal == "Norm of Preconditioned Initial Residual") {
        needToScale = true;
      }
      else if (implScal == "Norm of RHS") {
        // FIXME (mfh 26 Oct 2016) If we want to implement this, it
        // would make sense to combine that all-reduce with the
        // all-reduce for computing the initial residual norms.  We
        // could modify computeResiduals to have an option to do this.
        TEUCHOS_TEST_FOR_EXCEPTION
          (true, std::logic_error,
           "\"Norm of RHS\" scaling option not implemented");
      }
      else if (implScal == "None") {
        needToScale = false;
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION
          (true, std::invalid_argument, "\"Implicit Residual Scaling\""
           " has an invalid value \"" << implScal << "\".");
      }
    }

    int maxNumIters = maxNumIters_;
    if (params.isParameter ("Maximum Iterations")) {
      maxNumIters = params.get<int> ("Maximum Iterations");
      TEUCHOS_TEST_FOR_EXCEPTION
        (maxNumIters < 0, std::invalid_argument,
         "\"Maximum Iterations\" = " << maxNumIters << " < 0.");
    }

    int verbosity = verbosity_;
    if (params.isType<int> ("Verbosity")) {
      verbosity = params.get<int> ("Verbosity");
    }
    else if (params.isType<bool> ("Verbosity")) {
      const bool verbBool = params.get<bool> ("Verbosity");
      verbosity = verbBool ? 1 : 0;
    }

    tol_ = tol;
    maxNumIters_ = maxNumIters;
    verbosity_ = verbosity;
    needToScale_ = needToScale;
  }

private:
  static void
  computeResiduals (Kokkos::DualView<mag_type*, device_type>& norms,
                    mv_type& R,
                    const op_type& A,
                    const mv_type& X,
                    const mv_type& B)
  {
    typedef typename device_type::memory_space dev_mem_space;

    const SC ONE = STS::one ();
    A.apply (X, R);
    R.update (ONE, B, -ONE); // R := B - A*X

    norms.template modify<dev_mem_space> ();
    Kokkos::View<mag_type*, device_type> norms_d =
      norms.template view<dev_mem_space> ();
    R.norm2 (norms_d);
  }

  static mag_type
  computeResidual (vec_type& R,
                   const op_type& A,
                   const vec_type& X,
                   const vec_type& B)
  {
    const SC ONE = STS::one ();
    A.apply (X, R);
    R.update (ONE, B, -ONE); // R := B - A*X
    const mag_type r_norm = R.norm2 ();
    return r_norm;
  }

public:
  //! Solve the linear system(s) AX=B.
  SolverOutput<SC, LO, GO, NT>
  solve (mv_type& X, const mv_type& B)
  {
    using Teuchos::FancyOStream;
    using Teuchos::getFancyOStream;
    using Teuchos::oblackholestream;
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::rcpFromRef;

    TEUCHOS_TEST_FOR_EXCEPTION
      (A_.is_null (), std::runtime_error, "Matrix A is null.  Please call "
       "setMatrix() with a nonnull argument before calling solve().");
    TEUCHOS_TEST_FOR_EXCEPTION
      (X.getNumVectors () != B.getNumVectors (), std::runtime_error,
       "X.getNumVectors() = " << X.getNumVectors () <<
       " != B.getNumVectors() = " << B.getNumVectors () << ".");

    RCP<FancyOStream> outPtr;
    if (verbosity_) {
      const int myRank = A_->getDomainMap ()->getComm ()->getRank ();
      if (myRank == 0) {
        outPtr = getFancyOStream (rcpFromRef (std::cout));
      }
      else {
        outPtr = getFancyOStream (rcp (new oblackholestream ()));
      }
    }
    return solveImpl (outPtr.getRawPtr (), X, B);
  }

private:
  //! Solve the linear system(s) AX=B.
  SolverOutput<SC, LO, GO, NT>
  solveImpl (Teuchos::FancyOStream* outPtr, mv_type& X, const mv_type& B)
  {
    using Teuchos::RCP;
    using std::endl;

    const size_t numVecs = B.getNumVectors ();
    Kokkos::DualView<mag_type*, device_type> norms ("norms", numVecs);
    mv_type R (B.getMap (), numVecs);

    computeResiduals (norms, R, *A_, X, B);
    norms.template sync<Kokkos::HostSpace> ();
    auto norms_h = norms.template view<Kokkos::HostSpace> ();

    SolverInput<SC, LO, GO, NT> input;
    input.tol = tol_;
    input.maxNumIters = maxNumIters_;
    input.needToScale = needToScale_;
    SolverOutput<SC, LO, GO, NT> allOutput;

    for (size_t j = 0; j < numVecs; ++j) {
      if (outPtr != NULL) {
        *outPtr << "Solve for column " << (j+1) << " of " << numVecs << ":" << endl;
        outPtr->pushTab ();
      }
      RCP<vec_type> R_j = R.getVectorNonConst (j);
      RCP<vec_type> X_j = X.getVectorNonConst (j);
      input.r_norm_orig = norms_h(j);
      SolverOutput<SC, LO, GO, NT> curOutput;
      solveOneVec (outPtr, curOutput, *X_j, *R_j, *A_, input);
      allOutput.combine (curOutput);
      if (outPtr != NULL) {
        outPtr->popTab ();
      }
    }
    return allOutput;
  }

  static mag_type
  getConvergenceMetric (const mag_type r_norm_new,
                        const SolverInput<SC, LO, GO, NT>& input)
  {
    if (input.needToScale) {
      return input.r_norm_orig == STM::zero () ?
        r_norm_new :
        (r_norm_new / input.r_norm_orig);
    }
    else {
      return r_norm_new;
    }
  }

  static void
  solveOneVec (Teuchos::FancyOStream* outPtr,
               SolverOutput<SC, LO, GO, NT>& output,
               vec_type& X, // in/out
               vec_type& R, // in/out
               const op_type& A,
               const SolverInput<SC, LO, GO, NT>& input)
  {
    using std::endl;

    const SC ONE = STS::one ();

    vec_type P (R, Teuchos::Copy);
    vec_type AP (R.getMap ());
    mag_type r_norm_old = input.r_norm_orig; // R.norm2 ()

    if (r_norm_old == STM::zero () || r_norm_old <= input.tol) {
      if (outPtr != NULL) {
        *outPtr << "Initial guess' residual norm " << r_norm_old
                << " meets tolerance " << input.tol << endl;
      }
      output.absResid = r_norm_old;
      output.relResid = r_norm_old;
      output.numIters = 0;
      output.converged = true;
      return;
    }

    mag_type r_norm_new = STM::zero ();
    for (int iter = 0; iter < input.maxNumIters; ++iter) {
      if (outPtr != NULL) {
        *outPtr << "Iteration " << (iter+1) << " of " << input.maxNumIters << ":" << endl;
        outPtr->pushTab ();
        *outPtr << "r_norm_old: " << r_norm_old << endl;
      }

      A.apply (P, AP);
      const mag_type PAP = P.dot (AP);
      if (outPtr != NULL) {
        *outPtr << "PAP: " << PAP << endl;
      }
      TEUCHOS_TEST_FOR_EXCEPTION
        (PAP <= STM::zero (), std::runtime_error, "At iteration " << (iter+1)
         << " out of " << input.maxNumIters << ", P.dot(AP) = " << PAP <<
         " <= 0.  This usually means that the matrix A is not symmetric "
         "(Hermitian) positive definite.");

      const mag_type alpha = (r_norm_old * r_norm_old) / PAP;
      if (outPtr != NULL) {
        *outPtr << "alpha: " << alpha << endl;
      }
      X.update (static_cast<SC> (alpha), P, ONE);
      R.update (static_cast<SC> (-alpha), AP, ONE);

      r_norm_new = R.norm2 ();
      if (outPtr != NULL) {
        *outPtr << "r_norm_new: " << r_norm_new << endl;
      }
      const mag_type metric = getConvergenceMetric (r_norm_new, input);
      if (outPtr != NULL) {
        *outPtr << "metric: " << metric << endl;
      }
      if (metric <= input.tol) {
        output.absResid = r_norm_new;
        output.relResid = input.r_norm_orig == STM::zero () ?
          r_norm_new :
          (r_norm_new / input.r_norm_orig);
        output.numIters = iter + 1;
        output.converged = true;
        return;
      }
      else if (iter + 1 < input.maxNumIters) { // not last iteration
        const mag_type beta = (r_norm_new * r_norm_new) /
          (r_norm_old * r_norm_old);
        P.update (ONE, R, static_cast<SC> (beta));
        r_norm_old = r_norm_new;
      }

      if (outPtr != NULL) {
        outPtr->popTab ();
      }
    }

    // Reached max iteration count without converging
    output.absResid = r_norm_new;
    output.relResid = input.r_norm_orig == STM::zero () ?
      r_norm_new :
      (r_norm_new / input.r_norm_orig);
    output.numIters = input.maxNumIters;
    output.converged = false;
  }

private:
  Teuchos::RCP<const op_type> A_;

  mag_type tol_;
  int maxNumIters_;
  int verbosity_;
  // FIXME (mfh 26 Oct 2016) Once we implement left preconditioning,
  // we'll have to keep separate preconditioned and unpreconditioned
  // absolute residuals.
  bool needToScale_;

  mag_type relResid_;
  mag_type absResid_;
  int numIters_;
  bool converged_;
};


template<class ScalarType, class MV, class OP>
class CgWrapper :
  public Belos::SolverManager<ScalarType, MV, OP>
{
public:
  virtual ~CgWrapper () {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented");
  }

  const Belos::LinearProblem<ScalarType,MV,OP>& getProblem () const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented");
  }

  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters () const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented");
  }

  Teuchos::RCP<const Teuchos::ParameterList> getCurrentParameters () const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented");
  }

  int getNumIters () const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented");
  }

  bool isLOADetected () const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented");
  }

  void setProblem (const Teuchos::RCP<Belos::LinearProblem<ScalarType,MV,OP> >& problem) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented");
  }

  void setParameters (const Teuchos::RCP<Teuchos::ParameterList>& params) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented");
  }

  void reset (const Belos::ResetType type) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented");
  }

  Belos::ReturnType solve () {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented");
  }
};

template<class SC, class LO, class GO, class NT>
class CgWrapper<SC,
                Tpetra::MultiVector<SC, LO, GO, NT>,
                Tpetra::Operator<SC, LO, GO, NT> > :
  public Belos::SolverManager<SC,
                              Tpetra::MultiVector<SC, LO, GO, NT>,
                              Tpetra::Operator<SC, LO, GO, NT> >
{
public:
  typedef Tpetra::MultiVector<SC, LO, GO, NT> MV;
  typedef Tpetra::Operator<SC, LO, GO, NT> OP;
  typedef Belos::LinearProblem<SC, MV, OP> belos_problem_type;

  virtual ~CgWrapper () {}

  const Belos::LinearProblem<SC,MV,OP>& getProblem () const {
    TEUCHOS_TEST_FOR_EXCEPTION
      (problem_.is_null (), std::runtime_error, "The linear problem has not "
       "yet been set.  Please call setProblem with a nonnull argument before "
       "calling this method.");
    return *problem_;
  }

  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters () const {
    using Teuchos::ParameterList;
    using Teuchos::RCP;

    RCP<ParameterList> params (new ParameterList ("CG"));
    const bool defaultValues = true;
    solver_.getParameters (*params, defaultValues);
    return params;
  }

  Teuchos::RCP<const Teuchos::ParameterList> getCurrentParameters () const {
    using Teuchos::ParameterList;
    using Teuchos::RCP;

    RCP<ParameterList> params (new ParameterList ("CG"));
    const bool defaultValues = false;
    solver_.getParameters (*params, defaultValues);
    return params;
  }

  int getNumIters () const {
    return lastSolverOutput_.numIters;
  }

  bool isLOADetected () const {
    return false; // this solver doesn't attempt to detect loss of accuracy
  }

  void
  setProblem (const Teuchos::RCP<belos_problem_type>& problem)
  {
    if (problem.is_null ()) {
      solver_.setMatrix (Teuchos::null);
    }
    else if (solver_.getMatrix ().getRawPtr () !=
             problem->getOperator ().getRawPtr ()) {
      // setMatrix resets state, so only call if necessary.
      solver_.setMatrix (problem->getOperator ());
    }
    problem_ = problem;
  }

  void setParameters (const Teuchos::RCP<Teuchos::ParameterList>& params) {
    if (! params.is_null ()) {
      solver_.setParameters (*params);
    }
  }

  void reset (const Belos::ResetType /* type */ ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented");
  }

  Belos::ReturnType solve () {
    using Teuchos::RCP;
    TEUCHOS_TEST_FOR_EXCEPTION
      (problem_.is_null (), std::runtime_error, "The linear problem has not "
       "yet been set.  Please call setProblem with a nonnull argument before "
       "calling this method.");
    RCP<const MV> B = problem_->getRHS ();
    TEUCHOS_TEST_FOR_EXCEPTION
      (B.is_null (), std::runtime_error, "The linear problem's right-hand "
       "side(s) B has/have not yet been set.  Please call setProblem with "
       "a nonnull argument before calling this method.");
    RCP<MV> X = problem_->getLHS ();
    TEUCHOS_TEST_FOR_EXCEPTION
      (X.is_null (), std::runtime_error, "The linear problem's left-hand "
       "side(s) X has/have not yet been set.  Please call setProblem with "
       "a nonnull argument before calling this method.");
    SolverOutput<SC, LO, GO, NT> result = solver_.solve (*X, *B);
    lastSolverOutput_ = result;
    return result.converged ? Belos::Converged : Belos::Unconverged;
  }

private:
  typedef SolverOutput<SC, LO, GO, NT> output_type;

  CG<SC, LO, GO, NT> solver_;
  //! Output of the last solve.
  ///
  /// This does not include the solution (multi)vector, just things
  /// like the residual nor, the iteration count, and whether the
  /// solve converged.  See SolverOutput documentation above for
  /// details.
  output_type lastSolverOutput_;
  Teuchos::RCP<belos_problem_type> problem_;
};

} //end of trilquda


/*****************************The kernels********************************/


/*************
1.
https://trilinos.org/docs/dev/packages/tpetra/doc/html/namespaceTpetra.html
2.
packages/tpetra/core/test/PerformanceCGSolve
**************/

#define DEBUG

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

  QUDA_WilsonClover_PC_Operator (const Teuchos::RCP<const map_type>& map, size_t numvecs = 1) :
    map_ (map), verbose_(true), numvecs_(numvecs) { }

  Teuchos::RCP<const map_type> getDomainMap () const { return map_; }

  Teuchos::RCP<const map_type> getRangeMap () const { return map_; }

  void SetVerbose(bool verbose) {verbose_ = verbose;}
  void SetNumvecs(size_t n)     {numvecs_ = n;}

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
      (numvecs_ != X.getNumVectors () and numvecs_ != Y.getNumVectors (),
       std::logic_error, "Block vector size does not match.");

    TEUCHOS_TEST_FOR_EXCEPTION
      ((inv_param.solution_type != QUDA_MATPC_SOLUTION) and (inv_param.solution_type != QUDA_MATPCDAG_MATPC_SOLUTION),
      std::logic_error, "This operator should be used for the preconditioned system.");

////Print input norms:
    Teuchos::Array<mag_type> X_norms (numvecs_);
    Teuchos::Array<mag_type> Y_norms (numvecs_);

    X.norm2 (X_norms ());

    if(verbose_){
      if (myRank == 0) {
        for (size_t j = 0; j < numvecs_; ++j) {
          std::cout << "Input array for column " << (j+1) << " of " << numvecs_
             << ": X norm: " << X_norms[j]
             << std::endl;
        }
      }
    }

    auto trilinos_input_norm = X_norms[0];

////
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

      cudaPointerAttributes ptr_attr;
      if(cudaPointerGetAttributes(&ptr_attr, X_j_lcl_ptr) == cudaErrorInvalidValue) errorQuda("For X_j_lcl_ptr, a pointer was not allocated in, mapped by or registered with current CUDA context.\n");

      if ( ptr_attr.memoryType == cudaMemoryTypeDevice )
        warningQuda("Device allocation. (%d, is managed : %d)\n", ptr_attr.memoryType, ptr_attr.isManaged);
      else 
        warningQuda("Host allocation.(%d)\n", ptr_attr.memoryType);
 
#ifndef DEBUG

      MatDagMatQuda(Y_j_lcl_ptr, X_j_lcl_ptr, &inv_param);

#else

      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(&inv_param);

      ColorSpinorParam cpuParam(X_j_lcl_ptr, inv_param, gauge_param.X, /*pc=*/ true, inv_param.input_location);
      ColorSpinorField *in_h = ColorSpinorField::Create(cpuParam);

      ColorSpinorParam cudaParam(cpuParam, inv_param);
      cudaColorSpinorField in(*in_h, cudaParam);
//
#if 0
csParam.create = QUDA_REFERENCE_FIELD_CREATE;
csParam.v = X_j_lcl_ptr;
csParam.norm = nullptr;
csParam.mem_type=QUDA_MEMORY_EXTERNAL;

if(cudaPointerGetAttributes(&ptr_attr, csParam.v) == cudaErrorInvalidValue) errorQuda("For csParam.v, a pointer was not allocated in, mapped by or registered with current CUDA context.\n");

if ( ptr_attr.memoryType == cudaMemoryTypeDevice )
   warningQuda("Device allocation. (%d, is managed : %d)\n", ptr_attr.memoryType, ptr_attr.isManaged);
else 
   warningQuda("Host allocation.(%d)\n", ptr_attr.memoryType);

cudaColorSpinorField in2(csParam);
std::cout << in2 << std::endl;
#endif
//
      auto quda_input_norm = sqrt(blas::norm2(in));

      auto input_norm_diff = fabs(quda_input_norm - trilinos_input_norm); 

      if (verbose_) printfQuda("Converted input QUDA %e (trilinos/quda %e)\n", quda_input_norm, input_norm_diff);

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
      auto quda_output_norm = sqrt(blas::norm2(out));

      delete out_h;
      delete in_h;

      Y.norm2 (Y_norms ());
////Print input norms:
      if(verbose_){
        if (myRank == 0) {
          for (size_t j = 0; j < numvecs_; ++j) {
            std::cout << "Output norm :"
               << ", Y norm: " << Y_norms[j]
               << std::endl;
          }
        }
      }

      auto trilinos_output_norm = Y_norms[0];
      auto output_norm_diff = fabs(quda_output_norm - trilinos_output_norm); 

      if (verbose_) printfQuda("Output QUDA %e (trilinos/quda %e)\n", quda_output_norm, output_norm_diff);
#endif
    }
  }//end apply

//wil_matpc(void *out, void **gauge, void *in, double kappa, QudaMatPCType matpc_type,  int daggerBit, QudaPrecision precision, QudaGaugeParam &param);

  void apply_cpu_wilson_clover_pc_operator(const Tpetra::MultiVector<SC, LO, GO, NT>& X,
              Tpetra::MultiVector<SC, LO, GO, NT>& Y) {
    using Teuchos::RCP;
    typedef Tpetra::MultiVector<SC, LO, GO, NT> MV;

    MV R (X.getMap (), X.getNumVectors ());

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
  size_t numvecs_;
};


template<class Node> 
int run ( ) {

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::tuple;
  using std::cout;
  using std::endl;

  typedef Tpetra::Vector<>::scalar_type                 Scalar;
  typedef typename Tpetra::Map<>::local_ordinal_type    LO;
  typedef typename Tpetra::Map<>::global_ordinal_type   GO;

//  typedef Tpetra::MpiPlatform<Node>                     Platform;
//  typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node>          crs_matrix_type;
  typedef Tpetra::Vector<Scalar,LO,GO,Node>             vec_type;
  typedef Tpetra::Map<LO,GO,Node>                       map_type;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node>        mv_type;
  typedef Tpetra::Operator<Scalar,LO,GO,Node> 	        op_type;
  typedef Belos::LinearProblem<Scalar,mv_type,op_type>  belos_problem_type;
  typedef typename vec_type::mag_type                   mag_type;

  typedef Teuchos::ScalarTraits<Scalar> STS;
  typedef Teuchos::ScalarTraits<mag_type> STM;

  const Scalar ZERO = STS::zero ();
  const Scalar ONE = STS::one ();

  auto comm = Tpetra::getDefaultComm ();

  myRank = comm->getRank ();

  const GO gblNumRows = cudaSpinor->Length();
  const GO indexBase  = 0;

  RCP<const map_type> map (new map_type (gblNumRows, indexBase, comm));

  size_t numvecs = 1;

  RCP<QUDA_WilsonClover_PC_Operator<Scalar, LO, GO, Node> > NormalMdagM = rcp (new QUDA_WilsonClover_PC_Operator<Scalar, LO, GO, Node> (map, numvecs));

  mv_type X (NormalMdagM->getDomainMap (), numvecs);
  mv_type B (NormalMdagM->getRangeMap (), numvecs);

  B.randomize ();

  {
    Teuchos::Array<mag_type> B_norms (B.getNumVectors ());
    B.norm2 (B_norms ());
    if (myRank == 0) {
      for (size_t j = 0; j < B.getNumVectors (); ++j) {
        cout << "Column " << (j+1) << " of " << B.getNumVectors ()
             << ": Source norm: " << B_norms[j]
             << endl;
      }
      cout << endl;
    }

    //std::cout << "Array size: " << gblNumRows << " Parity : " << cudaSpinor->SiteSubset() << " X[0] : " << cudaSpinor->X(0) << std::endl;
    //std::cout << "CSPARAM data" << std::endl;
    //csParam.print();
  }

  NormalMdagM->SetVerbose(true);

  trilquda::CG<Scalar, LO, GO, Node> solver (NormalMdagM);
  auto out = solver.solve (X, B);

  if (myRank == 0) {
    cout << "Stage 1" << endl;
    cout << out << endl;
  }

  {
    // Check the residual.
    mv_type X_copy (X, Teuchos::Copy);
    mv_type R (B.getMap (), B.getNumVectors ());

    //NormalMdagM->apply (X_copy, R);
    NormalMdagM->apply_cpu_wilson_clover_pc_operator(X_copy, R);

    R.update (ONE, B, -ONE);
    Teuchos::Array<mag_type> R_norms (R.getNumVectors ());
    R.norm2 (R_norms ());
    Teuchos::Array<mag_type> B_norms (B.getNumVectors ());
    B.norm2 (B_norms ());
    if (myRank == 0) {
      for (size_t j = 0; j < R.getNumVectors (); ++j) {
        const mag_type relResNorm = (B_norms[j] == STM::zero ()) ?
          R_norms[j] :
          R_norms[j] / B_norms[j];
        cout << "Column ??" << (j+1) << " of " << R.getNumVectors ()
             << ": Absolute residual norm: " << R_norms[j]
             << ", Relative residual norm: " << relResNorm
             << endl;
      }
      cout << endl;
    }
  }


#if 0
  {
    // Get ready for next solve by resetting initial guess to zero.
    X.putScalar (ZERO);

    RCP<Belos::LinearProblem<Scalar, mv_type, op_type> > lp = rcp (new belos_problem_type (NormalMdagM, rcpFromRef (X), rcpFromRef (B)));
    // Our CG implementation bypasses this, but we call it anyway, just
    // for interface consistency with Belos' other solvers.
    lp->setProblem ();

    trilquda::CgWrapper<Scalar, mv_type, op_type> solverWrapper;
    solverWrapper.setProblem (lp);
    const Belos::ReturnType belosResult = solverWrapper.solve ();
    if (myRank == 0) {
cout << "Stage 2" << endl;
      cout << "Belos solver wrapper result: "
           << (belosResult == Belos::Converged ? "Converged" : "Unconverged")
           << endl
           << "Number of iterations: " << solverWrapper.getNumIters ()
           << endl;
    }
  }

  {
    // Check the residual.
    mv_type X_copy (X, Teuchos::Copy);
    mv_type R (B.getMap (), B.getNumVectors ());

    NormalMdagM->apply (X_copy, R);

    R.update (ONE, B, -ONE);
    Teuchos::Array<mag_type> R_norms (R.getNumVectors ());
    R.norm2 (R_norms ());
    Teuchos::Array<mag_type> B_norms (B.getNumVectors ());
    B.norm2 (B_norms ());
    if (myRank == 0) {
      for (size_t j = 0; j < R.getNumVectors (); ++j) {
        const mag_type relResNorm = (B_norms[j] == STM::zero ()) ?
          R_norms[j] :
          R_norms[j] / B_norms[j];
        cout << "Column " << (j+1) << " of " << R.getNumVectors ()
             << ": Absolute residual norm: " << R_norms[j]
             << ", Relative residual norm: " << relResNorm
             << endl;
      }
      cout << endl;
    }
  }
#endif
#if 0
  {
    numvecs = 1;

    NormalMdagM->SetNumvecs(numvecs);

    mv_type blockX (NormalMdagM->getDomainMap (), numvecs);
    mv_type blockB (NormalMdagM->getRangeMap (), numvecs);

    blockB.randomize ();

//!  Belos::PseudoBlockCGSolMgr<SC, mv_type, op_type> belosSolver;
    Belos::BlockCGSolMgr<Scalar, mv_type, op_type> belosSolver;

    // Prepare for next linear solve by resetting initial guess to zero.
    blockX.putScalar (ZERO);

    RCP<Belos::LinearProblem<Scalar, mv_type, op_type> > lp = rcp (new belos_problem_type (NormalMdagM, rcpFromRef (blockX), rcpFromRef (blockB)));
    // Our CG implementation bypasses this, but we call it anyway, just
    // for interface consistency with Belos' other solvers.
    lp->setProblem ();

    belosSolver.setProblem (lp);
    const Belos::ReturnType belosResult = belosSolver.solve ();
    if (myRank == 0) {
      cout << "Stage 3" << endl;
      cout << "Belos solver (PseudoBlockCGSolMgr) result: "
           << (belosResult == Belos::Converged ? "Converged" : "Unconverged")
           << endl
           << "Number of iterations: " << belosSolver.getNumIters ()
           << endl;
    }

    // Check the residual.
    mv_type blockX_copy (blockX, Teuchos::Copy);
    mv_type blockR (blockB.getMap (), blockB.getNumVectors ());

    NormalMdagM->apply (blockX_copy, blockR);

    blockR.update (ONE, blockB, -ONE);
    Teuchos::Array<mag_type> blockR_norms (blockR.getNumVectors ());
    blockR.norm2 (blockR_norms ());
    Teuchos::Array<mag_type> blockB_norms (blockB.getNumVectors ());
    blockB.norm2 (blockB_norms ());
    if (myRank == 0) {
      for (size_t j = 0; j < blockR.getNumVectors (); ++j) {
        const mag_type relResNorm = (blockB_norms[j] == STM::zero ()) ?
          blockR_norms[j] :
          blockR_norms[j] / blockB_norms[j];
        cout << "Column " << (j+1) << " of " << blockR.getNumVectors ()
             << ": Absolute residual norm: " << blockR_norms[j]
             << ", Relative residual norm: " << relResNorm
             << endl;
      }
    }
  }
#endif

  return 0;  
}

extern void usage(char**);

typedef Kokkos::Compat::KokkosCudaWrapperNode  cuda_Node;
//typedef Kokkos::Compat::KokkosOpenMPWrapperNode  openmp_Node;
//typedef Kokkos::Compat::KokkosSerialWrapperNode  serial_Node;

int main (int argc, char* argv[])
{
   // initalize google test, includes command line options
   ::testing::InitGoogleTest(&argc, argv);

   for (int i =1;i < argc; i++) {
     if(process_command_line_option(argc, argv, &i) == 0) continue;
       
     fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
     usage(argv);
   }

   trilquda::initialize(argc, argv);

   auto check_n = run<cuda_Node> ();

   trilquda::end( );

   return EXIT_SUCCESS;
}
