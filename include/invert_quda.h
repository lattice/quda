#pragma once

#include <vector>
#include <memory>
#include <quda.h>
#include <quda_internal.h>
#include <timer.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>
#include <qio_field.h>
#include <eigensolve_quda.h>
#include <invert_x_update.h>
#include <madwf_param.h>

namespace quda {

  /**
     SolverParam is the meta data used to define linear solvers.
   */
  struct SolverParam {

    /**
       Which linear solver to use
    */
    QudaInverterType inv_type;

    /**
     * The inner Krylov solver used in the preconditioner.  Set to
     * QUDA_INVALID_INVERTER to disable the preconditioner entirely.
     */
    QudaInverterType inv_type_precondition;

    /**
     * Preconditioner instance, e.g., multigrid
     */
    void *preconditioner;

    /**
     * Deflation operator
     */
    void *deflation_op;

    /**
     * Whether to use the L2 relative residual, L2 absolute residual
     * or Fermilab heavy-quark residual, or combinations therein to
     * determine convergence.  To require that multiple stopping
     * conditions are satisfied, use a bitwise OR as follows:
     *
     * p.residual_type = (QudaResidualType) (QUDA_L2_RELATIVE_RESIDUAL
     *                                     | QUDA_HEAVY_QUARK_RESIDUAL);
     */
    QudaResidualType residual_type;

    /**< Whether deflate the initial guess */
    bool deflate;

    /**< Used to define deflation */
    QudaEigParam eig_param;

    /**< Whether to use an initial guess in the solver or not */
    QudaUseInitGuess use_init_guess;

    /**< Whether to solve linear system with zero RHS */
    QudaComputeNullVector compute_null_vector;

    /**< Reliable update tolerance */
    double delta;

    /**< Whether to user alternative reliable updates (CG only at the moment) */
    bool use_alternative_reliable;

    /**< Whether to keep the partial solution accumulator in sloppy precision */
    bool use_sloppy_partial_accumulator;

    /**< This parameter determines how often we accumulate into the
       solution vector from the direction vectors in the solver.
       E.g., running with solution_accumulator_pipeline = 4, means we
       will update the solution vector every four iterations using the
       direction vectors from the prior four iterations.  This
       increases performance of mixed-precision solvers since it means
       less high-precision vector round-trip memory travel, but
       requires more low-precision memory allocation. */
    int solution_accumulator_pipeline;

    /**< This parameter determines how many consecutive reliable update
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge */
    int max_res_increase;

    /**< This parameter determines how many total reliable update
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge */
    int max_res_increase_total;

    /**< This parameter determines how many consecutive heavy-quark
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge */
    int max_hq_res_increase;

    /**< This parameter determines how many total heavy-quark residual
    restarts we tolerate before terminating the solver, i.e., how long
    do we want to keep trying to converge */
    int max_hq_res_restart_total;

    /**< After how many iterations shall the heavy quark residual be updated */
    int heavy_quark_check;

    /**< Enable pipeline solver */
    int pipeline;

    /**< Solver tolerance in the L2 residual norm */
    double tol;

    /**< Solver tolerance in the L2 residual norm */
    double tol_restart;

    /**< Solver tolerance in the heavy quark residual norm */
    double tol_hq;

    /**< Whether to compute the true residual post solve */
    bool compute_true_res;

    /** Whether to declare convergence without checking the true residual */
    bool sloppy_converge;

    /**< Actual L2 residual norm achieved in solver */
    double true_res;

    /**< Actual heavy quark residual norm achieved in solver */
    double true_res_hq;

    /**< Maximum number of iterations in the linear solver */
    int maxiter;

    /**< The number of iterations performed by the solver */
    int iter;

    /**< The precision used by the QUDA solver */
    QudaPrecision precision;

    /**< The precision used by the QUDA sloppy operator */
    QudaPrecision precision_sloppy;

    /**< The precision used by the QUDA sloppy operator for multishift refinement */
    QudaPrecision precision_refinement_sloppy;

    /**< The precision used by the QUDA preconditioner */
    QudaPrecision precision_precondition;

    /**< The precision used by the QUDA eigensolver */
    QudaPrecision precision_eigensolver;

    /**< Whether the source vector should contain the residual vector
       when the solver returns */
    bool return_residual;

    /**< Domain overlap to use in the preconditioning */
    int overlap_precondition;

    /**< Number of sources in the multi-src solver */
    int num_src;

    // Multi-shift solver parameters

    /**< Number of offsets in the multi-shift solver */
    int num_offset;

    /** Offsets for multi-shift solver */
    double offset[QUDA_MAX_MULTI_SHIFT];

    /** Solver tolerance for each offset */
    double tol_offset[QUDA_MAX_MULTI_SHIFT];

    /** Solver tolerance for each shift when refinement is applied using the heavy-quark residual */
    double tol_hq_offset[QUDA_MAX_MULTI_SHIFT];

    /** Actual L2 residual norm achieved in solver for each offset */
    double true_res_offset[QUDA_MAX_MULTI_SHIFT];

    /** Iterated L2 residual norm achieved in multi shift solver for each offset */
    double iter_res_offset[QUDA_MAX_MULTI_SHIFT];

    /** Actual heavy quark residual norm achieved in solver for each offset */
    double true_res_hq_offset[QUDA_MAX_MULTI_SHIFT];

    /** Number of steps in s-step algorithms */
    int Nsteps;

    /** Maximum size of Krylov space used by solver */
    int Nkrylov;

    /** Number of preconditioner cycles to perform per iteration */
    int precondition_cycle;

    /** Tolerance in the inner solver */
    double tol_precondition;

    /** Maximum number of iterations allowed in the inner solver */
    int maxiter_precondition;

    /** Relaxation parameter used in GCR-DD (default = 1.0) */
    double omega;

    /** Basis for CA algorithms */
    QudaCABasis ca_basis;

    /** Minimum eigenvalue for Chebyshev CA basis */
    double ca_lambda_min;

    /** Maximum eigenvalue for Chebyshev CA basis */
    double ca_lambda_max; // -1 -> power iter generate

    /** Basis for CA algorithms in a preconditioner */
    QudaCABasis ca_basis_precondition;

    /** Minimum eigenvalue for Chebyshev CA basis in a preconditioner */
    double ca_lambda_min_precondition;

    /** Maximum eigenvalue for Chebyshev CA basis in a preconditioner */
    double ca_lambda_max_precondition; // -1 -> power iter generate

    /** Whether to use additive or multiplicative Schwarz preconditioning */
    QudaSchwarzType schwarz_type;

    /** The type of accelerator type to use for preconditioner */
    QudaAcceleratorType accelerator_type_precondition;

    /**< The time taken by the solver */
    double secs;

    /**< The Gflops rate of the solver */
    double gflops;

    // Incremental EigCG solver parameters
    /**< The precision of the Ritz vectors */
    QudaPrecision precision_ritz;//also search space precision

    int n_ev; // number of eigenvectors produced by EigCG
    int m;//Dimension of the search space
    int deflation_grid;
    int rhs_idx;

    int     eigcg_max_restarts;
    int     max_restart_num;
    double  inc_tol;
    double  eigenval_tol;

    QudaVerbosity verbosity_precondition; //! verbosity to use for preconditioner

    bool is_preconditioner; //! whether the solver acting as a preconditioner for another solver

    bool global_reduction; //! whether to use a global or local (node) reduction for this solver

    /** Whether the MG preconditioner (if any) is an instance of MG
        (used internally in MG) or of multigrid_solver (used in the
        interface)*/
    bool mg_instance;

    MadwfParam madwf_param;

    /** Whether to perform advanced features in a preconditioning inversion,
        including reliable updates, pipelining, and mixed precision. */
    bool precondition_no_advanced_feature;

    /** Which external lib to use in the solver */
    QudaExtLibType extlib_type;

    /**
       Default constructor
     */
    SolverParam() :
      compute_null_vector(QUDA_COMPUTE_NULL_VECTOR_NO),
      compute_true_res(true),
      sloppy_converge(false),
      verbosity_precondition(QUDA_SILENT),
      mg_instance(false)
    {
      ;
    }

    /**
       Constructor that matches the initial values to that of the
       QudaInvertParam instance
       @param param The QudaInvertParam instance from which the values are copied
     */
    SolverParam(const QudaInvertParam &param) :
      inv_type(param.inv_type),
      inv_type_precondition(param.inv_type_precondition),
      preconditioner(param.preconditioner),
      deflation_op(param.deflation_op),
      residual_type(param.residual_type),
      deflate(param.eig_param != 0),
      use_init_guess(param.use_init_guess),
      compute_null_vector(QUDA_COMPUTE_NULL_VECTOR_NO),
      delta(param.reliable_delta),
      use_alternative_reliable(param.use_alternative_reliable),
      use_sloppy_partial_accumulator(param.use_sloppy_partial_accumulator),
      solution_accumulator_pipeline(param.solution_accumulator_pipeline),
      max_res_increase(param.max_res_increase),
      max_res_increase_total(param.max_res_increase_total),
      max_hq_res_increase(param.max_hq_res_increase),
      max_hq_res_restart_total(param.max_hq_res_restart_total),
      heavy_quark_check(param.heavy_quark_check),
      pipeline(param.pipeline),
      tol(param.tol),
      tol_restart(param.tol_restart),
      tol_hq(param.tol_hq),
      compute_true_res(param.compute_true_res),
      sloppy_converge(false),
      true_res(param.true_res),
      true_res_hq(param.true_res_hq),
      maxiter(param.maxiter),
      iter(param.iter),
      precision(param.cuda_prec),
      precision_sloppy(param.cuda_prec_sloppy),
      precision_refinement_sloppy(param.cuda_prec_refinement_sloppy),
      precision_precondition(param.cuda_prec_precondition),
      precision_eigensolver(param.cuda_prec_eigensolver),
      return_residual(false),
      num_src(param.num_src),
      num_offset(param.num_offset),
      Nsteps(param.Nsteps),
      Nkrylov(param.gcrNkrylov),
      precondition_cycle(param.precondition_cycle),
      tol_precondition(param.tol_precondition),
      maxiter_precondition(param.maxiter_precondition),
      omega(param.omega),
      ca_basis(param.ca_basis),
      ca_lambda_min(param.ca_lambda_min),
      ca_lambda_max(param.ca_lambda_max),
      ca_basis_precondition(param.ca_basis_precondition),
      ca_lambda_min_precondition(param.ca_lambda_min_precondition),
      ca_lambda_max_precondition(param.ca_lambda_max_precondition),
      schwarz_type(param.schwarz_type),
      accelerator_type_precondition(param.accelerator_type_precondition),
      secs(param.secs),
      gflops(param.gflops),
      precision_ritz(param.cuda_prec_ritz),
      n_ev(param.n_ev),
      m(param.max_search_dim),
      deflation_grid(param.deflation_grid),
      rhs_idx(0),
      eigcg_max_restarts(param.eigcg_max_restarts),
      max_restart_num(param.max_restart_num),
      inc_tol(param.inc_tol),
      eigenval_tol(param.eigenval_tol),
      verbosity_precondition(param.verbosity_precondition),
      is_preconditioner(false),
      global_reduction(true),
      mg_instance(false),
      precondition_no_advanced_feature(param.schwarz_type == QUDA_ADDITIVE_SCHWARZ),
      extlib_type(param.extlib_type)
    {
      if (deflate) { eig_param = *(static_cast<QudaEigParam *>(param.eig_param)); }
      for (int i=0; i<num_offset; i++) {
        offset[i] = param.offset[i];
        tol_offset[i] = param.tol_offset[i];
        tol_hq_offset[i] = param.tol_hq_offset[i];
      }

      if (param.rhs_idx != 0
          && (param.inv_type == QUDA_INC_EIGCG_INVERTER || param.inv_type == QUDA_GMRESDR_PROJ_INVERTER)) {
        rhs_idx = param.rhs_idx;
      }

      madwf_param.madwf_diagonal_suppressor = param.madwf_diagonal_suppressor;
      madwf_param.madwf_ls = param.madwf_ls;
      madwf_param.madwf_null_miniter = param.madwf_null_miniter;
      madwf_param.madwf_null_tol = param.madwf_null_tol;
      madwf_param.madwf_train_maxiter = param.madwf_train_maxiter;
      madwf_param.madwf_param_load = param.madwf_param_load == QUDA_BOOLEAN_TRUE;
      madwf_param.madwf_param_save = param.madwf_param_save == QUDA_BOOLEAN_TRUE;
      if (madwf_param.madwf_param_load) madwf_param.madwf_param_infile = std::string(param.madwf_param_infile);
      if (madwf_param.madwf_param_save) madwf_param.madwf_param_outfile = std::string(param.madwf_param_outfile);
    }

    SolverParam(const SolverParam &param) :
      inv_type(param.inv_type),
      inv_type_precondition(param.inv_type_precondition),
      preconditioner(param.preconditioner),
      deflation_op(param.deflation_op),
      residual_type(param.residual_type),
      deflate(param.deflate),
      eig_param(param.eig_param),
      use_init_guess(param.use_init_guess),
      compute_null_vector(param.compute_null_vector),
      delta(param.delta),
      use_alternative_reliable(param.use_alternative_reliable),
      use_sloppy_partial_accumulator(param.use_sloppy_partial_accumulator),
      solution_accumulator_pipeline(param.solution_accumulator_pipeline),
      max_res_increase(param.max_res_increase),
      max_res_increase_total(param.max_res_increase_total),
      heavy_quark_check(param.heavy_quark_check),
      pipeline(param.pipeline),
      tol(param.tol),
      tol_restart(param.tol_restart),
      tol_hq(param.tol_hq),
      compute_true_res(param.compute_true_res),
      sloppy_converge(param.sloppy_converge),
      true_res(param.true_res),
      true_res_hq(param.true_res_hq),
      maxiter(param.maxiter),
      iter(param.iter),
      precision(param.precision),
      precision_sloppy(param.precision_sloppy),
      precision_refinement_sloppy(param.precision_refinement_sloppy),
      precision_precondition(param.precision_precondition),
      precision_eigensolver(param.precision_eigensolver),
      return_residual(param.return_residual),
      num_offset(param.num_offset),
      Nsteps(param.Nsteps),
      Nkrylov(param.Nkrylov),
      precondition_cycle(param.precondition_cycle),
      tol_precondition(param.tol_precondition),
      maxiter_precondition(param.maxiter_precondition),
      omega(param.omega),
      ca_basis(param.ca_basis),
      ca_lambda_min(param.ca_lambda_min),
      ca_lambda_max(param.ca_lambda_max),
      ca_basis_precondition(param.ca_basis_precondition),
      ca_lambda_min_precondition(param.ca_lambda_min_precondition),
      ca_lambda_max_precondition(param.ca_lambda_max_precondition),
      schwarz_type(param.schwarz_type),
      accelerator_type_precondition(param.accelerator_type_precondition),
      secs(param.secs),
      gflops(param.gflops),
      precision_ritz(param.precision_ritz),
      n_ev(param.n_ev),
      m(param.m),
      deflation_grid(param.deflation_grid),
      rhs_idx(0),
      eigcg_max_restarts(param.eigcg_max_restarts),
      max_restart_num(param.max_restart_num),
      inc_tol(param.inc_tol),
      eigenval_tol(param.eigenval_tol),
      verbosity_precondition(param.verbosity_precondition),
      is_preconditioner(param.is_preconditioner),
      global_reduction(param.global_reduction),
      mg_instance(param.mg_instance),
      madwf_param(param.madwf_param),
      precondition_no_advanced_feature(param.precondition_no_advanced_feature),
      extlib_type(param.extlib_type)
    {
      for (int i=0; i<num_offset; i++) {
	offset[i] = param.offset[i];
	tol_offset[i] = param.tol_offset[i];
	tol_hq_offset[i] = param.tol_hq_offset[i];
      }

      if((param.inv_type == QUDA_INC_EIGCG_INVERTER || param.inv_type == QUDA_EIGCG_INVERTER) && m % 16){//current hack for the magma library
        m = (m / 16) * 16 + 16;
        warningQuda("\nSwitched eigenvector search dimension to %d\n", m);
      }
      if(param.rhs_idx != 0 && (param.inv_type==QUDA_INC_EIGCG_INVERTER || param.inv_type==QUDA_GMRESDR_PROJ_INVERTER)){
        rhs_idx = param.rhs_idx;
      }
    }

    ~SolverParam() { }

    /**
       Update the QudaInvertParam with the data from this
       @param param the QudaInvertParam to be updated
     */
    void updateInvertParam(QudaInvertParam &param, int offset=-1) {
      param.true_res = true_res;
      param.true_res_hq = true_res_hq;
      param.iter += iter;
      comm_allreduce_sum(gflops);
      param.gflops += gflops;
      param.secs += secs;
      if (offset >= 0) {
	param.true_res_offset[offset] = true_res_offset[offset];
        param.iter_res_offset[offset] = iter_res_offset[offset];
	param.true_res_hq_offset[offset] = true_res_hq_offset[offset];
      } else {
	for (int i=0; i<num_offset; i++) {
	  param.true_res_offset[i] = true_res_offset[i];
          param.iter_res_offset[i] = iter_res_offset[i];
	  param.true_res_hq_offset[i] = true_res_hq_offset[i];
	}
      }
      //for incremental eigCG:
      param.rhs_idx = rhs_idx;

      param.ca_lambda_min = ca_lambda_min;
      param.ca_lambda_max = ca_lambda_max;

      param.ca_lambda_min_precondition = ca_lambda_min_precondition;
      param.ca_lambda_max_precondition = ca_lambda_max_precondition;

      if (deflate) *static_cast<QudaEigParam *>(param.eig_param) = eig_param;
    }

    void updateRhsIndex(QudaInvertParam &param) {
      //for incremental eigCG:
      rhs_idx = param.rhs_idx;
    }

  };

  class Solver {

  protected:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    const DiracMatrix &matPrecon;
    const DiracMatrix &matEig;

    SolverParam &param;
    TimeProfile &profile;
    int node_parity;
    EigenSolver *eig_solve; /** Eigensolver object. */
    bool deflate_init;      /** If true, the deflation space has been computed. */
    bool deflate_compute;   /** If true, instruct the solver to create a deflation space. */
    bool recompute_evals;   /** If true, instruct the solver to recompute evals from an existing deflation space. */
    std::vector<ColorSpinorField> evecs; /** Holds the eigenvectors. */
    std::vector<Complex> evals;          /** Holds the eigenvalues. */

    bool mixed() { return param.precision != param.precision_sloppy; }

  public:
    Solver(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
           const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile);
    virtual ~Solver();

    virtual void operator()(ColorSpinorField &out, ColorSpinorField &in) = 0;

    virtual void blocksolve(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @return Return the residual vector from the prior solve
    */
    virtual ColorSpinorField &get_residual()
    {
      errorQuda("Not implemented");
      static ColorSpinorField dummy;
      return dummy;
    }

    /**
      @brief a virtual method that performs the necessary training/preparation at the beginning of a solve.
        The default here is a no-op.
      @param Solver the solver to be used to collect the null space vectors.
      @param ColorSpinorField the vector used to perform the training.
     */
    virtual void train_param(Solver &, ColorSpinorField &)
    {
      // Do nothing
    }

    /**
      @brief a virtual method that performs the inversion and collect some vectors.
        The default here is a no-op and should not be called.
     */
    virtual void solve_and_collect(ColorSpinorField &, ColorSpinorField &, cvector_ref<ColorSpinorField> &, int, double)
    {
      errorQuda("NOT implemented.");
    }

    void set_tol(double tol) { param.tol = tol; }
    void set_maxiter(int maxiter) { param.maxiter = maxiter; }

    const DiracMatrix &M() { return mat; }
    const DiracMatrix &Msloppy() { return matSloppy; }
    const DiracMatrix &Mprecon() { return matPrecon; }
    const DiracMatrix &Meig() { return matEig; }

    /**
       @return Whether the solver is only for Hermitian systems
     */
    virtual bool hermitian() = 0;

    /**
       @brief Generic solver setup and parameter checking
       @param[in] x Solution vector
       @param[in] b Source vector
     */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

    /**
       @brief Solver factory
    */
    static Solver *create(SolverParam &param, const DiracMatrix &mat, const DiracMatrix &matSloppy,
                          const DiracMatrix &matPrecon, const DiracMatrix &matEig, TimeProfile &profile);

    /**
       @brief Set the solver L2 stopping condition
       @param[in] Desired solver tolerance
       @param[in] b2 L2 norm squared of the source vector
       @param[in] residual_type The type of residual we want to solve for
       @return L2 stopping condition
    */
    static double stopping(double tol, double b2, QudaResidualType residual_type);

    /**
       @briefTest for solver convergence
       @param[in] r2 L2 norm squared of the residual
       @param[in] hq2 Heavy quark residual
       @param[in] r2_tol Solver L2 tolerance
       @param[in] hq_tol Solver heavy-quark tolerance
       @return Whether converged
     */
    bool convergence(double r2, double hq2, double r2_tol, double hq_tol);

    /**
       @brief Test for HQ solver convergence -- ignore L2 residual
       @param[in] r2 L2 norm squared of the residual
       @param[in] hq2 Heavy quark residual
       @param[in] r2_tol Solver L2 tolerance
       @param[in[ hq_tol Solver heavy-quark tolerance
       @return Whether converged
     */
    bool convergenceHQ(double r2, double hq2, double r2_tol, double hq_tol);

    /**
       @brief Test for L2 solver convergence -- ignore HQ residual
       @param[in] r2 L2 norm squared of the residual
       @param[in] hq2 Heavy quark residual
       @param[in] r2_tol Solver L2 tolerance
       @param[in] hq_tol Solver heavy-quark tolerance
     */
    bool convergenceL2(double r2, double hq2, double r2_tol, double hq_tol);

    /**
       @brief Prints out the running statistics of the solver
       (requires a verbosity of QUDA_VERBOSE)
       @param[in] name Name of solver that called this
       @param[in] k iteration count
       @param[in] r2 L2 norm squared of the residual
       @param[in] hq2 Heavy quark residual
     */
    void PrintStats(const char *name, int k, double r2, double b2, double hq2);

    /**
       @brief Prints out the summary of the solver convergence
       (requires a verbosity of QUDA_SUMMARIZE).  Assumes
       SolverParam.true_res and SolverParam.true_res_hq has been set
       @param[in] name Name of solver that called this
       @param[in] k iteration count
       @param[in] r2 L2 norm squared of the residual
       @param[in] hq2 Heavy quark residual
       @param[in] r2_tol Solver L2 tolerance
       @param[in] hq_tol Solver heavy-quark tolerance
    */
    void PrintSummary(const char *name, int k, double r2, double b2, double r2_tol, double hq_tol);

    /**
       @brief Returns the epsilon tolerance for a given precision, by default returns
       the solver precision.
       @param[in] prec Input precision, default value is solver precision
    */
    double precisionEpsilon(QudaPrecision prec = QUDA_INVALID_PRECISION) const;

    /**
       @brief Constructs the deflation space and eigensolver
       @param[in] meta A sample ColorSpinorField with which to instantiate
       the eigensolver
       @param[in] mat The operator to eigensolve
       @param[in] Whether to compute the SVD
    */
    void constructDeflationSpace(const ColorSpinorField &meta, const DiracMatrix &mat);

    /**
       @brief Destroy the allocated deflation space
    */
    void destroyDeflationSpace();

    /**
       @brief Extends the deflation space to twice its size for SVD deflation
    */
    void extendSVDDeflationSpace();

    /**
       @brief Injects a deflation space into the solver from the
       vector argument.  Note the input space is reduced to zero size as a
       result of calling this function, with responsibility for the
       space transferred to the solver.
       @param[in,out] defl_space the deflation space we wish to
       transfer to the solver.
    */
    void injectDeflationSpace(std::vector<ColorSpinorField> &defl_space);

    /**
       @brief Extracts the deflation space from the solver to the
       vector argument.  Note the solver deflation space is reduced to
       zero size as a result of calling this function, with
       responsibility for the space transferred to the argument.
       @param[in,out] defl_space the extracted deflation space.  On
       input, this vector should have zero size.
    */
    void extractDeflationSpace(std::vector<ColorSpinorField> &defl_space);

    /**
       @brief Returns the size of deflation space
    */
    int deflationSpaceSize() const { return (int)evecs.size(); };

    /**
       @brief Sets the deflation compute boolean
       @param[in] flag Set to this boolean value
    */
    void setDeflateCompute(bool flag) { deflate_compute = flag; };

    /**
       @brief Sets the recompute evals boolean
       @param[in] flag Set to this boolean value
    */
    void setRecomputeEvals(bool flag) { recompute_evals = flag; };

    /**
       @brief Compute power iterations on a Dirac matrix
       @param[in] diracm Dirac matrix used for power iterations
       @param[in] start Starting rhs for power iterations; value preserved unless it aliases tempvec1 or tempvec2
       @param[in,out] tempvec1 Temporary vector used for power iterations (FIXME: can become a reference when std::swap
       can be used on ColorSpinorField)
       @param[in,out] tempvec2 Temporary vector used for power iterations (FIXME: can become a reference when std::swap
       can be used on ColorSpinorField)
       @param[in] niter Total number of power iteration iterations
       @param[in] normalize_freq Frequency with which intermediate vector gets normalized
       @param[in] args Parameter pack of ColorSpinorFields used as temporary passed to Dirac
       @return Norm of final power iteration result
    */
    template <typename... Args>
    static double performPowerIterations(const DiracMatrix &diracm, const ColorSpinorField &start,
                                         ColorSpinorField &tempvec1, ColorSpinorField &tempvec2, int niter,
                                         int normalize_freq, Args &&...args);

    /**
       @brief Generate a Krylov space in a given basis
       @param[in] diracm Dirac matrix used to generate the Krylov space
       @param[out] Ap dirac matrix times the Krylov basis vectors
       @param[in,out] p Krylov basis vectors; assumes p[0] is in place
       @param[in] n_krylov Size of krylov space
       @param[in] basis Basis type
       @param[in] m_map Slope mapping for Chebyshev basis; ignored for power basis
       @param[in] b_map Intercept mapping for Chebyshev basis; ignored for power basis
       @param[in] args Parameter pack of ColorSpinorFields used as temporary passed to Dirac
    */
    template <typename... Args>
    static void computeCAKrylovSpace(const DiracMatrix &diracm, std::vector<ColorSpinorField> &Ap,
                                     std::vector<ColorSpinorField> &p, int n_krylov, QudaCABasis basis, double m_map,
                                     double b_map, Args &&...args);

    /**
     * @brief Return flops
     * @return flops expended by this operator
     */
    virtual double flops() const { return 0; }
  };

  /**
     @brief  Conjugate-Gradient Solver.
   */
  class CG : public Solver {

  private:
    // pointers to fields to avoid multiple creation overhead
    ColorSpinorField *yp, *rp, *rnewp, *pp, *App, *tmpp, *rSloppyp, *xSloppyp;
    bool init = false;

  public:
    CG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, const DiracMatrix &matEig,
       SolverParam &param, TimeProfile &profile);
    virtual ~CG();
    /**
     * @brief Run CG.
     * @param out Solution vector.
     * @param in Right-hand side.
     */
    void operator()(ColorSpinorField &out, ColorSpinorField &in){
      (*this)(out, in, nullptr, 0.0);
    };

    /**
     * @brief Solve re-using an initial Krylov space defined by an initial r2_old_init and search direction p_init.
     * @details This can be used when continuing a CG, e.g. as refinement step after a multi-shift solve.
     * @param out Solution-vector.
     * @param in Right-hand side.
     * @param p_init Initial-search direction.
     * @param r2_old_init [description]
     */
    void operator()(ColorSpinorField &out, ColorSpinorField &in, ColorSpinorField *p_init, double r2_old_init);

    void blocksolve(ColorSpinorField& out, ColorSpinorField& in);

    virtual bool hermitian() { return true; } /** CG is only for Hermitian systems */

  protected:
    /**
     * @brief Separate codepath for performing a "simpler" CG solve when a heavy quark residual is requested.
     * @param out Solution-vector.
     * @param in Right-hand side.
     */
    void hqsolve(ColorSpinorField &out, ColorSpinorField &in);
  };

  class CGNE : public CG
  {

  private:
    DiracMMdag mmdag;
    DiracMMdag mmdagSloppy;
    DiracMMdag mmdagPrecon;
    DiracMMdag mmdagEig;
    ColorSpinorField xp;
    ColorSpinorField yp;
    bool init = false;

    /**
       @brief Initiate the fields needed by the solver
       @param[in] x Solution vector
       @param[in] b Source vector
    */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

  public:
    CGNE(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, const DiracMatrix &matEig,
         SolverParam &param, TimeProfile &profile);

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @return Return the residual vector from the prior solve
    */
    ColorSpinorField &get_residual();

    virtual bool hermitian() { return false; } /** CGNE is for any system */
  };

  class CGNR : public CG
  {

  private:
    DiracMdagM mdagm;
    DiracMdagM mdagmSloppy;
    DiracMdagM mdagmPrecon;
    DiracMdagM mdagmEig;
    ColorSpinorField br;
    bool init = false;

    /**
       @brief Initiate the fields needed by the solver
       @param[in] x Solution vector
       @param[in] b Source vector
    */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

  public:
    CGNR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, const DiracMatrix &matEig,
         SolverParam &param, TimeProfile &profile);

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @return Return the residual vector from the prior solve
    */
    ColorSpinorField &get_residual();

    virtual bool hermitian() { return false; } /** CGNR is for any system */
  };

  class CG3 : public Solver
  {

  private:
    // pointers to fields to avoid multiple creation overhead
    ColorSpinorField *yp, *rp, *tmpp, *ArSp, *rSp, *xSp, *xS_oldp, *tmpSp, *rS_oldp;
    bool init = false;

  public:
    CG3(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
        TimeProfile &profile);
    virtual ~CG3();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return true; } /** CG is only for Hermitian systems */
  };

  class CG3NE : public CG3
  {

  private:
    DiracMMdag mmdag;
    DiracMMdag mmdagSloppy;
    DiracMMdag mmdagPrecon;
    ColorSpinorField xp;
    ColorSpinorField yp;
    bool init = false;

    /**
       @brief Initiate the fields needed by the solver
       @param[in] x Solution vector
       @param[in] b Source vector
    */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

  public:
    CG3NE(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
          TimeProfile &profile);

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @return Return the residual vector from the prior solve
    */
    ColorSpinorField &get_residual();

    virtual bool hermitian() { return false; } /** CG3NE is for any system */
  };

  class CG3NR : public CG3
  {

  private:
    DiracMdagM mdagm;
    DiracMdagM mdagmSloppy;
    DiracMdagM mdagmPrecon;
    ColorSpinorField br;
    bool init = false;

    /**
       @brief Initiate the fields needed by the solver
       @param[in] x Solution vector
       @param[in] b Source vector
    */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

  public:
    CG3NR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
          TimeProfile &profile);

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @return Return the residual vector from the prior solve
    */
    ColorSpinorField &get_residual();

    virtual bool hermitian() { return false; } /** CG3NR is for any system */
  };

  class PreconCG : public Solver {
    private:
    std::shared_ptr<Solver> K;
    SolverParam Kparam; // parameters for preconditioner solve

    ColorSpinorField r;
    ColorSpinorField y;
    ColorSpinorField Ap;
    ColorSpinorField x_sloppy;
    ColorSpinorField r_sloppy;
    ColorSpinorField minvr;
    ColorSpinorField minvr_sloppy;
    ColorSpinorField minvr_pre;
    ColorSpinorField r_pre;
    XUpdateBatch x_update_batch;
    int Np; /** the size of the accumulator pipeline */

    bool init = false;

    /**
       @brief Allocate persistent fields and parameter checking
       @param[in] x Solution vector
       @param[in] b Source vector
     */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

  public:
    PreconCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
             const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile);

    virtual ~PreconCG();

    void operator()(ColorSpinorField &out, ColorSpinorField &in)
    {
      this->solve_and_collect(out, in, cvector_ref<ColorSpinorField>(), 0, 0);
    }

    /**
       @brief a virtual method that performs the inversion and collect the r vectors in PCG.
       @param out the output vector
       @param in the input vector
       @param v_r the series of vectors that is to be collected
       @param collect_miniter minimal iteration start from which the r vectors are to be collected
       @param collect_tol maxiter tolerance start from which the r vectors are to be collected
    */
    virtual void solve_and_collect(ColorSpinorField &out, ColorSpinorField &in, cvector_ref<ColorSpinorField> &v_r,
                                   int collect_miniter, double collect_tol);

    virtual bool hermitian() { return true; } /** PCG is only Hermitian system */
  };


  class BiCGstab : public Solver {

  private:
    const DiracMdagM matMdagM; // used by the eigensolver
    // pointers to fields to avoid multiple creation overhead
    ColorSpinorField *yp, *rp, *pp, *vp, *tmpp, *tp;
    bool init = false;

  public:
    BiCGstab(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
             const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile);
    virtual ~BiCGstab();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** BiCGStab is for any linear system */
  };

  /**
   * @brief Optimized version of the BiCGstabL solver described in
   * https://etna.math.kent.edu/vol.1.1993/pp11-32.dir/pp11-32.pdf
   */
  class BiCGstabL : public Solver {

  private:
    const DiracMdagM matMdagM; // used by the eigensolver

    /**
       The size of the Krylov space that BiCGstabL uses.
     */
    int n_krylov; // in the language of BiCGstabL, this is L.
    int pipeline; // pipelining factor for legacyGramSchmidt

    // Various coefficients and params needed on each iteration.
    Complex rho0, rho1, alpha, omega, beta;           // Various coefficients for the BiCG part of BiCGstab-L.
    std::vector<Complex> gamma, gamma_prime, gamma_prime_prime; // Parameters for MR part of BiCGstab-L. (L+1) length.
    std::vector<Complex> tau; // Parameters for MR part of BiCGstab-L. Tech. modified Gram-Schmidt coeffs. (L+1)x(L+1) length.
    std::vector<double> sigma; // Parameters for MR part of BiCGstab-L. Tech. the normalization part of Gram-Scmidt. (L+1) length.

    ColorSpinorField r_full; //! Full precision residual.
    ColorSpinorField y;      //! Full precision temporary.

    // sloppy precision fields
    ColorSpinorField temp; //! Sloppy temporary vector.
    std::vector<ColorSpinorField> r; // Current residual + intermediate residual values, along the MR.
    std::vector<ColorSpinorField> u; // Search directions.

    ColorSpinorField x_sloppy;  //! Sloppy solution vector.
    ColorSpinorField r0;        //! Shadow residual, in BiCG language.

    /**
       @brief Allocate persistent fields and parameter checking
       @param[in] x Solution vector
       @param[in] b Source vector
     */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

    /**
     @brief Internal routine for reliable updates. Made to not conflict with BiCGstab's implementation.
     */
    int reliable(double &rNorm, double &maxrx, double &maxrr, const double &r2, const double &delta);

    /**
     * @brief Internal routine for performing the MR part of BiCGstab-L
     *
     * @param x_sloppy [out] sloppy accumulator for x
     * @param fixed_iteration [in] whether or not this is for a fixed iteration solver
     */
    void computeMR(ColorSpinorField &x_sloppy, bool fixed_iteration);

    /**
       Legacy routines that encapsulate the original pipelined Gram-Schmit.
       In theory these should be a bit more numerically stable than the
       fully fused version in computeMR, but in practice that seems to be
       lost in the noise, and the fused nature of computeMR wins in terms of
       time to solution.
     */

    /**
     * @brief Internal routine that comptues the "tau" matrix as described in
     *        the original BiCGstab-L paper, supporting pipelining
     *
     * @param begin [in] begin offset for pipelining
     * @param size [in] length of pipelining
     * @param j [in] row of tau being computed
     */
    void computeTau(int begin, int size, int j);

    /**
     * @brief Internal routine that updates R as described in
     *        the original BiCGstab-L paper, supporting pipelining.
     *
     * @param begin [in] begin offset for pipelining
     * @param size [in] length of pipelining
     * @param j [in] row of tau being computed
     */
    void updateR(int begin, int size, int j);

    /**
     * @brief Internal legacy routine for performing the MR part of BiCGstab-L
     *        which more closely matches the paper
     *
     * @param x_sloppy [out] sloppy accumulator for x
     */
    void legacyComputeMR(ColorSpinorField &x_sloppy);

    /**
       Solver uses lazy allocation: this flag determines whether we have allocated or not.
     */
    bool init = false;

    std::string solver_name; // holds BiCGstab-l, where 'l' literally equals n_krylov.

  public:
    BiCGstabL(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matEig, SolverParam &param,
              TimeProfile &profile);
    virtual ~BiCGstabL();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** BiCGStab is for any linear system */
  };

  class GCR : public Solver {

  private:
    const DiracMdagM matMdagM; // used by the eigensolver

    Solver *K;
    SolverParam Kparam; // parameters for preconditioner solve

    /**
       The size of the Krylov space that GCR uses
     */
    int n_krylov;

    std::vector<Complex> alpha;
    std::vector<Complex> beta;
    std::vector<double> gamma;

    /**
       Solver uses lazy allocation: this flag to determine whether we have allocated.
     */
    bool init = false;

    ColorSpinorField r;       //! residual vector
    ColorSpinorField r_sloppy; //! sloppy residual vector

    std::vector<ColorSpinorField> p;  // GCR direction vectors
    std::vector<ColorSpinorField> Ap; // mat * direction vectors

    void computeBeta(std::vector<Complex> &beta, std::vector<ColorSpinorField> &Ap, int i, int N, int k);
    void updateAp(std::vector<Complex> &beta, std::vector<ColorSpinorField> &Ap, int begin, int size, int k);
    void orthoDir(std::vector<Complex> &beta, std::vector<ColorSpinorField> &Ap, int k, int pipeline);
    void backSubs(const std::vector<Complex> &alpha, const std::vector<Complex> &beta, const std::vector<double> &gamma,
                  std::vector<Complex> &delta, int n);
    void updateSolution(ColorSpinorField &x, const std::vector<Complex> &alpha, const std::vector<Complex> &beta,
                        std::vector<double> &gamma, int k, std::vector<ColorSpinorField> &p);

    /**
       @brief Initiate the fields needed by the solver
       @param[in] x Solution vector
       @param[in] b Source vector
    */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

  public:
    GCR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, const DiracMatrix &matEig,
        SolverParam &param, TimeProfile &profile);

    /**
       @param K Preconditioner
    */
    GCR(const DiracMatrix &mat, Solver &K, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
        const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile);
    virtual ~GCR();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** GCR is for any linear system */
  };

  class MR : public Solver {

  private:
    ColorSpinorField r;
    ColorSpinorField r_sloppy;
    ColorSpinorField Ar;
    ColorSpinorField x_sloppy;
    bool init = false;

    /**
       @brief Allocate persistent fields and parameter checking
       @param[in] x Solution vector
       @param[in] b Source vector
     */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

  public:
    MR(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @return Return the residual vector from the prior solve
    */
    ColorSpinorField &get_residual();

    bool hermitian() { return false; } /** MR is for any linear system */
  };

  /**
     @brief Communication-avoiding CG solver.  This solver does
     un-preconditioned CG, running in steps of n_krylov, build up a
     polynomial in the linear operator of length n_krylov, and then
     performs a steepest descent minimization on the resulting basis
     vectors.  For now only implemented using the power basis so is
     only useful as a preconditioner.
   */
  class CACG : public Solver {

  protected:
    bool init = false;

    bool lambda_init;
    QudaCABasis basis;

    std::vector<double> Q_AQandg; // Fused inner product matrix
    std::vector<double> Q_AS;     // inner product matrix
    std::vector<double> alpha;    // QAQ^{-1} g
    std::vector<double> beta;     // QAQ^{-1} QpolyS

    ColorSpinorField r;

    std::vector<ColorSpinorField> S;    // residual vectors
    std::vector<ColorSpinorField> AS;   // mat * residual vectors. Can be replaced by a single temporary.
    std::vector<ColorSpinorField> Q;    // CG direction vectors
    std::vector<ColorSpinorField> Qtmp; // CG direction vectors for pointer swap
    std::vector<ColorSpinorField> AQ;   // mat * CG direction vectors.
                                        // it's possible to avoid carrying these
                                        // around, but there's a stability penalty,
                                        // and computing QAQ becomes a pain (though
                                        // it does let you fuse the reductions...)

    /**
       @brief Initiate the fields needed by the solver
       @param[in] x Solution vector
       @param[in] b Source vector
    */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

    /**
       @brief Compute the alpha coefficients
    */
    void compute_alpha();

    /**
       @brief Compute the beta coefficients
    */
    void compute_beta();

    /**
       @ brief Check if it's time for a reliable update
    */
    int reliable(double &rNorm,  double &maxrr, int &rUpdate, const double &r2, const double &delta);

  public:
    CACG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, const DiracMatrix &matEig,
         SolverParam &param, TimeProfile &profile);
    virtual ~CACG();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @return Return the residual vector from the prior solve
    */
    ColorSpinorField &get_residual();

    virtual bool hermitian() { return true; } /** CG is only for Hermitian systems */
  };

  class CACGNE : public CACG {

  private:
    DiracMMdag mmdag;
    DiracMMdag mmdagSloppy;
    DiracMMdag mmdagPrecon;
    DiracMMdag mmdagEig;
    ColorSpinorField xp;
    ColorSpinorField yp;
    bool init = false;

    /**
       @brief Initiate the fields needed by the solver
       @param[in] x Solution vector
       @param[in] b Source vector
    */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

  public:
    CACGNE(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
           const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile);

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @return Return the residual vector from the prior solve
    */
    ColorSpinorField &get_residual();

    virtual bool hermitian() { return false; } /** CGNE is for any linear system */
  };

  class CACGNR : public CACG
  {

  private:
    DiracMdagM mdagm;
    DiracMdagM mdagmSloppy;
    DiracMdagM mdagmPrecon;
    DiracMdagM mdagmEig;
    ColorSpinorField br;
    bool init = false;

    /**
       @brief Initiate the fields needed by the solver
       @param[in] x Solution vector
       @param[in] b Source vector
    */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

  public:
    CACGNR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
           const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile);

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @return Return the residual vector from the prior solve
    */
    ColorSpinorField &get_residual();

    virtual bool hermitian() { return false; } /** CGNE is for any linear system */
  };

  /**
     @brief Communication-avoiding GCR solver.  This solver does
     un-preconditioned GCR, first building up a polynomial in the
     linear operator of length n_krylov, and then performs a minimum
     residual extrapolation on the resulting basis vectors.  For use as
     a multigrid smoother with minimum global synchronization.
   */
  class CAGCR : public Solver {

  private:
    const DiracMdagM matMdagM; // used by the eigensolver
    bool init = false;

    bool lambda_init;  // whether or not lambda_max has been initialized
    QudaCABasis basis; // CA basis

    std::vector<Complex> alpha; // Solution coefficient vectors

    ColorSpinorField r;

    std::vector<ColorSpinorField> p; // GCR direction vectors
    std::vector<ColorSpinorField> q; // mat * direction vectors

    /**
       @brief Initiate the fields needed by the solver
       @param[in] x Solution vector
       @param[in] b Source vector
    */
    void create(ColorSpinorField &x, const ColorSpinorField &b);

    /**
       @brief Solve the equation A p_k psi_k = q_k psi_k = b by minimizing the
       least square residual using Eigen's LDLT Cholesky for numerical stability
       @param[out] psi Array of coefficients
       @param[in] q Search direction vectors with the operator applied
       @param[in] b Source vector against which we are solving
    */
    void solve(std::vector<Complex> &psi, std::vector<ColorSpinorField> &q, ColorSpinorField &b);

  public:
    CAGCR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, const DiracMatrix &matEig,
          SolverParam &param, TimeProfile &profile);
    virtual ~CAGCR();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @return Return the residual vector from the prior solve
    */
    ColorSpinorField &get_residual();

    virtual bool hermitian() { return false; } /** GCR is for any linear system */
  };

  // Steepest descent solver used as a preconditioner
  class SD : public Solver {
    private:
      ColorSpinorField *Ar;
      ColorSpinorField *r;
      bool init = false;

    public:
      SD(const DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
      virtual ~SD();

      void operator()(ColorSpinorField &out, ColorSpinorField &in);

      virtual bool hermitian() { return false; } /** SD is for any linear system */
  };

  class PreconditionedSolver : public Solver
  {
private:
    Solver *solver;
    const Dirac &dirac;
    const char *prefix;

public:
  PreconditionedSolver(Solver &solver, const Dirac &dirac, SolverParam &param, TimeProfile &profile, const char *prefix) :
    Solver(solver.M(), solver.Msloppy(), solver.Mprecon(), solver.Meig(), param, profile),
    solver(&solver),
    dirac(dirac),
    prefix(prefix)
  {
  }

    virtual ~PreconditionedSolver() { delete solver; }

    void operator()(ColorSpinorField &x, ColorSpinorField &b) {
      pushOutputPrefix(prefix);

      QudaSolutionType solution_type = b.SiteSubset() == QUDA_FULL_SITE_SUBSET ? QUDA_MAT_SOLUTION : QUDA_MATPC_SOLUTION;

      ColorSpinorField *out=nullptr;
      ColorSpinorField *in=nullptr;

      if (dirac.hasSpecialMG()) {
        dirac.prepareSpecialMG(in, out, x, b, solution_type);
      } else {
        dirac.prepare(in, out, x, b, solution_type);
      }
      (*solver)(*out, *in);
      if (dirac.hasSpecialMG()) {
        dirac.reconstructSpecialMG(x, b, solution_type);
      } else {
        dirac.reconstruct(x, b, solution_type);
      }

      popOutputPrefix();
    }

    /**
     * @brief Return reference to the solver. Used when mass/mu
     *        rescaling an MG instance
     */
    Solver &ExposeSolver() const { return *solver; }

    virtual bool hermitian() { return solver->hermitian(); } /** Use the inner solver */
  };

  class MultiShiftSolver {

  protected:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    SolverParam &param;
    TimeProfile &profile;

    /**
       @brief Generic solver setup and parameter checking
       @param[in] x Solution vectors
       @param[in] b Source vector
     */
    void create(const std::vector<ColorSpinorField> &x, const ColorSpinorField &b);

  public:
    MultiShiftSolver(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
      mat(mat), matSloppy(matSloppy), param(param), profile(profile)
    {
    }

    virtual void operator()(std::vector<ColorSpinorField> &out, ColorSpinorField &in) = 0;
    bool convergence(const std::vector<double> &r2, const std::vector<double> &r2_tol, int n) const;
  };

  /**
   * @brief Multi-Shift Conjugate Gradient Solver.
   */
  class MultiShiftCG : public MultiShiftSolver {

    bool init = false;
    bool mixed;        // whether we will be using mixed precision
    bool reliable;     // whether we will be using reliable updates or not
    bool group_update; // whether we will be using solution group updates
    int num_offset;
    ColorSpinorField r;
    ColorSpinorField r_sloppy;
    ColorSpinorField Ap;
    std::vector<ColorSpinorField> x_sloppy;

    void create(std::vector<ColorSpinorField> &x, const ColorSpinorField &b, std::vector<ColorSpinorField> &p);

  public:
    MultiShiftCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);

    /**
     * @brief Run multi-shift and return Krylov-space at the end of the solve in p and r2_old_arry.
     *
     * @param out std::vector of pointer to solutions for all the shifts.
     * @param in right-hand side.
     * @param p std::vector of pointers to hold search directions. Note this will be resized as necessary.
     * @param r2_old_array pointer to last values of r2_old for old shifts. Needs to be large enough to hold r2_old for all shifts.
     */
    void operator()(std::vector<ColorSpinorField> &x, ColorSpinorField &b, std::vector<ColorSpinorField> &p,
                    std::vector<double> &r2_old_array);

    /**
     * @brief Run multi-shift and return Krylov-space at the end of the solve in p and r2_old_arry.
     *
     * @param out std::vector of pointer to solutions for all the shifts.
     * @param in right-hand side.
     */
    void operator()(std::vector<ColorSpinorField> &out, ColorSpinorField &in)
    {
      std::vector<double> r2_old(out.size());
      std::vector<ColorSpinorField> p;

      (*this)(out, in, p, r2_old);
    }
  };


  /**
     @brief This computes the optimum guess for the system Ax=b in the L2
     residual norm.  For use in the HMD force calculations using a
     minimal residual chronological method.  This computes the guess
     solution as a linear combination of a given number of previous
     solutions.  Following Brower et al, only the orthogonalised vector
     basis is stored to conserve memory.

     If Eigen support is enabled then Eigen's SVD algorithm is used
     for solving the linear system, else Gaussian elimination with
     partial pivots is used.
  */
  class MinResExt {

  protected:
    const DiracMatrix &mat;
    bool orthogonal; //! Whether to construct an orthogonal basis or not
    bool apply_mat; //! Whether to compute q = Ap or assume it is provided
    bool hermitian; //! Whether A is hermitian or not

    TimeProfile &profile;

    /**
       @brief Solve the equation A p_k psi_k = q_k psi_k = b by minimizing the
       residual and using Eigen's SVD algorithm for numerical stability
       @param[out] psi Array of coefficients
       @param[in] p Search direction vectors
       @param[in] q Search direction vectors with the operator applied
       @param[in] hermitian Whether the linear system is Hermitian or not
    */
    void solve(std::vector<Complex> &psi_, std::vector<ColorSpinorField> &p, std::vector<ColorSpinorField> &q,
               const ColorSpinorField &b, bool hermitian);

  public:
    /**
       @param mat The operator for the linear system we wish to solve
       @param orthogonal Whether to construct an orthogonal basis prior to constructing the linear system
       @param apply_mat Whether to apply the operator in place or assume q already contains this
       @profile Timing profile to use
    */
    MinResExt(const DiracMatrix &mat, bool orthogonal, bool apply_mat, bool hermitian, TimeProfile &profile = dummy);

    /**
       @param x The optimum for the solution vector.
       @param b The source vector in the equation to be solved. This is not preserved.
       @param p The basis vectors in which we are building the guess
       @param q The basis vectors multiplied by A
    */
    void operator()(ColorSpinorField &x, const ColorSpinorField &b, std::vector<ColorSpinorField> &p,
                    std::vector<ColorSpinorField> &q);
  };

  using ColorSpinorFieldSet = ColorSpinorField;

  //forward declaration
  class EigCGArgs;

  class IncEigCG : public Solver {

  private:
    Solver *K;
    SolverParam Kparam; // parameters for preconditioner solve

    ColorSpinorFieldSet *Vm;  //eigCG search vectors  (spinor matrix of size eigen_vector_length x m)

    ColorSpinorField *rp;       //! residual vector
    ColorSpinorField *yp;       //! high precision accumulator
    ColorSpinorField* p;  // conjugate vector
    ColorSpinorField* Ap; // mat * conjugate vector
    ColorSpinorField *Az;       // mat * conjugate vector from the previous iteration
    ColorSpinorField *r_pre;    //! residual passed to preconditioner
    ColorSpinorField *p_pre;    //! preconditioner result

    EigCGArgs *eigcg_args;

    TimeProfile &profile; // time profile for initCG solver

    bool init = false;

public:
  IncEigCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
           TimeProfile &profile);

  virtual ~IncEigCG();

  /**
     @brief Expands deflation space.
     @param V Composite field container of new eigenvectors
     @param n_ev number of vectors to load
   */
  void increment(ColorSpinorField &V, int n_ev);

  void RestartVT(const double beta, const double rho);
  void UpdateVm(ColorSpinorField &res, double beta, double sqrtr2);
  // EigCG solver:
  int eigCGsolve(ColorSpinorField &out, ColorSpinorField &in);
  // InitCG solver:
  int initCGsolve(ColorSpinorField &out, ColorSpinorField &in);
  // Incremental eigCG solver (for eigcg and initcg calls)
  void operator()(ColorSpinorField &out, ColorSpinorField &in);

  bool hermitian() { return true; } // EigCG is only for Hermitian systems
  };

//forward declaration
 class GMResDRArgs;

 class GMResDR : public Solver {

  private:
    Solver *K;
    SolverParam Kparam; // parameters for preconditioner solve

    ColorSpinorFieldSet *Vm;//arnoldi basis vectors, size (m+1)
    ColorSpinorFieldSet *Zm;//arnoldi basis vectors, size (m+1)

    ColorSpinorField *rp;       //! residual vector
    ColorSpinorField *yp;       //! high precision accumulator
    ColorSpinorField *r_sloppy; //! sloppy residual vector
    ColorSpinorField *r_pre;    //! residual passed to preconditioner
    ColorSpinorField *p_pre;    //! preconditioner result

    TimeProfile &profile;    //time profile for initCG solver

    GMResDRArgs *gmresdr_args;

    bool init = false;

  public:
    GMResDR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
            TimeProfile &profile);
    GMResDR(const DiracMatrix &mat, Solver &K, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
            SolverParam &param, TimeProfile &profile);

    virtual ~GMResDR();

    //GMRES-DR solver
    void operator()(ColorSpinorField &out, ColorSpinorField &in);
    //
    //GMRESDR method
    void RunDeflatedCycles (ColorSpinorField *out, ColorSpinorField *in, const double tol_threshold);
    //
    int FlexArnoldiProcedure (const int start_idx, const bool do_givens);

    void RestartVZH();

    void UpdateSolution(ColorSpinorField *x, ColorSpinorField *r, bool do_gels);

    bool hermitian() { return false; } // GMRESDR for any linear system
 };

 /**
    @brief This is an object that captures the state required for a
    deflated solver.
 */
 struct deflation_space : public Object {
   bool svd;                            /** Whether this space is for an SVD deflaton */
   std::vector<ColorSpinorField> evecs; /** Container for the eigenvectors */
   std::vector<Complex> evals;          /** The eigenvalues */
 };

 /**
   @brief Returns if a solver is CA or not
   @return true if CA, false otherwise
 */
 bool is_ca_solver(QudaInverterType type);

} // namespace quda
