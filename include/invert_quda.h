#pragma once

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>
#include <qio_field.h>
#include <eigensolve_quda.h>
#include <vector>
#include <memory>

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

    /**< Preserve the source or not in the linear solver (deprecated?) */
    QudaPreserveSource preserve_source;

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

    /** Whether to use additive or multiplicative Schwarz preconditioning */
    QudaSchwarzType schwarz_type;

    /**< The time taken by the solver */
    double secs;

    /**< The Gflops rate of the solver */
    double gflops;

    // Incremental EigCG solver parameters
    /**< The precision of the Ritz vectors */
    QudaPrecision precision_ritz;//also search space precision

    int nev;//number of eigenvectors produced by EigCG
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
      preserve_source(param.preserve_source),
      return_residual(preserve_source == QUDA_PRESERVE_SOURCE_NO ? true : false),
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
      schwarz_type(param.schwarz_type),
      secs(param.secs),
      gflops(param.gflops),
      precision_ritz(param.cuda_prec_ritz),
      nev(param.nev),
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
      preserve_source(param.preserve_source),
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
      schwarz_type(param.schwarz_type),
      secs(param.secs),
      gflops(param.gflops),
      precision_ritz(param.precision_ritz),
      nev(param.nev),
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
      reduceDouble(gflops);
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

    SolverParam &param;
    TimeProfile &profile;
    int node_parity;
    EigenSolver *eig_solve; /** Eigensolver object. */
    bool deflate_init;      /** If true, the deflation space has been computed. */
    bool deflate_compute;   /** If true, instruct the solver to create a deflation space. */
    bool recompute_evals;   /** If true, instruct the solver to recompute evals from an existing deflation space. */
    std::vector<ColorSpinorField *> evecs;     /** Holds the eigenvectors. */
    std::vector<Complex> evals;                /** Holds the eigenvalues. */

  public:
    Solver(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
           SolverParam &param, TimeProfile &profile);
    virtual ~Solver();

    virtual void operator()(ColorSpinorField &out, ColorSpinorField &in) = 0;

    virtual void blocksolve(ColorSpinorField &out, ColorSpinorField &in);

    const DiracMatrix& M() { return mat; }
    const DiracMatrix& Msloppy() { return matSloppy; }
    const DiracMatrix& Mprecon() { return matPrecon; }

    /**
       @return Whether the solver is only for Hermitian systems
     */
    virtual bool hermitian() = 0;

    /**
       @brief Solver factory
    */
    static Solver* create(SolverParam &param, const DiracMatrix &mat, const DiracMatrix &matSloppy,
			  const DiracMatrix &matPrecon, TimeProfile &profile);

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
    void injectDeflationSpace(std::vector<ColorSpinorField *> &defl_space);

    /**
       @brief Extracts the deflation space from the solver to the
       vector argument.  Note the solver deflation space is reduced to
       zero size as a result of calling this function, with
       responsibility for the space transferred to the argument.
       @param[in,out] defl_space the extracted deflation space.  On
       input, this vector should have zero size.
    */
    void extractDeflationSpace(std::vector<ColorSpinorField *> &defl_space);

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
    ColorSpinorField *yp, *rp, *rnewp, *pp, *App, *tmpp, *tmp2p, *tmp3p, *rSloppyp, *xSloppyp;
    std::vector<ColorSpinorField*> p;
    bool init;

  public:
    CG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile);
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
  };

  class CGNE : public CG
  {

  private:
    DiracMMdag mmdag;
    DiracMMdag mmdagSloppy;
    DiracMMdag mmdagPrecon;
    ColorSpinorField *xp;
    ColorSpinorField *yp;
    bool init;

  public:
    CGNE(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
         TimeProfile &profile);
    virtual ~CGNE();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** CGNE is for any system */
  };

  class CGNR : public CG
  {

  private:
    DiracMdagM mdagm;
    DiracMdagM mdagmSloppy;
    DiracMdagM mdagmPrecon;
    ColorSpinorField *bp;
    bool init;

  public:
    CGNR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
         TimeProfile &profile);
    virtual ~CGNR();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** CGNR is for any system */
  };

  class CG3 : public Solver
  {

  private:
    // pointers to fields to avoid multiple creation overhead
    ColorSpinorField *yp, *rp, *tmpp, *ArSp, *rSp, *xSp, *xS_oldp, *tmpSp, *rS_oldp, *tmp2Sp;
    bool init;

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
    ColorSpinorField *xp;
    ColorSpinorField *yp;
    bool init;

  public:
    CG3NE(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
          TimeProfile &profile);
    virtual ~CG3NE();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** CG3NE is for any system */
  };

  class CG3NR : public CG3
  {

  private:
    DiracMdagM mdagm;
    DiracMdagM mdagmSloppy;
    DiracMdagM mdagmPrecon;
    ColorSpinorField *bp;
    bool init;

  public:
    CG3NR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
          TimeProfile &profile);
    virtual ~CG3NR();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** CG3NR is for any system */
  };

  class MPCG : public Solver {
    private:
      void computeMatrixPowers(cudaColorSpinorField out[], cudaColorSpinorField &in, int nvec);
      void computeMatrixPowers(std::vector<cudaColorSpinorField>& out, std::vector<cudaColorSpinorField>& in, int nsteps);

    public:
      MPCG(const DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
      virtual ~MPCG();

      void operator()(ColorSpinorField &out, ColorSpinorField &in);
      virtual bool hermitian() { return true; } /** MPCG is only Hermitian system */
  };


  class PreconCG : public Solver {
    private:
      Solver *K;
      SolverParam Kparam; // parameters for preconditioner solve

    public:
      PreconCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile);

      virtual ~PreconCG();

      void operator()(ColorSpinorField &out, ColorSpinorField &in);
      virtual bool hermitian() { return true; } /** MPCG is only Hermitian system */
  };


  class BiCGstab : public Solver {

  private:
    // pointers to fields to avoid multiple creation overhead
    ColorSpinorField *yp, *rp, *pp, *vp, *tmpp, *tp;
    bool init;

  public:
    BiCGstab(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
	     SolverParam &param, TimeProfile &profile);
    virtual ~BiCGstab();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** BiCGStab is for any linear system */
  };

  class SimpleBiCGstab : public Solver {

  private:

    // pointers to fields to avoid multiple creation overhead
    cudaColorSpinorField *yp, *rp, *pp, *vp, *tmpp, *tp;
    bool init;

  public:
    SimpleBiCGstab(const DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
    virtual ~SimpleBiCGstab();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** BiCGStab is for any linear system */
  };

  class MPBiCGstab : public Solver {

  private:
    void computeMatrixPowers(std::vector<cudaColorSpinorField>& pr, cudaColorSpinorField& p, cudaColorSpinorField& r, int nsteps);

  public:
    MPBiCGstab(const DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
    virtual ~MPBiCGstab();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** BiCGStab is for any linear system */
  };

  class BiCGstabL : public Solver {

  private:
    /**
       The size of the Krylov space that BiCGstabL uses.
     */
    int nKrylov; // in the language of BiCGstabL, this is L.

    // Various coefficients and params needed on each iteration.
    Complex rho0, rho1, alpha, omega, beta; // Various coefficients for the BiCG part of BiCGstab-L.
    Complex *gamma, *gamma_prime, *gamma_prime_prime; // Parameters for MR part of BiCGstab-L. (L+1) length.
    Complex **tau; // Parameters for MR part of BiCGstab-L. Tech. modified Gram-Schmidt coeffs. (L+1)x(L+1) length.
    double *sigma; // Parameters for MR part of BiCGstab-L. Tech. the normalization part of Gram-Scmidt. (L+1) length.

    // pointers to fields to avoid multiple creation overhead
    // full precision fields
    ColorSpinorField *r_fullp;   //! Full precision residual.
    ColorSpinorField *yp;        //! Full precision temporary.
    // sloppy precision fields
    ColorSpinorField *tempp;     //! Sloppy temporary vector.
    std::vector<ColorSpinorField*> r; // Current residual + intermediate residual values, along the MR.
    std::vector<ColorSpinorField*> u; // Search directions.

    // Saved, preallocated vectors. (may or may not get used depending on precision.)
    ColorSpinorField *x_sloppy_saved_p; //! Sloppy solution vector.
    ColorSpinorField *r0_saved_p;       //! Shadow residual, in BiCG language.
    ColorSpinorField *r_sloppy_saved_p; //! Current residual, in BiCG language.

    /**
       Internal routine for reliable updates. Made to not conflict with BiCGstab's implementation.
     */
    int reliable(double &rNorm, double &maxrx, double &maxrr, const double &r2, const double &delta);

    /**
       Internal routines for pipelined Gram-Schmidt. Made to not conflict with GCR's implementation.
     */
    void computeTau(Complex **tau, double *sigma, std::vector<ColorSpinorField*> r, int begin, int size, int j);
    void updateR(Complex **tau, std::vector<ColorSpinorField*> r, int begin, int size, int j);
    void orthoDir(Complex **tau, double* sigma, std::vector<ColorSpinorField*> r, int j, int pipeline);

    void updateUend(Complex* gamma, std::vector<ColorSpinorField*> u, int nKrylov);
    void updateXRend(Complex* gamma, Complex* gamma_prime, Complex* gamma_prime_prime,
                                std::vector<ColorSpinorField*> r, ColorSpinorField& x, int nKrylov);

    /**
       Solver uses lazy allocation: this flag determines whether we have allocated or not.
     */
    bool init;

    std::string solver_name; // holds BiCGstab-l, where 'l' literally equals nKrylov.

  public:
    BiCGstabL(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);
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
    int nKrylov;

    Complex *alpha;
    Complex **beta;
    double *gamma;

    /**
       Solver uses lazy allocation: this flag to determine whether we have allocated.
     */
    bool init;

    ColorSpinorField *rp;       //! residual vector
    ColorSpinorField *tmpp;     //! temporary for mat-vec
    ColorSpinorField *tmp_sloppy; //! temporary for sloppy mat-vec
    ColorSpinorField *r_sloppy; //! sloppy residual vector

    std::vector<ColorSpinorField*> p;  // GCR direction vectors
    std::vector<ColorSpinorField*> Ap; // mat * direction vectors

  public:
    GCR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
	SolverParam &param, TimeProfile &profile);

    /**
       @param K Preconditioner
    */
    GCR(const DiracMatrix &mat, Solver &K, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
        TimeProfile &profile);
    virtual ~GCR();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** GCR is for any linear system */
  };

  class MR : public Solver {

  private:
    ColorSpinorField *rp;
    ColorSpinorField *r_sloppy;
    ColorSpinorField *Arp;
    ColorSpinorField *tmpp;
    ColorSpinorField *tmp_sloppy;
    ColorSpinorField *x_sloppy;
    bool init;

  public:
    MR(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);
    virtual ~MR();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** MR is for any linear system */
  };

  /**
     @brief Communication-avoiding CG solver.  This solver does
     un-preconditioned CG, running in steps of nKrylov, build up a
     polynomial in the linear operator of length nKrylov, and then
     performs a steepest descent minimization on the resulting basis
     vectors.  For now only implemented using the power basis so is
     only useful as a preconditioner.
   */
  class CACG : public Solver {

  private:
    bool init;

    bool lambda_init;
    QudaCABasis basis;

    Complex *Q_AQandg; // Fused inner product matrix
    Complex *Q_AS; // inner product matrix
    Complex *alpha; // QAQ^{-1} g
    Complex *beta; // QAQ^{-1} QpolyS

    ColorSpinorField *rp;
    ColorSpinorField *tmpp;
    ColorSpinorField *tmpp2;
    ColorSpinorField *tmp_sloppy;
    ColorSpinorField *tmp_sloppy2;

    std::vector<ColorSpinorField*> S;  // residual vectors
    std::vector<ColorSpinorField*> AS; // mat * residual vectors. Can be replaced by a single temporary.
    std::vector<ColorSpinorField*> Q;  // CG direction vectors
    std::vector<ColorSpinorField*> Qtmp; // CG direction vectors for pointer swap
    std::vector<ColorSpinorField*> AQ; // mat * CG direction vectors.
                                       // it's possible to avoid carrying these
                                       // around, but there's a stability penalty,
                                       // and computing QAQ becomes a pain (though
                                       // it does let you fuse the reductions...)

    /**
       @brief Initiate the fields needed by the solver
       @param[in] b Source vector used for solver meta data.  If we're
       not preserving the source vector and we have a uni-precision
       solver, we set p[0] = b to save memory and memory copying.
    */
    void create(ColorSpinorField &b);

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
    CACG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile);
    virtual ~CACG();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return true; } /** CG is only for Hermitian systems */
  };

  class CACGNE : public CACG {

  private:
    DiracMMdag mmdag;
    DiracMMdag mmdagSloppy;
    DiracMMdag mmdagPrecon;
    ColorSpinorField *xp;
    ColorSpinorField *yp;
    bool init;

  public:
    CACGNE(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile);
    virtual ~CACGNE();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** CGNE is for any linear system */
  };

  class CACGNR : public CACG {

  private:
    DiracMdagM mdagm;
    DiracMdagM mdagmSloppy;
    DiracMdagM mdagmPrecon;
    ColorSpinorField *bp;
    bool init;

  public:
    CACGNR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile);
    virtual ~CACGNR();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** CGNE is for any linear system */
  };

  /**
     @brief Communication-avoiding GCR solver.  This solver does
     un-preconditioned GCR, first building up a polynomial in the
     linear operator of length nKrylov, and then performs a minimum
     residual extrapolation on the resulting basis vectors.  For use as
     a multigrid smoother with minimum global synchronization.
   */
  class CAGCR : public Solver {

  private:
    const DiracMdagM matMdagM; // used by the eigensolver
    bool init;
    const bool use_source; // whether we can reuse the source vector

    // Basis. Currently anything except POWER_BASIS causes a warning
    // then swap to POWER_BASIS.
    QudaCABasis basis;

    Complex *alpha; // Solution coefficient vectors

    ColorSpinorField *rp;
    ColorSpinorField *tmpp;
    ColorSpinorField *tmp_sloppy;

    std::vector<ColorSpinorField*> p;  // GCR direction vectors
    std::vector<ColorSpinorField*> q;  // mat * direction vectors

    /**
       @brief Initiate the fields needed by the solver
       @param[in] b Source vector used for solver meta data.  If we're
       not preserving the source vector and we have a uni-precision
       solver, we set p[0] = b to save memory and memory copying.
    */
    void create(ColorSpinorField &b);

    /**
       @brief Solve the equation A p_k psi_k = q_k psi_k = b by minimizing the
       least square residual using Eigen's LDLT Cholesky for numerical stability
       @param[out] psi Array of coefficients
       @param[in] q Search direction vectors with the operator applied
       @param[in] b Source vector against which we are solving
    */
    void solve(Complex *psi_, std::vector<ColorSpinorField*> &q, ColorSpinorField &b);

public:
    CAGCR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile);
    virtual ~CAGCR();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    virtual bool hermitian() { return false; } /** GCR is for any linear system */
  };

  // Steepest descent solver used as a preconditioner
  class SD : public Solver {
    private:
      cudaColorSpinorField *Ar;
      cudaColorSpinorField *r;
      cudaColorSpinorField *y;
      bool init;

    public:
      SD(const DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
      virtual ~SD();

      void operator()(ColorSpinorField &out, ColorSpinorField &in);

      virtual bool hermitian() { return false; } /** CGNE is for any linear system */
  };

  // Extended Steepest Descent solver used for overlapping DD preconditioning
  class XSD : public Solver
  {
    private:
      cudaColorSpinorField *xx;
      cudaColorSpinorField *bx;
      SD *sd; // extended sd is implemented using standard sd
      bool init;
      int R[4];

    public:
      XSD(const DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
      virtual ~XSD();

      void operator()(ColorSpinorField &out, ColorSpinorField &in);

      virtual bool hermitian() { return false; } /** CGNE is for any linear system */
  };

  class PreconditionedSolver : public Solver
  {
private:
    Solver *solver;
    const Dirac &dirac;
    const char *prefix;

public:
    PreconditionedSolver(Solver &solver, const Dirac &dirac, SolverParam &param, TimeProfile &profile,
                         const char *prefix) :
      Solver(solver.M(), solver.Msloppy(), solver.Mprecon(), param, profile),
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

      dirac.prepare(in, out, x, b, solution_type);
      (*solver)(*out, *in);
      dirac.reconstruct(x, b, solution_type);

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

  public:
    MultiShiftSolver(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
      mat(mat), matSloppy(matSloppy), param(param), profile(profile) { ; }
    virtual ~MultiShiftSolver() { ; }

    virtual void operator()(std::vector<ColorSpinorField*> out, ColorSpinorField &in) = 0;
    bool convergence(const double *r2, const double *r2_tol, int n) const;
  };

/**
 * @brief Multi-Shift Conjugate Gradient Solver.
 */
  class MultiShiftCG : public MultiShiftSolver {

  public:
    MultiShiftCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);
    virtual ~MultiShiftCG();
/**
 * @brief Run multi-shift and return Krylov-space at the end of the solve in p and r2_old_arry.
 *
 * @param out std::vector of pointer to solutions for all the shifts.
 * @param in right-hand side.
 * @param p std::vector of pointers to hold search directions. Note this will be resized as necessary.
 * @param r2_old_array pointer to last values of r2_old for old shifts. Needs to be large enough to hold r2_old for all shifts.
 */
    void operator()(std::vector<ColorSpinorField*>x, ColorSpinorField &b, std::vector<ColorSpinorField*> &p, double* r2_old_array );

/**
 * @brief Run multi-shift and return Krylov-space at the end of the solve in p and r2_old_arry.
 *
 * @param out std::vector of pointer to solutions for all the shifts.
 * @param in right-hand side.
 */
    void operator()(std::vector<ColorSpinorField*> out, ColorSpinorField &in){
      std::unique_ptr<double[]> r2_old(new double[QUDA_MAX_MULTI_SHIFT]);
      std::vector<ColorSpinorField*> p;

      (*this)(out, in, p, r2_old.get());

      for (auto& pp : p) delete pp;
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
    bool hermitian; //! whether A is hermitian ot not
    TimeProfile &profile;

    /**
       @brief Solve the equation A p_k psi_k = q_k psi_k = b by minimizing the
       residual and using Eigen's SVD algorithm for numerical stability
       @param[out] psi Array of coefficients
       @param[in] p Search direction vectors
       @param[in] q Search direction vectors with the operator applied
       @param[in] hermitian Whether the linear system is Hermitian or not
    */
    void solve(Complex *psi_, std::vector<ColorSpinorField*> &p,
               std::vector<ColorSpinorField*> &q, ColorSpinorField &b, bool hermitian);

  public:
    /**
       @param mat The operator for the linear system we wish to solve
       @param orthogonal Whether to construct an orthogonal basis prior to constructing the linear system
       @param apply_mat Whether to apply the operator in place or assume q already contains this
       @profile Timing profile to use
    */
    MinResExt(const DiracMatrix &mat, bool orthogonal, bool apply_mat, bool hermitian, TimeProfile &profile);
    virtual ~MinResExt();

    /**
       @param x The optimum for the solution vector.
       @param b The source vector in the equation to be solved. This is not preserved and is overwritten by the new residual.
       @param basis Vector of pairs storing the basis (p,Ap)
    */
    void operator()(ColorSpinorField &x, ColorSpinorField &b,
		    std::vector<std::pair<ColorSpinorField*,ColorSpinorField*> > basis);

    /**
       @param x The optimum for the solution vector.
       @param b The source vector in the equation to be solved. This is not preserved.
       @param p The basis vectors in which we are building the guess
       @param q The basis vectors multiplied by A
    */
    void operator()(ColorSpinorField &x, ColorSpinorField &b,
		    std::vector<ColorSpinorField*> p,
		    std::vector<ColorSpinorField*> q);
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
    ColorSpinorField *tmpp;     //! temporary for mat-vec
    ColorSpinorField* Az; // mat * conjugate vector from the previous iteration
    ColorSpinorField *r_pre;    //! residual passed to preconditioner
    ColorSpinorField *p_pre;    //! preconditioner result

    EigCGArgs *eigcg_args;

    TimeProfile &profile; // time profile for initCG solver

    bool init;

public:
    IncEigCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile);

    virtual ~IncEigCG();

    /**
       @brief Expands deflation space.
       @param V Composite field container of new eigenvectors
       @param nev number of vectors to load
     */
    void increment(ColorSpinorField &V, int nev);

    void RestartVT(const double beta, const double rho);
    void UpdateVm(ColorSpinorField &res, double beta, double sqrtr2);
    //EigCG solver:
    int eigCGsolve(ColorSpinorField &out, ColorSpinorField &in);
    //InitCG solver:
    int initCGsolve(ColorSpinorField &out, ColorSpinorField &in);
    //Incremental eigCG solver (for eigcg and initcg calls)
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
    ColorSpinorField *tmpp;     //! temporary for mat-vec
    ColorSpinorField *r_sloppy; //! sloppy residual vector
    ColorSpinorField *r_pre;    //! residual passed to preconditioner
    ColorSpinorField *p_pre;    //! preconditioner result

    TimeProfile &profile;    //time profile for initCG solver

    GMResDRArgs *gmresdr_args;

    bool init;

  public:

    GMResDR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile);
    GMResDR(const DiracMatrix &mat, Solver &K, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile);

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
    bool svd;                              /** Whether this space is for an SVD deflaton */
    std::vector<ColorSpinorField *> evecs; /** Container for the eigenvectors */
    std::vector<Complex> evals;            /** The eigenvalues */
  };

} // namespace quda
