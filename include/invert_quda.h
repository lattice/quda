#ifndef _INVERT_QUDA_H
#define _INVERT_QUDA_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>
#include <vector>

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
     * Whether to use the L2 relative residual, L2 absolute residual
     * or Fermilab heavy-quark residual, or combinations therein to
     * determine convergence.  To require that multiple stopping
     * conditions are satisfied, use a bitwise OR as follows:
     *
     * p.residual_type = (QudaResidualType) (QUDA_L2_RELATIVE_RESIDUAL
     *                                     | QUDA_HEAVY_QUARK_RESIDUAL);
     */
    QudaResidualType residual_type;

    /**< Whether to use an initial guess in the solver or not */
    QudaUseInitGuess use_init_guess;

    /**< Whether to solve linear system with zero RHS */
    QudaComputeNullVector compute_null_vector;

    /**< Reliable update tolerance */
    double delta;

    /**< Whether to keep the partial solution accumulator in sloppy precision */
    bool use_sloppy_partial_accumulator;

    /**< This parameter determines how many consective reliable update
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge */
    int max_res_increase;

    /**< This parameter determines how many total reliable update
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge */
    int max_res_increase_total;

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

    /**< The precision used by the QUDA preconditioner */
    QudaPrecision precision_precondition;

    /**< Preserve the source or not in the linear solver (deprecated?) */
    QudaPreserveSource preserve_source;

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

    bool    use_reduced_vector_set;
    bool    use_cg_updates;
    double  cg_iterref_tol;
    int     eigcg_max_restarts;
    int     max_restart_num;
    double  inc_tol;
    double  eigenval_tol;

    QudaVerbosity verbosity_precondition; //! verbosity to use for preconditioner

    bool compute_true_res; //! whether to compute the true residual or not at the end

    bool is_preconditioner; //! whether the solver acting as a preconditioner for another solver

    bool global_reduction; //! whether to use a global or local (node) reduction for this solver

    /**
       Default constructor
     */
  SolverParam() : compute_null_vector(QUDA_COMPUTE_NULL_VECTOR_NO),
    verbosity_precondition(QUDA_SILENT), compute_true_res(true) { ; }

    /**
       Constructor that matches the initial values to that of the
       QudaInvertParam instance
       @param param The QudaInvertParam instance from which the values are copied
     */
    SolverParam(const QudaInvertParam &param) : inv_type(param.inv_type),
      inv_type_precondition(param.inv_type_precondition), preconditioner(param.preconditioner),
      residual_type(param.residual_type), use_init_guess(param.use_init_guess),
      compute_null_vector(QUDA_COMPUTE_NULL_VECTOR_NO), delta(param.reliable_delta),
      use_sloppy_partial_accumulator(param.use_sloppy_partial_accumulator),
      max_res_increase(param.max_res_increase), max_res_increase_total(param.max_res_increase_total),
      heavy_quark_check(param.heavy_quark_check), pipeline(param.pipeline),
      tol(param.tol), tol_restart(param.tol_restart), tol_hq(param.tol_hq),
      true_res(param.true_res), true_res_hq(param.true_res_hq),
      maxiter(param.maxiter), iter(param.iter),
      precision(param.cuda_prec), precision_sloppy(param.cuda_prec_sloppy),
      precision_precondition(param.cuda_prec_precondition),
      preserve_source(param.preserve_source), num_src(param.num_src), num_offset(param.num_offset),
      Nsteps(param.Nsteps), Nkrylov(param.gcrNkrylov), precondition_cycle(param.precondition_cycle),
      tol_precondition(param.tol_precondition), maxiter_precondition(param.maxiter_precondition),
      omega(param.omega), schwarz_type(param.schwarz_type), secs(param.secs), gflops(param.gflops),
      precision_ritz(param.cuda_prec_ritz), nev(param.nev), m(param.max_search_dim),
      deflation_grid(param.deflation_grid), rhs_idx(0), use_reduced_vector_set(param.use_reduced_vector_set),
      use_cg_updates(param.use_cg_updates), cg_iterref_tol(param.cg_iterref_tol),
      eigcg_max_restarts(param.eigcg_max_restarts), max_restart_num(param.max_restart_num),
      inc_tol(param.inc_tol), eigenval_tol(param.eigenval_tol),
      verbosity_precondition(param.verbosity_precondition), compute_true_res(true),
      is_preconditioner(false), global_reduction(true)
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

    SolverParam(const SolverParam &param) : inv_type(param.inv_type),
      inv_type_precondition(param.inv_type_precondition), preconditioner(param.preconditioner),
      residual_type(param.residual_type), use_init_guess(param.use_init_guess),
      delta(param.delta), use_sloppy_partial_accumulator(param.use_sloppy_partial_accumulator),
      max_res_increase(param.max_res_increase), max_res_increase_total(param.max_res_increase_total),
      heavy_quark_check(param.heavy_quark_check), pipeline(param.pipeline),
      tol(param.tol), tol_restart(param.tol_restart), tol_hq(param.tol_hq),
      true_res(param.true_res), true_res_hq(param.true_res_hq),
      maxiter(param.maxiter), iter(param.iter),
      precision(param.precision), precision_sloppy(param.precision_sloppy),
      precision_precondition(param.precision_precondition),
      preserve_source(param.preserve_source), num_offset(param.num_offset),
      Nsteps(param.Nsteps), Nkrylov(param.Nkrylov), precondition_cycle(param.precondition_cycle),
      tol_precondition(param.tol_precondition), maxiter_precondition(param.maxiter_precondition),
      omega(param.omega), schwarz_type(param.schwarz_type), secs(param.secs), gflops(param.gflops),
      precision_ritz(param.precision_ritz), nev(param.nev), m(param.m),
      deflation_grid(param.deflation_grid), rhs_idx(0), use_reduced_vector_set(param.use_reduced_vector_set),
      use_cg_updates(param.use_cg_updates), cg_iterref_tol(param.cg_iterref_tol),
      eigcg_max_restarts(param.eigcg_max_restarts), max_restart_num(param.max_restart_num),
      inc_tol(param.inc_tol), eigenval_tol(param.eigenval_tol),
      verbosity_precondition(param.verbosity_precondition), compute_true_res(param.compute_true_res),
      is_preconditioner(param.is_preconditioner), global_reduction(param.global_reduction)
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
    }

    void updateRhsIndex(QudaInvertParam &param) {
      //for incremental eigCG:
      rhs_idx = param.rhs_idx;
    }

  };

  class Solver {

  protected:
    SolverParam &param;
    TimeProfile &profile;

  public:
    Solver(SolverParam &param, TimeProfile &profile) : param(param), profile(profile) { ; }
    virtual ~Solver() { ; }

    virtual void operator()(ColorSpinorField &out, ColorSpinorField &in) = 0;

    virtual void solve(ColorSpinorField &out, ColorSpinorField &in);


    /**
       Solver factory
    */
    static Solver* create(SolverParam &param, DiracMatrix &mat, DiracMatrix &matSloppy,
			  DiracMatrix &matPrecon, TimeProfile &profile);

    /**
       Set the solver stopping condition
       @param b2 L2 norm squared of the source vector
     */
    static double stopping(const double &tol, const double &b2, QudaResidualType residual_type);

    /**
       Test for solver convergence
       @param r2 L2 norm squared of the residual
       @param hq2 Heavy quark residual
       @param r2_tol Solver L2 tolerance
       @param hq_tol Solver heavy-quark tolerance
     */
    bool convergence(const double &r2, const double &hq2, const double &r2_tol,
		     const double &hq_tol);

    /**
       Test for HQ solver convergence -- ignore L2 residual
       @param r2 L2 norm squared of the residual
       @param hq2 Heavy quark residual
       @param r2_tol Solver L2 tolerance
       @param hq_tol Solver heavy-quark tolerance
     */
    bool convergenceHQ(const double &r2, const double &hq2, const double &r2_tol,
         const double &hq_tol);

    /**
       Test for L2 solver convergence -- ignore HQ residual
       @param r2 L2 norm squared of the residual
       @param hq2 Heavy quark residual
       @param r2_tol Solver L2 tolerance
       @param hq_tol Solver heavy-quark tolerance
     */
    bool convergenceL2(const double &r2, const double &hq2, const double &r2_tol,
         const double &hq_tol);

    /**
       Prints out the running statistics of the solver (requires a verbosity of QUDA_VERBOSE)
     */
    void PrintStats(const char*, int k, const double &r2, const double &b2, const double &hq2);

    /**
	Prints out the summary of the solver convergence (requires a
	versbosity of QUDA_SUMMARIZE).  Assumes
	SolverParam.true_res and SolverParam.true_res_hq has
	been set
    */
    void PrintSummary(const char *name, int k, const double &r2, const double &b2);

    /**
     * Return flops
     * @return flops expended by this operator
     */
    virtual double flops() const { return 0; }
  };

  class CG : public Solver {

  private:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    // pointers to fields to avoid multiple creation overhead
    ColorSpinorField *yp, *rp, *App, *tmpp;
    bool init;

  public:
    CG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);
    virtual ~CG();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);
    void solve(ColorSpinorField& out, ColorSpinorField& in);
  };



  class MPCG : public Solver {
    private:
      const DiracMatrix &mat;
      void computeMatrixPowers(cudaColorSpinorField out[], cudaColorSpinorField &in, int nvec);
      void computeMatrixPowers(std::vector<cudaColorSpinorField>& out, std::vector<cudaColorSpinorField>& in, int nsteps);


    public:
      MPCG(DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
      virtual ~MPCG();

      void operator()(ColorSpinorField &out, ColorSpinorField &in);
  };



  class PreconCG : public Solver {
    private:
      const DiracMatrix &mat;
      const DiracMatrix &matSloppy;
      const DiracMatrix &matPrecon;

      Solver *K;
      SolverParam Kparam; // parameters for preconditioner solve

    public:
      PreconCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
               SolverParam &param, TimeProfile &profile);
      virtual ~PreconCG();

      void operator()(ColorSpinorField &out, ColorSpinorField &in);
  };


  class BiCGstab : public Solver {

  private:
    DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    const DiracMatrix &matPrecon;

    // pointers to fields to avoid multiple creation overhead
    ColorSpinorField *yp, *rp, *pp, *vp, *tmpp, *tp;
    bool init;

  public:
    BiCGstab(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
	     SolverParam &param, TimeProfile &profile);
    virtual ~BiCGstab();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);
  };

  class SimpleBiCGstab : public Solver {

  private:
    DiracMatrix &mat;

    // pointers to fields to avoid multiple creation overhead
    cudaColorSpinorField *yp, *rp, *pp, *vp, *tmpp, *tp;
    bool init;

  public:
    SimpleBiCGstab(DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
    virtual ~SimpleBiCGstab();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);
  };

  class MPBiCGstab : public Solver {

  private:
    DiracMatrix &mat;
    void computeMatrixPowers(std::vector<cudaColorSpinorField>& pr, cudaColorSpinorField& p, cudaColorSpinorField& r, int nsteps);

  public:
    MPBiCGstab(DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
    virtual ~MPBiCGstab();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);
  };

  class BiCGstabL : public Solver {

  private:
    DiracMatrix &mat;
    const DiracMatrix &matSloppy;

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
    BiCGstabL(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);
    virtual ~BiCGstabL();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);
  };

  class GCR : public Solver {

  private:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    const DiracMatrix &matPrecon;

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
    ColorSpinorField *yp;       //! high precision accumulator
    ColorSpinorField *tmpp;     //! temporary for mat-vec
    ColorSpinorField *x_sloppy; //! sloppy solution vector
    ColorSpinorField *r_sloppy; //! sloppy residual vector
    ColorSpinorField *r_pre;    //! residual passed to preconditioner
    ColorSpinorField *p_pre;    //! preconditioner result
    ColorSpinorField *rM;       //! residual vector for doing multi-cycle preconditioning

    std::vector<ColorSpinorField*> p;  // GCR direction vectors
    std::vector<ColorSpinorField*> Ap; // mat * direction vectors

  public:
    GCR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
	SolverParam &param, TimeProfile &profile);

    /**
       @param K Preconditioner
     */
    GCR(DiracMatrix &mat, Solver &K, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
	SolverParam &param, TimeProfile &profile);
    virtual ~GCR();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);
  };

  class MR : public Solver {

  private:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    ColorSpinorField *rp;
    ColorSpinorField *Arp;
    ColorSpinorField *tmpp;
    ColorSpinorField *yp;  //Holds initial guess if applicable
    bool init;
    bool allocate_r;
    bool allocate_y;

  public:
    MR(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);
    virtual ~MR();

    void operator()(ColorSpinorField &out, ColorSpinorField &in);
  };

  // Steepest descent solver used as a preconditioner
  class SD : public Solver {
    private:
      const DiracMatrix &mat;
      cudaColorSpinorField *Ar;
      cudaColorSpinorField *r;
      cudaColorSpinorField *y;
      bool init;

    public:
      SD(DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
      virtual ~SD();


      void operator()(ColorSpinorField &out, ColorSpinorField &in);
  };

  // Extended Steepest Descent solver used for overlapping DD preconditioning
  class XSD : public Solver {
    private:
      const DiracMatrix &mat;
      cudaColorSpinorField *xx;
      cudaColorSpinorField *bx;
      SD *sd; // extended sd is implemented using standard sd
      bool init;
      int R[4];

    public:
      XSD(DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
      virtual ~XSD();

      void operator()(ColorSpinorField &out, ColorSpinorField &in);
  };


  class PreconditionedSolver : public Solver {

  private:
    Solver *solver;
    const Dirac &dirac;
    const char *prefix;

  public:
  PreconditionedSolver(Solver &solver, const Dirac &dirac, SolverParam &param, TimeProfile &profile, const char *prefix)
    : Solver(param, profile), solver(&solver), dirac(dirac), prefix(prefix) { }
    virtual ~PreconditionedSolver() { delete solver; }

    void operator()(ColorSpinorField &x, ColorSpinorField &b) {
      setOutputPrefix(prefix);

      QudaSolutionType solution_type = b.SiteSubset() == QUDA_FULL_SITE_SUBSET ? QUDA_MAT_SOLUTION : QUDA_MATPC_SOLUTION;

      ColorSpinorField *out=nullptr;
      ColorSpinorField *in=nullptr;

      dirac.prepare(in, out, x, b, solution_type);
      (*solver)(*out, *in);
      dirac.reconstruct(x, b, solution_type);

      setOutputPrefix("");
    }
  };


  class MultiShiftSolver {

  protected:
    SolverParam &param;
    TimeProfile &profile;

  public:
    MultiShiftSolver(SolverParam &param, TimeProfile &profile) :
    param(param), profile(profile) { ; }
    virtual ~MultiShiftSolver() { ; }

    virtual void operator()(std::vector<ColorSpinorField*> out, ColorSpinorField &in) = 0;
  };

  class MultiShiftCG : public MultiShiftSolver {

  protected:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;

  public:
    MultiShiftCG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);
    virtual ~MultiShiftCG();

    void operator()(std::vector<ColorSpinorField*> out, ColorSpinorField &in);
  };



  /**
     This computes the optimum guess for the system Ax=b in the L2
     residual norm.  For use in the HMD force calculations using a
     minimal residual chronological method This computes the guess
     solution as a linear combination of a given number of previous
     solutions.  Following Brower et al, only the orthogonalised vector
     basis is stored to conserve memory.
  */
  class MinResExt {

  protected:
    const DiracMatrix &mat;
    TimeProfile &profile;

  public:
    MinResExt(DiracMatrix &mat, TimeProfile &profile);
    virtual ~MinResExt();

    /**
       param x The optimum for the solution vector.
       param b The source vector in the equation to be solved. This is not preserved.
       param p The basis vectors in which we are building the guess
       param q The basis vectors multipled by A
       param N The number of basis vectors
       return The residue of this guess.
    */
    void operator()(ColorSpinorField &x, ColorSpinorField &b,
		    std::vector<ColorSpinorField*> p,
		    std::vector<ColorSpinorField*> q, int N);
  };

  class DeflatedSolver {

  protected:
    SolverParam &param;
    TimeProfile *profile;

    //WARNING: eigcg_precision may not coinside with param.precision and param.precision_sloppy (both used for the initCG).
    //
    QudaPrecision eigcg_precision;//may be double or single.

  public:
    DeflatedSolver(SolverParam &param, TimeProfile *profile) : param(param), profile(profile)
    {
       eigcg_precision = param.precision_sloppy;//for mixed presicion use param.precision_sloppy
    }

    virtual ~DeflatedSolver() { ; }

    virtual void operator()(cudaColorSpinorField *out, cudaColorSpinorField *in) = 0;

//    virtual void Deflate(cudaColorSpinorField &out, cudaColorSpinorField &in) = 0;//extrenal method (not implemented yet)
    virtual void StoreRitzVecs(void *host_buffer, double *inv_eigenvals, const int *X, QudaInvertParam *inv_par, const int nev, bool cleanResources = false) = 0;//extrenal method

    virtual void CleanResources() = 0;

    // solver factory
    static DeflatedSolver* create(SolverParam &param, DiracMatrix *mat, DiracMatrix *matSloppy, DiracMatrix *matCGSloppy, DiracMatrix *matDeflate, TimeProfile *profile);

    bool convergence(const double &r2, const double &hq2, const double &r2_tol,
		     const double &hq_tol);

    /**
       Prints out the running statistics of the solver (requires a verbosity of QUDA_VERBOSE)
     */
    void PrintStats(const char*, int k, const double &r2, const double &b2, const double &hq2);

    /**
	Prints out the summary of the solver convergence (requires a
	versbosity of QUDA_SUMMARIZE).  Assumes
	SolverParam.true_res and SolverParam.true_res_hq has
	been set
    */
    void PrintSummary(const char *name, int k, const double &r2, const double &b2);

  };

  struct DeflationParam;//Forward declaration
  typedef cudaColorSpinorField cudaColorSpinorFieldSet;

  class IncEigCG : public DeflatedSolver {

  private:
    DiracMatrix *mat;
    DiracMatrix *matSloppy;

    DiracMatrix *matCGSloppy;

    const DiracMatrix *matDefl;

    QudaPrecision search_space_prec;
    cudaColorSpinorFieldSet *Vm;  //search vectors  (spinor matrix of size eigen_vector_length x m)

    SolverParam initCGparam; // parameters for initCG solver
    TimeProfile *profile;    //time profile for initCG solver

    bool eigcg_alloc;
    bool use_eigcg;

  public:

    IncEigCG(DiracMatrix *mat, DiracMatrix *matSloppy, DiracMatrix *matCGSloppy, DiracMatrix *matDefl, SolverParam &param, TimeProfile *profile);
    IncEigCG(SolverParam &param);

    virtual ~IncEigCG();

    //EigCG solver
    int EigCG(cudaColorSpinorField &out, cudaColorSpinorField &in);

    //Incremental eigCG solver (for eigcg and initcg calls)
    void operator()(cudaColorSpinorField *out, cudaColorSpinorField *in);

    //Compute  u dH^{-1} u^{dagger}b:
    void DeflateSpinor(cudaColorSpinorField &out, cudaColorSpinorField &in, DeflationParam *param, bool set2zero = true);
    //
    void DeflateSpinorReduced(cudaColorSpinorField &out, cudaColorSpinorField &in, DeflationParam *param, bool set2zero = true);

    //Deflation space management
    void CreateDeflationSpace(cudaColorSpinorField &eigcgSpinor, DeflationParam *&param);

    //extend projection matrix:
    //compute Q' = DiracM Q, (here U = [V, Q] - total Ritz set)
    //construct H-matrix components with Q'^{dag} Q', V^{dag} Q' and Q'^{dag} V
    //extend H-matrix with the components
    void ExpandDeflationSpace(DeflationParam *param, const int new_nev);
    //
    void DeleteDeflationSpace(DeflationParam *&param);
    //
    void DeleteEigCGSearchSpace();
    //
    void SaveEigCGRitzVecs(DeflationParam *param, bool cleanResources = false);
    //
    void StoreRitzVecs(void *host_buf, double *inv_eigenvals, const int *X, QudaInvertParam *inv_par, const int nev, bool cleanResources = false);
    //
    void CleanResources();

    void LoadEigenvectors(DeflationParam *param, int max_nevs, double tol = 1e-3);

    void ReportEigenvalueAccuracy(DeflationParam *param, int nevs_to_print);

  };

//forward declaration
 struct GMResDRDeflationParam;
 class GMResDRArgs;

 class GMResDR : public DeflatedSolver {

  private:

    DiracMatrix *mat;

    DiracMatrix *matSloppy;

    const DiracMatrix *matDefl;

    QudaPrecision gmres_space_prec;

    cudaColorSpinorFieldSet *Vm;//arnoldi basis vectors, size (m+1)

    TimeProfile *profile;    //time profile for initCG solver

    GMResDRArgs *args;

    bool gmres_alloc;

  public:

    GMResDR(DiracMatrix *mat, DiracMatrix *matSloppy, DiracMatrix *matDefl, SolverParam &param, TimeProfile *profile);
    GMResDR(SolverParam &param);

    virtual ~GMResDR();

    //GMRES-DR solver
    //void   GmresDRCycle(cudaColorSpinorField &out, cudaColorSpinorField &in, Complex *u);
    double GMResDRCycle(cudaColorSpinorField &x, double r2, Complex *u, const double stop);
    //GMRES-DR solver
    void operator()(cudaColorSpinorField *out, cudaColorSpinorField *in);
    //
    void StoreRitzVecs(void *host_buf, double *inv_eigenvals, const int *X, QudaInvertParam *inv_par, const int nev, bool cleanResources = false) {};
    //
    void CleanResources();
    //
    void PerformProjection(cudaColorSpinorField &x_sloppy, cudaColorSpinorField &r_sloppy, GMResDRDeflationParam *dpar);
    //GMRESDR method
    void RunDeflatedCycles (cudaColorSpinorField *out, cudaColorSpinorField *in, GMResDRDeflationParam *dpar, const double tol_threshold);
    //
    void RunProjectedCycles(cudaColorSpinorField *out, cudaColorSpinorField *in, GMResDRDeflationParam *dpar, const bool enforce_mixed_precision);

    void AllocateKrylovSubspace(ColorSpinorParam &csParam);

  };



} // namespace quda

#endif // _INVERT_QUDA_H
