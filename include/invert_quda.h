#ifndef _INVERT_QUDA_H
#define _INVERT_QUDA_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>

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

    /**< Reliable update tolerance */
    double delta;           

    /**< Whether to keep the partial solution accumulator in sloppy precision */
    bool use_sloppy_partial_accumulator;

    /**< Enable pipeline solver */
    int pipeline;

    /**< Solver tolerance in the L2 residual norm */
    double tol;             

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

    /** Actual heavy quark residual norm achieved in solver for each offset */
    double true_res_hq_offset[QUDA_MAX_MULTI_SHIFT]; 


    

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
    
    /**
       Constructor that matches the initial values to that of the
       QudaInvertParam instance
       @param param The QudaInvertParam instance from which the values are copied
     */
    SolverParam(QudaInvertParam &param) : inv_type(param.inv_type), 
      inv_type_precondition(param.inv_type_precondition), 
      residual_type(param.residual_type), use_init_guess(param.use_init_guess),
      delta(param.reliable_delta), use_sloppy_partial_accumulator(param.use_sloppy_partial_accumulator), 
      pipeline(param.pipeline), tol(param.tol), tol_hq(param.tol_hq), 
      true_res(param.true_res), true_res_hq(param.true_res_hq),
      maxiter(param.maxiter), iter(param.iter), 
      precision(param.cuda_prec), precision_sloppy(param.cuda_prec_sloppy), 
      precision_precondition(param.cuda_prec_precondition), 
      preserve_source(param.preserve_source), num_offset(param.num_offset), 
      Nkrylov(param.gcrNkrylov), precondition_cycle(param.precondition_cycle), 
      tol_precondition(param.tol_precondition), maxiter_precondition(param.maxiter_precondition), 
      omega(param.omega), schwarz_type(param.schwarz_type), secs(param.secs), gflops(param.gflops)
    { 
      for (int i=0; i<num_offset; i++) {
	offset[i] = param.offset[i];
	tol_offset[i] = param.tol_offset[i];
	tol_hq_offset[i] = param.tol_hq_offset[i];
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
      param.gflops = (param.gflops*param.secs + gflops*secs) / (param.secs + secs);
      param.secs += secs;
      if (offset >= 0) {
	param.true_res_offset[offset] = true_res_offset[offset];
	param.true_res_hq_offset[offset] = true_res_hq_offset[offset];
      } else {
	for (int i=0; i<num_offset; i++) {
	  param.true_res_offset[i] = true_res_offset[i];
	  param.true_res_hq_offset[i] = true_res_hq_offset[i];
	}
      }
    }

  };

  class Solver {

  protected:
    SolverParam &param;
    TimeProfile &profile;

  public:
    Solver(SolverParam &param, TimeProfile &profile) : param(param), profile(profile) { ; }
    virtual ~Solver() { ; }

    virtual void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in) = 0;

    // solver factory
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

  class CG : public Solver {

  private:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;

  public:
    CG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);
    virtual ~CG();

    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
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

      void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };


  class BiCGstab : public Solver {

  private:
    DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    const DiracMatrix &matPrecon;

    // pointers to fields to avoid multiple creation overhead
    cudaColorSpinorField *yp, *rp, *pp, *vp, *tmpp, *tp;
    bool init;

  public:
    BiCGstab(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
	     SolverParam &param, TimeProfile &profile);
    virtual ~BiCGstab();

    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };

  class GCR : public Solver {

  private:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    const DiracMatrix &matPrecon;

    Solver *K;
    SolverParam Kparam; // parameters for preconditioner solve

  public:
    GCR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
	SolverParam &param, TimeProfile &profile);
    virtual ~GCR();

    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };

  class MR : public Solver {

  private:
    const DiracMatrix &mat;
    cudaColorSpinorField *rp;
    cudaColorSpinorField *Arp;
    cudaColorSpinorField *tmpp;
    bool init;
    bool allocate_r;

  public:
    MR(DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
    virtual ~MR();

    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
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


      void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };

  // multigrid solver
  class alphaSA : public Solver {

  protected:
    const DiracMatrix &mat;

  public:
    alphaSA(DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
    virtual ~alphaSA() { ; }

    void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in);
  };

  class MultiShiftSolver {

  protected:
    SolverParam &param;
    TimeProfile &profile;

  public:
    MultiShiftSolver(SolverParam &param, TimeProfile &profile) : 
    param(param), profile(profile) { ; }
    virtual ~MultiShiftSolver() { ; }

    virtual void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in) = 0;
  };

  class MultiShiftCG : public MultiShiftSolver {

  protected:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;

  public:
    MultiShiftCG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);
    virtual ~MultiShiftCG();

    void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in);
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
    void operator()(cudaColorSpinorField &x, cudaColorSpinorField &b, cudaColorSpinorField **p,
		    cudaColorSpinorField **q, int N);
  };

} // namespace quda

#endif // _INVERT_QUDA_H
