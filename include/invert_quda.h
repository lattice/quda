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
     * Whether to use the L2 relative residual, Fermilab heavy-quark
     * residual, or both to determine convergence.  To require that both
     * stopping conditions are satisfied, use a bitwise OR as follows:
     *
     * p.residual_type = (QudaResidualType) (QUDA_L2_RELATIVE_RESIDUAL
     *                                     | QUDA_HEAVY_QUARK_RESIDUAL);
     */
    QudaResidualType residual_type;
    
    /**< Whether to use an initial guess in the solver or not */
    QudaUseInitGuess use_init_guess;       

    /**< Reliable update tolerance */
    double delta;           

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

    // Incremental EigCG solver parameters

    int nev;//number of eigenvectors produced by EigCG
    int m;//Dimension of the search space
    int deflation_grid;
    int rhs_idx;
    
    /**
       Constructor that matches the initial values to that of the
       QudaInvertParam instance
       @param param The QudaInvertParam instance from which the values are copied
     */
    SolverParam(QudaInvertParam &param) : inv_type(param.inv_type), 
      inv_type_precondition(param.inv_type_precondition), 
      residual_type(param.residual_type), use_init_guess(param.use_init_guess),
      delta(param.reliable_delta), pipeline(param.pipeline), tol(param.tol), tol_hq(param.tol_hq), 
      true_res(param.true_res), true_res_hq(param.true_res_hq),
      maxiter(param.maxiter), iter(param.iter), 
      precision(param.cuda_prec), precision_sloppy(param.cuda_prec_sloppy), 
      precision_precondition(param.cuda_prec_precondition), 
      preserve_source(param.preserve_source), num_offset(param.num_offset), 
      Nkrylov(param.gcrNkrylov), precondition_cycle(param.precondition_cycle), 
      tol_precondition(param.tol_precondition), maxiter_precondition(param.maxiter_precondition), 
      omega(param.omega), schwarz_type(param.schwarz_type), secs(param.secs), gflops(param.gflops),
      nev(param.nev), m(param.max_vect_size), deflation_grid(param.deflation_grid), rhs_idx(0) //! for IncEigCG
    { 
      for (int i=0; i<num_offset; i++) {
	offset[i] = param.offset[i];
	tol_offset[i] = param.tol_offset[i];
	tol_hq_offset[i] = param.tol_hq_offset[i];
      }

      if((param.inv_type == QUDA_INC_EIGCG_INVERTER) && m % 16){//current hack for the magma library
        m = (m / 16) * 16 + 16;
        warningQuda("\nSwitched eigenvector search dimension to %d\n", m);
      }
      if(param.rhs_idx != 0 && param.inv_type==QUDA_INC_EIGCG_INVERTER){
        rhs_idx = param.rhs_idx;
      }
    }
    ~SolverParam() { }

    /**
       Update the QudaInvertParam with the data from this
       @param param the QudaInvertParam to be updated
     */
    void updateInvertParam(QudaInvertParam &param) {
      param.true_res = true_res;
      param.true_res_hq = true_res_hq;
      param.iter += iter;
      param.gflops = (param.gflops*param.secs + gflops*secs) / (param.secs + secs);
      param.secs += secs;
      for (int i=0; i<num_offset; i++) {
	param.true_res_offset[i] = true_res_offset[i];
	param.true_res_hq_offset[i] = true_res_hq_offset[i];
      }
      //for incremental eigCG:
      param.rhs_idx = rhs_idx;
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

//experimantal EigCG solver
  class DeflatedSolver {

  protected:
    SolverParam &param;
    TimeProfile &profile;

  public:
    DeflatedSolver(SolverParam &param, TimeProfile &profile) : 
    param(param), profile(profile) { ; }
    virtual ~DeflatedSolver() { ; }

    virtual void operator()(cudaColorSpinorField *out, cudaColorSpinorField *eigvset, cudaColorSpinorField *in) = 0;

    // solver factory
    static DeflatedSolver* create(SolverParam &param, DiracMatrix &mat, DiracMatrix &matSloppy,
			  ColorSpinorParam *eigenvParam, TimeProfile &profile);

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

/*
Concerning used precisions:
external Ritz vectors must currently have full solver precision. This is also precision for the projection
matrix used. 
Internal eigCG deflation space may have, in principle, an arbitrary precision: it's unrelated to the solver precisions; 
but we pinned it currently to the single precision because half precision is currently not supported. This can be easely 
included into the framework, though.
*/

//for future : replace by IncEigCGargs.
  class ProjectionMatrix {

  private:
    //
    const DiracMatrix &mat;

    //device projection matrix:
    void *dproj; //VH A V

    //host projection matrix(used for small dimensions):
    void *hproj; //VH A V
 
    QudaPrecision prec;     //precision must match DiracMatrix precision and Ritz vectors precision
    int ld;                 //projection matrix leading dimension
    int tot_dim;            //full dimension (nev*deflation_grid)
    int curr_dim;           //current dimension (must match rhs_idx: dim = (rhs_idx < deflation_grid) ? nev * rhs_idx) 
    int prev_dim;
    int bytes;

  public:
    ProjectionMatrix(DiracMatrix &mat, SolverParam &param);
    virtual ~ProjectionMatrix();

    //Solve dH y = u^{dagger}r: (y has precision of dH)
    //For small dim: use CPU
    //For big dim: use GPU (e.g., dim > 128)
    //output: complex vector y
    void operator()(void *out, cudaColorSpinorField *r, cudaColorSpinorField *u);

    //Compute H=U^{dag}MU (also computes block of H-matrix):
    //void computeProj(cudaColorSpinorField *u);

    //extend projection matrix:
    //compute Q' = DiracM Q, (here U = [V, Q] - total Ritz set)
    //construct H-matrix components with Q'^{dag} Q', V^{dag} Q' and Q'^{dag} V
    //extend H-matrix with the components
    void ConstructProj(cudaColorSpinorField *u); 

    //reset current dimention:
    void ResetProjMatDim(const int n);    

    //copy from the host:
    void LoadProj(void *out);

    //copy to the host:
    void SaveProj(void *out);
  };

  class IncEigCG : public DeflatedSolver {

  private:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;

    Solver *initCG;//initCG solver for deflated inversions
    SolverParam initCGparam; // parameters for initCG solve

    bool eigcg_alloc;

    cudaColorSpinorField *Vm;  //deflation vectors  (spinor matrix of size eigen_vector_length x m)

    //move this to separate DeflatedSolverArgs class...
    //host Lanczos matrice, and its eigenvalue/vector arrays:
    std::complex<float> *hTm;//VH A V
    std::complex<float> *hTvecm;//eigenvectors of both T[m,  m  ] and T[m-1, m-1] (re-used)
    float  *hTvalm;   //eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)

    //device Lanczos matrix, and its eigenvalue/vector arrays:
    void *dTm;     //VH A V
    void *dTvecm0; //eigenvectors of T[m,  m  ]
    void *dTvecm1; //eigenvectors of T[m-1,m-1]

  public:
    IncEigCG(DiracMatrix &mat, DiracMatrix &matSloppy, ColorSpinorParam *eigvParam, SolverParam &param, TimeProfile &profile);
    virtual ~IncEigCG();

    void EigCG(cudaColorSpinorField &out, cudaColorSpinorField &nev_eigvecs, cudaColorSpinorField &in);

    void DeflateInitGuess(cudaColorSpinorField &in, const cudaColorSpinorField &u, const ProjectionMatrix *pM);

    void OrthRitz(cudaColorSpinorField &u);

    void operator()(cudaColorSpinorField *out, cudaColorSpinorField *in, cudaColorSpinorField *u, ProjectionMatrix *pM);
  };


} // namespace quda

#endif // _INVERT_QUDA_H
