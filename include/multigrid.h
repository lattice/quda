#ifndef _MG_QUDA_H
#define _MG_QUDA_H

#include <invert_quda.h>
#include <transfer.h>
#include <vector>
#include <complex_quda.h>

// at the moment double-precision multigrid is only enabled when debugging
#ifdef HOST_DEBUG
#define GPU_MULTIGRID_DOUBLE
#endif

namespace quda {

  // forward declarations
  class MG;
  class DiracCoarse;

  /**
     This struct contains all the metadata required to define the
     multigrid solver.  For each level of multigrid we will have an
     instance of MGParam describing all the meta data appropriate for
     given level.
   */
  struct MGParam : SolverParam {

    /** This points to the parameter struct that is passed into QUDA.
	We use this to set per-level parameters */
    QudaMultigridParam  &mg_global;

    /** What is the level of this instance */
    int level;

    /** Number of levels in the solver */
    int Nlevel;

    /** Geometric block size */
    int geoBlockSize[QUDA_MAX_DIM];

    /** Spin block size */
    int spinBlockSize;

    /** Number of vectors used to define coarse space */
    int Nvec;

    /** This is the next lower level */
    MG *coarse;

    /** This is the immediate finer level */
    MG *fine;

    /** The null space vectors */
    std::vector<ColorSpinorField*> &B;

    /** Number of pre-smoothing applications to perform */
    int nu_pre;

    /** Number of pre-smoothing applications to perform */
    int nu_post;

    /** Tolerance to use for the solver / smoother (if applicable) */
    double smoother_tol;

    /** Multigrid cycle type */
    QudaMultigridCycleType cycle_type;

    /** Whether to use global or local (node) reductions */
    QudaBoolean global_reduction;

    /** The Dirac operator to use for residual computation */
    DiracMatrix *matResidual;

    /** The Dirac operator to use for smoothing */
    DiracMatrix *matSmooth;

    /** The sloppy Dirac operator to use for smoothing */
    DiracMatrix *matSmoothSloppy;

    /** What type of smoother to use */
    QudaInverterType smoother;

    /** The type of residual to send to the next coarse grid, and thus the
	type of solution to receive back from this coarse grid */
    QudaSolutionType coarse_grid_solution_type;

    /** The type of smoother solve to do on each grid (e/o preconditioning or not)*/
    QudaSolveType smoother_solve_type;

    /** Where to compute this level of multigrid */
    QudaFieldLocation location;

    /** Filename for where to load/store the null space */
    char filename[100];

    /**
       This is top level instantiation done when we start creating the multigrid operator.
     */
    MGParam(QudaMultigridParam &param,
	    void *mg_instance,
	    std::vector<ColorSpinorField*> &B,
	    DiracMatrix *matResidual,
	    DiracMatrix *matSmooth,
	    DiracMatrix *matSmoothSloppy,
	    int level=0) :
      SolverParam(*(param.invert_param)), 
      mg_global(param), 
      level(level),
      Nlevel(param.n_level),
      spinBlockSize(param.spin_block_size[level]),
      Nvec(param.n_vec[level]),
      B(B), 
      nu_pre(param.nu_pre[level]),
      nu_post(param.nu_post[level]),
      smoother_tol(param.smoother_tol[level]),
      cycle_type(param.cycle_type[level]),
      global_reduction(param.global_reduction[level]),
      matResidual(matResidual),
      matSmooth(matSmooth),
      matSmoothSloppy(matSmoothSloppy),
      smoother(param.smoother[level]),
      coarse_grid_solution_type(param.coarse_grid_solution_type[level]),
      smoother_solve_type(param.smoother_solve_type[level]),
      location(param.location[level])
      { 
	// set the block size
	for (int i=0; i<QUDA_MAX_DIM; i++) geoBlockSize[i] = param.geo_block_size[level][i];

	// set the smoother relaxation factor
	omega = param.omega[level];
      }

    MGParam(const MGParam &param, 
	    void *mg_instance,
	    std::vector<ColorSpinorField*> &B,
	    DiracMatrix *matResidual,
	    DiracMatrix *matSmooth,
	    DiracMatrix *matSmoothSloppy,
	    int level=0) :
      SolverParam(param),
      mg_global(param.mg_global),
      level(level),
      Nlevel(param.Nlevel),
      spinBlockSize(param.mg_global.spin_block_size[level]),
      Nvec(param.mg_global.n_vec[level]),
      coarse(param.coarse),
      fine(param.fine),
      B(B),
      nu_pre(param.mg_global.nu_pre[level]),
      nu_post(param.mg_global.nu_post[level]),
      smoother_tol(param.mg_global.smoother_tol[level]),
      cycle_type(param.mg_global.cycle_type[level]),
      global_reduction(param.mg_global.global_reduction[level]),
      matResidual(matResidual),
      matSmooth(matSmooth),
      matSmoothSloppy(matSmoothSloppy),
      smoother(param.mg_global.smoother[level]),
      coarse_grid_solution_type(param.mg_global.coarse_grid_solution_type[level]),
      smoother_solve_type(param.mg_global.smoother_solve_type[level]),
      location(param.mg_global.location[level])
      {
	// set the block size
	for (int i=0; i<QUDA_MAX_DIM; i++) geoBlockSize[i] = param.mg_global.geo_block_size[level][i];

	// set the smoother relaxation factor
	omega = param.mg_global.omega[level];
      }

  };

  /**
     Adaptive Multigrid solver
   */
  class MG : public Solver {

  private:
    /** Local copy of the multigrid metadata */
    MGParam &param;

    /** This is the transfer operator that defines the prolongation and restriction operators */
    Transfer *transfer;

    /** This is the smoother used */
    Solver *presmoother, *postsmoother;

    /** TimeProfile for all levels (refers to profile from parent solver) */
    TimeProfile &profile_global;

    /** TimeProfile for this level */
    TimeProfile profile;

    /** Prefix label used for printf at this level */
    char prefix[128];

    /** Prefix label used for printf on next coarse level */
    char coarse_prefix[128];

    /** This is the next lower level */
    MG *coarse;

    /** This is the next coarser level */
    MG *fine;

    /** The coarse grid solver - this either points at "coarse" or a solver preconditioned by "coarse" */
    Solver *coarse_solver;

    /** Storage for the parameter struct for the coarse grid */
    MGParam *param_coarse;

    /** Storage for the parameter struct for the pre-smoother */
    SolverParam *param_presmooth;

    /** Storage for the parameter struct for the post-smoother */
    SolverParam *param_postsmooth;

    /** Storage for the parameter struct for the coarse solver */
    SolverParam *param_coarse_solver;

    /** The fine-grid representation of the null space vectors */
    std::vector<ColorSpinorField*> *B;

    /** The coarse-grid representation of the null space vectors */
    std::vector<ColorSpinorField*> *B_coarse;

    /** Residual vector */
    ColorSpinorField *r;

    /** Projected source vector for preconditioned syste, else just points to source */
    ColorSpinorField *b_tilde;

    /** Coarse residual vector */
    ColorSpinorField *r_coarse;

    /** Coarse solution vector */
    ColorSpinorField *x_coarse;

    /** Coarse temporary vector */
    ColorSpinorField *tmp_coarse;

    /** The coarse operator used for computing inter-grid residuals */
    Dirac *diracCoarseResidual;

    /** The coarse operator used for doing smoothing */
    Dirac *diracCoarseSmoother;

    /** The coarse operator used for doing sloppy smoothing */
    Dirac *diracCoarseSmootherSloppy;

    /** Wrapper for the residual coarse grid operator */
    DiracMatrix *matCoarseResidual;

    /** Wrapper for the smoothing coarse grid operator */
    DiracMatrix *matCoarseSmoother;

    /** Wrapper for the sloppy smoothing coarse grid operator */
    DiracMatrix *matCoarseSmootherSloppy;

  public:
    /** 
      Constructor for MG class
      @param param MGParam struct that defines all meta data
      @param profile Timeprofile instance used to profile
    */
    MG(MGParam &param, void *mg_instance, TimeProfile &profile);

    /**
       Destructor for MG class. Frees any existing coarse grid MG
       instance
     */
    virtual ~MG();

    /**
       @brief Create the smoothers
    */
    void createSmoother();

    /**
       @brief Free the smoothers
    */
    void destroySmoother();

    /**
       This method is a placeholder for reseting the solver, e.g.,
       when a parameter has changed such as the mass.  For now, all it
       does is call Transfer::setSiteSubset to resize the on-GPU
       null-space components to single-parity if we're doing a
       single-parity solve (memory saving technique).
     */
    void reset();

    /**
       This method verifies the correctness of the MG method.  It checks:
       1. Null-space vectors are exactly preserved: v_k = P R v_k
       2. Any coarse vector is exactly preserved on the fine grid: eta_c = R P eta_c
       3. The emulated coarse Dirac operator matches the native one: D_c = R D P
     */
    void verify();

    /**
       This applies the V-cycle to the residual vector returning the residual vector
       @param out The solution vector
       @param in The residual vector (or equivalently the right hand side vector)
     */
    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @brief Load the null space vectors in from file
       @param B Loaded null-space vectors (pre-allocated)
    */
    void loadVectors(std::vector<ColorSpinorField*> &B);

    /**
       @brief Save the null space vectors in from file
       @param B Save null-space vectors from here
    */
    void saveVectors(std::vector<ColorSpinorField*> &B);

    /**
       @brief Generate the null-space vectors
       @param B Generated null-space vectors
     */
    void generateNullVectors(std::vector<ColorSpinorField*> B);

    /**
       @brief Return the total flops done on this and all coarser levels.
     */
    double flops() const;

  };

  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity = QUDA_INVALID_PARITY,
		   bool dslash=true, bool clover=true, bool dagger=false);

  /**
     @brief Coarse operator construction from a fine-grid operator (Wilson / Clover)
     @param Y[out] Coarse link field
     @param X[out] Coarse clover field
     @param Xinv[out] Coarse clover inverse field
     @param Yhat[out] Preconditioned coarse link field
     @param T[in] Transfer operator that defines the coarse space
     @param gauge[in] Gauge field from fine grid
     @param clover[in] Clover field on fine grid (optional)
     @param kappa[in] Kappa parameter
     @param mu[in] Mu parameter (set to non-zero for twisted-mass/twisted-clover)
     @param mu_factor[in] Multiplicative factor for the mu parameter
     @param matpc[in] The type of even-odd preconditioned fine-grid
     operator we are constructing the coarse grid operator from.  If
     matpc==QUDA_MATPC_INVALID then we assume the operator is not
     even-odd preconditioned and we coarsen the full operator.
   */
  void CoarseOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, const Transfer &T,
		const cudaGaugeField &gauge, const cudaCloverField *clover,
		double kappa, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc);

  /**
     @brief Coarse operator construction from an intermediate-grid operator (Coarse)
     @param Y[out] Coarse link field
     @param X[out] Coarse clover field
     @param Xinv[out] Coarse clover inverse field
     @param Y[out] Preconditioned coarse link field
     @param T[in] Transfer operator that defines the new coarse space
     @param gauge[in] Link field from fine grid
     @param clover[in] Clover field on fine grid
     @param cloverInv[in] Clover inverse field on fine grid
     @param kappa[in] Kappa parameter
     @param mu[in] Mu parameter (set to non-zero for twisted-mass/twisted-clover)
     @param mu_factor[in] Multiplicative factor for the mu parameter
     @param matpc[in] The type of even-odd preconditioned fine-grid
     operator we are constructing the coarse grid operator from.  If
     matpc==QUDA_MATPC_INVALID then we assume the operator is not
     even-odd preconditioned and we coarsen the full operator.
   */
  void CoarseCoarseOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, const Transfer &T,
		      const GaugeField &gauge, const GaugeField &clover, const GaugeField &cloverInv,
		      double kappa, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc);

  /**
     This is an object that captures an entire MG preconditioner
     state.  A bit of a hack at the moment, this is used to allow us
     to store and reuse the mg solver between solves.  This is use by
     the newMultigridQuda and destroyMultigridQuda interface functions.
   */
  struct multigrid_solver {
    Dirac *d;
    Dirac *dSmooth;
    Dirac *dSmoothSloppy;

    DiracM *m;
    DiracM *mSmooth;
    DiracM *mSmoothSloppy;

    std::vector<ColorSpinorField*> B;

    MGParam *mgParam;

    MG *mg;
    TimeProfile &profile;

    multigrid_solver(QudaMultigridParam &mg_param, void *mg_instance, TimeProfile &profile);

    virtual ~multigrid_solver()
    {
      profile.TPSTART(QUDA_PROFILE_FREE);
      if (mg) delete mg;

      if (mgParam) delete mgParam;

      for (unsigned int i=0; i<B.size(); i++) delete B[i];

      if (m) delete m;
      if (mSmooth) delete mSmooth;
      if (mSmoothSloppy) delete mSmoothSloppy;

      if (d) delete d;
      if (dSmooth) delete dSmooth;
      if (dSmoothSloppy && dSmoothSloppy != dSmooth) delete dSmoothSloppy;
      profile.TPSTOP(QUDA_PROFILE_FREE);
    }
  };

} // namespace quda

#endif // _MG_QUDA_H
