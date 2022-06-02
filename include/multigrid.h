#pragma once

#include <invert_quda.h>
#include <transfer.h>
#include <vector>
#include <complex_quda.h>
#include <memory>
#include <instantiate.h>

// at the moment double-precision multigrid is only enabled when debugging
#ifdef HOST_DEBUG
//#define GPU_MULTIGRID_DOUBLE
#endif

namespace quda {

  /**
     @brief Helper function for returning if multigrid is enabled
  */
#ifdef GPU_MULTIGRID
  constexpr bool is_enabled_multigrid() { return true; };
#else
  constexpr bool is_enabled_multigrid() { return false; };
#endif

  /**
     @brief Helper function for returning if double-precision
     multigrid is enabled
  */
#ifdef GPU_MULTIGRID_DOUBLE
  constexpr bool is_enabled_multigrid_double() { return true; }
#else
  constexpr bool is_enabled_multigrid_double() { return false; }
#endif

  /**
     @brief The instantiatePrecision function is used to instantiate
     the precision
     @param[in] field LatticeField we wish to instantiate
     @param[in,out] args Any additional arguments required for the
     computation at hand
  */
  template <template <typename> class Apply, typename F, typename... Args>
  constexpr void instantiatePrecisionMG(F &field, Args &&... args)
  {
    if (field.Precision() == QUDA_DOUBLE_PRECISION) {
      if constexpr (is_enabled_multigrid_double())
        Apply<double>(field, args...);
      else
        errorQuda("Multigrid not supported in double precision");
    } else if (field.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled<QUDA_SINGLE_PRECISION>())
        Apply<float>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
    } else if (field.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled<QUDA_HALF_PRECISION>())
        Apply<short>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
    } else if (field.Precision() == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled<QUDA_QUARTER_PRECISION>())
        Apply<int8_t>(field, args...);
      else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
    } else {
      errorQuda("Unsupported precision %d\n", field.Precision());
    }
  }

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

    /** Number of times to apply Gram-Schmidt within a block */
    int NblockOrtho;

    /** Whether we are doing the two-pass Block Orthogonalize */
    bool blockOrthoTwoPass;

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

    /** Where to compute this level of the multigrid setup*/
    QudaFieldLocation setup_location;

    /** Filename for where to load/store the null space */
    char filename[100];

    /** Whether or not this is a staggered solve or not */
    QudaTransferType transfer_type;

    /** Whether to use tensor cores (if available) */
    bool use_mma;

    /**
       This is top level instantiation done when we start creating the multigrid operator.
     */
    MGParam(QudaMultigridParam &param, std::vector<ColorSpinorField *> &B, DiracMatrix *matResidual,
            DiracMatrix *matSmooth, DiracMatrix *matSmoothSloppy, int level = 0) :
      SolverParam(*(param.invert_param)),
      mg_global(param),
      level(level),
      Nlevel(param.n_level),
      spinBlockSize(param.spin_block_size[level]),
      Nvec(param.n_vec[level]),
      NblockOrtho(param.n_block_ortho[level]),
      blockOrthoTwoPass(param.block_ortho_two_pass[level]),
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
      location(param.location[level]),
      setup_location(param.setup_location[level]),
      transfer_type(param.transfer_type[level]),
      use_mma(param.use_mma == QUDA_BOOLEAN_TRUE)
    {
      // set the block size
      for (int i = 0; i < QUDA_MAX_DIM; i++) geoBlockSize[i] = param.geo_block_size[level][i];

      // set the smoother relaxation factor
      omega = param.omega[level];
    }

    MGParam(const MGParam &param, std::vector<ColorSpinorField *> &B, DiracMatrix *matResidual, DiracMatrix *matSmooth,
            DiracMatrix *matSmoothSloppy, int level = 0) :
      SolverParam(param),
      mg_global(param.mg_global),
      level(level),
      Nlevel(param.Nlevel),
      spinBlockSize(param.mg_global.spin_block_size[level]),
      Nvec(param.mg_global.n_vec[level]),
      NblockOrtho(param.mg_global.n_block_ortho[level]),
      blockOrthoTwoPass(param.mg_global.block_ortho_two_pass[level]),
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
      location(param.mg_global.location[level]),
      setup_location(param.mg_global.setup_location[level]),
      transfer_type(param.mg_global.transfer_type[level]),
      use_mma(param.use_mma)
    {
      // set the block size
      for (int i = 0; i < QUDA_MAX_DIM; i++) geoBlockSize[i] = param.mg_global.geo_block_size[level][i];

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

    /** This tell to reset() if transfer needs to be rebuilt */
    bool resetTransfer;

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

    /** The coarse-grid representation of the null space vectors */
    std::vector<ColorSpinorField*> B_coarse;

    /** Residual vector */
    ColorSpinorField *r;

    /** Projected source vector for preconditioned system, else just points to source */
    ColorSpinorField *b_tilde;

    /** Coarse residual vector */
    ColorSpinorField *r_coarse;

    /** Coarse solution vector */
    ColorSpinorField *x_coarse;

    /** Coarse temporary vector */
    ColorSpinorField *tmp_coarse;

    /** Coarse temporary vector */
    ColorSpinorField *tmp2_coarse;

    /** Sloppy coarse temporary vector */
    ColorSpinorField *tmp_coarse_sloppy;

    /** Sloppy coarse temporary vector */
    ColorSpinorField *tmp2_coarse_sloppy;

    /** Kahler-Dirac Xinv */
    std::shared_ptr<GaugeField> xInvKD;

    /** Kahler-Dirac Xinv, sloppy field */
    std::shared_ptr<GaugeField> xInvKD_sloppy;

    /** The fine operator used for computing inter-grid residuals */
    const Dirac *diracResidual;

    /** The fine operator used for doing smoothing */
    const Dirac *diracSmoother;

    /** The fine operator used for doing sloppy smoothing */
    const Dirac *diracSmootherSloppy;

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

    /** Parallel hyper-cubic random number generator for generating null-space vectors */
    RNG *rng;

    /**
       @brief Helper function called on entry to each MG function
       @param[in] level The level we working on
    */
    void pushLevel(int level) const;

    /**
       @brief Helper function called on exit to each MG member function
    */
    void popLevel() const;

  public:
    /**
       Constructor for MG class
       @param param MGParam struct that defines all meta data
       @param profile Timeprofile instance used to profile
    */
    MG(MGParam &param, TimeProfile &profile);

    /**
       Destructor for MG class. Frees any existing coarse grid MG
       instance
     */
    virtual ~MG();

    /**
       @return MG can solve non-Hermitian systems
     */
    bool hermitian() { return false; };

    /**
       @brief This method resets the solver, e.g., when a parameter has changed such as the mass.
       @param Whether we are refreshing the null-space components or just updating the operators
     */
    void reset(bool refresh=false);

    /**
       @brief This method only resets the KD operators with the updated fine links and rebuilds
              the KD inverse
     */
    void resetStaggeredKD(cudaGaugeField *gauge_in, cudaGaugeField *fat_gauge_in, cudaGaugeField *long_gauge_in,
                          cudaGaugeField *gauge_sloppy_in, cudaGaugeField *fat_gauge_sloppy_in,
                          cudaGaugeField *long_gauge_sloppy_in, double mass);

    /**
       @brief Dump the null-space vectors to disk.  Will recurse dumping all levels.
    */
    void dumpNullVectors() const;

    /**
       @brief Create the smoothers
    */
    void createSmoother();

    /**
       @brief Destroy the smoothers
    */
    void destroySmoother();

    /**
       @brief Create the coarse dirac operator
    */
    void createCoarseDirac();

    /**
       @brief Create the optimized KD operator
    */
    void createOptimizedKdDirac();

    /**
       @brief Create the solver wrapper
    */
    void createCoarseSolver();

    /**
       @brief Destroy the solver wrapper
    */
    void destroyCoarseSolver();

    /**
       @brief Verify the correctness of the MG method, optionally recursively
       starting from the top down.
       @details This method verifies the correctness of the MG method.  It checks:
       1. Null-space vectors are exactly preserved: v_k = P R v_k
       2. Any coarse vector is exactly preserved on the fine grid: eta_c = R P eta_c
       3. The emulated coarse Dirac operator matches the native one: D_c = R D P
       4. The preconditioned operator was correctly formulated: \hat{D}_c - X^{-1} D_c
       5. The normal operator is indeed normal: im(<x|D^\dag D|x>) < epsilon
       @param recursively[in] Whether or not to recursively verify coarser levels, default false
     */
    void verify(bool recursively = false);

    /**
       This applies the V-cycle to the residual vector returning the residual vector
       @param out The solution vector
       @param in The residual vector (or equivalently the right hand side vector)
     */
    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @brief Load the null space vectors in from file
    */
    void loadVectors();

    /**
       @brief Save the null space vectors in from file
       @param B Save null-space vectors from here
    */
    void saveVectors(const std::vector<ColorSpinorField *> &B) const;

    /**
       @brief Initialize a set of vectors to random noise
       @param B set of vectors
    */
    void initializeRandomVectors(const std::vector<ColorSpinorField*> &B) const;

    /**
       @brief Wrapping function that manages logic for constructing near-null vectors
       @param refresh whether or not we're only refreshing near null vectors
    */
    void constructNearNulls(bool refresh = false);

    /**
       @brief Generate the null-space vectors via inverse iterations
       @param B Generated null-space vectors
       @param iterations if non-zero, override the max number of iterations
    */
    void generateInverseIterations(std::vector<ColorSpinorField*> &B, int iterations = 0);

    /**
       @brief Generate the null-space vectors via a Chebyshev filter; this approach is
              based on arXiv:2103.05034, P. Boyle and A. Yamaguchi.
       @param B Generated null-space vectors
    */
    void generateChebyshevFilter(std::vector<ColorSpinorField*> &B);

    /**
       @brief Generate lowest eigenvectors
       @param B Generated null-space vectors
       @param post_restrict whether or not we're generating extra eigenvectors after restricting
              some fine eigenvectors; if true we override the requested n_conv in the multigrid
              eigenvalue struct
    */
    void generateEigenvectors(std::vector<ColorSpinorField*> &B, bool post_restrict = false);

    /**
       @brief Generate near-null vectors via restricting finer near-nulls, generating extras if need be
       @param refresh Whether or not we're only refreshing extra near nulls
    */
    void generateRestrictedVectors(bool refresh = false);

    /**
       @brief Build free-field null-space vectors
       @param B Generated null-space vectors
    */
    void generateFreeVectors(std::vector<ColorSpinorField*> &B);

    /**
      @brief Orthonormalize a vector of ColorSpinorField, erroring out if
             the fields are not sufficiently linearly independent
      @param vecs vector of ColorSpinorFields to normalize
      @param count number of near-null vectors to orthonormalize, default all
    */
    void orthonormalize_vectors(const std::vector<ColorSpinorField*>& vecs, int count = -1) const;

    /**
       @brief Return the total flops done on this and all coarser levels.
     */
    double flops() const;

    /**
      @brief Return if we're on a fine grid right now
    */
    bool is_fine_grid() const
    {

      // Check if we're on a KD fine grid
      bool kd_nearnull_gen = ((param.level == 1)
                              && (param.mg_global.transfer_type[0] == QUDA_TRANSFER_OPTIMIZED_KD
                                  || param.mg_global.transfer_type[0] == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG));

      return (param.level == 0 || kd_nearnull_gen);
    }

    /**
      @brief Return if we're on the coarsest grid right now
    */
    bool is_coarsest_grid() const {
      return (param.level == (param.Nlevel - 1));
    }

  };

  /**
     @brief Apply the coarse dslash stencil.  This single driver
     accounts for all variations with and without the clover field,
     with and without dslash, and both single and full parity fields
     @param[out] out The result vector
     @param[in] inA The first input vector
     @param[in] inB The second input vector
     @param[in] Y Coarse link field
     @param[in] X Coarse clover field
     @param[in] kappa Scaling parameter
     @param[in] parity Parity of the field (if single parity)
     @param[in] dslash Are we applying dslash?
     @param[in] clover Are we applying clover?
     @param[in] dagger Apply dagger operator?
     @param[in] commDim Which dimensions are partitioned?
     @param[in] halo_precision What precision to use for the halos (if QUDA_INVALID_PRECISION, use field precision)
   */
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity = QUDA_INVALID_PARITY,
		   bool dslash=true, bool clover=true, bool dagger=false, const int *commDim=0,
                   QudaPrecision halo_precision=QUDA_INVALID_PRECISION);

  /**
     @brief Coarse operator construction from a fine-grid operator (Wilson / Clover)
     @param Y[out] Coarse link field
     @param X[out] Coarse clover field
     @param T[in] Transfer operator that defines the coarse space
     @param gauge[in] Gauge field from fine grid
     @param clover[in] Clover field on fine grid (optional)
     @param kappa[in] Kappa parameter
     @param mass[in] Mass parameter
     @param mu[in] Mu parameter (set to non-zero for twisted-mass/twisted-clover)
     @param mu_factor[in] Multiplicative factor for the mu parameter
     @param matpc[in] The type of even-odd preconditioned fine-grid
     operator we are constructing the coarse grid operator from.  If
     matpc==QUDA_MATPC_INVALID then we assume the operator is not
     even-odd preconditioned and we coarsen the full operator.
   */
  void CoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const cudaGaugeField &gauge, const CloverField *clover,
                double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc);

  /**
     @brief Coarse operator construction from a fine-grid operator (Staggered)
     @param Y[out] Coarse link field
     @param X[out] Coarse clover field
     @param T[in] Transfer operator that defines the coarse space
     @param gauge[in] Gauge field from fine grid
     @param longGauge[in] Long link field in case of HISQ operator
     @param XinvKD[in] Inverse Kahler-Dirac block
     @param mass[in] Mass parameter
     @param allow_truncation[in] Whether or not we can drop the long links for small aggregation dimensions
     @param matpc[in] The type of even-odd preconditioned fine-grid
     operator we are constructing the coarse grid operator from.
     For staggered, should always be QUDA_MATPC_INVALID.
   */
  void StaggeredCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const cudaGaugeField &gauge,
                         const cudaGaugeField &longGauge, const GaugeField &XinvKD, double mass, bool allow_truncation,
                         QudaDiracType dirac, QudaMatPCType matpc);

  /**
     @brief Coarse operator construction from an intermediate-grid operator (Coarse)
     @param Y[out] Coarse link field
     @param X[out] Coarse clover field
     @param T[in] Transfer operator that defines the new coarse space
     @param gauge[in] Link field from fine grid
     @param clover[in] Clover field on fine grid
     @param cloverInv[in] Clover inverse field on fine grid
     @param kappa[in] Kappa parameter
     @param mass[in] Mass parameter
     @param mu[in] Mu parameter (set to non-zero for twisted-mass/twisted-clover)
     @param mu_factor[in] Multiplicative factor for the mu parameter
     @param matpc[in] The type of even-odd preconditioned fine-grid
     operator we are constructing the coarse grid operator from.  If
     matpc==QUDA_MATPC_INVALID then we assume the operator is not
     even-odd preconditioned and we coarsen the full operator.
     @param need_bidirectional[in] Whether or not we need to force a bi-directional
     build, even if the given level isn't preconditioned---if any previous level is
     preconditioned, we've violated that symmetry.
     @param use_mma[in] Whether or not use MMA (tensor core) to do the calculation, default to false
   */
  void CoarseCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &gauge, const GaugeField &clover,
                      const GaugeField &cloverInv, double kappa, double mass, double mu, double mu_factor,
                      QudaDiracType dirac, QudaMatPCType matpc, bool need_bidirectional, bool use_mma = false);

  /**
     @brief Calculate preconditioned coarse links and coarse clover inverse field
     @param Yhat[out] Preconditioned coarse link field
     @param Xinv[out] Coarse clover inverse field
     @param Y[in] Coarse link field
     @param X[in] Coarse clover field
     @param use_mma[in] Whether or not use MMA (tensor core) to do the calculation, default to false
   */
  void calculateYhat(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X, bool use_mma = false);

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

    multigrid_solver(QudaMultigridParam &mg_param, TimeProfile &profile);

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
