#ifndef _MG_QUDA_H
#define _MG_QUDA_H

#include <invert_quda.h>
#include <transfer.h>
#include <vector>
#include <complex_quda.h>

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

    /**
       This is top level instantiation done when we start creating the multigrid operator.
     */
    MGParam(const QudaMultigridParam &param, 
	    std::vector<ColorSpinorField*> &B,
	    DiracMatrix &matResidual, 
	    DiracMatrix &matSmooth,
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
      matResidual(matResidual),
      matSmooth(matSmooth),
      smoother(param.smoother[level]),
      location(param.location[level])
      { 
	// set the block size
	for (int i=0; i<QUDA_MAX_DIM; i++) geoBlockSize[i] = param.geo_block_size[level][i];

	// set the smoother relaxation factor
	omega = param.omega[level];
      }

    MGParam(const MGParam &param, 
	    std::vector<ColorSpinorField*> &B,
	    DiracMatrix &matResidual, 
	    DiracMatrix &matSmooth,
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
      matResidual(matResidual),
      matSmooth(matSmooth),
      smoother(param.mg_global.smoother[level]),
      location(param.mg_global.location[level])
      {
	// set the block size
	for (int i=0; i<QUDA_MAX_DIM; i++) geoBlockSize[i] = param.mg_global.geo_block_size[level][i];
      }

    /** This points to the parameter struct that is passed into QUDA.
	We use this to set per-level parameters */
    const QudaMultigridParam  &mg_global;

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

    /** The Dirac operator to use for residual computation */
    DiracMatrix &matResidual;

    /** The Dirac operator to use for smoothing */
    DiracMatrix &matSmooth;

    /** What type of smoother to use */
    QudaInverterType smoother;

    /** Where to compute this level of multigrid */
    QudaFieldLocation location;

    /** Filename for where to load/store the null space */
    char filename[100];
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

    /** This is the next lower level */
    MG *coarse;

    /** This is the next coarser level */
    MG *fine;

    /** Storage for the parameter struct for the coarse grid */
    MGParam *param_coarse;

    /** Storage for the parameter struct for the pre-smoother */
    SolverParam *param_presmooth;

    /** Storage for the parameter struct for the post-smoother */
    SolverParam *param_postsmooth;

    /** The fine-grid representation of the null space vectors */
    std::vector<ColorSpinorField*> *B;

    /** The coarse-grid representation of the null space vectors */
    std::vector<ColorSpinorField*> *B_coarse;

    /** Residual vector */
    ColorSpinorField *r;

    /** Coarse residual vector */
    ColorSpinorField *r_coarse;

    /** Coarse solution vector */
    ColorSpinorField *x_coarse;

    /** The coarse grid operator */
    DiracCoarse *diracCoarse;

    /** Wrapper for the coarse grid operator */
    DiracMatrix *matCoarse;

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
       Load the null space vectors in from file
       @param B Loaded null-space vectors (pre-allocated)
    */
    void loadVectors(std::vector<ColorSpinorField*> &B);

    /**
       Save the null space vectors in from file
       @param B Save null-space vectors from here
    */
    void saveVectors(std::vector<ColorSpinorField*> &B);

    /**
       Generate the null-space vectors
       @param B Generated null-space vectors
     */
    void generateNullVectors(std::vector<ColorSpinorField*> B);

  };

  void CoarseOp(const Transfer &T, GaugeField &Y, GaugeField &X, QudaPrecision precision, const cudaGaugeField &gauge);

  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity = QUDA_INVALID_PARITY,
		   bool dslash=true, bool clover=true);

  void CoarseCoarseOp(const Transfer &T, GaugeField &Y, GaugeField &x, const cpuGaugeField &gauge, 
		      const cpuGaugeField &clover, double kappa);

  /**
     This is an object that captures an entire MG preconditioner
     state.  A bit of a hack at the moment, this is used to allow us
     to store and reuse the mg solver between solves.  This is use by
     the newMultigridQuda and destroyMultigridQuda interface functions.
   */
  struct multigrid_solver {
    Dirac *d;
    Dirac *dSloppy;
    Dirac *dPre;

    DiracM *m;
    DiracM *mSloppy;
    DiracM *mPre;

    std::vector<ColorSpinorField*> B;

    MGParam *mgParam;

    MG *mg;
    TimeProfile &profile;

    multigrid_solver(QudaMultigridParam &mg_param, TimeProfile &profile);

    virtual ~multigrid_solver() {
      profile.TPSTART(QUDA_PROFILE_FREE);
      delete mg;

      delete mgParam;

      for (unsigned int i=0; i<B.size(); i++) delete B[i];

      delete m;
      delete mSloppy;
      delete mPre;

      delete d;
      delete dSloppy;
      delete dPre;
      profile.TPSTOP(QUDA_PROFILE_FREE);
    }
  };

} // namespace quda

#endif // _MG_QUDA_H
