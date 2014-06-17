#ifndef _MG_QUDA_H
#define _MG_QUDA_H

#include <invert_quda.h>
#include <transfer.h>
#include <vector>
#include <complex_quda.h>

#define QUDA_MAX_MG_LEVEL 3

extern char vecfile[];
extern int nvec; 
  
namespace quda {

  // FIXME - these definitions are strictly temporary
  void loadVectors(std::vector<ColorSpinorField*> &B);

  // forward declarations
  class MG;
  class DiracCoarse;

  /**
     This struct contains all the metadata required to define the
     multigrid solver
   */
  struct MGParam : SolverParam {

    MGParam(const QudaInvertParam &invParam, std::vector<ColorSpinorField*> &B, 
	    DiracMatrix &matResidual, DiracMatrix &matSmooth) :
    SolverParam(invParam), B(B), matResidual(matResidual), matSmooth(matSmooth) { ; }

    MGParam(const MGParam &param, const std::vector<ColorSpinorField*> &B, 
	    DiracMatrix &matResidual, DiracMatrix &matSmooth) :
    SolverParam(param), level(param.level), Nlevel(param.Nlevel), spinBlockSize(param.spinBlockSize),
      Nvec(param.Nvec), coarse(param.coarse), fine(param.fine),  B(B), nu_pre(param.nu_pre), 
    nu_post(param.nu_post),  matResidual(matResidual), matSmooth(matSmooth), smoother(param.smoother) { 
      for (int i=0; i<QUDA_MAX_DIM; i++) geoBlockSize[i] = param.geoBlockSize[i];
    }

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
    const std::vector<ColorSpinorField*> &B;

    /** Number of pre-smoothing applications to perform */
    int nu_pre;

    /** Number of pre-smoothing applications to perform */
    int nu_post;

    /** The Dirac operator to use for residual computation */
    DiracMatrix &matResidual;

    /** The Dirac operator to use for smoothing */
    DiracMatrix &matSmooth;

    /** Filename for where to load/store the null space */
    char filename[100];

    /** What type of smoother to use */
    QudaInverterType smoother;
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

    /** This is the next lower level */
    MG *coarse;

    /** This is the next coarser level */
    MG *fine;

    /** Storage for the parameter struct for the coarse grid */
    MGParam *param_coarse;

    /** Storage for the parameter struct for the pre-smoother */
    MGParam *param_presmooth;

    /** Storage for the parameter struct for the post-smoother */
    MGParam *param_postsmooth;

    /** The coarse-grid representation of the null space vectors */
    std::vector<ColorSpinorField*> *B_coarse;

    /** Residual vector */
    ColorSpinorField *r;

    /** Coarse residual vector */
    ColorSpinorField *r_coarse;

    /** Coarse solution vector */
    ColorSpinorField *x_coarse;

    /** The coarse grid operator */
    DiracCoarse *matCoarse;

    // hack vectors
    ColorSpinorField *hack1, *hack2, *hack3, *hack4;

    //Auxiliary field to preserve solution vector
    ColorSpinorField *y;

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

    void verify();

    /**
       This applies the V-cycle to the residual vector returning the residual vector
       @param out The solution vector
       @param in The residual vector (or equivalently the right hand side vector)
     */
    void operator()(ColorSpinorField &out, ColorSpinorField &in);

    /**
       This applies the V-cycle to the residual vector returning the residual vector
       @param out The solution vector
       @param in The residual vector (or equivalently the right hand side vector)
     */
    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in) {
      operator()(static_cast<ColorSpinorField&>(out), static_cast<ColorSpinorField&>(in));
    }
  };

  void CoarseOp(const Transfer &T, GaugeField &Y, GaugeField &X, QudaPrecision precision, const cudaGaugeField &gauge);

  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Y, 
		   const GaugeField &X, QudaPrecision precision, double kappa);

  class DiracCoarse : public DiracMatrix {

    // restrictor / prolongator defined here
    const Transfer *t;
    ColorSpinorField &tmp;  // fine colorspinor field
    ColorSpinorField &tmp2; // fine colorspinor field
    cpuGaugeField *Y; //Coarse gauge field
    cpuGaugeField *X; //Coarse clover term

    void initializeCoarse();  //Initialize the coarse gauge field

  public:
    DiracCoarse(const Dirac &d, const Transfer &t, ColorSpinorField &tmp, ColorSpinorField &tmp2);
    DiracCoarse(const Dirac *d, const Transfer *t, ColorSpinorField &tmp, ColorSpinorField &tmp2);
    virtual ~DiracCoarse();

    void operator()(ColorSpinorField &out, const ColorSpinorField &in) const;

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, ColorSpinorField &dummy) const
    { (*this)(out,in); }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in, 
		    ColorSpinorField &dummy, ColorSpinorField &dummy2) const
    { (*this)(out,in); }

  };

} // namespace quda

#endif // _MG_QUDA_H
