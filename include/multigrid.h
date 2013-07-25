#ifndef _MG_QUDA_H
#define _MG_QUDA_H

#include <invert_quda.h>
#include <transfer.h>
#include <vector>

#define QUDA_MAX_MG_LEVEL 2

extern char vecfile[];
extern int nvec; 
  
namespace quda {

  // FIXME - these definitions are strictly temporary
  void loadVectors(std::vector<ColorSpinorField*> &B);

  class MG;

  /**
     This struct contains all the metadata required to define the
     multigrid solver
   */
  struct MGParam : SolverParam {

    MGParam(const QudaInvertParam &invParam, std::vector<ColorSpinorField*> &B, 
	    DiracMatrix &matResidual, DiracMatrix &matSmooth) :
    SolverParam(), B(B), matResidual(matResidual), matSmooth(matSmooth) { ; }

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
    Solver *smoother;

    /** This is the next lower level */
    MG *coarse;

    /** This is the next coarser level */
    MG *fine;

    /** Residual vector */
    ColorSpinorField *r;

    /** Coarse residual vector */
    ColorSpinorField *r_coarse;

    /** Coarse solution vector */
    ColorSpinorField *x_coarse;

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

  class DiracCoarse : public DiracMatrix {

    // restrictor / prolongator defined here
    const Transfer *t;
    /*
      The types of these vectors is very important.  If the wrong
      types are passed, then this will bork.  This is a hack for
      functionality to work around to the fact that Transfer is only
      implemented to work on CPU, whereas the Dirac operator will only
      act on GPU fields.
     */
    ColorSpinorField &tmp;  // must be a cpuColorSpinorField double/single with arbitrary ordering
    ColorSpinorField &tmp2; // must be a cpuColorSpinorField double/single with arbitrary ordering
    ColorSpinorField &tmp3; // must be a cudaColorSpinorField with QUDA internal ordering and UKQCD Dirac basis
    ColorSpinorField &tmp4; // must be a cudaColorSpinorField with QUDA internal ordering and UKQCD Dirac basis

  public:
  DiracCoarse(const Dirac &d, const Transfer &t, ColorSpinorField &tmp, ColorSpinorField &tmp2, ColorSpinorField &tmp3, ColorSpinorField &tmp4) 
    : DiracMatrix(d), t(&t), tmp(tmp), tmp2(tmp2), tmp3(tmp3), tmp4(tmp4) { }
  DiracCoarse(const Dirac *d, const Transfer *t, ColorSpinorField &tmp, ColorSpinorField &tmp2, ColorSpinorField &tmp3, ColorSpinorField &tmp4)
    : DiracMatrix(d), t(t), tmp(tmp), tmp2(tmp2), tmp3(tmp3), tmp4(tmp4) { }

    void operator()(ColorSpinorField &out, const ColorSpinorField &in) const
    {
      t->P(tmp, in);
      tmp3 = tmp;
      dirac->M(tmp4, tmp3);
      tmp2 = tmp4;
      t->R(out, tmp2);
    }

    // FIXME - additional dummy fields not used
    void operator()(ColorSpinorField &out, const ColorSpinorField &in, ColorSpinorField &dummy) const
    {
      t->P(tmp, in);
      tmp3 = tmp;
      dirac->M(tmp4, tmp3);
      tmp2 = tmp4;
      t->R(out, tmp2);
    }

    // FIXME - additional dummy fields not used
    void operator()(ColorSpinorField &out, const ColorSpinorField &in, 
		    ColorSpinorField &dummy, ColorSpinorField &dummy2) const
    {
      t->P(tmp, in);
      tmp3 = tmp;
      dirac->M(tmp4, tmp3);
      tmp2 = tmp4;
      t->R(out, tmp2);
    }
  };

} // namespace quda

#endif // _MG_QUDA_H
