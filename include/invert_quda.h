#ifndef _INVERT_QUDA_H
#define _INVERT_QUDA_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>

namespace quda {

  class Solver {

  protected:
    QudaInvertParam &invParam;
    TimeProfile &profile;

  public:
  Solver(QudaInvertParam &invParam, TimeProfile &profile) : invParam(invParam), profile(profile) { ; }
    virtual ~Solver() { ; }

    virtual void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in) = 0;

    // solver factory
    static Solver* create(QudaInvertParam &param, DiracMatrix &mat, DiracMatrix &matSloppy,
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
	QudaInvertParam.true_res and QudaInvertParam.true_res_hq has
	been set
    */
    void PrintSummary(const char *name, int k, const double &r2, const double &b2);

  };

  class CG : public Solver {

  private:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;

  public:
    CG(DiracMatrix &mat, DiracMatrix &matSloppy, QudaInvertParam &invParam, TimeProfile &profile);
    virtual ~CG();

    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };

  class BiCGstab : public Solver {

  private:
    DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    const DiracMatrix &matPrecon;

    // pointers to fields to avoid multiple creation overhead
    cudaColorSpinorField *yp, *rp, *pp, *vp, *tmpp, *tp, *wp, *zp;
    bool init;

  public:
    BiCGstab(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
	     QudaInvertParam &invParam, TimeProfile &profile);
    virtual ~BiCGstab();

    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };

  class GCR : public Solver {

  private:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    const DiracMatrix &matPrecon;

    Solver *K;
    QudaInvertParam Kparam; // parameters for preconditioner solve

  public:
    GCR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
	QudaInvertParam &invParam, TimeProfile &profile);
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
    MR(DiracMatrix &mat, QudaInvertParam &invParam, TimeProfile &profile);
    virtual ~MR();

    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };

  // multigrid solver
  class alphaSA : public Solver {

  protected:
    const DiracMatrix &mat;

  public:
    alphaSA(DiracMatrix &mat, QudaInvertParam &invParam, TimeProfile &profile);
    virtual ~alphaSA() { ; }

    void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in);
  };

  class MultiShiftSolver {

  protected:
    QudaInvertParam &invParam;
    TimeProfile &profile;

  public:
    MultiShiftSolver(QudaInvertParam &invParam, TimeProfile &profile) : 
    invParam(invParam), profile(profile) { ; }
    virtual ~MultiShiftSolver() { ; }

    virtual void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in) = 0;
  };

  class MultiShiftCG : public MultiShiftSolver {

  protected:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;

  public:
    MultiShiftCG(DiracMatrix &mat, DiracMatrix &matSloppy, QudaInvertParam &invParam, TimeProfile &profile);
    virtual ~MultiShiftCG();

    void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in);
  };

  /**
  This computes the optimum guess for the system Ax=b in the L2
  residual norm.  For use in the HMD force calculations using a
  minimal residual chronological method This computes the guess
  solution as a linear combination of a given number of previous
  solutions.  Following Brower et al, only the orthogonalised vector
  basis is stored to conserve memory.*/
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
