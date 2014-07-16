#ifndef _LANCZOS_QUDA_H
#define _LANCZOS_QUDA_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <ritz_quda.h>
#include <color_spinor_field.h>
#include <eig_variables.h>

namespace quda {

  class Eig_Solver {

  protected:
    QudaEigParam &eigParam;
    TimeProfile &profile;

  public:
  Eig_Solver(QudaEigParam &eigParam, TimeProfile &profile) : eigParam(eigParam), profile(profile) { ; }
    virtual ~Eig_Solver() { ; }

    virtual void operator()(double *alpha, double *beta, cudaColorSpinorField **Eig_Vec, 
                            cudaColorSpinorField &r, cudaColorSpinorField &Apsi, int k0, int m) = 0;

    // solver factory
    static Eig_Solver* create(QudaEigParam &param, RitzMat &ritz_mat, TimeProfile &profile);

    bool convergence(const double &r2, const double &hq2, const double &r2_tol, 
		     const double &hq_tol);
 
    /**
       Prints out the running statistics of the solver (requires a verbosity of QUDA_VERBOSE)
     */
    void PrintStats(const char*, int k, const double &r2, const double &b2, const double &hq2);

    /** 
	Prints out the summary of the solver convergence (requires a
	versbosity of QUDA_SUMMARIZE).  Assumes
	QudaEigParam.true_res and QudaEigParam.true_res_hq has
	been set
    */
    void PrintSummary(const char *name, int k, const double &r2, const double &b2);

    void GrandSchm_test(cudaColorSpinorField &psi, cudaColorSpinorField **Eig_Vec, int Nvec, double *delta);
  };

  class Lanczos : public Eig_Solver {

  private:
    const RitzMat &ritz_mat;

  public:
    Lanczos(RitzMat &ritz_mat, QudaEigParam &eigParam, TimeProfile &profile);
    virtual ~Lanczos();

    void operator()(double *alpha, double *beta, cudaColorSpinorField **Eig_Vec, 
                    cudaColorSpinorField &r, cudaColorSpinorField &Apsi, int k0, int m);
  };

  class ImpRstLanczos : public Eig_Solver {

  private:
    const RitzMat &ritz_mat;

  public:
    ImpRstLanczos(RitzMat &ritz_mat, QudaEigParam &eigParam, TimeProfile &profile);
    virtual ~ImpRstLanczos();

    void operator()(double *alpha, double *beta, cudaColorSpinorField **Eig_Vec, 
                    cudaColorSpinorField &r, cudaColorSpinorField &Apsi, int k0, int m);
  };


} // namespace quda

#endif // _LANCZOS_QUDA_H
