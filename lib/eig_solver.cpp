#include <quda_internal.h>
#include <invert_quda.h>
#include <lanczos_quda.h>

namespace quda {

  static void report(const char *type) {
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating a %s Eig solver\n", type);
  }

  // solver factory
  Eig_Solver* Eig_Solver::create(QudaEigParam &param, RitzMat &ritz_mat, TimeProfile &profile)
  {
    Eig_Solver *eig_solver=0;

    switch (param.eig_type) {
    case QUDA_LANCZOS:
      report("Lanczos solver");
      eig_solver = new Lanczos(ritz_mat, param, profile);
      break;
    case QUDA_IMP_RST_LANCZOS:
      report("BiCGstab");
      eig_solver = new ImpRstLanczos(ritz_mat, param, profile);
      break;
    default:
      errorQuda("Invalid eig solver type");
    }
    
    return eig_solver;
  }

  bool Eig_Solver::convergence(const double &r2, const double &hq2, const double &r2_tol, 
			   const double &hq_tol) {

    return true;
  }

  void Eig_Solver::PrintStats(const char* name, int k, const double &r2, 
			  const double &b2, const double &hq2) {
  }

  void Eig_Solver::PrintSummary(const char *name, int k, const double &r2, const double &b2) {
  }
  
  void Eig_Solver::GrandSchm_test(cudaColorSpinorField &psi, cudaColorSpinorField **Eig_Vec, int Nvec, double *delta) {
    Complex xp(0.0,0.0);
    for(int i = 0; i<Nvec; ++i)
    {
      xp =cDotProductCuda(*(Eig_Vec[i]), psi);

      if (getVerbosity() >= QUDA_VERBOSE) {
        if(fabs(xp.real()) > 1e-13 || fabs(xp.imag()) > 1e-13)
          printf("[%d] %e %e\n", i, xp.real(),xp.imag());
      }

      xp *= -1.0;
      caxpyCuda(xp, *(Eig_Vec[i]), psi);

      if(i==Nvec-1 && delta)  *delta = xp.real();   //  Re ( vec[Nvec-1],  psi ) needed for Lanczos' delta
    }
  }
} // namespace quda
