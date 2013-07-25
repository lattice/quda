#include <quda_internal.h>
#include <invert_quda.h>

namespace quda {

  static void report(const char *type) {
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating a %s solver\n", type);
  }

  // solver factory
  Solver* Solver::create(SolverParam &param, DiracMatrix &mat, DiracMatrix &matSloppy,
			 DiracMatrix &matPrecon, TimeProfile &profile)
  {
    Solver *solver=0;

    switch (param.inv_type) {
    case QUDA_CG_INVERTER:
      report("CG");
      solver = new CG(mat, matSloppy, param, profile);
      break;
    case QUDA_BICGSTAB_INVERTER:
      report("BiCGstab");
      solver = new BiCGstab(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_GCR_INVERTER:
      report("GCR");
      solver = new GCR(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_MR_INVERTER:
      report("MR");
      solver = new MR(mat, param, profile);
      break;
    default:
      errorQuda("Invalid solver type");
    }
    
    return solver;
  }

  bool Solver::convergence(const double &r2, const double &hq2, const double &r2_tol, 
			   const double &hq_tol) {
    //printf("converge: L2 %e / %e and HQ %e / %e\n", r2, r2_tol, hq2, hq_tol);

    // check the heavy quark residual norm if necessary
    if ( (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) && (hq2 > hq_tol) ) 
      return false;

    // check the L2 relative residual norm if necessary
    if ( (param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) && (r2 > r2_tol) ) 
      return false;

    return true;
  }

  void Solver::PrintStats(const char* name, int k, const double &r2, 
			  const double &b2, const double &hq2) {
    if (param.verbosity >= QUDA_VERBOSE) {
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
	printfQuda("%s: %d iterations, <r,r> = %e, |r|/|b| = %e, heavy-quark residual = %e\n", 
		   name, k, r2, sqrt(r2/b2), hq2);
      } else {
	printfQuda("%s: %d iterations, <r,r> = %e, |r|/|b| = %e\n", 
		   name, k, r2, sqrt(r2/b2));
      }
    }
  }

  void Solver::PrintSummary(const char *name, int k, const double &r2, const double &b2) {
    if (param.verbosity >= QUDA_SUMMARIZE) {
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
	printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e, heavy-quark residual = %e\n", name, k, sqrt(r2/b2), param.true_res, param.true_res_hq);    
      } else {
	printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e\n", 
		   name, k, sqrt(r2/b2), param.true_res);
      }

    }
  }

} // namespace quda
