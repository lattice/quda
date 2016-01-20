#include <quda_internal.h>
#include <invert_quda.h>
#include <cmath>

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
    case QUDA_SD_INVERTER:
      report("SD");
      solver = new SD(mat, param, profile);
      break;
    case QUDA_XSD_INVERTER:
#ifdef MULTI_GPU
      report("XSD");
      solver = new XSD(mat, param, profile);
#else
      errorQuda("Extended Steepest Descent is multi-gpu only");
#endif
      break;
    case QUDA_PCG_INVERTER:
      report("PCG");
      solver = new PreconCG(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_MPCG_INVERTER:
      report("MPCG");
      solver = new MPCG(mat, param, profile);
      break;
    case QUDA_MPBICGSTAB_INVERTER:
      report("MPBICGSTAB");
      solver = new MPBiCGstab(mat, param, profile);
      break;
    default:
      errorQuda("Invalid solver type");
    }
    
    return solver;
  }

  double Solver::stopping(const double &tol, const double &b2, QudaResidualType residual_type) {

    double stop=0.0;
    if ( (residual_type & QUDA_L2_ABSOLUTE_RESIDUAL) &&
	 (residual_type & QUDA_L2_RELATIVE_RESIDUAL) ) {
      // use the most stringent stopping condition
      double lowest = (b2 < 1.0) ? b2 : 1.0;
      stop = lowest*tol*tol;
    } else if (residual_type & QUDA_L2_ABSOLUTE_RESIDUAL) {
      stop = tol*tol;
    } else {
      stop = b2*tol*tol;
    }

    return stop;
  }

  bool Solver::convergence(const double &r2, const double &hq2, const double &r2_tol, 
			   const double &hq_tol) {
    //printf("converge: L2 %e / %e and HQ %e / %e\n", r2, r2_tol, hq2, hq_tol);

    // check the heavy quark residual norm if necessary
    if ( (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) && (hq2 > hq_tol) ) 
      return false;

    // check the L2 relative residual norm if necessary
    if ( ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) ||
	  (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) && (r2 > r2_tol) ) 
      return false;

    return true;
  }

// 
  bool Solver::convergenceHQ(const double &r2, const double &hq2, const double &r2_tol, 
         const double &hq_tol) {
    //printf("converge: L2 %e / %e and HQ %e / %e\n", r2, r2_tol, hq2, hq_tol);

    // check the heavy quark residual norm if necessary
    if ( (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) && (hq2 > hq_tol) ) 
      return false;

    return true;
  }

  bool Solver::convergenceL2(const double &r2, const double &hq2, const double &r2_tol, 
         const double &hq_tol) {
    //printf("converge: L2 %e / %e and HQ %e / %e\n", r2, r2_tol, hq2, hq_tol);

    // check the L2 relative residual norm if necessary
    if ( ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) ||
    (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) && (r2 > r2_tol) ) 
      return false;

    return true;
  }

  void Solver::PrintStats(const char* name, int k, const double &r2, 
			  const double &b2, const double &hq2) {
    if (getVerbosity() >= QUDA_VERBOSE) {
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
	printfQuda("%s: %d iterations, <r,r> = %e, |r|/|b| = %e, heavy-quark residual = %e\n", 
		   name, k, r2, sqrt(r2/b2), hq2);
      } else {
	printfQuda("%s: %d iterations, <r,r> = %e, |r|/|b| = %e\n", 
		   name, k, r2, sqrt(r2/b2));
      }
    }

    if (std::isnan(r2)) errorQuda("Solver appears to have diverged");
  }

  void Solver::PrintSummary(const char *name, int k, const double &r2, const double &b2) {
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
	printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e, heavy-quark residual = %e\n",
		   name, k, sqrt(r2/b2), param.true_res, param.true_res_hq);
      } else {
	printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e\n", 
		   name, k, sqrt(r2/b2), param.true_res);
      }

    }
  }

  // Deflated solver factory
  DeflatedSolver* DeflatedSolver::create(SolverParam &param, DiracMatrix *mat, DiracMatrix *matSloppy, DiracMatrix *matCGSloppy, DiracMatrix *matDeflate, TimeProfile *profile)
  {
    DeflatedSolver* solver=0;

    if (param.inv_type == QUDA_INC_EIGCG_INVERTER || param.inv_type == QUDA_EIGCG_INVERTER) {
      report("Incremental EIGCG");
      if(profile != NULL)
        solver = new IncEigCG(mat, matSloppy, matCGSloppy, matDeflate, param, profile);
      else
        solver = new IncEigCG(param);//hack to clean deflation resources
    }else if (param.inv_type == QUDA_GMRESDR_INVERTER || param.inv_type == QUDA_GMRESDR_PROJ_INVERTER ){
      report("GMRESDR");
      if(profile != NULL)
        solver = new GMResDR(mat, matSloppy, matDeflate, param, profile);
      else
        solver = new GMResDR(param);
    }else{
      errorQuda("Invalid solver type");
    }
    
    return solver;
  }

  bool DeflatedSolver::convergence(const double &r2, const double &hq2, const double &r2_tol, 
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

  void DeflatedSolver::PrintStats(const char* name, int k, const double &r2, 
			  const double &b2, const double &hq2) {
    if (getVerbosity() >= QUDA_VERBOSE) {
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
	printfQuda("%s: %d iterations, <r,r> = %e, |r|/|b| = %e, heavy-quark residual = %e\n", 
		   name, k, r2, sqrt(r2/b2), hq2);
      } else {
	printfQuda("%s: %d iterations, <r,r> = %e, |r|/|b| = %e\n", 
		   name, k, r2, sqrt(r2/b2));
      }
    }

    if (std::isnan(r2)) errorQuda("Solver appears to have diverged");
  }

  void DeflatedSolver::PrintSummary(const char *name, int k, const double &r2, const double &b2) {
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
	printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e, heavy-quark residual = %e\n", name, k, sqrt(r2/b2), param.true_res, param.true_res_hq);    
      } else {
	printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e\n", 
		   name, k, sqrt(r2/b2), param.true_res);
      }

    }
  }


} // namespace quda
