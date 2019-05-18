#include <quda_internal.h>
#include <invert_quda.h>
#include <multigrid.h>
#include <eigensolve_quda.h>
#include <cmath>

namespace quda {

  static void report(const char *type) {
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating a %s solver\n", type);
  }

  Solver::Solver(SolverParam &param, TimeProfile &profile) : param(param), profile(profile), node_parity(0) {
    // compute parity of the node
    for (int i=0; i<4; i++) node_parity += commCoords(i);
    node_parity = node_parity % 2;
  }

  // solver factory
  Solver* Solver::create(SolverParam &param, DiracMatrix &mat, DiracMatrix &matSloppy,
			 DiracMatrix &matPrecon, TimeProfile &profile)
  {
    Solver *solver = nullptr;

    if (param.preconditioner && param.inv_type != QUDA_GCR_INVERTER)
      errorQuda("Explicit preconditoner not supported for %d solver", param.inv_type);

    if (param.preconditioner && param.inv_type_precondition != QUDA_MG_INVERTER)
      errorQuda("Explicit preconditoner not supported for %d preconditioner", param.inv_type_precondition);

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
      if (param.preconditioner) {
	Solver *mg = param.mg_instance ? static_cast<MG*>(param.preconditioner) : static_cast<multigrid_solver*>(param.preconditioner)->mg;
	// FIXME dirty hack to ensure that preconditioner precision set in interface isn't used in the outer GCR-MG solver
	if (!param.mg_instance) param.precision_precondition = param.precision_sloppy;
	solver = new GCR(mat, *(mg), matSloppy, matPrecon, param, profile);
      } else {
	solver = new GCR(mat, matSloppy, matPrecon, param, profile);
      }
      break;
    case QUDA_CA_CG_INVERTER:
      report("CA-CG");
      solver = new CACG(mat, matSloppy, param, profile);
      break;
    case QUDA_CA_CGNE_INVERTER:
      report("CA-CGNE");
      solver = new CACGNE(mat, matSloppy, param, profile);
      break;
    case QUDA_CA_CGNR_INVERTER:
      report("CA-CGNR");
      solver = new CACGNR(mat, matSloppy, param, profile);
      break;
    case QUDA_CA_GCR_INVERTER:
      report("CA-GCR");
      solver = new CAGCR(mat, matSloppy, param, profile);
      break;
    case QUDA_MR_INVERTER:
      report("MR");
      solver = new MR(mat, matSloppy, param, profile);
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
    case QUDA_BICGSTABL_INVERTER:
      report("BICGSTABL");
      solver = new BiCGstabL(mat, matSloppy, param, profile);
      break;
    case QUDA_EIGCG_INVERTER:
      report("EIGCG");
      solver = new IncEigCG(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_INC_EIGCG_INVERTER:
      report("INC EIGCG");
      solver = new IncEigCG(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_GMRESDR_INVERTER:
      report("GMRESDR");
      if (param.preconditioner) {
	multigrid_solver *mg = static_cast<multigrid_solver*>(param.preconditioner);
	// FIXME dirty hack to ensure that preconditioner precision set in interface isn't used in the outer GCR-MG solver
	param.precision_precondition = param.precision_sloppy;
	solver = new GMResDR(mat, *(mg->mg), matSloppy, matPrecon, param, profile);
      } else {
	solver = new GMResDR(mat, matSloppy, matPrecon, param, profile);
      }
      break;
    case QUDA_CGNE_INVERTER:
      report("CGNE");
      solver = new CGNE(mat, matSloppy, param, profile);
      break;
    case QUDA_CGNR_INVERTER:
      report("CGNR");
      solver = new CGNR(mat, matSloppy, param, profile);
      break;
    case QUDA_CG3_INVERTER:
      report("CG3");
      solver = new CG3(mat, matSloppy, param, profile);
      break;
    case QUDA_CG3NE_INVERTER:
      report("CG3NE");
      solver = new CG3NE(mat, matSloppy, param, profile);
      break;
    case QUDA_CG3NR_INVERTER:
      report("CG3NR");
      // CG3NR is included in CG3NE
      solver = new CG3NE(mat, matSloppy, param, profile);
      break;
    default:
      errorQuda("Invalid solver type %d", param.inv_type);
    }

    return solver;
  }

  void Solver::blocksolve(ColorSpinorField& out, ColorSpinorField& in){
    for (int i = 0; i < param.num_src; i++) {
      (*this)(out.Component(i), in.Component(i));
      param.true_res_offset[i] = param.true_res;
      param.true_res_hq_offset[i] = param.true_res_hq;
    }
  }

  double Solver::stopping(double tol, double b2, QudaResidualType residual_type) {

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

  bool Solver::convergence(double r2, double hq2, double r2_tol, double hq_tol) {

    // check the heavy quark residual norm if necessary
    if ( (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) && (hq2 > hq_tol) )
      return false;

    // check the L2 relative residual norm if necessary
    if ( ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) ||
	  (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) && (r2 > r2_tol) )
      return false;

    return true;
  }

  bool Solver::convergenceHQ(double r2, double hq2, double r2_tol, double hq_tol) {

    // check the heavy quark residual norm if necessary
    if ( (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) && (hq2 > hq_tol) )
      return false;

    return true;
  }

  bool Solver::convergenceL2(double r2, double hq2, double r2_tol, double hq_tol) {

    // check the L2 relative residual norm if necessary
    if ( ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) ||
    (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) && (r2 > r2_tol) )
      return false;

    return true;
  }

  void Solver::PrintStats(const char* name, int k, double r2, double b2, double hq2) {
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

  void Solver::PrintSummary(const char *name, int k, double r2, double b2,
                            double r2_tol, double hq_tol) {
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      if (param.compute_true_res) {
	if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
	  printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e "
                     "(requested = %e), heavy-quark residual = %e (requested = %e)\n",
		     name, k, sqrt(r2/b2), param.true_res, sqrt(r2_tol/b2), param.true_res_hq, hq_tol);
	} else {
	  printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e (requested = %e)\n",
		     name, k, sqrt(r2/b2), param.true_res, sqrt(r2_tol/b2));
	}
      } else {
	if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
	  printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e "
                     "(requested = %e), heavy-quark residual = %e (requested = %e)\n",
		     name, k, sqrt(r2/b2), sqrt(r2_tol/b2), param.true_res_hq, hq_tol);
	} else {
	  printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %e (requested = %e)\n",
                     name, k, sqrt(r2/b2), sqrt(r2_tol/b2));
	}
      }
    }
  }

  /*
  void Solver::loadVectors(std::vector<ColorSpinorField*> &eig_vecs, std::string vec_infile) {

    //profile.TPSTOP(QUDA_PROFILE_INIT);
    //profile.TPSTART(QUDA_PROFILE_IO);

#ifdef HAVE_QIO
    const int Nvec = eig_vecs.size();
    if (strcmp(vec_infile.c_str(),"")!=0) {
      printfQuda("Start loading %04d vectors from %s\n", Nvec, vec_infile.c_str());

      std::vector<ColorSpinorField*> tmp;
      if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
        ColorSpinorParam csParam(*eig_vecs[0]);
        csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        csParam.setPrecision(eig_vecs[0]->Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION :
eig_vecs[0]->Precision()); csParam.location = QUDA_CPU_FIELD_LOCATION; csParam.create = QUDA_NULL_FIELD_CREATE; for (int
i=0; i<Nvec; i++) { tmp.push_back(ColorSpinorField::Create(csParam));
        }
      } else {
        for (int i=0; i<Nvec; i++) {
          tmp.push_back(eig_vecs[i]);
        }
      }

      void **V = static_cast<void**>(safe_malloc(Nvec*sizeof(void*)));
      for (int i=0; i<Nvec; i++) {
        V[i] = tmp[i]->V();
        if (V[i] == NULL) {
          printfQuda("Could not allocate space for eigenVector[%d]\n", i);
        }
      }

      read_spinor_field(vec_infile.c_str(), &V[0], eig_vecs[0]->Precision(),
                        eig_vecs[0]->X(), eig_vecs[0]->Ncolor(), eig_vecs[0]->Nspin(),
                        Nvec, 0,  (char**)0);
      host_free(V);
      if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
        for (int i=0; i<Nvec; i++) {
          *eig_vecs[i] = *tmp[i];
          delete tmp[i];
        }
      }

      printfQuda("Done loading vectors\n");
    } else {
      errorQuda("No eigenspace input file defined.");
    }
#else
    errorQuda("\nQIO library was not built.\n");
#endif
    //profile.TPSTOP(QUDA_PROFILE_IO);
    //profile.TPSTART(QUDA_PROFILE_INIT);

    return;
  }

  void Solver::saveVectors(std::vector<ColorSpinorField*> &eig_vecs,
                           std::string vec_outfile) {

    //profile.TPSTOP(QUDA_PROFILE_INIT);
    //profile.TPSTART(QUDA_PROFILE_IO);

#ifdef HAVE_QIO
    const int Nvec = eig_vecs.size();
    std::vector<ColorSpinorField*> tmp;
    if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
      ColorSpinorParam csParam(*eig_vecs[0]);
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.setPrecision(eig_vecs[0]->Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION :
eig_vecs[0]->Precision()); csParam.location = QUDA_CPU_FIELD_LOCATION; csParam.create = QUDA_NULL_FIELD_CREATE; for (int
i=0; i<Nvec; i++) { tmp.push_back(ColorSpinorField::Create(csParam)); *tmp[i] = *eig_vecs[i];
      }
    } else {
      for (int i=0; i<Nvec; i++) {
        tmp.push_back(eig_vecs[i]);
      }
    }
    if (strcmp(vec_outfile.c_str(),"")!=0) {
      printfQuda("Start saving %d vectors to %s\n", Nvec, vec_outfile.c_str());

      void **V = static_cast<void**>(safe_malloc(Nvec*sizeof(void*)));
      for (int i=0; i<Nvec; i++) {
        V[i] = tmp[i]->V();
        if (V[i] == NULL) {
          printfQuda("Could not allocate space for eigenVector[%04d]\n", i);
        }
      }

      write_spinor_field(vec_outfile.c_str(), &V[0], eig_vecs[0]->Precision(),
                         eig_vecs[0]->X(), eig_vecs[0]->Ncolor(), eig_vecs[0]->Nspin(),
                         Nvec, 0,  (char**)0);

      host_free(V);
      printfQuda("Done saving vectors\n");
      if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
        for (int i=0; i<Nvec; i++) delete tmp[i];
      }

    } else {
      errorQuda("No eigenspace output file defined.");
    }
#else
    errorQuda("\nQIO library was not built.\n");
#endif
    //profile.TPSTOP(QUDA_PROFILE_IO);
    //profile.TPSTART(QUDA_PROFILE_INIT);

    return;
  }
*/

  bool MultiShiftSolver::convergence(const double *r2, const double *r2_tol, int n) const {

    for (int i=0; i<n; i++) {
      // check the L2 relative residual norm if necessary
      if ( ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) ||
	    (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) && (r2[i] > r2_tol[i]) && r2_tol[i] != 0.0)
	return false;
    }

    return true;
  }

} // namespace quda
