#include <quda_internal.h>
#include <invert_quda.h>
#include <multigrid.h>
#include <eigensolve_quda.h>
#include <cmath>
#include <limits>

namespace quda {

  static void report(const char *type) {
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating a %s solver\n", type);
  }

  Solver::Solver(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                 const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile) :
    mat(mat),
    matSloppy(matSloppy),
    matPrecon(matPrecon),
    matEig(matEig),
    param(param),
    profile(profile),
    node_parity(0),
    eig_solve(nullptr),
    deflate_init(false),
    deflate_compute(true),
    recompute_evals(!param.eig_param.preserve_evals)
  {
    // compute parity of the node
    for (int i=0; i<4; i++) node_parity += commCoords(i);
    node_parity = node_parity % 2;
  }

  Solver::~Solver()
  {
    if (eig_solve) {
      delete eig_solve;
      eig_solve = nullptr;
    }
  }

  void Solver::create(ColorSpinorField &x, const ColorSpinorField &b)
  {
    if (checkPrecision(x, b) != param.precision)
      errorQuda("Precision mismatch %d %d", checkPrecision(x, b), param.precision);
  }

  // solver factory
  Solver *Solver::create(SolverParam &param, const DiracMatrix &mat, const DiracMatrix &matSloppy,
                         const DiracMatrix &matPrecon, const DiracMatrix &matEig, TimeProfile &profile)
  {
    Solver *solver = nullptr;

    if (param.preconditioner && param.inv_type != QUDA_GCR_INVERTER)
      errorQuda("Explicit preconditoner not supported for %d solver", param.inv_type);

    if (param.preconditioner && param.inv_type_precondition != QUDA_MG_INVERTER)
      errorQuda("Explicit preconditoner not supported for %d preconditioner", param.inv_type_precondition);

    switch (param.inv_type) {
    case QUDA_CG_INVERTER:
      report("CG");
      solver = new CG(mat, matSloppy, matPrecon, matEig, param, profile);
      break;
    case QUDA_BICGSTAB_INVERTER:
      report("BiCGstab");
      solver = new BiCGstab(mat, matSloppy, matPrecon, matEig, param, profile);
      break;
    case QUDA_GCR_INVERTER:
      report("GCR");
      if (param.preconditioner) {
	Solver *mg = param.mg_instance ? static_cast<MG*>(param.preconditioner) : static_cast<multigrid_solver*>(param.preconditioner)->mg;
	// FIXME dirty hack to ensure that preconditioner precision set in interface isn't used in the outer GCR-MG solver
	if (!param.mg_instance) param.precision_precondition = param.precision_sloppy;
        solver = new GCR(mat, *(mg), matSloppy, matPrecon, matEig, param, profile);
      } else {
        solver = new GCR(mat, matSloppy, matPrecon, matEig, param, profile);
      }
      break;
    case QUDA_CA_CG_INVERTER:
      report("CA-CG");
      solver = new CACG(mat, matSloppy, matPrecon, matEig, param, profile);
      break;
    case QUDA_CA_CGNE_INVERTER:
      report("CA-CGNE");
      solver = new CACGNE(mat, matSloppy, matPrecon, matEig, param, profile);
      break;
    case QUDA_CA_CGNR_INVERTER:
      report("CA-CGNR");
      solver = new CACGNR(mat, matSloppy, matPrecon, matEig, param, profile);
      break;
    case QUDA_CA_GCR_INVERTER:
      report("CA-GCR");
      solver = new CAGCR(mat, matSloppy, matPrecon, matEig, param, profile);
      break;
    case QUDA_MR_INVERTER:
      report("MR");
      solver = new MR(mat, matSloppy, param, profile);
      break;
    case QUDA_SD_INVERTER:
      report("SD");
      solver = new SD(mat, param, profile);
      break;
    case QUDA_PCG_INVERTER:
      report("PCG");
      solver = new PreconCG(mat, matSloppy, matPrecon, matEig, param, profile);
      break;
    case QUDA_BICGSTABL_INVERTER:
      report("BICGSTABL");
      solver = new BiCGstabL(mat, matSloppy, matEig, param, profile);
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
      solver = new CGNE(mat, matSloppy, matPrecon, matEig, param, profile);
      break;
    case QUDA_CGNR_INVERTER:
      report("CGNR");
      solver = new CGNR(mat, matSloppy, matPrecon, matEig, param, profile);
      break;
    case QUDA_CG3_INVERTER:
      report("CG3");
      solver = new CG3(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_CG3NE_INVERTER:
      report("CG3NE");
      solver = new CG3NE(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_CG3NR_INVERTER:
      report("CG3NR");
      solver = new CG3NR(mat, matSloppy, matPrecon, param, profile);
      break;
    default:
      errorQuda("Invalid solver type %d", param.inv_type);
    }

    if (!mat.hermitian() && solver->hermitian()) errorQuda("Cannot solve non-Hermitian system with Hermitian solver");
    return solver;
  }

  void Solver::constructDeflationSpace(const ColorSpinorField &meta, const DiracMatrix &mat)
  {
    if (deflate_init) return;

    // Deflation requested + first instance of solver
    bool profile_running = profile.isRunning(QUDA_PROFILE_INIT);
    if (!param.is_preconditioner && !profile_running) profile.TPSTART(QUDA_PROFILE_INIT);

    eig_solve = EigenSolver::create(&param.eig_param, mat, profile);

    // Clone from an existing vector
    ColorSpinorParam csParam(meta);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    // This is the vector precision used by matEig
    csParam.setPrecision(param.precision_eigensolver, QUDA_INVALID_PRECISION, true);

    if (deflate_compute) {

      deflation_space *space = reinterpret_cast<deflation_space *>(param.eig_param.preserve_deflation_space);

      if (space && space->evecs.size() != 0) {
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Restoring deflation space of size %lu\n", space->evecs.size());

        if ((!space->svd && param.eig_param.n_conv != (int)space->evecs.size())
            || (space->svd && 2 * param.eig_param.n_conv != (int)space->evecs.size()))
          errorQuda("Preserved deflation space size %lu does not match expected %d", space->evecs.size(),
                    param.eig_param.n_conv);

        // move vectors from preserved space to local space
        evecs = std::move(space->evecs);

        if (param.eig_param.n_conv != (int)space->evals.size())
          errorQuda("Preserved eigenvalues %lu does not match expected %lu", space->evals.size(), evals.size());

        // move vectors from preserved space to local space
        evals = std::move(space->evals);

        delete space;
        param.eig_param.preserve_deflation_space = nullptr;

        // we successfully got the deflation space so disable any subsequent recalculation
        deflate_compute = false;
      } else {
        // Computing the deflation space, rather than transferring, so we create space.
        resize(evecs, param.eig_param.n_conv, csParam);
        evals.resize(param.eig_param.n_conv, 0.0);
      }
    }

    deflate_init = true;

    if (!param.is_preconditioner && !profile_running) profile.TPSTOP(QUDA_PROFILE_INIT);
  }

  void Solver::destroyDeflationSpace()
  {
    if (deflate_init) {
      if (param.eig_param.preserve_deflation) {
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Preserving deflation space of size %lu\n", evecs.size());

        if (param.eig_param.preserve_deflation_space) {
          deflation_space *space = reinterpret_cast<deflation_space *>(param.eig_param.preserve_deflation_space);
          delete space;
        }

        deflation_space *space = new deflation_space;

        // if evecs size = 2x evals size then we are doing an SVD deflation
        space->svd = (evecs.size() == 2 * evals.size()) ? true : false;

        space->evecs = std::move(evecs);
        space->evals = std::move(evals);

        param.eig_param.preserve_deflation_space = space;
      }

      evecs.clear();
      evals.clear();
      deflate_init = false;
    }
  }

  void Solver::injectDeflationSpace(std::vector<ColorSpinorField> &defl_space)
  {
    if (!evecs.empty()) errorQuda("Solver deflation space should be empty, instead size=%lu\n", evecs.size());
    evecs = std::move(defl_space); // move defl_space to evecs
  }

  void Solver::extractDeflationSpace(std::vector<ColorSpinorField> &defl_space)
  {
    if (!defl_space.empty())
      errorQuda("Container deflation space should be empty, instead size=%lu\n", defl_space.size());
    defl_space = std::move(evecs); // move evecs to defl_space
  }

  void Solver::extendSVDDeflationSpace()
  {
    if (!deflate_init) errorQuda("Deflation space for this solver not computed");

    // Double the size deflation space to accomodate for the extra singular vectors
    resize(evecs, 2 * param.eig_param.n_conv, QUDA_ZERO_FIELD_CREATE);
  }

  void Solver::blocksolve(ColorSpinorField &out, ColorSpinorField &in)
  {
    for (int i = 0; i < param.num_src; i++) {
      (*this)(out.Component(i), in.Component(i));
      param.true_res_offset[i] = param.true_res;
      param.true_res_hq_offset[i] = param.true_res_hq;
    }
  }

  double Solver::stopping(double tol, double b2, QudaResidualType residual_type)
  {
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
    if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
      if (std::isnan(hq2) || std::isinf(hq2))
        errorQuda("Solver appears to have diverged with heavy quark residual %9.6e", hq2);

      if (hq2 > hq_tol) return false;
    }

    // check the L2 relative residual norm if necessary
    if ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) || (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) {
      if (std::isnan(r2) || std::isinf(r2)) errorQuda("Solver appears to have diverged with residual %9.6e", r2);

      if (r2 > r2_tol) return false;
    }

    return true;
  }

  bool Solver::convergenceHQ(double, double hq2, double, double hq_tol)
  {
    // check the heavy quark residual norm if necessary
    if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
      if (std::isnan(hq2) || std::isinf(hq2))
        errorQuda("Solver appears to have diverged with heavy quark residual %9.6e", hq2);

      if (hq2 > hq_tol) return false;
    }

    return true;
  }

  bool Solver::convergenceL2(double r2, double, double r2_tol, double)
  {
    // check the L2 relative residual norm if necessary
    if ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) || (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) {
      if (std::isnan(r2) || std::isinf(r2)) errorQuda("Solver appears to have diverged with residual %9.6e", r2);

      if (r2 > r2_tol) return false;
    }

    return true;
  }

  void Solver::PrintStats(const char* name, int k, double r2, double b2, double hq2) {
    if (getVerbosity() >= QUDA_VERBOSE) {
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
        printfQuda("%s: %5d iterations, <r,r> = %9.6e, |r|/|b| = %9.6e, heavy-quark residual = %9.6e\n", name, k, r2,
                   sqrt(r2 / b2), hq2);
      } else {
        printfQuda("%s: %5d iterations, <r,r> = %9.6e, |r|/|b| = %9.6e\n", name, k, r2, sqrt(r2 / b2));
      }
    }

    if (std::isnan(r2) || std::isinf(r2)) errorQuda("Solver appears to have diverged");
  }

  void Solver::PrintSummary(const char *name, int k, double r2, double b2,
                            double r2_tol, double hq_tol) {
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      if (param.compute_true_res) {
	if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
          printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %9.6e, true = %9.6e "
                     "(requested = %9.6e), heavy-quark residual = %9.6e (requested = %9.6e)\n",
                     name, k, sqrt(r2 / b2), param.true_res, sqrt(r2_tol / b2), param.true_res_hq, hq_tol);
        } else {
          printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %9.6e, true = %9.6e "
                     "(requested = %9.6e)\n",
                     name, k, sqrt(r2 / b2), param.true_res, sqrt(r2_tol / b2));
        }
      } else {
	if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
          printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %9.6e "
                     "(requested = %9.6e), heavy-quark residual = %9.6e (requested = %9.6e)\n",
                     name, k, sqrt(r2 / b2), sqrt(r2_tol / b2), param.true_res_hq, hq_tol);
        } else {
          printfQuda("%s: Convergence at %d iterations, L2 relative residual: iterated = %9.6e (requested = %9.6e)\n",
                     name, k, sqrt(r2 / b2), sqrt(r2_tol / b2));
        }
      }
    }
  }

  double Solver::precisionEpsilon(QudaPrecision prec) const
  {
    double eps = 0.;
    if (prec == QUDA_INVALID_PRECISION) { prec = param.precision; }

    switch (prec) {
    case QUDA_DOUBLE_PRECISION: eps = std::numeric_limits<double>::epsilon() / 2.; break;
    case QUDA_SINGLE_PRECISION: eps = std::numeric_limits<float>::epsilon() / 2.; break;
    case QUDA_HALF_PRECISION: eps = pow(2., -13); break;
    case QUDA_QUARTER_PRECISION: eps = pow(2., -6); break;
    default: errorQuda("Invalid precision %d", param.precision); break;
    }
    return eps;
  }

  void MultiShiftSolver::create(const std::vector<ColorSpinorField> &x, const ColorSpinorField &b)
  {
    if (checkPrecision(x[0], b) != param.precision)
      errorQuda("Precision mismatch %d %d", checkPrecision(x[0], b), param.precision);
  }

  bool MultiShiftSolver::convergence(const std::vector<double> &r2, const std::vector<double> &r2_tol, int n) const
  {
    // check the L2 relative residual norm if necessary
    if ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) || (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) {
      for (int i = 0; i < n; i++) {
        if (std::isnan(r2[i]) || std::isinf(r2[i]))
          errorQuda("Multishift solver appears to have diverged on shift %d with residual %9.6e", i, r2[i]);

        if (r2[i] > r2_tol[i] && r2_tol[i] != 0.0) return false;
      }
    }

    return true;
  }

  /**
    @brief Returns if a solver is CA or not
    @return true if CA, false otherwise
  */
  bool is_ca_solver(QudaInverterType type)
  {
    switch (type) {
    case QUDA_CA_GCR_INVERTER:
    case QUDA_CA_CG_INVERTER:
    case QUDA_CA_CGNR_INVERTER:
    case QUDA_CA_CGNE_INVERTER: return true;
    default: return false;
    }
  }

} // namespace quda
