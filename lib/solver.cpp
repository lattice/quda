#include <quda_internal.h>
#include <invert_quda.h>
#include <multigrid.h>
#include <eigensolve_quda.h>
#include <accelerator.h>
#include <madwf_ml.h> // For MADWF
#include <cmath>
#include <limits>

namespace quda {

  static void report(const char *type) { logQuda(QUDA_VERBOSE, "Creating a %s solver\n", type); }

  Solver::Solver(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                 const DiracMatrix &matEig, SolverParam &param) :
    mat(mat),
    matSloppy(matSloppy),
    matPrecon(matPrecon),
    matEig(matEig),
    param(param),
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

  void Solver::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    if (checkPrecision(x, b) != param.precision)
      errorQuda("Precision mismatch %d %d", checkPrecision(x, b), param.precision);

    param.true_res.resize(b.size());
    param.true_res_hq.resize(b.size());
  }

  // solver factory
  Solver *Solver::create(SolverParam &param, const DiracMatrix &mat, const DiracMatrix &matSloppy,
                         const DiracMatrix &matPrecon, const DiracMatrix &matEig)
  {
    Solver *solver = nullptr;

    if (param.preconditioner && param.inv_type != QUDA_GCR_INVERTER && param.inv_type != QUDA_PCG_INVERTER)
      errorQuda("Explicit preconditoner not supported for %d solver", param.inv_type);

    if (param.preconditioner && param.inv_type_precondition != QUDA_MG_INVERTER)
      errorQuda("Explicit preconditoner not supported for %d preconditioner", param.inv_type_precondition);

    switch (param.inv_type) {
    case QUDA_CG_INVERTER:
      report("CG");
      solver = new CG(mat, matSloppy, matPrecon, matEig, param);
      break;
    case QUDA_BICGSTAB_INVERTER:
      report("BiCGstab");
      solver = new BiCGstab(mat, matSloppy, matPrecon, matEig, param);
      break;
    case QUDA_GCR_INVERTER:
      report("GCR");
      if (param.preconditioner) {
	Solver *mg = param.mg_instance ? static_cast<MG*>(param.preconditioner) : static_cast<multigrid_solver*>(param.preconditioner)->mg;
	// FIXME dirty hack to ensure that preconditioner precision set in interface isn't used in the outer GCR-MG solver
	if (!param.mg_instance) param.precision_precondition = param.precision_sloppy;
        solver = new GCR(mat, *(mg), matSloppy, matPrecon, matEig, param);
      } else {
        solver = new GCR(mat, matSloppy, matPrecon, matEig, param);
      }
      break;
    case QUDA_CA_CG_INVERTER:
      report("CA-CG");
      solver = new CACG(mat, matSloppy, matPrecon, matEig, param);
      break;
    case QUDA_CA_CGNE_INVERTER:
      report("CA-CGNE");
      solver = new CGNE(mat, matSloppy, matPrecon, matEig, param);
      break;
    case QUDA_CA_CGNR_INVERTER:
      report("CA-CGNR");
      solver = new CGNR(mat, matSloppy, matPrecon, matEig, param);
      break;
    case QUDA_CA_GCR_INVERTER:
      report("CA-GCR");
      solver = new CAGCR(mat, matSloppy, matPrecon, matEig, param);
      break;
    case QUDA_MR_INVERTER:
      report("MR");
      solver = new MR(mat, matSloppy, param);
      break;
    case QUDA_SD_INVERTER:
      report("SD");
      solver = new SD(mat, param);
      break;
    case QUDA_PCG_INVERTER:
      report("PCG");
      if (param.preconditioner) {
        Solver *mg = param.mg_instance ? static_cast<MG *>(param.preconditioner) :
                                         static_cast<multigrid_solver *>(param.preconditioner)->mg;
        // FIXME dirty hack to ensure that preconditioner precision set in interface isn't used in the outer GCR-MG solver
        if (!param.mg_instance) param.precision_precondition = param.precision_sloppy;
        solver = new PCG(mat, *(mg), matSloppy, matPrecon, matEig, param);
      } else {
        solver = new PCG(mat, matSloppy, matPrecon, matEig, param);
      }
      break;
    case QUDA_BICGSTABL_INVERTER:
      report("BICGSTABL");
      solver = new BiCGstabL(mat, matSloppy, matEig, param);
      break;
    case QUDA_EIGCG_INVERTER:
      report("EIGCG");
      solver = new IncEigCG(mat, matSloppy, matPrecon, param);
      break;
    case QUDA_INC_EIGCG_INVERTER:
      report("INC EIGCG");
      solver = new IncEigCG(mat, matSloppy, matPrecon, param);
      break;
    case QUDA_GMRESDR_INVERTER:
      report("GMRESDR");
      if (param.preconditioner) {
	multigrid_solver *mg = static_cast<multigrid_solver*>(param.preconditioner);
	// FIXME dirty hack to ensure that preconditioner precision set in interface isn't used in the outer GCR-MG solver
	param.precision_precondition = param.precision_sloppy;
        solver = new GMResDR(mat, *(mg->mg), matSloppy, matPrecon, param);
      } else {
        solver = new GMResDR(mat, matSloppy, matPrecon, param);
      }
      break;
    case QUDA_CGNE_INVERTER:
      report("CGNE");
      solver = new CGNE(mat, matSloppy, matPrecon, matEig, param);
      break;
    case QUDA_CGNR_INVERTER:
      report("CGNR");
      solver = new CGNR(mat, matSloppy, matPrecon, matEig, param);
      break;
    case QUDA_CG3_INVERTER:
      report("CG3");
      solver = new CG3(mat, matSloppy, matPrecon, param);
      break;
    case QUDA_CG3NE_INVERTER:
      report("CG3NE");
      solver = new CGNE(mat, matSloppy, matPrecon, matEig, param);
      break;
    case QUDA_CG3NR_INVERTER:
      report("CG3NR");
      solver = new CGNR(mat, matSloppy, matPrecon, matEig, param);
      break;
    default:
      errorQuda("Invalid solver type %d", param.inv_type);
    }

    if (!mat.hermitian() && solver->hermitian()) errorQuda("Cannot solve non-Hermitian system with Hermitian solver");
    return solver;
  }

  // preconditioner solver factory
  std::shared_ptr<Solver> Solver::createPreconditioner(const DiracMatrix &mat, const DiracMatrix &matSloppy,
                                                       const DiracMatrix &matPrecon, const DiracMatrix &matEig,
                                                       SolverParam &param, SolverParam &Kparam)
  {
    Solver *K = nullptr;
    if (param.accelerator_type_precondition == QUDA_MADWF_ACCELERATOR) {
      if (param.inv_type_precondition == QUDA_CG_INVERTER) {
        K = new AcceleratedSolver<MadwfAcc, CG>(mat, matSloppy, matPrecon, matEig, Kparam);
      } else if (param.inv_type_precondition == QUDA_CA_CG_INVERTER) {
        K = new AcceleratedSolver<MadwfAcc, CACG>(mat, matSloppy, matPrecon, matEig, Kparam);
      } else { // unknown preconditioner
        errorQuda("Unknown inner solver %d for MADWF", param.inv_type_precondition);
      }
    } else {
      if (param.inv_type_precondition == QUDA_CG_INVERTER) {
        K = new CG(mat, matSloppy, matPrecon, matEig, Kparam);
      } else if (param.inv_type_precondition == QUDA_CA_CG_INVERTER) {
        K = new CACG(mat, matSloppy, matPrecon, matEig, Kparam);
      } else if (param.inv_type_precondition == QUDA_MR_INVERTER) {
        K = new MR(mat, matSloppy, Kparam);
      } else if (param.inv_type_precondition == QUDA_SD_INVERTER) {
        K = new SD(mat, Kparam);
      } else if (param.inv_type_precondition == QUDA_CA_GCR_INVERTER) {
        K = new CAGCR(mat, matSloppy, matPrecon, matEig, Kparam);
      } else if (param.inv_type_precondition != QUDA_INVALID_INVERTER) { // unknown preconditioner
        errorQuda("Unknown inner solver %d", param.inv_type_precondition);
      }
    }
    return std::shared_ptr<Solver>(K);
  }

  // set the required parameters for the inner solver
  void Solver::fillInnerSolverParam(SolverParam &inner, const SolverParam &outer)
  {
    inner.tol = outer.tol_precondition;
    inner.delta = 1e-20; // no reliable updates within the inner solver

    // different solvers assume different behavior
    if (outer.inv_type == QUDA_GCR_INVERTER) {
      inner.precision = outer.precision_sloppy;
      inner.precision_sloppy = outer.precision_precondition;
    } else if (outer.inv_type == QUDA_PCG_INVERTER) {
      inner.precision = ((outer.inv_type_precondition == QUDA_CG_INVERTER
                          || outer.inv_type_precondition == QUDA_CA_CG_INVERTER || outer.inv_type == QUDA_MG_INVERTER)
                         && !outer.precondition_no_advanced_feature) ?
        outer.precision_sloppy :
        outer.precision_precondition;
      inner.precision_sloppy = outer.precision_precondition;
    } else {
      errorQuda("Unexpected preconditioned solver %d", outer.inv_type);
    }

    // allows the inner solver to early exit if it converges quickly
    inner.residual_type = QUDA_L2_RELATIVE_RESIDUAL;

    inner.iter = 0;
    inner.inv_type_precondition = QUDA_INVALID_INVERTER;
    inner.is_preconditioner = true; // tell inner solver it is a preconditioner
    inner.pipeline = true;

    inner.schwarz_type = outer.schwarz_type;
    inner.global_reduction = inner.schwarz_type == QUDA_INVALID_SCHWARZ ? true : false;

    inner.use_init_guess = QUDA_USE_INIT_GUESS_NO;

    inner.maxiter = outer.maxiter_precondition;
    if (outer.inv_type_precondition == QUDA_CA_GCR_INVERTER || outer.inv_type_precondition == QUDA_CA_CG_INVERTER) {
      inner.Nkrylov = inner.maxiter / outer.precondition_cycle;
      inner.ca_basis = outer.ca_basis_precondition;
      inner.ca_lambda_min = outer.ca_lambda_min_precondition;
      inner.ca_lambda_max = outer.ca_lambda_max_precondition;
    } else {
      inner.Nsteps = outer.precondition_cycle;
    }

    inner.verbosity_precondition = outer.verbosity_precondition;

    inner.compute_true_res = false;
    inner.sloppy_converge = true;
  }

  void Solver::extractInnerSolverParam(SolverParam &outer, const SolverParam &inner)
  {
    // extract a_max, which may have been determined via power iterations
    if ((outer.inv_type_precondition == QUDA_CA_CG_INVERTER || outer.inv_type_precondition == QUDA_CA_GCR_INVERTER)
        && outer.ca_basis_precondition == QUDA_CHEBYSHEV_BASIS) {
      outer.ca_lambda_max_precondition = inner.ca_lambda_max;
    }
  }

  // preconditioner solver wrapper
  std::shared_ptr<Solver> Solver::wrapExternalPreconditioner(const Solver &K)
  {
    return std::shared_ptr<Solver>(&const_cast<Solver &>(K), [](Solver *) {});
  }

  void Solver::constructDeflationSpace(const ColorSpinorField &meta, const DiracMatrix &mat)
  {
    if (deflate_init) return;

    // Deflation requested + first instance of solver
    if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_INIT);

    eig_solve = EigenSolver::create(&param.eig_param, mat);

    // Clone from an existing vector
    ColorSpinorParam csParam(meta);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    // This is the vector precision used by matEig
    csParam.setPrecision(param.precision_eigensolver, QUDA_INVALID_PRECISION, true);

    if (deflate_compute) {

      deflation_space *space = reinterpret_cast<deflation_space *>(param.eig_param.preserve_deflation_space);

      if (space && space->evecs.size() != 0) {
        logQuda(QUDA_VERBOSE, "Restoring deflation space of size %lu\n", space->evecs.size());

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

    if (!param.is_preconditioner) getProfile().TPSTOP(QUDA_PROFILE_INIT);
  }

  void Solver::destroyDeflationSpace()
  {
    getProfile().TPSTART(QUDA_PROFILE_FREE);

    if (deflate_init) {
      if (param.eig_param.preserve_deflation) {
        logQuda(QUDA_VERBOSE, "Preserving deflation space of size %lu\n", evecs.size());

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

    getProfile().TPSTOP(QUDA_PROFILE_FREE);
  }

  void Solver::injectDeflationSpace(std::vector<ColorSpinorField> &defl_space)
  {
    if (!evecs.empty()) errorQuda("Solver deflation space should be empty, instead size=%lu\n", evecs.size());
    evecs = std::move(defl_space); // move defl_space to evecs
    evals.resize(defl_space.size()); // create space for the eigenvalues
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
      param.true_res_offset[i] = static_cast<double>(param.true_res);
      param.true_res_hq_offset[i] = static_cast<double>(param.true_res_hq);
    }
  }

  vector<double> Solver::stopping(double tol, cvector<double> &b2, QudaResidualType residual_type)
  {
    vector<double> stop(b2.size(), 0.0);
    if ( (residual_type & QUDA_L2_ABSOLUTE_RESIDUAL) &&
	 (residual_type & QUDA_L2_RELATIVE_RESIDUAL) ) {
      for (auto i = 0u; i < b2.size(); i++) {
        // use the most stringent stopping condition
        double lowest = (b2[i] < 1.0) ? b2[i] : 1.0;
        stop[i] = lowest * tol * tol;
      }
    } else if (residual_type & QUDA_L2_ABSOLUTE_RESIDUAL) {
      for (auto i = 0u; i < b2.size(); i++) stop[i] = tol * tol;
    } else if (residual_type & QUDA_L2_RELATIVE_RESIDUAL) {
      for (auto i = 0u; i < b2.size(); i++) stop[i] = b2[i] * tol * tol;
    } else {
      // if invalid residual then convergence is set by iteration count only
      for (auto i = 0u; i < b2.size(); i++) stop[i] = 0.0;
    }

    return stop;
  }

  bool Solver::convergence(cvector<double> &r2, cvector<double> &hq2, cvector<double> &r2_tol, cvector<double> &hq_tol)
  {
    if (r2.size() != hq2.size() || r2.size() != r2_tol.size() || r2.size() != hq_tol.size())
      errorQuda("Mismatched vector lengths r2 = %lu hq2 = %lu r2_tol = %lu hq_tol = %lu", r2.size(), hq2.size(),
                r2_tol.size(), hq_tol.size());

    for (auto i = 0u; i < r2.size(); i++) {
      // check the heavy quark residual norm if necessary
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
        if (std::isnan(hq2[i]) || std::isinf(hq2[i]))
          errorQuda("Solver appears to have diverged with heavy quark residual %9.6e", hq2[i]);
        if (hq2[i] > hq_tol[i]) return false;
      }

      // check the L2 relative residual norm if necessary
      if ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) || (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) {
        if (std::isnan(r2[i]) || std::isinf(r2[i]))
          errorQuda("Solver appears to have diverged with residual %9.6e", r2[i]);
        if (r2[i] > r2_tol[i]) return false;
      }
    }
    return true;
  }

  bool Solver::convergenceHQ(cvector<double> &hq2, cvector<double> &hq_tol)
  {
    if (hq2.size() != hq_tol.size())
      errorQuda("Mismatched vector lengths hq2 = %lu hq_tol = %lu", hq2.size(), hq_tol.size());

    for (auto i = 0u; i < hq2.size(); i++) {
      // check the heavy quark residual norm if necessary
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
        if (std::isnan(hq2[i]) || std::isinf(hq2[i]))
          errorQuda("Solver appears to have diverged with heavy quark residual %9.6e", hq2[i]);
        if (hq2[i] > hq_tol[i]) return false;
      }
    }
    return true;
  }

  bool Solver::convergenceL2(cvector<double> &r2, cvector<double> &r2_tol)
  {
    if (r2.size() != r2_tol.size())
      errorQuda("Mismatched vector lengths r2 = %lu r2_tol = %lu", r2.size(), r2_tol.size());

    for (auto i = 0u; i < r2.size(); i++) {
      // check the L2 relative residual norm if necessary
      if ((param.residual_type & QUDA_L2_RELATIVE_RESIDUAL) || (param.residual_type & QUDA_L2_ABSOLUTE_RESIDUAL)) {
        if (std::isnan(r2[i]) || std::isinf(r2[i]))
          errorQuda("Solver appears to have diverged with residual %9.6e", r2[i]);
        if (r2[i] > r2_tol[i]) return false;
      }
    }
    return true;
  }

  std::string set_rhs_str(unsigned int i, size_t n)
  {
    std::string rhs_str;
    if (n > 1) rhs_str += "n = " + std::to_string(i) + std::string(", ");
    return rhs_str;
  }

  void Solver::PrintStats(const char *name, int k, cvector<double> &r2, cvector<double> &b2, cvector<double> &hq2_)
  {
    auto hq2 = hq2_.size() == 0 ? vector<double>(r2.size(), 0.0) : hq2_;
    if (r2.size() != b2.size() || r2.size() != hq2.size())
      errorQuda("Mismatched vector lengths r2 = %lu b2 = %lu hq2 = %lu", r2.size(), b2.size(), hq2.size());

    for (auto i = 0u; i < r2.size(); i++) {
      auto rhs_str = set_rhs_str(i, r2.size());
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
        logQuda(QUDA_VERBOSE, "%s: %5d iterations, %s<r,r> = %9.6e, |r|/|b| = %9.6e, heavy-quark residual = %9.6e\n",
                name, k, rhs_str.c_str(), r2[i], sqrt(r2[i] / b2[i]), hq2[i]);
      } else {
        logQuda(QUDA_VERBOSE, "%s: %5d iterations, %s<r,r> = %9.6e, |r|/|b| = %9.6e\n", name, k, rhs_str.c_str(), r2[i],
                sqrt(r2[i] / b2[i]));
      }

      if (std::isnan(r2[i]) || std::isinf(r2[i])) errorQuda("Solver appears to have diverged for n = %d", i);
    }
  }

  void Solver::PrintSummary(const char *name, int k, cvector<double> &r2, cvector<double> &b2, cvector<double> &r2_tol,
                            cvector<double> &hq_tol_)
  {
    auto hq_tol = hq_tol_.size() == 0 ? vector<double>(r2.size(), 0.0) : hq_tol_;
    if (r2.size() != b2.size() || r2.size() != r2_tol.size() || r2.size() != hq_tol.size())
      errorQuda("Mismatched vector lengths r2 = %lu b2 = %lu r2_tol = %lu hq_tol = %lu", r2.size(), b2.size(),
                r2_tol.size(), hq_tol.size());

    for (auto i = 0u; i < r2.size(); i++) {
      auto rhs_str = set_rhs_str(i, r2.size());
      if (param.compute_true_res) {
        if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
          logQuda(QUDA_SUMMARIZE,
                  "%s: Convergence at %d iterations, %sL2 relative residual: iterated = %9.6e, true = %9.6e "
                  "(requested = %9.6e), heavy-quark residual = %9.6e (requested = %9.6e)\n",
                  name, k, rhs_str.c_str(), sqrt(r2[i] / b2[i]), param.true_res[i], sqrt(r2_tol[i] / b2[i]),
                  param.true_res_hq[i], hq_tol[i]);
        } else {
          logQuda(QUDA_SUMMARIZE,
                  "%s: Convergence at %d iterations, %sL2 relative residual: iterated = %9.6e, true = %9.6e "
                  "(requested = %9.6e)\n",
                  name, k, rhs_str.c_str(), sqrt(r2[i] / b2[i]), param.true_res[i], sqrt(r2_tol[i] / b2[i]));
        }
      } else {
        if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
          logQuda(QUDA_SUMMARIZE,
                  "%s: Convergence at %d iterations, %sL2 relative residual: iterated = %9.6e "
                  "(requested = %9.6e), heavy-quark residual = %9.6e (requested = %9.6e)\n",
                  name, k, rhs_str.c_str(), sqrt(r2[i] / b2[i]), sqrt(r2_tol[i] / b2[i]), param.true_res_hq[i],
                  hq_tol[i]);
        } else {
          logQuda(QUDA_SUMMARIZE,
                  "%s: Convergence at %d iterations, %sL2 relative residual: iterated = %9.6e (requested = %9.6e)\n",
                  name, k, rhs_str.c_str(), sqrt(r2[i] / b2[i]), sqrt(r2_tol[i] / b2[i]));
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
    case QUDA_HALF_PRECISION: eps = std::pow(2., -13); break;
    case QUDA_QUARTER_PRECISION: eps = std::pow(2., -6); break;
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

  // check we're not solving on a zero-valued source
  bool Solver::is_zero_src(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b, cvector<double> &b2)
  {
    // if computing null vectors then zero sources are fine
    if (param.compute_null_vector != QUDA_COMPUTE_NULL_VECTOR_NO) return false;

    bool zero_src = true;
    for (auto i = 0u; i < b.size(); i++) {
      if (b2[i] == 0) {
        warningQuda("source %d is zero", i);
        x[i] = b[i];
        param.true_res[i] = 0.0;
        param.true_res_hq[i] = 0.0;
      } else {
        zero_src = false;
      }
    }
    return zero_src;
  }

  void SolverParam::updateInvertParam(QudaInvertParam &param, int offset)
  {
    for (auto i = 0u; i < true_res.size(); i++) param.true_res[i] = true_res[i];
    for (auto i = 0u; i < true_res_hq.size(); i++) param.true_res_hq[i] = true_res_hq[i];
    param.iter += iter;
    if (offset >= 0) {
      param.true_res_offset[offset] = true_res_offset[offset];
      param.iter_res_offset[offset] = iter_res_offset[offset];
      param.true_res_hq_offset[offset] = true_res_hq_offset[offset];
    } else {
      for (int i = 0; i < num_offset; i++) {
        param.true_res_offset[i] = true_res_offset[i];
        param.iter_res_offset[i] = iter_res_offset[i];
        param.true_res_hq_offset[i] = true_res_hq_offset[i];
      }
    }
    // for incremental eigCG:
    param.rhs_idx = rhs_idx;

    param.ca_lambda_min = ca_lambda_min;
    param.ca_lambda_max = ca_lambda_max;

    param.ca_lambda_min_precondition = ca_lambda_min_precondition;
    param.ca_lambda_max_precondition = ca_lambda_max_precondition;

    if (deflate) *static_cast<QudaEigParam *>(param.eig_param) = eig_param;
  }

  void joinInvertParam(QudaInvertParam &out, const QudaInvertParam &in, const CommKey &split_key, int split_rank)
  {
    auto num_sub_partition = quda::product(split_key);

    int sub_partition_dims[]
      = {comm_dim(0) / split_key[0], comm_dim(1) / split_key[1], comm_dim(2) / split_key[2], comm_dim(3) / split_key[3]};

    int sub_partition_coords[] = {comm_coord(0) / sub_partition_dims[0], comm_coord(1) / sub_partition_dims[1],
                                  comm_coord(2) / sub_partition_dims[2], comm_coord(3) / sub_partition_dims[3]};

    auto j = sub_partition_coords[3];
    for (auto d = 2; d >= 0; d--) j = j * split_key[d] + sub_partition_coords[d];

    std::vector<double> true_res(in.num_src, 0.0);
    std::vector<double> true_res_hq(in.num_src, 0.0);
    if (split_rank == 0) { // only rank 0 in each sub partition sets the residuals
      for (auto i = 0; i < in.num_src_per_sub_partition; i++) {
        true_res[i * num_sub_partition + j] = in.true_res[i];
        true_res_hq[i * num_sub_partition + j] = in.true_res_hq[i];
      }
    }

    // communicate to all ranks
    comm_allreduce_sum(true_res_hq);
    comm_allreduce_sum(true_res);
    memcpy(out.true_res, true_res.data(), true_res.size() * sizeof(double));
    memcpy(out.true_res_hq, true_res_hq.data(), true_res_hq.size() * sizeof(double));

    out.iter = in.iter;
    comm_allreduce_int(out.iter);

    out.ca_lambda_min = in.ca_lambda_min;
    out.ca_lambda_max = in.ca_lambda_max;
    out.ca_lambda_min_precondition = in.ca_lambda_min_precondition;
    out.ca_lambda_max_precondition = in.ca_lambda_max_precondition;

    // now broadcast from global rank 0 to ensure uniformity
    comm_broadcast(&out, sizeof(QudaInvertParam));
  }

} // namespace quda
