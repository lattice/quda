#include "invert_quda.h"

namespace quda
{

  // vector of spinors used for forecasting solutions in HMC
#define QUDA_MAX_CHRONO 12
  // each entry is one p
  std::vector<std::vector<ColorSpinorField>> chronoResident(QUDA_MAX_CHRONO);

  void flushChrono(int i)
  {
    if (i >= QUDA_MAX_CHRONO) errorQuda("Requested chrono index %d is outside of max %d", i, QUDA_MAX_CHRONO);

    if (i >= 0)
      chronoResident[i].clear();
    else
      for (auto i = 0; i < QUDA_MAX_CHRONO; i++) chronoResident[i].clear();
  }

  void massRescale(cvector_ref<ColorSpinorField> &b, QudaInvertParam &param, bool for_multishift)
  {
    double kappa5 = (0.5 / (5.0 + param.m5));
    double kappa = (param.dslash_type == QUDA_DOMAIN_WALL_DSLASH || param.dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
                    || param.dslash_type == QUDA_MOBIUS_DWF_DSLASH || param.dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) ?
      kappa5 :
      param.kappa;

    logQuda(QUDA_DEBUG_VERBOSE, "Mass rescale: Kappa is: %g\n", kappa);
    logQuda(QUDA_DEBUG_VERBOSE, "Mass rescale: mass normalization: %d\n", param.mass_normalization);
    if (getVerbosity() > QUDA_DEBUG_VERBOSE) {
      auto b2 = blas::norm2(b);
      for (auto &b2i : b2) printfQuda("Mass rescale: norm of source in = %g\n", b2i);
    }

    // staggered dslash uses mass normalization internally
    if (param.dslash_type == QUDA_ASQTAD_DSLASH || param.dslash_type == QUDA_STAGGERED_DSLASH) {
      switch (param.solution_type) {
      case QUDA_MAT_SOLUTION:
      case QUDA_MATPC_SOLUTION:
        if (param.mass_normalization == QUDA_KAPPA_NORMALIZATION) blas::ax(2.0 * param.mass, b);
        break;
      case QUDA_MATDAG_MAT_SOLUTION:
      case QUDA_MATPCDAG_MATPC_SOLUTION:
        if (param.mass_normalization == QUDA_KAPPA_NORMALIZATION) blas::ax(4.0 * param.mass * param.mass, b);
        break;
      default: errorQuda("Not implemented");
      }
      return;
    }

    // multiply the source to compensate for normalization of the Dirac operator, if necessary
    // you are responsible for restoring what's in param.offset
    switch (param.solution_type) {
    case QUDA_MAT_SOLUTION:
      if (param.mass_normalization == QUDA_MASS_NORMALIZATION
          || param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
        blas::ax(2.0 * kappa, b);
        if (for_multishift)
          for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 2.0 * kappa;
      }
      break;
    case QUDA_MATDAG_MAT_SOLUTION:
      if (param.mass_normalization == QUDA_MASS_NORMALIZATION
          || param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
        blas::ax(4.0 * kappa * kappa, b);
        if (for_multishift)
          for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 4.0 * kappa * kappa;
      }
      break;
    case QUDA_MATPC_SOLUTION:
      if (param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        blas::ax(4.0 * kappa * kappa, b);
        if (for_multishift)
          for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 4.0 * kappa * kappa;
      } else if (param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
        blas::ax(2.0 * kappa, b);
        if (for_multishift)
          for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 2.0 * kappa;
      }
      break;
    case QUDA_MATPCDAG_MATPC_SOLUTION:
      if (param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        blas::ax(16.0 * std::pow(kappa, 4), b);
        if (for_multishift)
          for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 16.0 * std::pow(kappa, 4);
      } else if (param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
        blas::ax(4.0 * kappa * kappa, b);
        if (for_multishift)
          for (int i = 0; i < param.num_offset; i++) param.offset[i] *= 4.0 * kappa * kappa;
      }
      break;
    default: errorQuda("Solution type %d not supported", param.solution_type);
    }

    if (getVerbosity() > QUDA_DEBUG_VERBOSE) {
      auto b2 = blas::norm2(b);
      for (auto &b2i : b2) printfQuda("Mass rescale: norm of source out = %g\n", b2i);
    }
  }

  void distanceReweight(cvector_ref<ColorSpinorField> &b, QudaInvertParam &param, bool inverse)
  {
    // Force the alpha0 to be positive.
    // A negative alpha0 matches something like Eq.(12) in arXiv:1006.4028.
    // Disable the negative situation as QUDA already has multigrid for light quarks.
    const double alpha0 = abs(param.distance_pc_alpha0);
    const int t0 = param.distance_pc_t0;
    if (alpha0 != 0.0 && t0 >= 0) {
      if (param.dslash_type != QUDA_WILSON_DSLASH && param.dslash_type != QUDA_CLOVER_WILSON_DSLASH) {
        errorQuda("Only Wilson and Wilson-clover dslash support distance preconditioning, but get dslash_type %d\n",
                  param.dslash_type);
      }
      if (param.inv_type == QUDA_MG_INVERTER) errorQuda("Multigrid solver doesn't support distance preconditioning");

      if (param.cuda_prec != QUDA_DOUBLE_PRECISION || param.cuda_prec_sloppy != QUDA_DOUBLE_PRECISION) {
        warningQuda(
          "Using single or half (sloppy) precision in distance preconditioning sometimes makes the solver diverge");
      }

      if (inverse)
        for (auto i = 0u; i < b.size(); i++) spinorDistanceReweight(b[i], -alpha0, t0);
      else
        for (auto i = 0u; i < b.size(); i++) spinorDistanceReweight(b[i], alpha0, t0);
    }
  }

  void solve(cvector_ref<ColorSpinorField> &x, cvector_ref<ColorSpinorField> &b, Dirac &dirac, Dirac &diracSloppy,
             Dirac &diracPre, Dirac &diracEig, QudaInvertParam &param)
  {
    getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);

    bool mat_solution = (param.solution_type == QUDA_MAT_SOLUTION) || (param.solution_type == QUDA_MATPC_SOLUTION);
    bool direct_solve = (param.solve_type == QUDA_DIRECT_SOLVE) || (param.solve_type == QUDA_DIRECT_PC_SOLVE);
    bool norm_error_solve = (param.solve_type == QUDA_NORMERR_SOLVE) || (param.solve_type == QUDA_NORMERR_PC_SOLVE);

    auto nb = blas::norm2(b);
    for (auto &bi : nb) {
      if (bi == 0.0) errorQuda("Source has zero norm");
      logQuda(QUDA_VERBOSE, "Source: %g\n", bi);
    }
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      auto x_norm = blas::norm2(x);
      for (auto &xi : x_norm) logQuda(QUDA_VERBOSE, "Initial guess: %g\n", xi);
    }
    // rescale the source and solution vectors to help prevent the onset of underflow
    if (param.solver_normalization == QUDA_SOURCE_NORMALIZATION) {
      auto nb_inv(nb);
      for (auto &bi : nb_inv) bi = 1 / sqrt(bi);
      blas::ax(nb_inv, b);
      blas::ax(nb_inv, x);
    }

    massRescale(b, param, false);
    distanceReweight(b, param, true);

    std::vector<ColorSpinorField> in(b.size());
    std::vector<ColorSpinorField> out(b.size());

    // if we're doing a managed memory MG solve and prefetching is
    // enabled, prefetch all the Dirac matrices. There's probably
    // a better place to put this...
    if (param.inv_type_precondition == QUDA_MG_INVERTER) {
      dirac.prefetch(QUDA_CUDA_FIELD_LOCATION);
      diracSloppy.prefetch(QUDA_CUDA_FIELD_LOCATION);
      diracPre.prefetch(QUDA_CUDA_FIELD_LOCATION);
    }

    dirac.prepare(out, in, x, b, param.solution_type);

    if (getVerbosity() >= QUDA_VERBOSE) {
      auto in_norm = blas::norm2(in);
      auto out_norm = blas::norm2(out);
      for (auto i = 0u; i < in.size(); i++)
        logQuda(QUDA_VERBOSE, "Prepared: source = %g, solution = %g\n", in_norm[i], out_norm[i]);
    }

    // solution_type specifies *what* system is to be solved.
    // solve_type specifies *how* the system is to be solved.
    //
    // We have the following four cases (plus preconditioned variants):
    //
    // solution_type    solve_type    Effect
    // -------------    ----------    ------
    // MAT              DIRECT        Solve Ax=b
    // MATDAG_MAT       DIRECT        Solve A^dag y = b, followed by Ax=y
    // MAT              NORMOP        Solve (A^dag A) x = (A^dag b)
    // MATDAG_MAT       NORMOP        Solve (A^dag A) x = b
    // MAT              NORMERR       Solve (A A^dag) y = b, then x = A^dag y
    //
    // We generally require that the solution_type and solve_type
    // preconditioning match.  As an exception, the unpreconditioned MAT
    // solution_type may be used with any solve_type, including
    // DIRECT_PC and NORMOP_PC.  In these cases, preparation of the
    // preconditioned source and reconstruction of the full solution are
    // taken care of by Dirac::prepare() and Dirac::reconstruct(),
    // respectively.

    getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);

    if (mat_solution && !direct_solve && !norm_error_solve) { // prepare source: b' = A^dag b
      auto tmp = getFieldTmp(cvector_ref<ColorSpinorField>(in));
      blas::copy(tmp, in);
      dirac.Mdag(in, tmp);
    } else if (!mat_solution && direct_solve) { // perform the first of two solves: A^dag y = b
      DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre), mEig(diracEig);
      SolverParam solverParam(param);
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, mEig);
      (*solve)(out, in);
      blas::copy(in, out);
      delete solve;
      solverParam.updateInvertParam(param);
    }

    if (direct_solve) {
      DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre), mEig(diracEig);
      SolverParam solverParam(param);

      // chronological forecasting
      if (param.chrono_use_resident && chronoResident[param.chrono_index].size() > 0) {
        bool hermitian = false;
        auto &mChrono = param.chrono_precision == param.cuda_prec ? m : mSloppy;
        chronoExtrapolate(out[0], in[0], chronoResident[param.chrono_index], mChrono, hermitian);
      }

      Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, mEig);
      (*solve)(out, in);
      delete solve;
      solverParam.updateInvertParam(param);
    } else if (!norm_error_solve) {
      DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre), mEig(diracEig);
      SolverParam solverParam(param);

      // chronological forecasting
      if (param.chrono_use_resident && chronoResident[param.chrono_index].size() > 0) {
        bool hermitian = true;
        auto &mChrono = param.chrono_precision == param.cuda_prec ? m : mSloppy;
        chronoExtrapolate(out[0], in[0], chronoResident[param.chrono_index], mChrono, hermitian);
      }

      // if using a Schwarz preconditioner with a normal operator then we must use the DiracMdagMLocal operator
      if (param.inv_type_precondition != QUDA_INVALID_INVERTER && param.schwarz_type != QUDA_INVALID_SCHWARZ) {
        DiracMdagMLocal mPreLocal(diracPre);
        Solver *solve = Solver::create(solverParam, m, mSloppy, mPreLocal, mEig);
        (*solve)(out, in);
        delete solve;
        solverParam.updateInvertParam(param);
      } else {
        Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, mEig);
        (*solve)(out, in);
        delete solve;
        solverParam.updateInvertParam(param);
      }
    } else { // norm_error_solve
      DiracMMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre), mEig(diracEig);
      auto tmp = getFieldTmp(cvector_ref<ColorSpinorField>(in));
      SolverParam solverParam(param);
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, mEig);
      (*solve)(tmp, in);    // y = (M M^\dag) b
      dirac.Mdag(out, tmp); // x = M^dag y
      delete solve;
      solverParam.updateInvertParam(param);
    }

    if (getVerbosity() >= QUDA_VERBOSE) {
      auto x_norm = blas::norm2(out);
      for (auto i = 0u; i < x.size(); i++) printfQuda("Solution = %g\n", x_norm[i]);
    }

    getProfile().TPSTART(QUDA_PROFILE_EPILOGUE);
    if (param.chrono_make_resident) {
      const int i = param.chrono_index;
      if (i >= QUDA_MAX_CHRONO) errorQuda("Requested chrono index %d is outside of max %d\n", i, QUDA_MAX_CHRONO);

      auto &basis = chronoResident[i];

      if (param.chrono_max_dim < (int)basis.size()) {
        errorQuda("Requested chrono_max_dim %i is smaller than already existing chronology %lu", param.chrono_max_dim,
                  basis.size());
      }

      if (not param.chrono_replace_last) {
        // if we have not filled the space yet just augment
        if ((int)basis.size() < param.chrono_max_dim) {
          ColorSpinorParam cs_param(out[0]);
          cs_param.setPrecision(param.chrono_precision);
          basis.emplace_back(cs_param);
        }

        // shuffle every entry down one and bring the last to the front
        std::rotate(basis.begin(), basis.end() - 1, basis.end());
      }
      basis[0] = out[0]; // set first entry to new solution
    }

    dirac.reconstruct(x, b, param.solution_type);

    distanceReweight(x, param, false);

    if (param.solver_normalization == QUDA_SOURCE_NORMALIZATION) {
      // rescale the solution
      for (auto &bi : nb) bi = sqrt(bi);
      blas::ax(nb, x);
    }

    if (getVerbosity() >= QUDA_VERBOSE) {
      auto x_norm = blas::norm2(x);
      for (auto i = 0u; i < x.size(); i++) printfQuda("Reconstructed Solution = %g\n", x_norm[i]);
    }

    if (param.compute_action) {
      auto action = blas::cDotProduct(b, x);
      param.action[0] = action[0].real();
      param.action[1] = action[0].imag();
    }

    getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);
  }

  void createDiracWithEig(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, Dirac *&dEig, QudaInvertParam &param, bool pc_solve,
                          bool use_smeared_gauge);

  extern std::vector<ColorSpinorField> solutionResident;

  void solve(const std::vector<void *> &hp_x, const std::vector<void *> &hp_b, QudaInvertParam &param,
             const GaugeField &u)
  {
    pushVerbosity(param.verbosity);

    if (hp_b.size() != hp_x.size())
      errorQuda("Number of solutions %lu != number of solves %lu", hp_x.size(), hp_b.size());
    int n_src = hp_b.size();

    // It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
    // solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
    // for now, though, so here we factorize everything for convenience.

    bool pc_solution
      = (param.solution_type == QUDA_MATPC_SOLUTION) || (param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
    bool pc_solve = (param.solve_type == QUDA_DIRECT_PC_SOLVE) || (param.solve_type == QUDA_NORMOP_PC_SOLVE)
      || (param.solve_type == QUDA_NORMERR_PC_SOLVE);

    param.iter = 0;

    Dirac *dirac = nullptr;
    Dirac *diracSloppy = nullptr;
    Dirac *diracPre = nullptr;
    Dirac *diracEig = nullptr;

    // Create the dirac operator and operators for sloppy, precondition,
    // and an eigensolver
    createDiracWithEig(dirac, diracSloppy, diracPre, diracEig, param, pc_solve,
                       param.eig_param ? static_cast<QudaEigParam *>(param.eig_param)->use_smeared_gauge : false);

    // wrap CPU host side pointers
    ColorSpinorParam cpuParam(hp_b[0], param, u.X(), pc_solution, param.input_location);
    std::vector<ColorSpinorField> h_b(n_src);
    for (auto i = 0u; i < h_b.size(); i++) {
      cpuParam.v = hp_b[i];
      h_b[i] = ColorSpinorField(cpuParam);
    }

    std::vector<ColorSpinorField> h_x(n_src);
    cpuParam.location = param.output_location;
    for (auto i = 0u; i < h_x.size(); i++) {
      cpuParam.v = hp_x[i];
      h_x[i] = ColorSpinorField(cpuParam);
    }

    // download source
    ColorSpinorParam cudaParam(cpuParam, param, QUDA_CUDA_FIELD_LOCATION);
    cudaParam.create = QUDA_NULL_FIELD_CREATE;
    std::vector<ColorSpinorField> b;
    resize(b, n_src, cudaParam);
    blas::copy(b, h_b);

    // now check if we need to invalidate the solutionResident vectors
    std::vector<ColorSpinorField> x;
    resize(x, n_src, cudaParam);
    if (param.use_resident_solution == 1) {
      for (auto &v : solutionResident) {
        if (b[0].Precision() != v.Precision() || b[0].SiteSubset() != v.SiteSubset()) {
          solutionResident.clear();
          break;
        }
      }

      if (!solutionResident.size()) {
        cudaParam.create = QUDA_NULL_FIELD_CREATE;
        solutionResident = std::vector<ColorSpinorField>(1, cudaParam);
      }
      x[0] = solutionResident[0].create_alias(cudaParam);
    } else {
      cudaParam.create = QUDA_NULL_FIELD_CREATE;
      x[0] = ColorSpinorField(cudaParam);
    }

    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES && !param.chrono_use_resident) { // download initial guess
      // initial guess only supported for single-pass solvers
      if ((param.solution_type == QUDA_MATDAG_MAT_SOLUTION || param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION)
          && (param.solve_type == QUDA_DIRECT_SOLVE || param.solve_type == QUDA_DIRECT_PC_SOLVE)) {
        errorQuda("Initial guess not supported for two-pass solver");
      }

      blas::copy(x, h_x); // solution
    } else {              // zero initial guess
      blas::zero(x);
    }

    solve(x, b, *dirac, *diracSloppy, *diracPre, *diracEig, param);

    if (!param.make_resident_solution) blas::copy(h_x, x);

    delete dirac;
    delete diracSloppy;
    delete diracPre;
    delete diracEig;

    popVerbosity();
  }

} // namespace quda
