#include <cstring>

#include "multigrid.h"
#include "tune_quda.h"
#include "random_quda.h"
#include "vector_io.h"

// for dwf verify
#include "dslash_quda.h"

// for building the KD inverse op
#include "staggered_kd_build_xinv.h"

namespace quda
{

  using namespace blas;

  MG::MG(MGParam &param) :
    Solver(*param.matResidual, *param.matSmooth, *param.matSmoothSloppy, *param.matSmoothSloppy, param),
    param(param),
    xInvKD(nullptr),
    xInvKD_sloppy(nullptr),
    diracResidual(param.matResidual->Expose()),
    diracSmoother(param.matSmooth->Expose()),
    diracSmootherSloppy(param.matSmoothSloppy->Expose()),
    diracNull(param.matNull->Expose()),
    diracNullSloppy(param.matNullSloppy->Expose())
  {
    sprintf(prefix, "MG level %d (%s): ", param.level, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");
    pushLevel(param.level);

    if (param.level >= QUDA_MAX_MG_LEVEL)
      errorQuda("Level=%d is greater than limit of multigrid recursion depth", param.level);

    if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type != QUDA_DIRECT_PC_SOLVE)
      errorQuda("Cannot use preconditioned coarse grid solution without preconditioned smoother solve");

    // for 4-d-aggregated dwf MG, we need to be careful about the size of RNG
    if (diracResidual->getLs() == 1) {
      rng = new RNG(param.B[0], 1234);
    } else {
      // we need to create a dummy larger field
      ColorSpinorParam csParam(param.B[0]);
      csParam.nDim = 5;
      csParam.x[4] = diracResidual->getLs();
      csParam.create = QUDA_REFERENCE_FIELD_CREATE; // just create a metadata "container"
      csParam.v = nullptr;
      ColorSpinorField dummy(csParam);
      rng = new RNG(dummy, 1234);
    }

    if (param.transfer_type == QUDA_TRANSFER_AGGREGATE && param.level < param.Nlevel - 1) {
      if (param.B[0].Ndim() == 5 && param.level == 0)
        errorQuda("DWF does not support traditional aggregation, use 4-d aggregation");
      else
        createNullVectors();
    }

    // in case of iterative setup with MG the coarse level may be already built
    if (!transfer) reset();

    popLevel();
  }

  void MG::reset(bool refresh) {
    pushLevel(param.level);

    logQuda(QUDA_VERBOSE, "%s level %d\n", transfer ? "Resetting" : "Creating", param.level);

    destroySmoother();
    destroyCoarseSolver();

    // reset the Dirac operator pointers since these may have changed
    diracResidual = param.matResidual->Expose();
    diracSmoother = param.matSmooth->Expose();
    diracSmootherSloppy = param.matSmoothSloppy->Expose();
    diracNull = param.matNull->Expose();
    diracNullSloppy = param.matNullSloppy->Expose();

    // Only refresh if we needed to generate near-nulls, that is,
    // if we aren't doing a staggered KD solve
    if (param.level != 0 || param.transfer_type == QUDA_TRANSFER_AGGREGATE) {
      // Refresh the null-space vectors if we need to
      if (refresh && param.level < param.Nlevel - 1) {
        if (param.mg_global.setup_maxiter_refresh[param.level]) generateNullVectors(param.B, refresh);
      }
    }

    // if not on the coarsest level, update next
    if (param.level < param.Nlevel-1) {

      if (transfer) {
        // restoring FULL parity in Transfer changed at the end of this procedure
        transfer->setSiteSubset(QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY);
        if (resetTransfer || refresh) {
          transfer->reset();
          resetTransfer = false;
        }
      } else {
        // create transfer operator
        logQuda(QUDA_VERBOSE, "Creating transfer operator %s\n", param.transfer_use_mma ? "with MMA enabled" : "");
        transfer = new Transfer(param.B, param.Nvec, param.NblockOrtho, param.blockOrthoTwoPass, param.geoBlockSize,
                                param.spinBlockSize, param.mg_global.precision_null[param.level],
                                param.mg_global.transfer_type[param.level]);
        transfer->set_use_mma(param.transfer_use_mma);
        for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++)
          param.mg_global.geo_block_size[param.level][i] = param.geoBlockSize[i];

        auto customLs = is_pv() ? reinterpret_cast<const DiracDomainWall *>(diracSmoother)->getLs() : -1;

        // create coarse residual vector if not already created in verify()
        if (r_coarse.empty()) {
          r_coarse.resize(1);
          r_coarse[0] = param.B[0].create_coarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, customLs,
                                                 param.mg_global.invert_param->cuda_prec_sloppy,
                                                 param.mg_global.location[param.level + 1]);
          if (is_pv() && param.level == 0) r_coarse[0].GammaBasis(QUDA_UKQCD_GAMMA_BASIS);
        }

        // create coarse solution vector if not already created in verify()
        if (x_coarse.empty()) {
          x_coarse.resize(1);
          x_coarse[0] = param.B[0].create_coarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, customLs,
                                                 param.mg_global.invert_param->cuda_prec_sloppy,
                                                 param.mg_global.location[param.level + 1]);
          if (is_pv() && param.level == 0) x_coarse[0].GammaBasis(QUDA_UKQCD_GAMMA_BASIS);
        }

        int nVec_coarse = std::max(param.Nvec, param.mg_global.n_vec[param.level + 1]);
        B_coarse.resize(nVec_coarse);

        // only have single precision B vectors on the coarse grid
        QudaPrecision B_coarse_precision = std::max(param.mg_global.precision_null[param.level+1], QUDA_SINGLE_PRECISION);

        // the -1 is to preserve the dimensionality of the near-null vectors
        for (int i = 0; i < nVec_coarse; i++)
          B_coarse[i] = param.B[0].create_coarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, is_pv() ? 1 : -1,
                                                 B_coarse_precision, param.mg_global.setup_location[param.level + 1]);

        // if we're not generating on all levels then we need to propagate the vectors down
        if ((param.level != 0 || param.Nlevel - 1) && param.mg_global.generate_all_levels == QUDA_BOOLEAN_FALSE) {
          logQuda(QUDA_VERBOSE, "Restricting null space vectors\n");
          for (int i = 0; i < param.Nvec; i++) {
            zero(B_coarse[i]);
            transfer->R(B_coarse[i], param.B[i]);
          }
        }
        logQuda(QUDA_VERBOSE, "Transfer operator done\n");
      }

      // we no longer need the B fields for this level, can evict them to host memory
      // (only if using managed memory and prefetching is enabled, otherwise no-op)
      for (int i = 0; i < param.Nvec; i++) { param.B[i].prefetch(QUDA_CPU_FIELD_LOCATION); }

      buildNextDirac();
    }

    // delay allocating smoother until after coarse-links have been created
    createSmoother();

    if (param.level < param.Nlevel-1) {
      // creating or resetting the coarse level temporaries and solvers
      if (coarse) {
        coarse->param.updateInvertParam(*param.mg_global.invert_param);
        coarse->param.delta = 1e-20;
        coarse->param.precision = param.mg_global.invert_param->cuda_prec_precondition;
        coarse->param.matResidual = matCoarseResidual;
        coarse->param.matSmooth = matCoarseSmoother;
        coarse->param.matSmoothSloppy = matCoarseSmootherSloppy;
        coarse->param.matSmooth = matCoarseNull;
        coarse->param.matSmoothSloppy = matCoarseNullSloppy;
        coarse->reset(refresh);
      } else {
        // create the next multigrid level
        param_coarse = new MGParam(param, B_coarse, matCoarseResidual, matCoarseSmoother, matCoarseSmootherSloppy,
                                   matCoarseNull, matCoarseNullSloppy, param.level + 1);
        param_coarse->fine = this;
        param_coarse->delta = 1e-20;
        param_coarse->precision = param.mg_global.invert_param->cuda_prec_precondition;

        coarse = new MG(*param_coarse);
      }
      setOutputPrefix(prefix); // restore since we just popped back from coarse grid

      createCoarseSolver();

      // If enabled, verify the coarse links and fine solvers were correctly built
      if (param.mg_global.run_verify) verify();
    }

    // We're going back up the coarse construct stack now, prefetch the gauge fields on
    // this level back to device memory.
    diracResidual->prefetch(QUDA_CUDA_FIELD_LOCATION);
    diracSmoother->prefetch(QUDA_CUDA_FIELD_LOCATION);
    diracSmootherSloppy->prefetch(QUDA_CUDA_FIELD_LOCATION);
    diracNull->prefetch(QUDA_CUDA_FIELD_LOCATION);
    diracNullSloppy->prefetch(QUDA_CUDA_FIELD_LOCATION);

    logQuda(QUDA_VERBOSE, "Setup of level %d done\n", param.level);

    popLevel();
  }

  void MG::resetStaggeredKD(GaugeField *gauge_in, GaugeField *fat_gauge_in, GaugeField *long_gauge_in,
                            GaugeField *gauge_sloppy_in, GaugeField *fat_gauge_sloppy_in,
                            GaugeField *long_gauge_sloppy_in, double mass)
  {
    if (param.level != 0) errorQuda("The staggered KD operator can only be updated from level 0");

    if (param.transfer_type != QUDA_TRANSFER_OPTIMIZED_KD && param.transfer_type != QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG)
      errorQuda("Attempting to update fine gauge fields of a \"coarse\" but non-KD operator");

    // Need to be careful here: if we're preconditioning an ASQTAD op with
    // a StaggeredKD op, we need to pass the StaggeredKD op the fat links
    auto dirac_type = diracSmoother->getDiracType();

    if ((dirac_type == QUDA_ASQTAD_DIRAC || dirac_type == QUDA_ASQTADPC_DIRAC)
        && param.transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {
      // last nullptr is for the clover field
      diracCoarseResidual->updateFields(fat_gauge_in, fat_gauge_in, long_gauge_in, nullptr);
      diracCoarseSmoother->updateFields(fat_gauge_in, fat_gauge_in, long_gauge_in, nullptr);
      diracCoarseSmootherSloppy->updateFields(fat_gauge_sloppy_in, fat_gauge_sloppy_in, long_gauge_sloppy_in, nullptr);
      diracCoarseNull->updateFields(fat_gauge_in, fat_gauge_in, long_gauge_in, nullptr);
      diracCoarseNullSloppy->updateFields(fat_gauge_sloppy_in, fat_gauge_sloppy_in, long_gauge_sloppy_in, nullptr);
    } else {
      // last nullptr is for the clover field
      diracCoarseResidual->updateFields(gauge_in, fat_gauge_in, long_gauge_in, nullptr);
      diracCoarseSmoother->updateFields(gauge_in, fat_gauge_in, long_gauge_in, nullptr);
      diracCoarseSmootherSloppy->updateFields(gauge_sloppy_in, fat_gauge_sloppy_in, long_gauge_sloppy_in, nullptr);
      diracCoarseNull->updateFields(gauge_in, fat_gauge_in, long_gauge_in, nullptr);
      diracCoarseNullSloppy->updateFields(gauge_sloppy_in, fat_gauge_sloppy_in, long_gauge_sloppy_in, nullptr);
    }

    diracCoarseResidual->setMass(mass);
    diracCoarseSmoother->setMass(mass);
    diracCoarseSmootherSloppy->setMass(mass);
    diracCoarseNull->setMass(mass);
    diracCoarseNullSloppy->setMass(mass);

    // to-do: think about updating Xinv
  }

  void MG::pushLevel(int level) const
  {
    postTrace();
    pushVerbosity(param.mg_global.verbosity[level]);
    pushOutputPrefix(prefix);
  }

  void MG::popLevel() const
  {
    popVerbosity();
    popOutputPrefix();
    postTrace();
  }

  void MG::destroySmoother()
  {
    pushLevel(param.level);

    if (presmoother) {
      delete presmoother;
      presmoother = nullptr;
    }

    if (param_presmooth) {
      delete param_presmooth;
      param_presmooth = nullptr;
    }

    if (postsmoother) {
      delete postsmoother;
      postsmoother = nullptr;
    }

    if (param_postsmooth) {
      delete param_postsmooth;
      param_postsmooth = nullptr;
    }

    popLevel();
  }

  void MG::createSmoother() {
    pushLevel(param.level);

    // create the smoother for this level
    logQuda(QUDA_VERBOSE, "Creating smoother\n");
    destroySmoother();
    param_presmooth = new SolverParam(param);

    param_presmooth->is_preconditioner = false;
    param_presmooth->return_residual = true; // pre-smoother returns the residual vector for subsequent coarsening
    param_presmooth->use_init_guess = QUDA_USE_INIT_GUESS_NO;

    param_presmooth->precision = param.mg_global.invert_param->cuda_prec_sloppy;
    param_presmooth->precision_sloppy = (is_fine_grid()) ? param.mg_global.invert_param->cuda_prec_precondition :
                                                           param.mg_global.invert_param->cuda_prec_sloppy;
    param_presmooth->precision_precondition = (is_fine_grid()) ? param.mg_global.invert_param->cuda_prec_precondition :
                                                                 param.mg_global.invert_param->cuda_prec_sloppy;

    param_presmooth->inv_type = param.smoother;
    param_presmooth->inv_type_precondition = QUDA_INVALID_INVERTER;
    param_presmooth->residual_type = (param_presmooth->inv_type == QUDA_MR_INVERTER) ? QUDA_INVALID_RESIDUAL : QUDA_L2_RELATIVE_RESIDUAL;
    param_presmooth->Nsteps = param.mg_global.smoother_schwarz_cycle[param.level];
    param_presmooth->maxiter = (param.level < param.Nlevel-1) ? param.nu_pre : param.nu_pre + param.nu_post;

    param_presmooth->Nkrylov = param_presmooth->maxiter;
    param_presmooth->pipeline = param_presmooth->maxiter;

    if (is_ca_solver(param_presmooth->inv_type)) {
      param_presmooth->ca_basis = param.mg_global.smoother_solver_ca_basis[param.level];
      param_presmooth->ca_lambda_min = param.mg_global.smoother_solver_ca_lambda_min[param.level];
      param_presmooth->ca_lambda_max = param.mg_global.smoother_solver_ca_lambda_max[param.level];
    }

    param_presmooth->tol = param.smoother_tol;
    param_presmooth->global_reduction = param.global_reduction;

    param_presmooth->sloppy_converge = true; // this means we don't check the true residual before declaring convergence

    param_presmooth->schwarz_type = param.mg_global.smoother_schwarz_type[param.level];
    // inner solver should recompute the true residual after each cycle if using Schwarz preconditioning
    param_presmooth->compute_true_res = (param_presmooth->schwarz_type != QUDA_INVALID_SCHWARZ) ? true : false;

    presmoother = ((param.level < param.Nlevel - 1 || param_presmooth->schwarz_type != QUDA_INVALID_SCHWARZ)
                   && param_presmooth->inv_type != QUDA_INVALID_INVERTER && param_presmooth->maxiter > 0) ?
      Solver::create(*param_presmooth, *param.matSmooth, *param.matSmoothSloppy, *param.matSmoothSloppy,
                     *param.matSmoothSloppy) :
      nullptr;
    if (param.level < param.Nlevel - 1) { // Create the post smoother
      param_postsmooth = new SolverParam(*param_presmooth);
      param_postsmooth->return_residual = false;  // post smoother does not need to return the residual vector
      param_postsmooth->use_init_guess = QUDA_USE_INIT_GUESS_YES;

      param_postsmooth->maxiter = param.nu_post;
      param_postsmooth->Nkrylov = param_postsmooth->maxiter;
      param_postsmooth->pipeline = param_postsmooth->maxiter;

      // we never need to compute the true residual for a post smoother
      param_postsmooth->compute_true_res = false;

      postsmoother = (param_postsmooth->inv_type != QUDA_INVALID_INVERTER && param_postsmooth->maxiter > 0) ?
        Solver::create(*param_postsmooth, *param.matSmooth, *param.matSmoothSloppy, *param.matSmoothSloppy,
                       *param.matSmoothSloppy) :
        nullptr;
    }
    logQuda(QUDA_VERBOSE, "Smoother done\n");

    popLevel();
  }

  void MG::createNullVectors()
  {
    if (param.mg_global.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES) {
      if (param.mg_global.generate_all_levels == QUDA_BOOLEAN_TRUE || param.level == 0) {
        // Initializing to random vectors
        for (int i = 0; i < (int)param.B.size(); i++) { spinorNoise(param.B[i], *rng, QUDA_NOISE_UNIFORM); }
      }
      if (param.mg_global.num_setup_iter[param.level] > 0) {
        if (param.mg_global.vec_load[param.level] == QUDA_BOOLEAN_TRUE
            && strcmp(param.mg_global.vec_infile[param.level], "")
              != 0) { // only load if infile is defined and not computing
          loadVectors(param.B);
        } else if (param.mg_global.use_eig_solver[param.level]) {
          generateEigenVectors(param.B); // Run the eigensolver
        } else {
          generateNullVectors(param.B);
        }
      }
    } else if (strcmp(param.mg_global.vec_infile[param.level], "")
               != 0) { // only load if infile is defined and not computing
      if (param.mg_global.num_setup_iter[param.level] > 0) generateNullVectors(param.B);
    } else if (param.mg_global.vec_load[param.level] == QUDA_BOOLEAN_TRUE) { // only conditional load of null vectors
      loadVectors(param.B);
    } else { // generate free field vectors
      buildFreeVectors(param.B);
    }
  }

  void MG::buildNextDirac()
  {
    pushLevel(param.level);

    logQuda(QUDA_VERBOSE, "Creating coarse Dirac operator\n");

    // use even-odd preconditioning for the coarse grid solver
    if (diracCoarseResidual) delete diracCoarseResidual;
    if (diracCoarseSmoother) delete diracCoarseSmoother;
    if (diracCoarseSmootherSloppy) delete diracCoarseSmootherSloppy;
    if (diracCoarseNull) delete diracCoarseNull;
    if (diracCoarseNullSloppy) delete diracCoarseNullSloppy;

    // check for a pseudo-fine solve
    bool is_pseudo_fine_kd = (param.mg_global.transfer_type[param.level] == QUDA_TRANSFER_OPTIMIZED_KD
                              || param.mg_global.transfer_type[param.level] == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG);

    if (is_pseudo_fine_kd && param.level != 0) errorQuda("Unexpected KD pseudo-fine build from level %d", param.level);

    bool is_pseudo_fine_pv = is_pv() && param.level == 0;

    bool is_coarse_pv = is_pv() && param.level != 0;

    // custom setup for pseudo-fine
    if (is_pseudo_fine_kd) {
      createOptimizedKdDirac();
    } else if (is_pseudo_fine_pv) {
      createDwfPvDirac();
    } else if (is_coarse_pv) {
      createCoarsePvDirac();
    } else {
      createCoarseDirac();
    }

    if (matCoarseResidual) delete matCoarseResidual;
    if (matCoarseSmoother) delete matCoarseSmoother;
    if (matCoarseSmootherSloppy) delete matCoarseSmootherSloppy;
    if (matCoarseNull) delete matCoarseNull;
    if (matCoarseNullSloppy) delete matCoarseNullSloppy;

    matCoarseResidual = new DiracM(*diracCoarseResidual);
    matCoarseSmoother = new DiracM(*diracCoarseSmoother);
    matCoarseSmootherSloppy = new DiracM(*diracCoarseSmootherSloppy);
    matCoarseNull = new DiracM(*diracCoarseNull);
    matCoarseNullSloppy = new DiracM(*diracCoarseNullSloppy);

    logQuda(QUDA_VERBOSE, "Coarse Dirac operator done\n");

    popLevel();
  }

  void MG::createCoarseDirac()
  {
    // check if we are coarsening the preconditioned system then
    bool preconditioned_coarsen
      = (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE);
    QudaMatPCType matpc_type = param.mg_global.invert_param->matpc_type;

    // create coarse grid operator
    DiracParam diracParam;
    diracParam.transfer = transfer;

    // Parameters that matter for coarse construction and application
    diracParam.dirac = preconditioned_coarsen ? const_cast<Dirac *>(diracSmoother) : const_cast<Dirac *>(diracResidual);
    diracParam.kappa = (param.B[0].Nspin() == 1) ?
      -1.0 :
      diracParam.dirac->Kappa(); // -1 cancels automatic kappa in application of Y fields
    diracParam.mass = diracParam.dirac->Mass();
    diracParam.mu = diracParam.dirac->Mu();
    diracParam.mu_factor = param.mg_global.mu_factor[param.level + 1] - param.mg_global.mu_factor[param.level];

    // Need to figure out if we need to force bi-directional build. If any previous level (incl this one) was
    // preconditioned, or a KD op, we have to force bi-directional builds.
    diracParam.need_bidirectional = QUDA_BOOLEAN_FALSE;
    for (int i = 0; i <= param.level; i++) {
      if ((param.mg_global.coarse_grid_solution_type[i] == QUDA_MATPC_SOLUTION
           && param.mg_global.smoother_solve_type[i] == QUDA_DIRECT_PC_SOLVE)
          || (param.mg_global.transfer_type[i] == QUDA_TRANSFER_OPTIMIZED_KD
              || param.mg_global.transfer_type[i] == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG)) {
        diracParam.need_bidirectional = QUDA_BOOLEAN_TRUE;
      }
    }

    diracParam.dagger = QUDA_DAG_NO;
    diracParam.matpcType = matpc_type;
    diracParam.type = QUDA_COARSE_DIRAC;
    diracParam.halo_precision = param.mg_global.precision_null[param.level];
    diracParam.setup_use_mma = param.mg_global.setup_use_mma[param.level];
    // level + 1 since this is for the coarse grid
    diracParam.dslash_use_mma = param.mg_global.dslash_use_mma[param.level + 1];
    diracParam.allow_truncation = (param.mg_global.allow_truncation == QUDA_BOOLEAN_TRUE) ? true : false;

    diracCoarseResidual = new DiracCoarse(diracParam, param.setup_location == QUDA_CUDA_FIELD_LOCATION ? true : false);

    // create smoothing operators
    diracParam.dirac = const_cast<Dirac *>(param.matSmooth->Expose());
    diracParam.halo_precision = param.mg_global.smoother_halo_precision[param.level + 1];

    if (param.mg_global.smoother_solve_type[param.level + 1] == QUDA_DIRECT_PC_SOLVE) {
      diracParam.type = QUDA_COARSEPC_DIRAC;
      diracCoarseSmoother = new DiracCoarsePC(static_cast<DiracCoarse &>(*diracCoarseResidual), diracParam);
      diracCoarseNull = new DiracCoarsePC(static_cast<DiracCoarse &>(*diracCoarseResidual), diracParam);
      {
        bool schwarz = param.mg_global.smoother_schwarz_type[param.level + 1] != QUDA_INVALID_SCHWARZ;
        for (int i = 0; i < 4; i++) diracParam.commDim[i] = schwarz ? 0 : 1;
      }
      diracCoarseSmootherSloppy = new DiracCoarsePC(static_cast<DiracCoarse &>(*diracCoarseSmoother), diracParam);
      diracCoarseNullSloppy = new DiracCoarsePC(static_cast<DiracCoarse &>(*diracCoarseNull), diracParam);
    } else {
      diracParam.type = QUDA_COARSE_DIRAC;
      diracCoarseSmoother = new DiracCoarse(static_cast<DiracCoarse &>(*diracCoarseResidual), diracParam);
      diracCoarseNull = new DiracCoarse(static_cast<DiracCoarse &>(*diracCoarseResidual), diracParam);
      {
        bool schwarz = param.mg_global.smoother_schwarz_type[param.level + 1] != QUDA_INVALID_SCHWARZ;
        for (int i = 0; i < 4; i++) diracParam.commDim[i] = schwarz ? 0 : 1;
      }
      diracCoarseSmootherSloppy = new DiracCoarse(static_cast<DiracCoarse &>(*diracCoarseSmoother), diracParam);
      diracCoarseNullSloppy = new DiracCoarse(static_cast<DiracCoarse &>(*diracCoarseNull), diracParam);
    }
  }

  void MG::createOptimizedKdDirac()
  {
    auto dirac_type = diracSmoother->getDiracType();

    auto smoother_solve_type = param.mg_global.smoother_solve_type[param.level + 1];
    if (smoother_solve_type != QUDA_DIRECT_SOLVE) {
      errorQuda("Invalid solve type %d for optimized KD operator", smoother_solve_type);
    }

    // Determine if we're doing a mixed precision solve for setup or not
    bool mixed_precision_setup
      = (param.mg_global.invert_param->cuda_prec_precondition != param.mg_global.invert_param->cuda_prec_sloppy);

    // Determine if the dirac_type is naive staggered
    bool is_naive_staggered = (dirac_type == QUDA_STAGGERED_DIRAC || dirac_type == QUDA_STAGGEREDPC_DIRAC);
    bool is_improved_staggered = (dirac_type == QUDA_ASQTAD_DIRAC || dirac_type == QUDA_ASQTADPC_DIRAC);

    bool is_coarse_naive_staggered = is_naive_staggered
      || (is_improved_staggered && param.mg_global.transfer_type[param.level] == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG);

    auto fine_gauge = diracSmoother->getStaggeredShortLinkField();
    auto sloppy_gauge = mixed_precision_setup ? diracSmootherSloppy->getStaggeredShortLinkField() : fine_gauge;

    // Allocate and build the KD inverse block (inverse coarse clover)
    xInvKD = AllocateAndBuildStaggeredKahlerDiracInverse(
      *fine_gauge, diracSmoother->Mass(), param.mg_global.staggered_kd_dagger_approximation == QUDA_BOOLEAN_TRUE);

    // Unique to the KD operator as a "coarse level", we can do a mixed-precision
    // near null generation.
    if (mixed_precision_setup) {
      GaugeFieldParam xinv_param(*xInvKD);

      // true is to force FLOAT2
      xinv_param.setPrecision(param.mg_global.invert_param->cuda_prec_precondition, true);

      xInvKD_sloppy = std::shared_ptr<GaugeField>(reinterpret_cast<GaugeField *>(new GaugeField(xinv_param)));
      xInvKD_sloppy->copy(*xInvKD);

    } else {
      // We can just alias fields
      xInvKD_sloppy = xInvKD;
    }

    DiracParam diracParamKD;
    diracParamKD.kappa
      = -1.0; // Cancels automatic kappa in Y field application, which may be relevant if it propagates down
    diracParamKD.mass = diracSmoother->Mass();
    diracParamKD.mu = diracSmoother->Mu(); // doesn't matter
    diracParamKD.mu_factor = 1.0;          // doesn't matter
    diracParamKD.dagger = QUDA_DAG_NO;
    diracParamKD.matpcType = QUDA_MATPC_EVEN_EVEN; // We can use this to track left vs right block jacobi in the future
    diracParamKD.gauge = fine_gauge;
    diracParamKD.xInvKD = xInvKD.get(); // FIXME: pulling a raw unmanaged pointer out of a unique_ptr...
    diracParamKD.dirac
      = const_cast<Dirac *>(diracSmoother); // used to determine if the outer solve is preconditioned or not

    if (is_coarse_naive_staggered) {
      diracParamKD.type = QUDA_STAGGEREDKD_DIRAC;

      diracCoarseResidual = new DiracStaggeredKD(diracParamKD);
      diracCoarseSmoother = new DiracStaggeredKD(diracParamKD);
      diracCoarseNull = new DiracStaggeredKD(diracParamKD);
      if (mixed_precision_setup) {
        diracParamKD.gauge = sloppy_gauge;
        diracParamKD.xInvKD = xInvKD_sloppy.get();
        diracParamKD.dirac = nullptr;
      }
      diracCoarseSmootherSloppy = new DiracStaggeredKD(diracParamKD);
      diracCoarseNullSloppy = new DiracStaggeredKD(diracParamKD);
    } else if (is_improved_staggered) {
      diracParamKD.type = QUDA_ASQTADKD_DIRAC;

      diracParamKD.fatGauge = fine_gauge;
      diracParamKD.longGauge = diracSmoother->getStaggeredLongLinkField();

      diracCoarseResidual = new DiracImprovedStaggeredKD(diracParamKD);
      diracCoarseSmoother = new DiracImprovedStaggeredKD(diracParamKD);
      diracCoarseNull = new DiracStaggeredKD(diracParamKD);
      if (mixed_precision_setup) {
        diracParamKD.fatGauge = sloppy_gauge;
        diracParamKD.longGauge = diracSmootherSloppy->getStaggeredLongLinkField();
        diracParamKD.xInvKD = xInvKD_sloppy.get();
        diracParamKD.dirac = nullptr;
      }
      diracCoarseSmootherSloppy = new DiracImprovedStaggeredKD(diracParamKD);
      diracCoarseNullSloppy = new DiracStaggeredKD(diracParamKD);
    } else {
      errorQuda("Invalid dirac_type %d", dirac_type);
    }
  }

  void MG::createDwfPvDirac()
  {
    auto dirac_type = diracSmoother->getDiracType();

    // Determine if we're doing a mixed precision solve for setup or not
    bool mixed_precision_setup
      = (param.mg_global.invert_param->cuda_prec_precondition != param.mg_global.invert_param->cuda_prec_sloppy);

    // Check to make sure the smoother is for the full operator
    auto smoother_solve_type = param.mg_global.smoother_solve_type[param.level + 1];
    if (smoother_solve_type != QUDA_DIRECT_SOLVE) {
      errorQuda("Invalid solve type %d for optimized PV operator", smoother_solve_type);
    }

    // Get the fine and sloppy gauge fields
    auto fine_gauge = diracSmoother->getGaugeField();
    auto sloppy_gauge = mixed_precision_setup ? diracSmootherSloppy->getGaugeField() : fine_gauge;

    // Create the Dirac operators, first common parameters
    DiracParam diracParamPV;
    diracParamPV.kappa = param.mg_global.kappa_dwf_null;
    diracParamPV.mass = diracSmoother->Mass();
    diracParamPV.mu = diracSmoother->Mu();              // doesn't matter
    diracParamPV.mu_factor = diracSmoother->MuFactor(); // doesn't matter
    diracParamPV.m5 = reinterpret_cast<const DiracDomainWall *>(diracSmoother)->M5();
    diracParamPV.Ls = reinterpret_cast<const DiracDomainWall *>(diracSmoother)->getLs();
    diracParamPV.dagger = QUDA_DAG_NO;
    diracParamPV.matpcType = QUDA_MATPC_EVEN_EVEN; // I guess we could hack this for left vs right block Jacobi?
    diracParamPV.gauge = fine_gauge;

    // then DWF-4D vs Mobius-specific
    if (dirac_type == QUDA_DOMAIN_WALL_4D_DIRAC) {
      diracParamPV.type = QUDA_DOMAIN_WALL_4DPV_DIRAC;

      diracCoarseResidual = new DiracDomainWall4DPV(diracParamPV);
      diracCoarseSmoother = new DiracDomainWall4DPV(diracParamPV);
      if (mixed_precision_setup) {
        diracParamPV.gauge = sloppy_gauge;
        diracParamPV.dirac = nullptr;
      }
      diracCoarseSmootherSloppy = new DiracDomainWall4DPV(diracParamPV);
    } else if (dirac_type == QUDA_MOBIUS_DOMAIN_WALL_DIRAC) {
      auto b5 = reinterpret_cast<const DiracMobius *>(diracSmoother)->getB5();
      auto c5 = reinterpret_cast<const DiracMobius *>(diracSmoother)->getC5();
      for (int i = 0; i < diracParamPV.Ls; i++) {
        diracParamPV.b_5[i] = b5[i];
        diracParamPV.c_5[i] = c5[i];
      }
      diracParamPV.type = QUDA_MOBIUS_DOMAIN_WALLPV_DIRAC;

      diracCoarseResidual = new DiracMobiusPV(diracParamPV);
      diracCoarseSmoother = new DiracMobiusPV(diracParamPV);
      if (mixed_precision_setup) {
        diracParamPV.gauge = sloppy_gauge;
        diracParamPV.dirac = nullptr;
      }
      diracCoarseSmootherSloppy = new DiracMobiusPV(diracParamPV);
    } else {
      errorQuda("Invalid fine domain wall operator type %d", dirac_type);
    }

    // near-null vectors are generated with the Wilson operator
    diracParamPV.type = QUDA_WILSON_DIRAC;
    diracParamPV.kappa = param.mg_global.kappa_dwf_null;
    diracParamPV.gauge = fine_gauge;
    diracCoarseNull = new DiracWilson(diracParamPV);
    diracParamPV.gauge = sloppy_gauge;
    diracCoarseNullSloppy = new DiracWilson(diracParamPV);
  }

  void MG::createCoarsePvDirac()
  {
    auto dirac_type = diracSmoother->getDiracType();

    if (dirac_type != QUDA_DOMAIN_WALL_4DPV_DIRAC && dirac_type != QUDA_MOBIUS_DOMAIN_WALLPV_DIRAC)
      errorQuda("Unexpected Dirac type %d for CoarsePv build", dirac_type);

    // lots of checks to make sure the coarse system is also a full system
    if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION)
      errorQuda("Unexpected coarse grid solution type %d, expecting %d", param.coarse_grid_solution_type,
                QUDA_MAT_SOLUTION);

    // Check to make sure the smoother is for the full operator
    auto smoother_solve_type = param.mg_global.smoother_solve_type[param.level + 1];
    if (smoother_solve_type != QUDA_DIRECT_SOLVE) {
      errorQuda("Invalid solve type %d for coarse PV operator", smoother_solve_type);
    }

    QudaMatPCType matpc_type = param.mg_global.invert_param->matpc_type;

    // create coarse grid operator
    DiracParam diracParam;
    diracParam.transfer = transfer;

    // Here's a sneaky one: we build the null operator *first*, since it's just a vanilla
    // DiracCoarse and it'll get us the coarse X and Y fields we need.
    // Note that this bakes a "kappa" into the operator, which we'll have to undo later.

    // Parameters that matter for coarse construction and application
    diracParam.dirac = const_cast<Dirac *>(diracNull);
    diracParam.kappa = param.mg_global.kappa_dwf_null;
    diracParam.mass = diracParam.dirac->Mass();
    diracParam.mu = diracParam.dirac->Mu(); // ignored
    diracParam.mu_factor = param.mg_global.mu_factor[param.level + 1] - param.mg_global.mu_factor[param.level]; // ignored

    // We're always coarsening the simple Wilson operator so we don't need a bidirectional build
    diracParam.need_bidirectional = QUDA_BOOLEAN_FALSE;

    diracParam.dagger = QUDA_DAG_NO;
    diracParam.matpcType = matpc_type; // ignored
    diracParam.type = QUDA_COARSE_DIRAC;
    diracParam.halo_precision = param.mg_global.precision_null[param.level];
    diracParam.setup_use_mma = param.mg_global.setup_use_mma[param.level];
    // level + 1 since this is for the coarse grid
    diracParam.dslash_use_mma = param.mg_global.dslash_use_mma[param.level + 1];
    diracParam.allow_truncation = false;

    diracParam.type = QUDA_COARSE_DIRAC;
    diracCoarseNull = new DiracCoarse(diracParam, param.setup_location == QUDA_CUDA_FIELD_LOCATION ? true : false);

    // create the sloppy null
    diracParam.halo_precision = param.mg_global.smoother_halo_precision[param.level + 1];
    {
      bool schwarz = param.mg_global.smoother_schwarz_type[param.level + 1] != QUDA_INVALID_SCHWARZ;
      for (int i = 0; i < 4; i++) diracParam.commDim[i] = schwarz ? 0 : 1;
    }
    diracCoarseNullSloppy = new DiracCoarse(static_cast<DiracCoarse &>(*diracCoarseNull), diracParam);

    // Now we build the CoarsePV operator
    diracParam.dirac = const_cast<Dirac *>(param.matSmooth->Expose());
    diracParam.Ls = diracParam.dirac->getLs(); // param.mg_global.custom_ls[param.level + 1]
    diracParam.m5 = diracParam.dirac->M5();
    diracParam.parent_dwf = get_outermost_dirac_type();

    if (diracParam.parent_dwf == QUDA_MOBIUS_DOMAIN_WALL_DIRAC) {
      auto b5 = diracSmoother->getB5();
      auto c5 = diracSmoother->getC5();
      for (int i = 0; i < diracParam.Ls; i++) {
        diracParam.b_5[i] = b5[i];
        diracParam.c_5[i] = c5[i];
      }
    }

    diracParam.type = QUDA_COARSEPV_DIRAC;

    // temporarily restore values
    diracParam.halo_precision = param.mg_global.precision_null[param.level];
    for (int i = 0; i < 4; i++) diracParam.commDim[i] = 1;

    // build the residual operator
    diracCoarseResidual = new DiracCoarsePV(static_cast<DiracCoarse &>(*diracCoarseNull), diracParam);

    // create smoothing operators
    diracParam.halo_precision = param.mg_global.smoother_halo_precision[param.level + 1];
    diracCoarseSmoother = new DiracCoarsePV(static_cast<DiracCoarse &>(*diracCoarseNull), diracParam);
    {
      bool schwarz = param.mg_global.smoother_schwarz_type[param.level + 1] != QUDA_INVALID_SCHWARZ;
      for (int i = 0; i < 4; i++) diracParam.commDim[i] = schwarz ? 0 : 1;
    }
    diracCoarseSmootherSloppy = new DiracCoarsePV(static_cast<DiracCoarse &>(*diracCoarseSmoother), diracParam);
  }

  void MG::destroyCoarseSolver() {
    pushLevel(param.level);

    if (param.cycle_type == QUDA_MG_CYCLE_VCYCLE && param.level < param.Nlevel-2) {
      // nothing to do
    } else if (param.cycle_type == QUDA_MG_CYCLE_RECURSIVE || param.level == param.Nlevel-2) {
      if (coarse_solver) {
        auto &coarse_solver_inner = reinterpret_cast<PreconditionedSolver *>(coarse_solver)->ExposeSolver();
        // int defl_size = coarse_solver_inner.evecs.size();
        int defl_size = coarse_solver_inner.deflationSpaceSize();
        if (defl_size > 0 && transfer && param.mg_global.preserve_deflation) {
          // Deflation space exists and we are going to create a new solver. Extract deflation space.
          logQuda(QUDA_VERBOSE, "Extracting deflation space size %d to MG\n", defl_size);
          coarse_solver_inner.extractDeflationSpace(evecs);
        }
        delete coarse_solver;
        coarse_solver = nullptr;
      }
      if (param_coarse_solver) {
        delete param_coarse_solver;
        param_coarse_solver = nullptr;
      }
    } else {
      errorQuda("Multigrid cycle type %d not supported", param.cycle_type);
    }

    popLevel();
  }

  void MG::createCoarseSolver() {
    pushLevel(param.level);

    logQuda(QUDA_VERBOSE, "Creating coarse solver wrapper\n");
    destroyCoarseSolver();
    if (param.cycle_type == QUDA_MG_CYCLE_VCYCLE && param.level < param.Nlevel-2) {
      // if coarse solver is not a bottom solver and on the second to bottom level then we can just use the coarse solver as is
      coarse_solver = coarse;
      logQuda(QUDA_VERBOSE, "Assigned coarse solver to coarse MG operator\n");
    } else if (param.cycle_type == QUDA_MG_CYCLE_RECURSIVE || param.level == param.Nlevel-2) {

      param_coarse_solver = new SolverParam(param);
      param_coarse_solver->inv_type = param.mg_global.coarse_solver[param.level + 1];
      param_coarse_solver->is_preconditioner = false;
      param_coarse_solver->sloppy_converge = true; // this means we don't check the true residual before declaring convergence
      param_coarse_solver->return_residual = false; // coarse solver does need to return residual vector

      param_coarse_solver->use_init_guess = QUDA_USE_INIT_GUESS_NO;
      // Coarse level deflation is triggered if the eig param structure exists
      // on the coarsest level, and we are on the next to coarsest level.
      if (param.mg_global.use_eig_solver[param.Nlevel - 1] && (param.level == param.Nlevel - 2)) {
        param_coarse_solver->eig_param = *param.mg_global.eig_param[param.Nlevel - 1];
        param_coarse_solver->deflate = QUDA_BOOLEAN_TRUE;
        // Due to coherence between these levels, an initial guess
        // might be beneficial.
        if (param.mg_global.coarse_guess == QUDA_BOOLEAN_TRUE) {
          param_coarse_solver->use_init_guess = QUDA_USE_INIT_GUESS_YES;
        }

        // Deflation on the coarse is supported for 6 solvers only
        if (param_coarse_solver->inv_type != QUDA_CA_CGNR_INVERTER && param_coarse_solver->inv_type != QUDA_CGNR_INVERTER
            && param_coarse_solver->inv_type != QUDA_CA_CGNE_INVERTER && param_coarse_solver->inv_type != QUDA_CGNE_INVERTER
            && param_coarse_solver->inv_type != QUDA_CA_GCR_INVERTER && param_coarse_solver->inv_type != QUDA_GCR_INVERTER
            && param_coarse_solver->inv_type != QUDA_BICGSTABL_INVERTER) {
          errorQuda("Coarse grid deflation not supported with coarse solver %d", param_coarse_solver->inv_type);
        }

        if (strcmp(param_coarse_solver->eig_param.vec_infile, "") == 0 && // check that input file not already set
            param.mg_global.vec_load[param.level + 1] == QUDA_BOOLEAN_TRUE
            && (strcmp(param.mg_global.vec_infile[param.level + 1], "") != 0)) {
          std::string vec_infile(param.mg_global.vec_infile[param.level + 1]);
          vec_infile += "_level_";
          vec_infile += std::to_string(param.level + 1);
          vec_infile += "_defl_";
          vec_infile += std::to_string(param.mg_global.n_vec[param.level + 1]);
          strcpy(param_coarse_solver->eig_param.vec_infile, vec_infile.c_str());
        }

        if (strcmp(param_coarse_solver->eig_param.vec_outfile, "") == 0 && // check that output file not already set
            param.mg_global.vec_store[param.level + 1] == QUDA_BOOLEAN_TRUE
            && (strcmp(param.mg_global.vec_outfile[param.level + 1], "") != 0)) {
          std::string vec_outfile(param.mg_global.vec_outfile[param.level + 1]);
          vec_outfile += "_level_";
          vec_outfile += std::to_string(param.level + 1);
          vec_outfile += "_defl_";
          vec_outfile += std::to_string(param.mg_global.n_vec[param.level + 1]);
          strcpy(param_coarse_solver->eig_param.vec_outfile, vec_outfile.c_str());
          param_coarse_solver->eig_param.partfile = param.mg_global.mg_vec_partfile[param.level + 1];
        }
      }

      param_coarse_solver->tol = param.mg_global.coarse_solver_tol[param.level+1];
      param_coarse_solver->global_reduction = true;
      param_coarse_solver->compute_true_res = false;
      param_coarse_solver->delta = 1e-8;
      param_coarse_solver->pipeline = 8;

      param_coarse_solver->maxiter = param.mg_global.coarse_solver_maxiter[param.level+1];
      param_coarse_solver->Nkrylov = param_coarse_solver->maxiter < param_coarse_solver->Nkrylov ?
        param_coarse_solver->maxiter :
        param_coarse_solver->Nkrylov;
      if (is_ca_solver(param_coarse_solver->inv_type)) {
        param_coarse_solver->ca_basis = param.mg_global.coarse_solver_ca_basis[param.level+1];
        param_coarse_solver->ca_lambda_min = param.mg_global.coarse_solver_ca_lambda_min[param.level+1];
        param_coarse_solver->ca_lambda_max = param.mg_global.coarse_solver_ca_lambda_max[param.level+1];
        param_coarse_solver->Nkrylov = param.mg_global.coarse_solver_ca_basis_size[param.level+1];
      } else if (param_coarse_solver->inv_type == QUDA_BICGSTABL_INVERTER) {
        param_coarse_solver->Nkrylov = param.mg_global.coarse_solver_ca_basis_size[param.level + 1];
      }
      param_coarse_solver->inv_type_precondition = (param.level<param.Nlevel-2 || coarse->presmoother) ? QUDA_MG_INVERTER : QUDA_INVALID_INVERTER;
      param_coarse_solver->preconditioner = (param.level<param.Nlevel-2 || coarse->presmoother) ? coarse : nullptr;
      param_coarse_solver->mg_instance = true;
      param_coarse_solver->verbosity_precondition = param.mg_global.verbosity[param.level+1];

      // preconditioned solver wrapper is uniform precision
      param_coarse_solver->precision = r_coarse[0].Precision();
      param_coarse_solver->precision_sloppy = param_coarse_solver->precision;
      param_coarse_solver->precision_precondition = param_coarse_solver->precision_sloppy;

      if (param.mg_global.coarse_grid_solution_type[param.level + 1] == QUDA_MATPC_SOLUTION) {
        Solver *solver = Solver::create(*param_coarse_solver, *matCoarseSmoother, *matCoarseSmoother,
                                        *matCoarseSmoother, *matCoarseSmoother);
        sprintf(coarse_prefix, "MG level %d (%s): ", param.level + 1,
                param.mg_global.location[param.level + 1] == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");
        coarse_solver
          = new PreconditionedSolver(*solver, *matCoarseSmoother->Expose(), *param_coarse_solver, coarse_prefix);
      } else {
        Solver *solver = Solver::create(*param_coarse_solver, *matCoarseResidual, *matCoarseResidual,
                                        *matCoarseResidual, *matCoarseResidual);
        sprintf(coarse_prefix, "MG level %d (%s): ", param.level + 1,
                param.mg_global.location[param.level + 1] == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");
        coarse_solver
          = new PreconditionedSolver(*solver, *matCoarseResidual->Expose(), *param_coarse_solver, coarse_prefix);
      }

      setOutputPrefix(prefix); // restore since we just popped back from coarse grid

      if (param.level == param.Nlevel - 2 && param.mg_global.use_eig_solver[param.level + 1]) {

        // Test if a coarse grid deflation space needs to be transferred to the coarse solver to prevent recomputation
        int defl_size = evecs.size();
        auto &coarse_solver_inner = reinterpret_cast<PreconditionedSolver *>(coarse_solver)->ExposeSolver();
        if (defl_size > 0 && transfer && param.mg_global.preserve_deflation) {
          // We shall not recompute the deflation space, we shall transfer
          // vectors stored in the parent MG instead
          coarse_solver_inner.setDeflateCompute(false);
          coarse_solver_inner.setRecomputeEvals(true);
          logQuda(QUDA_VERBOSE, "Transferring deflation space size %d to coarse solver\n", defl_size);
          // Create space in coarse solver to hold deflation space, destroy space in MG.
          coarse_solver_inner.injectDeflationSpace(evecs);
        }

        // Run a dummy solve so that the deflation space is constructed and computed if needed during the MG setup,
        // or the eigenvalues are recomputed during transfer.
        spinorNoise(r_coarse[0], *coarse->rng, QUDA_NOISE_UNIFORM);
        param_coarse_solver->maxiter = 1; // do a single iteration on the dummy solve
        (*coarse_solver)(x_coarse[0], r_coarse[0]);
        setOutputPrefix(prefix); // restore since we just popped back from coarse grid
        param_coarse_solver->maxiter = param.mg_global.coarse_solver_maxiter[param.level + 1];
      }

      logQuda(QUDA_VERBOSE, "Assigned coarse solver to preconditioned GCR solver\n");
    } else {
      errorQuda("Multigrid cycle type %d not supported", param.cycle_type);
    }
    logQuda(QUDA_VERBOSE, "Coarse solver wrapper done\n");

    popLevel();
  }

  MG::~MG()
  {
    pushLevel(param.level);

    if (param.level < param.Nlevel - 1) {
      if (coarse) delete coarse;
      if (param.level == param.Nlevel-1 || param.cycle_type == QUDA_MG_CYCLE_RECURSIVE) {
        if (coarse_solver) delete coarse_solver;
        if (param_coarse_solver) delete param_coarse_solver;
      }

      if (transfer) delete transfer;
      if (matCoarseSmootherSloppy) delete matCoarseSmootherSloppy;
      if (diracCoarseSmootherSloppy) delete diracCoarseSmootherSloppy;
      if (matCoarseSmoother) delete matCoarseSmoother;
      if (diracCoarseSmoother) delete diracCoarseSmoother;
      if (matCoarseResidual) delete matCoarseResidual;
      if (diracCoarseResidual) delete diracCoarseResidual;
      if (matCoarseNullSloppy) delete matCoarseNullSloppy;
      if (diracCoarseNullSloppy) delete diracCoarseNullSloppy;
      if (matCoarseNull) delete matCoarseNull;
      if (diracCoarseNull) delete diracCoarseNull;
      if (postsmoother) delete postsmoother;
      if (param_postsmooth) delete param_postsmooth;
    }

    if (rng) {
      delete rng;
    }

    if (presmoother) delete presmoother;
    if (param_presmooth) delete param_presmooth;
    if (param_coarse) delete param_coarse;

    popLevel();
  }

  bool check_deviation(double deviation, double tol)
  {
    return (deviation > tol || std::isnan(deviation) || std::isinf(deviation));
  }

  /**
     Verification that the constructed multigrid operator is valid
  */
  void MG::verify(bool recursively)
  {
    pushLevel(param.level);

    QudaPrecision prec
      = std::min(param.mg_global.precision_null[param.level], param.mg_global.invert_param->cuda_prec_sloppy);
    // may want to revisit this---these were relaxed for cases where ghost_precision < precision
    // these were set while hacking in tests of quarter precision ghosts
    // moreover, we can improve the precision of block ortho with a tighter max than 1.0
    double tol;
    switch (prec) {
    case QUDA_QUARTER_PRECISION: tol = 5e-2; break;
    case QUDA_HALF_PRECISION: tol = 5e-2; break;
    case QUDA_SINGLE_PRECISION: tol = 2e-3; break;
    default: tol = 1e-8;
    }

    // temporary fields used for verification
    std::vector<ColorSpinorField> fine_tmp(param.Nvec);
    ColorSpinorParam fine_param(param.B[0]);
    fine_param.setPrecision(param.mg_global.invert_param->cuda_prec_sloppy, QUDA_INVALID_PRECISION,
                            fine_param.location == QUDA_CUDA_FIELD_LOCATION ? true : false);
    if (param.transfer_type == QUDA_TRANSFER_DWF_PV && param.level == 1)
      fine_param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    else
      fine_param.gammaBasis
        = (param.level > 0 || param.B[0].Nspin() == 1) ? QUDA_DEGRAND_ROSSI_GAMMA_BASIS : QUDA_UKQCD_GAMMA_BASIS;
    for (auto &f : fine_tmp) f = ColorSpinorField(fine_param);

    std::vector<ColorSpinorField> coarse_tmp(param.Nvec);
    ColorSpinorParam coarse_param(r_coarse[0]);
    coarse_param.create = QUDA_NULL_FIELD_CREATE;
    if (is_pv() && param.level != 0) {
      coarse_param.nDim = 4;
      coarse_param.x[4] = 1;
    }
    for (auto &c : coarse_tmp) c = ColorSpinorField(coarse_param);

    auto &tmp1 = fine_tmp[0];
    auto &tmp2 = fine_tmp[1];
    auto &tmp_coarse = coarse_tmp[0];

    vector<double> B_norm;
    if (param.transfer_type == QUDA_TRANSFER_AGGREGATE) B_norm = norm2(param.B);

    // No need to check (projector) v_k for staggered case
    if (param.transfer_type == QUDA_TRANSFER_AGGREGATE) {

      logQuda(QUDA_SUMMARIZE, "Checking 0 = (1 - P P^\\dagger) v_k for %d vectors\n", param.Nvec);

      // change fine_tmp to match B basis to allow comparison
      auto basis = fine_tmp[0].GammaBasis();
      for (auto &f : fine_tmp) f.GammaBasis(param.B[0].GammaBasis());
      transfer->R(coarse_tmp, param.B);
      transfer->P(fine_tmp, coarse_tmp);

      auto max_deviation = blas::max_deviation(param.B, fine_tmp);
      auto deviation = xmyNorm(param.B, fine_tmp);
      auto coarse_norm = norm2(coarse_tmp);
      auto fine_norm = norm2(coarse_tmp);
      for (auto i = 0; i < param.Nvec; i++) {
        auto l2_deviation = sqrt(deviation[i]) / B_norm[i];
        logQuda(
          QUDA_VERBOSE, "Vector %d: L2 norms v_k = %e P^\\dagger v_k = %e (1 - P P^\\dagger) v_k = %e; Deviations: L2 relative = %e, max = %e\n",
          i, B_norm[i], coarse_norm[i], fine_norm[i], l2_deviation, max_deviation[i][0]);
        if (check_deviation(l2_deviation, tol))
          errorQuda("k=%d orthonormality failed: L2 relative deviation %e > %e", i, l2_deviation, tol);
        if (check_deviation(max_deviation[i][0], tol))
          errorQuda("k=%d orthonormality failed: max deviation %e > %e", i, max_deviation[i][0], tol);
      }
      for (auto &f : fine_tmp) f.GammaBasis(basis); // restore basis

      // the oblique check
      if (param.mg_global.run_oblique_proj_check) {
        if (is_pv()) errorQuda("The oblique projector check does not currently work with DWF");

        sprintf(prefix, "MG level %d (%s): Null vector Oblique Projections : ", param.level + 1,
                param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");
        setOutputPrefix(prefix);

        // Oblique projections
        logQuda(QUDA_SUMMARIZE, "Checking 1 > || (1 - DP(P^dagDP)P^dag) v_k || / || v_k || for %d vectors\n", param.Nvec);

        for (int i = 0; i < param.Nvec; i++) {
          transfer->R(r_coarse[0], param.B[i]);
          (*coarse_solver)(x_coarse[0], r_coarse[0]); // this needs to be an exact solve to pass
          setOutputPrefix(prefix);                // restore prefix after return from coarse grid
          transfer->P(tmp2, x_coarse[0]);
          (*param.matResidual)(tmp1, tmp2);
          tmp2 = param.B[i];
          logQuda(QUDA_SUMMARIZE, "Vector %d: norms %e %e\n", i, B_norm[i], norm2(tmp1));
          logQuda(QUDA_SUMMARIZE, "relative residual = %e\n", sqrt(xmyNorm(tmp2, tmp1) / B_norm[i]));
        }

        sprintf(prefix, "MG level %d (%s): ", param.level + 1, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");
        setOutputPrefix(prefix);
      }
    }

#if 0
    logQuda(QUDA_SUMMARIZE, "Checking 1 > || (1 - D P (P^\\dagger D P) P^\\dagger v_k || / || v_k || for %d vectors\n",
            param.Nvec);

    for (int i=0; i<param.Nvec; i++) {
      transfer->R(r_coarse, param.B[i]);
      (*coarse)(x_coarse[0], r_coarse[0]); // this needs to be an exact solve to pass
      setOutputPrefix(prefix); // restore output prefix
      transfer->P(tmp2, x_coarse[0]);
      param.matResidual(tmp1, tmp2);
      tmp2 = param.B[i];
      logQuda(QUDA_SUMMARIZE, "Vector %d: norms %e %e ", i, B_norm[i], norm2(tmp1));
      logQuda(QUDA_SUMMARIZE, "relative residual = %e\n", sqrt(xmyNorm(tmp2, tmp1) / B_norm[i]) );
    }
#endif

    // create coarse residual vector if not already created in verify()
    if (r_coarse.empty()) {
      auto customLs = is_pv() ? reinterpret_cast<const DiracDomainWall *>(diracSmoother)->getLs() : -1;

      r_coarse.resize(1);
      r_coarse[0] = param.B[0].create_coarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, customLs,
                                             param.mg_global.invert_param->cuda_prec_sloppy,
                                             param.mg_global.location[param.level + 1]);
      if (is_pv() && param.level == 0) r_coarse[0].GammaBasis(QUDA_UKQCD_GAMMA_BASIS);
    }

    // create coarse solution vector if not already created in verify()
    if (x_coarse.empty()) {
      auto customLs = is_pv() ? reinterpret_cast<const DiracDomainWall *>(diracSmoother)->getLs() : -1;

      x_coarse.resize(1);
      x_coarse[0] = param.B[0].create_coarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, customLs,
                                             param.mg_global.invert_param->cuda_prec_sloppy,
                                             param.mg_global.location[param.level + 1]);
      if (is_pv() && param.level == 0) x_coarse[0].GammaBasis(QUDA_UKQCD_GAMMA_BASIS);
    }

    {
      logQuda(QUDA_SUMMARIZE, "Checking 0 = (1 - P^\\dagger P) eta_c\n");

      if (is_pv() && param.level != 0) {
        // right now, x_coarse[0] and r_coarse[0] are nColor == 24 (or whatever), Ls == 8 (or whatever) 5-d fields
        // we need to split them into a vector of 8 4-d fields

        // get Ls
        auto Ls = x_coarse[0].X(4);

        // populate x_coarse[0] with random values
        spinorNoise(x_coarse[0], *rng, QUDA_NOISE_UNIFORM);

        // create the set of 4-d coarse vectors
        ColorSpinorParam csParam(x_coarse[0]);
        csParam.nDim = 4;
        csParam.x[4] = 1;
        csParam.create = QUDA_NULL_FIELD_CREATE;

        // prepare vectors of coarse 4-d fields for x_coarse and r_coarse
        auto x_coarse_4d = getFieldTmp<ColorSpinorField>(Ls, csParam);
        auto r_coarse_4d = getFieldTmp<ColorSpinorField>(Ls, csParam);

        // create the set of 4-d fine vectors
        ColorSpinorParam csParamFine(tmp2);
        csParamFine.create = QUDA_NULL_FIELD_CREATE;

        // prepare vectors of fine 4-d fields for tmp2
        auto tmp2_4d = getFieldTmp<ColorSpinorField>(Ls, csParamFine);

        printfQuda("Testing Split/Join Workflow\n");

        // split x_coarse[0]
        Split5DTo4DFields(x_coarse_4d, x_coarse[0]);

        // Check split 5d to 4d: norm2 of the 5-d field should equal the sum of the norms of the 4-d fields
        auto x_coarse_5d_r2 = norm2(x_coarse[0]);
        {
          auto x_coarse_4d_r2 = norm2(x_coarse_4d);
          double accum_4d = 0.;
          for (int i = 0; i < Ls; i++) {
            // printfQuda("x_coarse_4d constituent %d has norm %e\n", i, x_coarse_4d_r2[i]);
            accum_4d += x_coarse_4d_r2[i];
          }
          auto l2_deviation = sqrt(abs(x_coarse_5d_r2 - accum_4d) / x_coarse_5d_r2);
          logQuda(QUDA_VERBOSE, "  Split5DTo4D: 5-d norm2 %e , summed 4-d norm2 %e; Deviations: L2 relative = %e\n",
                  x_coarse_5d_r2, accum_4d, l2_deviation);
          if (check_deviation(l2_deviation, tol))
            errorQuda("Split5DTo4D failed, L2 relative deviation = %e > %e", l2_deviation, tol);
        }

        transfer->P(tmp2_4d, x_coarse_4d);
        transfer->R(r_coarse_4d, tmp2_4d);

        // join r_coarse_4d
        Join4DTo5DField(r_coarse[0], r_coarse_4d);

        // check norms, etc
        auto r2 = norm2(r_coarse[0]);

        // Check join 4d to 5d: norm2 of the 5-d field should equal the sum of the norms of the 4-d fields
        {
          auto r_coarse_4d_r2 = norm2(r_coarse_4d);
          double accum_4d = 0.;
          for (int i = 0; i < Ls; i++) { accum_4d += r_coarse_4d_r2[i]; }
          auto l2_deviation = sqrt(abs(r2 - accum_4d) / r2);
          logQuda(QUDA_VERBOSE, "  Join4DTo5D: 5-d norm2 %e , summed 4-d norm2 %e; Deviations: L2 relative = %e\n", r2,
                  accum_4d, l2_deviation);
          if (check_deviation(l2_deviation, tol))
            errorQuda("Join4DTo5D failed, L2 relative deviation = %e > %e", l2_deviation, tol);
        }

        auto max_deviation = blas::max_deviation(r_coarse[0], x_coarse[0]);
        auto l2_deviation = sqrt(xmyNorm(x_coarse[0], r_coarse[0]) / norm2(x_coarse[0]));
        logQuda(QUDA_VERBOSE, "  Split/Join L2 norms %e %e (fine tmp %e); Deviations: L2 relative = %e, max = %e\n",
                norm2(x_coarse[0]), r2, norm2(tmp2), l2_deviation, max_deviation[0]);
        if (check_deviation(l2_deviation, tol))
          errorQuda("coarse span failed: L2 relative deviation = %e > %e", l2_deviation, tol);
        if (check_deviation(max_deviation[0], tol))
          errorQuda("coarse span failed: max deviation = %e > %e", max_deviation[0], tol);

        // now test the fused prolongate/restrict
        printfQuda("Testing 5-d field Workflow\n");

        // populate x_coarse[0] with random values
        spinorNoise(x_coarse[0], *rng, QUDA_NOISE_UNIFORM);

        // create a temporary 5-d fine vector
        csParamFine = ColorSpinorParam(tmp2);
        csParamFine.nDim = 5;
        csParamFine.x[4] = Ls;
        csParamFine.create = QUDA_NULL_FIELD_CREATE;

        // prepare vectors of fine 4-d fields for tmp2
        auto tmp2_5d = getFieldTmp<ColorSpinorField>(csParamFine);

        transfer->P(tmp2_5d, x_coarse[0]);
        transfer->R(r_coarse[0], tmp2_5d);

        r2 = norm2(r_coarse[0]);
        max_deviation = blas::max_deviation(r_coarse[0], x_coarse[0]);
        l2_deviation = sqrt(xmyNorm(x_coarse[0], r_coarse[0]) / norm2(x_coarse[0]));
        logQuda(QUDA_VERBOSE, "  5-d field L2 norms %e %e (fine tmp %e); Deviations: L2 relative = %e, max = %e\n",
                norm2(x_coarse[0]), r2, norm2(tmp2), l2_deviation, max_deviation[0]);
        if (check_deviation(l2_deviation, tol))
          errorQuda("coarse span failed: L2 relative deviation = %e > %e", l2_deviation, tol);
        if (check_deviation(max_deviation[0], tol))
          errorQuda("coarse span failed: max deviation = %e > %e", max_deviation[0], tol);
      } else {
        spinorNoise(x_coarse[0], *rng, QUDA_NOISE_UNIFORM);
        transfer->P(tmp2, x_coarse[0]);
        transfer->R(r_coarse[0], tmp2);
        auto r2 = norm2(r_coarse[0]);
        auto max_deviation = blas::max_deviation(r_coarse[0], x_coarse[0]);
        auto l2_deviation = sqrt(xmyNorm(x_coarse[0], r_coarse[0]) / norm2(x_coarse[0]));
        logQuda(QUDA_VERBOSE, "L2 norms %e %e (fine tmp %e); Deviations: L2 relative = %e, max = %e\n",
                norm2(x_coarse[0]), r2, norm2(tmp2), l2_deviation, max_deviation[0]);
        if (check_deviation(l2_deviation, tol))
          errorQuda("coarse span failed: L2 relative deviation = %e > %e", l2_deviation, tol);
        if (check_deviation(max_deviation[0], tol))
          errorQuda("coarse span failed: max deviation = %e > %e", max_deviation[0], tol);
      }
    }

    logQuda(QUDA_SUMMARIZE, "Checking 0 = (D_c - P^\\dagger D P) (native coarse operator to emulated operator)\n");
    zero(tmp_coarse);
    zero(r_coarse);

#if 0 // debugging trick: point source matrix elements
    tmp_coarse.Source(QUDA_POINT_SOURCE, 0, 0, 0);
#else
    spinorNoise(tmp_coarse, *rng, QUDA_NOISE_UNIFORM);
#endif

    // put a non-trivial vector on the fine level as well
    transfer->P(tmp1, tmp_coarse);

    // the three-hop terms in ASQTAD can break the verification depending on how we're coarsening the operator
    // and if the aggregate size is too small in a direction
    bool can_verify = true;

    bool is_verify_kd = (param.transfer_type == QUDA_TRANSFER_OPTIMIZED_KD
                         || param.transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG)
      && (diracSmoother->getDiracType() == QUDA_STAGGERED_DIRAC || diracSmoother->getDiracType() == QUDA_STAGGEREDPC_DIRAC
          || diracSmoother->getDiracType() == QUDA_ASQTAD_DIRAC || diracSmoother->getDiracType() == QUDA_ASQTADPC_DIRAC);

    bool is_verify_dwf_pv = (param.transfer_type == QUDA_TRANSFER_DWF_PV)
      && (diracSmoother->getDiracType() == QUDA_DOMAIN_WALL_4D_DIRAC
          || diracSmoother->getDiracType() == QUDA_MOBIUS_DOMAIN_WALL_DIRAC);

    bool is_verify_coarse_pv = (param.transfer_type == QUDA_TRANSFER_AGGREGATE)
      && (diracSmoother->getDiracType() == QUDA_DOMAIN_WALL_4DPV_DIRAC
          || diracSmoother->getDiracType() == QUDA_MOBIUS_DOMAIN_WALLPV_DIRAC
          || diracSmoother->getDiracType() == QUDA_COARSEPV_DIRAC);

    if (is_verify_kd) {
      // If we're doing an optimized build with the staggered operator, we need to skip the verify on level 0
      can_verify = false;
      logQuda(QUDA_VERBOSE,
              "Intentionally skipping staggered -> staggered KD verify because it's not a \"real\" coarsen\n");
    } else if (is_verify_dwf_pv) {
      // If we're doing PV-preconditioned domain wall, we need to skip the verify on level 0
      can_verify = false;
      logQuda(
        QUDA_VERBOSE, "Performing a custom verify for dwf -> pv^dagger dwf verify: reconstructing dwf from the multi-rhs Wilson + chiral projectors\n");

      verifyDwfPV();
    } else if (is_verify_coarse_pv) {
      can_verify = false;
      logQuda(QUDA_VERBOSE,
              "Performing a custom verify for coarse pv^dagger dwf verify: reconstructing coarse dwf from the "
              "multi-rhs coarse + chiral projectors\n");

      // check 4-d dslash
      {
        printfQuda("Checking the internal 4-d coarse operator\n");

        // make sure diracCoarseNull is the coarse operator
        if (diracCoarseNull->getDiracType() != QUDA_COARSE_DIRAC)
          errorQuda("  Unexpected Dirac type %d", diracCoarseNull->getDiracType());

        // create 4-d coarse vectors
        ColorSpinorParam csParam(x_coarse[0]);
        csParam.nDim = 4;
        csParam.x[4] = 1;
        csParam.create = QUDA_NULL_FIELD_CREATE;

        auto coarse_4_rhs = ColorSpinorField(csParam);
        std::vector<ColorSpinorField> coarse_4_lhs(2); // create two fields
        for (auto &f : coarse_4_lhs) f = ColorSpinorField(csParam);

        // populate the coarse rhs with random noise
        spinorNoise(coarse_4_rhs, *rng, QUDA_NOISE_UNIFORM);

        // create the set of 4-d fine vectors
        ColorSpinorParam csParamFine(tmp2);
        csParamFine.create = QUDA_NULL_FIELD_CREATE;
        if (param.level == 1)
          csParamFine.gammaBasis = (param.level == 1) ? QUDA_UKQCD_GAMMA_BASIS : QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
        auto fine_4_rhs = ColorSpinorField(csParamFine);
        auto fine_4_lhs = ColorSpinorField(csParamFine);

        // First verify that the coarse operator is good

        // emulated R D P
        transfer->P(fine_4_rhs, coarse_4_rhs);
        (*param.matNull)(fine_4_lhs, fine_4_rhs);
        transfer->R(coarse_4_lhs[0], fine_4_lhs);

        // coarse operator
        diracCoarseNull->M(coarse_4_lhs[1], coarse_4_rhs);

        // check
        double r_nrm = norm2(coarse_4_lhs[1]);
        auto max_deviation = blas::max_deviation(coarse_4_lhs[1], coarse_4_lhs[0]);
        auto l2_deviation = sqrt(xmyNorm(coarse_4_lhs[0], coarse_4_lhs[1]) / norm2(coarse_4_lhs[0]));

        logQuda(QUDA_VERBOSE, "  4-d L2 norms: Emulated = %e, Native = %e; Deviations: L2 relative = %e, max = %e\n",
                norm2(x_coarse[0]), r_nrm, l2_deviation, max_deviation[0]);

        if (check_deviation(l2_deviation, tol))
          errorQuda("  4-d Coarse operator failed: L2 relative deviation = %e > %e", l2_deviation, tol);
        if (check_deviation(max_deviation[0], tol))
          warningQuda("  4-d Coarse operator failed: max deviation = %e > %e", max_deviation[0], tol);
      }

      // check 5-d dslash
      {
        printfQuda("Checking the 5-d coarse dwf operator\n");

        // make sure diracCoarseResidual is the coarse pv operator
        if (diracCoarseResidual->getDiracType() != QUDA_COARSEPV_DIRAC)
          errorQuda("  Unexpected Dirac type %d", diracCoarseResidual->getDiracType());

        // create 5-d coarse vectors
        ColorSpinorParam csParam(x_coarse[0]);
        csParam.nDim = 5;
        csParam.x[4] = diracResidual->getLs();
        csParam.create = QUDA_NULL_FIELD_CREATE;
        csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

        auto coarse_5d_rhs = ColorSpinorField(csParam);
        std::vector<ColorSpinorField> coarse_5d_lhs(2); // create two fields
        for (auto &f : coarse_5d_lhs) f = ColorSpinorField(csParam);

        // populate the coarse rhs with random noise
        spinorNoise(coarse_5d_rhs, *rng, QUDA_NOISE_UNIFORM);

        // create 5-d fine vectors
        ColorSpinorParam csParamFine(tmp2);
        csParamFine.nDim = 5;
        csParamFine.x[4] = diracCoarseResidual->getLs();
        csParamFine.create = QUDA_NULL_FIELD_CREATE;
        csParamFine.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

        auto fine_5d_rhs = ColorSpinorField(csParamFine);
        auto fine_5d_lhs = ColorSpinorField(csParamFine);

        // emulated

        transfer->P(fine_5d_rhs, coarse_5d_rhs);

        // fine operator, which has different paths depending on if it's fine or coarse
        if (diracSmoother->getDiracType() == QUDA_COARSE_DIRAC) {
          errorQuda("  \"Fine\" PV level being coarse isn't supported yet");
        } else {
          // we need an extra basis change...
          csParamFine.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
          auto fine_5d_rhs_inter = ColorSpinorField(csParamFine);
          auto fine_5d_lhs_inter = ColorSpinorField(csParamFine);

          blas::copy(fine_5d_rhs_inter, fine_5d_rhs);
          if (diracSmoother->getDiracType() == QUDA_DOMAIN_WALL_4DPV_DIRAC) {
            static_cast<const DiracDomainWall4DPV *>(diracSmoother)->ApplyMDwf(fine_5d_lhs_inter, fine_5d_rhs_inter);
          } else {
            //errorQuda("  The coarse MobiusPV op has not been implemented yet");
            static_cast<const DiracMobiusPV*>(diracSmoother)->ApplyMDwf(fine_5d_lhs_inter, fine_5d_rhs_inter);
          }

          blas::copy(fine_5d_lhs, fine_5d_lhs_inter);
        }

        transfer->R(coarse_5d_lhs[0], fine_5d_lhs);

        // non-emulated
        static_cast<DiracCoarsePV *>(diracCoarseResidual)->ApplyMDwf(coarse_5d_lhs[1], coarse_5d_rhs);

        // check
        double r_nrm = norm2(coarse_5d_lhs[1]);
        auto max_deviation = blas::max_deviation(coarse_5d_lhs[1], coarse_5d_lhs[0]);
        auto l2_deviation = sqrt(xmyNorm(coarse_5d_lhs[0], coarse_5d_lhs[1]) / norm2(coarse_5d_lhs[0]));

        logQuda(QUDA_VERBOSE, "  5-d L2 norms: Emulated = %e, Native = %e; Deviations: L2 relative = %e, max = %e\n",
                norm2(coarse_5d_lhs[0]), r_nrm, l2_deviation, max_deviation[0]);

        if (check_deviation(l2_deviation, tol))
          errorQuda("  5-d Coarse operator failed: L2 relative deviation = %e > %e", l2_deviation, tol);
        if (check_deviation(max_deviation[0], tol))
          warningQuda("  5-d Coarse operator failed: max deviation = %e > %e", max_deviation[0], tol);
      }

      // check 5-d PV operator
      {
        printfQuda("Checking the 5-d coarse PV operator\n");

        // make sure diracCoarseResidual is the coarse pv operator
        if (diracCoarseResidual->getDiracType() != QUDA_COARSEPV_DIRAC)
          errorQuda("  Unexpected Dirac type %d", diracCoarseResidual->getDiracType());

        // create 5-d coarse vectors
        ColorSpinorParam csParam(x_coarse[0]);
        csParam.nDim = 5;
        csParam.x[4] = diracResidual->getLs();
        csParam.create = QUDA_NULL_FIELD_CREATE;
        csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

        auto coarse_5d_rhs = ColorSpinorField(csParam);
        std::vector<ColorSpinorField> coarse_5d_lhs(2); // create two fields
        for (auto &f : coarse_5d_lhs) f = ColorSpinorField(csParam);

        // populate the coarse rhs with random noise
        spinorNoise(coarse_5d_rhs, *rng, QUDA_NOISE_UNIFORM);

        // create 5-d fine vectors
        ColorSpinorParam csParamFine(tmp2);
        csParamFine.nDim = 5;
        csParamFine.x[4] = diracCoarseResidual->getLs();
        csParamFine.create = QUDA_NULL_FIELD_CREATE;
        csParamFine.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

        auto fine_5d_rhs = ColorSpinorField(csParamFine);
        auto fine_5d_lhs = ColorSpinorField(csParamFine);

        // emulated

        transfer->P(fine_5d_rhs, coarse_5d_rhs);

        // fine operator, which has different paths depending on if it's fine or coarse
        if (diracSmoother->getDiracType() == QUDA_COARSE_DIRAC) {
          errorQuda("  \"Fine\" PV level being coarse isn't supported yet");
        } else {
          // we need an extra basis change...
          csParamFine.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
          auto fine_5d_rhs_inter = ColorSpinorField(csParamFine);
          auto fine_5d_lhs_inter = ColorSpinorField(csParamFine);

          blas::copy(fine_5d_rhs_inter, fine_5d_rhs);
          if (diracSmoother->getDiracType() == QUDA_DOMAIN_WALL_4DPV_DIRAC) {
            static_cast<const DiracDomainWall4DPV *>(diracSmoother)->ApplyPVDagger(fine_5d_lhs_inter, fine_5d_rhs_inter);
          } else {
            //errorQuda("  The coarse MobiusPV op has not been implemented yet");
            static_cast<const DiracMobiusPV*>(diracSmoother)->ApplyPVDagger(fine_5d_lhs_inter, fine_5d_rhs_inter);
          }

          blas::copy(fine_5d_lhs, fine_5d_lhs_inter);
        }

        transfer->R(coarse_5d_lhs[0], fine_5d_lhs);

        // non-emulated
        static_cast<DiracCoarsePV *>(diracCoarseResidual)->ApplyPVDagger(coarse_5d_lhs[1], coarse_5d_rhs);

        // check
        double r_nrm = norm2(coarse_5d_lhs[1]);
        auto max_deviation = blas::max_deviation(coarse_5d_lhs[1], coarse_5d_lhs[0]);
        auto l2_deviation = sqrt(xmyNorm(coarse_5d_lhs[0], coarse_5d_lhs[1]) / norm2(coarse_5d_lhs[0]));

        logQuda(QUDA_VERBOSE, "  5-d PV L2 norms: Emulated = %e, Native = %e; Deviations: L2 relative = %e, max = %e\n",
                norm2(coarse_5d_lhs[0]), r_nrm, l2_deviation, max_deviation[0]);

        if (check_deviation(l2_deviation, tol))
          errorQuda("  5-d Coarse PV operator failed: L2 relative deviation = %e > %e", l2_deviation, tol);
        if (check_deviation(max_deviation[0], tol))
          warningQuda("  5-d Coarse PV operator failed: max deviation = %e > %e", max_deviation[0], tol);
      }

    } else if (diracSmoother->getDiracType() == QUDA_ASQTAD_DIRAC || diracSmoother->getDiracType() == QUDA_ASQTADKD_DIRAC
               || diracSmoother->getDiracType() == QUDA_ASQTADPC_DIRAC) {
      // If we're doing anything with the asqtad operator, the long links can make verification difficult

      if (param.transfer_type == QUDA_TRANSFER_COARSE_KD || param.transfer_type == QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG) {
        can_verify = false;
        logQuda(QUDA_VERBOSE, "Using the naively coarsened KD operator with asqtad long links, skipping verify...\n");
      } else if (param.transfer_type == QUDA_TRANSFER_AGGREGATE || param.transfer_type == QUDA_TRANSFER_OPTIMIZED_KD) {
        // need to see if the aggregate is smaller than 3 in any direction
        for (int d = 0; d < 4; d++) {
          if (param.mg_global.geo_block_size[param.level][d] < 3) {
            can_verify = false;
            logQuda(QUDA_VERBOSE,
                    "Aggregation geo_block_size[%d] = %d is less than 3, skipping verify for asqtad coarsen...\n", d,
                    param.mg_global.geo_block_size[param.level][d]);
          }
        }
      }
    }

    if (can_verify) {

      if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) {
        double kappa = diracResidual->Kappa();
        double mass = diracResidual->Mass();
        if (param.level == 0) {
          if (tmp1.Nspin() == 4) {
            diracSmoother->DslashXpay(tmp2.Even(), tmp1.Odd(), QUDA_EVEN_PARITY, tmp1.Even(), -kappa);
            diracSmoother->DslashXpay(tmp2.Odd(), tmp1.Even(), QUDA_ODD_PARITY, tmp1.Odd(), -kappa);
          } else if (tmp1.Nspin() == 2) { // if the coarse op is on top
            diracSmoother->DslashXpay(tmp2.Even(), tmp1.Odd(), QUDA_EVEN_PARITY, tmp1.Even(), 1.0);
            diracSmoother->DslashXpay(tmp2.Odd(), tmp1.Even(), QUDA_ODD_PARITY, tmp1.Odd(), 1.0);
          } else { // staggered
            diracSmoother->DslashXpay(tmp2.Even(), tmp1.Odd(), QUDA_EVEN_PARITY, tmp1.Even(),
                                      2.0 * mass); // stag convention
            diracSmoother->DslashXpay(tmp2.Odd(), tmp1.Even(), QUDA_ODD_PARITY, tmp1.Odd(),
                                      2.0 * mass); // stag convention
          }
        } else { // this is a hack since the coarse Dslash doesn't properly use the same xpay conventions yet
          diracSmoother->DslashXpay(tmp2.Even(), tmp1.Odd(), QUDA_EVEN_PARITY, tmp1.Even(), 1.0);
          diracSmoother->DslashXpay(tmp2.Odd(), tmp1.Even(), QUDA_ODD_PARITY, tmp1.Odd(), 1.0);
        }
      } else {
        (*param.matResidual)(tmp2, tmp1);
      }

      transfer->R(x_coarse[0], tmp2);
      static_cast<DiracCoarse *>(diracCoarseResidual)->M(r_coarse[0], tmp_coarse);

#if 0 // enable to print out emulated and actual coarse-grid operator vectors for debugging
      setOutputPrefix("");

      for (unsigned int rank = 0; rank < comm_size(); rank++) { // this ensures that we print each rank in order
        comm_barrier();
        printfQuda("\nemulated\n");
        comm_barrier();
        for (int parity = 0; parity < 2; parity++)
          for (unsigned int x_cb = 0; x_cb < x_coarse[0].VolumeCB(); x_cb++) x_coarse[0].PrintVector(parity, x_cb, rank);

        comm_barrier();
        printfQuda("\nactual\n");
        comm_barrier();
        for (int parity = 0; parity < 2; parity++)
          for (unsigned int x_cb = 0; x_cb < r_coarse[0].VolumeCB(); x_cb++) r_coarse[0].PrintVector(parity, x_cb, rank);
      }
      setOutputPrefix(prefix);
#endif

      double r_nrm = norm2(r_coarse[0]);
      auto max_deviation = blas::max_deviation(r_coarse[0], x_coarse[0]);
      auto l2_deviation = sqrt(xmyNorm(x_coarse[0], r_coarse[0]) / norm2(x_coarse[0]));

      if (diracResidual->Mu() != 0.0) {
        // When the mu is shifted on the coarse level; we can compute exactly the error we introduce in the check:
        //  it is given by 2*kappa*delta_mu || tmp_coarse ||; where tmp_coarse is the random vector generated for the test
        double delta_factor = param.mg_global.mu_factor[param.level + 1] - param.mg_global.mu_factor[param.level];
        if (fabs(delta_factor) > tol) {
          double delta_a
            = delta_factor * 2.0 * diracResidual->Kappa() * diracResidual->Mu() * transfer->Vectors().TwistFlavor();
          l2_deviation -= fabs(delta_a) * sqrt(norm2(tmp_coarse) / norm2(x_coarse[0]));
          l2_deviation = fabs(l2_deviation);
          max_deviation[0] -= fabs(delta_a);
        }
      }
      logQuda(QUDA_VERBOSE, "L2 norms: Emulated = %e, Native = %e; Deviations: L2 relative = %e, max = %e\n",
              norm2(x_coarse[0]), r_nrm, l2_deviation, max_deviation[0]);

      if (check_deviation(l2_deviation, tol))
        errorQuda("Coarse operator failed: L2 relative deviation = %e > %e", l2_deviation, tol);
      if (check_deviation(max_deviation[0], tol))
        warningQuda("Coarse operator failed: max deviation = %e > %e", max_deviation[0], tol);
    }

    // check the preconditioned operator construction on the lower level if applicable
    bool coarse_was_preconditioned = (param.mg_global.coarse_grid_solution_type[param.level + 1] == QUDA_MATPC_SOLUTION
                                      && param.mg_global.smoother_solve_type[param.level + 1] == QUDA_DIRECT_PC_SOLVE);
    if (coarse_was_preconditioned) {
      // check eo
      logQuda(QUDA_SUMMARIZE, "Checking Deo of preconditioned operator 0 = \\hat{D}_c - A^{-1} D_c\n");
      static_cast<DiracCoarse *>(diracCoarseResidual)->Dslash(r_coarse[0].Even(), tmp_coarse.Odd(), QUDA_EVEN_PARITY);
      static_cast<DiracCoarse *>(diracCoarseResidual)->CloverInv(x_coarse[0].Even(), r_coarse[0].Even(), QUDA_EVEN_PARITY);
      static_cast<DiracCoarsePC *>(diracCoarseSmoother)->Dslash(r_coarse[0].Even(), tmp_coarse.Odd(), QUDA_EVEN_PARITY);
      double r_nrm = norm2(r_coarse[0].Even());
      auto max_deviation = blas::max_deviation(r_coarse[0].Even(), x_coarse[0].Even());
      auto l2_deviation = sqrt(xmyNorm(x_coarse[0].Even(), r_coarse[0].Even()) / norm2(x_coarse[0].Even()));
      logQuda(QUDA_VERBOSE, "L2 norms: Emulated = %e, Native = %e; Deviations: L2 relative = %e, max = %e\n",
              norm2(x_coarse[0].Even()), r_nrm, l2_deviation, max_deviation[0]);
      if (check_deviation(l2_deviation, tol))
        errorQuda("Preconditioned Deo failed: L2 relative deviation = %e > %e", l2_deviation, tol);
      if (check_deviation(max_deviation[0], tol))
        errorQuda("Preconditioned Deo failed: max deviation = %e > %e", max_deviation[0], tol);

      // check Doe
      logQuda(QUDA_SUMMARIZE, "Checking Doe of preconditioned operator 0 = \\hat{D}_c - A^{-1} D_c\n");
      static_cast<DiracCoarse *>(diracCoarseResidual)->Dslash(r_coarse[0].Odd(), tmp_coarse.Even(), QUDA_ODD_PARITY);
      static_cast<DiracCoarse *>(diracCoarseResidual)->CloverInv(x_coarse[0].Odd(), r_coarse[0].Odd(), QUDA_ODD_PARITY);
      static_cast<DiracCoarsePC *>(diracCoarseSmoother)->Dslash(r_coarse[0].Odd(), tmp_coarse.Even(), QUDA_ODD_PARITY);
      r_nrm = norm2(r_coarse[0].Odd());
      max_deviation = blas::max_deviation(r_coarse[0].Odd(), x_coarse[0].Odd());
      l2_deviation = sqrt(xmyNorm(x_coarse[0].Odd(), r_coarse[0].Odd()) / norm2(x_coarse[0].Odd()));
      logQuda(QUDA_VERBOSE, "L2 norms: Emulated = %e, Native = %e; Deviations: L2 relative = %e, max = %e\n",
              norm2(x_coarse[0].Odd()), r_nrm, l2_deviation, max_deviation[0]);
      if (check_deviation(l2_deviation, tol))
        errorQuda("Preconditioned Doe failed: L2 relative deviation = %e > %e", l2_deviation, tol);
      if (check_deviation(max_deviation[0], tol))
        errorQuda("Preconditioned Doe failed: max deviation = %e > %e", max_deviation[0], tol);
    }

    // here we check that the Hermitian conjugate operator is working
    // as expected for both the smoother and residual Dirac operators
    if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) {
      logQuda(QUDA_SUMMARIZE, "Checking normality of preconditioned operator\n");
      if (tmp2.Nspin() == 1) { // if the outer op is the staggered op, just use M.
        diracSmoother->M(tmp2.Even(), tmp1.Odd());
      } else {
        diracSmoother->MdagM(tmp2.Even(), tmp1.Odd());
      }
      Complex dot = cDotProduct(tmp2.Even(), tmp1.Odd());
      double deviation = std::fabs(dot.imag()) / std::fabs(dot.real());
      logQuda(QUDA_VERBOSE,
              "Smoother normal operator test (eta^dag M^dag M eta): real=%e imag=%e, relative imaginary deviation=%e\n",
              real(dot), imag(dot), deviation);
      if (check_deviation(deviation, tol))
        errorQuda("Smoother operator normality failed: deviation = %e > %e", deviation, tol);
    }

    { // normal operator check for residual operator
      logQuda(QUDA_SUMMARIZE, "Checking normality of residual operator\n");
      Complex dot;
      if (diracResidual->getLs() != 1) {
        // dwf pv, create two temporary 

        ColorSpinorParam csParamFine(tmp2);
        csParamFine.nDim = 5;
        csParamFine.x[4] = diracResidual->getLs();
        csParamFine.create = QUDA_NULL_FIELD_CREATE;
        csParamFine.gammaBasis = (param.level == 0 || param.level == 1) ? QUDA_UKQCD_GAMMA_BASIS : QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

        // prepare vectors of fine 4-d fields for tmp2
        auto tmp1_5d = ColorSpinorField(csParamFine);
        auto tmp2_5d = ColorSpinorField(csParamFine);

        spinorNoise(tmp1_5d, *rng, QUDA_NOISE_UNIFORM);

        diracResidual->MdagM(tmp2_5d, tmp1_5d);

        dot = cDotProduct(tmp1_5d, tmp2_5d);
      } else if (tmp2.Nspin() != 1 || tmp2.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
        diracResidual->MdagM(tmp2, tmp1);
        dot = cDotProduct(tmp1, tmp2);
      } else {
        // staggered preconditioned op.
        diracResidual->M(tmp2, tmp1);
        dot = cDotProduct(tmp1, tmp2);
      }

      double deviation = std::fabs(dot.imag()) / std::fabs(dot.real());
      logQuda(QUDA_VERBOSE,
              "Normal operator test (eta^dag M^dag M eta): real=%e imag=%e, relative imaginary deviation=%e\n",
              real(dot), imag(dot), deviation);
      if (check_deviation(deviation, tol))
        errorQuda("Residual operator normality failed: deviation = %e > %e", deviation, tol);
    }

    // Not useful for staggered op since it's a unitary transform
    if (param.transfer_type == QUDA_TRANSFER_AGGREGATE) {
      if (param.mg_global.run_low_mode_check) {

        sprintf(prefix, "MG level %d (%s): eigenvector overlap : ", param.level + 1,
                param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");
        setOutputPrefix(prefix);

        // Reuse the space for the Null vectors. By this point,
        // the coarse grid has already been constructed.
        generateEigenVectors(param.B);

        for (int i = 0; i < param.Nvec; i++) {

          // Restrict Evec, place result in r_coarse
          transfer->R(r_coarse[0], param.B[i]);
          // Prolong r_coarse, place result in tmp2
          transfer->P(tmp2, r_coarse[0]);

          printfQuda("Vector %d: norms v_k = %e P^dag v_k = %e PP^dag v_k = %e\n", i, B_norm[i], norm2(r_coarse[0]),
                     norm2(tmp2));

          // Compare v_k and PP^dag v_k.
          auto max_deviation = blas::max_deviation(tmp2, param.B[i]);
          auto l2_deviation = sqrt(xmyNorm(param.B[i], tmp2) / B_norm[i]);
          printfQuda("L2 relative deviation = %e max deviation = %e\n", l2_deviation, max_deviation[0]);

          if (param.mg_global.run_oblique_proj_check) {

            sprintf(prefix, "MG level %d (%s): eigenvector Oblique Projections : ", param.level + 1,
                    param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");
            setOutputPrefix(prefix);

            // Oblique projections
            logQuda(QUDA_SUMMARIZE, "Checking 1 > || (1 - DP(P^dagDP)P^dag) v_k || / || v_k || for vector %d\n", i);

            transfer->R(r_coarse[0], param.B[i]);
            (*coarse_solver)(x_coarse[0], r_coarse[0]); // this needs to be an exact solve to pass
            setOutputPrefix(prefix);                // restore prefix after return from coarse grid
            transfer->P(tmp2, x_coarse[0]);
            (*param.matResidual)(tmp1, tmp2);

            logQuda(QUDA_SUMMARIZE, "Vector %d: norms v_k %e DP(P^dagDP)P^dag v_k %e\n", i, B_norm[i], norm2(tmp1));
            max_deviation = blas::max_deviation(tmp1, param.B[i]);
            logQuda(QUDA_SUMMARIZE, "L2 relative deviation = %e, max deviation = %e\n",
                    sqrt(xmyNorm(param.B[i], tmp1) / B_norm[i]), max_deviation[0]);
          }

          sprintf(prefix, "MG level %d (%s): ", param.level + 1,
                  param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");
          setOutputPrefix(prefix);
        }
      }
    }

    if (recursively && param.level < param.Nlevel - 2) coarse->verify(true);

    popLevel();
  }

  void MG::verifyDwfPV()
  {
    // quickly make sure diracCoarseNull is the Wilson operator
    if (diracCoarseNull->getDiracType() != QUDA_WILSON_DIRAC)
      errorQuda("Unexpected Dirac type %d", diracCoarseNull->getDiracType());

    const DiracWilson *d_wilson = reinterpret_cast<const DiracWilson *>(diracCoarseNull);

    // get Ls
    auto Ls = r_coarse[0].X(4);

    // we've verified that this is a DWF operator; grab the mass and m5
    auto mass = reinterpret_cast<const DiracDomainWall *>(diracSmoother)->Mass();
    auto m5 = reinterpret_cast<const DiracDomainWall *>(diracSmoother)->M5();

    logQuda(QUDA_VERBOSE, "Ls %d mass %f m5 %f\n", Ls, mass, m5);

    // prepare extra coarse vectors
    ColorSpinorParam coarse_param(r_coarse[0]);

    // change the basis for verification
    coarse_param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

    // create a random rhs
    coarse_param.create = QUDA_NULL_FIELD_CREATE;
    ColorSpinorField rhs(coarse_param);
    spinorNoise(rhs, *rng, QUDA_NOISE_UNIFORM);

    // place to store underlying "M" times rhs
    coarse_param.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField dwf_lhs(coarse_param);

    // place to store applying the dwf operator built from Wilson bits
    ColorSpinorField emul_lhs(coarse_param);

    // create a set of 4-d vectors
    ColorSpinorParam csParam(rhs);
    csParam.nDim = 4;
    csParam.x[4] = 1;
    csParam.create = QUDA_NULL_FIELD_CREATE;
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

    // prepare vectors of 4-d fields for the rhs and emulated lhs
    auto rhs_4d_ = getFieldTmp<ColorSpinorField>(Ls, csParam);
    auto emul_lhs_4d_ = getFieldTmp<ColorSpinorField>(Ls, csParam);
    cvector_ref<ColorSpinorField> rhs_4d(rhs_4d_), emul_lhs_4d(emul_lhs_4d_);

    // split rhs
    Split5DTo4DFields(rhs_4d, rhs);

    // prepare vectors to hold intermediate chiral projections
    auto chiral_plus_4d_ = getFieldTmp<ColorSpinorField>(Ls, csParam);
    auto chiral_minus_4d_ = getFieldTmp<ColorSpinorField>(Ls, csParam);
    cvector_ref<ColorSpinorField> chiral_plus(chiral_plus_4d_), chiral_minus(chiral_minus_4d_);

    // check 5-d dslash
    {
      printfQuda("Checking the 5-d Pauli-Villars operator\n");

      if (diracSmoother->getDiracType() == QUDA_DOMAIN_WALL_4D_DIRAC) {
        auto kappa5 = 0.5 / (5.0 + m5);

        // apply the exact DiracDomainWall4D::ApplyMDwf
        reinterpret_cast<const DiracDomainWall4DPV *>(diracCoarseSmoother)->ApplyMDwf(dwf_lhs, rhs);

        // This bit is equivalent to the DWF call:
        // ApplyDomainWall4D(out, in, *gauge, 0.0, 0.0, nullptr, nullptr, in, QUDA_INVALID_PARITY, dagger, commDim.data,
        //                   profile);

        d_wilson->Dslash(emul_lhs_4d.Even(), rhs_4d.Odd(), QUDA_EVEN_PARITY);
        d_wilson->Dslash(emul_lhs_4d.Odd(), rhs_4d.Even(), QUDA_ODD_PARITY);

        // This next block is equivalent to the DWF call:
        // ApplyDslash5(out, in, out, mass, 0.0, nullptr, nullptr, 1.0, dagger, Dslash5Type::DSLASH5_DWF);

        ApplyChiralProj(chiral_plus, rhs_4d, +1);  // for the backwards direction
        ApplyChiralProj(chiral_minus, rhs_4d, -1); // for the forwards direction
        for (int s = 0; s < Ls; s++) {
          // forwards direction
          blas::axpy((s == Ls - 1) ? -mass : 1, chiral_minus[(s + 1) % Ls], emul_lhs_4d[s]);
          // backwards direction
          blas::axpy((s == 0) ? -mass : 1, chiral_plus[(s + Ls - 1) % Ls], emul_lhs_4d[s]);
        }

        // This last bit is equivalent to the call:
        // blas::xpay(in, -kappa5, out);

        blas::xpay(rhs_4d, -kappa5, emul_lhs_4d);

      } else if (diracSmoother->getDiracType() == QUDA_MOBIUS_DOMAIN_WALL_DIRAC) {
        auto b_5 = reinterpret_cast<const DiracMobius *>(diracSmoother)->getB5();
        auto c_5 = reinterpret_cast<const DiracMobius *>(diracSmoother)->getC5();
        double mobius_kappa_b = 0.5 / (b_5[0].real() * (4.0 + m5) + 1.0);

        // from compute_coeff_mobius_pre and compute_coeff_mobius
        std::array<double, QUDA_MAX_DWF_LS> kappa, alpha, beta;

        // apply the exact DiracMobius::ApplyMDwf
        reinterpret_cast<const DiracMobiusPV *>(diracCoarseSmoother)->ApplyMDwf(dwf_lhs, rhs);

        // create a temporary
        auto tmp_4d_ = getFieldTmp<ColorSpinorField>(Ls, csParam);
        vector_ref<ColorSpinorField> tmp_4d(tmp_4d_);

        // This bit is equivalent to the following Mobius call:
        // ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, Dslash5Type::DSLASH5_MOBIUS_PRE);

        // from compute_coeff_mobius_pre
        for (int s = 0; s < Ls; s++) {
          beta[s] = b_5[s].real();
          alpha[s] = 0.5 * c_5[s].real(); // 0.5 from gamma matrices
        }

        ApplyChiralProj(chiral_plus, rhs_4d, +1);  // for the backwards direction
        ApplyChiralProj(chiral_minus, rhs_4d, -1); // for the forwards direction
        for (int s = 0; s < Ls; s++) {
          // forwards direction
          blas::axy(alpha[s] * ((s == Ls - 1) ? -mass : 1), chiral_minus[(s + 1) % Ls], emul_lhs_4d[s]);
          // backwards direction
          blas::axpy(alpha[s] * ((s == 0) ? -mass : 1), chiral_plus[(s + Ls - 1) % Ls], emul_lhs_4d[s]);
          // diagonal contribution
          blas::axpy(beta[s], rhs_4d[s], emul_lhs_4d[s]);
        }

        // This bit is equivalent to the next Mobius call:
        // ApplyDomainWall4D(tmp, out, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, dagger, commDim.data, profile);

        // a = 0; xpay false; much simpler than it looks
        d_wilson->Dslash(tmp_4d.Even(), emul_lhs_4d.Odd(), QUDA_EVEN_PARITY);
        d_wilson->Dslash(tmp_4d.Odd(), emul_lhs_4d.Even(), QUDA_ODD_PARITY);

        // This bit is equivalent to the last Mobius call:
        // ApplyDslash5(out, in, in, mass, m5, b_5, c_5, 0.0, dagger, Dslash5Type::DSLASH5_MOBIUS);

        // from compute_coeff_mobius
        for (int s = 0; s < Ls; s++) {
          kappa[s]
            = 0.5 * (c_5[s].real() * (m5 + 4.0) - 1.0) / (b_5[s].real() * (m5 + 4.0) + 1.0); // 0.5 from gamma matrices
        }

        // seems to be the same chiral projectors?
        for (int s = 0; s < Ls; s++) {
          // forwards direction
          blas::axy(kappa[s] * ((s == Ls - 1) ? -mass : 1), chiral_minus[(s + 1) % Ls], emul_lhs_4d[s]);
          // backwards direction
          blas::axpy(kappa[s] * ((s == 0) ? -mass : 1), chiral_plus[(s + Ls - 1) % Ls], emul_lhs_4d[s]);
          // diagonal contribution
          blas::axpy(1.0, rhs_4d[s], emul_lhs_4d[s]);
        }

        // last, but not least, this call
        // blas::axpy(-mobius_kappa_b, tmp, out);

        blas::axpy(-mobius_kappa_b, tmp_4d, emul_lhs_4d);
      }

      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
        // for debugging purposes
        // std::vector<ColorSpinorField> dwf_lhs_4(Ls);
        // for (auto &f : dwf_lhs_4) f = ColorSpinorField(csParam);
        // auto dwf_lhs_4d = vector_ref<ColorSpinorField>(dwf_lhs_4);
        auto dwf_lhs_4d_ = getFieldTmp<ColorSpinorField>(Ls, csParam);
        vector_ref<ColorSpinorField> dwf_lhs_4d(dwf_lhs_4d_);

        Split5DTo4DFields(dwf_lhs_4d, dwf_lhs);

        // in theory emul_lhs contains the same thing as dwf_lhs on a per-component basis
        for (int s = 0; s < Ls; s++)
          printfQuda("s %d norm2 source %e norm2 chiral+ %e norm2 chiral- %e norm2 dwf %e norm2 emul %e\n", s,
                    blas::norm2(rhs_4d[s]), blas::norm2(chiral_plus[s]), blas::norm2(chiral_minus[s]),
                    blas::norm2(dwf_lhs_4d[s]), blas::norm2(emul_lhs_4d[s]));
      }

      // re-join
      Join4DTo5DField(emul_lhs, emul_lhs_4d);

      // verify
      double emul_nrm2 = blas::norm2(emul_lhs);
      double native_nrm2 = blas::norm2(dwf_lhs);
      auto max_deviation = blas::max_deviation(dwf_lhs, emul_lhs);
      auto l2_deviation = sqrt(xmyNorm(dwf_lhs, emul_lhs) / native_nrm2);

      logQuda(QUDA_VERBOSE, "L2 norms: Emulated = %e, Native = %e; Deviations: L2 relative = %e, max = %e\n", emul_nrm2,
              native_nrm2, l2_deviation, max_deviation[0]);

      // may want to revisit this---these were relaxed for cases where ghost_precision < precision
      // these were set while hacking in tests of quarter precision ghosts
      // moreover, we can improve the precision of block ortho with a tighter max than 1.0
      QudaPrecision prec
        = std::min(param.mg_global.precision_null[param.level], param.mg_global.invert_param->cuda_prec_sloppy);

      double tol;
      switch (prec) {
      case QUDA_QUARTER_PRECISION: tol = 5e-2; break;
      case QUDA_HALF_PRECISION: tol = 5e-2; break;
      case QUDA_SINGLE_PRECISION: tol = 2e-3; break;
      default: tol = 1e-8;
      }

      if (check_deviation(l2_deviation, tol))
        errorQuda("Coarse operator failed: L2 relative deviation = %e > %e", l2_deviation, tol);
      if (check_deviation(max_deviation[0], tol))
        warningQuda("Coarse operator failed: max deviation = %e > %e", max_deviation[0], tol);
    } // end of check 5-d Pauli-Villars operator

    // check 5-d Pauli-Villars dagger operator
    {
      printfQuda("Checking the 5-d Pauli-Villars dagger operator\n");

      double mass_pv = 1.0;

      if (diracSmoother->getDiracType() == QUDA_DOMAIN_WALL_4D_DIRAC) {
        auto kappa5 = 0.5 / (5.0 + m5);

        // apply the exact DiracDomainWall4D::ApplyPVDagger
        reinterpret_cast<const DiracDomainWall4DPV *>(diracCoarseSmoother)->ApplyPVDagger(dwf_lhs, rhs);

        d_wilson->flipDagger();
        d_wilson->Dslash(emul_lhs_4d.Even(), rhs_4d.Odd(), QUDA_EVEN_PARITY);
        d_wilson->Dslash(emul_lhs_4d.Odd(), rhs_4d.Even(), QUDA_ODD_PARITY);
        d_wilson->flipDagger();

        // This next block is equivalent to the DWF call:
        // ApplyDslash5(out, in, out, mass_pv, 0.0, nullptr, nullptr, 1.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_DWF);
        // the dagger version of the projector is the same as the non-dagger version, but with the direction of the projector flipped
        
        ApplyChiralProj(chiral_plus, rhs_4d, +1);  // for the backwards direction
        ApplyChiralProj(chiral_minus, rhs_4d, -1); // for the forwards direction
        for (int s = 0; s < Ls; s++) {
          // forwards direction
          blas::axpy((s == Ls - 1) ? -mass_pv : 1, chiral_plus[(s + 1) % Ls], emul_lhs_4d[s]);
          // backwards direction
          blas::axpy((s == 0) ? -mass_pv : 1, chiral_minus[(s + Ls - 1) % Ls], emul_lhs_4d[s]);
        }

        // This last bit is equivalent to the call:
        // blas::xpay(in, -kappa5, out);

        blas::xpay(rhs_4d, -kappa5, emul_lhs_4d);

      } else if (diracSmoother->getDiracType() == QUDA_MOBIUS_DOMAIN_WALL_DIRAC) {
        auto b_5 = reinterpret_cast<const DiracMobius *>(diracSmoother)->getB5();
        auto c_5 = reinterpret_cast<const DiracMobius *>(diracSmoother)->getC5();
        double mobius_kappa_b = 0.5 / (b_5[0].real() * (4.0 + m5) + 1.0);

        // from compute_coeff_mobius_pre and compute_coeff_mobius
        std::array<double, QUDA_MAX_DWF_LS> kappa, alpha, beta;

        // apply the exact DiracMobius::ApplyPVDagger
        reinterpret_cast<const DiracMobiusPV *>(diracCoarseSmoother)->ApplyPVDagger(dwf_lhs, rhs);

        // create a temporary
        auto tmp_4d_ = getFieldTmp<ColorSpinorField>(Ls, csParam);
        vector_ref<ColorSpinorField> tmp_4d(tmp_4d_);

        // This bit is equivalent:
        // ApplyDomainWall4D(out, in, *gauge, 0.0, m5, b_5, c_5, in, QUDA_INVALID_PARITY, QUDA_DAG_YES, commDim.data, profile);

        // a = 0; xpay false; much simpler than it looks
        d_wilson->flipDagger();
        d_wilson->Dslash(emul_lhs_4d.Even(), rhs_4d.Odd(), QUDA_EVEN_PARITY);
        d_wilson->Dslash(emul_lhs_4d.Odd(), rhs_4d.Even(), QUDA_ODD_PARITY);
        d_wilson->flipDagger();

        
        // This bit is equivalent to the following Mobius call:
        // ApplyDslash5(tmp, out, in, mass_pv, m5, b_5, c_5, 0.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_MOBIUS_PRE);

        // from compute_coeff_mobius_pre
        for (int s = 0; s < Ls; s++) {
          beta[s] = b_5[s].real();
          alpha[s] = 0.5 * c_5[s].real(); // 0.5 from gamma matrices
        }

        ApplyChiralProj(chiral_plus, emul_lhs_4d, +1);  // for the backwards direction
        ApplyChiralProj(chiral_minus, emul_lhs_4d, -1); // for the forwards direction
        for (int s = 0; s < Ls; s++) {
          // forwards direction
          blas::axy(alpha[s] * ((s == Ls - 1) ? -mass_pv : 1), chiral_plus[(s + 1) % Ls], tmp_4d[s]);
          // backwards direction
          blas::axpy(alpha[s] * ((s == 0) ? -mass_pv : 1), chiral_minus[(s + Ls - 1) % Ls], tmp_4d[s]);
          // diagonal contribution
          blas::axpy(beta[s], emul_lhs_4d[s], tmp_4d[s]);
        }

        // This bit is equivalent to the last Mobius call:
        // ApplyDslash5(out, in, in, mass_pv, m5, b_5, c_5, 0.0, QUDA_DAG_YES, Dslash5Type::DSLASH5_MOBIUS);

        // from compute_coeff_mobius
        for (int s = 0; s < Ls; s++) {
          kappa[s]
            = 0.5 * (c_5[s].real() * (m5 + 4.0) - 1.0) / (b_5[s].real() * (m5 + 4.0) + 1.0); // 0.5 from gamma matrices
        }

        ApplyChiralProj(chiral_plus, rhs_4d, +1);  // for the backwards direction
        ApplyChiralProj(chiral_minus, rhs_4d, -1); // for the forwards direction
        for (int s = 0; s < Ls; s++) {
          // forwards direction
          blas::axy(kappa[s] * ((s == Ls - 1) ? -mass_pv : 1), chiral_plus[(s + 1) % Ls], emul_lhs_4d[s]);
          // backwards direction
          blas::axpy(kappa[s] * ((s == 0) ? -mass_pv : 1), chiral_minus[(s + Ls - 1) % Ls], emul_lhs_4d[s]);
          // diagonal contribution
          blas::axpy(1.0, rhs_4d[s], emul_lhs_4d[s]);
        }

        // last, but not least, this call
        // blas::axpy(-mobius_kappa_b, tmp, out);

        blas::axpy(-mobius_kappa_b, tmp_4d, emul_lhs_4d);
      }

      if (getVerbosity() >= QUDA_VERBOSE) {
        // for debugging purposes
        // std::vector<ColorSpinorField> dwf_lhs_4(Ls);
        // for (auto &f : dwf_lhs_4) f = ColorSpinorField(csParam);
        // auto dwf_lhs_4d = vector_ref<ColorSpinorField>(dwf_lhs_4);
        auto dwf_lhs_4d_ = getFieldTmp<ColorSpinorField>(Ls, csParam);
        vector_ref<ColorSpinorField> dwf_lhs_4d(dwf_lhs_4d_);

        Split5DTo4DFields(dwf_lhs_4d, dwf_lhs);

        // in theory emul_lhs contains the same thing as dwf_lhs on a per-component basis
        for (int s = 0; s < Ls; s++)
          printfQuda("s %d norm2 source %e norm2 chiral+ %e norm2 chiral- %e norm2 dwf %e norm2 emul %e\n", s,
                    blas::norm2(rhs_4d[s]), blas::norm2(chiral_plus[s]), blas::norm2(chiral_minus[s]),
                    blas::norm2(dwf_lhs_4d[s]), blas::norm2(emul_lhs_4d[s]));
      }

      // re-join
      Join4DTo5DField(emul_lhs, emul_lhs_4d);

      // verify
      double emul_nrm2 = blas::norm2(emul_lhs);
      double native_nrm2 = blas::norm2(dwf_lhs);
      auto max_deviation = blas::max_deviation(dwf_lhs, emul_lhs);
      auto l2_deviation = sqrt(xmyNorm(dwf_lhs, emul_lhs) / native_nrm2);

      logQuda(QUDA_VERBOSE, "L2 norms: Emulated = %e, Native = %e; Deviations: L2 relative = %e, max = %e\n", emul_nrm2,
              native_nrm2, l2_deviation, max_deviation[0]);

      // may want to revisit this---these were relaxed for cases where ghost_precision < precision
      // these were set while hacking in tests of quarter precision ghosts
      // moreover, we can improve the precision of block ortho with a tighter max than 1.0
      QudaPrecision prec
        = std::min(param.mg_global.precision_null[param.level], param.mg_global.invert_param->cuda_prec_sloppy);

      double tol;
      switch (prec) {
      case QUDA_QUARTER_PRECISION: tol = 5e-2; break;
      case QUDA_HALF_PRECISION: tol = 5e-2; break;
      case QUDA_SINGLE_PRECISION: tol = 2e-3; break;
      default: tol = 1e-8;
      }

      if (check_deviation(l2_deviation, tol))
        errorQuda("Coarse operator failed: L2 relative deviation = %e > %e", l2_deviation, tol);
      if (check_deviation(max_deviation[0], tol))
        warningQuda("Coarse operator failed: max deviation = %e > %e", max_deviation[0], tol);
    } // end of check 5-d Pauli-Villars dagger operator
  }

  void MG::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    pushOutputPrefix(prefix);

    QudaMatPCType matpc_type = param.mg_global.invert_param->matpc_type;
    QudaParity parity = (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) ?
      QUDA_EVEN_PARITY :
      QUDA_ODD_PARITY;

    if (param.level < param.Nlevel - 1) { // set parity for the solver in the transfer operator
      QudaSiteSubset site_subset
        = param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION ? QUDA_PARITY_SITE_SUBSET : QUDA_FULL_SITE_SUBSET;
      transfer->setSiteSubset(site_subset, parity); // use this to force location of transfer
    }

    QudaSolutionType outer_solution_type = b.SiteSubset() == QUDA_FULL_SITE_SUBSET ? QUDA_MAT_SOLUTION : QUDA_MATPC_SOLUTION;
    QudaSolutionType inner_solution_type = param.coarse_grid_solution_type;
    // is the smoother consistent with the coarse grid correction
    bool smoother_solver_uniform
      = (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE && inner_solution_type == QUDA_MATPC_SOLUTION)
      || (param.smoother_solve_type == QUDA_DIRECT_SOLVE && inner_solution_type == QUDA_MAT_SOLUTION);

    // if using preconditioned smoother then need to reconstruct full residual
    // FIXME extend this check for precision, Schwarz, etc.
    bool use_solver_residual = presmoother && smoother_solver_uniform;

    // need to compute residual vector if presmoothing and smoother not consistent with coarse grid correction
    bool compute_residual = presmoother && !smoother_solver_uniform;

    ColorSpinorParam csParam(b[0]);
    auto r = getFieldTmp<ColorSpinorField>(presmoother ? b.size() : 0, csParam);
    resize(r_coarse, b.size(), QUDA_NULL_FIELD_CREATE);
    resize(x_coarse, b.size(), QUDA_NULL_FIELD_CREATE);

    if (outer_solution_type == QUDA_MATPC_SOLUTION && inner_solution_type == QUDA_MAT_SOLUTION)
      errorQuda("Unsupported solution type combination");

    if (inner_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type != QUDA_DIRECT_PC_SOLVE)
      errorQuda("For this coarse grid solution type, a preconditioned smoother is required");

    if (param.level < param.Nlevel - 1) {
      std::vector<ColorSpinorField> out(b.size()), in(b.size());
      diracSmoother->prepare(out, in, x, b, outer_solution_type);

      if (presmoother) (*presmoother)(out, in);

      if (!smoother_solver_uniform) diracSmoother->reconstruct(x, b, inner_solution_type);

      if (compute_residual) {
        (*param.matResidual)(r, x);
        axpby(1.0, b, -1.0, r);
      }

      // We need this to ensure that the coarse level has been created.
      // e.g. in case of iterative setup with MG we use just pre- and post-smoothing at the first iteration.
      if (transfer) {
        const auto &residual = use_solver_residual ? presmoother->get_residual() :
          !presmoother && smoother_solver_uniform  ? in :
          !presmoother                             ? b :
                                                     r;

        // restrict to the coarse grid
        transfer->R(r_coarse, residual);

        // recurse to the next lower level
        (*coarse_solver)(x_coarse, r_coarse);

        // prolongate back to this grid
        if (!presmoother) {
          transfer->P(inner_solution_type == outer_solution_type ? x : x(parity), x_coarse);
        } else { // we must sum to the presmoother solution
          auto res = inner_solution_type == outer_solution_type ? cvector_ref<ColorSpinorField>(r) :
                                                                  cvector_ref<ColorSpinorField>(r)(parity);
          transfer->P(res, x_coarse);
          xpy(res, inner_solution_type == outer_solution_type ? x : x(parity));
        }
      }

      if (!smoother_solver_uniform) diracSmoother->prepare(out, in, x, b, inner_solution_type);

      if (postsmoother) (*postsmoother)(out, in);

      diracSmoother->reconstruct(x, b, outer_solution_type);

    } else { // do the coarse grid solve

      std::vector<ColorSpinorField> out(b.size()), in(b.size());
      diracSmoother->prepare(out, in, x, b, outer_solution_type);
      if (presmoother) (*presmoother)(out, in);
      diracSmoother->reconstruct(x, b, outer_solution_type);
    }

    popOutputPrefix();
  }

  // supports separate reading or single file read
  void MG::loadVectors(cvector_ref<ColorSpinorField> &B)
  {
    if (param.transfer_type != QUDA_TRANSFER_AGGREGATE) {
      warningQuda("Cannot load near-null vectors for top level of staggered MG solve.");
    } else {
      getProfile().TPSTART(QUDA_PROFILE_IO);
      pushLevel(param.level);
      std::string vec_infile(param.mg_global.vec_infile[param.level]);
      vec_infile += "_level_";
      vec_infile += std::to_string(param.level);
      vec_infile += "_nvec_";
      vec_infile += std::to_string(param.mg_global.n_vec[param.level]);
      VectorIO io(vec_infile);
      io.load(B);
      popLevel();
      getProfile().TPSTOP(QUDA_PROFILE_IO);
    }
  }

  void MG::saveVectors(cvector_ref<const ColorSpinorField> &B) const
  {
    if (param.transfer_type != QUDA_TRANSFER_AGGREGATE) {
      warningQuda("Cannot save near-null vectors for top level of staggered MG solve.");
    } else {
      getProfile().TPSTART(QUDA_PROFILE_IO);
      pushLevel(param.level);
      std::string vec_outfile(param.mg_global.vec_outfile[param.level]);
      vec_outfile += "_level_";
      vec_outfile += std::to_string(param.level);
      vec_outfile += "_nvec_";
      vec_outfile += std::to_string(param.mg_global.n_vec[param.level]);
      VectorIO io(vec_outfile, false, param.mg_global.mg_vec_partfile[param.level]);
      io.save(B);
      popLevel();
      getProfile().TPSTOP(QUDA_PROFILE_IO);
    }
  }

  void MG::dumpNullVectors() const
  {
    if (param.transfer_type != QUDA_TRANSFER_AGGREGATE) {
      warningQuda("Cannot dump near-null vectors for top level of staggered MG solve.");
    } else {
      saveVectors(param.B);
    }
    if (param.level < param.Nlevel - 2) coarse->dumpNullVectors();
  }

  void MG::generateNullVectors(std::vector<ColorSpinorField> &B, bool refresh)
  {
    pushLevel(param.level);

    SolverParam solverParam(param); // Set solver field parameters:
    // set null-space generation options - need to expose these
    solverParam.maxiter
      = refresh ? param.mg_global.setup_maxiter_refresh[param.level] : param.mg_global.setup_maxiter[param.level];
    solverParam.tol = param.mg_global.setup_tol[param.level];
    solverParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;
    solverParam.delta = 1e-1;
    solverParam.inv_type = param.mg_global.setup_inv_type[param.level];
    // Hard coded for now...
    if (is_ca_solver(solverParam.inv_type)) {
      solverParam.ca_basis = param.mg_global.setup_ca_basis[param.level];
      solverParam.ca_lambda_min = param.mg_global.setup_ca_lambda_min[param.level];
      solverParam.ca_lambda_max = param.mg_global.setup_ca_lambda_max[param.level];
      solverParam.Nkrylov = param.mg_global.setup_ca_basis_size[param.level];
    } else if (solverParam.inv_type == QUDA_GCR_INVERTER || solverParam.inv_type == QUDA_BICGSTABL_INVERTER) {
      solverParam.Nkrylov = param.mg_global.setup_ca_basis_size[param.level];
    } else {
      solverParam.Nkrylov = 4;
    }
    solverParam.pipeline
      = (solverParam.inv_type == QUDA_BICGSTAB_INVERTER ? 0 : 4); // FIXME: pipeline != 0 breaks BICGSTAB
    solverParam.precision = param.mg_global.invert_param->cuda_prec_sloppy;

    if (is_fine_grid()) {
      solverParam.precision_sloppy = param.mg_global.invert_param->cuda_prec_precondition;
      solverParam.precision_precondition = param.mg_global.invert_param->cuda_prec_precondition;
    } else {
      solverParam.precision_precondition = solverParam.precision;
    }

    solverParam.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
    solverParam.compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;
    ColorSpinorParam csParam(B[0]);                                           // Create spinor field parameters:
    csParam.setPrecision(solverParam.precision, solverParam.precision, true); // ensure native ordering
    csParam.location = QUDA_CUDA_FIELD_LOCATION; // hard code to GPU location for null-space generation for now
    csParam.gammaBasis = B[0].Nspin() == 1 ? QUDA_DEGRAND_ROSSI_GAMMA_BASIS :
                                             QUDA_UKQCD_GAMMA_BASIS; // degrand-rossi required for staggered
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    std::vector<ColorSpinorField> b, x;
    resize(b, param.n_vec_batch, csParam);
    resize(x, param.n_vec_batch, csParam);

    // if we're not using GCR/MG smoother then we need to switch off Schwarz since regular Krylov solvers do not support it
    bool schwarz_reset = solverParam.inv_type != QUDA_MG_INVERTER
      && param.mg_global.smoother_schwarz_type[param.level] != QUDA_INVALID_SCHWARZ;
    if (schwarz_reset) {
      logQuda(QUDA_VERBOSE, "Disabling Schwarz for null-space finding");
      int commDim[QUDA_MAX_DIM];
      for (int i = 0; i < QUDA_MAX_DIM; i++) commDim[i] = 1;
      diracNullSloppy->setCommDim(commDim);
    }

    // if quarter precision halo, promote for null-space finding to half precision
    QudaPrecision halo_precision = diracSmootherSloppy->HaloPrecision();
    if (halo_precision == QUDA_QUARTER_PRECISION) diracSmootherSloppy->setHaloPrecision(QUDA_HALF_PRECISION);

    Solver *solve;
    DiracMdagM *mdagm = (solverParam.inv_type == QUDA_CG_INVERTER || solverParam.inv_type == QUDA_CA_CG_INVERTER) ?
      new DiracMdagM(*diracNull) :
      nullptr;
    DiracMdagM *mdagmSloppy = (solverParam.inv_type == QUDA_CG_INVERTER || solverParam.inv_type == QUDA_CA_CG_INVERTER) ?
      new DiracMdagM(*diracNullSloppy) :
      nullptr;
    if (solverParam.inv_type == QUDA_CG_INVERTER || solverParam.inv_type == QUDA_CA_CG_INVERTER) {
      solve = Solver::create(solverParam, *mdagm, *mdagmSloppy, *mdagmSloppy, *mdagmSloppy);
    } else if (solverParam.inv_type == QUDA_MG_INVERTER) {
      // in case MG has not been created, we create the Smoother
      if (!transfer) createSmoother();

      // run GCR with the MG as a preconditioner
      solverParam.inv_type_precondition = QUDA_MG_INVERTER;
      solverParam.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
      solverParam.precondition_cycle = 1;
      solverParam.tol_precondition = 1e-1;
      solverParam.maxiter_precondition = 1;
      solverParam.omega = 1.0;
      solverParam.verbosity_precondition = param.mg_global.verbosity[param.level+1];
      solverParam.precision_sloppy = solverParam.precision;
      solverParam.compute_true_res = 0;
      solverParam.preconditioner = this;

      solverParam.inv_type = QUDA_GCR_INVERTER;
      solve = Solver::create(solverParam, *param.matNull, *param.matNull, *param.matNullSloppy, *param.matNullSloppy);
      solverParam.inv_type = QUDA_MG_INVERTER;
    } else {
      solve
        = Solver::create(solverParam, *param.matNull, *param.matNullSloppy, *param.matNullSloppy, *param.matNullSloppy);
    }

    for (int si = 0; si < param.mg_global.num_setup_iter[param.level]; si++) {
      logQuda(QUDA_VERBOSE, "Running vectors setup on level %d iter %d of %d\n", param.level, si + 1,
              param.mg_global.num_setup_iter[param.level]);

      // global orthonormalization of the initial null-space vectors
      if (param.mg_global.pre_orthonormalize) {
        for (auto i = 0u; i < B.size(); i++) {
          for (auto j = 0u; j < i; j++) {
            Complex alpha = cDotProduct(B[j], B[i]); // <j,i>
            caxpy(-alpha, B[j], B[i]);               // i-<j,i>j
          }
          double nrm2 = norm2(B[i]);
          if (nrm2 > 1e-16)
            ax(1.0 / sqrt(nrm2), B[i]); // i/<i,i>
          else errorQuda("\nCannot normalize %u vector\n", i);
        }
      }

      // launch solver for each source
      if (B.size() % param.n_vec_batch != 0) errorQuda("Bad batch size %d", param.n_vec_batch);
      for (auto i = 0u; i < B.size(); i += param.n_vec_batch) {
        if (param.mg_global.setup_type
            == QUDA_TEST_VECTOR_SETUP) { // DDalphaAMG test vector idea solving against the vector
          copy({b.begin(), b.begin() + param.n_vec_batch}, {B.begin() + i, B.begin() + i + param.n_vec_batch});
          zero(x); // with zero initial guess
        } else {
          copy({x.begin(), x.begin() + param.n_vec_batch}, {B.begin() + i, B.begin() + i + param.n_vec_batch});
          zero(b);
        }

        if (getVerbosity() >= QUDA_VERBOSE) {
          auto nrm2 = norm2(x);
          auto b2 = norm2(b);
          for (auto j = 0; j < param.n_vec_batch; j++)
            printfQuda("%d Initial guess = %g, Initial rhs = %g\n", i + j, nrm2[j], b2[j]);
        }

        std::vector<ColorSpinorField> out(param.n_vec_batch), in(param.n_vec_batch);
        diracNull->prepare(out, in, x, b, QUDA_MAT_SOLUTION);
        (*solve)(out, in);
        diracNull->reconstruct(x, b, QUDA_MAT_SOLUTION);

        if (getVerbosity() >= QUDA_VERBOSE) {
          auto nrm2 = norm2(x);
          for (auto j = 0; j < param.n_vec_batch; j++) printfQuda("%d Solution = %g\n", i + j, nrm2[j]);
        }

        copy({B.begin() + i, B.begin() + i + param.n_vec_batch}, {x.begin(), x.begin() + param.n_vec_batch});
      }

      // global orthonormalization of the generated null-space vectors
      if (param.mg_global.post_orthonormalize) {
        for (auto i = 0u; i < B.size(); i++) {
          for (auto j = 0u; j < i; j++) {
            Complex alpha = cDotProduct(B[j], B[i]); // <j,i>
            caxpy(-alpha, B[j], B[i]);               // i-<j,i>j
          }
          double nrm2 = norm2(B[i]);
          if (sqrt(nrm2) > 1e-16)
            ax(1.0 / sqrt(nrm2), B[i]); // i/<i,i>
          else errorQuda("\nCannot normalize %u vector (nrm=%e)\n", i, sqrt(nrm2));
        }
      }

      if (solverParam.inv_type == QUDA_MG_INVERTER) {

        if (transfer) {
          resetTransfer = true;
          reset();
          if ( param.level < param.Nlevel-2 ) {
            if ( param.mg_global.generate_all_levels == QUDA_BOOLEAN_TRUE ) {
              coarse->generateNullVectors(B_coarse, refresh);
            } else {
              logQuda(QUDA_VERBOSE, "Restricting null space vectors\n");
              for (auto i = 0; i < param.Nvec; i++) {
                zero(B_coarse[i]);
                transfer->R(B_coarse[i], param.B[i]);
              }
              // rebuild the transfer operator in the coarse level
              coarse->resetTransfer = true;
              coarse->reset();
            }
          }
        } else {
          reset();
        }
      }
    }

    delete solve;
    if (mdagm) delete mdagm;
    if (mdagmSloppy) delete mdagmSloppy;

    diracNullSloppy->setHaloPrecision(halo_precision); // restore halo precision

    // reenable Schwarz
    if (schwarz_reset) {
      logQuda(QUDA_VERBOSE, "Reenabling Schwarz for null-space finding\n");
      int commDim[QUDA_MAX_DIM];
      for (int i = 0; i < QUDA_MAX_DIM; i++) commDim[i] = 0;
      diracNullSloppy->setCommDim(commDim);
    }

    if (param.mg_global.vec_store[param.level] == QUDA_BOOLEAN_TRUE) { // conditional store of null vectors
      saveVectors(B);
    }

    popLevel();
  }

  // generate a full span of free vectors.
  // FIXME: Assumes fine level is SU(3).
  void MG::buildFreeVectors(std::vector<ColorSpinorField> &B)
  {
    pushLevel(param.level);
    const int Nvec = B.size();

    // Given the number of colors and spins, figure out if the number
    // of vectors in 'B' makes sense.
    const int Ncolor = B[0].Ncolor();
    const int Nspin = B[0].Nspin();

    if (Ncolor == 3) // fine level
    {
      if (Nspin == 4) // Wilson or Twisted Mass (singlet)
      {
        // There needs to be 6 null vectors -> 12 after chirality.
        if (Nvec != 6) errorQuda("\nError in MG::buildFreeVectors: Wilson-type fermions require Nvec = 6");

        logQuda(QUDA_VERBOSE, "Building %d free field vectors for Wilson-type fermions\n", Nvec);

        // Zero the null vectors.
        for (int i = 0; i < Nvec; i++) zero(B[i]);

        // Create a temporary vector.
        ColorSpinorParam csParam(B[0]);
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField tmp(csParam);

        int counter = 0;
        for (int c = 0; c < Ncolor; c++) {
          for (int s = 0; s < 2; s++) {
            tmp.Source(QUDA_CONSTANT_SOURCE, 1, s, c);
            xpy(tmp, B[counter]);
            tmp.Source(QUDA_CONSTANT_SOURCE, 1, s + 2, c);
            xpy(tmp, B[counter]);
            counter++;
          }
        }

      } else if (Nspin == 1) { // Staggered

        // There needs to be 24 null vectors -> 48 after chirality.
        if (Nvec != 24) errorQuda("\nError in MG::buildFreeVectors: Staggered-type fermions require Nvec = 24\n");

        logQuda(QUDA_VERBOSE, "Building %d free field vectors for Staggered-type fermions\n", Nvec);

        // Zero the null vectors.
        for (int i = 0; i < Nvec; i++) zero(B[i]);

        // Create a temporary vector.
        ColorSpinorParam csParam(B[0]);
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField tmp(csParam);

        // Build free null vectors.
        for (int c = 0; c < B[0].Ncolor(); c++) {
          // Need to pair an even+odd corner together
          // since they'll get split up.
          // 0000, 0001
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0x0, c);
          xpy(tmp, B[8 * c + 0]);
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0x1, c);
          xpy(tmp, B[8 * c + 0]);

          // 0010, 0011
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0x2, c);
          xpy(tmp, B[8 * c + 1]);
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0x3, c);
          xpy(tmp, B[8 * c + 1]);

          // 0100, 0101
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0x4, c);
          xpy(tmp, B[8 * c + 2]);
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0x5, c);
          xpy(tmp, B[8 * c + 2]);

          // 0110, 0111
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0x6, c);
          xpy(tmp, B[8 * c + 3]);
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0x7, c);
          xpy(tmp, B[8 * c + 3]);

          // 1000, 1001
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0x8, c);
          xpy(tmp, B[8 * c + 4]);
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0x9, c);
          xpy(tmp, B[8 * c + 4]);

          // 1010, 1011
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0xA, c);
          xpy(tmp, B[8 * c + 5]);
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0xB, c);
          xpy(tmp, B[8 * c + 5]);

          // 1100, 1101
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0xC, c);
          xpy(tmp, B[8 * c + 6]);
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0xD, c);
          xpy(tmp, B[8 * c + 6]);

          // 1110, 1111
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0xE, c);
          xpy(tmp, B[8 * c + 7]);
          tmp.Source(QUDA_CORNER_SOURCE, 1, 0xF, c);
          xpy(tmp, B[8 * c + 7]);
        }

      } else {
        errorQuda("\nError in MG::buildFreeVectors: Unsupported combo of Nc %d, Nspin %d", Ncolor, Nspin);
      }
    } else { // coarse level
      if (Nspin == 2) {
        // There needs to be Ncolor null vectors.
        if (Nvec != Ncolor) errorQuda("\nError in MG::buildFreeVectors: Coarse fermions require Nvec = Ncolor");

        logQuda(QUDA_VERBOSE, "Building %d free field vectors for Coarse fermions\n", Ncolor);

        // Zero the null vectors.
        for (int i = 0; i < Nvec; i++) zero(B[i]);

        // Create a temporary vector.
        ColorSpinorParam csParam(B[0]);
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField tmp(csParam);

        for (int c = 0; c < Ncolor; c++) {
          tmp.Source(QUDA_CONSTANT_SOURCE, 1, 0, c);
          xpy(tmp, B[c]);
          tmp.Source(QUDA_CONSTANT_SOURCE, 1, 1, c);
          xpy(tmp, B[c]);
        }

      } else if (Nspin == 1) {
        // There needs to be Ncolor null vectors.
        if (Nvec != Ncolor) errorQuda("\nError in MG::buildFreeVectors: Coarse fermions require Nvec = Ncolor");

        logQuda(QUDA_VERBOSE, "Building %d free field vectors for Coarse fermions\n", Ncolor);

        // Zero the null vectors.
        for (int i = 0; i < Nvec; i++) zero(B[i]);

        // Create a temporary vector.
        ColorSpinorParam csParam(B[0]);
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField tmp(csParam);

        for (int c = 0; c < Ncolor; c++) {
          tmp.Source(QUDA_CONSTANT_SOURCE, 1, 0, c);
          xpy(tmp, B[c]);
        }

      } else {
        errorQuda("\nError in MG::buildFreeVectors: Unexpected Nspin = %d for coarse fermions", Nspin);
      }
    }

    // global orthonormalization of the generated null-space vectors
    if(param.mg_global.post_orthonormalize) {
      for (auto i = 0u; i < B.size(); i++) {
        double nrm2 = norm2(B[i]);
        if (nrm2 > 1e-16)
          ax(1.0 / sqrt(nrm2), B[i]); // i/<i,i>
        else errorQuda("\nCannot normalize %u vector\n", i);
      }
    }

    logQuda(QUDA_VERBOSE, "Done building free vectors\n");
    popLevel();
  }

  void MG::generateEigenVectors(std::vector<ColorSpinorField> &B)
  {
    pushLevel(param.level);

    // Extract eigensolver params
    int n_conv = param.mg_global.eig_param[param.level]->n_conv;
    bool dagger = param.mg_global.eig_param[param.level]->use_dagger;
    bool normop = param.mg_global.eig_param[param.level]->use_norm_op;

    // Dummy array to keep the eigensolver happy.
    ColorSpinorParam csParam(param.B[0]);
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    // This is the vector precision used by matResidual
    csParam.setPrecision(param.mg_global.invert_param->cuda_prec_sloppy, QUDA_INVALID_PRECISION, true);

    std::vector<Complex> evals(n_conv, 0.0);
    std::vector<ColorSpinorField> B_evecs(n_conv);
    for (auto &b : B_evecs) b = ColorSpinorField(csParam);

    // before entering the eigen solver, let's free the B vectors to save some memory
    ColorSpinorParam bParam(B[0]);
    for (auto &b : B) b = ColorSpinorField();

    EigenSolver *eig_solve;
    if (!normop && !dagger) {
      DiracM *mat = new DiracM(*diracNull);
      eig_solve = EigenSolver::create(param.mg_global.eig_param[param.level], *mat);
      (*eig_solve)(B_evecs, evals);
      delete eig_solve;
      delete mat;
    } else if (!normop && dagger) {
      DiracMdag *mat = new DiracMdag(*diracNull);
      eig_solve = EigenSolver::create(param.mg_global.eig_param[param.level], *mat);
      (*eig_solve)(B_evecs, evals);
      delete eig_solve;
      delete mat;
    } else if (normop && !dagger) {
      DiracMdagM *mat = new DiracMdagM(*diracNull);
      eig_solve = EigenSolver::create(param.mg_global.eig_param[param.level], *mat);
      (*eig_solve)(B_evecs, evals);
      delete eig_solve;
      delete mat;
    } else if (normop && dagger) {
      DiracMMdag *mat = new DiracMMdag(*diracNull);
      eig_solve = EigenSolver::create(param.mg_global.eig_param[param.level], *mat);
      (*eig_solve)(B_evecs, evals);
      delete eig_solve;
      delete mat;
    }

    // now reallocate the B vectors copy in e-vectors
    bParam.create = QUDA_NULL_FIELD_CREATE;
    for (auto i = 0u; i < param.B.size(); i++) {
      B[i] = ColorSpinorField(bParam);
      B[i] = B_evecs[i];
    }

    // only save if outfile is defined
    if (strcmp(param.mg_global.vec_outfile[param.level], "") != 0) { saveVectors(B); }

    popLevel();
  }

} // namespace quda
