#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include <util_quda.h>
#include <test_util.h>
#include <test_params.h>

#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

namespace quda {
  extern void setTransferGPU(bool);
}

void
display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    prec_sloppy   matpc_type  recon  recon_sloppy S_dimension T_dimension Ls_dimension   dslash_type "
             " normalization\n");
  printfQuda("%6s   %6s    %12s     %2s     %2s         %3d/%3d/%3d     %3d         %2d     %14s  %8s\n",
             get_prec_str(prec), get_prec_str(prec_sloppy), get_matpc_str(matpc_type), get_recon_str(link_recon),
             get_recon_str(link_recon_sloppy), xdim, ydim, zdim, tdim, Lsdim, get_dslash_str(dslash_type),
             get_mass_normalization_str(normalization));

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));

  printfQuda("Deflation space info: location   mem_type\n");
  printfQuda("                     %5s     %8s\n", get_ritz_location_str(location_ritz),
             get_memory_type_str(mem_type_ritz));

  return ;
}

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;
QudaPrecision &cuda_prec_refinement_sloppy = prec_refinement_sloppy;
QudaPrecision &cuda_prec_ritz = prec_ritz;

void setGaugeParam(QudaGaugeParam &gauge_param) {
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;

  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_precondition;

  gauge_param.cuda_prec_refinement_sloppy = cuda_prec_refinement_sloppy;
  gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =std::max(x_face_size, y_face_size);
  pad_size = std::max(pad_size, z_face_size);
  pad_size = std::max(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;
#endif
}


void setInvertParam(QudaInvertParam &inv_param) {
  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_refinement_sloppy = cuda_prec_refinement_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

//  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  inv_param.dslash_type = dslash_type;

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1 + 3/anisotropy);
  }

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.clover_coeff = clover_coeff;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;

  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  // inv_param.solution_type = QUDA_MATPC_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;

  if (inv_type != QUDA_EIGCG_INVERTER && inv_type != QUDA_INC_EIGCG_INVERTER && inv_type != QUDA_GMRESDR_INVERTER)
    errorQuda("Unknown deflated solver type %d.", inv_type);

  //! For deflated solvers only:
  inv_param.inv_type = inv_type;
  inv_param.tol      = tol;
  inv_param.tol_hq   = tol_hq; // specify a tolerance for the residual for heavy quark residual

  inv_param.rhs_idx  = 0;

  inv_param.nev = nev;
  inv_param.max_search_dim = max_search_dim;
  inv_param.deflation_grid = deflation_grid;
  inv_param.tol_restart = tol_restart;
  inv_param.eigcg_max_restarts = eigcg_max_restarts;
  inv_param.max_restart_num = max_restart_num;
  inv_param.inc_tol = inc_tol;
  inv_param.eigenval_tol = eigenval_tol;


  if(inv_param.inv_type == QUDA_EIGCG_INVERTER || inv_param.inv_type == QUDA_INC_EIGCG_INVERTER ){
    inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
  }else if(inv_param.inv_type == QUDA_GMRESDR_INVERTER) {
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    inv_param.tol_restart = 0.0;//restart is not requested...
  }

  inv_param.cuda_prec_ritz = cuda_prec_ritz;
  inv_param.verbosity = verbosity;
  inv_param.verbosity_precondition = verbosity;

  inv_param.inv_type_precondition = precon_type;
  inv_param.gcrNkrylov = 6;

  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = 1e-1;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-2;
  inv_param.maxiter_precondition = 10;
  inv_param.omega = 1.0;

  inv_param.extlib_type = solver_ext_lib;
}

void setDeflationParam(QudaEigParam &df_param) {

  df_param.import_vectors = QUDA_BOOLEAN_NO;
  df_param.run_verify     = QUDA_BOOLEAN_NO;

  df_param.nk             = df_param.invert_param->nev;
  df_param.np             = df_param.invert_param->nev*df_param.invert_param->deflation_grid;
  df_param.extlib_type    = deflation_ext_lib;

  df_param.cuda_prec_ritz = prec_ritz;
  df_param.location       = location_ritz;
  df_param.mem_type_ritz  = mem_type_ritz;

  // set file i/o parameters
  strcpy(df_param.vec_infile, eig_vec_infile);
  strcpy(df_param.vec_outfile, eig_vec_outfile);
}

int main(int argc, char **argv)
{

  // command line options
  auto app = make_app();
  // add_eigen_option_group(app);
  add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_ritz == QUDA_INVALID_PRECISION) prec_ritz = prec_sloppy;
  if (prec_refinement_sloppy == QUDA_INVALID_PRECISION) prec_refinement_sloppy = prec_sloppy;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** QUDA parameters begin here.

  if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH
      && dslash_type != QUDA_TWISTED_MASS_DSLASH && dslash_type != QUDA_DOMAIN_WALL_4D_DSLASH
      && dslash_type != QUDA_MOBIUS_DWF_DSLASH && dslash_type != QUDA_TWISTED_CLOVER_DSLASH
      && dslash_type != QUDA_DOMAIN_WALL_DSLASH && dslash_type != QUDA_MOBIUS_DWF_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);


  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  double kappa5 = 0.0;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.epsilon = epsilon;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
             || dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    inv_param.m5 = -1.8;
    kappa5 = 0.5 / (5 + inv_param.m5);
    inv_param.Ls = Lsdim;
    for (int k = 0; k < Lsdim; k++) // for mobius only
    {
      // b5[k], c[k] values are chosen for arbitrary values,
      // but the difference of them are same as 1.0
      inv_param.b_5[k] = 1.452;
      inv_param.c_5[k] = 0.452;
    }
  }

  QudaEigParam  df_param = newQudaEigParam();
  df_param.invert_param = &inv_param;
  setDeflationParam(df_param);

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    dw_setDims(gauge_param.X, inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }

  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover=0, *clover_inv=0;

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate an SU(3) field
    if (unit_gauge) {
      // unit SU(3) field
      construct_gauge_field(gauge, 0, gauge_param.cpu_prec, &gauge_param);
    } else {
      // random SU(3) field
      construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
    }
  }

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    double norm = 0.1; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = inv_param.clover_cpu_prec;
    clover = malloc(V*cloverSiteSize*cSize);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    if (!compute_clover) construct_clover_field(clover, norm, diag, inv_param.clover_cpu_prec);

    inv_param.compute_clover = compute_clover;
    if (compute_clover) inv_param.return_clover = 1;
    inv_param.compute_clover_inverse = 1;
    inv_param.return_clover_inverse = 1;
  }

  void *spinorIn = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

  void *spinorOut = NULL;
  spinorOut = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  // this line ensure that if we need to construct the clover inverse (in either the smoother or the solver) we do so
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) loadCloverQuda(clover, clover_inv, &inv_param);

  void *df_preconditioner  = newDeflationQuda(&df_param);
  inv_param.deflation_op   = df_preconditioner;

  for (int i=0; i<Nsrc; i++) {
    // create a point source at 0 (in each subvolume...  FIXME)
    memset(spinorIn, 0, inv_param.Ls*V*spinorSiteSize*sSize);
    memset(spinorCheck, 0, inv_param.Ls*V*spinorSiteSize*sSize);
    memset(spinorOut, 0, inv_param.Ls*V*spinorSiteSize*sSize);

    if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
      //((float*)spinorIn)[i] = 1.0;
      for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((float*)spinorIn)[i] = rand() / (float)RAND_MAX;
    } else {
      //((double*)spinorIn)[i] = 1.0;
      for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((double*)spinorIn)[i] = rand() / (double)RAND_MAX;
    }

    invertQuda(spinorOut, spinorIn, &inv_param);
    printfQuda("\nDone for %d rhs.\n", inv_param.rhs_idx);
  }

  destroyDeflationQuda(df_preconditioner);

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  // printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n",
  // inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);
  printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", inv_param.iter, inv_param.secs,
             inv_param.gflops / inv_param.secs, 0.0);

  if (inv_param.solution_type == QUDA_MAT_SOLUTION) {

    if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      wil_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);

    } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
      dw_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
    } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
      dw_4d_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
    } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
      double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
      for (int xs = 0; xs < Lsdim; xs++) {
        kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
        kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
      }
      mdw_mat(spinorCheck, gauge, spinorOut, kappa_b, kappa_c, inv_param.dagger, inv_param.cpu_prec, gauge_param,
              inv_param.mass, inv_param.b_5, inv_param.c_5);
      free(kappa_b);
      free(kappa_c);
    } else {
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if(inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tm_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0, inv_param.cpu_prec, gauge_param);
        } else {
          printfQuda("Unsupported dslash_type\n");
          exit(-1);
        }
      }
    }

    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
      if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
          || dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
        ax(0.5 / kappa5, spinorCheck, V * spinorSiteSize * inv_param.Ls, inv_param.cpu_prec);
      } else if (dslash_type == QUDA_TWISTED_MASS_DSLASH && twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
        ax(0.5 / inv_param.kappa, spinorCheck, 2 * V * spinorSiteSize, inv_param.cpu_prec);
      } else {
        ax(0.5 / inv_param.kappa, spinorCheck, V * spinorSiteSize, inv_param.cpu_prec);
      }
    }

  } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {

    if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      wil_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
    } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
      dw_matpc(spinorCheck, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param,
               inv_param.mass);
    } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
      dw_4d_matpc(spinorCheck, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param,
                  inv_param.mass);
    } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
      double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
      for (int xs = 0; xs < Lsdim; xs++) {
        kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
        kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
      }
      mdw_matpc(spinorCheck, gauge, spinorOut, kappa_b, kappa_c, inv_param.matpc_type, 0, inv_param.cpu_prec,
                gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
      free(kappa_b);
      free(kappa_c);
    } else {
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tm_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        } else {
          printfQuda("Unsupported dslash_type\n");
          exit(-1);
        }
      }
    }

    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
      if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
          || dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
        ax(0.25 / (kappa5 * kappa5), spinorCheck, V * spinorSiteSize * inv_param.Ls, inv_param.cpu_prec);
      } else {
        ax(0.25 / (inv_param.kappa * inv_param.kappa), spinorCheck, Vh * spinorSiteSize, inv_param.cpu_prec);
      }
    }
  }

  int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
  mxpy(spinorIn, spinorCheck, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
  double l2r = sqrt(nrm2 / src2);

  printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
	     inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq);


  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }

  for (int dir = 0; dir<4; dir++) free(gauge[dir]);

  return 0;
}
