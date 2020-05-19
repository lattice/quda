#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>

#include <dslash_reference.h>
//#include <blas_reference.h>
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

void display_test_info()
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

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // initialize the QUDA library
  initQuda(device);

  // *** QUDA parameters begin here.

  if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH
      && dslash_type != QUDA_TWISTED_MASS_DSLASH && dslash_type != QUDA_DOMAIN_WALL_4D_DSLASH
      && dslash_type != QUDA_MOBIUS_DWF_DSLASH && dslash_type != QUDA_TWISTED_CLOVER_DSLASH
      && dslash_type != QUDA_DOMAIN_WALL_DSLASH && dslash_type != QUDA_MOBIUS_DWF_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  QudaInvertParam inv_param = newQudaInvertParam();
  setDeflatedInvertParam(inv_param);

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

  QudaEigParam df_param = newQudaEigParam();
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

  // Allocate host side memory for the gauge field.
  //----------------------------------------------------------------------------
  void *gauge[4];
  // Allocate space on the host (always best to allocate and free in the same scope)
  for (int dir = 0; dir < 4; dir++) gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  // Allocate host side memory for clover terms if needed.
  //----------------------------------------------------------------------------
  void *clover = nullptr;
  void *clover_inv = nullptr;
  // Allocate space on the host (always best to allocate and free in the same scope)
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    clover = malloc(V * clover_site_size * host_clover_data_type_size);
    clover_inv = malloc(V * clover_site_size * host_spinor_data_type_size);
    constructHostCloverField(clover, clover_inv, inv_param);
    // Load the clover terms to the device
    loadCloverQuda(clover, clover_inv, &inv_param);
  }

  void *spinorIn = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
  void *spinorCheck = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);

  void *spinorOut = NULL;
  spinorOut = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);

  // start the timer
  double time0 = -((double)clock());

  // this line ensure that if we need to construct the clover inverse (in either the smoother or the solver) we do so
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) loadCloverQuda(clover, clover_inv, &inv_param);

  void *df_preconditioner  = newDeflationQuda(&df_param);
  inv_param.deflation_op   = df_preconditioner;

  for (int i=0; i<Nsrc; i++) {
    // create a point source at 0 (in each subvolume...  FIXME)
    memset(spinorIn, 0, inv_param.Ls * V * spinor_site_size * host_spinor_data_type_size);
    memset(spinorCheck, 0, inv_param.Ls * V * spinor_site_size * host_spinor_data_type_size);
    memset(spinorOut, 0, inv_param.Ls * V * spinor_site_size * host_spinor_data_type_size);

    if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
      //((float*)spinorIn)[i] = 1.0;
      for (int i = 0; i < inv_param.Ls * V * spinor_site_size; i++) ((float *)spinorIn)[i] = rand() / (float)RAND_MAX;
    } else {
      //((double*)spinorIn)[i] = 1.0;
      for (int i = 0; i < inv_param.Ls * V * spinor_site_size; i++) ((double *)spinorIn)[i] = rand() / (double)RAND_MAX;
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
        ax(0.5 / kappa5, spinorCheck, V * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
      } else if (dslash_type == QUDA_TWISTED_MASS_DSLASH && twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
        ax(0.5 / inv_param.kappa, spinorCheck, 2 * V * spinor_site_size, inv_param.cpu_prec);
      } else {
        ax(0.5 / inv_param.kappa, spinorCheck, V * spinor_site_size, inv_param.cpu_prec);
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
        ax(0.25 / (kappa5 * kappa5), spinorCheck, V * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
      } else {
        ax(0.25 / (inv_param.kappa * inv_param.kappa), spinorCheck, Vh * spinor_site_size, inv_param.cpu_prec);
      }
    }
  }

  int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
  mxpy(spinorIn, spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, vol * spinor_site_size * inv_param.Ls, inv_param.cpu_prec);
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
