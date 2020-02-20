#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits>

#include <util_quda.h>
#include <random_quda.h>
#include <test_util.h>
#include <test_params.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include <qio_field.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

namespace quda
{
  extern void setTransferGPU(bool);
}

void display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    prec_sloppy   multishift  matpc_type  recon  recon_sloppy S_dimension T_dimension Ls_dimension   dslash_type  normalization\n");
  printfQuda("%6s   %6s          %d     %12s     %2s     %2s         %3d/%3d/%3d     %3d         %2d       %14s  %8s\n",
             get_prec_str(prec), get_prec_str(prec_sloppy), multishift, get_matpc_str(matpc_type),
             get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim, tdim, Lsdim,
             get_dslash_str(dslash_type), get_mass_normalization_str(normalization));

  printfQuda("\n   Eigensolver parameters\n");
  printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
  printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
  printfQuda(" - number of eigenvectors requested %d\n", eig_nConv);
  printfQuda(" - size of eigenvector search space %d\n", eig_nEv);
  printfQuda(" - size of Krylov space %d\n", eig_nKr);
  printfQuda(" - solver tolerance %e\n", eig_tol);
  printfQuda(" - convergence required (%s)\n", eig_require_convergence ? "true" : "false");
  if (eig_compute_svd) {
    printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
    printfQuda(" - ***********************************************************\n");
    printfQuda(" - **** Overriding any previous choices of operator type. ****\n");
    printfQuda(" - ****    SVD demands normal operator, will use MdagM    ****\n");
    printfQuda(" - ***********************************************************\n");
  } else {
    printfQuda(" - Operator: daggered (%s) , norm-op (%s)\n", eig_use_dagger ? "true" : "false",
               eig_use_normop ? "true" : "false");
  }
  if (eig_use_poly_acc) {
    printfQuda(" - Chebyshev polynomial degree %d\n", eig_poly_deg);
    printfQuda(" - Chebyshev polynomial minumum %e\n", eig_amin);
    if (eig_amax <= 0)
      printfQuda(" - Chebyshev polynomial maximum will be computed\n");
    else
      printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
  }

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  
  return ;  
}

int main(int argc, char **argv)
{
  mg_verbosity[0] = QUDA_SILENT; // set default preconditioner verbosity

  if (multishift) solution_type = QUDA_MATPCDAG_MATPC_SOLUTION; // set a correct default for the multi-shift solver

  // command line options
  auto app = make_app();
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_refinement_sloppy == QUDA_INVALID_PRECISION) prec_refinement_sloppy = prec_sloppy;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  // Only these fermions are supported in this file
  if (dslash_type != QUDA_WILSON_DSLASH &&
      dslash_type != QUDA_CLOVER_WILSON_DSLASH &&
      dslash_type != QUDA_TWISTED_MASS_DSLASH &&
      dslash_type != QUDA_DOMAIN_WALL_4D_DSLASH &&
      dslash_type != QUDA_MOBIUS_DWF_DSLASH &&
      dslash_type != QUDA_TWISTED_CLOVER_DSLASH &&
      dslash_type != QUDA_DOMAIN_WALL_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  // Check for deflation
  QudaEigParam eig_param = newQudaEigParam();
  if(inv_deflate) {
    inv_param.eig_param = &eig_param;
    setEigParam(eig_param);
  } else {
    inv_param.eig_param = nullptr;
  }

  // *********************************************************** //
  // *** Everything between here and the call to initQuda() is   //
  // *** application-specific.                                   //
  // *********************************************************** //

  // Set parameters for the reference Dslash, and prepare fields to be loaded
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
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
    double norm = 0.01; // clover components are random numbers in the range (-norm, norm)
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

  void *spinorOut = NULL, **spinorOutMulti = NULL;
  if (multishift) {
    spinorOutMulti = (void**)malloc(inv_param.num_offset*sizeof(void *));
    for (int i=0; i<inv_param.num_offset; i++) {
      spinorOutMulti[i] = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
    }
  } else {
    spinorOut = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  }

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // load the clover term, if desired
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
    loadCloverQuda(clover, clover_inv, &inv_param);

  double *time = new double[Nsrc];
  double *gflops = new double[Nsrc];
  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 1234);
  rng->Init();

  for (int i = 0; i < Nsrc; i++) {

    construct_spinor_source(spinorIn, 4, 3, inv_param.cpu_prec, gauge_param.X, *rng);

    // if deflating preserve the deflation space between solves
    eig_param.preserve_deflation = i < Nsrc - 1 ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    if (multishift) {
      invertMultiShiftQuda(spinorOutMulti, spinorIn, &inv_param);
    } else {
      invertQuda(spinorOut, spinorIn, &inv_param);
    }

    time[i] = inv_param.secs;
    gflops[i] = inv_param.gflops / inv_param.secs;
    printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs,
               inv_param.gflops / inv_param.secs);
  }

  rng->Release();
  delete rng;

  if (Nsrc > 1) {
    auto mean_time = 0.0;
    auto mean_time2 = 0.0;
    auto mean_gflops = 0.0;
    auto mean_gflops2 = 0.0;
    // skip first solve due to allocations, potential UVM swapping overhead
    for (int i = 1; i < Nsrc; i++) {
      mean_time += time[i];
      mean_time2 += time[i] * time[i];
      mean_gflops += gflops[i];
      mean_gflops2 += gflops[i] * gflops[i];
    }

    auto NsrcM1 = Nsrc - 1;

    mean_time /= NsrcM1;
    mean_time2 /= NsrcM1;
    auto stddev_time = NsrcM1 > 1 ? sqrt((NsrcM1 / ((double)NsrcM1 - 1.0)) * (mean_time2 - mean_time * mean_time)) :
                                    std::numeric_limits<double>::infinity();
    mean_gflops /= NsrcM1;
    mean_gflops2 /= NsrcM1;
    auto stddev_gflops = NsrcM1 > 1 ?
      sqrt((NsrcM1 / ((double)NsrcM1 - 1.0)) * (mean_gflops2 - mean_gflops * mean_gflops)) :
      std::numeric_limits<double>::infinity();
    printfQuda(
      "%d solves, with mean solve time %g (stddev = %g), mean GFLOPS %g (stddev = %g) [excluding first solve]\n", Nsrc,
      mean_time, stddev_time, mean_gflops, stddev_gflops);
  }

  delete[] time;
  delete[] gflops;

  if (multishift) {
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
      errorQuda("Mass normalization not supported for multi-shift solver in invert_test");
    }

    void *spinorTmp = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

    printfQuda("Host residuum checks: \n");
    for(int i=0; i < inv_param.num_offset; i++) {
      ax(0, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
      
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          int tm_offset = Vh*spinorSiteSize;
	  void *out0 = spinorCheck;
	  void *out1 = (char*)out0 + tm_offset*cpu_prec;

	  void *tmp0 = spinorTmp;
	  void *tmp1 = (char*)tmp0 + tm_offset*cpu_prec;

	  void *in0  = spinorOutMulti[i];
	  void *in1  = (char*)in0 + tm_offset*cpu_prec;

	  tm_ndeg_matpc(tmp0, tmp1, gauge, in0, in1, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	  tm_ndeg_matpc(out0, out1, gauge, tmp0, tmp1, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	} else {
	  tm_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		   inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	  tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		   inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	}
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
	  errorQuda("Twisted mass solution type not supported");
	tmc_matpc(spinorTmp, gauge, spinorOutMulti[i], clover, clover_inv, inv_param.kappa, inv_param.mu,
		  inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        tmc_matpc(spinorCheck, gauge, spinorTmp, clover, clover_inv, inv_param.kappa, inv_param.mu,
		  inv_param.twist_flavor, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_WILSON_DSLASH) {
        wil_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0,
                  inv_param.cpu_prec, gauge_param);
        wil_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
                  inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_matpc(spinorTmp, gauge, clover, clover_inv, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0,
		     inv_param.cpu_prec, gauge_param);
        clover_matpc(spinorCheck, gauge, clover, clover_inv, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
		     inv_param.cpu_prec, gauge_param);
      } else {
        printfQuda("Domain wall not supported for multi-shift\n");
        exit(-1);
      }

      axpy(inv_param.offset[i], spinorOutMulti[i], spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
      mxpy(spinorIn, spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
      double nrm2 = norm_2(spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
      double src2 = norm_2(spinorIn, Vh*spinorSiteSize, inv_param.cpu_prec);
      double l2r = sqrt(nrm2 / src2);

      printfQuda("Shift %d residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
		 i, inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r, 
		 inv_param.tol_hq_offset[i], inv_param.true_res_hq_offset[i]);
    }
    free(spinorTmp);

  } else {
    
    if (inv_param.solution_type == QUDA_MAT_SOLUTION) {

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	if(inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
	  tm_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0, inv_param.cpu_prec, gauge_param);
	} else {
          int tm_offset = V*spinorSiteSize;
	  void *evenOut = spinorCheck;
	  void *oddOut  = (char*)evenOut + tm_offset*cpu_prec;

	  void *evenIn  = spinorOut;
	  void *oddIn   = (char*)evenIn + tm_offset*cpu_prec;

	  tm_ndeg_mat(evenOut, oddOut, gauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon, 0, inv_param.cpu_prec, gauge_param);
	}
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	tmc_mat(spinorCheck, gauge, clover, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0,
		inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_WILSON_DSLASH) {
        wil_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_mat(spinorCheck, gauge, clover, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
        dw_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
      } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
        dw_4d_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
      } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
        double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
        double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
        for(int xs = 0 ; xs < Lsdim ; xs++)
        {
          kappa_b[xs] = 1.0/(2*(inv_param.b_5[xs]*(4.0 + inv_param.m5) + 1.0));
          kappa_c[xs] = 1.0/(2*(inv_param.c_5[xs]*(4.0 + inv_param.m5) - 1.0));
        }
	mdw_mat(spinorCheck, gauge, spinorOut, kappa_b, kappa_c, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
	free(kappa_b);
	free(kappa_c);
      } else {
        errorQuda("Unsupported dslash_type");
      }
      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || 
            dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
            dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
          ax(0.5/kappa5, spinorCheck, V*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
        } else if (dslash_type == QUDA_TWISTED_MASS_DSLASH && twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
          ax(0.5/inv_param.kappa, spinorCheck, 2*V*spinorSiteSize, inv_param.cpu_prec);
	} else {
          ax(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
        }
      }

    } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          int tm_offset = Vh*spinorSiteSize;
	  void *out0 = spinorCheck;
	  void *out1 = (char*)out0 + tm_offset*cpu_prec;

	  void *in0  = spinorOut;
	  void *in1  = (char*)in0 + tm_offset*cpu_prec;

	  tm_ndeg_matpc(out0, out1, gauge, in0, in1, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	} else {
	  tm_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		   inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	}
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
	  errorQuda("Twisted mass solution type not supported");
        tmc_matpc(spinorCheck, gauge, spinorOut, clover, clover_inv, inv_param.kappa, inv_param.mu,
		  inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_WILSON_DSLASH) {
        wil_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0,
                  inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_matpc(spinorCheck, gauge, clover, clover_inv, spinorOut, inv_param.kappa, inv_param.matpc_type, 0,
		     inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
        dw_matpc(spinorCheck, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
      } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
        dw_4d_matpc(spinorCheck, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
      } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
        double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
        double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
        for(int xs = 0 ; xs < Lsdim ; xs++)
        {
          kappa_b[xs] = 1.0/(2*(inv_param.b_5[xs]*(4.0 + inv_param.m5) + 1.0));
          kappa_c[xs] = 1.0/(2*(inv_param.c_5[xs]*(4.0 + inv_param.m5) - 1.0));
        }
        mdw_matpc(spinorCheck, gauge, spinorOut, kappa_b, kappa_c, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        free(kappa_b);
        free(kappa_c);
      } else {
        errorQuda("Unsupported dslash_type");
      }

      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
            dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
            dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
          ax(0.25/(kappa5*kappa5), spinorCheck, V*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
        } else {
          ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
      
	}
      }

    } else if (inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {

      void *spinorTmp = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

      ax(0, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
      
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
          int tm_offset = Vh*spinorSiteSize;
	  void *out0 = spinorCheck;
	  void *out1 = (char*)out0 + tm_offset*cpu_prec;

	  void *tmp0 = spinorTmp;
	  void *tmp1 = (char*)tmp0 + tm_offset*cpu_prec;

	  void *in0  = spinorOut;
	  void *in1  = (char*)in0 + tm_offset*cpu_prec;

	  tm_ndeg_matpc(tmp0, tmp1, gauge, in0, in1, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	  tm_ndeg_matpc(out0, out1, gauge, tmp0, tmp1, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	} else {
	  tm_matpc(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		   inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	  tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		   inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
	}
      } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
	  errorQuda("Twisted mass solution type not supported");
        tmc_matpc(spinorTmp, gauge, spinorOut, clover, clover_inv, inv_param.kappa, inv_param.mu,
		  inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        tmc_matpc(spinorCheck, gauge, spinorTmp, clover, clover_inv, inv_param.kappa, inv_param.mu,
		  inv_param.twist_flavor, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_WILSON_DSLASH) {
        wil_matpc(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0,
                  inv_param.cpu_prec, gauge_param);
        wil_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
                  inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        clover_matpc(spinorTmp, gauge, clover, clover_inv, spinorOut, inv_param.kappa,
		     inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        clover_matpc(spinorCheck, gauge, clover, clover_inv, spinorTmp, inv_param.kappa,
		     inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
        dw_matpc(spinorTmp, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
        dw_matpc(spinorCheck, gauge, spinorTmp, kappa5, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param, inv_param.mass);
      } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
        dw_4d_matpc(spinorTmp, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
        dw_4d_matpc(spinorCheck, gauge, spinorTmp, kappa5, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param, inv_param.mass);
      } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
        double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
        double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
        for(int xs = 0 ; xs < Lsdim ; xs++)
        {
          kappa_b[xs] = 1.0/(2*(inv_param.b_5[xs]*(4.0 + inv_param.m5) + 1.0));
          kappa_c[xs] = 1.0/(2*(inv_param.c_5[xs]*(4.0 + inv_param.m5) - 1.0));
        }
        mdw_matpc(spinorTmp, gauge, spinorOut, kappa_b, kappa_c, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        mdw_matpc(spinorCheck, gauge, spinorTmp, kappa_b, kappa_c, inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        free(kappa_b);
        free(kappa_c);
      } else {
        errorQuda("Unsupported dslash_type");
      }

      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	errorQuda("Mass normalization not implemented");
      }

      free(spinorTmp);
    }


    int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
    mxpy(spinorIn, spinorCheck, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
    double nrm2 = norm_2(spinorCheck, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
    double src2 = norm_2(spinorIn, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
    double l2r = sqrt(nrm2 / src2);

    printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
	       inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq);

  }

  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();
  
  // finalize the QUDA library
  endQuda();

  finalizeComms();

  free(spinorIn);
  free(spinorCheck);
  if (multishift) {
    for (int i = 0; i < inv_param.num_offset; i++) free(spinorOutMulti[i]);
    free(spinorOutMulti);
  } else {
    free(spinorOut);
  }

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }

  for (int dir = 0; dir<4; dir++) free(gauge[dir]);

  return 0;
}
