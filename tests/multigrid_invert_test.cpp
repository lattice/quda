#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include "face_quda.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

// Wilson, clover-improved Wilson, twisted mass, and domain wall are supported.
extern QudaDslashType dslash_type;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaPrecision prec_sloppy;
extern QudaPrecision prec_precondition;
extern QudaPrecision prec_null;
extern QudaReconstructType link_recon_sloppy;
extern QudaReconstructType link_recon_precondition;
extern double mass;
extern double mu;
extern double anisotropy;
extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern char latfile[];
extern int Nsrc; // number of spinors to apply to simultaneously
extern int niter;
extern int gcrNkrylov; // number of inner iterations for GCR, or l for BiCGstab-l
extern int pipeline; // length of pipeline for fused operations in GCR or BiCGstab-l
extern int nvec[];
extern int mg_levels;

extern bool generate_nullspace;
extern bool generate_all_levels;
extern bool verify_results;
extern int nu_pre;
extern int nu_post;
extern int geo_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM];
extern double mu_factor[QUDA_MAX_MG_LEVEL];

extern QudaVerbosity mg_verbosity[QUDA_MAX_MG_LEVEL];

extern QudaSetupType setup_type;
extern bool pre_orthonormalize;
extern bool post_orthonormalize;
extern double omega;
extern QudaInverterType smoother_type;
extern QudaInverterType coarsest_solver;
extern double coarsest_tol;
extern int coarsest_maxiter;

extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;

extern char vec_infile[];
extern char vec_outfile[];

//Twisted mass flavor type
extern QudaTwistFlavorType twist_flavor;

extern void usage(char** );

extern double clover_coeff;
extern bool compute_clover;

extern bool verify_results;

namespace quda {
  extern void setTransferGPU(bool);
}

void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim, Lsdim);     

  printfQuda("MG parameters\n");
  printfQuda(" - number of levels %d\n", mg_levels);
  for (int i=0; i<mg_levels-1; i++) printfQuda(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);
  printfQuda(" - number of pre-smoother applications %d\n", nu_pre);
  printfQuda(" - number of post-smoother applications %d\n", nu_post);

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  
  return ;
  
}

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

void setGaugeParam(QudaGaugeParam &gauge_param) {
  setDefaultGaugeParam(gauge_param);
  
  // Add here additional changings

}

void setMultigridParam(QudaMultigridParam &mg_param) {
  setDefaultMultigridParam(mg_param);

  // Add here additional changings

}

void setInvertParam(QudaInvertParam &inv_param) {
  setDefaultInvertParam(inv_param);

  // Add here additional changings
  inv_param.verbosity_precondition = mg_verbosity[0];
  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
}

int main(int argc, char **argv)
{
  process_command_line(argc, argv);

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** QUDA parameters begin here.

  if (dslash_type != QUDA_WILSON_DSLASH &&
      dslash_type != QUDA_CLOVER_WILSON_DSLASH &&
      dslash_type != QUDA_TWISTED_MASS_DSLASH &&
      dslash_type != QUDA_TWISTED_CLOVER_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  mg_param.invert_param = &mg_inv_param;

  setMultigridParam(mg_param);


  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  setDims(gauge_param.X);

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
  } else { // else generate a random SU(3) field
    //generate a random SU(3) field
    //construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
    //generate a unit SU(3) field
    construct_gauge_field(gauge, 0, gauge_param.cpu_prec, &gauge_param);
    
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
  if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE) inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) loadCloverQuda(clover, clover_inv, &inv_param);

  inv_param.solve_type = solve_type; // restore actual solve_type we want to do

  // setup the multigrid solver
  void *mg_preconditioner = newMultigridQuda(&mg_param);
  inv_param.preconditioner = mg_preconditioner;

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
    //printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
    //inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);
    printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
               inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, 0.0);
  }

  // free the multigrid solver
  destroyMultigridQuda(mg_preconditioner);



  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
    
  printfQuda("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
	 inv_param.spinorGiB, gauge_param.gaugeGiB);
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) printfQuda("   Clover: %f GiB\n", inv_param.cloverGiB);

  if (inv_param.solution_type == QUDA_MAT_SOLUTION) {
    
    if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      wil_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
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
      ax(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
    }
    
  } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {
    
    if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      wil_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0, 
		inv_param.cpu_prec, gauge_param);
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
      ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
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
