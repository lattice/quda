#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_params.h>
#include <test_util.h>
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

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

void
display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    prec_sloppy   multishift  matpc_type  recon  recon_sloppy S_dimension T_dimension Ls_dimension   dslash_type  normalization\n");
  printfQuda("%6s   %6s          %d     %12s     %2s     %2s         %3d/%3d/%3d     %3d         %2d       %14s  %8s\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy), multishift, get_matpc_str(matpc_type),
	     get_recon_str(link_recon),
	     get_recon_str(link_recon_sloppy),
	     xdim, ydim, zdim, tdim, Lsdim,
	     get_dslash_str(dslash_type),
	     get_mass_normalization_str(normalization));

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

  auto app = make_app();
  // app->get_formatter()->column_width(40);
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION){
    prec_sloppy = prec;
  }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID){
    link_recon_sloppy = link_recon;
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  // *** QUDA parameters begin here.

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

  QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec = prec;
  QudaPrecision cuda_prec_sloppy = prec_sloppy;
  QudaPrecision cuda_prec_precondition = QUDA_HALF_PRECISION;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();

  double kappa5;

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;
  inv_param.Ls = 1;

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
  gauge_param.reconstruct_precondition = link_recon_sloppy;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.dslash_type = dslash_type;

  inv_param.mass = mass;
  inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = 0.12;
    inv_param.epsilon = 0.1385;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
             dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
    inv_param.m5 = -1.8;
    kappa5 = 0.5/(5 + inv_param.m5);
    inv_param.Ls = Lsdim;
  } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    inv_param.m5 = -1.8;
    kappa5 = 0.5/(5 + inv_param.m5);
    inv_param.Ls = Lsdim;
    for(int k = 0; k < Lsdim; k++)
    {
      // b5[k], c[k] values are chosen for arbitrary values,
      // but the difference of them are same as 1.0
      inv_param.b_5[k] = 1.452;
      inv_param.c_5[k] = 0.452;
    }
  }

  // offsets used only by multi-shift solver
  inv_param.num_offset = 12;
  inv_param.num_src = Nsrc;
  double offset[12] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12};
  for (int i=0; i<inv_param.num_offset; i++) inv_param.offset[i] = offset[i];

  inv_param.inv_type = inv_type;
  if (inv_param.dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.solution_type = QUDA_MAT_SOLUTION;
  } else {
    inv_param.solution_type = multishift ? QUDA_MATPCDAG_MATPC_SOLUTION : QUDA_MATPC_SOLUTION;
  }
  inv_param.matpc_type = matpc_type;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      dslash_type == QUDA_MOBIUS_DWF_DSLASH ||
      dslash_type == QUDA_TWISTED_MASS_DSLASH ||
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH ||
      multishift || inv_type == QUDA_CG_INVERTER) {
    inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
  } else {
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  }

  inv_param.pipeline = 0;

  inv_param.Nsteps = 2;
  inv_param.gcrNkrylov = 10;
  inv_param.tol = tol;
  inv_param.tol_restart = 1e-3; //now theoretical background for this parameter...
  if(tol_hq == 0 && tol == 0){
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }
  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType_s>(0);
  inv_param.residual_type = (tol != 0) ? static_cast<QudaResidualType_s> ( inv_param.residual_type | QUDA_L2_RELATIVE_RESIDUAL) : inv_param.residual_type;
  inv_param.residual_type = (tol_hq != 0) ? static_cast<QudaResidualType_s> (inv_param.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : inv_param.residual_type;

  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta =  1e-1;
  inv_param.use_sloppy_partial_accumulator = 0;
  inv_param.max_res_increase = 1;

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = precon_type;

  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = QUDA_SILENT;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  gauge_param.ga_pad = 0; // 24*24*24/2;
  inv_param.sp_pad = 0; // 24*24*24/2;
  inv_param.cl_pad = 0; // 24*24*24/2;

  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;
#endif

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = 1.5*inv_param.kappa;
  }

  inv_param.verbosity = QUDA_VERBOSE;

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
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

  void *gauge[4], *clover_inv=0, *clover=0;

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

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    double norm = 0.0; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = (inv_param.clover_cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    construct_clover_field(clover_inv, norm, diag, inv_param.clover_cpu_prec);

    // The uninverted clover term is only needed when solving the unpreconditioned
    // system or when using "asymmetric" even/odd preconditioning.
    int preconditioned = (inv_param.solve_type == QUDA_DIRECT_PC_SOLVE ||
			  inv_param.solve_type == QUDA_NORMOP_PC_SOLVE);
    int asymmetric = preconditioned &&
                         (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ||
                          inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);
    if (!preconditioned) {
      clover = clover_inv;
      clover_inv = NULL;
    } else if (asymmetric) { // fake it by using the same random matrix
      clover = clover_inv;   // for both clover and clover_inv
    } else {
      clover = NULL;
    }
  }



  void **spinorIn = (void**)malloc(inv_param.num_src*sizeof(void *));
  void **spinorCheck = (void**)malloc(inv_param.num_src*sizeof(void *));
  for (int i=0; i<inv_param.num_src; i++) {
    spinorIn[i] = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
    spinorCheck[i] = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  }

  void **spinorOutMulti = NULL;
//  if (multishift) {
    spinorOutMulti = (void**)malloc(inv_param.num_src*sizeof(void *));
    for (int i=0; i<inv_param.num_src; i++) {
      spinorOutMulti[i] = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
    }
//  } else {
//    spinorOut = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
//  }


//  if (multishift) {
    for (int i=0; i<inv_param.num_src; i++) {
      memset(spinorOutMulti[i], 0, inv_param.Ls*V*spinorSiteSize*sSize);
      memset(spinorIn[i], 0, inv_param.Ls*V*spinorSiteSize*sSize);
      memset(spinorCheck[i], 0, inv_param.Ls*V*spinorSiteSize*sSize);
    }
//  } else {
//    memset(spinorOut, 0, inv_param.Ls*V*spinorSiteSize*sSize);
//  }

  // create a point source at 0 (in each subvolume...  FIXME)

  // create a point source at 0 (in each subvolume...  FIXME)
    for (int j=0; j<inv_param.num_src; j++) {
  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
    //((float*)spinorIn)[0] = 1.0;
    for (int i=0; i<inv_param.Ls*V; i++) ((float*)spinorIn[j])[i*spinorSiteSize+j] = rand() / (float)RAND_MAX;
  } else {
    //((double*)spinorIn)[0] = 1.0;
    for (int i=0; i<inv_param.Ls*V; i++) ((double*)spinorIn[j])[i*spinorSiteSize+j] = rand() / (double)RAND_MAX;
  }
    }
  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  // load the clover term, if desired
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) loadCloverQuda(clover, clover_inv, &inv_param);

  if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) loadCloverQuda(NULL, NULL, &inv_param);

  // perform the inversion
//  if (multishift) {
//    invertMultiShiftQuda(spinorOutMulti, spinorIn, &inv_param);
//  } else {
    invertMultiSrcQuda(spinorOutMulti, spinorIn, &inv_param);


  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n",
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

   for(int i = 0; i < inv_param.num_src ; i++){
     invertQuda(spinorOutMulti[i], spinorIn[i], &inv_param);
   }

//  if (true) {
//    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
//      errorQuda("Mass normalization not supported for multi-shift solver in invert_test");
//    }
//
//    void *spinorTmp = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
//
//    printfQuda("Host residuum checks: \n");
//    for(int i=0; i < inv_param.num_src; i++) {
//      ax(0, spinorCheck[i], V*spinorSiteSize, inv_param.cpu_prec);
//
//      if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
//	if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
//	  errorQuda("Twisted mass solution type not supported");
//        tm_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
//                 inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
//        tm_matpc(spinorCheck[i], gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
//                 inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
//      } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
//        wil_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0,
//                  inv_param.cpu_prec, gauge_param);
//        wil_matpc(spinorCheck[i], gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
//                  inv_param.cpu_prec, gauge_param);
//      } else {
//        printfQuda("Domain wall not supported for multi-shift\n");
//        exit(-1);
//      }
//
//      axpy(inv_param.offset[i], spinorOutMulti[i], spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
//      mxpy(spinorIn, spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
//      double nrm2 = norm_2(spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
//      double src2 = norm_2(spinorIn, Vh*spinorSiteSize, inv_param.cpu_prec);
//      double l2r = sqrt(nrm2 / src2);
//
//      printfQuda("Shift %d residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
//		 i, inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r,
//		 inv_param.tol_hq_offset[i], inv_param.true_res_hq_offset[i]);
//    }
//    free(spinorTmp);
//  }
//    #if 0
//   else
  for (int i=0; i <inv_param.num_src; i++){

    if (inv_param.solution_type == QUDA_MAT_SOLUTION) {

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	if(inv_param.twist_flavor == QUDA_TWIST_SINGLET)
	  tm_mat(spinorCheck[i], gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0, inv_param.cpu_prec, gauge_param);
	else
	{
          int tm_offset = V*spinorSiteSize; //12*spinorRef->Volume();
	  void *evenOut = spinorCheck[i];
	  void *oddOut  = cpu_prec == sizeof(double) ? (void*)((double*)evenOut + tm_offset): (void*)((float*)evenOut + tm_offset);

	  void *evenIn  = spinorOutMulti[i];
	  void *oddIn   = cpu_prec == sizeof(double) ? (void*)((double*)evenIn + tm_offset): (void*)((float*)evenIn + tm_offset);

	  tm_ndeg_mat(evenOut, oddOut, gauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon, 0, inv_param.cpu_prec, gauge_param);
	}
      } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        wil_mat(spinorCheck[i], gauge, spinorOutMulti[i], inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
        dw_mat(spinorCheck[i], gauge, spinorOutMulti[i], kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
//      } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
//        dw_4d_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
//      } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
//        mdw_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
      } else {
        printfQuda("Unsupported dslash_type\n");
        exit(-1);
      }
      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
            dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
            dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
          ax(0.5/kappa5, spinorCheck[i], V*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
        } else if (dslash_type == QUDA_TWISTED_MASS_DSLASH && twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
          ax(0.5/inv_param.kappa, spinorCheck[i], 2*V*spinorSiteSize, inv_param.cpu_prec);
	} else {
          ax(0.5/inv_param.kappa, spinorCheck[i], V*spinorSiteSize, inv_param.cpu_prec);
        }
      }

    } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
	  errorQuda("Twisted mass solution type not supported");
        tm_matpc(spinorCheck[i], gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                 inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        wil_matpc(spinorCheck[i], gauge, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0,
                  inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
        dw_matpc(spinorCheck[i], gauge, spinorOutMulti[i], kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
      } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
        dw_4d_matpc(spinorCheck[i], gauge, spinorOutMulti[i], kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
      } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
        double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
        double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
        for(int xs = 0 ; xs < Lsdim ; xs++)
        {
          kappa_b[xs] = 1.0/(2*(inv_param.b_5[xs]*(4.0 + inv_param.m5) + 1.0));
          kappa_c[xs] = 1.0/(2*(inv_param.c_5[xs]*(4.0 + inv_param.m5) - 1.0));
        }
        mdw_matpc(spinorCheck[i], gauge, spinorOutMulti[i], kappa_b, kappa_c, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        free(kappa_b);
        free(kappa_c);
      } else {
        printfQuda("Unsupported dslash_type\n");
        exit(-1);
      }

      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
            dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
            dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
          ax(0.25/(kappa5*kappa5), spinorCheck[i], V*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
        } else {
          ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck[i], Vh*spinorSiteSize, inv_param.cpu_prec);

	}
      }

    } else if (inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {

      void *spinorTmp = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

      ax(0, spinorCheck[i], V*spinorSiteSize, inv_param.cpu_prec);

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	if (inv_param.twist_flavor != QUDA_TWIST_SINGLET)
	  errorQuda("Twisted mass solution type not supported");
        tm_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                 inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        tm_matpc(spinorCheck[i], gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                 inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        wil_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0,
                  inv_param.cpu_prec, gauge_param);
        wil_matpc(spinorCheck[i], gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
                  inv_param.cpu_prec, gauge_param);
      } else {
        printfQuda("Unsupported dslash_type\n");
        exit(-1);
      }

      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	errorQuda("Mass normalization not implemented");
      }

      free(spinorTmp);
    }


    int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
    mxpy(spinorIn[i], spinorCheck[i], vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
    double nrm2 = norm_2(spinorCheck[i], vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
    double src2 = norm_2(spinorIn[i], vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
    double l2r = sqrt(nrm2 / src2);

//    printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
//	       inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq);
          printfQuda("rhs %d residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
         i, inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r,
         inv_param.tol_hq_offset[i], inv_param.true_res_hq_offset[i]);
  }

  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();

  // finalize the QUDA library
  endQuda();

  finalizeComms();

  return 0;
}
