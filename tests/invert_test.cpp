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

// Parameters defining the eigensolver
void setEigParam(QudaEigParam &eig_param)
{
  eig_param.eig_type = eig_type;
  eig_param.spectrum = eig_spectrum;
  if ((eig_type == QUDA_EIG_TR_LANCZOS || eig_type == QUDA_EIG_IR_LANCZOS)
      && !(eig_spectrum == QUDA_SPECTRUM_LR_EIG || eig_spectrum == QUDA_SPECTRUM_SR_EIG)) {
    errorQuda("Only real spectrum type (LR or SR) can be passed to Lanczos type solver");
  }

  // The solver will exit when nConv extremal eigenpairs have converged
  if (eig_nConv < 0) {
    eig_param.nConv = eig_nEv;
    eig_nConv = eig_nEv;
  } else {
    eig_param.nConv = eig_nConv;
  }

  eig_param.nEv = eig_nEv;
  eig_param.nKr = eig_nKr;
  eig_param.tol = eig_tol;
  eig_param.batched_rotate = eig_batched_rotate;
  eig_param.require_convergence = eig_require_convergence ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.check_interval = eig_check_interval;
  eig_param.max_restarts = eig_max_restarts;
  eig_param.cuda_prec_ritz = prec;

  eig_param.use_norm_op = eig_use_normop ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.use_dagger = eig_use_dagger ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.compute_svd = eig_compute_svd ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  if (eig_compute_svd) {
    eig_param.use_dagger = QUDA_BOOLEAN_NO;
    eig_param.use_norm_op = QUDA_BOOLEAN_YES;
  }

  eig_param.use_poly_acc = eig_use_poly_acc ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.poly_deg = eig_poly_deg;
  eig_param.a_min = eig_amin;
  eig_param.a_max = eig_amax;

  eig_param.arpack_check = eig_arpack_check ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  strcpy(eig_param.arpack_logfile, eig_arpack_logfile);
  strcpy(eig_param.QUDA_logfile, eig_QUDA_logfile);

  strcpy(eig_param.vec_infile, eig_vec_infile);
  strcpy(eig_param.vec_outfile, eig_vec_outfile);
}

int main(int argc, char **argv)
{

  mg_verbosity[0] = QUDA_SILENT; // set default preconditioner verbosity

  if (multishift) solution_type = QUDA_MATPCDAG_MATPC_SOLUTION; // set a correct default for the multi-shift solver

  // command line options
  auto app = make_app();
  // app->get_formatter()->column_width(40);
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // for (int i = 1; i < argc; i++){
  //   if(process_command_line_option(argc, argv, &i) == 0){
  //     continue;
  //   }
  //   printfQuda("ERROR: Invalid option:%s\n", argv[i]);
  //   usage(argv);
  // }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_refinement_sloppy == QUDA_INVALID_PRECISION) prec_refinement_sloppy = prec_sloppy;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

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
  QudaPrecision cuda_prec_refinement_sloppy = prec_refinement_sloppy;
  QudaPrecision cuda_prec_precondition = prec_precondition;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
  QudaEigParam eig_param = newQudaEigParam();
  setEigParam(eig_param);
  inv_param.eig_param = inv_deflate ? &eig_param : nullptr;

  double kappa5 = 0;

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
  gauge_param.reconstruct_precondition = link_recon_precondition;
  gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
  gauge_param.cuda_prec_refinement_sloppy = cuda_prec_refinement_sloppy;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.dslash_type = dslash_type;

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1 + 3/gauge_param.anisotropy);
  }
  inv_param.mu = mu;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.epsilon = epsilon;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
             dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
	     dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    inv_param.m5 = m5;
    kappa5 = 0.5/(5 + inv_param.m5);  
    inv_param.Ls = Lsdim;
    for(int k = 0; k < Lsdim; k++) // for mobius only
    {
      // b5[k], c[k] values are chosen for arbitrary values,
      // but the difference of them are same as 1.0
      inv_param.b_5[k] = b5;
      inv_param.c_5[k] = c5;
    }
  }

  // offsets used only by multi-shift solver
  inv_param.num_offset = 12;
  double offset[12] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12};
  for (int i=0; i<inv_param.num_offset; i++) inv_param.offset[i] = offset[i];

  inv_param.inv_type = inv_type;
  inv_param.solution_type = solution_type;
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;


  inv_param.pipeline = pipeline;

  inv_param.Nsteps = 2;
  inv_param.gcrNkrylov = gcrNkrylov;
  inv_param.ca_basis = ca_basis;
  inv_param.ca_lambda_min = ca_lambda_min;
  inv_param.ca_lambda_max = ca_lambda_max;
  inv_param.tol = tol;
  inv_param.tol_restart = tol_restart; // now theoretical background for this parameter...
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
  inv_param.reliable_delta = reliable_delta;
  inv_param.use_alternative_reliable = alternative_reliable;
  inv_param.use_sloppy_partial_accumulator = 0;
  inv_param.solution_accumulator_pipeline = solution_accumulator_pipeline;
  inv_param.max_res_increase = 1;

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = precon_type;
    
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = mg_verbosity[0];
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_refinement_sloppy = cuda_prec_refinement_sloppy;
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
    inv_param.clover_cuda_prec_refinement_sloppy = cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = clover_coeff;
  }

  inv_param.verbosity = verbosity;

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
    eig_param.preserve_deflation = i < Nsrc - 1 ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

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

  auto mean_time = 0.0;
  auto mean_time2 = 0.0;
  auto mean_gflops = 0.0;
  auto mean_gflops2 = 0.0;
  for (int i = 0; i < Nsrc; i++) {
    mean_time += time[i];
    mean_time2 += time[i] * time[i];
    mean_gflops += gflops[i];
    mean_gflops2 += gflops[i] * gflops[i];
  }

  mean_time /= Nsrc;
  mean_time2 /= Nsrc;
  auto stddev_time = Nsrc > 1 ? sqrt((Nsrc / ((double)Nsrc - 1.0)) * (mean_time2 - mean_time * mean_time)) : std::numeric_limits<double>::infinity();
  mean_gflops /= Nsrc;
  mean_gflops2 /= Nsrc;
  auto stddev_gflops = Nsrc > 1 ? sqrt((Nsrc / ((double)Nsrc - 1.0)) * (mean_gflops2 - mean_gflops * mean_gflops)) : std::numeric_limits<double>::infinity();
  printfQuda("%d solves, with mean solve time %g (stddev = %g), mean GFLOPS %g (stddev = %g)\n", Nsrc, mean_time,
             stddev_time, mean_gflops, stddev_gflops);

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
