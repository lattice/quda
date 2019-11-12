#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <misc.h>
#include <test_util.h>
#include <test_params.h>
#include <dslash_util.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <llfat_reference.h>
#include <gauge_field.h>
#include <unitarization_links.h>
#include <blas_reference.h>
#include <random_quda.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#define mySpinorSiteSize 6

// extern void usage(char** argv);

void** ghost_fatlink, **ghost_longlink;

// extern int device;

QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
size_t gSize = sizeof(double);

static int n_naiks = 1;

// For loading the gauge fields
int argc_copy;
char** argv_copy;

int X[4];

cpuColorSpinorField *in;
cpuColorSpinorField *out;
cpuColorSpinorField *ref;
cpuColorSpinorField *tmp;

static void end();

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

void setGaugeParam(QudaGaugeParam &gauge_param)
{
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
  gauge_param.cuda_prec_sloppy = prec_sloppy;
  gauge_param.cuda_prec_refinement_sloppy = prec_refinement_sloppy;
  gauge_param.cuda_prec_precondition = prec_precondition;

  gauge_param.anisotropy = 1.0;

  // For HISQ, this must always be set to 1.0, since the tadpole
  // correction is baked into the coefficients for the first fattening.
  // The tadpole doesn't mean anything for the second fattening
  // since the input fields are unitarized.
  gauge_param.tadpole_coeff = 1.0;

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.scale = -1.0 / 24.0;
    if (eps_naik != 0) { gauge_param.scale *= (1.0 + eps_naik); }
  } else {
    gauge_param.scale = 1.0;
  }
  gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param.type = QUDA_WILSON_LINKS;

  gauge_param.ga_pad = 0;

#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2;
  int t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;
#endif
}

void setInvertParam(QudaInvertParam &inv_param)
{
  // Solver params
  inv_param.verbosity = QUDA_VERBOSE;
  inv_param.mass = mass;
  inv_param.kappa = kappa = 1.0 / (8.0 + mass); // for Laplace operator
  inv_param.laplace3D = laplace3D;              // for Laplace operator

  // outer solver parameters
  inv_param.inv_type = inv_type;
  inv_param.tol = tol;
  inv_param.tol_restart = tol_restart; // now theoretical background for this parameter...
  inv_param.maxiter = niter;
  inv_param.reliable_delta = reliable_delta;
  inv_param.use_alternative_reliable = alternative_reliable;
  inv_param.use_sloppy_partial_accumulator = false;
  inv_param.solution_accumulator_pipeline = solution_accumulator_pipeline;
  inv_param.pipeline = pipeline;

  inv_param.Ls = Nsrc;

  if (tol_hq == 0 && tol == 0) {
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }
  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType_s>(0);
  inv_param.residual_type = (tol != 0) ?
    static_cast<QudaResidualType_s>(inv_param.residual_type | QUDA_L2_RELATIVE_RESIDUAL) :
    inv_param.residual_type;
  inv_param.residual_type = (tol_hq != 0) ?
    static_cast<QudaResidualType_s>(inv_param.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) :
    inv_param.residual_type;
  inv_param.heavy_quark_check = (inv_param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL ? 5 : 0);

  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  inv_param.Nsteps = 2;

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = precon_type;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = QUDA_SILENT;

  // Specify Krylov sub-size for GCR, BICGSTAB(L), basis size for CA-CG, CA-GCR
  inv_param.gcrNkrylov = gcrNkrylov;

  // Specify basis for CA-CG, lambda min/max for Chebyshev basis
  //   lambda_max < lambda_max . use power iters to generate
  inv_param.ca_basis = ca_basis;
  inv_param.ca_lambda_min = ca_lambda_min;
  inv_param.ca_lambda_max = ca_lambda_max;

  inv_param.solution_type = solution_type;
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = prec;
  inv_param.cuda_prec_sloppy = prec_sloppy;
  inv_param.cuda_prec_precondition = prec_precondition;
  inv_param.cuda_prec_refinement_sloppy = prec_refinement_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // this is meaningless, but must be thus set
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.dslash_type = dslash_type;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  int tmpint = MAX(X[1] * X[2] * X[3], X[0] * X[2] * X[3]);
  tmpint = MAX(tmpint, X[0] * X[1] * X[3]);
  tmpint = MAX(tmpint, X[0] * X[1] * X[2]);

  inv_param.sp_pad = tmpint;
}

int invert_test()
{

  // Ensure that the default is improved staggered
  if (dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH)
    dslash_type = QUDA_ASQTAD_DSLASH;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  QudaEigParam eig_param = newQudaEigParam();
  setEigParam(eig_param);
  inv_param.eig_param = inv_deflate ? &eig_param : nullptr;

  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gauge_param.X);
  dw_setDims(gauge_param.X, Nsrc); // so we can use 5-d indexing from dwf
  setSpinorSiteSize(6);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void* qdp_inlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_fatlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_longlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* milc_fatlink = nullptr;
  void* milc_longlink = nullptr;

  for (int dir = 0; dir < 4; dir++) {
    qdp_inlink[dir] = malloc(V*gaugeSiteSize*gSize);
    qdp_fatlink[dir] = malloc(V*gaugeSiteSize*gSize);
    qdp_longlink[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  milc_fatlink = malloc(4*V*gaugeSiteSize*gSize);
  milc_longlink = malloc(4 * V * gaugeSiteSize * gSize);

  // for load, etc
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;

  // load a field WITHOUT PHASES
  if (strcmp(latfile, "")) {
    read_gauge_field(latfile, qdp_inlink, gauge_param.cpu_prec, gauge_param.X, argc_copy, argv_copy);
    if (dslash_type != QUDA_LAPLACE_DSLASH) {
      applyGaugeFieldScaling_long(qdp_inlink, Vh, &gauge_param, QUDA_STAGGERED_DSLASH, gauge_param.cpu_prec);
    }
  } else {
    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      construct_gauge_field(qdp_inlink, 1, gauge_param.cpu_prec, &gauge_param);
    } else {
      construct_fat_long_gauge_field(qdp_inlink, qdp_longlink, 1, gauge_param.cpu_prec, &gauge_param,
                                     compute_fatlong ? QUDA_STAGGERED_DSLASH : dslash_type);
    }
  }

  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);

  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // QUDA_STAGGERED_DSLASH follows the same codepath whether or not you
  // "compute" the fat/long links or not.
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink[dir], qdp_inlink[dir], V * gaugeSiteSize * gSize);
      memset(qdp_longlink[dir],0,V*gaugeSiteSize*gSize);
    }
  } else { // QUDA_ASQTAD_DSLASH

    if (compute_fatlong) {
      computeFatLongGPU(qdp_fatlink, qdp_longlink, qdp_inlink, gauge_param, gSize, n_naiks, eps_naik);
    } else {
      for (int dir = 0; dir < 4; dir++) {
        memcpy(qdp_fatlink[dir],qdp_inlink[dir], V*gaugeSiteSize*gSize);
      }
    }

    // Compute fat link plaquette
    computeStaggeredPlaquetteQDPOrder(qdp_fatlink, plaq, gauge_param, dslash_type);

    printfQuda("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }

  // Alright, we've created all the void** links.
  // Create the void* pointers
  reorderQDPtoMILC(milc_fatlink, qdp_fatlink, V, gaugeSiteSize, gauge_param.cpu_prec, gauge_param.cpu_prec);
  reorderQDPtoMILC(milc_longlink, qdp_longlink, V, gaugeSiteSize, gauge_param.cpu_prec, gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  csParam.nColor=3;
  csParam.nSpin = 1;
  csParam.nDim = 5;
  for (int d = 0; d < 4; d++) csParam.x[d] = gauge_param.X[d];
  bool pc = (inv_param.solution_type == QUDA_MATPC_SOLUTION || inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  if (pc) csParam.x[0] /= 2;
  csParam.x[4] = Nsrc;

  csParam.setPrecision(inv_param.cpu_prec);
  csParam.pad = 0;
  csParam.siteSubset = pc ? QUDA_PARITY_SITE_SUBSET : QUDA_FULL_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  in = new cpuColorSpinorField(csParam);
  out = new cpuColorSpinorField(csParam);
  ref = new cpuColorSpinorField(csParam);
  tmp = new cpuColorSpinorField(csParam);

  // Construct source
  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 1234);
  rng->Init();

  construct_spinor_source(in->V(), 1, 3, inv_param.cpu_prec, csParam.x, *rng);

  rng->Release();
  delete rng;

#ifdef MULTI_GPU
  int tmp_value = MAX(ydim*zdim*tdim/2, xdim*zdim*tdim/2);
  tmp_value = MAX(tmp_value, xdim * ydim * tdim / 2);
  tmp_value = MAX(tmp_value, xdim * ydim * zdim / 2);

  int fat_pad = tmp_value;
  int link_pad =  3*tmp_value;

  // FIXME: currently assume staggered is SU(3)
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuFatParam(milc_fatlink, gauge_param);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuGaugeField *cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink = (void**)cpuFat->Ghost();

  gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(milc_longlink, gauge_param);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuGaugeField *cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink = (void**)cpuLong->Ghost();

#else
  int fat_pad = 0;
  int link_pad = 0;
#endif

  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;
  gauge_param.ga_pad = fat_pad;
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
    gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
  } else {
    gauge_param.reconstruct = gauge_param.reconstruct_sloppy = gauge_param.reconstruct_refinement_sloppy
      = QUDA_RECONSTRUCT_NO;
  }
  gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;

  loadGaugeQuda(milc_fatlink, &gauge_param);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
    gauge_param.ga_pad = link_pad;
    gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
    gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
    gauge_param.reconstruct_precondition = link_recon_precondition;
    loadGaugeQuda(milc_longlink, &gauge_param);
  }

  double time0 = -((double)clock()); // Start the timer

  double nrm2=0;
  double src2=0;
  int ret = 0;

  int len = 0;
  if (solution_type == QUDA_MAT_SOLUTION || solution_type == QUDA_MATDAG_MAT_SOLUTION) {
    len = V*Nsrc;
  } else {
    len = Vh*Nsrc;
  }

  switch (test_type) {
  case 0: // full parity solution
  case 1: // solving prec system, reconstructing
  case 2:

    invertQuda(out->V(), in->V(), &inv_param);
    time0 += clock(); // stop the timer
    time0 /= CLOCKS_PER_SEC;

    // In QUDA, the full staggered operator has the sign convention
    //{{m, -D_eo},{-D_oe,m}}, while the CPU verify function does not
    // have the minus sign. Passing in QUDA_DAG_YES solves this
    // discrepancy
    staggered_dslash(reinterpret_cast<cpuColorSpinorField *>(&ref->Even()), qdp_fatlink, qdp_longlink, ghost_fatlink,
                     ghost_longlink, reinterpret_cast<cpuColorSpinorField *>(&out->Odd()), QUDA_EVEN_PARITY,
                     QUDA_DAG_YES, inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);
    staggered_dslash(reinterpret_cast<cpuColorSpinorField *>(&ref->Odd()), qdp_fatlink, qdp_longlink, ghost_fatlink,
                     ghost_longlink, reinterpret_cast<cpuColorSpinorField *>(&out->Even()), QUDA_ODD_PARITY,
                     QUDA_DAG_YES, inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);

    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      xpay(out->V(), kappa, ref->V(), ref->Length(), gauge_param.cpu_prec);
      ax(0.5 / kappa, ref->V(), ref->Length(), gauge_param.cpu_prec);
    } else {
      axpy(2 * mass, out->V(), ref->V(), ref->Length(), gauge_param.cpu_prec);
    }

    // Reference debugging code: print the first component
    // of the even and odd partities within a solution vector.
    /*
    printfQuda("\nLength: %lu\n", ref->Length());

    // for verification
    printfQuda("\n\nEven:\n");
    printfQuda("CUDA: %f\n", ((double*)(in->Even().V()))[0]);
    printfQuda("Soln: %f\n", ((double*)(out->Even().V()))[0]);
    printfQuda("CPU:  %f\n", ((double*)(ref->Even().V()))[0]);

    printfQuda("\n\nOdd:\n");
    printfQuda("CUDA: %f\n", ((double*)(in->Odd().V()))[0]);
    printfQuda("Soln: %f\n", ((double*)(out->Odd().V()))[0]);
    printfQuda("CPU:  %f\n", ((double*)(ref->Odd().V()))[0]);
    printfQuda("\n\n");
    */

    mxpy(in->V(), ref->V(), len * mySpinorSiteSize, inv_param.cpu_prec);
    nrm2 = norm_2(ref->V(), len * mySpinorSiteSize, inv_param.cpu_prec);
    src2 = norm_2(in->V(), len * mySpinorSiteSize, inv_param.cpu_prec);

    break;

  case 3: // even
  case 4:

    invertQuda(out->V(), in->V(), &inv_param);

    time0 += clock();
    time0 /= CLOCKS_PER_SEC;

    matdagmat(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, out, mass, 0, inv_param.cpu_prec,
              gauge_param.cpu_prec, tmp, test_type == 3 ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY, dslash_type);

    if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
      printfQuda("%f %f\n", ((float *)in->V())[12], ((float *)ref->V())[12]);
    } else {
      printfQuda("%f %f\n", ((double *)in->V())[12], ((double *)ref->V())[12]);
    }

    mxpy(in->V(), ref->V(), len * mySpinorSiteSize, inv_param.cpu_prec);
    nrm2 = norm_2(ref->V(), len * mySpinorSiteSize, inv_param.cpu_prec);
    src2 = norm_2(in->V(), len * mySpinorSiteSize, inv_param.cpu_prec);

    break;

  case 5: // multi mass CG, even
  case 6:

#define NUM_OFFSETS 12

  {
        double masses[NUM_OFFSETS] ={0.06, 0.061, 0.064, 0.070, 0.077, 0.081, 0.1, 0.11, 0.12, 0.13, 0.14, 0.205};
        inv_param.num_offset = NUM_OFFSETS;
        // these can be set independently
        for (int i = 0; i < inv_param.num_offset; i++) {
          inv_param.tol_offset[i] = inv_param.tol;
          inv_param.tol_hq_offset[i] = inv_param.tol_hq;
        }
        void* outArray[NUM_OFFSETS];

        cpuColorSpinorField* spinorOutArray[NUM_OFFSETS];
        spinorOutArray[0] = out;
        for (int i = 1; i < inv_param.num_offset; i++) { spinorOutArray[i] = new cpuColorSpinorField(csParam); }

        for (int i = 0; i < inv_param.num_offset; i++) {
          outArray[i] = spinorOutArray[i]->V();
          inv_param.offset[i] = 4*masses[i]*masses[i];
        }

        invertMultiShiftQuda(outArray, in->V(), &inv_param);

        cudaDeviceSynchronize();
        time0 += clock(); // stop the timer
        time0 /= CLOCKS_PER_SEC;

        printfQuda("done: total time = %g secs, compute time = %g, %i iter / %g secs = %g gflops\n", time0,
            inv_param.secs, inv_param.iter, inv_param.secs, inv_param.gflops / inv_param.secs);

        printfQuda("checking the solution\n");
        QudaParity parity = QUDA_INVALID_PARITY;
        if (inv_param.solve_type == QUDA_NORMOP_SOLVE) {
          //parity = QUDA_EVENODD_PARITY;
          errorQuda("full parity not supported\n");
        } else if (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN) {
          parity = QUDA_EVEN_PARITY;
        } else if (inv_param.matpc_type == QUDA_MATPC_ODD_ODD) {
          parity = QUDA_ODD_PARITY;
        } else {
          errorQuda("ERROR: invalid spinor parity \n");
        }
        for(int i=0;i < inv_param.num_offset;i++){
          printfQuda("%dth solution: mass=%f, ", i, masses[i]);
          matdagmat(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, spinorOutArray[i], masses[i], 0,
                    inv_param.cpu_prec, gauge_param.cpu_prec, tmp, parity, dslash_type);

          mxpy(in->V(), ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
          double nrm2 = norm_2(ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
          double src2 = norm_2(in->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
          double hqr = sqrt(blas::HeavyQuarkResidualNorm(*spinorOutArray[i], *ref).z);
          double l2r = sqrt(nrm2/src2);

          printfQuda("Shift %d residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, "
                     "host = %g\n",
              i, inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r, inv_param.tol_hq_offset[i],
              inv_param.true_res_hq_offset[i], hqr);

          //emperical, if the cpu residue is more than 1 order the target accuracy, the it fails to converge
          if (sqrt(nrm2/src2) > 10*inv_param.tol_offset[i]){
            ret |=1;
          }
        }

        for(int i=1; i < inv_param.num_offset;i++) delete spinorOutArray[i];
  } break;

    default:
      errorQuda("Unsupported test type");

    } // switch

  if (test_type <=4){

    double hqr = sqrt(blas::HeavyQuarkResidualNorm(*out, *ref).z);
    double l2r = sqrt(nrm2/src2);

    printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, host = %g\n",
               inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq, hqr);

    printfQuda("done: total time = %g secs, compute time = %g secs, %i iter / %g secs = %g gflops, \n", time0,
        inv_param.secs, inv_param.iter, inv_param.secs, inv_param.gflops / inv_param.secs);
  }

  // Clean up gauge fields, at least
  for (int dir = 0; dir < 4; dir++) {
    if (qdp_inlink[dir] != nullptr) { free(qdp_inlink[dir]); qdp_inlink[dir] = nullptr; }
    if (qdp_fatlink[dir] != nullptr) { free(qdp_fatlink[dir]); qdp_fatlink[dir] = nullptr; }
    if (qdp_longlink[dir] != nullptr) { free(qdp_longlink[dir]); qdp_longlink[dir] = nullptr; }
  }
  if (milc_fatlink != nullptr) { free(milc_fatlink); milc_fatlink = nullptr; }
  if (milc_longlink != nullptr) { free(milc_longlink); milc_longlink = nullptr; }

#ifdef MULTI_GPU
  if (cpuFat != nullptr) { delete cpuFat; cpuFat = nullptr; }
  if (cpuLong != nullptr) { delete cpuLong; cpuLong = nullptr; }
#endif

  end();
  return ret;
}

static void end(void)
{
  delete in;
  delete out;
  delete ref;
  delete tmp;

  endQuda();
}

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon test_type  S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %s         %d/%d/%d          %d \n", get_prec_str(prec),
      get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy),
      get_staggered_test_type(test_type), xdim, ydim, zdim, tdim);

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
      dimPartitioned(3));

  return ;

}

//   void
// usage_extra(char** argv )
// {
//   printfQuda("Extra options:\n");
//   printfQuda("    --test <0/1/2/3/4/5/6>                      # Test method\n");
//   printfQuda("                                                0: Full parity inverter\n");
//   printfQuda("                                                1: Even even spinor CG inverter, reconstruct to full
//   parity\n"); printfQuda("                                                2: Odd odd spinor CG inverter, reconstruct
//   to full parity\n"); printfQuda("                                                3: Even even spinor CG
//   inverter\n"); printfQuda("                                                4: Odd odd spinor CG inverter\n"); printfQuda("
//   5: Even even spinor multishift CG inverter\n"); printfQuda("                                                6: Odd
//   odd spinor multishift CG inverter\n"); printfQuda("    --cpu-prec <double/single/half>             # Set CPU
//   precision\n");

//   return ;
// }
int main(int argc, char **argv)
{

  // Set a default
  solve_type = QUDA_INVALID_SOLVE;
  // command line options
  auto app = make_app();
  // app->get_formatter()->column_width(40);
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"full", 0}, {"full_ee_prec", 1}, {"full_oo_prec", 2}, {"even", 3},
                                          {"odd", 4},  {"mcg_even", 5},     {"mcg_odd", 6}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // for (int i = 1; i < argc; i++) {

  //   if (process_command_line_option(argc, argv, &i) == 0) { continue; }

  //   if (strcmp(argv[i], "--cpu-prec") == 0) {
  //     if (i+1 >= argc){
  //       usage(argv);
  //     }
  //     cpu_prec= get_prec(argv[i+1]);
  //     i++;
  //     continue;
  //   }

  //   printf("ERROR: Invalid option:%s\n", argv[i]);
  //   usage(argv);
  // }

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  if (test_type < 0 || test_type > 6) {
    errorQuda("Test type %d is outside the valid range.\n", test_type);
  }

  // Ensure a reasonable default
  // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH) {
    warningQuda("The dslash_type %d isn't staggered, asqtad, or laplace. Defaulting to asqtad.\n", dslash_type);
    dslash_type = QUDA_ASQTAD_DSLASH;
  }

  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    if (test_type != 0) {
      errorQuda("Test type %d is not supported for the Laplace operator.\n", test_type);
    }

    solve_type = QUDA_DIRECT_SOLVE;
    solution_type = QUDA_MAT_SOLUTION;
    matpc_type = QUDA_MATPC_EVEN_EVEN; // doesn't matter

  } else {

    if (test_type == 0 && (inv_type == QUDA_CG_INVERTER || inv_type == QUDA_PCG_INVERTER) &&
        solve_type != QUDA_NORMOP_SOLVE && solve_type != QUDA_DIRECT_PC_SOLVE) {
      warningQuda("The full spinor staggered operator (test 0) can't be inverted with (P)CG. Switching to BiCGstab.\n");
      inv_type = QUDA_BICGSTAB_INVERTER;
    }

    if (solve_type == QUDA_INVALID_SOLVE) {
      if (test_type == 0) {
        solve_type = QUDA_DIRECT_SOLVE;
      } else {
        solve_type = QUDA_DIRECT_PC_SOLVE;
      }
    }

    if (test_type == 1 || test_type == 3 || test_type == 5) {
      matpc_type = QUDA_MATPC_EVEN_EVEN;
    } else if (test_type == 2 || test_type == 4 || test_type == 6) {
      matpc_type = QUDA_MATPC_ODD_ODD;
    } else if (test_type == 0) {
      matpc_type = QUDA_MATPC_EVEN_EVEN; // it doesn't matter
    }

    if (test_type == 0 || test_type == 1 || test_type == 2) {
      solution_type = QUDA_MAT_SOLUTION;
    } else {
      solution_type = QUDA_MATPC_SOLUTION;
    }
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION){
    prec_sloppy = prec;
  }

  if (prec_refinement_sloppy == QUDA_INVALID_PRECISION){
    prec_refinement_sloppy = prec_sloppy;
  }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID){
    link_recon_sloppy = link_recon;
  }

  if(inv_type != QUDA_CG_INVERTER && (test_type == 5 || test_type == 6)) {
    errorQuda("Preconditioning is currently not supported in multi-shift solver solvers");
  }


  // Set n_naiks to 2 if eps_naik != 0.0
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    if (eps_naik != 0.0) {
      if (compute_fatlong) {
        n_naiks = 2;
        printfQuda("Note: epsilon-naik != 0, testing epsilon correction links.\n");
      } else {
        eps_naik = 0.0;
        printfQuda("Not computing fat-long, ignoring epsilon correction.\n");
      }
    } else {
      printfQuda("Note: epsilon-naik = 0, testing original HISQ links.\n");
    }
  }

  display_test_info();

  printfQuda("dslash_type = %d\n", dslash_type);

  argc_copy = argc;
  argv_copy = argv;

  int ret = invert_test();

  // finalize the communications layer
  finalizeComms();

  return ret;
}
