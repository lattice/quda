#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <staggered_dslash_reference.h>
#include <quda.h>
#include <string.h>
#include "misc.h"
#include <gauge_field.h>
#include <blas_quda.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#define MAX(a,b) ((a)>(b)?(a):(b))
#define my_spinor_site_size 6

void *qdp_fatlink[4];
void *qdp_longlink[4];

void *fatlink;
void *longlink;

#ifdef MULTI_GPU
void** ghost_fatlink, **ghost_longlink;
#endif


QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;

cpuColorSpinorField* in;
cpuColorSpinorField* out;
cpuColorSpinorField* ref;
cpuColorSpinorField* tmp;

cpuGaugeField *cpuFat = NULL;
cpuGaugeField *cpuLong = NULL;

static void end();

template<typename Float>
void constructSpinorField(Float *res) {
  for(int i = 0; i < Vh; i++) {
    for (int s = 0; s < 1; s++) {
      for (int m = 0; m < 3; m++) {
        res[i*(1*3*2) + s*(3*2) + m*(2) + 0] = rand() / (Float)RAND_MAX;
        res[i*(1*3*2) + s*(3*2) + m*(2) + 1] = rand() / (Float)RAND_MAX;
      }
    }
  }
}


static void
set_params(QudaGaugeParam* gaugeParam, QudaInvertParam* inv_param,
    int X1, int  X2, int X3, int X4,
    QudaPrecision cpu_prec, QudaPrecision prec, QudaPrecision prec_sloppy,
    QudaReconstructType link_recon, QudaReconstructType link_recon_sloppy,
    double mass, double tol, int maxiter, double reliable_delta,
    double tadpole_coeff
    )
{
  gaugeParam->X[0] = X1;
  gaugeParam->X[1] = X2;
  gaugeParam->X[2] = X3;
  gaugeParam->X[3] = X4;

  gaugeParam->cpu_prec = cpu_prec;
  gaugeParam->cuda_prec = prec;
  gaugeParam->reconstruct = link_recon;
  gaugeParam->cuda_prec_sloppy = prec_sloppy;
  gaugeParam->reconstruct_sloppy = link_recon_sloppy;
  gaugeParam->gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam->anisotropy = 1.0;
  gaugeParam->tadpole_coeff = tadpole_coeff;

  if (dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_STAGGERED_DSLASH)
    dslash_type = QUDA_ASQTAD_DSLASH;

  gaugeParam->scale = dslash_type == QUDA_STAGGERED_DSLASH ? 1.0 : -1.0/(24.0*tadpole_coeff*tadpole_coeff);

  gaugeParam->t_boundary = QUDA_ANTI_PERIODIC_T;
  gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER;
  gaugeParam->ga_pad = X1*X2*X3/2;

  inv_param->verbosity = QUDA_VERBOSE;
  inv_param->mass = mass;

  // outer solver parameters
  inv_param->inv_type = inv_type;
  inv_param->tol = tol;
  inv_param->tol_restart = 1e-3; //now theoretical background for this parameter...
  inv_param->maxiter = niter;
  inv_param->reliable_delta = 0;//1e-1;
  inv_param->use_sloppy_partial_accumulator = false;
  inv_param->pipeline = false;

  inv_param->Ls = 1;


  if(tol_hq == 0 && tol == 0){
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }
  // require both L2 relative and heavy quark residual to determine convergence
  inv_param->residual_type = static_cast<QudaResidualType_s>(0);
  inv_param->residual_type = (tol != 0) ? static_cast<QudaResidualType_s> ( inv_param->residual_type | QUDA_L2_RELATIVE_RESIDUAL) : inv_param->residual_type;
  inv_param->residual_type = (tol_hq != 0) ? static_cast<QudaResidualType_s> (inv_param->residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : inv_param->residual_type;

  inv_param->tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  inv_param->Nsteps = 2;


  //inv_param->inv_type = QUDA_GCR_INVERTER;
  //inv_param->gcrNkrylov = 10;

  // domain decomposition preconditioner parameters
  inv_param->inv_type_precondition = QUDA_SD_INVERTER;
  inv_param->tol_precondition = 1e-1;
  inv_param->maxiter_precondition = 10;
  inv_param->verbosity_precondition = QUDA_SILENT;
  inv_param->cuda_prec_precondition = QUDA_HALF_PRECISION;

  inv_param->solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  inv_param->solve_type = QUDA_NORMOP_PC_SOLVE;
  inv_param->matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param->dagger = QUDA_DAG_NO;
  inv_param->mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param->cpu_prec = cpu_prec;
  inv_param->cuda_prec = prec;
  inv_param->cuda_prec_sloppy = prec_sloppy;
  inv_param->preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // this is meaningless, but must be thus set
  inv_param->dirac_order = QUDA_DIRAC_ORDER;

  inv_param->dslash_type = dslash_type;

  inv_param->sp_pad = X1*X2*X3/2;
  inv_param->use_init_guess = QUDA_USE_INIT_GUESS_YES;

  inv_param->input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param->output_location = QUDA_CPU_FIELD_LOCATION;
}


  int
invert_test(void)
{
  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();

  set_params(&gaugeParam, &inv_param,
      xdim, ydim, zdim, tdim,
      cpu_prec, prec, prec_sloppy,
      link_recon, link_recon_sloppy, mass, tol, 500, 1e-3,
      0.8);

  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gaugeParam.X);
  dw_setDims(gaugeParam.X,1); // so we can use 5-d indexing from dwf
  setSpinorSiteSize(6);

  for (int dir = 0; dir < 4; dir++) {
    qdp_fatlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_longlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  }
  fatlink = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
  longlink = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

  construct_fat_long_gauge_field(qdp_fatlink, qdp_longlink, 1, gaugeParam.cpu_prec,
				 &gaugeParam, dslash_type);

  for(int dir=0; dir<4; ++dir){
    for(int i=0; i<V; ++i){
      for (int j = 0; j < gauge_site_size; ++j) {
        if(gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION){
          ((double *)fatlink)[(i * 4 + dir) * gauge_site_size + j]
            = ((double *)qdp_fatlink[dir])[i * gauge_site_size + j];
          ((double *)longlink)[(i * 4 + dir) * gauge_site_size + j]
            = ((double *)qdp_longlink[dir])[i * gauge_site_size + j];
        }else{
          ((float *)fatlink)[(i * 4 + dir) * gauge_site_size + j] = ((float *)qdp_fatlink[dir])[i * gauge_site_size + j];
          ((float *)longlink)[(i * 4 + dir) * gauge_site_size + j]
            = ((float *)qdp_longlink[dir])[i * gauge_site_size + j];
        }
      }
    }
  }


  ColorSpinorParam csParam;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=5;
  for (int d = 0; d < 4; d++) csParam.x[d] = gaugeParam.X[d];
  csParam.x[0] /= 2;
  csParam.x[4] = 1;

  csParam.setPrecision(inv_param.cpu_prec);
  csParam.pad = 0;
  csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder  = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  in = new cpuColorSpinorField(csParam);
  out = new cpuColorSpinorField(csParam);
  ref = new cpuColorSpinorField(csParam);
  tmp = new cpuColorSpinorField(csParam);

  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION){
    constructSpinorField((float*)in->V());
  }else{
    constructSpinorField((double*)in->V());
  }

#ifdef MULTI_GPU
  int tmp_value = MAX(ydim*zdim*tdim/2, xdim*zdim*tdim/2);
  tmp_value = MAX(tmp_value, xdim*ydim*tdim/2);
  tmp_value = MAX(tmp_value, xdim*ydim*zdim/2);

  int fat_pad = tmp_value;
  int link_pad =  3*tmp_value;

  // FIXME: currently assume staggered is SU(3)
  gaugeParam.type = dslash_type == QUDA_STAGGERED_DSLASH ?
    QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuFatParam(fatlink, gaugeParam);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink = (void**)cpuFat->Ghost();

  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(longlink, gaugeParam);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink = (void**)cpuLong->Ghost();


#else
  int fat_pad = 0;
  int link_pad = 0;
#endif

  gaugeParam.type = dslash_type == QUDA_STAGGERED_DSLASH ?
    QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.ga_pad = fat_pad;
  if (dslash_type == QUDA_STAGGERED_DSLASH) {
    gaugeParam.reconstruct = link_recon;
    gaugeParam.reconstruct_sloppy = link_recon_sloppy;
  } else {
    gaugeParam.reconstruct= gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  }
  gaugeParam.cuda_prec_precondition = QUDA_HALF_PRECISION;
  loadGaugeQuda(fatlink, &gaugeParam);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
    gaugeParam.ga_pad = link_pad;
    gaugeParam.reconstruct= link_recon;
    gaugeParam.reconstruct_sloppy = link_recon_sloppy;
    loadGaugeQuda(longlink, &gaugeParam);
  }

  double time0 = -((double)clock()); // Start the timer

  double nrm2=0;
  double src2=0;
  int ret = 0;



  switch(test_type){
    case 0: //even
      if(inv_type == QUDA_GCR_INVERTER){
      	inv_param.inv_type = QUDA_GCR_INVERTER;
      	inv_param.gcrNkrylov = 50;
      }else if(inv_type == QUDA_PCG_INVERTER){
	inv_param.inv_type = QUDA_PCG_INVERTER;
      }
      inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
      #define NUM_SRC 20
      inv_param.num_src=Nsrc; // number of spinors to apply to simultaneously
      void* outArray[NUM_SRC];
      void* inArray[NUM_SRC];
      // int len;

      cpuColorSpinorField* spinorOutArray[NUM_SRC];
      cpuColorSpinorField* spinorInArray[NUM_SRC];
      spinorOutArray[0] = out;
      spinorInArray[0] = in;
      // in = new cpuColorSpinorField(csParam);
      // out = new cpuColorSpinorField(csParam);
      // ref = new cpuColorSpinorField(csParam);
      // tmp = new cpuColorSpinorField(csParam);

      if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION){
        constructSpinorField((float*)in->V());
      }else{
        constructSpinorField((double*)in->V());
      }

      for(int i=1;i < inv_param.num_src; i++){
        spinorOutArray[i] = new cpuColorSpinorField(csParam);
        spinorInArray[i] = new cpuColorSpinorField(csParam);
        if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION){
          constructSpinorField((float*)spinorInArray[i]->V());
        }else{
          constructSpinorField((double*)spinorInArray[i]->V());
        }
      }

      for(int i=0;i < inv_param.num_src; i++){
        outArray[i] = spinorOutArray[i]->V();
        inArray[i] = spinorInArray[i]->V();
        // inv_param.offset[i] = 4*masses[i]*masses[i];
      }
      invertMultiSrcQuda(outArray, inArray, &inv_param);

      time0 += clock();
      time0 /= CLOCKS_PER_SEC;



#ifdef MULTI_GPU
      // matdagmat_mg4dir(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink,
      // out, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_EVEN_PARITY);
#else
      matdagmat(ref->V(), qdp_fatlink, qdp_longlink, out->V(), mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp->V(), QUDA_EVEN_PARITY);
#endif

      mxpy(in->V(), ref->V(), Vh * my_spinor_site_size, inv_param.cpu_prec);
      nrm2 = norm_2(ref->V(), Vh * my_spinor_site_size, inv_param.cpu_prec);
      src2 = norm_2(in->V(), Vh * my_spinor_site_size, inv_param.cpu_prec);

      for(int i=1; i < inv_param.num_src;i++) delete spinorOutArray[i];
      for(int i=1; i < inv_param.num_src;i++) delete spinorInArray[i];


      break;

    case 1: //odd
      if(inv_type == QUDA_GCR_INVERTER){
      	inv_param.inv_type = QUDA_GCR_INVERTER;
      	inv_param.gcrNkrylov = 50;
      }else if(inv_type == QUDA_PCG_INVERTER){
	inv_param.inv_type = QUDA_PCG_INVERTER;
      }

      inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
      invertQuda(out->V(), in->V(), &inv_param);
      time0 += clock(); // stop the timer
      time0 /= CLOCKS_PER_SEC;

#ifdef MULTI_GPU
      // matdagmat_mg4dir(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink,
      // out, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_ODD_PARITY);
#else
      matdagmat(ref->V(), qdp_fatlink, qdp_longlink, out->V(), mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp->V(), QUDA_ODD_PARITY);
#endif
      mxpy(in->V(), ref->V(), Vh * my_spinor_site_size, inv_param.cpu_prec);
      nrm2 = norm_2(ref->V(), Vh * my_spinor_site_size, inv_param.cpu_prec);
      src2 = norm_2(in->V(), Vh * my_spinor_site_size, inv_param.cpu_prec);

      break;

    case 2: //full spinor

      errorQuda("full spinor not supported\n");
      break;

    case 3: //multi mass CG, even
    case 4:

#define NUM_OFFSETS 12

      {
        double masses[NUM_OFFSETS] ={0.06, 0.061, 0.064, 0.070, 0.077, 0.081, 0.1, 0.11, 0.12, 0.13, 0.14, 0.205};
        inv_param.num_offset = NUM_OFFSETS;
        // these can be set independently
        for (int i=0; i<inv_param.num_offset; i++) {
          inv_param.tol_offset[i] = inv_param.tol;
          inv_param.tol_hq_offset[i] = inv_param.tol_hq;
        }
        void* outArray[NUM_OFFSETS];
        int len;

        cpuColorSpinorField* spinorOutArray[NUM_OFFSETS];
        spinorOutArray[0] = out;
        for(int i=1;i < inv_param.num_offset; i++){
          spinorOutArray[i] = new cpuColorSpinorField(csParam);
        }

        for(int i=0;i < inv_param.num_offset; i++){
          outArray[i] = spinorOutArray[i]->V();
          inv_param.offset[i] = 4*masses[i]*masses[i];
        }

        len=Vh;

        if (test_type == 3) {
          inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
        } else {
          inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
        }

        invertMultiShiftQuda(outArray, in->V(), &inv_param);

        cudaDeviceSynchronize();
        time0 += clock(); // stop the timer
        time0 /= CLOCKS_PER_SEC;

        printfQuda("done: total time = %g secs, compute time = %g, %i iter / %g secs = %g gflops\n",
            time0, inv_param.secs, inv_param.iter, inv_param.secs,
            inv_param.gflops/inv_param.secs);


        printfQuda("checking the solution\n");
        QudaParity parity = QUDA_INVALID_PARITY;
        if (inv_param.solve_type == QUDA_NORMOP_SOLVE){
          //parity = QUDA_EVENODD_PARITY;
          errorQuda("full parity not supported\n");
        }else if (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN){
          parity = QUDA_EVEN_PARITY;
        }else if (inv_param.matpc_type == QUDA_MATPC_ODD_ODD){
          parity = QUDA_ODD_PARITY;
        }else{
          errorQuda("ERROR: invalid spinor parity \n");
          exit(1);
        }
        for(int i=0;i < inv_param.num_offset;i++){
          printfQuda("%dth solution: mass=%f, ", i, masses[i]);
#ifdef MULTI_GPU
          // matdagmat_mg4dir(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink,
          //     spinorOutArray[i], masses[i], 0, inv_param.cpu_prec,
          //     gaugeParam.cpu_prec, tmp, parity);
#else
          matdagmat(ref->V(), qdp_fatlink, qdp_longlink, outArray[i], masses[i], 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp->V(), parity);
#endif

          mxpy(in->V(), ref->V(), len * my_spinor_site_size, inv_param.cpu_prec);
          double nrm2 = norm_2(ref->V(), len * my_spinor_site_size, inv_param.cpu_prec);
          double src2 = norm_2(in->V(), len * my_spinor_site_size, inv_param.cpu_prec);
          double hqr = sqrt(blas::HeavyQuarkResidualNorm(*spinorOutArray[i], *ref).z);
          double l2r = sqrt(nrm2 / src2);

          printfQuda("Shift %d residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, "
                     "host = %g\n",
                     i, inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r, inv_param.tol_hq_offset[i],
                     inv_param.true_res_hq_offset[i], hqr);

          // emperical, if the cpu residue is more than 1 order the target accuracy, the it fails to converge
          if (sqrt(nrm2 / src2) > 10 * inv_param.tol_offset[i]) { ret |= 1; }
        }

        for(int i=1; i < inv_param.num_offset;i++) delete spinorOutArray[i];
      }
      break;

    default:
      errorQuda("Unsupported test type");

  }//switch

  if (test_type <=2){

    double hqr = sqrt(blas::HeavyQuarkResidualNorm(*out, *ref).z);
    double l2r = sqrt(nrm2/src2);

    printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, host = %g\n",
        inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq, hqr);

    printfQuda("done: total time = %g secs, compute time = %g secs, %i iter / %g secs = %g gflops, \n",
        time0, inv_param.secs, inv_param.iter, inv_param.secs,
        inv_param.gflops/inv_param.secs);
  }

  end();
  return ret;
}



  static void
end(void)
{
  for(int i=0;i < 4;i++){
    free(qdp_fatlink[i]);
    free(qdp_longlink[i]);
  }

  free(fatlink);
  free(longlink);

  delete in;
  delete out;
  delete ref;
  delete tmp;

  if (cpuFat) delete cpuFat;
  if (cpuLong) delete cpuLong;

  endQuda();
}


  void
display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon test_type  S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %s         %d/%d/%d          %d \n",
      get_prec_str(prec),get_prec_str(prec_sloppy),
      get_recon_str(link_recon),
      get_recon_str(link_recon_sloppy), get_test_type(test_type), xdim, ydim, zdim, tdim);

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n",
      dimPartitioned(0),
      dimPartitioned(1),
      dimPartitioned(2),
      dimPartitioned(3));

  return ;

}

int main(int argc, char** argv)
{
  // command line options
  auto app = make_app();
  CLI::TransformPairs<int> test_type_map {{"full", 0}, {"full_ee_prec", 1}, {"full_oo_prec", 2}, {"even", 3}, {"odd", 4}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
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

  if(inv_type != QUDA_CG_INVERTER){
    if(test_type != 0 && test_type != 1) errorQuda("Preconditioning is currently not supported in multi-shift solver solvers");
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  printfQuda("dslash_type = %d\n", dslash_type);

  int ret = invert_test();

  // finalize the communications layer
  finalizeComms();

  return ret;
}
