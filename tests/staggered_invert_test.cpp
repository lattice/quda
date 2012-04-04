#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <test_util.h>
#include <blas_reference.h>
#include <staggered_dslash_reference.h>
#include <quda.h>
#include <string.h>
#include <face_quda.h>
#include "misc.h"
#include <gauge_field.h>

#ifdef MULTI_GPU
#include <face_quda.h>
#endif

#define MAX(a,b) ((a)>(b)?(a):(b))
#define mySpinorSiteSize 6

extern void usage(char** argv);
void *fatlink[4];
void *longlink[4];  

#ifdef MULTI_GPU
void** ghost_fatlink, **ghost_longlink;
#endif

extern int device;
extern bool tune;

extern QudaReconstructType link_recon;
extern QudaPrecision prec;
QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;

extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision  prec_sloppy;
cpuColorSpinorField* in;
cpuColorSpinorField* out;
cpuColorSpinorField* ref;
cpuColorSpinorField* tmp;

cpuGaugeField *cpuFat = NULL;
cpuGaugeField *cpuLong = NULL;

static double tol = 1e-6;

static int testtype = 0;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern bool kernelPackT;
extern int gridsize_from_cmdline[];


static void end();

extern int Z[4];
extern int V;
extern int Vh;
static int Vs_x, Vs_y, Vs_z, Vs_t;
extern int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
static int Vsh[4];


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

void
setDimConstants(int *X)
{
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];
  }
  Vh = V/2;

  Vs_x = X[1]*X[2]*X[3];
  Vs_y = X[0]*X[2]*X[3];
  Vs_z = X[0]*X[1]*X[3];
  Vs_t = X[0]*X[1]*X[2];


  Vsh_x = Vs_x/2;
  Vsh_y = Vs_y/2;
  Vsh_z = Vs_z/2;
  Vsh_t = Vs_t/2;

  Vsh[0] = Vsh_x;
  Vsh[1] = Vsh_y;
  Vsh[2] = Vsh_z;
  Vsh[3] = Vsh_t;
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
  gaugeParam->t_boundary = QUDA_ANTI_PERIODIC_T;
  gaugeParam->gauge_order = QUDA_QDP_GAUGE_ORDER;
  gaugeParam->ga_pad = X1*X2*X3/2;

  inv_param->verbosity = QUDA_VERBOSE;
  inv_param->mass = mass;

  // outer solver parameters
  inv_param->inv_type = QUDA_CG_INVERTER;
  inv_param->tol = tol;
  inv_param->maxiter = 10000;
  inv_param->reliable_delta = 1e-1; // ignored by multi-shift solver

  //inv_param->inv_type = QUDA_GCR_INVERTER;
  //inv_param->gcrNkrylov = 10;

  // domain decomposition preconditioner parameters
  //inv_param->inv_type_precondition = QUDA_MR_INVERTER;
  //inv_param->tol_precondition = 1e-1;
  //inv_param->maxiter_precondition = 100;
  //inv_param->verbosity_precondition = QUDA_SILENT;
  //inv_param->prec_precondition = prec_sloppy;

  inv_param->solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  inv_param->solve_type = QUDA_NORMEQ_PC_SOLVE;
  inv_param->matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param->dagger = QUDA_DAG_NO;
  inv_param->mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param->cpu_prec = cpu_prec;
  inv_param->cuda_prec = prec; 
  inv_param->cuda_prec_sloppy = prec_sloppy;
  inv_param->preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // this is meaningless, but must be thus set
  inv_param->dirac_order = QUDA_DIRAC_ORDER;
  inv_param->dslash_type = QUDA_ASQTAD_DSLASH;
  inv_param->tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;
  inv_param->sp_pad = X1*X2*X3/2;
  inv_param->use_init_guess = QUDA_USE_INIT_GUESS_YES;

}

int
invert_test(void)
{
  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();

  double mass = 0.1;

  set_params(&gaugeParam, &inv_param,
	     xdim, ydim, zdim, tdim,
	     cpu_prec, prec, prec_sloppy,
	     link_recon, link_recon_sloppy, mass, tol, 500, 1e-3,
	     0.8);
  
  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gaugeParam.X);
  setDimConstants(gaugeParam.X);

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  for (int dir = 0; dir < 4; dir++) {
    fatlink[dir] = malloc(V*gaugeSiteSize*gSize);
    longlink[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  
  construct_fat_long_gauge_field(fatlink, longlink, 1, gaugeParam.cpu_prec, &gaugeParam);
    
  for (int dir = 0; dir < 4; dir++) {
    for(int i = 0;i < V*gaugeSiteSize;i++){
      if (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION){
	((double*)fatlink[dir])[i] = 0.5 *rand()/RAND_MAX;
      }else{
	((float*)fatlink[dir])[i] = 0.5* rand()/RAND_MAX;
      }
    }
  }  
 
  ColorSpinorParam csParam;
  csParam.fieldLocation = QUDA_CPU_FIELD_LOCATION;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=4;
  for(int d = 0; d < 4; d++) {
    csParam.x[d] = gaugeParam.X[d];
  }
  csParam.x[0] /= 2;
  
  csParam.precision = inv_param.cpu_prec;
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

  int tmp_value = MAX(ydim*zdim*tdim/2, xdim*zdim*tdim/2);
   tmp_value = MAX(tmp_value, xdim*ydim*tdim/2);
   tmp_value = MAX(tmp_value, xdim*ydim*zdim/2);

  int fat_pad = tmp_value;
  int link_pad =  3*tmp_value;

#ifdef MULTI_GPU
  gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuFatParam(fatlink, gaugeParam);
  cpuFat = new cpuGaugeField(cpuFatParam);
  cpuFat->exchangeGhost();
  ghost_fatlink = (void**)cpuFat->Ghost();
  
  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(longlink, gaugeParam);
  cpuLong = new cpuGaugeField(cpuLongParam);
  cpuLong->exchangeGhost();
  ghost_longlink = (void**)cpuLong->Ghost();
#endif
  
  if(testtype == 6){    
    record_gauge(gaugeParam.X, fatlink, fat_pad,
		 longlink, link_pad,
		 link_recon, link_recon_sloppy,
		 &gaugeParam);        
   }else{ 
    
#ifdef MULTI_GPU
 

    gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
    gaugeParam.ga_pad = fat_pad;
    gaugeParam.reconstruct= gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(fatlink, &gaugeParam);
    
    gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
    gaugeParam.ga_pad = link_pad;
    gaugeParam.reconstruct= link_recon;
    gaugeParam.reconstruct_sloppy = link_recon_sloppy;
    loadGaugeQuda(longlink, &gaugeParam);
#else
    gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(fatlink, &gaugeParam);
    
    gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
    gaugeParam.reconstruct = link_recon;
    gaugeParam.reconstruct_sloppy = link_recon_sloppy;
    loadGaugeQuda(longlink, &gaugeParam);
#endif
  }

  
  double time0 = -((double)clock()); // Start the timer
  
  double nrm2=0;
  double src2=0;
  int ret = 0;

  switch(testtype){
  case 0: //even
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    
    invertQuda(out->V(), in->V(), &inv_param);
    
    time0 += clock(); 
    time0 /= CLOCKS_PER_SEC;

#ifdef MULTI_GPU    
    matdagmat_mg4dir(ref, fatlink, longlink, ghost_fatlink, ghost_longlink, 
		     out, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_EVEN_PARITY);
#else
    matdagmat(ref->V(), fatlink, longlink, out->V(), mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp->V(), QUDA_EVEN_PARITY);
#endif
    
    mxpy(in->V(), ref->V(), Vh*mySpinorSiteSize, inv_param.cpu_prec);
    nrm2 = norm_2(ref->V(), Vh*mySpinorSiteSize, inv_param.cpu_prec);
    src2 = norm_2(in->V(), Vh*mySpinorSiteSize, inv_param.cpu_prec);

    break;

  case 1: //odd
	
    inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
    invertQuda(out->V(), in->V(), &inv_param);	
    time0 += clock(); // stop the timer
    time0 /= CLOCKS_PER_SEC;
    
#ifdef MULTI_GPU
    matdagmat_mg4dir(ref, fatlink, longlink, ghost_fatlink, ghost_longlink, 
		     out, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_ODD_PARITY);
#else
    matdagmat(ref->V(), fatlink, longlink, out->V(), mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp->V(), QUDA_ODD_PARITY);	
#endif
    mxpy(in->V(), ref->V(), Vh*mySpinorSiteSize, inv_param.cpu_prec);
    nrm2 = norm_2(ref->V(), Vh*mySpinorSiteSize, inv_param.cpu_prec);
    src2 = norm_2(in->V(), Vh*mySpinorSiteSize, inv_param.cpu_prec);
	
    break;
    
  case 2: //full spinor

    errorQuda("full spinor not supported\n");
    break;
    
  case 3: //multi mass CG, even
  case 4:
  case 5:
  case 6:

#define NUM_OFFSETS 7
        
    double masses[NUM_OFFSETS] ={5.05, 1.23, 2.64, 2.33, 2.70, 2.77, 2.81};
    double offsets[NUM_OFFSETS];	
    int num_offsets =NUM_OFFSETS;
    void* outArray[NUM_OFFSETS];
    int len;
    
    cpuColorSpinorField* spinorOutArray[NUM_OFFSETS];
    spinorOutArray[0] = out;    
    for(int i=1;i < num_offsets; i++){
      spinorOutArray[i] = new cpuColorSpinorField(csParam);       
    }
    
    for(int i=0;i < num_offsets; i++){
      outArray[i] = spinorOutArray[i]->V();
    }

    for (int i=0; i< num_offsets;i++){
      offsets[i] = 4*masses[i]*masses[i];
    }
    
    len=Vh;

    if (testtype == 3 || testtype == 6){
      inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;      
    } else if (testtype == 4){
      inv_param.matpc_type = QUDA_MATPC_ODD_ODD;      
    }else { //testtype ==5
      errorQuda("test 5 not supported\n");
    }
    
    double residue_sq;
    if (testtype == 6){
      invertMultiShiftQudaMixed(outArray, in->V(), &inv_param, offsets, num_offsets, &residue_sq);
    }else{      
      invertMultiShiftQuda(outArray, in->V(), &inv_param, offsets, num_offsets, &residue_sq);	
    }
    cudaThreadSynchronize();
    printfQuda("Final residue squred =%g\n", residue_sq);
    time0 += clock(); // stop the timer
    time0 /= CLOCKS_PER_SEC;
    
    printfQuda("done: total time = %g secs, %i iter / %g secs = %g gflops, \n", 
	       time0, inv_param.iter, inv_param.secs,
	       inv_param.gflops/inv_param.secs);
    
    
    printfQuda("checking the solution\n");
    QudaParity parity = QUDA_INVALID_PARITY;
    if (inv_param.solve_type == QUDA_NORMEQ_SOLVE){
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
    for(int i=0;i < num_offsets;i++){
      printfQuda("%dth solution: mass=%f, ", i, masses[i]);
#ifdef MULTI_GPU
      matdagmat_mg4dir(ref, fatlink, longlink, ghost_fatlink, ghost_longlink, 
		       spinorOutArray[i], masses[i], 0, inv_param.cpu_prec, 
		       gaugeParam.cpu_prec, tmp, parity);
#else
      matdagmat(ref->V(), fatlink, longlink, outArray[i], masses[i], 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp->V(), parity);
#endif
      mxpy(in->V(), ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
      double nrm2 = norm_2(ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
      double src2 = norm_2(in->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
      
      printfQuda("relative residual, requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

      //emperical, if the cpu residue is more than 1 order the target accuracy, the it fails to converge
      if (sqrt(nrm2/src2) > 10*inv_param.tol){
	ret |=1;
      }
    }

    if (ret ==1){
      errorQuda("Converge failed!\n");
    }

    for(int i=1; i < num_offsets;i++){
      delete spinorOutArray[i];
    }

    
  }//switch
    

  if (testtype <=2){

    printfQuda("Relative residual, requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));
	
    printfQuda("done: total time = %g secs, %i iter / %g secs = %g gflops, \n", 
	       time0, inv_param.iter, inv_param.secs,
	       inv_param.gflops/inv_param.secs);
    
    //emperical, if the cpu residue is more than 2 order the target accuracy, the it fails to converge
    if (sqrt(nrm2/src2) > 100*inv_param.tol){
      ret = 1;
      errorQuda("Convergence failed!\n");
    }
  }

  end();
  return ret;
}



static void
end(void) 
{
  for(int i=0;i < 4;i++){
    free(fatlink[i]);
    free(longlink[i]);
  }

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
	     get_recon_str(link_recon_sloppy), get_test_type(testtype), xdim, ydim, zdim, tdim);     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     commDimPartitioned(0),
	     commDimPartitioned(1),
	     commDimPartitioned(2),
	     commDimPartitioned(3)); 
  
  return ;
  
}

void
usage_extra(char** argv )
{
  printfQuda("Extra options:\n");
  printfQuda("    --tol  <resid_tol>                       # Set residual tolerance\n");
  printfQuda("    --test <0/1>                             # Test method\n");
  printfQuda("                                                0: Even even spinor CG inverter\n");
  printfQuda("                                                1: Odd odd spinor CG inverter\n");
  printfQuda("                                                3: Even even spinor multishift CG inverter\n");
  printfQuda("                                                4: Odd odd spinor multishift CG inverter\n");
  printfQuda("                                                6: Even even spinor mixed precision multishift CG inverter\n");
  printfQuda("    --cpu_prec <double/single/half>          # Set CPU precision\n");
  
  return ;
}
int main(int argc, char** argv)
{

  int i;
  for (i =1;i < argc; i++){

    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }   
    

    if( strcmp(argv[i], "--tol") == 0){
      float tmpf;
      if (i+1 >= argc){
        usage(argv);
      }
      sscanf(argv[i+1], "%f", &tmpf);
      if (tmpf <= 0){
        printf("ERROR: invalid tol(%f)\n", tmpf);
        usage(argv);
      }
      tol = tmpf;
      i++;
      continue;
    }
    
    
    if( strcmp(argv[i], "--test") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      testtype = atoi(argv[i+1]);
      i++;
      continue;	    
    }
    
    if( strcmp(argv[i], "--cpu_prec") == 0){
      if (i+1 >= argc){
	usage(argv);
      }
      cpu_prec= get_prec(argv[i+1]);
      i++;
      continue;
    }
   
        
    printf("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }


  if (prec_sloppy == QUDA_INVALID_PRECISION){
    prec_sloppy = prec;
  }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID){
    link_recon_sloppy = link_recon;
  }
  

  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);
  display_test_info();
  
  int ret = invert_test();

  endCommsQuda();

  return ret;
}
