#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <test_util.h>
#include <blas_reference.h>
#include <staggered_dslash_reference.h>
#include <quda.h>
#include <string.h>
#include "misc.h"
#include "gauge_quda.h"

#ifdef MULTI_GPU
#include "exchange_face.h"
#include "mpicomm.h"
#include <mpi.h>
#endif


#define mySpinorSiteSize 6

void *fatlink[4];
void *longlink[4];  

#ifdef MULTI_GPU
void *cpu_fwd_nbr_spinor, *cpu_back_nbr_spinor;
void* ghost_fatlink, *ghost_longlink;
#endif

void *spinorIn;
void *spinorOut;
void *spinorCheck;
void *tmp;

int device = 0;

extern FullGauge cudaFatLinkPrecise;
extern FullGauge cudaFatLinkSloppy;
extern FullGauge cudaLongLinkPrecise;
extern FullGauge cudaLongLinkSloppy;

QudaReconstructType link_recon = QUDA_RECONSTRUCT_12;
QudaPrecision prec = QUDA_SINGLE_PRECISION;
QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;

QudaReconstructType link_recon_sloppy = QUDA_RECONSTRUCT_INVALID;
QudaPrecision  prec_sloppy = QUDA_INVALID_PRECISION;

static double tol = 1e-8;

static int testtype = 0;
static int sdim = 24;
static int tdim = 24;

extern int V;
static void end();


template<typename Float>
void constructSpinorField(Float *res) {
  for(int i = 0; i < V; i++) {
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
  gaugeParam->tadpole_coeff = tadpole_coeff;
  gaugeParam->t_boundary = QUDA_ANTI_PERIODIC_T;
  gaugeParam->gauge_order = QUDA_QDP_GAUGE_ORDER;
  gaugeParam->ga_pad = X1*X2*X3/2;

  inv_param->verbosity = QUDA_DEBUG_VERBOSE;
  inv_param->inv_type = QUDA_CG_INVERTER;    
  inv_param->mass = mass;
  inv_param->tol = tol;
  inv_param->maxiter = 500;
  inv_param->reliable_delta = 1e-3;

  inv_param->solution_type = QUDA_MATDAG_MAT_SOLUTION;
  inv_param->solve_type = QUDA_NORMEQ_PC_SOLVE;
  inv_param->matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param->dagger = QUDA_DAG_NO;
  inv_param->mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param->cpu_prec = cpu_prec;
  inv_param->cuda_prec = prec; 
  inv_param->cuda_prec_sloppy = prec_sloppy;
  inv_param->preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param->dirac_order = QUDA_DIRAC_ORDER;
  inv_param->dslash_type = QUDA_ASQTAD_DSLASH;
  inv_param->dirac_tune = QUDA_TUNE_NO;
  inv_param->preserve_dirac = QUDA_PRESERVE_DIRAC_NO;
  inv_param->sp_pad = X1*X2*X3/2;
  inv_param->use_init_guess = QUDA_USE_INIT_GUESS_YES;
    
}

static int
invert_test(void)
{
  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();

  int Vs = sdim*sdim*sdim;
  int Vsh = Vs/2;  
  
  double mass = 0.95;

  set_params(&gaugeParam, &inv_param,
	     sdim, sdim, sdim, tdim,
	     cpu_prec, prec, prec_sloppy,
	     link_recon, link_recon_sloppy, mass, tol, 500, 1e-3,
	     0.8);
  

  setDims(gaugeParam.X);

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  
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
   
#ifdef MULTI_GPU
  ghost_fatlink = malloc(Vs*gaugeSiteSize*gSize);
  ghost_longlink = malloc(3*Vs*gaugeSiteSize*gSize);
  if (ghost_fatlink == NULL || ghost_longlink == NULL){
    errorQuda("ERROR: malloc failed for ghost fatlink/longlink\n");
  }
  exchange_cpu_links(gaugeParam.X, fatlink, ghost_fatlink, longlink, ghost_longlink, gaugeParam.cpu_prec);
#endif

 
  spinorIn = malloc(V*mySpinorSiteSize*sSize);
  spinorOut = malloc(V*mySpinorSiteSize*sSize);
  spinorCheck = malloc(V*mySpinorSiteSize*sSize);
  tmp = malloc(V*mySpinorSiteSize*sSize);
   
  memset(spinorIn, 0, V*mySpinorSiteSize*sSize);
  memset(spinorOut, 0, V*mySpinorSiteSize*sSize);
  memset(spinorCheck, 0, V*mySpinorSiteSize*sSize);
  memset(tmp, 0, V*mySpinorSiteSize*sSize);

  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION){
    constructSpinorField((float*)spinorIn);    
  }else{
    constructSpinorField((double*)spinorIn);
  }
  
  void* spinorInOdd = ((char*)spinorIn) + Vh*mySpinorSiteSize*sSize;
  void* spinorOutOdd = ((char*)spinorOut) + Vh*mySpinorSiteSize*sSize;
  void* spinorCheckOdd = ((char*)spinorCheck) + Vh*mySpinorSiteSize*sSize;
  
  initQuda(device);
  
  
#ifdef MULTI_GPU
  //create ghost spinors
  cpu_fwd_nbr_spinor = malloc(Vsh* mySpinorSiteSize *3*sizeof(double));
  cpu_back_nbr_spinor = malloc(Vsh*mySpinorSiteSize *3*sizeof(double));
  if (cpu_fwd_nbr_spinor == NULL || cpu_back_nbr_spinor == NULL){
    errorQuda("ERROR: malloc failed for cpu_fwd_nbr_spinor/cpu_back_nbr_spinor\n");
  }
#endif


#ifdef MULTI_GPU

  if(testtype == 6){
    record_gauge(fatlink, ghost_fatlink, Vsh,
		 longlink, ghost_longlink, 3*Vsh,
		 link_recon, link_recon_sloppy,
		 &gaugeParam);
   }else{
    int num_faces =1;
    gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
    gaugeParam.ga_pad = Vsh;
    gaugeParam.reconstruct= gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda_general_mg(fatlink, ghost_fatlink, &gaugeParam, &cudaFatLinkPrecise, &cudaFatLinkSloppy, num_faces);
    
    num_faces =3;
    gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
    gaugeParam.ga_pad = 3*Vsh;
    gaugeParam.reconstruct= link_recon;
    gaugeParam.reconstruct_sloppy = link_recon_sloppy;
    loadGaugeQuda_general_mg(longlink,ghost_longlink, &gaugeParam, &cudaLongLinkPrecise, &cudaLongLinkSloppy, num_faces);
  }

#else
  gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  loadGaugeQuda(fatlink, &gaugeParam);
  
  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
  gaugeParam.reconstruct = link_recon;
  gaugeParam.reconstruct_sloppy = link_recon_sloppy;
  loadGaugeQuda(longlink, &gaugeParam);
#endif
  
  double time0 = -((double)clock()); // Start the timer
  
  unsigned long volume = Vh;
  unsigned long nflops=2*1187; //from MILC's CG routine
  double nrm2=0;
  double src2=0;
  int ret = 0;


  switch(testtype){
  case 0: //even
    volume = Vh;
    inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    
    invertQuda(spinorOut, spinorIn, &inv_param);
    
    time0 += clock(); 
    time0 /= CLOCKS_PER_SEC;

#ifdef MULTI_GPU    
    matdagmat_mg(spinorCheck, fatlink, ghost_fatlink, longlink, ghost_longlink, 
		      spinorOut, cpu_fwd_nbr_spinor, cpu_back_nbr_spinor, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_EVEN);
    
#else
    matdagmat(spinorCheck, fatlink, longlink, spinorOut, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_EVEN);
#endif
    
    mxpy(spinorIn, spinorCheck, Vh*mySpinorSiteSize, inv_param.cpu_prec);
    nrm2 = norm_2(spinorCheck, Vh*mySpinorSiteSize, inv_param.cpu_prec);
    src2 = norm_2(spinorIn, Vh*mySpinorSiteSize, inv_param.cpu_prec);
    break;

  case 1: //odd
	
    volume = Vh;    
    inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
    inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
    invertQuda(spinorOutOdd, spinorInOdd, &inv_param);	
    time0 += clock(); // stop the timer
    time0 /= CLOCKS_PER_SEC;
    
#ifdef MULTI_GPU
    matdagmat_mg(spinorCheckOdd, fatlink, ghost_fatlink, longlink, ghost_longlink, 
		 spinorOutOdd, cpu_fwd_nbr_spinor, cpu_back_nbr_spinor, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_ODD);
#else
    matdagmat(spinorCheckOdd, fatlink, longlink, spinorOutOdd, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_ODD);	
#endif
    mxpy(spinorInOdd, spinorCheckOdd, Vh*mySpinorSiteSize, inv_param.cpu_prec);
    nrm2 = norm_2(spinorCheckOdd, Vh*mySpinorSiteSize, inv_param.cpu_prec);
    src2 = norm_2(spinorInOdd, Vh*mySpinorSiteSize, inv_param.cpu_prec);
	
    break;
    
  case 2: //full spinor

    volume = Vh; //FIXME: the time reported is only parity time
    inv_param.solve_type = QUDA_NORMEQ_SOLVE;
    inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION;
    invertQuda(spinorOut, spinorIn, &inv_param);
    
    time0 += clock(); // stop the timer
    time0 /= CLOCKS_PER_SEC;

#ifdef MULTI_GPU    
#else
    matdagmat(spinorCheck, fatlink, longlink, spinorOut, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_EVENODD);
#endif    
    mxpy(spinorIn, spinorCheck, V*mySpinorSiteSize, inv_param.cpu_prec);
    nrm2 = norm_2(spinorCheck, V*mySpinorSiteSize, inv_param.cpu_prec);
    src2 = norm_2(spinorIn, V*mySpinorSiteSize, inv_param.cpu_prec);

    break;

  case 3: //multi mass CG, even
  case 4:
  case 5:
  case 6:

#define NUM_OFFSETS 4
        
    nflops = 2*(1205 + 15* NUM_OFFSETS); //from MILC's multimass CG routine
    double masses[NUM_OFFSETS] ={5.05, 1.23, 2.64, 2.33};
    double offsets[NUM_OFFSETS];	
    int num_offsets =NUM_OFFSETS;
    void* spinorOutArray[NUM_OFFSETS];
    void* in;
    int len;
    
    for (int i=0; i< num_offsets;i++){
      offsets[i] = 4*masses[i]*masses[i];
    }
    
    if (testtype == 3){
      in=spinorIn;
      len=Vh;
      volume = Vh;
      
      inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
      inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;      
      
      spinorOutArray[0] = spinorOut;
      for (int i=1; i< num_offsets;i++){
	spinorOutArray[i] = malloc(Vh*mySpinorSiteSize*sSize);
      }		
    }
    
    else if (testtype == 4||testtype == 6){
      in=spinorInOdd;
      len = Vh;
      volume = Vh;

      inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
      inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
      
      spinorOutArray[0] = spinorOutOdd;
      for (int i=1; i< num_offsets;i++){
	spinorOutArray[i] = malloc(Vh*mySpinorSiteSize*sSize);
      }
    }else { //testtype ==5
      in=spinorIn;
      len= V;
      inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION;
      inv_param.solve_type = QUDA_NORMEQ_SOLVE;
      volume = Vh; //FIXME: the time reported is only parity time
      spinorOutArray[0] = spinorOut;
      for (int i=1; i< num_offsets;i++){
	spinorOutArray[i] = malloc(V*mySpinorSiteSize*sSize);
      }		
    }
    
    double residue_sq;
    if (testtype == 6){
      invertMultiShiftQudaMixed(spinorOutArray, in, &inv_param, offsets, num_offsets, &residue_sq);
    }else{      
      invertMultiShiftQuda(spinorOutArray, in, &inv_param, offsets, num_offsets, &residue_sq);	
    }
    cudaThreadSynchronize();
    printfQuda("Final residue squred =%g\n", residue_sq);
    time0 += clock(); // stop the timer
    time0 /= CLOCKS_PER_SEC;
    
    printfQuda("done: total time = %g secs, %i iter / %g secs = %g gflops, \n", 
	       time0, inv_param.iter, inv_param.secs,
	       inv_param.gflops/inv_param.secs);
    
    
    printfQuda("checking the solution\n");
    MyQudaParity parity;
    if (inv_param.solve_type == QUDA_NORMEQ_SOLVE){
      parity = QUDA_EVENODD;
    }else if (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN){
      parity = QUDA_EVEN;
    }else if (inv_param.matpc_type == QUDA_MATPC_ODD_ODD){
      parity = QUDA_ODD;
    }else{
      errorQuda("ERROR: invalid spinor parity \n");
      exit(1);
    }
    
    for(int i=0;i < num_offsets;i++){
      printfQuda("%dth solution: mass=%f, ", i, masses[i]);
#ifdef MULTI_GPU
      matdagmat_mg(spinorCheck, fatlink, ghost_fatlink, longlink, ghost_longlink, 
		   spinorOutArray[i], cpu_fwd_nbr_spinor, cpu_back_nbr_spinor, masses[i], 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, parity);
#else
      matdagmat(spinorCheck, fatlink, longlink, spinorOutArray[i], masses[i], 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, parity);
#endif
      mxpy(in, spinorCheck, len*mySpinorSiteSize, inv_param.cpu_prec);
      double nrm2 = norm_2(spinorCheck, len*mySpinorSiteSize, inv_param.cpu_prec);
      double src2 = norm_2(in, len*mySpinorSiteSize, inv_param.cpu_prec);

      printfQuda("relative residual, requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

      //emperical, if the cpu residue is more than 2 order the target accuracy, the it fails to converge
      if (sqrt(nrm2/src2) > 100*inv_param.tol){
	ret |=1;
	errorQuda("Converge failed!\n");
      }
    }
    
    for(int i=1; i < num_offsets;i++){
      free(spinorOutArray[i]);
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
#ifdef MULTI_GPU
  free(ghost_fatlink);
  free(ghost_longlink);
#endif  
  
  free(spinorIn);
  free(spinorOut);
  free(spinorCheck);
  free(tmp);

#ifdef MULTI_GPU
  free(cpu_fwd_nbr_spinor);
  free(cpu_back_nbr_spinor);
#endif
  endQuda();
}


void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon test_type  S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %s         %d          %d \n",
	 get_prec_str(prec),get_prec_str(prec_sloppy),
	 get_recon_str(link_recon), 
	 get_recon_str(link_recon_sloppy), get_test_type(testtype), sdim, tdim);     
  return ;
  
}

void
usage(char** argv )
{
  printfQuda("Usage: %s <args>\n", argv[0]);
  printfQuda("--prec         <double/single/half>     Spinor/gauge precision\n"); 
  printfQuda("--prec_sloppy  <double/single/half>     Spinor/gauge sloppy precision\n"); 
  printfQuda("--recon        <8/12>                   Long link reconstruction type\n"); 
  printfQuda("--test         <0/1/2/3/4/5>            Testing type(0=even, 1=odd, 2=full, 3=multimass even,\n" 
	 "                                                     4=multimass odd, 5=multimass full)\n"); 
  printfQuda("--tdim                                  T dimension\n");
  printfQuda("--sdim                                  S dimension\n");
  printfQuda("--help                                  Print out this message\n"); 
  exit(1);
  return ;
}


int main(int argc, char** argv)
{
#ifdef MULTI_GPU
  MPI_Init(&argc, &argv);
  comm_init();
#endif
  
  int i;
  for (i =1;i < argc; i++){
	
    if( strcmp(argv[i], "--help")== 0){
      usage(argv);
    }
	
    if( strcmp(argv[i], "--prec") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      prec = get_prec(argv[i+1]);
      i++;
      continue;	    
    }
    
    if( strcmp(argv[i], "--prec_sloppy") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      prec_sloppy =  get_prec(argv[i+1]);
      i++;
      continue;	    
    }
    
    
    if( strcmp(argv[i], "--recon") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      link_recon =  get_recon(argv[i+1]);
      i++;
      continue;	    
    }
    if( strcmp(argv[i], "--tol") == 0){
      float tmpf;
      if (i+1 >= argc){
        usage(argv);
      }
      sscanf(argv[i+1], "%f", &tmpf);
      if (tol <= 0){
        printfQuda("ERROR: invalid tol(%f)\n", tmpf);
        usage(argv);
      }
      tol = tmpf;
      i++;
      continue;
    }


	
    if( strcmp(argv[i], "--recon_sloppy") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      link_recon_sloppy =  get_recon(argv[i+1]);
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

    if( strcmp(argv[i], "--cprec") == 0){
      if (i+1 >= argc){
	usage(argv);
      }
      cpu_prec= get_prec(argv[i+1]);
      i++;
      continue;
    }

    if( strcmp(argv[i], "--tdim") == 0){
      if (i+1 >= argc){
	usage(argv);
      }
      tdim= atoi(argv[i+1]);
      if (tdim < 0 || tdim > 128){
	printfQuda("ERROR: invalid T dimention (%d)\n", tdim);
	usage(argv);
      }
      i++;
      continue;
    }		
    if( strcmp(argv[i], "--sdim") == 0){
      if (i+1 >= argc){
	usage(argv);
      }
      sdim= atoi(argv[i+1]);
      if (sdim < 0 || sdim > 128){
	printfQuda("ERROR: invalid S dimention (%d)\n", sdim);
	usage(argv);
      }
      i++;
      continue;
    }
    if( strcmp(argv[i], "--device") == 0){
          if (i+1 >= argc){
              usage(argv);
          }
          device =  atoi(argv[i+1]);
          if (device < 0){
	    printfQuda("Error: invalid device number(%d)\n", device);
              exit(1);
          }
          i++;
          continue;
    }


    printfQuda("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }


  if (prec_sloppy == QUDA_INVALID_PRECISION){
    prec_sloppy = prec;
  }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID){
    link_recon_sloppy = link_recon;
  }
  
  display_test_info();

  int ret = invert_test();

#ifdef MULTI_GPU  
  comm_cleanup();
#endif

  return ret;
}
