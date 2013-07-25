#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <quda_internal.h>
#include <gauge_field.h>
#include <util_quda.h>

#include <test_util.h>

#include <color_spinor_field.h>
#include <blas_quda.h>

#include <qio_field.h>
#include <transfer.h>

#include <cstring>
#include <wilson_dslash_reference.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif


#define MAX(a,b) ((a)>(b)?(a):(b))

using namespace quda;

ColorSpinorParam csParam, coarsecsParam;

float kappa = 1.0/(2.0*(4-.4125));

extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern char latfile[];
extern int gridsize_from_cmdline[];
    
extern int nvec;
extern char vecfile[];

std::vector<ColorSpinorField*> W; // array of bad guys
cpuColorSpinorField *V1, *V2;
int Nvec; // number of bad guys for the transfer operator
QudaPrecision prec_cpu = QUDA_SINGLE_PRECISION;

// where is the packing / unpacking taking place
//most orders are CPU only currently
const QudaFieldLocation location = QUDA_CPU_FIELD_LOCATION;
QudaGaugeParam gauge_param;
QudaInvertParam inv_param;
void *hostGauge[4];

DiracWilson *dirac;
extern int test_type;

// Dirac operator type
extern QudaDslashType dslash_type;
cudaGaugeField *cudagauge;
cpuGaugeField *cpugauge;


void init(int argc, char **argv) {

  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;

  csParam.x[0] = xdim;
  csParam.x[1] = ydim;
  csParam.x[2] = zdim;
  csParam.x[3] = tdim;
  setDims(csParam.x);

  csParam.precision = prec_cpu;
  csParam.pad = 0;
  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_ZERO_FIELD_CREATE;

  Nvec = nvec;
  W.resize(Nvec);
  for (int i=0; i<Nvec; i++) {
    W[i] = new cpuColorSpinorField(csParam);
    if (W[i] == NULL) {
      printf("Could not allocate W[%d]\n", i);      
    }
  }
  V1 = new cpuColorSpinorField(csParam);
  V2 = new cpuColorSpinorField(csParam);


  gauge_param = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  //setDims(gauge_param.X);
  setKernelPackT(false);
  Ls = 1;
  setSpinorSiteSize(24);

  gauge_param.anisotropy = 1.0;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  gauge_param.cpu_prec = prec_cpu;
  gauge_param.cuda_prec = prec_cpu;
  gauge_param.reconstruct = link_recon;
  gauge_param.reconstruct_sloppy = link_recon;
  gauge_param.cuda_prec_sloppy = prec_cpu;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;


#ifndef MULTI_GPU // free parameter for single GPU
  gauge_param.ga_pad = 0;
#else // must be this one c/b face for multi gpu
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif

  inv_param.kappa = kappa;

  inv_param.Ls = (inv_param.twist_flavor != QUDA_TWIST_NONDEG_DOUBLET) ? Ls : 1;
  
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  inv_param.dagger = QUDA_DAG_NO;

  inv_param.cpu_prec = prec_cpu;
  if (inv_param.cpu_prec != gauge_param.cpu_prec) {
    errorQuda("Gauge and spinor CPU precisions must match");
  }
  inv_param.cuda_prec = prec_cpu;
  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;


  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // test code only supports DeGrand-Rossi Basis
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  switch(test_type) {
  case 0:
  case 1:
    inv_param.solution_type = QUDA_MATPC_SOLUTION;
    break;
  case 2:
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    break;
  case 3:
    inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
    break;
  case 4:
    inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION;
    break;
  default:
    errorQuda("Test type %d not defined\n", test_type);
  }

  inv_param.dslash_type = dslash_type;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(V*gaugeSiteSize*gauge_param.cpu_prec);


 if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
   read_gauge_field(latfile, hostGauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
   construct_gauge_field(hostGauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate a random SU(3) field
   construct_gauge_field(hostGauge, 1, gauge_param.cpu_prec, &gauge_param);
  }
 GaugeFieldParam gParam(hostGauge, gauge_param);  
  cpugauge = new cpuGaugeField(gParam);

  GaugeFieldParam gParam2(0, gauge_param);  
  gParam2.create = QUDA_NULL_FIELD_CREATE;
  cudagauge = new cudaGaugeField(gParam2);
  cudagauge->loadCPUField(*cpugauge, QUDA_CPU_FIELD_LOCATION);


    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, 0);
    diracParam.verbose = QUDA_VERBOSE;
    //diracParam.tmp1 = tmp1;
    //diracParam.tmp2 = tmp2;
    
    dirac = new DiracWilson(diracParam);

  initQuda(device);
}

void end() {
  // release memory
  
  for (int i=0; i<Nvec; i++) delete W[i];
  //delete []W;
  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  delete cpugauge;
  delete cudagauge;
  delete V1;
  delete V2;
  delete dirac;

  endQuda();
}

void loadTest() {


  void **V = new void*[Nvec];
  for (int i=0; i<Nvec; i++) { 
    V[i] = W[i]->V();
    if (V[i] == NULL) {
      printf("Could not allocate V[%d]\n", i);      
    }
  }
    //supports seperate reading or single file read

  if (strcmp(vecfile,"")!=0) {
#if 0
    read_spinor_field(vecfile, &V[0], W[0]->Precision(), W[0]->X(), 
		      W[0]->Ncolor(), W[0]->Nspin(), Nvec, 0,  (char**)0);
#else 
    for (int i=0; i<Nvec; i++) {
      char filename[256];
      sprintf(filename, "%s.%d", vecfile, i);
      printf("Reading vector %d from file %s\n", i, filename);
      read_spinor_field(filename, &V[i], W[i]->Precision(), W[i]->X(), 
			W[i]->Ncolor(), W[i]->Nspin(), 1, 0,  (char**)0);
    }
#endif
  }

  for (int i=0; i<Nvec; i++) printfQuda("Vector %d has norm = %e\n", i, blas::norm2(*W[i]));


  int geom_bs[] = {4, 4, 4, 4};
  int spin_bs = 2;

  printfQuda("Creating transfer operator with Nvec=%d\n", Nvec);
  Transfer T(W, Nvec, geom_bs, spin_bs);

  cpuColorSpinorField Wfine(csParam);
  ColorSpinorField *Wcoarse = W[0]->CreateCoarse(geom_bs, spin_bs, Nvec);

  // test that the prolongator preserves the components which were
  // used to define it
  for (int i=0; i<Nvec; i++) {
    T.R(*Wcoarse,*W[i]);
    T.P(Wfine, *Wcoarse);
    blas::axpy(-1.0, *W[i], Wfine);
    printfQuda("%d Absolute Norm^2 of the difference = %e\n", i, blas::norm2(Wfine));
    //Wfine.Compare(Wfine,*W[i]); // strong check
  }

  //  V1->Source(QUDA_RANDOM_SOURCE);
  //    printfQuda("Absolute Norm^2 of V1 = %e\n", blas::norm2(*V1));
  //wil_mat(V2->V(), hostGauge, V1->V(), kappa, QUDA_DAG_NO, prec_cpu, gauge_param);
  //    printfQuda("Absolute Norm^2 of V2 = %e\n", blas::norm2(*V2));

      int ndim = 4;
      std::complex<float> *Y[2*ndim+1];
      int Nc = Wcoarse->Ncolor();
      int Ns = Wcoarse->Nspin();
      printfQuda("Vol = %d, Nc_c = %d, Ns_c = %d\n", Wcoarse->Volume(), Nc, Ns);
      for(int d = 0; d < 2*ndim+1; d++) {
	Y[d] = (std::complex<float> *) malloc(Wcoarse->Volume()*Ns*Ns*Nc*Nc*sizeof(float));
      }

  delete Wcoarse;
      for(int d = 0; d < 2*ndim+1; d++) {
	free(Y[d]);
      }

}

extern void usage(char**);

int main(int argc, char **argv) {

  for (int i=1; i<argc; i++){    
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }  
    
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }


  initComms(argc, argv, gridsize_from_cmdline);

  init(argc, argv);
  loadTest();
  end();

  finalizeComms();


  return 0;
}

