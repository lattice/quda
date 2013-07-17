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

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif


using namespace quda;

ColorSpinorParam csParam, coarsecsParam;

float kappa = 1.0;

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

cpuColorSpinorField **W; // array of bad guys
int Nvec; // number of bad guys for the transfer operator
QudaPrecision prec_cpu = QUDA_DOUBLE_PRECISION;

// where is the packing / unpacking taking place
//most orders are CPU only currently
const QudaFieldLocation location = QUDA_CPU_FIELD_LOCATION;

void init() {

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
  W = new cpuColorSpinorField*[Nvec];
  for (int i=0; i<Nvec; i++) {
    W[i] = new cpuColorSpinorField(csParam);
    if (W[i] == NULL) {
      printf("Could not allocate W[%d]\n", i);      
    }
  }

  initQuda(device);
}

void end() {
  // release memory
  
  for (int i=0; i<Nvec; i++) delete W[i];
  delete []W;

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

  for (int i=0; i<Nvec; i++) printfQuda("Vector %d has norm = %e\n", i, norm2(*W[i]));


  int geom_bs[] = {4, 2, 2, 2};
  int spin_bs = 2;

  Transfer T(W, Nvec, geom_bs, spin_bs);

  coarsecsParam.nColor = Nvec;
  //coarsecsParam.nColor = 3;
  coarsecsParam.nSpin = 4/spin_bs;
  //coarsecsParam.nSpin = 4;
  coarsecsParam.nDim = 4;

  #if 1
  coarsecsParam.x[0] = xdim/geom_bs[0];
  coarsecsParam.x[1] = ydim/geom_bs[1];
  coarsecsParam.x[2] = zdim/geom_bs[2];
  coarsecsParam.x[3] = tdim/geom_bs[3];
  #else
  coarsecsParam.x[0] = xdim;
  coarsecsParam.x[1] = ydim;
  coarsecsParam.x[2] = zdim;
  coarsecsParam.x[3] = tdim;
  #endif
  setDims(coarsecsParam.x);

  coarsecsParam.precision = prec_cpu;
  coarsecsParam.pad = 0;
  coarsecsParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  coarsecsParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  coarsecsParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  coarsecsParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  coarsecsParam.create = QUDA_ZERO_FIELD_CREATE;

  for (int j=0; j<3; j++) {

#if 0
    printfQuda("W[%d]\n",j);
  for (int i = 0; i < 1;i++) {
    W[j]->PrintVector(i);
  }
#endif

  cpuColorSpinorField Wfine2(csParam);
  cpuColorSpinorField Wfine(csParam);
  cpuColorSpinorField Wcoarse2(coarsecsParam);
  cpuColorSpinorField Wcoarse(coarsecsParam);

  //T.R(Wcoarse,*W[j]);
  //Test prolongator
  #if 0
  std::complex<double> *fcoarse = (std::complex<double> *)Wcoarse.V();
  for (int i = 0; i < Wcoarse.Volume(); i++) {
    for (int s = 0; s < Wcoarse.Nspin(); s++) {
      fcoarse[i*Wcoarse.Nspin()*Wcoarse.Ncolor()+s*Wcoarse.Ncolor()+j] = 1.0;
    }
  }
  #endif
  //T.P(Wfine,Wcoarse);

  //printfQuda("Wcoarse\n");
#if 0
  for (int i = 0; i < 1;i++) {
    Wcoarse.PrintVector(i);
  }
  #endif
  #if 1
  std::complex<double> *fcoarse = (std::complex<double> *)Wfine.V();
  for (int i = 0; i < Wfine.Volume(); i++) {
    for (int s = 0; s < Wfine.Nspin(); s++) {
      fcoarse[i*Wfine.Nspin()*Wfine.Ncolor()+s*Wfine.Ncolor()+j] = 1.0;
    }
  }
  #endif

    printfQuda("Wfine\n");
#if 1
  for (int i = 0; i < 1;i++) {
    Wfine.PrintVector(i);
  }
  #endif
  printfQuda("Now restrict\n");
  T.R(Wcoarse2,Wfine);
    printfQuda("Wcoarse2\n");
#if 1
  for (int i = 0; i < 1;i++) {
    Wcoarse2.PrintVector(i);
  }
  #endif
  printfQuda("now prolongate\n");
  T.P(Wfine2,Wcoarse2);
    printfQuda("Wfine2\n");
#if 1
  for (int i = 0; i < 1;i++) {
    Wfine2.PrintVector(i);
  }
  #endif
  Wfine.Compare(Wfine,Wfine2);	
  //Wfine.Compare(*W[0],Wfine);
  }
      
  delete []V;

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

  init();
  loadTest();
  end();

  finalizeComms();


  return 0;
}

