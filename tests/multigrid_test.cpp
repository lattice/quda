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
#include <multigrid.h>

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

std::vector<ColorSpinorField*> W; // array of bad guys
int Nvec; // number of bad guys for the transfer operator
QudaPrecision prec_cpu = QUDA_SINGLE_PRECISION;

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
  W.resize(Nvec);
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
  //delete []W;

  endQuda();
}

void loadTest() {

  loadVectors(W);
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

  delete Wcoarse;

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

