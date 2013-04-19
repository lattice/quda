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

using namespace quda;

ColorSpinorParam csParam;

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
//Test2
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
  csParam.create = QUDA_NULL_FIELD_CREATE;

  Nvec = nvec;
  W = new cpuColorSpinorField*[Nvec];
  for (int i=0; i<Nvec; i++) W[i] = new cpuColorSpinorField(csParam);

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
  for (int i=0; i<Nvec; i++) V[i] = W[i]->V();
    //supports seperate reading or single file read

  if (strcmp(vecfile,"")!=0) {
#if 0
    read_spinor_field(vecfile, &V[0], W[0]->Precision(), W[0]->X(), 
		      W[0]->Ncolor(), W[0]->Nspin(), Nvec, 0,  (char**)0);
#else 
    for (int i=0; i<Nvec; i++) {
      char filename[256];
      sprintf(filename, "%s.%d", vecfile, i);
      read_spinor_field(filename, &V[i], W[i]->Precision(), W[i]->X(), 
			W[i]->Ncolor(), W[i]->Nspin(), 1, 0,  (char**)0);
    }
#endif
  }

  for (int i=0; i<Nvec; i++) printfQuda("Vector %d has norm = %e\n", i, norm2(*W[i]));

  int geom_bs[] = {4, 2, 2, 2};
  int spin_bs = 2;

  Transfer P(W, Nvec, geom_bs, spin_bs);

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

  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);

  init();
  loadTest();
  end();

  endCommsQuda();
}

