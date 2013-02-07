#include <cstdio>
#include <cstdlib>
#include <iostream> 
#include <iomanip>
#include <quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <domain_decomposition.h>
#include <dslash_quda.h>

#include <test_utilities.h>

using namespace std;
using namespace quda;


void setGaugeField(cudaGaugeField& preciseGauge,
    cudaGaugeField& sloppyGauge,
    cudaGaugeField& preconGauge,
    QudaLinkType type)
{
  // These are defined in a different translation unit.
  //  Will this work?
  extern cudaGaugeField* gaugeFatPrecise;
  extern cudaGaugeField* gaugeLongPrecise;

  if(type == QUDA_ASQTAD_FAT_LINKS){
    gaugeFatPrecise = &preciseGauge;
  } else if(type == QUDA_ASQTAD_LONG_LINKS){
    gaugeLongPrecise = &preciseGauge;
  } else {
    errorQuda("Unsupported link type %d", type);
  }
  return;
}


void applyDDSolver(cudaColorSpinorField* const solution,
                   cudaColorSpinorField& source,
    const cudaGaugeField& preciseFatGauge,
    const cudaGaugeField& preciseLongGauge,
    const cudaGaugeField& sloppyFatGauge,
    const cudaGaugeField& sloppyLongGauge,
    const cudaGaugeField& preconFatGauge,
    const cudaGaugeField& preconLongGauge)
{
  // assign the gauge fields in interface_quda.cpp
  // Have to cast away the constantness to get it to work.
  // Yuck!
  setGaugeField(const_cast<cudaGaugeField&>(preciseFatGauge), 
                const_cast<cudaGaugeField&>(sloppyFatGauge), 
                const_cast<cudaGaugeField&>(preconFatGauge), 
                QUDA_ASQTAD_FAT_LINKS);


  setGaugeField(const_cast<cudaGaugeField&>(preciseLongGauge), 
                const_cast<cudaGaugeField&>(sloppyLongGauge), 
                const_cast<cudaGaugeField&>(preconLongGauge), 
                QUDA_ASQTAD_LONG_LINKS);


  DecompParam decompParam;
  initDecompParam(&decompParam, preciseFatGauge.X(), preconLongGauge.X());

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  QudaInvertParam param;
  bool pc_solve = true;

  createDirac(d, dSloppy, dPre, param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre; 

  const int *X = preciseFatGauge.X();
  const int *Y = preconFatGauge.X();

  massRescale(QUDA_ASQTAD_DSLASH, 
              param.kappa, 
              QUDA_MATPCDAG_MATPC_SOLUTION, 
              QUDA_MASS_NORMALIZATION, 
              source);

 
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre); 

/*
  Solver *solve = Solver::create(*param, m, mSloppy, mPre, profileInvert);
  solve->operator()(solution, source);
*/

  delete d;
  delete dSloppy;
  delete dPre;

  return;
}


int main(int argc, char *argv[])
{


  return EXIT_SUCCESS;
}
