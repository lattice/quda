#include <cstdio>
#include <cstdlib>
#include <iostream> 
#include <iomanip>
#include <quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <gauge_field.h>

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
    const cudaColorSpinorField& source,
    const cudaGaugeField& preciseFatGauge,
    const cudaGaugeField& preciseLongGauge,
    const cudaGaugeField& sloppyFatGauge,
    const cudaGaugeField& sloppyLongGauge,
    const cudaGaugeField& preconFatGauge,
    const cudaGaugeField& preconLongGauge)
{
  // assign the gauge fields in interface_quda.cpp
  setGaugeField(preciseFatGauge, sloppyFatGauge, preconFatGauge, QUDA_ASQTAD_FAT_LINKS);
  setGaugeField(preciseLongGauge, sloppyLongGauge, preconLongGauge, QUDA_ASQTAD_LONG_LINKS);
  
  return;
}


int main(int argc, char *argv[])
{


  return EXIT_SUCCESS;
}
