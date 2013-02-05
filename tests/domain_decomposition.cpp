#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <debug_utilities.h>

using namespace quda;


// First I need to create two sets of fields and assign two values 
// The usual quark fields and quark fields for the overlapping subdomains. 
// Then, I copy
namespace domain_decomposition
{

  void setDefaultPrecision(QudaPrecision* cpu_prec,
                           QudaPrecision* cuda_prec,
                           QudaPrecision* cuda_prec_sloppy,
                           QudaPrecision* cuda_prec_precon)
  {
    *cpu_prec = QUDA_DOUBLE_PRECISION;
    *cuda_prec = QUDA_SINGLE_PRECISION;
    *cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
    *cuda_prec_precon = QUDA_SINGLE_PRECISION; 
    return;
  }

  void setGaugePrecision(QudaGaugeParam* const gaugeParam,
                         QudaPrecision cpu_prec,
                         QudaPrecision cuda_prec,
                         QudaPrecision cuda_prec_sloppy
                         QudaPrecision cuda_prec_precondition)
  {
    gaugeParam->cpu_prec = cpu_prec;
    gaugeParam->cuda_prec = cuda_prec;
    gaugeParam->cuda_prec_sloppy = cuda_prec_sloppy;
    gaugeParam->cuda_prec_precondition = cuda_prec_precondition;
    return;
  }

  void setGaugeHISQDefaults(QudaGaugeParam* const gaugeParam)
  {
    // Things that might change
    gaugeParam->reconstruct = QUDA_RECONSTRUCT_NO;
    gaugeParam->reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER;  
    gaugeParam->ga_pad = dim[0]*dim[1]*dim[2]/2;

    // Things that probably won't change
    gaugeParam->type = QUDA_GENERAL_LINKS; // Same as QUDA_ASQTAD_FATLINKS
    gaugeParam->gauge_fix = QUDA_GAUGE_FIXED_NO;
    gaugeParam->anisotropy = 1.0;
    gaugeParam->tadpole_coeff = 1.0;
    gaugeParam->t_boundary = QUDA_PERIODIC_T; 

    return;
  }

  // Maybe I don't need to work with QudaGaugeParam 
  // since I won't be calling this code from a C client. 
  // Nonetheless, this is good to keep around.
  void setGaugeParams(QudaGaugeParam* const gaugeParam,
                      const int dim[4],
                      const QudaPrecision cpu_prec,
                      const QudaPrecision cuda_prec,
                      const QudaPrecision cuda_prec_sloppy,
                      const QudaPrecision cuda_prec_precon,
                      int nFace)
  {

    for(int dir=0; dir<4; ++dir) gaugeParam->X[dir] = dim[dir];

    gaugeParam->ga_pad = dim[0]*dim[1]*dim[2]/2; 
    gaugeParam->nFace = nFace;

    setGaugeHISQDefaults(gaugeParam);
    setGaugePrecision(gaugeParam, 
                      cpu_prec, 
                      cuda_prec, 
                      cuda_prec_sloppy,
                      cuda_prec_precon);

    return;
  }

// Code from gauge_field.h

struct GaugeFieldParam : public LatticeFieldParam {

//  int nColor; 
//  int nFace;
//    QudaReconstructType reconstruct;
// *************************************  QudaGaugeFieldOrder order;
//  QudaGaugeFixed fixed;
// *************************************  QudaLinkType link_type;
// *************************************   QudaTboundary t_boundary;

//  double anisotropy;
//  double tadpole;

//  void *gauge; // used when we reference an external field
//*************************************  QudaFieldCreate create; // used to determine the type of field created
//  QudaFieldGeometry geometry; // whether the field is a scalar, vector, or tensor.
//  int pinned; // used in cpu field only, where the host memory is pinned

  // Default constructor
  GaugeFieldParam(void* const h_gauge=NULL) : LatticeFieldParam(),
    nColor(3),
    nFace(0),
    reconstruct(QUDA_RECONSTRUCT_NO),
    order(QUDA_INVALID_GAUGE_ORDER),
    fixed(QUDA_GAUGE_FIXED_NO);
    link_type(QUDA_WILSON_LINKS),
    t_boundary(QUDA_INVALID_T_BOUNDARY),
    precision = QUDA_INVALID_PRECISION

};





// 




  void testDomainDecomposition();
  {
    QudaPrecision host_precion, device_precision, device_precision_sloppy, device_precision_precon;
    // set default precisions
    // Will specify command-line arguments that will override these
    setDefaultPrecision(&host_precision, 
                        &device_precision, 
                        &device_precision_sloppy,
                        &device_precision_precon);
 
    const int dim[4] = {8,8,8,8};
    const int nFace = 2;
    // instantiate QudaGaugeParam and invalidate every member variable
    QudaGaugeParam gaugeParam = newQudaGaugeParam(); 
    // set the gauge parameters for regular (not overlapping) gauge fields            
    setGaugeParam(&gaugeParam, dim, host_precision, 
                  device_precision, device_precision_sloppy, devce_precision_precon);
  

 
    cudaGaugeField fatGauge;
    cudaGaugeField longGauge;
    cudaGaugeField overlapGauge; 
    
    return;
  }


  int main(int argc, char* argv[])
  {
    return EXIT_SUCCESS;
  }

} // namespace domain_decomposition
