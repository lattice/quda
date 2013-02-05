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


  // Look at the link variables
  void createSiteLinks()
  {
    void* siteLink_2d[4];
    void* siteLink_ex_2d[4];

    // Allocate page-locked memory
    for(int dir=0; dir<4; ++dir){
#ifdef GPU_DIRECT
      cudaMallocHost(&siteLink_2d[dir], V*gaugeSiteSize*qudaGaugeParam.cpu_prec);
      cudaMallocHost(&siteLink_ex_2d[dir], V_ex*gaugeSiteSite*qudaGaugeParam.cpu_prec);
#else
      siteLink_2d[dir] = malloc(V*gaugeSiteSize*qudaGaugeParam.cpu_prec);
      siteLink_ex_2d[dir] = malloc(V_ex*gaugeSiteSize*qudaGaugeParam.cpu_prec);
#endif
      memset(siteLink_2d[dir], 0, V*gaugeSiteSize*qudaGaugeParam.cpu_prec);
      memset(siteLink_ex_2d[dir], 0, V_ex*gaugeSiteSize*qudaGaugeParam.cpu_prec);
    }

    // fill the gauge field with random numbers
    createSiteLinkCPU(siteLink_2d, qudaGaugeParam.cpu_prec, 1);

    int X1 = Z[0];
    int X2 = Z[1];
    int X3 = Z[2];
    int X4 = Z[3];

    for(int i=0; i<V_ex; ++i){
      int sid = i;
      int oddBit=0;
      if(i >= Vh_ex){
        sid = i - Vh_ex;
        oddBit = 1;
      }
      // sid = x4*E3*E2*E1h + x3*E2*E1h + x2*E1h + x1h;
      x1h = sid % E1h;
      x2  = (sid / E1h) % E2;
      x3  = (sid / (E2*E1h) ) % E3;
      x4  =  sid / (E3*E2*E1h);  
      int x1odd = (x2 + x3 + x4 + oddBit) & 1;
      int x1 = 2*x1h + x1odd;
      if( x1 < 2 || x1 >= X1+2
      ||  x2 < 2 || x2 >= X2+2
      ||  x3 < 2 || x3 >= X3+2
      ||  x4 < 2 || x4 >= X4+2 ){ continue; }
      // do nothing if the sites are outside the interior
      x1 = (x1 - 2 + X1 ) % X1;
      x2 = (x2 - 2 + X2 ) % X2;
      x3 = (x3 - 2 + X3 ) % X3;
      x4 = (x4 - 2 + X4 ) % X4;
    
      int idx = (x4*X3*X2*X1 + x3*X2*X1 + x2*X1 + x1) >> 1;
      if(oddBit){
        idx += Vh;
      }
      for(int dir=0; dir<4; ++dir){
        memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
      } // dir
    }  // i

 
    return;
  }



  // create two fat 
  void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
  {
    GaugeFieldParam gauge_param(h_gauge, *param);
    cpuGaugeField *cpu = new cpuGaugeField(gauge_param);

    gauge_param.create = QUDA_NULL_FIELD_CREATE;
    gauge_param.precision = param->cuda_prec;
    gauge_param.reconstruct = param->reconstruct;
    gauge_param.pad = param->ga_pad;
    gauge_param.order = (gauge_param.precision == QUDA_DOUBLE_PRECISION ||
                         gauge_param.reconstruct == QUDA_RECONSTRUCT_NO) ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  }

  // Write a function loadGauge, which loads each of the gauge fields separately
  // call once for the outer precision, call once for the sloppy precision
  // call once for the preconditioning.
  // I need a flag to distinguish between the precision types
  // the signature should probably be
  void loadGauge(cudaGaugeField* const out, const cpuGaugeField&  in, const GaugeFieldParam& param);
  {
    return; 
  }

  void setGaugeFields(const cudaGaugeField& precise_field,
                      const cudaGaugeField& sloppy_field,
                      const cudaGaugeField& precon_field
                      QudaLinkType type)
  {
    extern cudaGaugeField *gaugePrecise;
    extern cuda
    if(type == QUDA_ASQTAD_FAT_LINKS){
      gaugeFatPrecise      = &precise_field;
      gaugeFatSloppy       = &sloppy_field;
      gaugeFatPrecondition = &precon_field;
    }else if(type == QUDA_ASQTAD_LONG_LINKS){
      gaugeLongPrecise      = &precon_field;
      gaugeLongSloppy       = &sloppy_field;
      gaugeLongPrecondition = &precon_field;
    }else{
      errorQuda("Only fat and long links currently supported");
    }
    return;
  }

  void domainDecomposedInverter(const cudaGaugeField& field,
                                const cudaGaugeField& field

  int main(int argc, char* argv[])
  {
    return EXIT_SUCCESS;
  }

} // namespace domain_decomposition
