#include <cstdio>
#include <cstdlib>
#include <quda.h>
#include <dslash_quda.h>

#include "include/milc_utilities.h"
#include "external_headers/quda_milc_interface.h"

static void 
setGaugeParams(QudaGaugeParam* gaugeParam,
              const int dim[4],
              QudaPrecision precision,
              QudaReconstructType recon = QUDA_RECONSTRUCT_NO)
{
  for(int dir=0; dir<4; ++dir) gaugeParam->X[dir] = dim[dir];

  gaugeParam->type = QUDA_GENERAL_LINKS;
  gaugeParam->t_boundary = QUDA_PERIODIC_T;

  gaugeParam->cpu_prec = gaugeParam->cuda_prec = precision;
  gaugeParam->cuda_prec_sloppy = precision;
  gaugeParam->reconstruct = recon;
  gaugeParam->reconstruct_sloppy = recon;
  gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER;
  gaugeParam->anisotropy = 1.0;
  gaugeParam->tadpole_coeff = 1.0;
  gaugeParam->gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam->ga_pad = 0;
  gaugeParam->scale = 1;

  gaugeParam->cuda_prec_precondition = precision;
  gaugeParam->reconstruct_precondition = recon;


  return;
}

void*  qudaCreateExtendedGaugeField(void* gauge, int precision)
{
  using namespace milc_interface;
 
  QudaPrecision qudaPrecision = (precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION; 
  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  Layout layout;
  const int* dim = layout.getLocalDim();
  setGaugeParams(&gaugeParam, dim, qudaPrecision);

  //return createExtendedGaugeField(gauge, &gaugeParam);
  return NULL;
}




void qudaCloverDerivative(void* out, void* gauge, void* oprod, int mu, int nu, int precision, int parity)
{

  using namespace milc_interface;

  QudaParity qudaParity = (parity==2) ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY;
  QudaPrecision qudaPrecision = (precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  
  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  
  Layout layout;

  const int* dim = layout.getLocalDim();
  setGaugeParams(&gaugeParam, dim, qudaPrecision);

  computeCloverDerivativeQuda(out, gauge, oprod, mu, nu, qudaParity, &gaugeParam);


  return;
}
