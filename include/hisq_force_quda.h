#ifndef _HISQ_FORCE_QUDA_H
#define _HISQ_FORCE_QUDA_H

#include "hisq_force_utils.h"

#include <gauge_field.h>

namespace quda {
  namespace fermion_force {

  void hisqForceInitCuda(QudaGaugeParam* param);

  void hisqStaplesForceCuda(const double path_coeff[6], 
                              const QudaGaugeParam& param,
                              const cudaGaugeField& oprod, 
                              const cudaGaugeField& link, 
                              cudaGaugeField *newOprod);


   void hisqLongLinkForceCuda(double coeff,
                             const QudaGaugeParam& param,
                             const cudaGaugeField &oprod,
                             const cudaGaugeField &link,
                             cudaGaugeField *newOprod);


   void hisqCompleteForceCuda(const QudaGaugeParam &param,
				 const cudaGaugeField &oprod,
                                 const cudaGaugeField &link,
                                 cudaGaugeField *force);



  void setUnitarizeForceConstants(double unitarize_eps, double hisq_force_filter, double max_det_error,
				     bool allow_svd, bool svd_only,
				     double svd_rel_error,
				     double svd_abs_error);

  void unitarizeForceCuda(const QudaGaugeParam &param,
			    cudaGaugeField &cudaOldForce,
                            cudaGaugeField &cudaGauge,
                            cudaGaugeField *cudaNewForce,
			    int* unitarization_failed);

  void unitarizeForceCPU( const QudaGaugeParam &param,
			    cpuGaugeField &cpuOldForce,
                            cpuGaugeField &cpuGauge,
                            cpuGaugeField *cpuNewForce);


 } // namespace fermion_force
}  // namespace quda

#endif // _HISQ_FORCE_QUDA_H
