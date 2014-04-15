#ifndef _KS_IMPROVED_FORCE_H
#define _KS_IMPROVED_FORCE_H

#include <quda_internal.h>
#include <quda.h>
#include <gauge_field.h>

namespace quda {
  namespace fermion_force {

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

  void unitarizeForceCuda(cudaGaugeField &cudaOldForce,
                          cudaGaugeField &cudaGauge,
                          cudaGaugeField *cudaNewForce,
			  int* unitarization_failed);

  void unitarizeForceCPU( cpuGaugeField &cpuOldForce,
                          cpuGaugeField &cpuGauge,
                          cpuGaugeField *cpuNewForce);


 } // namespace fermion_force
}  // namespace quda

#endif // _KS_IMPROVED_FORCE_H
