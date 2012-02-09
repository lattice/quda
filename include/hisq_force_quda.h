#ifndef _HISQ_FORCE_QUDA_H
#define _HISQ_FORCE_QUDA_H

#include "hisq_force_utils.h"

#include <gauge_field.h>

namespace hisq {
  namespace fermion_force {

  void hisq_force_init_cuda(QudaGaugeParam* param);

  void hisq_staples_force_cuda(const double path_coeff[6], 
                              const QudaGaugeParam& param,
                              const cudaGaugeField& oprod, 
                              const cudaGaugeField& link, 
                              cudaGaugeField *newOprod);


   void hisq_longlink_force_cuda(double coeff,
                             const QudaGaugeParam& param,
                             const cudaGaugeField &oprod,
                             const cudaGaugeField &link,
                             cudaGaugeField *newOprod);


   void hisq_complete_force_cuda(const QudaGaugeParam &param,
				 const cudaGaugeField &oprod,
                                 const cudaGaugeField &link,
                                 cudaGaugeField *force);



  void set_unitarize_force_constants(double unitarize_eps, double hisq_force_filter, double max_det_error,
				     bool allow_svd, bool svd_only,
				     double svd_rel_error,
				     double svd_abs_error);

  void unitarize_force_cuda(const QudaGaugeParam &param,
			    cudaGaugeField &cudaOldForce,
                            cudaGaugeField &cudaGauge,
                            cudaGaugeField *cudaNewForce,
			    int* unitarization_failed);

  void unitarize_force_cpu( const QudaGaugeParam &param,
			    cpuGaugeField &cpuOldForce,
                            cpuGaugeField &cpuGauge,
                            cpuGaugeField *cpuNewForce);


 } // namespace fermion_force
}  // namespace hisq

#endif // _HISQ_FORCE_QUDA_H
