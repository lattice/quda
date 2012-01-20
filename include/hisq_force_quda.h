#ifndef _HISQ_FORCE_QUDA_H
#define _HISQ_FORCE_QUDA_H

#include "hisq_force_utils.h"

#include <gauge_field.h>

namespace hisq {
  namespace fermion_force {
  void hisq_force_init_cuda(QudaGaugeParam* param);
  void hisq_staples_force_cuda(void* act_path_coeff, 
                              cudaGaugeField &oprod, 
                              cudaGaugeField &link, 
                              QudaGaugeParam* param,
                              cudaGaugeField &newOprod);

  void rewriteOprodCuda(cudaGaugeField &cudaForceMatrix, cudaGaugeField &cudaOprod, QudaGaugeParam* param);


  void set_unitarize_force_contants(double unitarize_eps, double hisq_force_filter);

  void unitarize_force_cuda(cudaGaugeField &cudaOldForce,
                            cudaGaugeField &cudaGauge,
                            cudaGaugeField &cudaNewForce);

  void unitarize_force_cpu(cpuGaugeField &cpuOldForce,
                            cpuGaugeField &cpuGauge,
                            cpuGaugeField &cpuNewForce);

  void rescaleHalfFieldCuda(cudaGaugeField &cudaField, 
                            const QudaGaugeParam& param,
                            int oddBit,
                            double coeff);


   void hisq_naik_force_cuda(void* path_coeff_array,
                             cudaGaugeField &oprod,
                             cudaGaugeField &link,
                             QudaGaugeParam* param,
                             cudaGaugeField &newOprod);

   void hisq_complete_force_cuda(cudaGaugeField &oprod,
                                 cudaGaugeField &link,
                                 QudaGaugeParam* param,
                                 cudaGaugeField &force);




 } // namespace fermion_force
}  // namespace hisq

#endif // _HISQ_FORCE_QUDA_H
