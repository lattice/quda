#ifndef _HISQ_FORCE_QUDA_H
#define _HISQ_FORCE_QUDA_H

#include "hisq_force_utils.h"

#include <gauge_field.h>

namespace hisq {
  namespace fermion_force {
  void hisq_force_init_cuda(QudaGaugeParam* param);
  void hisq_force_cuda(void* act_path_coeff, 
                       cudaGaugeField &cudaOprod, 
                       cudaGaugeField &cudaSiteLink, 
                       QudaGaugeParam* param,
                       cudaGaugeField &cudaForceMatrix);

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
                             cudaGaugeField &cudaOprod,
                             cudaGaugeField &cudaLink,
                             QudaGaugeParam* param,
                             cudaGaugeField &cudaForce);

   void hisq_contract_cuda(cudaGaugeField &cudaOprod,
                           cudaGaugeField &cudaLink,
                           QudaGaugeParam* param,
                           cudaGaugeField &cudaForce);




 } // namespace fermion_force
}  // namespace hisq

#endif // _HISQ_FORCE_QUDA_H
