#ifndef _FERMION_FORCE_QUDA_H
#define _FERMION_FORCE_QUDA_H

namespace quda {

  void fermion_force_init_cuda(QudaGaugeParam* param);
  void fermion_force_cuda(double eps, double weight1, double weight2, void* act_path_coeff,
			  FullHw cudaHw, cudaGaugeField &cudaSiteLink, cudaGaugeField &cudaMom, 
			  QudaGaugeParam* param);
    
} // namespace quda

#endif // _FERMION_FORCE_QUDA_H
