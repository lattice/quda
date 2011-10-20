#ifndef _FERMION_FORCE_QUDA_H
#define _FERMION_FORCE_QUDA_H

#ifdef __cplusplus
extern "C"{
#endif

  void fermion_force_init_cuda(QudaGaugeParam* param);
  void fermion_force_cuda(double eps, double weight1, double weight2, void* act_path_coeff,
			  FullHw cudaHw, cudaGaugeField &cudaSiteLink, cudaGaugeField &cudaMom, 
			  QudaGaugeParam* param);
    
#ifdef __cplusplus
}
#endif

#endif // _FERMION_FORCE_QUDA_H
