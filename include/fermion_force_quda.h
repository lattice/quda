#ifndef __FERMION_FORCE_QUDA_H__
#define __FERMION_FORCE_QUDA_H__

#ifdef __cplusplus
extern "C"{
#endif
    void fermion_force_init_cuda(QudaGaugeParam* param);
    void fermion_force_cuda(double eps, double weight1, double weight2, void* act_path_coeff,
			    FullHw cudaHw, FullGauge cudaSiteLink, FullMom cudaMom, QudaGaugeParam* param);
    
#ifdef __cplusplus
}
#endif

#endif

