#ifndef _GAUGE_FORCE_QUDA_H
#define _GAUGE_FORCE_QUDA_H

#ifdef __cplusplus
extern "C"{
#endif
    
  void gauge_force_init_cuda(QudaGaugeParam* param, int max_length);
    
  void gauge_force_cuda(FullMom cudaMom, int dir, double eb3, FullGauge cudaSiteLink,
			QudaGaugeParam* param, int** input_path, int* length,
			void* path_coeff, int num_paths, int max_length);

#ifdef __cplusplus
}
#endif

#endif // _GAUGE_FORCE_QUDA_H
