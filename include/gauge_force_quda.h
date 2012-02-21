#ifndef _GAUGE_FORCE_QUDA_H
#define _GAUGE_FORCE_QUDA_H

#ifdef __cplusplus
extern "C"{
#endif
  
  void gauge_force_init_cuda(QudaGaugeParam* param, int max_length);
  void gauge_force_cuda(cudaGaugeField& cudaMom, double eb3, cudaGaugeField& cudaSiteLink,
			QudaGaugeParam* param, int*** input_path, int* length,
			void* path_coeff, int num_paths, int max_length);
  int computeGaugeForceQuda(void* mom, void* sitelink,  int*** input_path_buf, int* path_length,
			    void* loop_coeff, int num_paths, int max_length, double eb3,
			    QudaGaugeParam* qudaGaugeParam);
#ifdef __cplusplus
}
#endif

#endif // _GAUGE_FORCE_QUDA_H
