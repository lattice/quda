#ifndef _GAUGE_FORCE_QUDA_H
#define _GAUGE_FORCE_QUDA_H

namespace quda {

  typedef struct kernel_param_s {
    unsigned long threads;
    int ghostDim[4]; // Whether a ghost zone has been allocated for a given dimension
  } kernel_param_t;
 
 
  void gauge_force_init_cuda(QudaGaugeParam* param, int max_length);
  void gauge_force_cuda(cudaGaugeField& cudaMom, double eb3, cudaGaugeField& cudaSiteLink,
			QudaGaugeParam* param, int*** input_path, int* length,
			double* path_coeff, int num_paths, int max_length);

} // namespace quda


#endif // _GAUGE_FORCE_QUDA_H
