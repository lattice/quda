#ifndef _GAUGE_FORCE_QUDA_H
#define _GAUGE_FORCE_QUDA_H

namespace quda {

  void gauge_force_cuda(cudaGaugeField& cudaMom, double eb3, cudaGaugeField& cudaSiteLink,
			QudaGaugeParam* param, int*** input_path, int* length,
			double* path_coeff, int num_paths, int max_length);

} // namespace quda


#endif // _GAUGE_FORCE_QUDA_H
