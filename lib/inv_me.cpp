#include <cassert>
#include <invert_quda.h>
#include <blas_quda.h>

namespace quda {

  static void getPolynomialCoefficients(double coeff[], 
      double offset,
      const double prev_offset[], 
      int num_offsets)
  {
    assert(offset != 0.0);

    if(num_offsets == 0) return;
    if(num_offsets == 1) coeff[0] = 1.0;

    double sum = 0;
    for(int i=0; i<num_offsets; ++i){
      double temp = 1.0;
      for(int j=0; j<num_offsets; ++j){
        if(i == j) continue;
        if(prev_offset[i] == prev_offset[j]) errorQuda("Offsets must be distinct\n");
        temp *= (current_offset - prev_offset[j])/(prev_offset[i] - prev_offset[j]) 
      }
      coeff[i] = temp;
    }
    return;
  }


  // Compute trial solution x[idx] by 
  // applying polynomial extrapolation in the 
  // masses to the previous solutions x[i>idx]
  void polyMassExt(cudaColorSpinorField **x,
                   const QudaInvertParam& param,
                   int idx)
  {
    const int last_idx = param.num_offset-1;
    if(idx > last_idx) errorQuda("index exceeds number of solutions\n");

    if(idx == last_idx) return; // nothing to do 

    double* prev_offset = new double[param.num_offset];
    double* coeff = new double[param.num_offset];

    const int order = last_idx - idx;

    for(int j=0; j<order; ++j) 
      prev_offset[j] = param.offset[idx+j+1];

    getPolynomialCoefficients(coeff, param.offset[idx], prev_offset, order);

    for(int j=0; j<order; ++j)
      axpyCuda(coeff[j], *x[idx+j+1], *x[idx]);

    delete[] prev_offset;
    delete[] coeff;

    return;
  }


}
