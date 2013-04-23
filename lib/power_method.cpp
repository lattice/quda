#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>
#include <face_quda.h>
#include <domain_decomposition.h>
#include <resize_quda.h>

// I need to add functionality to the conjugate-gradient solver.
// Should I do this by simple inheritance, or should I use a decorator? 
// Is it even possible to use a decorator?

namespace quda {

  // Uses the power method to compute the maximum eigenvalue of the DiracMatrix 
  // on a given subdomain.
  
  // Use the power method to calculate the largest eigenvalue of the local Dirac operator
  void maxEigenValue(double* max_eval, const DiracMatrix& mat, const cudaColorSpinorField& b, int max_iter)
  {
    globalReduce = false;

    cudaColorSpinorField x(b); 
    cudaColorSpinorField tmp(b);
    cudaColorSpinorField Ax(b);
    
    int k=0;
    while(k < max_iter){
     double norm_Ax = sqrt(norm2(Ax));
     x = Ax;
     axCuda(1./norm_Ax, x);
     mat(Ax, x, tmp);
    }

    double xAx = reDotProductCuda(x, Ax);
    double x2  = norm2(x);

    *max_eval = xAx/x2;
    
    globalReduce = true;
     
    return;
  }

} // namespace quda
