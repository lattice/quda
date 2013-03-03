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


  SimpleCG::SimpleCG(const DiracMatrix &mat,  QudaInvertParam &invParam, TimeProfile &profile) :
    Solver(invParam, profile), mat(mat) 
  {
  }


  SimpleCG::~SimpleCG() {
  }

  void SimpleCG::operator()(cudaColorSpinorField& x, cudaColorSpinorField &b)
  {
    int k=0;

    // Find the maximum domain overlap.
    // This will determine the number of faces needed by the vector r.
    // Have to be very careful to ensure that setting the number of 
    // ghost faces here doesn't screw anything up further down the line.
    ColorSpinorParam param(b);
    param.create = QUDA_COPY_FIELD_CREATE; 
    cudaColorSpinorField r(b);
    cudaColorSpinorField y(b);


    mat(r, x, y); // operator()(cudaColorSpinorField& out, cudaColorSpinorField& in,
    // => r = A*x;
    double r2 = xmyNormCuda(b,r);
    cudaColorSpinorField p(r);
    cudaColorSpinorField Ap(r);


    double alpha = 0.0, beta=0.0;
    double pAp;
    double r2_old;


    while( k < invParam.maxiter-1 ){
      mat(Ap, p, y);
      pAp = reDotProductCuda(p, Ap);
      
      alpha = r2/pAp; 
      axpyCuda(-alpha, Ap, r); // r --> r - alpha*Ap
      r2_old = r2;
      r2 = norm2(r);
      beta = r2/r2_old;
      axpyZpbxCuda(alpha, p, x, r, beta);
      // x = x + alpha*p
      // p = r + beta*p
      ++k;
    }

    mat(Ap, p, y);
    pAp = reDotProductCuda(p, Ap);
    alpha  = r2/pAp;
    axpyCuda(alpha, p, x); // x --> x + alpha*p
    
    return;
  }


} // namespace quda
