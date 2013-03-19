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


  inline void accumulate_time(double* time_difference, const timeval& start, const timeval& stop)
  {
    long ds = stop.tv_sec - start.tv_sec;
    long dus = stop.tv_usec - start.tv_usec;
    *time_difference += ds + 0.000001*dus;
  }

  void SimpleCG::operator()(cudaColorSpinorField& x, cudaColorSpinorField &b, double* time)
  {
    profile[QUDA_PROFILE_INIT].Start();

    int k=0;

    // Find the maximum domain overlap.
    // This will determine the number of faces needed by the vector r.
    // Have to be very careful to ensure that setting the number of 
    // ghost faces here doesn't screw anything up further down the line.
    ColorSpinorParam param(b);
    param.create = QUDA_COPY_FIELD_CREATE; 
    cudaColorSpinorField r(b);
    cudaColorSpinorField y(b);

    timeval mat_start, mat_stop;


    gettimeofday(&mat_start, NULL);
    mat(r, x, y); // operator()(cudaColorSpinorField& out, cudaColorSpinorField& in,
    // => r = A*x;
    double r2 = xmyNormCuda(b,r);
    gettimeofday(&mat_stop, NULL);

    accumulate_time(time, mat_start, mat_stop);

    cudaColorSpinorField p(r);
    cudaColorSpinorField Ap(r);
    
    profile[QUDA_PROFILE_INIT].Stop();
    profile[QUDA_PROFILE_PREAMBLE].Start();

    double alpha = 0.0, beta=0.0;
    double pAp;
    double r2_old;

    profile[QUDA_PROFILE_PREAMBLE].Stop();
    profile[QUDA_PROFILE_COMPUTE].Start();

    while( k < invParam.maxiter-1 ){
      gettimeofday(&mat_start, NULL);
      mat(Ap, p, y);
      pAp = reDotProductCuda(p, Ap);
      gettimeofday(&mat_stop, NULL);
            
      accumulate_time(time, mat_start, mat_stop);

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

    gettimeofday(&mat_start, NULL);
    mat(Ap, p, y);
    pAp = reDotProductCuda(p, Ap);
    gettimeofday(&mat_start, NULL);
    accumulate_time(time, mat_start, mat_stop);

    alpha  = r2/pAp;
    axpyCuda(alpha, p, x); // x --> x + alpha*p
   
    profile[QUDA_PROFILE_COMPUTE].Stop();
 
    return;
  }


} // namespace quda
