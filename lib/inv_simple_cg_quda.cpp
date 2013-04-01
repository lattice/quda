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
    if(init){ 
      delete Ap;
      delete r;
    }
  }


  inline void accumulate_time(double* time_difference, const timeval& start, const timeval& stop)
  {
    long ds = stop.tv_sec - start.tv_sec;
    long dus = stop.tv_usec - start.tv_usec;
    *time_difference += ds + 0.000001*dus;
  }

  //void SimpleCG::operator()(cudaColorSpinorField& x, cudaColorSpinorField &b, double* time)
  void SimpleCG::operator()(cudaColorSpinorField& x, cudaColorSpinorField& b, cudaColorSpinorField& y, double* time)
  {
#ifdef PRECON_TIME
    timeval tstart, tstop;
    gettimeofday(&tstart, NULL);
#endif

    // profile[QUDA_PROFILE_INIT].Start();
    timeval mat_start, mat_stop;
    timeval start1, stop1;

    int k=0;



    // Find the maximum domain overlap.
    // This will determine the number of faces needed by the vector r.
    // Have to be very careful to ensure that setting the number of 
    // ghost faces here doesn't screw anything up further down the line.
    //    gettimeofday(&start1, NULL);

    if(!init){
      r = new cudaColorSpinorField(b);
      Ap = new cudaColorSpinorField(b); // I don't need to instantiate this every time either!
      init = true;                                  // There has to be a nicer way of doing this!
    }

    gettimeofday(&mat_start, NULL);
    mat(*r, x, y); // operator()(cudaColorSpinorField& out, cudaColorSpinorField& in,
                  // Switching to a zero source would get rid of this operation. 
                  // Will it affect the number of iterations
                  // => r = A*x;
    double r2 = xmyNormCuda(b,*r);

    cudaColorSpinorField p(*r); // I can also get rid of this initial copy
                               // by having a special case

    double alpha = 0.0, beta=0.0;
    double pAp;
    double r2_old;
    while( k < invParam.maxiter-1 ){
      gettimeofday(&mat_start, NULL);
      mat(*Ap, p, y);
      pAp = reDotProductCuda(p, *Ap);
      alpha = r2/pAp; 
      axpyCuda(-alpha, *Ap, *r); // r --> r - alpha*Ap
      r2_old = r2;
      r2 = norm2(*r);
      beta = r2/r2_old;
      axpyZpbxCuda(alpha, p, x, *r, beta);
      // x = x + alpha*p
      // p = r + beta*p
      ++k;
    }
    mat(*Ap, p, y);
    pAp = reDotProductCuda(p, *Ap);
    alpha  = r2/pAp;
    axpyCuda(alpha, p, x); // x --> x + alpha*p
#ifdef PRECON_TIME    
    cudaDeviceSynchronize();
    gettimeofday(&tstop, NULL);
    accumulate_time(&(time[2]), tstart, tstop);
#endif 
    return;
  }


} // namespace quda
