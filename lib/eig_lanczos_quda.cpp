#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>
#include <lanczos_quda.h>

#include <iostream>

namespace quda {

  Lanczos::Lanczos(RitzMat &ritz_mat, QudaEigParam &eigParam, TimeProfile &profile) :
    Eig_Solver(eigParam, profile), ritz_mat(ritz_mat)
  {

  }

  Lanczos::~Lanczos() 
  {

  }

  void Lanczos::operator()(double *alpha, double *beta, cudaColorSpinorField **Eig_Vec, cudaColorSpinorField &r, cudaColorSpinorField &Apsi, int k0, int m) 
  {
    using namespace blas;

    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // Check to see that we'reV not trying to invert on a zero-field source    
    const double b2 = norm2(r);
    if(b2 == 0){
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      printfQuda("Warning: initial residual is already zero\n");
      return;
    }

    double ff;
    ff = sqrt(norm2(r));
    zero(*(Eig_Vec[k0]));
    axpy(1.0/ff, r, *(Eig_Vec[k0]));
  
    for (int k = k0; k < m; ++k)
    {
      if( k == 0 )
      {
        // r_k = A*v_k , r_k is used for temporary buffer.
        ritz_mat(r, *(Eig_Vec[0]));
        // alpha_k = ( v_k , r_k )
        alpha[0] = reDotProduct(*(Eig_Vec[0]), r);
        // r_k = (A - alpha_k)v_k - beta_{k-1}*v_{k-1} = r_k - alpha_k*v_k
        // k = 1 case, beta_0 is defined to 0, so we don`t need beta related term.
        axpy(-alpha[0],*(Eig_Vec[0]), r);
        // beta_k = ||r_k||
        beta[0] = sqrt(norm2(r));


        zero(*(Eig_Vec[1]));
        axpy(1.0/beta[0], r, *(Eig_Vec[1]));

      }
      else
      {
        // r_k = A*v_k , r_k is used for temporary buffer.
        ritz_mat(r, *(Eig_Vec[k]));
        // r_k = (A - alpha_k)v_k - beta_{k-1}*v_{k-1} = r_k - alpha_k*v_k
        // 1st: r_k = r_k - beta_{k-1}*v_{k-1}
        // 2nd: alpha = (v_k, A*v_k) = (v_k , r_k)
        // 3rd: r_k = r_k - alpha_k*v_k
        axpy(-beta[k-1],*(Eig_Vec[k-1]), r);
        alpha[k] = reDotProduct(*(Eig_Vec[k]), r);
        axpy(-alpha[k],*(Eig_Vec[k]), r);
        // beta_k = ||r_k||
        beta[k] = sqrt(norm2(r));

        GrandSchm_test(r, Eig_Vec, k+1, 0);
      
        if(k+1 < m)
        {
          zero(*(Eig_Vec[k+1]));
          axpy(1.0/beta[k], r, *(Eig_Vec[k+1]));
        }
      }
    }
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    return;
  }
  
#if 0
  ImpRstLanczos::ImpRstLanczos(RitzMat &ritz_mat, QudaEigParam &eigParam, TimeProfile &profile) :
    Eig_Solver(eigParam, profile), ritz_mat(ritz_mat)
  { }

  ImpRstLanczos::~ImpRstLanczos()
  { }

  void ImpRstLanczos::operator()(double *alpha, double *beta, cudaColorSpinorField **Eig_Vec,
				 cudaColorSpinorField &r, cudaColorSpinorField &Apsi, int k0, int m)
  { }
#endif

} // namespace quda
