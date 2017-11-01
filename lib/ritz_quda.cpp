#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <lanczos_quda.h>
#include <ritz_quda.h>

namespace quda {

  void RitzMat::operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    using namespace blas;
    
    const double alpha = pow(cheby_param[0], 2);
    const double beta  = pow(cheby_param[1]+fabs(shift), 2);

    const double c1 = 2.0*(alpha+beta)/(alpha-beta); 
    const double c0 = 2.0/(alpha+beta); 

    bool reset1 = newTmp( &tmp1, in);
    bool reset2 = newTmp( &tmp2, in);

    *(tmp2) = in;
    dirac_mat( *(tmp1), in);

    axpby(-0.5*c1, const_cast<cudaColorSpinorField&>(in), 0.5*c0*c1, *(tmp1));
    for(int i=2; i < N_Poly+1; ++i)
    {
      dirac_mat(out,*(tmp1));
      axpby(-c1,*(tmp1),c0*c1,out);
      axpy(-1.0,*(tmp2),out);
      //printfQuda("ritzMat: Ritz mat loop %d\n",i);

      if(i != N_Poly)
      {
        // tmp2 = tmp
        // tmp = out
        cudaColorSpinorField *swap_Tmp = tmp2;
        tmp2 = tmp1;
        tmp1 = swap_Tmp;
        *(tmp1) = out;
      }
    }
    deleteTmp(&(tmp1), reset1);
    deleteTmp(&(tmp2), reset2);

  }
  RitzMat::~RitzMat() {;}
  bool RitzMat::newTmp(cudaColorSpinorField **tmp, const cudaColorSpinorField &a) const{
    if (*tmp) return false;
    ColorSpinorParam param(a);
    param.create = QUDA_ZERO_FIELD_CREATE;
    *tmp = new cudaColorSpinorField(a, param);
    return true;
  }

  void RitzMat::deleteTmp(cudaColorSpinorField **a, const bool &reset) const{
    if (reset) {
      delete *a;
      *a = NULL;
    }
  }
}
