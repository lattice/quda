#ifndef _RITZ_QUDA_H
#define _RITZ_QUDA_H

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <dslash_quda.h>

#include <face_quda.h>
#include <blas_quda.h>
#include <dirac_quda.h>

#include <typeinfo>

namespace quda {

  class RitzMat {

    friend class DiracMatrix;

    protected:
    const DiracMatrix &dirac_mat; 
    int N_Poly;
    double shift;
    double *cheby_param;

    mutable cudaColorSpinorField *tmp1; // temporary hack
    mutable cudaColorSpinorField *tmp2; // temporary hack



    bool newTmp(cudaColorSpinorField **tmp, const cudaColorSpinorField &a) const{
      if (*tmp) return false;
      ColorSpinorParam param(a);
      param.create = QUDA_ZERO_FIELD_CREATE;
      *tmp = new cudaColorSpinorField(a, param);
      return true;
    }

    void deleteTmp(cudaColorSpinorField **a, const bool &reset) const{
      if (reset) {
        delete *a;
        *a = NULL;
      }
    }

    public:
    RitzMat(DiracMatrix &d, const QudaEigParam &param) 
      : dirac_mat(d), N_Poly(param.NPoly), cheby_param(param.MatPoly_param), shift(param.eigen_shift)
    {
      //   int bytes = N_Poly*sizeof(double);
      //   cheby_param = device_malloc(bytes);
      //   cudaMemcpy(cheby_param, param.MatPoly_params, bytes, cudaMemcpyHostToDevice);
    }
    RitzMat(DiracMatrix *d, const QudaEigParam &param)
      : dirac_mat(*d), N_Poly(param.NPoly), cheby_param(param.MatPoly_param), shift(param.eigen_shift)
    { 
      //   int bytes = N_Poly*sizeof(double);
      //   cheby_param = device_malloc(bytes);
      //   cudaMemcpy(cheby_param, param.MatPoly_params, bytes, cudaMemcpyHostToDevice);
    }
    virtual ~RitzMat();

    void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
    {
      const double alpha = pow(cheby_param[0], 2);
      const double beta  = pow(cheby_param[1]+fabs(shift), 2);

      const double c1 = 2.0*(alpha+beta)/(alpha-beta); 
      const double c0 = 2.0/(alpha+beta); 

      bool reset1 = newTmp( &tmp1, in);
      bool reset2 = newTmp( &tmp2, in);

      *(tmp2) = in;

      dirac_mat( *(tmp1), in);

      axpbyCuda(-0.5*c1, const_cast<cudaColorSpinorField&>(in), 0.5*c0*c1, *(tmp1));
      for(int i=2; i < N_Poly+1; ++i)
      {
        dirac_mat(out,*(tmp1));
        axpbyCuda(-c1,*(tmp1),c0*c1,out);
        axpyCuda(-1.0,*(tmp2),out);

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

    //    unsigned long long flops() const { return (dirac_mat->dirac)->Flops(); }

    //    std::string Type() const { return typeid(*(dirac_mat->dirac)).name(); }
  };

//  inline RitzMat::~RitzMat()
//  {
//
//  }

} // namespace quda

#endif // _RITZ_QUDA_H
