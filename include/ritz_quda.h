#ifndef _RITZ_QUDA_H
#define _RITZ_QUDA_H

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <dslash_quda.h>
#include <blas_quda.h>
#include <dirac_quda.h>
#include <typeinfo>

namespace quda {

  /**
   Ritz matrix is targeted matrix object what we want to calculate its eigen values and eigen vectors.
   In lattice QCD application, this is normally Dirac operator
   */
  class RitzMat {

    friend class DiracMatrix;

    protected:
    const DiracMatrix &dirac_mat; 
    int N_Poly;  // Chebychev polynomial order
    double shift;   // eigen shift offset 
    double *cheby_param;  // Chebychev polynomial coefficients values

    mutable cudaColorSpinorField *tmp1; // temporary hack
    mutable cudaColorSpinorField *tmp2; // temporary hack

    bool newTmp(cudaColorSpinorField **tmp, const cudaColorSpinorField &a) const;
    void deleteTmp(cudaColorSpinorField **a, const bool &reset) const;

    public:
    RitzMat(DiracMatrix &d, const QudaEigParam &param) 
      : dirac_mat(d), N_Poly(param.NPoly), shift(param.eigen_shift), 
        cheby_param(param.MatPoly_param), tmp1(NULL), tmp2(NULL)
    {;}
    RitzMat(DiracMatrix *d, const QudaEigParam &param)
      : dirac_mat(*d), N_Poly(param.NPoly), shift(param.eigen_shift),
        cheby_param(param.MatPoly_param), tmp1(NULL), tmp2(NULL)
    {;}
    virtual ~RitzMat();

    void operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const;

    //    unsigned long long flops() const { return (dirac_mat->dirac)->Flops(); }

    //    std::string Type() const { return typeid(*(dirac_mat->dirac)).name(); }
  };

} // namespace quda

#endif // _RITZ_QUDA_H
