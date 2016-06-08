#ifndef _CONTRACT_QUDA_H
#define _CONTRACT_QUDA_H

#include <quda_internal.h>
#include <quda.h>

namespace quda {
  void contractCuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, const QudaContractType contract_type, const QudaParity parity, TimeProfile &profile);
  void contractCuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, const QudaContractType contract_type, const int tSlice, const QudaParity parity, TimeProfile &profile);
  void covDev(ColorSpinorField *out, cudaGaugeField &gauge, const ColorSpinorField *in, const int parity, const int mu, TimeProfile &profile);

  class CovD {
    protected:
      cudaGaugeField *gauge;
      unsigned long long flops;
  
      QudaTune tune;
  
      int commDim[QUDA_MAX_DIM]; // whether do comms or not
  
      TimeProfile *profile;
  
    public:
      CovD(cudaGaugeField *gauge, TimeProfile &profile);
      ~CovD();
      CovD& operator=(const CovD &cov);
  
      void checkFullSpinor(const ColorSpinorField &, const ColorSpinorField &) const;
      void checkSpinorAlias(const ColorSpinorField &, const ColorSpinorField &) const;
      void checkParitySpinor(const ColorSpinorField &out, const ColorSpinorField &in) const;
  
      void Apply(ColorSpinorField &out, const ColorSpinorField &in, const QudaParity parity, const int mu);
      void M(ColorSpinorField &out, const ColorSpinorField &in, const int mu);
  
      unsigned long long Flops();
  };

}

#endif
