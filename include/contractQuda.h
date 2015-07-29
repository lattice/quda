#ifndef _CONTRACT_QUDA_H
#define _CONTRACT_QUDA_H

#include <quda_internal.h>
#include <quda.h>

namespace quda {
  void contractCuda(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type, const QudaParity parity, TimeProfile &profile);
  void contractCuda(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type, const int tSlice, const QudaParity parity, TimeProfile &profile);
  void covDev(cudaColorSpinorField *out, cudaGaugeField &gauge, const cudaColorSpinorField *in, const int parity, const int mu, TimeProfile &profile);

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
  
      void checkFullSpinor(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
      void checkSpinorAlias(const cudaColorSpinorField &, const cudaColorSpinorField &) const;
      void checkParitySpinor(const cudaColorSpinorField &out, const cudaColorSpinorField &in) const;
  
      void Apply(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity, const int mu);
      void M(cudaColorSpinorField &out, const cudaColorSpinorField &in, const int mu);
  
      unsigned long long Flops();
  };

}

#endif
