#ifndef _CONTRACT_QUDA_H
#define _CONTRACT_QUDA_H

#include <quda_internal.h>
#include <quda.h>

namespace quda {
  void contractCuda(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type, const QudaParity parity, TimeProfile &profile);
  void contractCuda(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type, const int tSlice, const QudaParity parity, TimeProfile &profile);
  
  class Contract {

  protected:
    TimeProfile profile;
    
  public:
    Contract(TimeProfile &profile);
    ~Contract();
    
    void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y,
		      void *result, TimeProfile &profile);
  };
}

#endif
