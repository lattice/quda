#ifndef _CONTRACT_QUDA_H
#define _CONTRACT_QUDA_H

#include <quda_internal.h>
#include <quda.h>

namespace quda {
  void contractCuda(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type, const QudaParity parity, TimeProfile &profile);
  void contractCuda(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type, const int tSlice, const QudaParity parity, TimeProfile &profile);
  void gamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in);
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

  /**
   * Compute a volume or time-slice contraction of two spinors.
   * @param x     Spinor to contract. This is conjugated before contraction.
   * @param y     Spinor to contract.
   * @param ctrn  Contraction output. The size must be Volume*16
   * @param cType Contraction type, allows for volume or time-slice contractions.
   * @param tC    Time-slice to contract in case the contraction is in a single time-slice.
   */
  void contract(const cudaColorSpinorField x, const cudaColorSpinorField y, void *ctrn, const QudaContractType cType);

  void contract(const cudaColorSpinorField x, const cudaColorSpinorField y, void *ctrn, const QudaContractType cType, const int tC);

}

/* The interfaces are not mature enough to be included in the master branch at this moment */
/*
void loopPlainCG(void *hp_x, void *hp_b, QudaInvertParam *param, void *ct, void *cDgv[4]);
void loopHPECG(void *hp_x, void *hp_b, QudaInvertParam *param, void *ct, void *cDgv[4]);
void oneEndTrickCG(void *hp_x, void *hp_b, QudaInvertParam *param, void *ct_gv, void *ct_vv, void *cDgv[4], void *cDvv[4], void *cCgv[4], void *cCvv[4], void *cXgv[4], void *cXvv[4]);
void tuneOneEndTrick(void *hp_x, void *hp_b, QudaInvertParam *param, void ***cnRes_gv, void ***cnRs2_gv, void **cnRes_vv, void **cnRs2_vv,
                     const int nSteps, const bool Cr, void ***cnCor_gv, void ***cnCr2_gv, void **cnCor_vv, void **cnCr2_vv);
void tDilHPECG(void *hp_x, void *hp_b, QudaInvertParam *param, void **cnRes, const int tSlice, const int nCoh, const int HPE);
*/

#endif
