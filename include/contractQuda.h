#ifndef _CONTRACT_QUDA_H
#define _CONTRACT_QUDA_H

#include <quda_internal.h>
#include <quda.h>

namespace quda {
  void contractCuda(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type, const QudaParity parity);
  void contractCuda(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const QudaContractType contract_type, const int tSlice, const QudaParity parity);
  void gamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in);
  void covDev(cudaColorSpinorField *out, cudaGaugeField &gauge, const cudaColorSpinorField *in, const int parity, const int mu, TimeProfile &profile);
}

/* The interfaces are not mature enough to be included in the master branch at this moment */
/*
void loopPlainCG(void *hp_x, void *hp_b, QudaInvertParam *param, void *ct, void *cDgv[4]);
void loopHPECG(void *hp_x, void *hp_b, QudaInvertParam *param, void *ct, void *cDgv[4]);
void oneEndTrickCG(void *hp_x, void *hp_b, QudaInvertParam *param, void *ct_gv, void *ct_vv, void *cDgv[4], void *cDvv[4], void *cCgv[4], void *cCvv[4]);
void tuneOneEndTrick(void *hp_x, void *hp_b, QudaInvertParam *param, void ***cnRes_gv, void ***cnRs2_gv, void **cnRes_vv, void **cnRs2_vv,
                     const int nSteps, const bool Cr, void ***cnCor_gv, void ***cnCr2_gv, void **cnCor_vv, void **cnCr2_vv);
void tDilutionCG(void *hp_x, void *hp_b, QudaInvertParam *param, void **cnRes, const int tSlice, const int nCoh);
void tDilHPECG(void *hp_x, void *hp_b, QudaInvertParam *param, void **cnRes, const int tSlice, const int nCoh);
*/
#endif
