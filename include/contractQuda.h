#ifndef _CONTRACT_QUDA_H
#define _CONTRACT_QUDA_H

#include <quda_internal.h>
#include <quda.h>

namespace quda
{

	void	contractCuda		(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const int sign, const dim3 &blockDim, int *XS, const int Parity);
	void	contractGamma5Cuda	(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const int sign, const dim3 &blockDim, int *XS, const int Parity);
	void	contractTsliceCuda	(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const int sign, const dim3 &blockDim, int *XS, const int tslice, const int Parity);
	void	gamma5Cuda		(cudaColorSpinorField *out, const cudaColorSpinorField *in, const dim3 &block);
	void	covDevQuda		(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in, const int parity, const int mu, const int *commOverride);
}

	void	loopPlainCG	(void *hp_x, void *hp_b, QudaInvertParam *param, void *ct, void *cDgv[4]);
	void	loopHPECG	(void *hp_x, void *hp_b, QudaInvertParam *param, void *ct, void *cDgv[4]);
	void	oneEndTrickCG	(void *hp_x, void *hp_b, QudaInvertParam *param, void *ct_gv, void *ct_vv, void *cDgv[4], void *cDvv[4], void *cCgv[4], void *cCvv[4]);
	void	tDilutionCG	(void *hp_x, void *hp_b, QudaInvertParam *param, void **cnRes, const int tSlice, const int nCoh);
	void	tDilHPECG	(void *hp_x, void *hp_b, QudaInvertParam *param, void **cnRes, const int tSlice, const int nCoh);

#endif
