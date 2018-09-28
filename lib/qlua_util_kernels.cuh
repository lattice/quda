#ifndef QLUA_UTIL_KERNELS_H__
#define QLUA_UTIL_KERNELS_H__

#include <qlua_contract.h>

namespace quda {
  __global__ void phaseMatrix_kernel(complex<QUDA_REAL> *phaseMatrix, int *momMatrix,
				     MomProjArg *arg);
  __global__ void QluaSiteOrderCheck_kernel(QluaUtilArg *utilArg);
  __global__ void convertSiteOrder_QudaQDP_to_momProj_kernel(void *dst, const void *src,
							     QluaUtilArg *arg);
  __global__ void qcCopyCudaLink_kernel(Arg_CopyCudaLink *arg);
  __global__ void qcSetGaugeToUnity_kernel(Arg_SetUnityLink *arg);
} //- namespace quda

#endif //- QLUA_UTIL_KERNELS_H__
