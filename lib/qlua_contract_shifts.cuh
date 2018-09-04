/* C. Kallidonis: Header file for qlua_contract_shifts.cu
 * August 2018 
 */

#ifndef QLUA_CONTRACT_SHIFTS_H
#define QLUA_CONTRACT_SHIFTS_H

#include <qlua_contract.h>

namespace quda {

  __global__ void ShiftGauge_nonCov_kernel(Arg_ShiftGauge_nonCov *arg,
					   qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn);

  __global__ void ShiftLink_Cov_kernel(Arg_ShiftLink_Cov *arg,
				       qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn);

  __global__ void ShiftLink_AdjSplitCov_kernel(Arg_ShiftLink_AdjSplitCov *arg,
					       qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn);

  __global__ void ShiftCudaVec_nonCov_kernel(Arg_ShiftCudaVec_nonCov *arg,
                                             qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn);

  __global__ void ShiftCudaVec_Cov_kernel(Arg_ShiftCudaVec_Cov *arg,
					  qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn);

} //- namespace quda

#endif //- QLUA_CONTRACT_SHIFTS_H
