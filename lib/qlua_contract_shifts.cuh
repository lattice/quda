/* C. Kallidonis: Header file for qlua_contract_shifts.cu
 * August 2018 
 */

#ifndef QLUA_CONTRACT_SHIFTS_H
#define QLUA_CONTRACT_SHIFTS_H

#include <qlua_contract.h>

namespace quda {

  __device__ void ShiftVectorOnAxis_dev(Vector &shfVec, QluaCntrTMDArg *TMDarg, int ivec,
					qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType,
					int x_cb, int pty);

  __global__ void ShiftVectorOnAxis_kernel(QluaCntrTMDArg TMDarg, int ivec,
					   qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn, qcTMD_ShiftType shfType);


} //- namespace quda

#endif //- QLUA_CONTRACT_SHIFTS_H
