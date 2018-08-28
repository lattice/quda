/* C. Kallidonis: Header file for qlua_contract_shifts.cu
 * August 2018 
 */

#ifndef QLUA_CONTRACT_SHIFTS_H
#define QLUA_CONTRACT_SHIFTS_H

#include <qlua_contract.h>

namespace quda {

  __device__ void CovShiftPropPM1_dev(Vector *shfVec, QluaContractArg *arg, Propagator prop[],				      
				      int dir, qcCovShiftType shiftType,
				      int x_cb, int pty);

  __global__ void CovShiftPropPM1_kernel(QluaContractArg *arg,
					 int shfDir, qcCovShiftType shiftType);



  __device__ void NonCovShiftPropOnAxis_kernel_dev(Vector *shfVec, QluaContractArg *arg,
						   qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn,
						   int x_cb, int pty);

  __global__ void NonCovShiftPropOnAxis_kernel(QluaContractArg *arg,
					       qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn);



  __device__ void NonCovShiftVectorOnAxis_dev(Vector &shfVec, QluaCntrTMDArg *TMDarg, int ivec,
                                              qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn,
                                              int x_cb, int pty);

  __global__ void NonCovShiftVectorOnAxis_kernel(QluaCntrTMDArg *TMDarg, int ivec,
                                                 qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn);


} //- namespace quda

#endif //- QLUA_CONTRACT_SHIFTS_H
