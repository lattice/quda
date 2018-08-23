/* C. Kallidonis: Header file for qlua_contract_shifts.cu
 * August 2018 
 */

#ifndef QLUA_CONTRACT_SHIFTS_H
#define QLUA_CONTRACT_SHIFTS_H

#include <qlua_contract.h>

namespace quda {

  __device__ void CovShiftPropPM1_dev(QluaContractArg *arg,
				      Vector *outShf, Propagator prop[],
				      int x_cb, int pty,
				      int dir, qcCovShiftType shiftType);

  __global__ void CovShiftPropPM1_kernel(QluaContractArg *arg, QluaAuxCntrArg *auxArg,
					 int shfDir, qcCovShiftType shiftType);



  __device__ void NonCovShiftPropOnAxis_kernel_dev(QluaContractArg *arg, Vector *outShf,
						   int x_cb, int pty,
						   qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn);

  __global__ void NonCovShiftPropOnAxis_kernel(QluaContractArg *arg, QluaAuxCntrArg *auxArg,
					       qcTMD_ShiftDir shfDir, qcTMD_ShiftSgn shfSgn);

} //- namespace quda

#endif //- QLUA_CONTRACT_SHIFTS_H
