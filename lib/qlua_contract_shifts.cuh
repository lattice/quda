/* C. Kallidonis: Header file for qlua_contract_shifts.cu
 * August 2018 
 */

#ifndef QLUA_CONTRACT_SHIFTS_H
#define QLUA_CONTRACT_SHIFTS_H

#include <qlua_contract.h>

namespace quda {

  __global__ void CovShiftDevicePropPM1(QluaContractArg *arg,
                                        Vector *outShf, Propagator prop[],
                                        int dir, qcShiftType shiftType);

}

#endif //- QLUA_CONTRACT_SHIFTS_H
