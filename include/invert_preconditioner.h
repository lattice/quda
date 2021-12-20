#pragma once

#include <invert_quda.h>
#include <accelerator.h>

#include <madwf_ml.h> // For MADWF

namespace quda
{
  std::unique_ptr<Solver> create_preconditioner(const DiracMatrix &matPrecon, const DiracMatrix &matEig, SolverParam &param, SolverParam &Kparam, TimeProfile &profile)
  {
    Solver *K = nullptr;
    if (param.schwarz_type == QUDA_ADDITIVE_MADWF_SCHWARZ) {
      if (param.inv_type_precondition == QUDA_CG_INVERTER) {
        K = new AcceleratedSolver<MadwfAcc, CG>(matPrecon, matPrecon, matPrecon, matEig, Kparam, profile);
      } else { // unknown preconditioner
        errorQuda("Unknown inner solver %d for MADWF", param.inv_type_precondition);
      }
    } else {
      if (param.inv_type_precondition == QUDA_CG_INVERTER) {
        K = new CG(matPrecon, matPrecon, matPrecon, matEig, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_MR_INVERTER) {
        K = new MR(matPrecon, matPrecon, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_SD_INVERTER) {
        K = new SD(matPrecon, Kparam, profile);
      } else if (param.inv_type_precondition != QUDA_INVALID_INVERTER) { // unknown preconditioner
        errorQuda("Unknown inner solver %d", param.inv_type_precondition);
      }
    }
    return std::unique_ptr<Solver>(K);
  }
}

