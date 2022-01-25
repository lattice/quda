#pragma once

#include <invert_quda.h>
#include <accelerator.h>

#include <madwf_ml.h> // For MADWF

namespace quda
{
  /**
    @brief Create a preconditioning solver given the operators and parameters. Currently only a few solvers are
      instantiated, and only the MADWF accelerator is currently supported.
    @param[in] matPrecon the preconditioner
    @param[in] matEig the eigen-space operator that is to be used to construct the solver
    @param[in] param the outer solver param
    @param[in] Kparam the inner solver param
    @param[in] profile the timer profile
    @return the created preconditioning solver, decorated by std::shared_ptr
  */
  std::shared_ptr<Solver> create_preconditioner(const DiracMatrix &matPrecon, const DiracMatrix &matEig,
                                                SolverParam &param, SolverParam &Kparam, TimeProfile &profile)
  {
    Solver *K = nullptr;
    if (param.accelerator_type_precondition == QUDA_MADWF_ACCELERATOR) {
      if (param.inv_type_precondition == QUDA_CG_INVERTER) {
        K = new AcceleratedSolver<MadwfAcc, CG>(matPrecon, matPrecon, matPrecon, matEig, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_CA_CG_INVERTER) {
        K = new AcceleratedSolver<MadwfAcc, CACG>(matPrecon, matPrecon, matPrecon, matEig, Kparam, profile);
      } else { // unknown preconditioner
        errorQuda("Unknown inner solver %d for MADWF", param.inv_type_precondition);
      }
    } else {
      if (param.inv_type_precondition == QUDA_CG_INVERTER) {
        K = new CG(matPrecon, matPrecon, matPrecon, matEig, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_CA_CG_INVERTER) {
        K = new CACG(matPrecon, matPrecon, matPrecon, matEig, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_MR_INVERTER) {
        K = new MR(matPrecon, matPrecon, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_SD_INVERTER) {
        K = new SD(matPrecon, Kparam, profile);
      } else if (param.inv_type_precondition != QUDA_INVALID_INVERTER) { // unknown preconditioner
        errorQuda("Unknown inner solver %d", param.inv_type_precondition);
      }
    }
    return std::shared_ptr<Solver>(K);
  }

  /**
    @brief Wrap an external, existing, unmanaged preconditioner in a custom `std::shared_ptr` that
      doesn't deallocate when it falls out of scope. This is a temporary WAR for how MG solvers
      are managed and should be removed when they themselves are passed around via shared_ptr or
      potentially directly by reference.
      @param[in] K the externally allocated preconditioner
      @return the external preconditioner wrapped in a non-deallocating std::shared_ptr
   */
  std::shared_ptr<Solver> wrap_external_preconditioner(const Solver& K)
  {
    return std::shared_ptr<Solver>(&const_cast<Solver&>(K), [](Solver*) { });
  }
} // namespace quda
