#include <memory>

#include <invert_quda.h>
#include <invert_preconditioner.h>
#include <accelerator.h>

#include <madwf_ml.h> // For MADWF

namespace quda
{
  /**
    @brief Create a preconditioning solver given the operators and parameters. Currently only a few solvers are
      instantiated, and only the MADWF accelerator is currently supported.
    @param[in] matSloppy the sloppy matrix, which gets distinguished in GCR (FIXME necessary?)
    @param[in] matPrecon the preconditioner
    @param[in] matEig the eigen-space operator that is to be used to construct the solver
    @param[in] param the outer solver param
    @param[in] Kparam the inner solver param
    @param[in] profile the timer profile
    @return the created preconditioning solver, decorated by std::shared_ptr
  */
  std::shared_ptr<Solver> create_preconditioner(const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, const DiracMatrix &matEig,
                                                SolverParam &param, SolverParam &Kparam, TimeProfile &profile)
  {
    Solver *K = nullptr;
    if (param.accelerator_type_precondition == QUDA_MADWF_ACCELERATOR) {
      if (param.inv_type_precondition == QUDA_CG_INVERTER) {
        K = new AcceleratedSolver<MadwfAcc, CG>(matSloppy, matPrecon, matPrecon, matEig, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_CA_CG_INVERTER) {
        K = new AcceleratedSolver<MadwfAcc, CACG>(matSloppy, matPrecon, matPrecon, matEig, Kparam, profile);
      } else { // unknown preconditioner
        errorQuda("Unknown inner solver %d for MADWF", param.inv_type_precondition);
      }
    } else {
      if (param.inv_type_precondition == QUDA_CG_INVERTER) {
        K = new CG(matSloppy, matPrecon, matPrecon, matEig, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_CA_CG_INVERTER) {
        K = new CACG(matSloppy, matPrecon, matPrecon, matEig, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_MR_INVERTER) {
        K = new MR(matSloppy, matPrecon, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_SD_INVERTER) {
        K = new SD(matSloppy, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_CA_GCR_INVERTER) {
        K = new CAGCR(matSloppy, matPrecon, matPrecon, matEig, Kparam, profile);
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
