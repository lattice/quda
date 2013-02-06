#include <quda_internal.h>
#include <invert_quda.h>

namespace quda {

  static void report(const char *type) {
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating a %s solver\n", type);
  }

  // solver factory
  Solver* Solver::create(QudaInvertParam &param, DiracMatrix &mat, DiracMatrix &matSloppy,
			 DiracMatrix &matPrecon, TimeProfile &profile)
  {
    Solver *solver=0;

    switch (param.inv_type) {
    case QUDA_PCG_INVERTER:
      report("PCG");
      solver = new PreconCG(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_CG_INVERTER:
      report("CG");
      solver = new CG(mat, matSloppy, param, profile);
      break;
    case QUDA_BICGSTAB_INVERTER:
      report("BiCGstab");
      solver = new BiCGstab(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_GCR_INVERTER:
      report("GCR");
      solver = new GCR(mat, matSloppy, matPrecon, param, profile);
      break;
    case QUDA_MR_INVERTER:
      report("MR");
      solver = new MR(mat, param, profile);
      break;
    default:
      errorQuda("Invalid solver type");
    }
    
    return solver;
  }

} // namespace quda
