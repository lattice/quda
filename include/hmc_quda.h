#pragma once

namespace quda {

  void computeLeapfrogTrajectory(cudaGaugeField& mom, cudaGaugeField& gauge, QudaHMCParam *hmc_param);
  
} // namespace quda
