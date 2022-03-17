#include <quda.h>
#include <timer.h>
#include <tune_quda.h>
#include <dslash_quda.h>

#include <gauge_force_quda.h>
#include <gauge_update_quda.h>

#include <unitarization_links.h>

// Forward declarations for profiling and parameter checking
// The helper functions are defined in interface_quda.cpp
TimeProfile &getProfileBLAS();
TimeProfile &getProfileInit();
TimeProfile &getProfileWilsonForce();
TimeProfile &getProfileCloverForce();
TimeProfile &getProfileGauss();
TimeProfile &getProfileGauge();
TimeProfile &getProfileMomAction();
TimeProfile &getProfileGaugeForce();
TimeProfile &getProfileGaugeUpdate();
bool &getQudaInitialised();

namespace quda {

  void computeLeapfrogTrajectory(cudaGaugeField& mom, cudaGaugeField& gauge, QudaHMCParam *hmc_param) {
    // HMC action parameters
    //-------------------------------------------------------------------------  
    // Define gauge action coefficients
    QudaGaugeActionType gauge_action_type = QUDA_GAUGE_ACTION_TYPE_WILSON;
    double path_coeff[3] = {1.0, 1.0, 1.0};

    // Define fermion actions
    int n_ferm_mono = hmc_param->n_fermion_monomials;
    std::vector<QudaHMCFermionActionType> fermion_action_types;
    fermion_action_types.reserve(n_ferm_mono);
    //for(int i = 0; i<n_ferm_mono; i++) fermion_action_types.push_back(hmc_param->fermion_action_types[i]);
    
    // Define step size
    double epsilon = (hmc_param->traj_length/(1.0*hmc_param->traj_steps));
  
    // DMH (3->Nc)
    double hmc_coeff = hmc_param->beta*epsilon/3.0;
    //-------------------------------------------------------------------------
    
    GaugeField *gauge_temp = createExtendedGauge(gauge, gauge.R(), getProfileGauge());
    
    // Create device force field
    GaugeFieldParam force_param(gauge);
    force_param.link_type = QUDA_GENERAL_LINKS;
    force_param.create = QUDA_ZERO_FIELD_CREATE;
    force_param.reconstruct = QUDA_RECONSTRUCT_NO;
    cudaGaugeField *force = new cudaGaugeField(force_param);
    
    for(int k=0; k<hmc_param->traj_steps; k++) {

      getProfileGaugeUpdate().TPSTART(QUDA_PROFILE_TOTAL);
      getProfileGaugeUpdate().TPSTART(QUDA_PROFILE_COMPUTE);
      updateGaugeField(*gauge_temp, 0.5*epsilon, gauge, mom, false, true);
      copyExtendedGauge(gauge, *gauge_temp, QUDA_CUDA_FIELD_LOCATION);
      QudaGaugeObservableParam gauge_obs_param = newQudaGaugeObservableParam();
      gauge_obs_param.compute_plaquette = QUDA_BOOLEAN_TRUE;
      gauge_obs_param.compute_qcharge = QUDA_BOOLEAN_TRUE;  
      
      // Measure gauge action and Q charge DMH(6->2*Nc)
      gaugeObservablesQuda(&gauge_obs_param);
      double gauge_action = 6.0 * (1.0 - gauge_obs_param.plaquette[0]) * gauge.Volume() * hmc_param->beta;
      printfQuda("Gauge action %d = %e\n", k, gauge_action);
      gauge.exchangeExtendedGhost(gauge.R(), false);
      getProfileGaugeUpdate().TPSTOP(QUDA_PROFILE_COMPUTE);
      getProfileGaugeUpdate().TPSTOP(QUDA_PROFILE_TOTAL);

      //int *num_failures_h = nullptr;
      //int *num_failures_d = nullptr;
      //quda::unitarizeLinks(gauge, num_failures_d); // unitarize on the gpu
      //if (*num_failures_h>0) errorQuda("Error in reunitarization: %d failures\n", *num_failures_h);

      getProfileGaugeForce().TPSTART(QUDA_PROFILE_TOTAL);
      getProfileGaugeForce().TPSTART(QUDA_PROFILE_COMPUTE);
      gaugeForceNew(mom, gauge, gauge_action_type, hmc_coeff, path_coeff);
      getProfileGaugeForce().TPSTOP(QUDA_PROFILE_COMPUTE);
      getProfileGaugeForce().TPSTOP(QUDA_PROFILE_TOTAL);
    
      //getProfileWilsonForce().TPSTART(QUDA_PROFILE_TOTAL);
      //getProfileWilsonForce().TPSTART(QUDA_PROFILE_COMPUTE);
      //computeFermionForce(*device_mom, *gaugeEvolved, hmc_param);
      //getProfileWilsonForce().TPSTOP(QUDA_PROFILE_COMPUTE);
      //getProfileWilsonForce().TPSTOP(QUDA_PROFILE_TOTAL);    
      getProfileGaugeUpdate().TPSTART(QUDA_PROFILE_TOTAL);
      getProfileGaugeUpdate().TPSTART(QUDA_PROFILE_COMPUTE);
      updateGaugeField(*gauge_temp, 0.5*epsilon, gauge, mom, false, true);
      copyExtendedGauge(gauge, *gauge_temp, QUDA_CUDA_FIELD_LOCATION);
      gauge.exchangeExtendedGhost(gauge.R(), false);
      getProfileGaugeUpdate().TPSTOP(QUDA_PROFILE_COMPUTE);
      getProfileGaugeUpdate().TPSTOP(QUDA_PROFILE_TOTAL);
    }
    delete force;
  }
}
