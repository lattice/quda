#include <gauge_field.h>
#include <gauge_tools.h>
#include <gauge_path_quda.h>

namespace quda
{

  void gaugeObservables(GaugeField &u, QudaGaugeObservableParam &param)
  {
    auto &profile = getProfile();
    if (param.su_project) {
      int *num_failures_h = static_cast<int *>(pool_pinned_malloc(sizeof(int)));
      int *num_failures_d = static_cast<int *>(get_mapped_device_pointer(num_failures_h));
      *num_failures_h = 0;
      auto tol = u.Precision() == QUDA_DOUBLE_PRECISION ? 1e-14 : 1e-6;
      projectSU3(u, tol, num_failures_d);
      if (*num_failures_h > 0) errorQuda("Error in the SU(3) unitarization: %d failures\n", *num_failures_h);
      pool_pinned_free(num_failures_h);
    }

    if (param.compute_plaquette) {
      auto plaq = plaquette(u);
      param.plaquette[0] = double(plaq[0]);
      param.plaquette[1] = double(plaq[1]);
      param.plaquette[2] = double(plaq[2]);
    }

    if (param.compute_polyakov_loop) {
      auto ploop = gaugePolyakovLoop(u, 3, profile);
      param.ploop[0] = double(ploop[0]);
      param.ploop[1] = double(ploop[1]);
    }

    if (param.compute_gauge_loop_trace) {
      // wrap 1-d arrays in std::vector
      std::vector<int> path_length_v(param.num_paths);
      std::vector<real_t> loop_coeff_v(param.num_paths);
      for (int i = 0; i < param.num_paths; i++) {
        path_length_v[i] = param.path_length[i];
        loop_coeff_v[i] = param.loop_coeff[i];
      }

      // input_path should encode exactly 1 direction
      std::vector<int **> input_path_v(1);
      for (int d = 0; d < 1; d++) { input_path_v[d] = param.input_path_buff; }

      // prepare trace storage
      std::vector<complex_t> loop_traces(param.num_paths);

      // actually do the computation
      gaugeLoopTrace(u, loop_traces, param.factor, input_path_v, path_length_v, loop_coeff_v, param.num_paths,
                     param.max_length);

      for (int i = 0; i < param.num_paths; i++) { memcpy(param.traces + i, &loop_traces[i], sizeof(complex_t)); }
    }

    // no point constructing Fmunu unless we are going to use it
    if (!param.compute_qcharge && !param.compute_qcharge_density) return;

    // create the Fmunu field
    // u is an extended field we need to shrink for the Fmunu field
    lat_dim_t x;
    for (int i = 0; i < 4; i++) x[i] = u.X()[i] - 2 * u.R()[i];
    GaugeFieldParam tensorParam(x, u.Precision(), QUDA_RECONSTRUCT_NO, 0, QUDA_TENSOR_GEOMETRY);
    tensorParam.location = QUDA_CUDA_FIELD_LOCATION;
    tensorParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    tensorParam.order = QUDA_FLOAT2_GAUGE_ORDER;
    tensorParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    GaugeField gaugeFmunu(tensorParam);

    computeFmunu(gaugeFmunu, u);

    if (param.compute_qcharge || param.compute_qcharge_density) {
      profile.TPSTART(QUDA_PROFILE_INIT);
      if (param.compute_qcharge_density && !param.qcharge_density)
        errorQuda("Charge density requested, but destination field not defined");
      size_t size = gaugeFmunu.Volume() * gaugeFmunu.Precision();
      void *d_qDensity = param.compute_qcharge_density ? pool_device_malloc(size) : nullptr;
      profile.TPSTOP(QUDA_PROFILE_INIT);

      array<real_t, 3> energy;
      real_t qcharge;
      if (param.compute_qcharge_density)
        qcharge = computeQChargeDensity(energy, d_qDensity, gaugeFmunu);
      else
        qcharge = computeQCharge(energy, gaugeFmunu);
      for (int i = 0; i < 3; i++) param.energy[i] = double(energy[i]);
      param.qcharge = double(qcharge);

      if (param.compute_qcharge_density) {
        profile.TPSTART(QUDA_PROFILE_D2H);
        qudaMemcpy(param.qcharge_density, d_qDensity, size, qudaMemcpyDeviceToHost);
        profile.TPSTOP(QUDA_PROFILE_D2H);

        pool_device_free(d_qDensity);
      }
    }
  }

} // namespace quda
