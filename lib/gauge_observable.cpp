#include <gauge_field.h>
#include <gauge_tools.h>
#include <gauge_path_quda.h>

namespace quda
{

  void gaugeObservables(GaugeField &u, QudaGaugeObservableParam &param, TimeProfile &profile)
  {
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
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
      double3 plaq = plaquette(u);
      param.plaquette[0] = plaq.x;
      param.plaquette[1] = plaq.y;
      param.plaquette[2] = plaq.z;
    }
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    if (param.compute_polyakov_loop) { gaugePolyakovLoop(param.ploop, u, 3, profile); }

    if (param.compute_gauge_loop_trace) {
      // wrap 1-d arrays in std::vector
      std::vector<int> path_length_v(param.num_paths);
      std::vector<double> loop_coeff_v(param.num_paths);
      for (int i = 0; i < param.num_paths; i++) {
        path_length_v[i] = param.path_length[i];
        loop_coeff_v[i] = param.loop_coeff[i];
      }

      // input_path should encode exactly 1 direction
      std::vector<int **> input_path_v(1);
      for (int d = 0; d < 1; d++) { input_path_v[d] = param.input_path_buff; }

      // prepare trace storage
      std::vector<Complex> loop_traces(param.num_paths);

      // actually do the computation
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
      gaugeLoopTrace(u, loop_traces, param.factor, input_path_v, path_length_v, loop_coeff_v, param.num_paths,
                     param.max_length);
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);

      for (int i = 0; i < param.num_paths; i++) { memcpy(param.traces + i, &loop_traces[i], sizeof(Complex)); }
    }

    // no point constructing Fmunu unless we are going to use it
    if (!param.compute_qcharge && !param.compute_qcharge_density) return;

    // create the Fmunu field
    profile.TPSTART(QUDA_PROFILE_INIT);
    // u is an extended field we need to shrink for the Fmunu field
    lat_dim_t x;
    for (int i = 0; i < 4; i++) x[i] = u.X()[i] - 2 * u.R()[i];
    GaugeFieldParam tensorParam(x, u.Precision(), QUDA_RECONSTRUCT_NO, 0, QUDA_TENSOR_GEOMETRY);
    tensorParam.location = QUDA_CUDA_FIELD_LOCATION;
    tensorParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    tensorParam.order = QUDA_FLOAT2_GAUGE_ORDER;
    tensorParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    cudaGaugeField gaugeFmunu(tensorParam);
    profile.TPSTOP(QUDA_PROFILE_INIT);

    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    computeFmunu(gaugeFmunu, u);
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTOP(QUDA_PROFILE_TOTAL);

    if (param.compute_qcharge || param.compute_qcharge_density) {
      profile.TPSTART(QUDA_PROFILE_TOTAL);
      profile.TPSTART(QUDA_PROFILE_INIT);
      if (param.compute_qcharge_density && !param.qcharge_density)
        errorQuda("Charge density requested, but destination field not defined");
      size_t size = gaugeFmunu.Volume() * gaugeFmunu.Precision();
      void *d_qDensity = param.compute_qcharge_density ? pool_device_malloc(size) : nullptr;
      profile.TPSTOP(QUDA_PROFILE_INIT);

      profile.TPSTART(QUDA_PROFILE_COMPUTE);

      if (param.compute_qcharge_density)
        computeQChargeDensity(param.energy, param.qcharge, d_qDensity, gaugeFmunu);
      else
        computeQCharge(param.energy, param.qcharge, gaugeFmunu);

      profile.TPSTOP(QUDA_PROFILE_COMPUTE);

      if (param.compute_qcharge_density) {
        profile.TPSTART(QUDA_PROFILE_D2H);
        qudaMemcpy(param.qcharge_density, d_qDensity, size, qudaMemcpyDeviceToHost);
        profile.TPSTOP(QUDA_PROFILE_D2H);

        profile.TPSTART(QUDA_PROFILE_FREE);
        pool_device_free(d_qDensity);
        profile.TPSTOP(QUDA_PROFILE_FREE);
      }
    }
  }

} // namespace quda
