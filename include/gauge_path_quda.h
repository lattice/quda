#pragma once

namespace quda
{

  /**
     @brief Compute the gauge-force contribution to the momentum
     @param[out] mom Momentum field
     @param[in] u Gauge field (extended when running on multiple GPUs)
     @param[in] coeff Step-size coefficient
     @param[in] input_path Host-array holding all path contributions for the gauge action
     @param[in] length Host array holding the length of all paths
     @param[in] path_coeff Coefficient of each path
     @param[in] num_paths Numer of paths
     @param[in] max_length Maximum length of each path
   */
  void gaugeForce(GaugeField &mom, const GaugeField &u, double coeff, std::vector<int **> &input_path,
                  std::vector<int> &length, std::vector<double> &path_coeff, int num_paths, int max_length);

  /**
     @brief Compute the product of gauge-links along the given path
     @param[out] out Gauge field which the result is added to
     @param[in] u Gauge field (extended when running on multiple GPUs)
     @param[in] coeff Global coefficient for the result
     @param[in] input_path Host-array holding all path contributions
     @param[in] length Host array holding the length of all paths
     @param[in] path_coeff Coefficient of each path
     @param[in] num_paths Numer of paths
     @param[in] max_length Maximum length of each path
   */
  void gaugePath(GaugeField &out, const GaugeField &u, double coeff, std::vector<int **> &input_path,
                 std::vector<int> &length, std::vector<double> &path_coeff, int num_paths, int max_length);

  /**
     @brief Compute the trace of an arbitrary set of gauge loops
     @param[in] u Gauge field (extended when running on multiple GPUs)
     @param[in, out] loop_traces Output traces of loops
     @param[in] input_path Host-array holding all path contributions for the gauge action
     @param[in] factor Multiplicative factor for each loop (i.e., volume normalization, etc)
     @param[in] length Host array holding the length of all paths
     @param[in] path_coeff Coefficient of each path
     @param[in] num_paths Numer of paths
     @param[in] path_max_length Maximum length of each path
   */
  void gaugeLoopTrace(const GaugeField &u, std::vector<Complex> &loop_traces, double factor,
                      std::vector<int **> &input_path, std::vector<int> &length, std::vector<double> &path_coeff_h,
                      int num_paths, int path_max_length);

} // namespace quda
