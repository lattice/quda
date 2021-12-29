#pragma once

#include <vector>
#include <random>
#include <unordered_map>

#include <color_spinor_field.h>
#include <blas_quda.h>
#include <madwf_transfer.h>
#include <polynomial.h>

#include <random_quda.h>
#include <tune_quda.h>

#include <device_vector.h>
#include <invert_quda.h>
#include <madwf_param.h>

namespace quda
{

  /**
    @brief A class for performing the Mobius Accelerated Domain Wall Fermion (MADWF) with the machine learning (ML)
    version, i.e. the transfer matrix between the vectors with different Ls are _trained_.
    - As indicated, the parameters, ultimately contained in `device_param`, need to be trained by calling the
    `train` method.
    - Once trained, the accelerated operator can be applied with the `apply` method.
  */
  struct MadwfAcc {

    using transfer_float = float;

    static constexpr madwf_ml::transfer_5D_t transfer_t = madwf_ml::transfer_5D_t::Spin;

    // The parameters to be trained.
    using device_container = device_vector<transfer_float>;

    // Has device_param been trained?
    bool trained = false;

  private:
    device_container device_param;

    MadwfParam param;

    double mu; // internal copy of the suppression factor

    // persistent buffers for reuse.
    std::unique_ptr<ColorSpinorField> forward_tmp;
    std::unique_ptr<ColorSpinorField> backward_tmp;

    // The underlying preconditioning precision
    QudaPrecision prec_precondition;

    // A static map that holds the cached parameters
    static std::unordered_map<std::string, std::vector<transfer_float>> host_training_param_cache; // empty map

    // Time profile
    TimeProfile &profile;

    /**
      @brief Fill a host vector with Gaussian random numbers.
      @param[in] v the host vector
    */
    void fill_random(std::vector<transfer_float> &v);

    /**
      @brief Return the chi squared of in the machine larning training.
      @param[in] ref the reference operator
      @param[in] base the base solver
      @param[out] out the output vector
      @param[in] in the input vector
    */
    double cost(const DiracMatrix &ref, Solver &base, ColorSpinorField &out, const ColorSpinorField &in);

  public:
    /**
      @brief constructor.
      @param[in] solve_param the standard solve_param
    */
    MadwfAcc(const SolverParam &solve_param, TimeProfile &profile);

    /**
      @brief Apply the (trained) parameter and perform the accelerated operator
      @param[in] base the solver to use
      @param[out] out the input vector
      @param[in] in the output vector

    */
    void apply(Solver &base, ColorSpinorField &out, const ColorSpinorField &in);

    /**
      @brief Train the parameters
      @param[in] ref the reference operator
      @param[in] base the base solver
      @param[in] null the solver used to generate null space vectors
      @param[in] in the input vector
      @param[in] tune_suppressor whether or not tune the mu suppression factor
    */
    void train(const DiracMatrix &ref, Solver &base, Solver &null, const ColorSpinorField &in,
               bool tune_suppressor = false);
  };

} // namespace quda
