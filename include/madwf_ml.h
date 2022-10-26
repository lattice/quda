#pragma once

#include <vector>
#include <random>
#include <unordered_map>

#include <color_spinor_field.h>
#include <blas_quda.h>
#include <polynomial.h>

#include <random_quda.h>
#include <tune_quda.h>

#include <device_vector.h>
#include <invert_quda.h>
#include <madwf_param.h>

namespace quda
{
  namespace madwf_ml
  {
    /**
      @brief defines the types of 5th dimension transfer matrices:
      - Wilson: has both the color and spin indices, d.o.f = (4 x 3) * (4 x 3) = 144
      - Spin: only has the spin index, d.o.f = (4) * (4) = 16
      - Chiral: only has the chiral index, d.o.f = 2
    */
    enum class transfer_5D_t { Wilson = 144, Spin = 16, Chiral = 2 };

    /**
      @brief A helper class to instantiate the precisions (half and quarter) for madwf_tensor and madwf_transfer
    */
    template <template <class T> class F, class... Args>
    void instantiate_madwf(ColorSpinorField &out, const ColorSpinorField &in, Args... args)
    {
      switch (checkPrecision(out, in)) {
      case QUDA_HALF_PRECISION: {
        if constexpr (static_cast<bool>(QUDA_PRECISION & 2)) {
          F<short> w(out, in, args...);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
        }
      } break;
      case QUDA_QUARTER_PRECISION: {
        if constexpr (static_cast<bool>(QUDA_PRECISION & 1)) {
          F<int8_t> w(out, in, args...);
        } else {
          errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
        }
      } break;
      default: errorQuda("Unsupported precision %d", in.Precision());
      }
    }
  } // namespace madwf_ml

  /**
    @brief A class for performing the Mobius Accelerated Domain Wall Fermion (MADWF) with the machine learning (ML)
    version, i.e. the transfer matrix between the vectors with different Ls are _trained_.
    - As indicated, the parameters, ultimately contained in `device_param`, need to be trained by calling the
    `train` method.
    - Once trained, the accelerated operator can be applied with the `apply` method.
    @param transfer_float the floating-point type we use for the transfer matrix
    @param trained whether this object has been trained or not
    @param mu internal copy of the suppression factor
    @param forward_tmp, backward_tmp persistent ColorSpinorField buffers for reuse
    @param prec_precondition the underlying preconditioning precision
    @param host_training_param_cache a static map that holds the cached parameters, so we do not need to load the same
      set of parameters multiple times from disk
  */
  struct MadwfAcc {

    using transfer_float = float;

    static constexpr madwf_ml::transfer_5D_t transfer_t = madwf_ml::transfer_5D_t::Spin;

    using device_container = device_vector<transfer_float>;

    bool trained = false;

  private:
    device_container device_param;

    MadwfParam param;

    double mu;

    ColorSpinorField forward_tmp;
    ColorSpinorField backward_tmp;

    QudaPrecision prec_precondition;

    static std::unordered_map<std::string, std::vector<transfer_float>> host_training_param_cache; // empty map

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

    /**
      @brief Save the current parameter to disk
      @param[in] Ls the original (larger) 5th dimension size
      @param[in] Ls_base the reduced 5th dimension size
    */
    void save_parameter(int Ls, int Ls_base);

    /**
      @brief Load the parameter from disk
      @param[in] Ls the original (larger) 5th dimension size
      @param[in] Ls_base the reduced 5th dimension size
    */
    void load_parameter(int Ls, int Ls_base);

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
