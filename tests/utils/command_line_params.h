#pragma once

#include <CLI11.hpp>
#include <array>
#include <quda.h>
#include "misc.h"

// for compatibility while porting - remove later
extern void usage(char **);

// Put this is quda_constants.h?
#define QUDA_MAX_SOURCES 128

// Put this is quda_constants.h?
#define QUDA_MAX_MASSES 128

namespace quda
{
  template <typename T> using mgarray = std::array<T, QUDA_MAX_MG_LEVEL>;
  template <typename T> using file_array = std::array<T, QUDA_MAX_SOURCES>;
  template <typename T> using source_array = std::array<T, QUDA_MAX_SOURCES>;
  template <typename T> using mass_array = std::array<T, QUDA_MAX_MASSES>;
}

class QUDAApp : public CLI::App
{

public:
  QUDAApp(std::string app_description = "", std::string app_name = "") : CLI::App(app_description, app_name) {};

  virtual ~QUDAApp() {};

  template <typename T>
  CLI::Option *add_mgoption(std::string option_name, std::array<T, QUDA_MAX_MG_LEVEL> &variable, CLI::Validator trans,
                            std::string option_description = "", bool = false)
  {

    CLI::callback_t f = [&variable, &option_name, trans](CLI::results_t vals) {
      size_t l;
      T j; // results_t is just a vector of strings
      bool worked = true;

      CLI::Range validlevel(0, QUDA_MAX_MG_LEVEL);
      for (size_t i {0}; i < vals.size() / 2; ++i) { // will always be a multiple of 2
        auto levelok = validlevel(vals.at(2 * i));
        auto transformok = trans(vals.at(2 * i + 1));
        if (!levelok.empty()) throw CLI::ValidationError(option_name, levelok);
        if (!transformok.empty()) throw CLI::ValidationError(option_name, transformok);
        worked = worked and CLI::detail::lexical_cast(vals.at(2 * i), l);
        worked = worked and CLI::detail::lexical_cast(vals.at(2 * i + 1), j);

        if (worked) variable[l] = j;
      }
      return worked;
    };
    CLI::Option *opt = add_option(option_name, f, option_description);
    auto valuename = std::string("LEVEL ") + std::string(CLI::detail::type_name<T>());
    opt->type_name(valuename)->type_size(-2);
    opt->expected(-1);
    opt->check(CLI::Validator(trans.get_description()));
    // opt->transform(trans);
    // opt->default_str("");

    return opt;
  }

  template <typename T>
  CLI::Option *add_mgoption(CLI::Option_group *group, std::string option_name, std::array<T, QUDA_MAX_MG_LEVEL> &variable,
                            CLI::Validator trans, std::string option_description = "", bool = false)
  {

    CLI::callback_t f = [&variable, &option_name, trans](CLI::results_t vals) {
      size_t l;
      // T j; // results_t is just a vector of strings
      bool worked = true;

      CLI::Range validlevel(0, QUDA_MAX_MG_LEVEL);
      for (size_t i {0}; i < vals.size() / 2; ++i) { // will always be a multiple of 2
        auto levelok = validlevel(vals.at(2 * i));
        auto transformok = trans(vals.at(2 * i + 1));
        if (!levelok.empty()) throw CLI::ValidationError(option_name, levelok);
        if (!transformok.empty()) throw CLI::ValidationError(option_name, transformok);
        worked = worked and CLI::detail::lexical_cast(vals.at(2 * i), l);
        auto &j = variable[l];
        worked = worked and CLI::detail::lexical_cast(vals.at(2 * i + 1), j);

        // if (worked) variable[l] = j;
      }
      return worked;
    };
    CLI::Option *opt = add_option(option_name, f, option_description);
    auto valuename = std::string("LEVEL ") + std::string(CLI::detail::type_name<T>());
    opt->type_name(valuename)->type_size(-2);
    opt->expected(-1);
    opt->check(CLI::Validator(trans.get_description()));
    // opt->transform(trans);
    // opt->default_str("");
    group->add_option(opt);
    return opt;
  }

  template <typename T>
  CLI::Option *add_mgoption(CLI::Option_group *group, std::string option_name,
                            std::array<std::array<T, 4>, QUDA_MAX_MG_LEVEL> &variable, CLI::Validator trans,
                            std::string option_description = "", bool = false)
  {

    CLI::callback_t f = [&variable, &option_name, trans](CLI::results_t vals) {
      size_t l;
      T j; // results_t is just a vector of strings
      bool worked = true;

      CLI::Range validlevel(0, QUDA_MAX_MG_LEVEL);
      for (size_t i {0}; i < vals.size() / (4 + 1); ++i) {
        auto levelok = validlevel(vals.at((4 + 1) * i));

        if (!levelok.empty()) throw CLI::ValidationError(option_name, levelok);
        worked = worked and CLI::detail::lexical_cast(vals.at((4 + 1) * i), l);

        for (int k = 0; k < 4; k++) {
          auto transformok = trans(vals.at((4 + 1) * i + k + 1));
          if (!transformok.empty()) throw CLI::ValidationError(option_name, transformok);
          worked = worked and CLI::detail::lexical_cast(vals.at((4 + 1) * i + k + 1), j);
          if (worked) variable[l][k] = j;
        }
      }
      return worked;
    };
    CLI::Option *opt = add_option(option_name, f, option_description);
    auto valuename = std::string("LEVEL ") + std::string(CLI::detail::type_name<T>());
    opt->type_name(valuename)->type_size(-4 - 1);
    opt->expected(-1);
    opt->check(CLI::Validator(trans.get_description()));
    // opt->transform(trans);
    // opt->default_str("");
    group->add_option(opt);
    return opt;
  }

  // Add option to parse multiple point source locations
  template <typename T>
  CLI::Option *add_psoption(CLI::Option_group *group, std::string option_name,
                            std::array<std::array<T, 4>, QUDA_MAX_SOURCES> &variable, CLI::Validator trans,
                            std::string option_description = "")
  {

    CLI::callback_t f = [&variable, &option_name, trans](CLI::results_t vals) {
      size_t l;
      T j; // results_t is just a vector of strings
      bool worked = true;

      CLI::Range validsource(0, QUDA_MAX_SOURCES);
      for (size_t i {0}; i < vals.size() / (4 + 1); ++i) {
        auto sourceok = validsource(vals.at((4 + 1) * i));

        if (!sourceok.empty()) throw CLI::ValidationError(option_name, sourceok);
        worked = worked and CLI::detail::lexical_cast(vals.at((4 + 1) * i), l);

        for (int k = 0; k < 4; k++) {
          auto transformok = trans(vals.at((4 + 1) * i + k + 1));
          if (!transformok.empty()) throw CLI::ValidationError(option_name, transformok);
          worked = worked and CLI::detail::lexical_cast(vals.at((4 + 1) * i + k + 1), j);
          if (worked) variable[l][k] = j;
        }
      }
      return worked;
    };
    CLI::Option *opt = add_option(option_name, f, option_description);
    auto valuename = std::string("SOURCE ") + std::string(CLI::detail::type_name<T>());
    opt->type_name(valuename)->type_size(-4 - 1);
    opt->expected(-1);
    opt->check(CLI::Validator(trans.get_description()));

    group->add_option(opt);
    return opt;
  }

  // Add option to parse multiple files.
  template <typename T>
  CLI::Option *add_fileoption(CLI::Option_group *group, std::string option_name,
                              std::array<T, QUDA_MAX_SOURCES> &variable, CLI::Validator trans,
                              std::string option_description = "")
  {

    CLI::callback_t f = [&variable, &option_name, trans](CLI::results_t vals) {
      size_t l;
      // T j; // results_t is just a vector of strings
      bool worked = true;

      CLI::Range validsource(0, QUDA_MAX_SOURCES);
      for (size_t i {0}; i < vals.size() / 2; ++i) { // will always be a multiple of 2
        auto sourceok = validsource(vals.at(2 * i));
        auto transformok = trans(vals.at(2 * i + 1));
        if (!sourceok.empty()) throw CLI::ValidationError(option_name, sourceok);
        if (!transformok.empty()) throw CLI::ValidationError(option_name, transformok);
        worked = worked and CLI::detail::lexical_cast(vals.at(2 * i), l);
        auto &j = variable[l];
        worked = worked and CLI::detail::lexical_cast(vals.at(2 * i + 1), j);

        // if (worked) variable[l] = j;
      }
      return worked;
    };
    CLI::Option *opt = add_option(option_name, f, option_description);
    auto valuename = std::string("SOURCE ") + std::string(CLI::detail::type_name<T>());
    opt->type_name(valuename)->type_size(-2);
    opt->expected(-1);
    opt->check(CLI::Validator(trans.get_description()));

    group->add_option(opt);
    return opt;
  }

  template <typename T>
  CLI::Option *add_massoption(CLI::Option_group *group, std::string option_name, std::array<T, QUDA_MAX_MASSES> &variable,
                              CLI::Validator trans, std::string option_description = "", bool = false)
  {

    CLI::callback_t f = [&variable, &option_name, trans](CLI::results_t vals) {
      size_t l;
      // T j; // results_t is just a vector of strings
      bool worked = true;

      CLI::Range validlevel(0, QUDA_MAX_MASSES);
      for (size_t i {0}; i < vals.size() / 2; ++i) { // will always be a multiple of 2
        auto levelok = validlevel(vals.at(2 * i));
        auto transformok = trans(vals.at(2 * i + 1));
        if (!levelok.empty()) throw CLI::ValidationError(option_name, levelok);
        if (!transformok.empty()) throw CLI::ValidationError(option_name, transformok);
        worked = worked and CLI::detail::lexical_cast(vals.at(2 * i), l);
        auto &j = variable[l];
        worked = worked and CLI::detail::lexical_cast(vals.at(2 * i + 1), j);

        // if (worked) variable[l] = j;
      }
      return worked;
    };
    CLI::Option *opt = add_option(option_name, f, option_description);
    auto valuename = std::string("FLAVOR<INT> ") + std::string(CLI::detail::type_name<T>());
    opt->type_name(valuename)->type_size(-2);
    opt->expected(-1);
    opt->check(CLI::Validator(trans.get_description()));
    // opt->transform(trans);
    // opt->default_str("");
    group->add_option(opt);
    return opt;
  }
};

std::shared_ptr<QUDAApp> make_app(std::string app_description = "QUDA internal test", std::string app_name = "");
void add_eigen_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_deflation_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_multigrid_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_eofa_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_madwf_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_su3_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_heatbath_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_propagator_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_contraction_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_gaugefix_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_comms_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_testing_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_quark_smear_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_clover_force_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_covdev_option_group(std::shared_ptr<QUDAApp> quda_app);

template <typename T> std::string inline get_string(CLI::TransformPairs<T> &map, T val)
{
  auto it
    = std::find_if(map.begin(), map.end(), [&val](const decltype(map.back()) &p) -> bool { return p.second == val; });
  return it->first;
}

// template<typename T>
// const char* inline get_cstring(CLI::TransformPairs<T> &map, T val){
//   return get_string(map,val).c_str();
// }
// parameters

extern int device_ordinal;
extern int rank_order;
extern bool native_blas_lapack;
extern std::array<int, 4> gridsize_from_cmdline;
extern std::array<int, 4> dim_partitioned;
extern QudaReconstructType link_recon;
extern QudaReconstructType link_recon_sloppy;
extern QudaReconstructType link_recon_precondition;
extern QudaReconstructType link_recon_eigensolver;
extern QudaPrecision prec;
extern QudaPrecision prec_sloppy;
extern QudaPrecision prec_refinement_sloppy;
extern QudaPrecision prec_precondition;
extern QudaPrecision prec_eigensolver;
extern QudaPrecision prec_null;
extern QudaPrecision prec_ritz;
extern QudaVerbosity verbosity;
extern std::array<int, 4> dim;
extern int &xdim;
extern int &ydim;
extern int &zdim;
extern int &tdim;
extern int Lsdim;
extern bool dagger;
extern QudaDslashType dslash_type;
extern int laplace3D;
extern std::string latfile;
extern bool unit_gauge;
extern double gaussian_sigma;
extern std::string gauge_outfile;
extern int Nsrc;
extern int Msrc;
extern int Nsrc_tile;
extern int Msrc_tile;
extern int niter;
extern int nrepeat;
extern int maxiter_precondition;
extern QudaVerbosity verbosity_precondition;
extern int gcrNkrylov;
extern QudaCABasis ca_basis;
extern double ca_lambda_min;
extern double ca_lambda_max;
extern QudaCABasis ca_basis_precondition;
extern double ca_lambda_min_precondition;
extern double ca_lambda_max_precondition;
extern int pipeline;
extern int solution_accumulator_pipeline;
extern int test_type;
extern quda::mgarray<int> nvec;
extern quda::mgarray<int> nvec_batch;
extern quda::mgarray<std::string> mg_vec_infile;
extern quda::mgarray<std::string> mg_vec_outfile;
extern quda::mgarray<bool> mg_vec_partfile;
extern QudaInverterType inv_type;
extern bool inv_deflate;
extern bool inv_multigrid;
extern bool gauge_smear;
extern QudaInverterType precon_type;
extern QudaSchwarzType precon_schwarz_type;
extern QudaAcceleratorType precon_accelerator_type;

extern double madwf_diagonal_suppressor;
extern int madwf_ls;
extern int madwf_null_miniter;
extern double madwf_null_tol;
extern int madwf_train_maxiter;
extern bool madwf_param_load;
extern bool madwf_param_save;
extern std::string madwf_param_infile;
extern std::string madwf_param_outfile;

extern int precon_schwarz_cycle;
extern int multishift;
extern std::vector<double> multishift_shifts;
extern std::vector<double> multishift_masses;
extern std::vector<double> multishift_tols;
extern std::vector<double> multishift_tols_hq;
extern bool verify_results;
extern bool low_mode_check;
extern bool oblique_proj_check;
extern double mass;
extern double kappa;
extern quda::mass_array<double> kappa_array;
extern double mu;
extern double epsilon;
extern double evmax;
extern double m5;
extern double b5;
extern double c5;
extern double anisotropy;
extern double tadpole_factor;
extern double eps_naik;
extern int n_naiks;
extern double clover_csw;
extern double clover_coeff;
extern bool compute_clover;
extern bool compute_clover_trlog;
extern bool compute_fatlong;
extern double tol;
extern double tol_precondition;
extern double tol_hq;
extern double reliable_delta;
extern bool alternative_reliable;
extern QudaTwistFlavorType twist_flavor;
extern QudaMassNormalization normalization;
extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;
extern QudaSolutionType solution_type;
extern QudaTboundary fermion_t_boundary;
extern std::array<int, 4> dilution_block_size;
extern double distance_pc_alpha0;
extern int distance_pc_t0;

extern int mg_levels;

extern int max_res_increase;
extern int max_res_increase_total;

extern quda::mgarray<QudaFieldLocation> solver_location;
extern quda::mgarray<QudaFieldLocation> setup_location;
extern quda::mgarray<int> nu_pre;
extern quda::mgarray<int> nu_post;
extern quda::mgarray<int> n_block_ortho;
extern quda::mgarray<bool> block_ortho_two_pass;
extern quda::mgarray<double> mu_factor;
extern quda::mgarray<QudaVerbosity> mg_verbosity;
extern quda::mgarray<bool> mg_setup_use_mma;
extern quda::mgarray<bool> mg_dslash_use_mma;
extern quda::mgarray<QudaInverterType> setup_inv;
extern quda::mgarray<QudaSolveType> coarse_solve_type;
extern quda::mgarray<QudaSolveType> smoother_solve_type;
extern quda::mgarray<int> num_setup_iter;
extern quda::mgarray<double> setup_tol;
extern quda::mgarray<int> setup_maxiter;
extern quda::mgarray<int> setup_maxiter_refresh;
extern quda::mgarray<QudaCABasis> setup_ca_basis;
extern quda::mgarray<int> setup_ca_basis_size;
extern quda::mgarray<double> setup_ca_lambda_min;
extern quda::mgarray<double> setup_ca_lambda_max;
extern QudaSetupType setup_type;
extern bool pre_orthonormalize;
extern bool post_orthonormalize;
extern double omega;
extern quda::mgarray<QudaInverterType> coarse_solver;
extern quda::mgarray<double> coarse_solver_tol;
extern quda::mgarray<QudaInverterType> smoother_type;
extern quda::mgarray<QudaCABasis> smoother_solver_ca_basis;
extern quda::mgarray<double> smoother_solver_ca_lambda_min;
extern quda::mgarray<double> smoother_solver_ca_lambda_max;
extern QudaPrecision smoother_halo_prec;
extern quda::mgarray<double> smoother_tol;
extern quda::mgarray<int> coarse_solver_maxiter;
extern quda::mgarray<QudaCABasis> coarse_solver_ca_basis;
extern quda::mgarray<int> coarse_solver_ca_basis_size;
extern quda::mgarray<double> coarse_solver_ca_lambda_min;
extern quda::mgarray<double> coarse_solver_ca_lambda_max;
extern bool generate_nullspace;
extern bool generate_all_levels;
extern quda::mgarray<QudaSchwarzType> mg_schwarz_type;
extern quda::mgarray<int> mg_schwarz_cycle;
extern bool mg_evolve_thin_updates;
extern QudaTransferType staggered_transfer_type;

extern quda::mgarray<std::array<int, 4>> geo_block_size;
extern bool mg_allow_truncation;
extern bool mg_staggered_kd_dagger_approximation;

extern bool use_mobius_fused_kernel;

extern int n_ev;
extern int max_search_dim;
extern int deflation_grid;
extern double tol_restart;

extern int eigcg_max_restarts;
extern int max_restart_num;
extern double inc_tol;
extern double eigenval_tol;

extern QudaExtLibType solver_ext_lib;
extern QudaExtLibType deflation_ext_lib;
extern QudaFieldLocation location_ritz;
extern QudaMemoryType mem_type_ritz;

// Parameters for the stand alone eigensolver
extern int eig_ortho_block_size;
extern int eig_evals_batch_size;
extern int eig_block_size;
extern int eig_n_ev;
extern int eig_n_kr;
extern int eig_n_conv;         // If unchanged, will be set to n_ev
extern int eig_n_ev_deflate;   // If unchanged, will be set to n_conv
extern int eig_batched_rotate; // If unchanged, will be set to maximum
extern bool eig_require_convergence;
extern int eig_check_interval;
extern int eig_max_restarts;
extern int eig_max_ortho_attempts;
extern double eig_tol;
extern double eig_qr_tol;
extern bool eig_use_eigen_qr;
extern bool eig_use_poly_acc;
extern int eig_poly_deg;
extern double eig_amin;
extern double eig_amax;
extern bool eig_use_normop;
extern bool eig_use_dagger;
extern bool eig_use_pc;
extern bool eig_compute_svd;
extern bool eig_compute_gamma5;
extern QudaEigSpectrumType eig_spectrum;
extern QudaEigType eig_type;
extern bool eig_arpack_check;
extern std::string eig_arpack_logfile;
extern std::string eig_vec_infile;
extern std::string eig_vec_outfile;
extern bool eig_io_parity_inflate;
extern QudaPrecision eig_save_prec;
extern bool eig_partfile;

// Parameters for the MG eigensolver.
// The coarsest grid params are for deflation,
// all others are for PR vectors.
extern quda::mgarray<bool> mg_eig;
extern quda::mgarray<int> mg_eig_ortho_block_size;
extern quda::mgarray<int> mg_eig_evals_batch_size;
extern quda::mgarray<int> mg_eig_block_size;
extern quda::mgarray<int> mg_eig_n_ev_deflate;
extern quda::mgarray<int> mg_eig_n_ev;
extern quda::mgarray<int> mg_eig_n_kr;
extern quda::mgarray<int> mg_eig_batched_rotate;
extern quda::mgarray<bool> mg_eig_require_convergence;
extern quda::mgarray<int> mg_eig_check_interval;
extern quda::mgarray<int> mg_eig_max_restarts;
extern quda::mgarray<int> mg_eig_max_ortho_attempts;
extern quda::mgarray<double> mg_eig_tol;
extern quda::mgarray<double> mg_eig_qr_tol;
extern quda::mgarray<bool> mg_eig_use_eigen_qr;
extern quda::mgarray<bool> mg_eig_use_poly_acc;
extern quda::mgarray<int> mg_eig_poly_deg;
extern quda::mgarray<double> mg_eig_amin;
extern quda::mgarray<double> mg_eig_amax;
extern quda::mgarray<bool> mg_eig_use_normop;
extern quda::mgarray<bool> mg_eig_use_dagger;
extern quda::mgarray<bool> mg_eig_use_pc;
extern quda::mgarray<QudaEigSpectrumType> mg_eig_spectrum;
extern quda::mgarray<QudaEigType> mg_eig_type;
extern quda::mgarray<QudaPrecision> mg_eig_save_prec;

extern bool mg_eig_coarse_guess;
extern bool mg_eig_preserve_deflation;

extern double heatbath_beta_value;
extern int heatbath_warmup_steps;
extern int heatbath_num_steps;
extern int heatbath_num_heatbath_per_step;
extern int heatbath_num_overrelax_per_step;
extern bool heatbath_coldstart;

extern int gf_gauge_dir;
extern int gf_maxiter;
extern int gf_verbosity_interval;
extern double gf_ovr_relaxation_boost;
extern double gf_fft_alpha;
extern int gf_reunit_interval;
extern double gf_tolerance;
extern bool gf_theta_condition;
extern bool gf_fft_autotune;

extern int eofa_pm;
extern double eofa_shift;
extern double eofa_mq1;
extern double eofa_mq2;
extern double eofa_mq3;

// SU(3) smearing options
extern double gauge_smear_rho;
extern double gauge_smear_epsilon;
extern double gauge_smear_alpha;
extern double gauge_smear_alpha1;
extern double gauge_smear_alpha2;
extern double gauge_smear_alpha3;
extern int gauge_smear_steps;
extern int gauge_smear_dir_ignore;
extern int measurement_interval;
extern QudaGaugeSmearType gauge_smear_type;
extern bool su_project;

extern double smear_coeff;
extern int    smear_n_steps;
extern int    smear_t0;
extern bool   smear_compute_two_link;
extern bool   smear_delete_two_link;

extern std::array<int, 4> grid_partition;

// contract options
extern QudaContractType contract_type;
extern char correlator_save_dir[256];
extern char correlator_file_affix[256];
extern std::array<int, 4> momentum;
extern bool open_flavor;

extern quda::file_array<char[256]> prop_source_infile;
extern quda::file_array<char[256]> prop_source_outfile;
extern quda::file_array<char[256]> prop_sink_infile;
extern quda::file_array<char[256]> prop_sink_outfile;
extern quda::source_array<std::array<int, 4>> prop_source_position;
extern int prop_source_smear_steps;
extern int prop_sink_smear_steps;
extern double prop_source_smear_coeff;
extern double prop_sink_smear_coeff;
extern bool prop_read_sources;
extern int prop_n_sources;
extern QudaPrecision prop_save_prec;

extern bool enable_testing;

extern bool detratio;

extern std::array<int, 4> covdev_mu;
