#pragma once

#include <array>
#include <externals/CLI11.hpp>
#include <quda.h>

// for compatibility while porting - remove later
extern void usage(char **);

namespace quda
{
  template <typename T> using mgarray = std::array<T, QUDA_MAX_MG_LEVEL>;
}

class QUDAApp : public CLI::App
{

public:
  QUDAApp(std::string app_description = "", std::string app_name = "") : CLI::App(app_description, app_name) {};

  virtual ~QUDAApp() {};

  template <typename T>
  CLI::Option *add_mgoption(std::string option_name, std::array<T, QUDA_MAX_MG_LEVEL> &variable, CLI::Validator trans,
                            std::string option_description = "", bool defaulted = false)
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
                            CLI::Validator trans, std::string option_description = "", bool defaulted = false)
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
  CLI::Option *add_mgoption(CLI::Option_group *group, std::string option_name, std::array<std::array<T, 4>, QUDA_MAX_MG_LEVEL> &variable,
                            CLI::Validator trans, std::string option_description = "", bool defaulted = false)
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
};

std::shared_ptr<QUDAApp> make_app(std::string app_description = "QUDA internal test", std::string app_name = "");
void add_eigen_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_deflation_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_multigrid_option_group(std::shared_ptr<QUDAApp> quda_app);

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

extern int device;
extern int rank_order;
extern std::array<int, 4> gridsize_from_cmdline;
extern std::array<int, 4> dim_partitioned;
extern QudaReconstructType link_recon;
extern QudaReconstructType link_recon_sloppy;
extern QudaReconstructType link_recon_precondition;
extern QudaPrecision prec;
extern QudaPrecision prec_sloppy;
extern QudaPrecision prec_refinement_sloppy;
extern QudaPrecision prec_precondition;
extern QudaPrecision prec_null;
extern QudaPrecision prec_ritz;
extern QudaVerbosity verbosity;
extern std::array<int, 4> dim;
extern int &xdim;
extern int &ydim;
extern int &zdim;
extern int &tdim;
extern int Lsdim;
extern QudaDagType dagger;
extern QudaDslashType dslash_type;
extern int laplace3D;
extern char latfile[256];
extern bool unit_gauge;
extern double gaussian_sigma;
extern char gauge_outfile[256];
extern int Nsrc;
extern int Msrc;
extern int niter;
extern int gcrNkrylov;
extern QudaCABasis ca_basis;
extern double ca_lambda_min;
extern double ca_lambda_max;
extern int pipeline;
extern int solution_accumulator_pipeline;
extern int test_type;
extern quda::mgarray<int> nvec;
extern quda::mgarray<char[256]> mg_vec_infile;
extern quda::mgarray<char[256]> mg_vec_outfile;
extern QudaInverterType inv_type;
extern bool inv_deflate;
extern QudaInverterType precon_type;
extern int multishift;
extern bool verify_results;
extern bool low_mode_check;
extern bool oblique_proj_check;
extern double mass;
extern double kappa;
extern double mu;
extern double epsilon;
extern double m5;
extern double b5;
extern double c5;
extern double anisotropy;
extern double tadpole_factor;
extern double eps_naik;
extern double clover_coeff;
extern bool compute_clover;
extern bool compute_fatlong;
extern double tol;
extern double tol_hq;
extern double reliable_delta;
extern bool alternative_reliable;
extern QudaTwistFlavorType twist_flavor;
extern QudaMassNormalization normalization;
extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;
extern QudaSolutionType solution_type;

extern int mg_levels;

extern quda::mgarray<QudaFieldLocation> solver_location;
extern quda::mgarray<QudaFieldLocation> setup_location;

extern quda::mgarray<int> nu_pre;
extern quda::mgarray<int> nu_post;
extern quda::mgarray<int> n_block_ortho;
extern quda::mgarray<double> mu_factor;
extern quda::mgarray<QudaVerbosity> mg_verbosity;
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
extern QudaPrecision smoother_halo_prec;
extern quda::mgarray<double> smoother_tol;
extern quda::mgarray<int> coarse_solver_maxiter;
extern quda::mgarray<QudaCABasis> coarse_solver_ca_basis;
extern quda::mgarray<int> coarse_solver_ca_basis_size;
extern quda::mgarray<double> coarse_solver_ca_lambda_min;
extern quda::mgarray<double> coarse_solver_ca_lambda_max;
extern bool generate_nullspace;
extern bool generate_all_levels;
extern quda::mgarray<QudaSchwarzType> schwarz_type;
extern quda::mgarray<int> schwarz_cycle;

extern quda::mgarray<std::array<int, 4>> geo_block_size;
extern int nev;
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
extern int eig_nEv;
extern int eig_nKr;
extern int eig_nConv; // If unchanged, will be set to nEv
extern int eig_batched_rotate; // If unchanged, will be set to maximum
extern bool eig_require_convergence;
extern int eig_check_interval;
extern int eig_max_restarts;
extern double eig_tol;
extern bool eig_use_poly_acc;
extern int eig_poly_deg;
extern double eig_amin;
extern double eig_amax;
extern bool eig_use_normop;
extern bool eig_use_dagger;
extern bool eig_compute_svd;
extern QudaEigSpectrumType eig_spectrum;
extern QudaEigType eig_type;
extern bool eig_arpack_check;
extern char eig_arpack_logfile[256];
extern char eig_QUDA_logfile[256];
extern char eig_vec_infile[256];
extern char eig_vec_outfile[256];

// Parameters for the MG eigensolver.
// The coarsest grid params are for deflation,
// all others are for PR vectors.
extern quda::mgarray<bool> mg_eig;
extern quda::mgarray<int> mg_eig_nEv;
extern quda::mgarray<int> mg_eig_nKr;
extern quda::mgarray<int> mg_eig_batched_rotate;
extern quda::mgarray<bool> mg_eig_require_convergence;
extern quda::mgarray<int> mg_eig_check_interval;
extern quda::mgarray<int> mg_eig_max_restarts;
extern quda::mgarray<double> mg_eig_tol;
extern quda::mgarray<bool> mg_eig_use_poly_acc;
extern quda::mgarray<int> mg_eig_poly_deg;
extern quda::mgarray<double> mg_eig_amin;
extern quda::mgarray<double> mg_eig_amax;
extern quda::mgarray<bool> mg_eig_use_normop;
extern quda::mgarray<bool> mg_eig_use_dagger;
extern quda::mgarray<QudaEigSpectrumType> mg_eig_spectrum;
extern quda::mgarray<QudaEigType> mg_eig_type;
extern bool mg_eig_coarse_guess;
extern bool mg_eig_preserve_deflation;

extern double heatbath_beta_value;
extern int heatbath_warmup_steps;
extern int heatbath_num_steps;
extern int heatbath_num_heatbath_per_step;
extern int heatbath_num_overrelax_per_step;
extern bool heatbath_coldstart;

extern QudaContractType contract_type;
