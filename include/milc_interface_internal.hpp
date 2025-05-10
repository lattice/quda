#pragma once

#include <string>
#include <vector>
#include <array>
#include "quda.h"

namespace quda
{

  // Structure used to handle loading from input file
  struct mgInputStruct {

    /** Total number of multigrid levels   */
    int mg_levels = 4;

    /** Whether to perform correctness checks on multigrid setup */
    bool verify_results = true;

    /** Precision for near-nulls, coarse links */
    QudaPrecision preconditioner_precision = QUDA_HALF_PRECISION;

    /** Use the optimized KD operator (true), naive coarsened operator (false), or optimized dropped links (drop) */
    QudaTransferType optimized_kd = QUDA_TRANSFER_OPTIMIZED_KD;

    /** Whether to accelerate setup using MMA routines */
    bool setup_use_mma[QUDA_MAX_MG_LEVEL] = {true, true, true, true, true};

    /** Whether to accelerate dslash using MMA routines */
    bool dslash_use_mma[QUDA_MAX_MG_LEVEL] = {true, true, true, true, true};

    /** Whether to accelerate transfer using MMA routines; currently set to false b/c it's generally not faster for now */
    bool transfer_use_mma[QUDA_MAX_MG_LEVEL] = {false, false, false, false, false};

    /** Whether to allow dropping the long links for small (less than three) aggregate directions */
    bool allow_truncation = false;

    /** Whether to use the dagger approximation to Xinv */
    bool dagger_approximation = false;

    /** Number of rhs to solve at once in the block solver, -1 means all */
    int block_solver_batch_size = -1;

    /**
     * Setup:
     * There is no near-null vector generation on the first and last (coarsest) level.
     * - The second level is the KD preconditioned staggered/HISQ operator, which is not a coarsening of the fine operator
     * - By definition there is no coarsening of the coarsest level
     * For this reason most of these variables are ignored on the first and last level.
     * We do reuse `nvec` on the coarsest level to specify the size of coarsest-level deflation basis
     * For reference: geo_block_size[0] does get defined internally (1 1 1 1 for optimized, 2 2 2 2 for coarse KD)
     */

    /** Number of vectors to use for the setup solver; ignored on first level, reused for deflation size on last level */
    int nvec[QUDA_MAX_MG_LEVEL] = {24, 24, 24, 24, 24};

    /** Solver type for the setup solver */
    QudaInverterType setup_inv[QUDA_MAX_MG_LEVEL]
      = {QUDA_CGNR_INVERTER, QUDA_CGNR_INVERTER, QUDA_CGNR_INVERTER, QUDA_CGNR_INVERTER, QUDA_CGNR_INVERTER};

    /** Tolerance for the setup solver; ignored on first and last level */
    double setup_tol[QUDA_MAX_MG_LEVEL] = {1e-5, 1e-5, 1e-5, 1e-5, 1e-5};

    /** Maximum number of iterations for the setup solver; ignored on first and last level */
    double setup_maxiter[QUDA_MAX_MG_LEVEL] = {500, 500, 500, 500, 500};

    /** Size of the basis for communications avoiding solvers (CA-GCR, etc); ignored on first and last level */
    int setup_ca_basis_size[QUDA_MAX_MG_LEVEL] = {4, 4, 4, 4, 4};

    /** Input file for loading pre-existing near-null vectors; ignored on first and last level */
    char mg_vec_infile[QUDA_MAX_MG_LEVEL][256] = {"", "", "", "", ""};

    /** Output file for saving near-null vectors after generation; ignored on first and last level */
    char mg_vec_outfile[QUDA_MAX_MG_LEVEL][256] = {"", "", "", "", ""};

    /** Whether or not to save near-null vectors in partfile or not; loading will autodetect; ignored on first and last level */
    bool mg_vec_partfile[QUDA_MAX_MG_LEVEL] = {false, false, false, false, false};

    /** The aggregation size for each multigrid level; ignored on first and last level, the first level's values are prescribed */
    int geo_block_size[QUDA_MAX_MG_LEVEL][4];

    /**
     * Solve:
     * The coarse solver parameters are ignored on the first level because it is
     * the outer solver, and as such we reuse values specified in MILC (tolerance, max iterations)
     * Some of these are fixed (for now) and will be exposed in the future:
     * - Solve type (for now fixed to full operator, will eventually expose Schur operator)
     * - Solver (for now fixed to GCR, will eventually expose PCG for Schur operator)
     * The smoother types are ignored for the coarsest level because, by definition, there is no
     * still coarser operator to smooth
     */

    /** Whether to use the full or preconditioned operator and solver; ignored on the first and second level because it's prescribed */
    QudaSolveType coarse_solve_type[QUDA_MAX_MG_LEVEL]
      = {QUDA_DIRECT_PC_SOLVE, QUDA_DIRECT_PC_SOLVE, QUDA_DIRECT_PC_SOLVE, QUDA_DIRECT_PC_SOLVE, QUDA_DIRECT_PC_SOLVE};

    /** Solver type for the MG solve, currently only GCR is supported */
    QudaInverterType coarse_solver[QUDA_MAX_MG_LEVEL]
      = {QUDA_GCR_INVERTER, QUDA_GCR_INVERTER, QUDA_GCR_INVERTER, QUDA_GCR_INVERTER, QUDA_GCR_INVERTER};

    /** Tolerance for the coarse solver; ignored on the first level because it's prescribed by the external solve */
    double coarse_solver_tol[QUDA_MAX_MG_LEVEL] = {0.25, 0.25, 0.25, 0.25, 0.25};

    /** Maximum number of iterations for each coarse solve; ignored on the first level because it's prescribed by the external solve */
    int coarse_solver_maxiter[QUDA_MAX_MG_LEVEL] = {16, 16, 16, 16, 16};

    /** Size of the basis for communications avoiding solvers (CA-GCR, etc), only used for the last level */
    int coarse_solver_ca_basis_size[QUDA_MAX_MG_LEVEL] = {16, 16, 16, 16, 16};

    /** Solver type to use for the MG smoother, ignored on the last level */
    QudaInverterType smoother_type[QUDA_MAX_MG_LEVEL]
      = {QUDA_CA_GCR_INVERTER, QUDA_CA_GCR_INVERTER, QUDA_CA_GCR_INVERTER, QUDA_CA_GCR_INVERTER, QUDA_CA_GCR_INVERTER};

    /** Number of pre-smoothing iterations to perform on all levels; ignored on the last level b/c there's no smoother */
    int nu_pre[QUDA_MAX_MG_LEVEL] = {0, 0, 0, 0, 0};

    /** Number of post-smoothing iterations to perform on all levels; ignored on the last level b/c there's no smoother */
    int nu_post[QUDA_MAX_MG_LEVEL] = {2, 2, 2, 2, 2};

    /** Verbosity values to use on each level */
    QudaVerbosity mg_verbosity[QUDA_MAX_MG_LEVEL]
      = {QUDA_SUMMARIZE, QUDA_SUMMARIZE, QUDA_SUMMARIZE, QUDA_SUMMARIZE, QUDA_SUMMARIZE};

    // Coarsest level deflation

    /** Size of initial factorization; desired eigenvectors plus 2 is "typical" */
    int deflate_n_ev = 66;

    /** Size of Krylov space after extension; 1.5 or 2 times the desired number of eigenvectors is "typical" */
    int deflate_n_kr = 128;

    /** Number of times to restart the eigensolver before exiting */
    int deflate_max_restarts = 50;

    /** Target tolerance of the eigenvalues */
    double deflate_tol = 1e-5;

    /** Block size for the eigensolver; if it's 1 the eigensolver is TRLM, if it's greater than 1 it uses block TRLM */
    int deflate_block_size = 1;

    /** Whether or not to use polynomial acceleration */
    bool deflate_use_poly_acc = false;

    /** Bottom of Chebyshev window, eigenvalues below this value are enhanced -> converge more easily; ignored if
     * there's no polynomial acceleration */
    double deflate_a_min = 1e-2;

    /** Degree of the polynomial used for acceleration; larger values trade better resolution of eigenvalues against
     * cost; ignored if there's no polynomial acceleration */
    int deflate_poly_deg = 50;

    /** Whether or not to save eigenvectors in partfile format or not */
    bool deflate_vec_partfile = false;

    /**
     * @brief Sets best-practice defaults for multigrid parameters
     */
    void setBestPracticeDefaults()
    {
      /* required or best-practice values for typical solves */
      nvec[0] = 3;              // must be this
      geo_block_size[0][0] = 1; // must be this...
      geo_block_size[0][1] = 1; // "
      geo_block_size[0][2] = 1; // "
      geo_block_size[0][3] = 1; // "

      nvec[1] = 64;
      geo_block_size[1][0] = 2;
      geo_block_size[1][1] = 2;
      geo_block_size[1][2] = 2;
      geo_block_size[1][3] = 2;

      nvec[2] = 96;
      geo_block_size[2][0] = 2;
      geo_block_size[2][1] = 2;
      geo_block_size[2][2] = 2;
      geo_block_size[2][3] = 2;

      /* Setup */

      /* level 0 -> 1 is K-D, no customization */

      /* Level 1 (pseudo-fine) to 2 (intermediate) */
      setup_inv[1] = QUDA_CGNR_INVERTER;
      setup_tol[1] = 1e-5;
      setup_maxiter[1] = 500;
      setup_ca_basis_size[1] = 4;
      mg_vec_infile[1][0] = 0;
      mg_vec_outfile[1][0] = 0;

      /* Level 2 (intermediate) to 3 (coarsest) */
      setup_inv[2] = QUDA_CGNR_INVERTER;
      setup_tol[2] = 1e-5;
      setup_maxiter[2] = 500;
      setup_ca_basis_size[2] = 4;
      mg_vec_infile[2][0] = 0;
      mg_vec_outfile[2][0] = 0;

      /* Solve info */

      /* Level 0 only needs a smoother */
      smoother_type[0] = QUDA_CA_GCR_INVERTER;
      nu_pre[0] = 0;
      nu_post[0] = 4;

      /* Level 1 */
      coarse_solver[1] = QUDA_GCR_INVERTER;
      coarse_solver_tol[1] = 5e-2;
      coarse_solver_maxiter[1] = 4;
      coarse_solver_ca_basis_size[1] = 4; // generally unused b/c not coarsest level
      smoother_type[1] = QUDA_CA_GCR_INVERTER;
      nu_pre[1] = 0;
      nu_post[1] = 2;

      /* Level 2 */
      coarse_solve_type[2] = QUDA_DIRECT_PC_SOLVE;
      coarse_solver[2] = QUDA_GCR_INVERTER;
      coarse_solver_tol[2] = 0.25;
      coarse_solver_maxiter[2] = 4;
      coarse_solver_ca_basis_size[2] = 4; // generally unused b/c not coarsest level
      smoother_type[2] = QUDA_CA_GCR_INVERTER;
      nu_pre[2] = 0;
      nu_post[2] = 2;

      /* Level 3 */
      coarse_solve_type[3] = QUDA_DIRECT_PC_SOLVE;
      coarse_solver[3] = QUDA_CA_GCR_INVERTER; // use CGNR for non-deflated... sometimes
      coarse_solver_tol[3] = 0.25;
      coarse_solver_maxiter[3] = 16;       // use larger for non-deflated
      coarse_solver_ca_basis_size[3] = 16; // ignored for non-CA solvers

      /* Misc */
      mg_verbosity[0] = QUDA_SUMMARIZE;
      mg_verbosity[1] = QUDA_SUMMARIZE;
      mg_verbosity[2] = QUDA_SUMMARIZE;
      mg_verbosity[3] = QUDA_SUMMARIZE;

      /* Deflation */
      nvec[3] = 0; // 64; // do not deflate
      mg_vec_infile[3][0] = 0;
      mg_vec_outfile[3][0] = 0;
      deflate_n_ev = 66;
      deflate_n_kr = 128;
      deflate_tol = 1e-3;
      deflate_max_restarts = 50;
      deflate_block_size = 1; // TRLM, > 1 is block TRLM
      deflate_use_poly_acc = false;
      deflate_a_min = 1e-2;
      deflate_poly_deg = 20;
    }

    // set defaults
    mgInputStruct()
    {
      /* Initialize the aggregation sizes, this one's a bit too painful
         to include as a default value */
      for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
        for (int d = 0; d < 4; d++) { geo_block_size[i][d] = 2; }
      }

      /* Set some defaults for best practices */
      setBestPracticeDefaults();
    }

    /**
     * @brief Convert string to QudaInverterType enum
     * @param[in] name String name of inverter type
     * @return QudaInverterType enum value, or QUDA_INVALID_INVERTER if invalid
     */
    QudaInverterType getQudaInverterType(const char *name) const
    {
      if (strcmp(name, "gcr") == 0) {
        return QUDA_GCR_INVERTER;
      } else if (strcmp(name, "cgnr") == 0) {
        return QUDA_CGNR_INVERTER;
      } else if (strcmp(name, "cgne") == 0) {
        return QUDA_CGNE_INVERTER;
      } else if (strcmp(name, "ca-cgnr") == 0) {
        return QUDA_CA_CGNR_INVERTER;
      } else if (strcmp(name, "ca-cgne") == 0) {
        return QUDA_CA_CGNE_INVERTER;
      } else if (strcmp(name, "bicgstab") == 0) {
        return QUDA_BICGSTAB_INVERTER;
      } else if (strcmp(name, "bicgstab-l") == 0) {
        return QUDA_BICGSTABL_INVERTER;
      } else if (strcmp(name, "ca-gcr") == 0) {
        return QUDA_CA_GCR_INVERTER;
      } else {
        return QUDA_INVALID_INVERTER;
      }
    }

    /**
     * @brief Convert string to QudaPrecision enum
     * @param[in] name String name of precision ("single" or "half")
     * @return QudaPrecision enum value, or QUDA_INVALID_PRECISION if invalid
     */
    QudaPrecision getQudaPrecision(const char *name) const
    {
      if (strcmp(name, "single") == 0) {
        return QUDA_SINGLE_PRECISION;
      } else if (strcmp(name, "half") == 0) {
        return QUDA_HALF_PRECISION;
      } else {
        return QUDA_INVALID_PRECISION;
      }
    }

    /**
     * @brief Convert string to QudaSolveType enum
     * @param[in] name String name of solve type ("direct" or "direct-pc")
     * @return QudaSolveType enum value, or QUDA_INVALID_SOLVE if invalid
     */
    QudaSolveType getQudaSolveType(const char *name) const
    {
      if (strcmp(name, "direct") == 0) {
        return QUDA_DIRECT_SOLVE;
      } else if (strcmp(name, "direct-pc") == 0) {
        return QUDA_DIRECT_PC_SOLVE;
      } else {
        return QUDA_INVALID_SOLVE;
      }
    }

    /**
     * @brief Convert string to QudaTransferType enum
     * @param[in] name String name of transfer type ("true", "false", or "drop")
     * @return QudaTransferType enum value, or QUDA_TRANSFER_INVALID if invalid
     */
    QudaTransferType getQudaTransferType(const char *name) const
    {
      if (strcmp(name, "true") == 0) {
        return QUDA_TRANSFER_OPTIMIZED_KD;
      } else if (strcmp(name, "false") == 0) {
        return QUDA_TRANSFER_COARSE_KD;
      } else if (strcmp(name, "drop") == 0) {
        return QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG;
      } else {
        return QUDA_TRANSFER_INVALID;
      }
    }

    /**
     * @brief Convert string to QudaVerbosity enum
     * @param[in] name String name of verbosity level ("silent", "summarize"/"false", "verbose"/"true", or "debug")
     * @return QudaVerbosity enum value, or QUDA_INVALID_VERBOSITY if invalid
     */
    QudaVerbosity getQudaVerbosity(const char *name) const
    {
      if (strcmp(name, "silent") == 0) {
        return QUDA_SILENT;
      } else if (strcmp(name, "summarize") == 0 || strcmp(name, "false") == 0) {
        // false == summary is for backwards compatibility
        return QUDA_SUMMARIZE;
      } else if (strcmp(name, "verbose") == 0 || strcmp(name, "true") == 0) {
        // true == verbose is for backwards compatibility
        return QUDA_VERBOSE;
      } else if (strcmp(name, "debug") == 0) {
        return QUDA_DEBUG_VERBOSE;
      } else {
        return QUDA_INVALID_VERBOSITY;
      }
    }

    /**
     * @brief Parses an input from a multigrid commands file that is general across the entire preconditioner
     *
     * @tparam Parser Type of the parser lambda that processes the second argument
     * @param[out] error_code Set to 1 if insufficient arguments are provided
     * @param[in] input_line Vector of strings containing the input arguments
     * @param[in] key The expected first argument to match against
     * @param[in] parse Parser lambda that processes the second argument
     * @return true if the key matches and parsing was successful, false otherwise
     */
    template <typename Parser>
    bool parse_2_args(int &error_code, const std::vector<std::string> &input_line, const char *key, Parser &&parse)
    {
      if (strcmp(input_line[0].c_str(), key) == 0) {
        if (input_line.size() < 2) {
          error_code = 1;
          return false;
        } else {
          parse(input_line[1].c_str());
          return true;
        }
      }
      return false;
    }

    /**
     * @brief Parses an input from a multigrid commands file that's applied to a specific level
     *
     * @tparam Parser Type of the parser lambda that processes the second and third arguments
     * @param[out] error_code Set to 1 if insufficient arguments are provided
     * @param[in] input_line Vector of strings containing the input arguments
     * @param[in] key The expected first argument to match against
     * @param[in] parse Parser lambda that processes the second argument, which is the preconditioner level, and third
     * argument, which is a value
     * @return true if the key matches and parsing was successful, false otherwise
     */
    template <typename Parser>
    bool parse_3_args(int &error_code, const std::vector<std::string> &input_line, const char *key, Parser &&parse)
    {
      if (strcmp(input_line[0].c_str(), key) == 0) {
        if (input_line.size() < 3) {
          error_code = 1;
          return false;
        } else {
          parse(atoi(input_line[1].c_str()), input_line[2].c_str());
          return true;
        }
      }
      return false;
    }

    /**
     * @brief Parses an input from a multigrid commands file that's applied to a specific level and has four geometric
     * parameters
     *
     * @tparam Parser Type of the parser lambda that processes the second argument and geometric parameters
     * @param[out] error_code Set to 1 if insufficient arguments are provided
     * @param[in] input_line Vector of strings containing the input arguments
     * @param[in] key The expected first argument to match against
     * @param[in] parse Parser lambda that processes the second argument, which is the preconditioner level, and four
     * geometric parameters
     * @return true if the key matches and parsing was successful, false otherwise
     */
    template <typename Parser>
    bool parse_3_geo_args(int &error_code, const std::vector<std::string> &input_line, const char *key, Parser &&parse)
    {
      if (strcmp(input_line[0].c_str(), key) == 0) {
        if (input_line.size() < 6) {
          error_code = 1;
          return false;
        } else {
          std::array<const char *, 4> vals
            = {input_line[2].c_str(), input_line[3].c_str(), input_line[4].c_str(), input_line[5].c_str()};
          parse(atoi(input_line[1].c_str()), vals);
          return true;
        }
      }
      return false;
    }

    /**
     * @brief Update internal parameters based on input line
     * @param[in] input_line Vector of strings containing input parameters
     * @return True if update was successful, false if error occurred
     */
    bool update(std::vector<std::string> &input_line);
  };

  // Internal structure that maintains `QudaMultigridParam`,
  // `QudaInvertParam`, `QudaEigParam`s, and the traditional
  // void* returned by `newMultigridQuda`.
  // last_mass tracks rebuilds based on changing the mass.
  struct milcMultigridPack {
    QudaMultigridParam mg_param;
    QudaInvertParam mg_inv_param;
    QudaEigParam mg_eig_param[QUDA_MAX_MG_LEVEL];
    QudaPrecision preconditioner_precision;
    double last_mass;
    void *mg_preconditioner;
    mgInputStruct input_struct;
  };

  /**
   * @brief Sets the eigensolver parameters for multigrid based on input parameters
   *
   * @param[out] mg_eig_param The QUDA eigensolver parameters to be set
   * @param[in] input_struct The input structure containing multigrid parameters
   * @param[in] level The multigrid level for which parameters are being set
   */
  void milcSetMultigridEigParam(QudaEigParam &mg_eig_param, const mgInputStruct &input_struct, int level);

  /**
   * @brief Sets up the multigrid parameters for MILC interface
   *
   * @param[out] mg_pack The multigrid parameter pack to be configured
   * @param[in] host_precision The precision to use for host operations
   * @param[in] device_precision The precision to use for device operations
   * @param[in] device_precision_sloppy The precision to use for sloppy device operations
   * @param[in] mass The quark mass parameter
   * @param[in] mg_param_file Path to the multigrid parameter input file
   */
  void milcSetMultigridParam(milcMultigridPack *mg_pack, QudaPrecision host_precision, QudaPrecision device_precision,
                             QudaPrecision device_precision_sloppy, double mass, const char *const mg_param_file);

}; // namespace quda
