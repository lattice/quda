#pragma once

namespace quda
{
  /**
    @brief Parameter structure for holding the various MADWF parameters.
  */
  struct MadwfParam {

    /** The diagonal constant to suppress the low modes when performing 5D transfer */
    double madwf_diagonal_suppressor;

    /** The target MADWF Ls to be used in the accelerator */
    int madwf_ls;

    /** The minimum number of iterations after which to generate the null vectors for MADWF */
    int madwf_null_miniter;

    /** The maximum tolerance after which to generate the null vectors for MADWF */
    double madwf_null_tol;

    /** The maximum number of iterations for the training iterations */
    int madwf_train_maxiter;

    /** Whether to load the MADWF parameters from the file system */
    bool madwf_param_load;

    /** Whether to save the MADWF parameters to the file system */
    bool madwf_param_save;

    /** Path to load from the file system */
    std::string madwf_param_infile;

    /** Path to save to the file system */
    std::string madwf_param_outfile;
  };

} // namespace quda
