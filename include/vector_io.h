#pragma once

#include <string>

namespace quda {

  extern "C" {
    /**
       @brief Set whether to inflate single-fields to full fields when
        doing IO (required for compatability with MILC).
        @param[in] parity_inflate Whether to inflate
    */
    void set_io_parity_inflation(QudaBoolean parity_inflate);
  }

  /**
     @brief VectorIO is a simple wrapper class for loading and saving
     sets of vectors using QIO.
   */
  class VectorIO {
    const std::string filename;

  public:
    static bool parity_inflate;

    /**
       Constructor for VectorIO class
       @param[in] filename The filename associated with this IO object
    */
    VectorIO(const std::string &filename);

    /**
       @brief Load vectors from filename
       @param[in] eig_vecs The eigenvectors to load
    */
    void load(std::vector<ColorSpinorField *> &vecs);

    /**
       @brief Save vectors to filename
       @param[in] eig_vecs The eigenvectors to save
    */
    void save(const std::vector<ColorSpinorField *> &vecs);
  };

}
