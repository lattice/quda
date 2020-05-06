#pragma once

#include <string>

namespace quda
{

  /**
     @brief VectorIO is a simple wrapper class for loading and saving
     sets of vector fields using QIO.
   */
  class VectorIO
  {
    const std::string filename;
    bool parity_inflate;

  public:

    /**
       Constructor for VectorIO class
       @param[in] filename The filename associated with this IO object
       @param[in] parity_inflate Whether to inflate single_parity
       field to dual parity fields for I/O
    */
    VectorIO(const std::string &filename, bool parity_inflate = false);

    /**
       @brief Load vectors from filename
       @param[in] vecs The set of vectors to load
    */
    void load(std::vector<ColorSpinorField *> &vecs);

    /**
       @brief Save vectors to filename
       @param[in] vecs The set of vectors to save
    */
    void save(const std::vector<ColorSpinorField *> &vecs);
  };

} // namespace quda
