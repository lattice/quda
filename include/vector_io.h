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
#ifdef HAVE_QIO
    bool parity_inflate;
#endif
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

    /**
       @brief Create alias pointers to a vector space of lower precision
       @param[in] vecs_high_prec The set of vectors with high precision
       @param[in] vecs_low_prec The set of vectors with lower precision
       @param[in] low_prec The low precsision value
    */

    void downPrec(const std::vector<ColorSpinorField *> &vecs_high_prec, std::vector<ColorSpinorField *> &vecs_low_prec,
                  const QudaPrecision save_prec);
  };

} // namespace quda
