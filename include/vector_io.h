#pragma once

#include <string>
#include <color_spinor_field.h>
#include <reference_wrapper_helper.h>

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
    bool partfile;

  public:
    /**
       Constructor for VectorIO class
       @param[in] filename The filename associated with this IO object
       @param[in] parity_inflate Whether to inflate single_parity
       field to dual parity fields for I/O
       @param[in] partfile Whether or not to save in partfiles (ignored on load)
    */
    VectorIO(const std::string &filename, bool parity_inflate = false, bool partfile = false);

    /**
       @brief Load vectors from filename
       @param[in] vecs The set of vectors to load
    */
    void load(cvector_ref<ColorSpinorField> &vecs);

    /**
       @brief Load propagator (12 vecs, Chroma compliant) from filename
       @param[in] vecs The set of vectors to load
    */
    void loadProp(vector_ref<ColorSpinorField> &vecs);

    /**
       @brief Save vectors to filename
       @param[in] vecs The set of vectors to save
       @param[in] prec Optional change of precision when saving
       @param[in] size Optional cap to number of vectors saved
    */
    void save(cvector_ref<const ColorSpinorField> &vecs, QudaPrecision prec = QUDA_INVALID_PRECISION, uint32_t size = 0);

    /**
       @brief Save propagator (12 vecs, Chroma compliant) to filename
       @param[in] vecs The set of vectors to save
    */
    void saveProp(cvector_ref<ColorSpinorField> &vecs);

    /**
       @brief Create alias pointers to a vector space of lower precision
       @param[in] vecs_high_prec The set of vectors with high precision
       @param[in] vecs_low_prec The set of vectors with lower precision
       @param[in] low_prec The low precsision value
    */
    void downPrec(cvector_ref<ColorSpinorField> &vecs_high_prec, vector_ref<ColorSpinorField> &vecs_low_prec,
                  const QudaPrecision save_prec);
  };

} // namespace quda
