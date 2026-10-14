#pragma once

#include <string>
#include <vector>
#include <color_spinor_field.h>
#include <reference_wrapper_helper.h>
#include <io/field_io.h>

namespace quda
{

  /**
     @brief VectorIO is a simple wrapper class for loading and saving
     sets of vector fields.
   */
  class VectorIO
  {
    const std::string filename;
    const FileIOParam param;

  public:
    /**
       Constructor for VectorIO class
       @param[in] filename The filename associated with this IO object
       @param[in] param File format and I/O options, including whether
       to inflate single-parity fields to dual-parity fields for I/O and
       whether to save in partfiles (ignored on load)
    */
    VectorIO(const std::string &filename, const FileIOParam &param = {});

    /**
       @brief Load vectors from filename
       @param[in] vecs The set of vectors to load
       @param[out] evals Optional eigenvalues, filled if the file format stores them
    */
    void load(cvector_ref<ColorSpinorField> &vecs, std::vector<complex_t> *evals = nullptr);

    /**
       @brief Save vectors to filename
       @param[in] vecs The set of vectors to save
       @param[in] prec Optional change of precision when saving
       @param[in] size Optional cap to number of vectors saved
       @param[in] evals Optional eigenvalues, saved if the file format stores them
    */
    void save(cvector_ref<const ColorSpinorField> &vecs, QudaPrecision prec = QUDA_INVALID_PRECISION, uint32_t size = 0,
              const std::vector<complex_t> *evals = nullptr);
  };

} // namespace quda
