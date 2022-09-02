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
    void load(vector_ref<ColorSpinorField> &&vecs);

    /**
       @brief Load vectors from filename.  Generic interface that
       accepts vector of fields or vector of pointers
       @param[in] vecs The set of vectors to load
    */
    template <typename T> void load(T &&vecs)
    {
      load(make_set(vecs));
    }

    /**
       @brief Save vectors to filename
       @param[in] vecs The set of vectors to save
       @param[in] prec Optional change of precision when saving
       @param[in] size Optional cap to number of vectors saved
    */
    void save(vector_ref<const ColorSpinorField> &&vecs, QudaPrecision prec = QUDA_INVALID_PRECISION, uint32_t size = 0);

    /**
       @brief Save vectors to filename.  Generic interface that
       accepts vector of fields or vector of pointers
       @param[in] vecs The set of vectors to save
       @param[in] prec Optional change of precision when saving
       @param[in] size Optional cap to number of vectors saved
    */
    template <typename T>
    void save(T &&vecs, QudaPrecision prec = QUDA_INVALID_PRECISION, uint32_t size = 0)
    {
      save(make_set(vecs), prec, size);
    }

  };

} // namespace quda
