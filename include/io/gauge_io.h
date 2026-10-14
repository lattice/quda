#pragma once

#include <string>
#include <io/field_io.h>

namespace quda
{

  /**
     @brief GaugeIO is a simple wrapper class for loading and saving
     host gauge fields in QDP order (one array per direction).
   */
  class GaugeIO
  {
    const std::string filename;
    const FileIOParam param;

  public:
    /**
       @brief Constructor for GaugeIO class
       @param[in] filename The filename associated with this IO object
       @param[in] param File format and I/O options
    */
    GaugeIO(const std::string &filename, const FileIOParam &param = {});

    /**
       @brief Load a gauge field from filename
       @param[out] gauge The host gauge field, one array per direction
       @param[in] prec The precision of the host gauge field
       @param[in] X The local lattice dimensions
    */
    void load(void *const gauge[], QudaPrecision prec, const int *X);

    /**
       @brief Save a gauge field to filename
       @param[in] gauge The host gauge field, one array per direction
       @param[in] prec The precision of the host gauge field
       @param[in] X The local lattice dimensions
    */
    void save(const void *const gauge[], QudaPrecision prec, const int *X);
  };

} // namespace quda
