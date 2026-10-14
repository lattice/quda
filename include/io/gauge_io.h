#pragma once

#include <string>
#include <io/field_io.h>

namespace quda
{

  /**
     @brief Metadata found when loading a gauge field file
   */
  struct GaugeFileInfo {
    QudaFileFormat format = QUDA_FILE_FORMAT_INVALID; /** The format the file was read as */
    QudaPrecision file_prec = QUDA_INVALID_PRECISION; /** Precision of the data in the file */
    bool ildg = false;                                /** Whether the file has the ILDG records */
    bool has_usqcd_info = false;                      /** Whether the file has a usqcdInfo record */
    double plaq = 0.0;                                /** usqcdInfo average Re Tr U_P / 3 */
    double linktr = 0.0;                              /** usqcdInfo average Re Tr U_mu / 3 */
    bool usqcd_info_matches = false;                  /** Whether the usqcdInfo values match the loaded field */
  };

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
       @return Metadata found in the file
    */
    GaugeFileInfo load(void *const gauge[], QudaPrecision prec, const int *X);

    /**
       @brief Save a gauge field to filename
       @param[in] gauge The host gauge field, one array per direction
       @param[in] prec The precision of the host gauge field
       @param[in] X The local lattice dimensions
    */
    void save(const void *const gauge[], QudaPrecision prec, const int *X);
  };

} // namespace quda
