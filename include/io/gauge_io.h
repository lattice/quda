#pragma once

#include <cstdint>
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
    bool has_plaq_linktr = false;                     /** Whether the metadata gives plaq and linktr */
    double plaq = 0.0;                /** Average Re Tr U_P / 3 from a usqcdInfo record or MILC .info file */
    double linktr = 0.0;              /** Average Re Tr U_mu / 3 from a usqcdInfo record or MILC .info file */
    bool plaq_linktr_matches = false; /** Whether plaq and linktr match the loaded field */
    bool byte_swapped = false;        /** MILC: whether the file is in the opposite byte order */
    bool has_info_file = false;       /** MILC: whether a .info file was found and checked */
    bool has_checksum = false;        /** Whether the file has checksums (ILDG files from other codes have none) */
    bool checksum_matches = false;    /** Whether the checksums of the data and the file (and MILC .info file) agree */
    uint32_t sum29 = 0;               /** The checksums computed from the data: sum29 and sum31 for MILC files, suma */
    uint32_t sum31 = 0;               /** and sumb for QIO and ILDG files */

    /** Largest deviation from unitarity of the loaded links over all ranks, as MILC measures it, before any
        reunitarization; negative if not measured */
    double unitarity_deviation = -1.0;
    /** The same after reunitarization; negative if the field was not reunitarized */
    double reunitarized_deviation = -1.0;
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
