#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <io/field_io.h>

namespace quda
{

  /**
     @brief A line of a MILC .info file
   */
  struct MILCInfoItem {
    std::string key;  /** The key, as milc::read_info parses it (empty if the line has none) */
    std::string line; /** The line as it is in the file, without its newline */
  };

  /**
     @brief The metadata of a gauge field file beyond the links, their
     checksums and the plaquette: what a conversion to another file
     can carry over.  Fields that a format doesn't have are left empty.
   */
  struct GaugeMetadata {
    std::string user_file_xml;   /** SciDAC: the user file XML (none in ILDG files from other codes) */
    std::string user_record_xml; /** SciDAC: the user record XML of the field, verbatim */
    bool has_usqcd_info = false; /** SciDAC: whether the user record XML is a usqcdInfo record */
    /** SciDAC: the content of the <info> element of the usqcdInfo record, verbatim; none without usqcdInfo or
        without an <info> element */
    std::optional<std::string> usqcd_info;
    std::optional<std::string> ildg_lfn; /** ILDG: the ildg-data-lfn record of the field, if there is one */
    /** SciDAC: the <date> of the field's private record XML, verbatim, if it has one */
    std::optional<std::string> record_date;
    std::string milc_time_stamp;         /** MILC: the time stamp of the header */
    std::vector<MILCInfoItem> milc_info; /** MILC: the lines of the .info file in file order, blank ones left out */
  };

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

    GaugeMetadata metadata; /** The file's other metadata */
  };

  /**
     @brief The observables of a gauge field that MILC records in its .info files
   */
  struct GaugeObservables {
    double plaq = 0.0;           /** Average Re Tr U_P / 3 over all plaquettes */
    double plaq_spatial = 0.0;   /** The same over the spatial plaquettes; MILC's gauge.ssplaq is 3 times this */
    double plaq_temporal = 0.0;  /** The same over the temporal plaquettes; MILC's gauge.stplaq is 3 times this */
    double linktr = 0.0;         /** Average Re Tr U_mu / 3: MILC's gauge.nersc_linktr */
    uint32_t nersc_checksum = 0; /** MILC's gauge.nersc_checksum: the sum modulo 2^32 of the 32-bit patterns of the
                                     real and imaginary parts, as floats, of rows 0 and 1 of every link */
  };

  /**
     @brief Compute the observables of a host gauge field in QDP order,
     on the device for the plaquettes and the link trace, so after
     initQuda.  The links are taken with periodic boundaries, as they
     are in a file.  The checksum is exact, so it is the same on any
     rank layout and thread count.  Collective.
     @param[in] gauge The host gauge field, one array per direction
     @param[in] prec The precision of the host gauge field
     @param[in] X The local lattice dimensions
   */
  GaugeObservables gauge_observables(const void *const gauge[], QudaPrecision prec, const int *X);

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
