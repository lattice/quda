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

     A load fills it.  A save can be given one, and writes the fields of
     the format it saves, in either backend; empty fields mean what a
     save without it writes:
     - user_file_xml: written verbatim; empty means QUDA's <qudaFile>.
     - has_usqcd_info: a usqcdInfo record with freshly computed
       plaquette and link trace and usqcd_info as its <info> (none means
       "QUDA"), if FileIOParam::write_usqcd_info is set; otherwise QUDA's
       own record XML.  Without it, a non-empty user_record_xml is written
       verbatim, and no usqcdInfo record.
     - ildg_lfn: the LFN of ILDG saves, ahead of FileIOParam::ildg_lfn
       (an empty one counts as none).
     - record_date: written verbatim as the private record XML's <date>;
       none means the time of saving.
     - milc_time_stamp: the MILC header's time stamp and the .info
       file's time_stamp, at most 64 bytes; empty means the time of saving.
     - milc_info: written to the .info file verbatim, in order, after the
       lines QUDA writes.  A line whose key QUDA writes (magic_number,
       time_stamp, checksums, nx to nt, and the four keys of
       FileIOParam::write_usqcd_info) is replaced by QUDA's line in its
       place, or left out if QUDA doesn't write it; only the first line
       of a key is kept.
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

  class GaugeField;

  /**
     @brief Compute the observables of a gauge field as a file would
     hold it: on its links in double or single precision (single for a
     half or quarter field), on the device for the plaquettes and the
     link trace, so after initQuda.  The links are taken with periodic
     boundaries, as they are in a file.  The checksum is exact, so it
     is the same on any rank layout and thread count.  Collective.
     @param[in] u The gauge field; any field GaugeIO::save accepts
     @param[in] convert_link_conventions As FileIOParam's: compute on
     the plain links of a field that records staggered phases or an
     antiperiodic t boundary (see GaugeIO), rather than on its links
     as they are
   */
  GaugeObservables gauge_observables(const GaugeField &u, bool convert_link_conventions = false);

  /**
     @brief GaugeIO loads and saves gauge fields.

     It takes any GaugeField of 3 colors and vector geometry holding
     links (QUDA_SU3_LINKS, QUDA_GENERAL_LINKS, QUDA_THREE_LINKS or
     QUDA_SMEARED_LINKS), in any location, order, precision and
     reconstruct GaugeField::copy supports.  The formats take a host
     copy in QDP order without reconstruct; a host QDP field of the
     file's precision (double or single, no reconstruct, not
     extended, and for loads without a ghost zone) is read or written
     in place, any other field through a temporary.

     Files hold plain links: no staggered phases, periodic t.  By
     default the links are moved to and from the file as they are,
     whatever the field records.  With
     FileIOParam::convert_link_conventions the links are converted on
     the host copy (never on the caller's field), and each conversion
     is announced:
     - the staggered phases (StaggeredPhase(): MILC, TIFR or CHROMA,
       as applyStaggeredPhase defines them) are in the links when
       StaggeredPhaseApplied() is set; long links (QUDA_THREE_LINKS)
       take MILC's, the phase of each of their three links;
     - the antiperiodic t boundary sign is in the links when
       TBoundary() is QUDA_ANTI_PERIODIC_T and either the field has
       no staggered phase type or its phases are applied (as
       applyStaggeredPhase applies the sign with the phases): on the
       t links of the last global time slice, or of the last three
       for long links.
     It is an error to convert phases with a nonzero iMu(), on long
     links with phases other than MILC's, with another phase type, or
     with an odd local extent in a dimension the phases depend on.
     Anisotropy, the tadpole factor and gauge fixing are never
     converted.
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
       @brief Load a gauge field from filename.  The links are read
       into a host copy of the field's precision (single for a half or
       quarter field), checked against the file's plaquette and link
       trace and reunitarized there, then converted (if requested) and
       copied into the field, whose ghost zone is then exchanged.  A
       reconstruct-12/8 field keeps the compressed links.  Extended
       fields are not supported.
       @param[out] u The gauge field; its lattice is the file's
       @return Metadata found in the file
    */
    GaugeFileInfo load(GaugeField &u);

    /**
       @brief Save a gauge field to filename.  An extended field saves
       its interior.
       @param[in] u The gauge field; its lattice is the file's
       @param[in] prec The precision of the host copy the links are
       saved from, double or single; the default is the field's, single
       for a half or quarter field.  SciDAC and ILDG files hold the
       links in this precision.  MILC files always hold the links
       rounded to single precision, so for them it doesn't change the
       file.
       @param[in] metadata Metadata to write, as a load reports it; see GaugeMetadata
    */
    void save(const GaugeField &u, QudaPrecision prec = QUDA_INVALID_PRECISION, const GaugeMetadata &metadata = {});
  };

} // namespace quda
