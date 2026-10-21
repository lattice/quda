#pragma once

#include <array>
#include <optional>
#include <string>
#include <quda.h>

namespace quda
{

  /** The sign convention of staggered fields saved by QUDA: QUDA's own, or MILC's */
  enum class StaggeredSignConvention { QUDA, MILC };

  /**
     @brief Internal options for loading and saving fields.  This is
     the C++ counterpart of the public QudaFileIOParam, plus the
     vector-specific options that the public interface keeps
     elsewhere (QudaEigParam::partfile, io_parity_inflate, etc.).
   */
  struct FileIOParam {
    QudaFileFormat format = QUDA_FILE_FORMAT_AUTO; /** AUTO, QIO, MILC or ILDG; never an _EXTERNAL value */
    bool external_qio = false;                     /** QIO and ILDG: use the external QIO library */
    QudaFileIOMode io_mode = QUDA_FILE_IO_PARALLEL;
    bool partfile = false;       /** QIO only, ignored on load */
    bool parity_inflate = false; /** Inflate single-parity vectors to full fields for I/O */
    std::string ildg_lfn;        /** Empty means use the file name */
    bool write_usqcd_info = true;
    bool allow_checksum_mismatch = false; /** Warn instead of error on a checksum mismatch */

    /** How to reunitarize a gauge field after loading it; ignored for saves and vectors */
    QudaGaugeReunitarize reunitarize = QUDA_GAUGE_REUNITARIZE_NONE;

    /** Number of I/O partitions in each dimension; all zeros means one per rank */
    std::array<int, 4> io_partition_grid {};

    /**
       Native partfile saves, for tests: the number of volumes in each
       dimension, as if saved by a process grid of that shape with one
       partition per rank, numbered as QUDA's default rank map numbers
       ranks (lexicographically, t fastest).
       Rank 0 creates them all, also on one rank.  All zeros means one
       volume per I/O partition, as QIO saves.
    */
    std::array<int, 4> partfile_grid {};

    /** Save vectors in MILC's SciDAC eigenvector format (QudaFileIOParam::milc_eigenvectors); ignored on load */
    bool milc_eigenvectors = false;

    /**
       Staggered vector saves in QUDA's format: the sign convention the vectors are in, recorded in each record's
       XML; none records nothing.  Ignored on load.
    */
    std::optional<StaggeredSignConvention> staggered_sign_convention;

    /**
       Native vector saves, for tests: all the vectors in one record,
       interleaved at each site, as QUDA saved them before it saved one
       record per vector
    */
    bool interleaved_vectors = false;

    /**
       SciDAC saves, for tests: the strings QUDA saved its files with
       before they had a file XML, USQCD datatypes and record XML of
       version 1.0 (the user file XML "Dummy user file XML", QUDA's own
       datatypes, and record XML of version BETA with a <type>).
       Implied by interleaved_vectors.
    */
    bool legacy_strings = false;

    FileIOParam() = default;

    /**
       @brief Construct from the public interface struct.  An _EXTERNAL
       format becomes the plain format with external_qio set.
       @param[in] param Public parameters, validated here; nullptr means defaults
     */
    explicit FileIOParam(const QudaFileIOParam *param);
  };

  /**
     @brief Return a string naming a file format, for messages
   */
  const char *get_file_format_str(QudaFileFormat format);

  /**
     @brief Return a string naming a reunitarization method, for messages
   */
  const char *get_gauge_reunitarize_str(QudaGaugeReunitarize reunitarize);

  /**
     @brief Return the name of a staggered sign convention ("QUDA" or "MILC"), as files record it
   */
  const char *get_staggered_sign_convention_str(StaggeredSignConvention convention);

  /**
     @brief Determine the format of an existing file from its
     contents.  Rank 0 inspects the file and broadcasts the result.
     @param[in] filename File to inspect; for QIO partfiles this may be
     the base name without the volume suffix
     @return The detected format, or QUDA_FILE_FORMAT_INVALID if it
     cannot be determined
   */
  QudaFileFormat detect_file_format(const std::string &filename);

  /**
     @brief Determine the format to use when loading a file.  AUTO
     returns the detected format; an explicit format is checked for
     consistency with the file contents.  Errors if the format cannot
     be determined or does not match.
     @param[in] filename File to be loaded
     @param[in] requested The requested format
     @return The format to use
   */
  QudaFileFormat resolve_load_format(const std::string &filename, QudaFileFormat requested);

  /**
     @brief Determine the format to use when saving a file (AUTO means QIO)
   */
  inline QudaFileFormat resolve_save_format(QudaFileFormat requested)
  {
    return requested == QUDA_FILE_FORMAT_AUTO ? QUDA_FILE_FORMAT_QIO : requested;
  }

} // namespace quda
