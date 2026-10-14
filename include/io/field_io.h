#pragma once

#include <string>
#include <quda.h>

namespace quda
{

  /**
     @brief Internal options for loading and saving fields.  This is
     the C++ counterpart of the public QudaFileIOParam, plus the
     vector-specific options that the public interface keeps
     elsewhere (QudaEigParam::partfile, io_parity_inflate, etc.).
   */
  struct FileIOParam {
    QudaFileFormat format = QUDA_FILE_FORMAT_AUTO;
    QudaFileIOMode io_mode = QUDA_FILE_IO_PARALLEL;
    bool partfile = false;       /** QIO only, ignored on load */
    bool parity_inflate = false; /** Inflate single-parity vectors to full fields for I/O */
    std::string ildg_lfn;        /** Empty means use the file name */
    bool write_usqcd_info = true;
    bool allow_checksum_mismatch = false; /** MILC only: warn instead of error on a checksum mismatch */

    FileIOParam() = default;

    /**
       @brief Construct from the public interface struct
       @param[in] param Public parameters, validated here; nullptr means defaults
     */
    explicit FileIOParam(const QudaFileIOParam *param);
  };

  /**
     @brief Return a string naming a file format, for messages
   */
  const char *get_file_format_str(QudaFileFormat format);

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
