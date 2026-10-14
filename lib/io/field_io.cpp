#include <cstdint>
#include <cstdio>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/field_io.h>

// defined in interface_quda.cpp from check_params.h
void checkFileIOParam(QudaFileIOParam &param);

namespace quda
{

  FileIOParam::FileIOParam(const QudaFileIOParam *param)
  {
    if (!param) return;

    QudaFileIOParam checked = *param;
    checkFileIOParam(checked);

    switch (checked.format) {
    case QUDA_FILE_FORMAT_QIO_EXTERNAL: format = QUDA_FILE_FORMAT_QIO; break;
    case QUDA_FILE_FORMAT_ILDG_EXTERNAL: format = QUDA_FILE_FORMAT_ILDG; break;
    default: format = checked.format;
    }
    external_qio = format != checked.format;
#ifndef HAVE_QIO
    if (external_qio)
      errorQuda("The %s file format needs the external QIO library, which requires building QUDA with QUDA_QIO=ON",
                get_file_format_str(checked.format));
#endif
    io_mode = checked.io_mode;
    ildg_lfn = checked.ildg_lfn;
    write_usqcd_info = checked.write_usqcd_info == QUDA_BOOLEAN_TRUE;
    allow_checksum_mismatch = checked.allow_checksum_mismatch == QUDA_BOOLEAN_TRUE;
  }

  const char *get_file_format_str(QudaFileFormat format)
  {
    switch (format) {
    case QUDA_FILE_FORMAT_AUTO: return "auto";
    case QUDA_FILE_FORMAT_QIO: return "QIO";
    case QUDA_FILE_FORMAT_MILC: return "MILC";
    case QUDA_FILE_FORMAT_ILDG: return "ILDG";
    case QUDA_FILE_FORMAT_QIO_EXTERNAL: return "QIO_external";
    case QUDA_FILE_FORMAT_ILDG_EXTERNAL: return "ILDG_external";
    default: return "unknown";
    }
  }

  namespace
  {
    constexpr uint32_t lime_magic = 0x456789ab;       // LIME headers are always big endian
    constexpr uint32_t milc_gauge_magic = 0x00004e87; // MILC v5 gauge files use the writer's byte order

    /**
       @brief Read the first four bytes of a file as a big-endian integer
       @return false if the file cannot be opened or is too short
     */
    bool read_magic(const std::string &filename, uint32_t &magic)
    {
      FILE *fp = fopen(filename.c_str(), "rb");
      if (!fp) return false;
      unsigned char b[4];
      bool ok = fread(b, 1, 4, fp) == 4;
      fclose(fp);
      if (ok) magic = (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
      return ok;
    }
  } // namespace

  QudaFileFormat detect_file_format(const std::string &filename)
  {
    int format = QUDA_FILE_FORMAT_INVALID;

    if (comm_rank() == 0) {
      uint32_t magic = 0;
      // QIO partfiles are named <filename>.volNNNN, except on a single rank where QIO omits the suffix
      if (read_magic(filename, magic) || read_magic(filename + ".vol0000", magic)) {
        if (magic == lime_magic)
          format = QUDA_FILE_FORMAT_QIO;
        else if (magic == milc_gauge_magic || __builtin_bswap32(magic) == milc_gauge_magic)
          format = QUDA_FILE_FORMAT_MILC;
      }
    }
    comm_broadcast(&format, sizeof(format));

    return static_cast<QudaFileFormat>(format);
  }

  QudaFileFormat resolve_load_format(const std::string &filename, QudaFileFormat requested)
  {
    auto detected = detect_file_format(filename);

    if (requested == QUDA_FILE_FORMAT_AUTO) {
      if (detected == QUDA_FILE_FORMAT_INVALID)
        errorQuda("File %s is missing or its format could not be determined", filename.c_str());
      return detected;
    }

    // ILDG files are LIME files, so they cannot be told apart from SciDAC files by their first bytes
    auto expected = requested == QUDA_FILE_FORMAT_ILDG ? QUDA_FILE_FORMAT_QIO : requested;
    // MILC KS eigenvector files carry no magic number
    bool undetectable = requested == QUDA_FILE_FORMAT_MILC && detected == QUDA_FILE_FORMAT_INVALID;

    if (detected != expected && !undetectable) {
      if (detected == QUDA_FILE_FORMAT_INVALID)
        errorQuda("File %s is missing or is not a %s file", filename.c_str(), get_file_format_str(requested));
      errorQuda("File %s was requested as %s but appears to be %s", filename.c_str(), get_file_format_str(requested),
                get_file_format_str(detected));
    }

    return requested;
  }

} // namespace quda
