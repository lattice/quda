#pragma once

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <io/gauge_io.h>
#include <io/site_io.h>

namespace quda
{

  namespace milc
  {

    /**
       @brief The current local time in the format of ctime(), without the trailing newline
     */
    std::string time_stamp();

    /**
       @brief Write <filename>.info on rank 0, one "key = value" line per item
     */
    void write_info(const std::string &filename, const std::vector<std::pair<std::string, std::string>> &items);

    /**
       @brief Read <filename>.info if it exists.  Rank 0 reads it and
       broadcasts its contents.  Lines may have the form "key = value"
       or "key value".
       @return The lines that aren't blank in file order, each with its key, or nothing if there is no file
     */
    std::optional<std::vector<MILCInfoItem>> read_info(const std::string &filename);

    /**
       @brief The values of the lines of a .info file by key, with
       surrounding quotes removed; a key given more than once takes the
       value of its last line
     */
    std::map<std::string, std::string> info_values(const std::vector<MILCInfoItem> &items);

    /**
       @brief Load a MILC v5 gauge field
       @param[in] filename The file
       @param[out] gauge The host gauge field in QDP order, one array per direction
       @param[in] prec The precision of the host gauge field
       @param[in] X The local lattice dimensions
       @param[in] param File I/O options
       @return Metadata found in the file and any .info file
     */
    GaugeFileInfo load_gauge(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                             const FileIOParam &param);

    /**
       @brief Save a MILC v5 gauge field, and its .info file
       @param[in] filename The file
       @param[in] gauge The host gauge field in QDP order, one array per direction
       @param[in] prec The precision of the host gauge field
       @param[in] X The local lattice dimensions
       @param[in] param File I/O options
       @return The checksums written
     */
    io::Checksum save_gauge(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                            const FileIOParam &param);

  } // namespace milc

} // namespace quda
