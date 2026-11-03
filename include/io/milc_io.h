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
       @brief Write <filename>.info on rank 0: the other lines verbatim,
       in their order, and one "key = value" line per item.  The first
       other line with an item's key is replaced by the item's line, and
       later lines of that key are left out, as are lines with a key in
       dropped.  An item whose key no other line has goes right after
       the line of the item before it, or first if it is the first item,
       so runs of such items stay in the items' order.
       @param[in] items QUDA's keys and values, in QUDA's order
       @param[in] lines The other lines, without newlines
       @param[in] dropped The keys of other lines to leave out
     */
    void write_info(const std::string &filename, const std::vector<std::pair<std::string, std::string>> &items,
                    const std::vector<MILCInfoItem> &lines = {}, const std::vector<std::string> &dropped = {});

    /**
       @brief The lines of the text of a .info file, as read_info gives
       them: those that aren't blank, in order, each with its key.  Lines
       may have the form "key = value" or "key value".
     */
    std::vector<MILCInfoItem> info_items(const std::string &text);

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
       @brief The keys of the .info lines save_gauge always writes, in
       its order: magic_number, time_stamp, checksums, nx, ny, nz, nt
     */
    const std::vector<std::string> &generic_info_keys();

    /**
       @brief The keys of the .info lines save_gauge computes from the
       links with param.write_usqcd_info, in its order: gauge.ssplaq,
       gauge.stplaq, gauge.nersc_linktr, gauge.nersc_checksum
     */
    const std::vector<std::string> &computed_info_keys();

    /**
       @brief The keys of computed_info_keys and their values for a
       field's observables, formatted as MILC's write_gauge_info_item
       writes them (generic/io_lat_utils.c): each value as a float, with
       %f, %f, %e and %u
     */
    std::vector<std::pair<std::string, std::string>> computed_info_items(const GaugeObservables &obs);

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
       @brief Save a MILC v5 gauge field, and its .info file.  With
       param.write_usqcd_info, the .info file also holds MILC's
       gauge.ssplaq, gauge.stplaq, gauge.nersc_linktr and
       gauge.nersc_checksum of the links as saved, in single precision,
       which needs initQuda.
       @param[in] filename The file
       @param[in] gauge The host gauge field in QDP order, one array per direction
       @param[in] prec The precision of the host gauge field
       @param[in] X The local lattice dimensions
       @param[in] param File I/O options
       @param[in] metadata The time stamp and the other .info lines to write (see GaugeMetadata)
       @return The checksums written
     */
    io::Checksum save_gauge(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                            const FileIOParam &param, const GaugeMetadata &metadata = {});

  } // namespace milc

} // namespace quda
