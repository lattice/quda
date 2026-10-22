#pragma once

// Collective MPI-IO access to single files (QUDA_FILE_IO_MPI_IO): an open MPI file, the independent reads and
// writes of rank 0's metadata, and the collective read or write of each rank's box of a record's data

#include <array>
#include <cstddef>
#include <string>

namespace quda
{

  namespace io
  {

    /**
       @brief A file opened through MPI-IO by every rank of QUDA's
       communicator; defined in mpi_file.cpp
     */
    struct MpiHandle;

    /**
       @brief Open a file through MPI-IO, collectively.  Errors in
       builds without MPI_COMMS or QMP_COMMS.
       @param[in] filename The file
       @param[in] create Whether to create it write-only, truncating it
       to zero bytes if it exists (in place, so that it keeps its
       striping), rather than open it read-only
     */
    MpiHandle *mpi_open(const std::string &filename, bool create);

    /**
       @brief Close a file opened with mpi_open, collectively, raising
       an error if that fails
     */
    void mpi_close(MpiHandle &file);

    /**
       @brief Close a file if it is still open, ignoring errors, and
       free it; the deleter of a File's handle.  Closing is collective,
       so destroying a File that is still open is collective too.
     */
    void mpi_free(MpiHandle *file);

    /**
       @brief Read bytes at a byte offset on the calling rank alone
     */
    void mpi_read_at(const MpiHandle &file, void *data, size_t bytes, size_t offset);

    /**
       @brief Write bytes at a byte offset on the calling rank alone
     */
    void mpi_write_at(const MpiHandle &file, const void *data, size_t bytes, size_t offset);

    /**
       @brief The size of the file in bytes
     */
    size_t mpi_size(const MpiHandle &file);

    /**
       @brief Read or write each rank's box of a region of sites with
       one collective call.  The region is stored at offset in
       lexicographic order with x running fastest, and each rank's
       buffer holds its box of it in the same order.  Every rank must
       call it, also those whose box is empty.
       @param[in] file The file
       @param[in] write Whether to write rather than read
       @param[in,out] buffer The rank's sites, site_bytes each
       @param[in] offset Byte offset of the region's first site
       @param[in] region The region's extent in each dimension
       @param[in] lo The lowest corner of the rank's box, relative to the region's
       @param[in] extent The extent of the rank's box; any zero means the rank has no sites
       @param[in] site_bytes Bytes per site
     */
    void mpi_access(const MpiHandle &file, bool write, char *buffer, size_t offset, const std::array<int, 4> &region,
                    const std::array<int, 4> &lo, const std::array<int, 4> &extent, size_t site_bytes);

  } // namespace io

} // namespace quda
