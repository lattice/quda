#include <algorithm>
#include <climits>
#include <cstdio>
#include <memory>
#include <string>
#include <util_quda.h>
#include <io/mpi_file.h>

#if defined(MPI_COMMS) || defined(QMP_COMMS)
#include <mpi.h>
#include <mpi_comm_handle.h>
#endif

namespace quda
{

  namespace io
  {

#if defined(MPI_COMMS) || defined(QMP_COMMS)

    struct MpiHandle {
      MPI_File fh = MPI_FILE_NULL;
      std::string filename;
    };

    namespace
    {
      // MPI file handles return their errors (MPI_ERRORS_RETURN is the default for files)
      void check(int rc, const char *operation, const std::string &filename)
      {
        if (rc == MPI_SUCCESS) return;
        char message[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        if (MPI_Error_string(rc, message, &length) != MPI_SUCCESS)
          snprintf(message, sizeof(message), "MPI error code %d", rc);
        errorQuda("MPI-IO %s of %s failed: %s", operation, filename.c_str(), message);
      }

      // the most bytes in one independent read or write, whose count MPI takes as an int
      constexpr size_t max_independent_bytes = size_t(1) << 30;
    } // namespace

    MpiHandle *mpi_open(const std::string &filename, bool create)
    {
      auto file = std::make_unique<MpiHandle>();
      file->filename = filename;
      const int amode = create ? MPI_MODE_WRONLY | MPI_MODE_CREATE : MPI_MODE_RDONLY;
      check(MPI_File_open(get_mpi_handle(), filename.c_str(), amode, MPI_INFO_NULL, &file->fh), "open", filename);
      if (create) {
        check(MPI_File_set_size(file->fh, 0), "truncation", filename);
        // the truncation is a write that conflicts with every rank's later writes (MPI-3.1 13.6.9), so in nonatomic
        // mode it must be synced, and complete on every rank, before any of them writes (13.6.1)
        check(MPI_File_sync(file->fh), "sync", filename);
        check(MPI_Barrier(get_mpi_handle()), "barrier", filename);
        check(MPI_File_sync(file->fh), "sync", filename);
      }
      return file.release();
    }

    void mpi_close(MpiHandle &file) { check(MPI_File_close(&file.fh), "close", file.filename); }

    void mpi_free(MpiHandle *file)
    {
      if (file && file->fh != MPI_FILE_NULL) MPI_File_close(&file->fh);
      delete file;
    }

    void mpi_read_at(const MpiHandle &file, void *data, size_t bytes, size_t offset)
    {
      auto p = static_cast<char *>(data);
      for (size_t done = 0; done < bytes;) {
        const int n = static_cast<int>(std::min(max_independent_bytes, bytes - done));
        MPI_Status status;
        check(MPI_File_read_at(file.fh, static_cast<MPI_Offset>(offset + done), p + done, n, MPI_BYTE, &status), "read",
              file.filename);
        int count = 0;
        check(MPI_Get_count(&status, MPI_BYTE, &count), "read", file.filename);
        if (count != n) errorQuda("Unexpected end of file %s", file.filename.c_str());
        done += n;
      }
    }

    void mpi_write_at(const MpiHandle &file, const void *data, size_t bytes, size_t offset)
    {
      auto p = static_cast<const char *>(data);
      for (size_t done = 0; done < bytes;) {
        const int n = static_cast<int>(std::min(max_independent_bytes, bytes - done));
        MPI_Status status;
        check(MPI_File_write_at(file.fh, static_cast<MPI_Offset>(offset + done), p + done, n, MPI_BYTE, &status),
              "write", file.filename);
        int count = 0;
        check(MPI_Get_count(&status, MPI_BYTE, &count), "write", file.filename);
        if (count != n) errorQuda("MPI-IO wrote %d of %d bytes to %s", count, n, file.filename.c_str());
        done += n;
      }
    }

    size_t mpi_size(const MpiHandle &file)
    {
      MPI_Offset size = 0;
      check(MPI_File_get_size(file.fh, &size), "size query", file.filename);
      return static_cast<size_t>(size);
    }

    void mpi_access(const MpiHandle &file, bool write, char *buffer, size_t offset, const std::array<int, 4> &region,
                    const std::array<int, 4> &lo, const std::array<int, 4> &extent, size_t site_bytes)
    {
      const auto &name = file.filename;
      if (site_bytes == 0 || site_bytes > INT_MAX) errorQuda("Invalid site size %lu for MPI-IO", site_bytes);
      size_t sites = 1;
      for (int d = 0; d < 4; d++) sites *= std::max(extent[d], 0);
      if (sites > INT_MAX)
        errorQuda("This rank's %lu sites of %s are more than one MPI-IO call can transfer", sites, name.c_str());

      // the view's elementary type is a site, the same on every rank; a rank with no sites keeps it as its file type
      MPI_Datatype site, box = MPI_DATATYPE_NULL;
      check(MPI_Type_contiguous(static_cast<int>(site_bytes), MPI_BYTE, &site), "site datatype", name);
      check(MPI_Type_commit(&site), "site datatype", name);
      if (sites > 0) {
        check(MPI_Type_create_subarray(4, region.data(), extent.data(), lo.data(), MPI_ORDER_FORTRAN, site, &box),
              "subarray datatype", name);
        check(MPI_Type_commit(&box), "subarray datatype", name);
      }
      check(MPI_File_set_view(file.fh, static_cast<MPI_Offset>(offset), site, sites > 0 ? box : site, "native",
                              MPI_INFO_NULL),
            "view", name);

      const int count = static_cast<int>(sites);
      MPI_Status status;
      if (write)
        check(MPI_File_write_all(file.fh, buffer, count, site, &status), "collective write", name);
      else
        check(MPI_File_read_all(file.fh, buffer, count, site, &status), "collective read", name);
      // ROMIO leaves the status of a rank that transfers nothing unset
      if (count > 0) {
        int done = 0;
        check(MPI_Get_count(&status, site, &done), write ? "collective write" : "collective read", name);
        if (done != count)
          errorQuda("MPI-IO %s %d of this rank's %d sites of %s", write ? "wrote" : "read", done, count, name.c_str());
      }

      // the independent reads and writes take byte offsets from the start of the file
      check(MPI_File_set_view(file.fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL), "view", name);
      if (box != MPI_DATATYPE_NULL) MPI_Type_free(&box);
      MPI_Type_free(&site);
    }

#else

    struct MpiHandle {
    };

    namespace
    {
      void no_mpi() { errorQuda("MPI-IO needs QUDA built with QUDA_MPI or QUDA_QMP"); }
    } // namespace

    MpiHandle *mpi_open(const std::string &, bool)
    {
      no_mpi();
      return nullptr;
    }

    void mpi_close(MpiHandle &) { no_mpi(); }

    void mpi_free(MpiHandle *file) { delete file; }

    void mpi_read_at(const MpiHandle &, void *, size_t, size_t) { no_mpi(); }

    void mpi_write_at(const MpiHandle &, const void *, size_t, size_t) { no_mpi(); }

    size_t mpi_size(const MpiHandle &)
    {
      no_mpi();
      return 0;
    }

    void mpi_access(const MpiHandle &, bool, char *, size_t, const std::array<int, 4> &, const std::array<int, 4> &,
                    const std::array<int, 4> &, size_t)
    {
      no_mpi();
    }

#endif

  } // namespace io

} // namespace quda
