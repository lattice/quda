#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/host_field.h>
#include <io/milc_io.h>

namespace quda
{

  namespace milc
  {

    using io::byte_swap;

    namespace
    {
      constexpr uint32_t gauge_magic = 20103; // 0x4e87
      constexpr int n_dim = 4;
      constexpr int matrix_reals = 18;

      struct GaugeHeader {
        uint32_t magic;
        uint32_t dims[4];
        char time_stamp[64];
        uint32_t order; // 0: sites in lexicographic order; otherwise a site list follows (checkpoint files)
        uint32_t sum29;
        uint32_t sum31;
      };
      static_assert(sizeof(GaugeHeader) == 96, "The MILC v5 gauge header is 96 bytes");

      /**
         @brief Check a .info file, if there is one, against the file's header, and pick up any
         plaquette and link trace it records
       */
      void check_info(const std::string &filename, const GaugeHeader &header, const FileIOParam &param,
                      GaugeFileInfo &info)
      {
        auto lines = read_info(filename);
        if (!lines) return;
        info.has_info_file = true;
        info.metadata.milc_info = *lines;
        const auto items = info_values(*lines);

        auto get = [&](const char *key) -> const std::string * {
          auto it = items.find(key);
          return it == items.end() ? nullptr : &it->second;
        };

        if (auto s = get("checksums")) {
          unsigned int sum29, sum31;
          if (sscanf(s->c_str(), "%x %x", &sum29, &sum31) != 2)
            errorQuda("Cannot read the checksums \"%s\" in %s.info", io::excerpt(*s).c_str(), filename.c_str());
          io::Checksum listed = {sum29, sum31}, stored = {header.sum29, header.sum31};
          if (listed != stored) {
            info.checksum_matches = false;
            io::report_checksum_mismatch("The checksums " + io::checksum_str(listed) + " in " + filename
                                           + ".info do not match those of the file, " + io::checksum_str(stored),
                                         param.allow_checksum_mismatch);
          }
        }

        const char *dim_keys[] = {"nx", "ny", "nz", "nt"};
        for (int d = 0; d < n_dim; d++) {
          if (auto s = get(dim_keys[d])) {
            if (std::strtol(s->c_str(), nullptr, 10) != static_cast<long>(header.dims[d]))
              errorQuda("%s = %s in %s.info does not match the file's dimension %u", dim_keys[d],
                        io::excerpt(*s).c_str(), filename.c_str(), header.dims[d]);
          }
        }

        if (auto s = get("time_stamp")) {
          std::string stamp(header.time_stamp, strnlen(header.time_stamp, sizeof(header.time_stamp)));
          if (*s != stamp)
            warningQuda("The time stamp \"%s\" in %s.info does not match the file's \"%s\"", io::excerpt(*s).c_str(),
                        filename.c_str(), stamp.c_str());
        }

        // MILC records the plaquettes as Re Tr (not divided by 3), averaged over spatial or temporal planes
        auto ssplaq = get("gauge.ssplaq");
        auto stplaq = get("gauge.stplaq");
        auto linktr = get("gauge.nersc_linktr");
        if (ssplaq && stplaq && linktr) {
          info.has_plaq_linktr = true;
          info.plaq = (std::strtod(ssplaq->c_str(), nullptr) + std::strtod(stplaq->c_str(), nullptr)) / 6.0;
          info.linktr = std::strtod(linktr->c_str(), nullptr);
        }
      }

      /**
         @brief The values of gauge.ssplaq, gauge.stplaq, gauge.nersc_linktr and gauge.nersc_checksum for the
         links as a MILC v5 file holds them, in single precision, formatted as MILC's write_gauge_info_item
         writes them (generic/io_lat_utils.c): each value as a float, with %f, %f, %e and %u.  Collective.
       */
      std::array<std::string, 4> computed_info(const void *const gauge[], QudaPrecision prec, const int *X)
      {
        // computed in double precision on the values rounded to float, which single-precision links are already
        const size_t reals = static_cast<size_t>(X[0]) * X[1] * X[2] * X[3] * matrix_reals;
        std::vector<std::vector<double>> rounded(n_dim, std::vector<double>(reals));
        for (int dir = 0; dir < n_dim; dir++) {
          auto u = rounded[dir].data();
          const auto d = static_cast<const double *>(gauge[dir]);
          const auto f = static_cast<const float *>(gauge[dir]);
#pragma omp parallel for
          for (size_t i = 0; i < reals; i++) u[i] = prec == QUDA_DOUBLE_PRECISION ? static_cast<float>(d[i]) : f[i];
        }
        const void *links[n_dim] = {rounded[0].data(), rounded[1].data(), rounded[2].data(), rounded[3].data()};
        const auto obs = gauge_observables(links, QUDA_DOUBLE_PRECISION, X);

        std::array<std::string, 4> values;
        auto print = [](const char *fmt, auto value) {
          char buf[64];
          snprintf(buf, sizeof(buf), fmt, value);
          return std::string(buf);
        };
        values[0] = print("%f", static_cast<double>(static_cast<float>(3.0 * obs.plaq_spatial)));
        values[1] = print("%f", static_cast<double>(static_cast<float>(3.0 * obs.plaq_temporal)));
        values[2] = print("%e", static_cast<double>(static_cast<float>(obs.linktr)));
        values[3] = print("%u", static_cast<unsigned int>(obs.nersc_checksum));
        return values;
      }
    } // namespace

    GaugeFileInfo load_gauge(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                             const FileIOParam &param)
    {
      io::HostField field(gauge, n_dim, prec, matrix_reals);

      GaugeHeader header;
      auto file_size = io::read_header(filename, &header, sizeof(header));

      bool swap = false;
      if (byte_swap(header.magic) == gauge_magic)
        swap = true;
      else if (header.magic != gauge_magic)
        errorQuda("%s is not a MILC v5 gauge file (magic number %#x)", filename.c_str(), header.magic);
      if (swap) {
        for (auto &n : header.dims) n = byte_swap(n);
        header.order = byte_swap(header.order);
        header.sum29 = byte_swap(header.sum29);
        header.sum31 = byte_swap(header.sum31);
      }

      if (header.order != 0)
        errorQuda("%s has a site list (order %u): MILC checkpoint files are not supported", filename.c_str(),
                  header.order);

      auto N = io::global_dims(X);
      for (int d = 0; d < n_dim; d++) {
        if (header.dims[d] != static_cast<uint32_t>(N[d]))
          errorQuda("%s has dimensions %u %u %u %u, but the lattice is %d %d %d %d", filename.c_str(), header.dims[0],
                    header.dims[1], header.dims[2], header.dims[3], N[0], N[1], N[2], N[3]);
      }

      // each site holds the four links as single-precision row-major 3x3 complex matrices
      auto layout = field.layout(QUDA_SINGLE_PRECISION, sizeof(GaugeHeader), swap, io::ChecksumType::MILC);
      auto expected_size = layout.offset + io::data_bytes(layout, N);
      if (file_size != expected_size)
        errorQuda("%s is %lu bytes, but a MILC v5 gauge file of this size should be %lu bytes", filename.c_str(),
                  file_size, expected_size);

      // checks the grid, also in MPI-IO mode, whose reads don't use the partitions
      const io::Partitions partitions(param.io_mode, param.io_partition_grid);
      io::Checksum sum;
      if (param.io_mode == QUDA_FILE_IO_MPI_IO) {
        io::File file(filename, io::File::Mode::READ, true);
        sum = io::read_data(file, layout, X, field.unpacker(QUDA_SINGLE_PRECISION));
        file.close();
      } else {
        sum = io::read_data(filename, layout, X, partitions, field.unpacker(QUDA_SINGLE_PRECISION));
      }

      GaugeFileInfo info;
      info.format = QUDA_FILE_FORMAT_MILC;
      info.file_prec = QUDA_SINGLE_PRECISION;
      info.byte_swapped = swap;
      info.sum29 = sum.sum29;
      info.sum31 = sum.sum31;
      info.has_checksum = true;
      info.checksum_matches
        = io::check_checksum(filename, sum, {header.sum29, header.sum31}, param.allow_checksum_mismatch);
      check_info(filename, header, param, info);

      info.metadata.milc_time_stamp
        = std::string(header.time_stamp, strnlen(header.time_stamp, sizeof(header.time_stamp)));
      logQuda(QUDA_VERBOSE, "MILC v5 time stamp \"%s\"\n", info.metadata.milc_time_stamp.c_str());
      return info;
    }

    io::Checksum save_gauge(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                            const FileIOParam &param, const GaugeMetadata &metadata)
    {
      // checks the grid and the metadata before the file is truncated
      const io::Partitions partitions(param.io_mode, param.io_partition_grid);
      if (metadata.milc_time_stamp.size() > sizeof(GaugeHeader::time_stamp))
        errorQuda("The time stamp \"%s\" given for %s is longer than the %lu bytes of a MILC v5 header",
                  io::excerpt(metadata.milc_time_stamp).c_str(), filename.c_str(), sizeof(GaugeHeader::time_stamp));
      for (auto &item : metadata.milc_info)
        if (item.line.find('\n') != std::string::npos)
          errorQuda("The .info line \"%s\" given for %s holds a newline", io::excerpt(item.line).c_str(),
                    filename.c_str());

      io::HostField field(gauge, n_dim, prec, matrix_reals);
      if (prec == QUDA_DOUBLE_PRECISION)
        logQuda(QUDA_SUMMARIZE,
                "MILC v5 gauge files are single precision, so the double-precision field is saved in "
                "single precision\n");

      auto N = io::global_dims(X);
      GaugeHeader header = {};
      header.magic = gauge_magic;
      for (int d = 0; d < n_dim; d++) header.dims[d] = N[d];

      std::string stamp = metadata.milc_time_stamp;
      if (comm_rank() == 0) {
        if (stamp.empty()) stamp = time_stamp();
        memcpy(header.time_stamp, stamp.c_str(), std::min(stamp.size(), sizeof(header.time_stamp)));
      }

      // rank 0 writes the header, and the checksums into it after the data
      auto layout = field.layout(QUDA_SINGLE_PRECISION, sizeof(GaugeHeader), false, io::ChecksumType::MILC);
      io::Checksum sum;
      auto write_checksums = [&](const io::File &file) {
        uint32_t sums[2] = {sum.sum29, sum.sum31};
        file.write(sums, sizeof(sums), offsetof(GaugeHeader, sum29));
      };
      if (param.io_mode == QUDA_FILE_IO_MPI_IO) {
        io::File file(filename, io::File::Mode::CREATE, true);
        if (comm_rank() == 0) file.write(&header, sizeof(header), 0);
        sum = io::write_data(file, layout, X, field.packer(QUDA_SINGLE_PRECISION));
        if (comm_rank() == 0) write_checksums(file);
        file.close();
        // every rank's data is in the file before anyone reads it
        comm_barrier();
      } else {
        if (comm_rank() == 0) {
          io::File file(filename, io::File::Mode::CREATE);
          file.write(&header, sizeof(header), 0);
          file.close();
        }
        sum = io::write_data(filename, layout, X, partitions, field.packer(QUDA_SINGLE_PRECISION));
        if (comm_rank() == 0) {
          io::File file(filename, io::File::Mode::WRITE);
          write_checksums(file);
          file.close();
        }
      }

      std::vector<std::pair<std::string, std::string>> items = {{"magic_number", std::to_string(gauge_magic)},
                                                                {"time_stamp", "\"" + stamp + "\""},
                                                                {"checksums", "\"" + io::checksum_str(sum) + "\""},
                                                                {"nx", std::to_string(N[0])},
                                                                {"ny", std::to_string(N[1])},
                                                                {"nz", std::to_string(N[2])},
                                                                {"nt", std::to_string(N[3])}};
      const std::vector<std::string> computed_keys
        = {"gauge.ssplaq", "gauge.stplaq", "gauge.nersc_linktr", "gauge.nersc_checksum"};
      if (param.write_usqcd_info) {
        const auto values = computed_info(gauge, prec, X);
        for (int k = 0; k < 4; k++) items.push_back({computed_keys[k], values[k]});
        logQuda(QUDA_SUMMARIZE, "MILC .info plaquettes %s %s, link trace %s, NERSC checksum %s\n", values[0].c_str(),
                values[1].c_str(), values[2].c_str(), values[3].c_str());
      }
      write_info(filename, items, metadata.milc_info,
                 param.write_usqcd_info ? std::vector<std::string>() : computed_keys);

      return sum;
    }

  } // namespace milc

} // namespace quda
