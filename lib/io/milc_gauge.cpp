#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/milc_io.h>

namespace quda
{

  namespace milc
  {

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

      DataLayout gauge_layout()
      {
        DataLayout layout;
        layout.offset = sizeof(GaugeHeader);
        layout.site_bytes = n_dim * matrix_reals * sizeof(float);
        layout.word_bytes = sizeof(float);
        return layout;
      }

      // each site holds the four links as single-precision row-major 3x3 complex matrices
      template <typename Float> void unpack_links(void *const gauge[], size_t index, const char *site)
      {
        for (int d = 0; d < n_dim; d++) {
          float link[matrix_reals];
          memcpy(link, site + d * sizeof(link), sizeof(link));
          auto dst = static_cast<Float *>(gauge[d]) + index * matrix_reals;
          for (int i = 0; i < matrix_reals; i++) dst[i] = link[i];
        }
      }

      template <typename Float> void pack_links(const void *const gauge[], size_t index, char *site)
      {
        for (int d = 0; d < n_dim; d++) {
          float link[matrix_reals];
          auto src = static_cast<const Float *>(gauge[d]) + index * matrix_reals;
          for (int i = 0; i < matrix_reals; i++) link[i] = src[i];
          memcpy(site + d * sizeof(link), link, sizeof(link));
        }
      }

      std::string checksum_str(uint32_t sum29, uint32_t sum31)
      {
        char buf[20];
        snprintf(buf, sizeof(buf), "%x %x", sum29, sum31);
        return buf;
      }

      void report_checksum_mismatch(const std::string &message, const FileIOParam &param)
      {
        if (param.allow_checksum_mismatch)
          warningQuda("%s", message.c_str());
        else
          errorQuda("%s (set allow_checksum_mismatch to load the file anyway)", message.c_str());
      }

      /**
         @brief Check a .info file, if there is one, against the file's header, and pick up any
         plaquette and link trace it records
       */
      void check_info(const std::string &filename, const GaugeHeader &header, const FileIOParam &param,
                      GaugeFileInfo &info)
      {
        auto items = read_info(filename);
        if (!items) return;
        info.has_info_file = true;

        auto get = [&](const char *key) -> const std::string * {
          auto it = items->find(key);
          return it == items->end() ? nullptr : &it->second;
        };

        if (auto s = get("checksums")) {
          unsigned int sum29, sum31;
          if (sscanf(s->c_str(), "%x %x", &sum29, &sum31) != 2)
            errorQuda("Cannot read the checksums \"%s\" in %s.info", s->c_str(), filename.c_str());
          if (sum29 != header.sum29 || sum31 != header.sum31) {
            info.checksum_matches = false;
            report_checksum_mismatch("The checksums " + checksum_str(sum29, sum31) + " in " + filename
                                       + ".info do not match those of the file, "
                                       + checksum_str(header.sum29, header.sum31),
                                     param);
          }
        }

        const char *dim_keys[] = {"nx", "ny", "nz", "nt"};
        for (int d = 0; d < n_dim; d++) {
          if (auto s = get(dim_keys[d])) {
            if (std::strtol(s->c_str(), nullptr, 10) != static_cast<long>(header.dims[d]))
              errorQuda("%s = %s in %s.info does not match the file's dimension %u", dim_keys[d], s->c_str(),
                        filename.c_str(), header.dims[d]);
          }
        }

        if (auto s = get("time_stamp")) {
          std::string stamp(header.time_stamp, strnlen(header.time_stamp, sizeof(header.time_stamp)));
          if (*s != stamp)
            warningQuda("The time stamp \"%s\" in %s.info does not match the file's \"%s\"", s->c_str(),
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
    } // namespace

    GaugeFileInfo load_gauge(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                             const FileIOParam &param)
    {
      if (prec != QUDA_DOUBLE_PRECISION && prec != QUDA_SINGLE_PRECISION)
        errorQuda("Unsupported host gauge field precision %d", prec);

      GaugeHeader header;
      auto file_size = read_header(filename, &header, sizeof(header));

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

      auto N = global_dims(X);
      for (int d = 0; d < n_dim; d++) {
        if (header.dims[d] != static_cast<uint32_t>(N[d]))
          errorQuda("%s has dimensions %u %u %u %u, but the lattice is %d %d %d %d", filename.c_str(), header.dims[0],
                    header.dims[1], header.dims[2], header.dims[3], N[0], N[1], N[2], N[3]);
      }

      auto layout = gauge_layout();
      auto expected_size = layout.offset + data_bytes(layout, N);
      if (file_size != expected_size)
        errorQuda("%s is %lu bytes, but a MILC v5 gauge file of this size should be %lu bytes", filename.c_str(),
                  file_size, expected_size);

      auto unpack = [&](int, size_t index, const char *site) {
        if (prec == QUDA_DOUBLE_PRECISION)
          unpack_links<double>(gauge, index, site);
        else
          unpack_links<float>(gauge, index, site);
      };
      auto sum = read_data(filename, layout, X, swap, param.io_mode, unpack);

      GaugeFileInfo info;
      info.format = QUDA_FILE_FORMAT_MILC;
      info.file_prec = QUDA_SINGLE_PRECISION;
      info.byte_swapped = swap;
      info.sum29 = sum.sum29;
      info.sum31 = sum.sum31;
      info.checksum_matches = sum.sum29 == header.sum29 && sum.sum31 == header.sum31;
      if (!info.checksum_matches)
        report_checksum_mismatch("Checksum mismatch in " + filename + ": computed " + checksum_str(sum.sum29, sum.sum31)
                                   + ", but the file has " + checksum_str(header.sum29, header.sum31),
                                 param);
      check_info(filename, header, param, info);

      std::string stamp(header.time_stamp, strnlen(header.time_stamp, sizeof(header.time_stamp)));
      logQuda(QUDA_SUMMARIZE, "Loaded a MILC v5 gauge field%s with time stamp \"%s\"; checksums %x %x %s%s\n",
              swap ? " (byte swapped)" : "", stamp.c_str(), sum.sum29, sum.sum31,
              info.checksum_matches ? "verified" : "NOT verified",
              info.has_info_file ? ", along with the .info file" : "");
      return info;
    }

    void save_gauge(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                    const FileIOParam &param)
    {
      if (prec != QUDA_DOUBLE_PRECISION && prec != QUDA_SINGLE_PRECISION)
        errorQuda("Unsupported host gauge field precision %d", prec);
      if (prec == QUDA_DOUBLE_PRECISION)
        logQuda(QUDA_SUMMARIZE,
                "MILC v5 gauge files are single precision, so the double-precision field is saved in "
                "single precision\n");

      auto N = global_dims(X);
      GaugeHeader header = {};
      header.magic = gauge_magic;
      for (int d = 0; d < n_dim; d++) header.dims[d] = N[d];

      std::string stamp;
      if (comm_rank() == 0) {
        stamp = time_stamp();
        memcpy(header.time_stamp, stamp.c_str(), std::min(stamp.size(), sizeof(header.time_stamp) - 1));
        File file(filename, File::Mode::CREATE);
        file.write(&header, sizeof(header), 0);
        file.close();
      }

      auto pack = [&](int, size_t index, char *site) {
        if (prec == QUDA_DOUBLE_PRECISION)
          pack_links<double>(gauge, index, site);
        else
          pack_links<float>(gauge, index, site);
      };
      auto sum = write_data(filename, gauge_layout(), X, param.io_mode, pack);

      if (comm_rank() == 0) {
        uint32_t sums[2] = {sum.sum29, sum.sum31};
        File file(filename, File::Mode::WRITE);
        file.write(sums, sizeof(sums), offsetof(GaugeHeader, sum29));
        file.close();
      }

      write_info(filename,
                 {{"magic_number", std::to_string(gauge_magic)},
                  {"time_stamp", "\"" + stamp + "\""},
                  {"checksums", "\"" + checksum_str(sum.sum29, sum.sum31) + "\""},
                  {"nx", std::to_string(N[0])},
                  {"ny", std::to_string(N[1])},
                  {"nz", std::to_string(N[2])},
                  {"nt", std::to_string(N[3])}});

      logQuda(QUDA_SUMMARIZE, "Saved a MILC v5 gauge field with checksums %x %x\n", sum.sum29, sum.sum31);
    }

  } // namespace milc

} // namespace quda
