#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/site_io.h>

namespace quda
{

  namespace io
  {

    File::File(const std::string &filename, Mode mode) : filename(filename)
    {
      int flags = mode == Mode::READ ? O_RDONLY : mode == Mode::WRITE ? O_WRONLY : O_WRONLY | O_CREAT | O_TRUNC;
      fd = open(filename.c_str(), flags, 0644);
      if (fd < 0) errorQuda("Cannot open %s: %s", filename.c_str(), strerror(errno));
    }

    File::~File()
    {
      if (fd >= 0) ::close(fd);
    }

    void File::close()
    {
      if (fd >= 0 && ::close(fd) != 0) errorQuda("Error closing %s: %s", filename.c_str(), strerror(errno));
      fd = -1;
    }

    void File::read(void *data, size_t bytes, size_t offset) const
    {
      auto p = static_cast<char *>(data);
      while (bytes > 0) {
        ssize_t n = pread(fd, p, bytes, static_cast<off_t>(offset));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) errorQuda("Error reading %s: %s", filename.c_str(), strerror(errno));
        if (n == 0) errorQuda("Unexpected end of file %s", filename.c_str());
        p += n;
        bytes -= n;
        offset += n;
      }
    }

    void File::write(const void *data, size_t bytes, size_t offset) const
    {
      auto p = static_cast<const char *>(data);
      while (bytes > 0) {
        ssize_t n = pwrite(fd, p, bytes, static_cast<off_t>(offset));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) errorQuda("Error writing %s: %s", filename.c_str(), strerror(errno));
        p += n;
        bytes -= n;
        offset += n;
      }
    }

    size_t File::size() const
    {
      struct stat s;
      if (fstat(fd, &s) != 0) errorQuda("Cannot stat %s: %s", filename.c_str(), strerror(errno));
      return s.st_size;
    }

    void byte_swap(void *data, size_t n_words, size_t word_bytes)
    {
      auto p = static_cast<char *>(data);
      if (word_bytes == 4) {
        for (size_t i = 0; i < n_words; i++) {
          uint32_t w;
          memcpy(&w, p + 4 * i, 4);
          w = __builtin_bswap32(w);
          memcpy(p + 4 * i, &w, 4);
        }
      } else if (word_bytes == 8) {
        for (size_t i = 0; i < n_words; i++) {
          uint64_t w;
          memcpy(&w, p + 8 * i, 8);
          w = __builtin_bswap64(w);
          memcpy(p + 8 * i, &w, 8);
        }
      } else {
        errorQuda("Unsupported word size %lu", word_bytes);
      }
    }

    namespace
    {
      using crc_tables_t = std::array<std::array<uint32_t, 256>, 8>;

      /**
         @brief Tables for slicing-by-8: table[0][n] is the CRC of the byte n, and table[k][n]
         advances table[k - 1][n] by one zero byte
       */
      constexpr crc_tables_t make_crc_tables()
      {
        constexpr uint32_t polynomial = 0xedb88320;
        crc_tables_t table {};
        for (uint32_t n = 0; n < 256; n++) {
          uint32_t c = n;
          for (int bit = 0; bit < 8; bit++) c = (c >> 1) ^ (polynomial & (0u - (c & 1u)));
          table[0][n] = c;
        }
        for (int k = 1; k < 8; k++)
          for (int n = 0; n < 256; n++) table[k][n] = (table[k - 1][n] >> 8) ^ table[0][table[k - 1][n] & 0xff];
        return table;
      }

      constexpr crc_tables_t crc_tables = make_crc_tables();

      inline uint32_t load_le32(const unsigned char *p)
      {
        return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
      }
    } // namespace

    uint32_t crc32(const void *data, size_t bytes, uint32_t crc)
    {
      const auto &t = crc_tables;
      auto p = static_cast<const unsigned char *>(data);
      crc = ~crc;
      for (; bytes >= 8; bytes -= 8, p += 8) {
        uint32_t a = crc ^ load_le32(p);
        uint32_t b = load_le32(p + 4);
        crc = t[7][a & 0xff] ^ t[6][(a >> 8) & 0xff] ^ t[5][(a >> 16) & 0xff] ^ t[4][a >> 24] ^ t[3][b & 0xff]
          ^ t[2][(b >> 8) & 0xff] ^ t[1][(b >> 16) & 0xff] ^ t[0][b >> 24];
      }
      for (; bytes > 0; bytes--, p++) crc = (crc >> 8) ^ t[0][(crc ^ *p) & 0xff];
      return ~crc;
    }

    size_t read_header(const std::string &filename, void *data, size_t bytes)
    {
      size_t size = 0;
      if (comm_rank() == 0) {
        File file(filename, File::Mode::READ);
        size = file.size();
        if (size < bytes) errorQuda("File %s is too short (%lu bytes)", filename.c_str(), size);
        file.read(data, bytes, 0);
      }
      comm_broadcast(&size, sizeof(size));
      comm_broadcast(data, bytes);
      return size;
    }

    std::array<int, 4> global_dims(const int *X)
    {
      std::array<int, 4> N;
      for (int d = 0; d < 4; d++) N[d] = X[d] * comm_dim(d);
      return N;
    }

    size_t data_bytes(const DataLayout &layout, const std::array<int, 4> &dims)
    {
      size_t sites = static_cast<size_t>(dims[0]) * dims[1] * dims[2] * dims[3];
      if (layout.parity == QUDA_EVEN_PARITY || layout.parity == QUDA_ODD_PARITY) sites /= 2;
      return layout.n_block * sites * layout.site_bytes;
    }

    std::string checksum_str(const Checksum &sum)
    {
      char buf[20];
      snprintf(buf, sizeof(buf), "%x %x", sum.sum29, sum.sum31);
      return buf;
    }

    void report_checksum_mismatch(const std::string &message, bool allow)
    {
      if (allow)
        warningQuda("%s", message.c_str());
      else
        errorQuda("%s (set allow_checksum_mismatch to load the file anyway)", message.c_str());
    }

    bool check_checksum(const std::string &filename, const Checksum &computed, const Checksum &stored, bool allow)
    {
      if (computed == stored) return true;
      report_checksum_mismatch("Checksum mismatch in " + filename + ": computed " + checksum_str(computed)
                                 + ", but the file has " + checksum_str(stored),
                               allow);
      return false;
    }

    namespace
    {
      using coords_t = std::array<int, 4>;

      /**
         @brief Where each rank's sites are in the file, and in QUDA's host ordering
       */
      struct Geometry {
        const DataLayout &layout;
        coords_t X;
        coords_t N;
        bool single_parity;
        size_t volume;     // local sites
        size_t row_sites;  // stored sites per local x row
        size_t n_row;      // local x rows
        size_t file_sites; // stored sites per block in the file

        Geometry(const DataLayout &layout, const int *X_) :
          layout(layout), single_parity(layout.parity == QUDA_EVEN_PARITY || layout.parity == QUDA_ODD_PARITY)
        {
          for (int d = 0; d < 4; d++) X[d] = X_[d];
          N = global_dims(X_);
          if (X[0] % 2 != 0) errorQuda("The local x dimension %d must be even", X[0]);
          if (layout.site_bytes % 4 != 0 || layout.site_bytes % layout.word_bytes != 0)
            errorQuda("Invalid site size %lu with word size %lu", layout.site_bytes, layout.word_bytes);
          // the SciDAC checksum's site index restarts in every record
          if (layout.n_block > 1 && layout.checksum == ChecksumType::SCIDAC)
            errorQuda("A SciDAC checksum covers one record; read or write each record with its own call");
          volume = static_cast<size_t>(X[0]) * X[1] * X[2] * X[3];
          row_sites = single_parity ? X[0] / 2 : X[0];
          n_row = static_cast<size_t>(X[1]) * X[2] * X[3];
          file_sites = static_cast<size_t>(N[0]) * N[1] * N[2] * N[3] / (single_parity ? 2 : 1);
        }

        size_t row_bytes() const { return row_sites * layout.site_bytes; }
        size_t local_bytes() const { return n_row * row_bytes(); }

        /**
           @brief The position in the file, counted in stored sites, of the start of local row r on the rank at c
         */
        size_t row_file_index(const coords_t &c, size_t r) const
        {
          size_t y = c[1] * X[1] + r % X[1];
          size_t z = c[2] * X[2] + (r / X[1]) % X[2];
          size_t t = c[3] * X[3] + r / (static_cast<size_t>(X[1]) * X[2]);
          size_t g = c[0] * X[0] + N[0] * (y + N[1] * (z + N[2] * t));
          return single_parity ? g / 2 : g;
        }

        size_t file_offset(int block, size_t file_index) const
        {
          return layout.offset + (block * file_sites + file_index) * layout.site_bytes;
        }

        /**
           @brief Call f(first_row, n_rows, file_index) for each run of local rows that are contiguous in the file
         */
        template <typename F> void for_each_run(const coords_t &c, F &&f) const
        {
          size_t first = 0;
          for (size_t r = 1; r <= n_row; r++) {
            if (r == n_row || row_file_index(c, r) != row_file_index(c, r - 1) + row_sites) {
              f(first, r - first, row_file_index(c, first));
              first = r;
            }
          }
        }

        /**
           @brief Call f(index, position) for each local site, where index is the QUDA host index
           and position is the site's place in the rank's buffer (in file order)
         */
        template <typename F> void for_each_site(const coords_t &c, F &&f) const
        {
          for (size_t r = 0; r < n_row; r++) {
            int y = r % X[1], z = (r / X[1]) % X[2], t = r / (static_cast<size_t>(X[1]) * X[2]);
            int base = (c[0] * X[0] + c[1] * X[1] + y + c[2] * X[2] + z + c[3] * X[3] + t) & 1;
            for (size_t k = 0; k < row_sites; k++) {
              size_t x = single_parity ? 2 * k + ((layout.parity ^ base) & 1) : k;
              size_t lex = x + X[0] * r;
              size_t parity = (base + x) & 1;
              f(single_parity ? lex / 2 : lex / 2 + parity * (volume / 2), r * row_sites + k);
            }
          }
        }
      };

      inline uint32_t rotl(uint32_t w, int r) { return r == 0 ? w : (w << r) | (w >> (32 - r)); }

      /**
         @brief Accumulate the checksum of n values starting with value number first, which are
         either 32-bit words or the CRC-32s of sites of site_bytes each
       */
      void accumulate(Checksum &sum, const DataLayout &layout, const char *data, size_t n, size_t first)
      {
        int r29 = first % 29;
        int r31 = first % 31;
        for (size_t i = 0; i < n; i++) {
          uint32_t w;
          if (layout.checksum == ChecksumType::MILC)
            memcpy(&w, data + 4 * i, 4);
          else
            w = crc32(data + i * layout.site_bytes, layout.site_bytes);
          sum.sum29 ^= rotl(w, r29);
          sum.sum31 ^= rotl(w, r31);
          if (++r29 == 29) r29 = 0;
          if (++r31 == 31) r31 = 0;
        }
      }

      Checksum combine(const Checksum &local)
      {
        uint64_t sums = (static_cast<uint64_t>(local.sum29) << 32) | local.sum31;
        comm_allreduce_xor(sums);
        return {static_cast<uint32_t>(sums >> 32), static_cast<uint32_t>(sums)};
      }

      coords_t my_coords()
      {
        coords_t c;
        for (int d = 0; d < 4; d++) c[d] = comm_coord(d);
        return c;
      }

      std::vector<coords_t> all_coords()
      {
        std::vector<double> c(4 * comm_size(), 0.0);
        for (int d = 0; d < 4; d++) c[4 * comm_rank() + d] = comm_coord(d);
        comm_allreduce_sum(c);

        std::vector<coords_t> coords(comm_size());
        for (auto i = 0u; i < coords.size(); i++)
          for (int d = 0; d < 4; d++) coords[i][d] = static_cast<int>(c[4 * i + d]);
        return coords;
      }

      // the MPI backend limits a message to INT_MAX bytes
      constexpr size_t max_message_bytes = size_t(1) << 30;
      constexpr int message_tag = 0x4e87;

      void send(const char *data, size_t bytes, int rank)
      {
        for (size_t offset = 0; offset < bytes; offset += max_message_bytes) {
          auto mh = comm_declare_send_rank(const_cast<char *>(data) + offset, rank, message_tag,
                                           std::min(max_message_bytes, bytes - offset));
          comm_start(mh);
          comm_wait(mh);
          comm_free(mh);
        }
      }

      void recv(char *data, size_t bytes, int rank)
      {
        for (size_t offset = 0; offset < bytes; offset += max_message_bytes) {
          auto mh = comm_declare_recv_rank(data + offset, rank, message_tag, std::min(max_message_bytes, bytes - offset));
          comm_start(mh);
          comm_wait(mh);
          comm_free(mh);
        }
      }

      void read_rows(const File &file, const Geometry &g, const coords_t &c, int block, char *buffer)
      {
        g.for_each_run(c, [&](size_t first_row, size_t n_rows, size_t file_index) {
          file.read(buffer + first_row * g.row_bytes(), n_rows * g.row_bytes(), g.file_offset(block, file_index));
        });
      }

      void write_rows(const File &file, const Geometry &g, const coords_t &c, int block, const char *buffer)
      {
        g.for_each_run(c, [&](size_t first_row, size_t n_rows, size_t file_index) {
          file.write(buffer + first_row * g.row_bytes(), n_rows * g.row_bytes(), g.file_offset(block, file_index));
        });
      }

      void checksum_rows(Checksum &sum, const Geometry &g, const coords_t &c, int block, const char *buffer)
      {
        const auto &layout = g.layout;
        uint32_t sum29 = 0, sum31 = 0;
#pragma omp parallel for reduction(^ : sum29, sum31)
        for (size_t r = 0; r < g.n_row; r++) {
          Checksum row;
          const char *data = buffer + r * g.row_bytes();
          size_t first_site = block * g.file_sites + g.row_file_index(c, r);
          if (layout.checksum == ChecksumType::MILC)
            accumulate(row, layout, data, g.row_bytes() / 4, first_site * layout.site_bytes / 4);
          else
            accumulate(row, layout, data, g.row_sites, first_site);
          sum29 ^= row.sum29;
          sum31 ^= row.sum31;
        }
        sum.sum29 ^= sum29;
        sum.sum31 ^= sum31;
      }
    } // namespace

    Checksum read_data(const std::string &filename, const DataLayout &layout, const int *X, QudaFileIOMode io_mode,
                       const SiteUnpack &unpack)
    {
      Geometry g(layout, X);
      const auto me = my_coords();
      std::vector<char> buffer(g.local_bytes());
      Checksum sum;

      auto process = [&](int block) {
        if (layout.checksum == ChecksumType::SCIDAC) checksum_rows(sum, g, me, block, buffer.data());
        if (layout.swap) byte_swap(buffer.data(), buffer.size() / layout.word_bytes, layout.word_bytes);
        if (layout.checksum == ChecksumType::MILC) checksum_rows(sum, g, me, block, buffer.data());
        g.for_each_site(
          me, [&](size_t index, size_t pos) { unpack(block, index, buffer.data() + pos * layout.site_bytes); });
      };

      if (io_mode == QUDA_FILE_IO_PARALLEL) {
        File file(filename, File::Mode::READ);
        for (int block = 0; block < layout.n_block; block++) {
          read_rows(file, g, me, block, buffer.data());
          process(block);
        }
      } else {
        const auto coords = all_coords();
        const bool root = comm_rank() == 0;
        std::unique_ptr<File> file(root ? new File(filename, File::Mode::READ) : nullptr);
        std::vector<char> staging(root && comm_size() > 1 ? g.local_bytes() : 0);
        for (int block = 0; block < layout.n_block; block++) {
          if (root) {
            for (auto rank = 1u; rank < coords.size(); rank++) {
              read_rows(*file, g, coords[rank], block, staging.data());
              send(staging.data(), staging.size(), rank);
            }
            read_rows(*file, g, me, block, buffer.data());
          } else {
            recv(buffer.data(), buffer.size(), 0);
          }
          process(block);
        }
      }

      return combine(sum);
    }

    Checksum write_data(const std::string &filename, const DataLayout &layout, const int *X, QudaFileIOMode io_mode,
                        const SitePack &pack)
    {
      Geometry g(layout, X);
      const auto me = my_coords();
      std::vector<char> buffer(g.local_bytes());
      Checksum sum;

      auto process = [&](int block) {
        g.for_each_site(me,
                        [&](size_t index, size_t pos) { pack(block, index, buffer.data() + pos * layout.site_bytes); });
        if (layout.checksum == ChecksumType::MILC) checksum_rows(sum, g, me, block, buffer.data());
        if (layout.swap) byte_swap(buffer.data(), buffer.size() / layout.word_bytes, layout.word_bytes);
        if (layout.checksum == ChecksumType::SCIDAC) checksum_rows(sum, g, me, block, buffer.data());
      };

      // rank 0 has created the file and written everything but the data
      comm_barrier();

      if (io_mode == QUDA_FILE_IO_PARALLEL) {
        File file(filename, File::Mode::WRITE);
        for (int block = 0; block < layout.n_block; block++) {
          process(block);
          write_rows(file, g, me, block, buffer.data());
        }
        file.close();
      } else {
        const auto coords = all_coords();
        const bool root = comm_rank() == 0;
        std::unique_ptr<File> file(root ? new File(filename, File::Mode::WRITE) : nullptr);
        std::vector<char> staging(root && comm_size() > 1 ? g.local_bytes() : 0);
        for (int block = 0; block < layout.n_block; block++) {
          process(block);
          if (root) {
            write_rows(*file, g, me, block, buffer.data());
            for (auto rank = 1u; rank < coords.size(); rank++) {
              recv(staging.data(), staging.size(), rank);
              write_rows(*file, g, coords[rank], block, staging.data());
            }
          } else {
            send(buffer.data(), buffer.size(), 0);
          }
        }
        if (root) file->close();
      }

      // every rank's data is in the file before anyone reads it
      comm_barrier();

      return combine(sum);
    }

  } // namespace io

} // namespace quda
