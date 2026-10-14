#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <sstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/milc_io.h>

namespace quda
{

  namespace milc
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

    size_t read_header(const std::string &filename, void *data, size_t bytes)
    {
      size_t size = 0;
      if (comm_rank() == 0) {
        File file(filename, File::Mode::READ);
        size = file.size();
        if (size < bytes) errorQuda("File %s is too short (%lu bytes) to be a MILC file", filename.c_str(), size);
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

        size_t word_index(int block, size_t file_index) const
        {
          return (block * file_sites + file_index) * layout.site_bytes / 4;
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

      void accumulate(Checksum &sum, const char *data, size_t n_words, size_t first_word)
      {
        int r29 = first_word % 29;
        int r31 = first_word % 31;
        for (size_t i = 0; i < n_words; i++) {
          uint32_t w;
          memcpy(&w, data + 4 * i, 4);
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
        g.for_each_run(c, [&](size_t first_row, size_t n_rows, size_t file_index) {
          accumulate(sum, buffer + first_row * g.row_bytes(), n_rows * g.row_bytes() / 4,
                     g.word_index(block, file_index));
        });
      }
    } // namespace

    Checksum read_data(const std::string &filename, const DataLayout &layout, const int *X, bool swap,
                       QudaFileIOMode io_mode, const SiteUnpack &unpack)
    {
      Geometry g(layout, X);
      const auto me = my_coords();
      std::vector<char> buffer(g.local_bytes());
      Checksum sum;

      auto process = [&](int block) {
        if (swap) byte_swap(buffer.data(), buffer.size() / layout.word_bytes, layout.word_bytes);
        checksum_rows(sum, g, me, block, buffer.data());
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
        checksum_rows(sum, g, me, block, buffer.data());
      };

      // rank 0 has created the file and written its header
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

    std::string time_stamp()
    {
      time_t now = time(nullptr);
      char buf[64] = {};
      if (!ctime_r(&now, buf)) return "";
      std::string stamp(buf);
      if (!stamp.empty() && stamp.back() == '\n') stamp.pop_back();
      return stamp;
    }

    void write_info(const std::string &filename, const std::vector<std::pair<std::string, std::string>> &items)
    {
      if (comm_rank() != 0) return;
      auto info_filename = filename + ".info";
      std::ofstream out(info_filename);
      for (auto &item : items) out << item.first << " = " << item.second << "\n";
      out.close();
      if (!out) warningQuda("Could not write %s", info_filename.c_str());
    }

    std::optional<std::map<std::string, std::string>> read_info(const std::string &filename)
    {
      int found = 0;
      size_t length = 0;
      std::string contents;
      if (comm_rank() == 0) {
        std::ifstream in(filename + ".info");
        if (in) {
          std::stringstream ss;
          ss << in.rdbuf();
          contents = ss.str();
          length = contents.size();
          found = 1;
        }
      }
      comm_broadcast(&found, sizeof(found));
      if (!found) return std::nullopt;
      comm_broadcast(&length, sizeof(length));
      contents.resize(length);
      if (length > 0) comm_broadcast(contents.data(), length);

      auto trim = [](const std::string &s) {
        auto first = s.find_first_not_of(" \t\r");
        if (first == std::string::npos) return std::string();
        return s.substr(first, s.find_last_not_of(" \t\r") - first + 1);
      };

      std::map<std::string, std::string> items;
      std::istringstream lines(contents);
      std::string line;
      while (std::getline(lines, line)) {
        line = trim(line);
        auto end = line.find_first_of(" \t=");
        if (line.empty() || end == 0) continue;
        auto key = line.substr(0, end);
        auto value = end == std::string::npos ? std::string() : trim(line.substr(end));
        if (!value.empty() && value[0] == '=') value = trim(value.substr(1));
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') value = value.substr(1, value.size() - 2);
        items[key] = value;
      }
      return items;
    }

  } // namespace milc

} // namespace quda
