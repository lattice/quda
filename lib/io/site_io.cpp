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

    File::File(const std::string &filename, Mode mode, bool mpi_io) : filename(filename)
    {
      if (mpi_io) {
        if (mode == Mode::WRITE)
          errorQuda("%s: MPI-IO files are opened to be read or created, not to be written in place", filename.c_str());
        mpi.reset(mpi_open(filename, mode == Mode::CREATE));
        return;
      }
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
      if (mpi) {
        mpi_close(*mpi);
        mpi.reset();
        return;
      }
      if (fd >= 0 && ::close(fd) != 0) errorQuda("Error closing %s: %s", filename.c_str(), strerror(errno));
      fd = -1;
    }

    void File::read(void *data, size_t bytes, size_t offset) const
    {
      if (mpi) {
        mpi_read_at(*mpi, data, bytes, offset);
        return;
      }
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
      if (mpi) {
        mpi_write_at(*mpi, data, bytes, offset);
        return;
      }
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
      if (mpi) return mpi_size(*mpi);
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

    void broadcast(void *data, size_t bytes)
    {
      auto p = static_cast<char *>(data);
      for (size_t offset = 0; offset < bytes; offset += max_message_bytes)
        comm_broadcast(p + offset, std::min(max_message_bytes, bytes - offset));
    }

    void broadcast(std::string &data)
    {
      size_t n = data.size();
      broadcast(&n, sizeof(n));
      data.resize(n);
      broadcast(data.data(), n);
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
      broadcast(data, bytes);
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

    std::string excerpt(const std::string &s)
    {
      constexpr size_t n = 200;
      if (s.size() <= n) return s;
      return s.substr(0, n) + "... (" + std::to_string(s.size()) + " bytes)";
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
         @brief A box of stored sites, [lo, hi) in each dimension
       */
      struct Box {
        coords_t lo;
        coords_t hi;

        size_t extent(int d) const { return hi[d] - lo[d]; }
        size_t volume() const { return extent(0) * extent(1) * extent(2) * extent(3); }
      };

      /**
         @brief Where each rank's sites of the region are in the lattice and the files, and in QUDA's host ordering.
         A rank's buffer holds its sites of the region in file order: its box of the region in lexicographic order.
       */
      struct Geometry {
        const DataLayout &layout;
        coords_t X;
        coords_t N;
        coords_t L; // stored local dimensions: X with the x extent halved when a single parity is stored
        coords_t S; // stored global dimensions
        Box region; // the stored sites the files hold
        bool single_parity;
        size_t volume;     // local sites
        size_t file_sites; // stored sites per block in the lattice

        Geometry(const DataLayout &layout, const int *X_, const Hypercube *box) :
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
          L = X;
          S = N;
          if (single_parity) {
            L[0] /= 2;
            S[0] /= 2;
          }
          file_sites = static_cast<size_t>(S[0]) * S[1] * S[2] * S[3];
          region = {{}, S};
          if (box) {
            for (int d = 0; d < 4; d++)
              if (box->lower[d] < 0 || box->lower[d] > box->upper[d] || box->upper[d] >= S[d])
                errorQuda("The hypercube %d %d %d %d .. %d %d %d %d is not a box of the lattice %d %d %d %d",
                          box->lower[0], box->lower[1], box->lower[2], box->lower[3], box->upper[0], box->upper[1],
                          box->upper[2], box->upper[3], S[0], S[1], S[2], S[3]);
            region.lo = box->lower;
            for (int d = 0; d < 4; d++) region.hi[d] = box->upper[d] + 1;
          }
        }

        /**
           @brief The stored sites of the rank at c in the region (empty if it holds none)
         */
        Box local_box(const coords_t &c) const
        {
          Box b;
          for (int d = 0; d < 4; d++) {
            b.lo[d] = std::max(c[d] * L[d], region.lo[d]);
            b.hi[d] = std::max(b.lo[d], std::min((c[d] + 1) * L[d], region.hi[d]));
          }
          return b;
        }

        size_t local_bytes(const coords_t &c) const { return local_box(c).volume() * layout.site_bytes; }

        /**
           @brief The part holding the region in one file
         */
        FilePart whole_region(const std::string &filename) const
        {
          FilePart part = {filename, layout.offset, region.lo, {}};
          for (int d = 0; d < 4; d++) part.extent[d] = region.extent(d);
          return part;
        }

        size_t file_offset(const FilePart &part, int block, size_t file_index) const
        {
          return part.offset + (block * part.volume() + file_index) * layout.site_bytes;
        }

        /**
           @brief Call f(buffer_site, file_index, n_sites) for each run of sites of the rank at c that are
           contiguous both in the rank's buffer (in file order) and in a file's box, where buffer_site is the
           run's position in the buffer and file_index its position in the file, both counted in stored sites
           @return The number of the rank's sites in the box
         */
        template <typename F> size_t for_each_run(const coords_t &c, const FilePart &part, F &&f) const
        {
          const auto b = local_box(c);
          coords_t lo, hi;
          for (int d = 0; d < 4; d++) {
            lo[d] = std::max(b.lo[d], part.corner[d]);
            hi[d] = std::min(b.hi[d], part.corner[d] + part.extent[d]);
            if (lo[d] >= hi[d]) return 0;
          }
          const auto &P = part.corner;
          const auto &E = part.extent;
          const size_t n = hi[0] - lo[0];
          size_t run_buffer = 0, run_file = 0, run_sites = 0, total = 0;
          for (int t = lo[3]; t < hi[3]; t++) {
            for (int z = lo[2]; z < hi[2]; z++) {
              for (int y = lo[1]; y < hi[1]; y++) {
                size_t row = (y - b.lo[1]) + b.extent(1) * ((z - b.lo[2]) + b.extent(2) * (t - b.lo[3]));
                size_t buffer_site = row * b.extent(0) + (lo[0] - b.lo[0]);
                size_t file_index
                  = (lo[0] - P[0]) + E[0] * ((y - P[1]) + E[1] * ((z - P[2]) + static_cast<size_t>(E[2]) * (t - P[3])));
                if (run_sites > 0 && buffer_site == run_buffer + run_sites && file_index == run_file + run_sites) {
                  run_sites += n;
                } else {
                  if (run_sites > 0) f(run_buffer, run_file, run_sites);
                  run_buffer = buffer_site;
                  run_file = file_index;
                  run_sites = n;
                }
                total += n;
              }
            }
          }
          if (run_sites > 0) f(run_buffer, run_file, run_sites);
          return total;
        }

        /**
           @brief Call f(index, position) for each local site in the region, where index is the QUDA host index
           and position is the site's place in the rank's buffer (in file order)
         */
        template <typename F> void for_each_site(const coords_t &c, F &&f) const
        {
          const auto b = local_box(c);
          if (b.volume() == 0) return;
          size_t position = 0;
          for (int t = b.lo[3]; t < b.hi[3]; t++) {
            for (int z = b.lo[2]; z < b.hi[2]; z++) {
              for (int y = b.lo[1]; y < b.hi[1]; y++) {
                const size_t r
                  = (y - c[1] * X[1]) + X[1] * ((z - c[2] * X[2]) + static_cast<size_t>(X[2]) * (t - c[3] * X[3]));
                const int base = (c[0] * X[0] + y + z + t) & 1;
                for (int k = b.lo[0] - c[0] * L[0]; k < b.hi[0] - c[0] * L[0]; k++) {
                  size_t x = single_parity ? 2 * k + ((layout.parity ^ base) & 1) : k;
                  size_t lex = x + X[0] * r;
                  size_t parity = (base + x) & 1;
                  f(single_parity ? lex / 2 : lex / 2 + parity * (volume / 2), position++);
                }
              }
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

      constexpr size_t max_open_files = 32;

      /**
         @brief The files of a field, opened when first used, keeping at
         most max_open_files open and closing the least recently used
       */
      class FileCache
      {
        const std::vector<FilePart> &parts;
        File::Mode mode;
        std::vector<std::pair<size_t, std::unique_ptr<File>>> open; // most recently used last

      public:
        FileCache(const std::vector<FilePart> &parts, File::Mode mode) : parts(parts), mode(mode) { }

        const File &get(size_t i)
        {
          for (auto it = open.begin(); it != open.end(); ++it) {
            if (it->first != i) continue;
            std::rotate(it, it + 1, open.end());
            return *open.back().second;
          }
          if (open.size() == max_open_files) {
            open.front().second->close();
            open.erase(open.begin());
          }
          open.emplace_back(i, std::make_unique<File>(parts[i].filename, mode));
          return *open.back().second;
        }

        void close()
        {
          for (auto &f : open) f.second->close();
          open.clear();
        }
      };

      /**
         @brief Read or write the sites of the rank at c, held in buffer in file order, from or to each file
       */
      template <typename Access>
      void access_rows(FileCache &files, const std::vector<FilePart> &parts, const Geometry &g, const coords_t &c,
                       int block, Access &&access)
      {
        size_t sites = 0;
        for (auto i = 0u; i < parts.size(); i++) {
          const File *file = nullptr;
          sites += g.for_each_run(c, parts[i], [&](size_t buffer_site, size_t file_index, size_t n) {
            if (!file) file = &files.get(i);
            access(*file, buffer_site * g.layout.site_bytes, n * g.layout.site_bytes,
                   g.file_offset(parts[i], block, file_index));
          });
        }
        const size_t expected = g.local_box(c).volume();
        if (sites != expected)
          errorQuda("The files hold %lu of the %lu sites of the rank at %d %d %d %d in the region", sites, expected,
                    c[0], c[1], c[2], c[3]);
      }

      void read_rows(FileCache &files, const std::vector<FilePart> &parts, const Geometry &g, const coords_t &c,
                     int block, char *buffer)
      {
        access_rows(files, parts, g, c, block, [&](const File &file, size_t pos, size_t bytes, size_t offset) {
          file.read(buffer + pos, bytes, offset);
        });
      }

      void write_rows(FileCache &files, const std::vector<FilePart> &parts, const Geometry &g, const coords_t &c,
                      int block, const char *buffer)
      {
        access_rows(files, parts, g, c, block, [&](const File &file, size_t pos, size_t bytes, size_t offset) {
          file.write(buffer + pos, bytes, offset);
        });
      }

      void checksum_rows(Checksum &sum, const Geometry &g, const coords_t &c, int block, const char *buffer)
      {
        const auto &layout = g.layout;
        const auto b = g.local_box(c);
        const size_t row_sites = b.extent(0), row_bytes = row_sites * layout.site_bytes;
        const size_t n_row = b.extent(1) * b.extent(2) * b.extent(3);
        const auto &S = g.S;
        uint32_t sum29 = 0, sum31 = 0;
#pragma omp parallel for reduction(^ : sum29, sum31)
        for (size_t r = 0; r < n_row; r++) {
          Checksum row;
          const char *data = buffer + r * row_bytes;
          const size_t y = b.lo[1] + r % b.extent(1);
          const size_t z = b.lo[2] + (r / b.extent(1)) % b.extent(2);
          const size_t t = b.lo[3] + r / (b.extent(1) * b.extent(2));
          // the site's index in the whole lattice, also for a record of a box of it
          size_t first_site = block * g.file_sites + b.lo[0] + S[0] * (y + S[1] * (z + S[2] * t));
          if (layout.checksum == ChecksumType::MILC)
            accumulate(row, layout, data, row_bytes / 4, first_site * layout.site_bytes / 4);
          else
            accumulate(row, layout, data, row_sites, first_site);
          sum29 ^= row.sum29;
          sum31 ^= row.sum31;
        }
        sum.sum29 ^= sum29;
        sum.sum31 ^= sum31;
      }
    } // namespace

    bool check_partition_grid(QudaFileIOMode io_mode, const std::array<int, 4> &grid, const std::array<int, 4> &dims)
    {
      if (grid == coords_t {}) return false;
      const bool serial = io_mode == QUDA_FILE_IO_SERIAL;

      for (int d = 0; d < 4; d++)
        if (grid[d] <= 0 || dims[d] % grid[d] != 0)
          errorQuda("The I/O partition grid %d %d %d %d does not divide the process grid %d %d %d %d", grid[0], grid[1],
                    grid[2], grid[3], dims[0], dims[1], dims[2], dims[3]);

      if (serial) {
        logQuda(QUDA_VERBOSE, "Serial I/O goes through rank 0 alone, so the I/O partition grid %d %d %d %d is ignored\n",
                grid[0], grid[1], grid[2], grid[3]);
        return false;
      }
      if (io_mode == QUDA_FILE_IO_MPI_IO) {
        if (grid != dims)
          errorQuda("The I/O partition grid %d %d %d %d groups the ranks of the process grid %d %d %d %d, but in "
                    "MPI-IO mode the MPI library chooses which ranks access the file: use a zero grid, and set the "
                    "aggregators through the MPI library (ROMIO's cb_nodes and cb_config_list hints, or Open MPI's "
                    "io_ompio_num_aggregators)",
                    grid[0], grid[1], grid[2], grid[3], dims[0], dims[1], dims[2], dims[3]);
        return false;
      }
      logQuda(QUDA_VERBOSE, "I/O partition grid %d %d %d %d: each partition has %d x %d x %d x %d ranks\n", grid[0],
              grid[1], grid[2], grid[3], dims[0] / grid[0], dims[1] / grid[1], dims[2] / grid[2], dims[3] / grid[3]);
      return true;
    }

    Partitions::Partitions(QudaFileIOMode io_mode, const std::array<int, 4> &grid, bool partfile) :
      rank_coords(all_coords())
    {
      const bool serial = io_mode == QUDA_FILE_IO_SERIAL && !partfile;
      const coords_t dims = {comm_dim(0), comm_dim(1), comm_dim(2), comm_dim(3)};
      const bool grouped = check_partition_grid(partfile ? QUDA_FILE_IO_PARALLEL : io_mode, grid, dims);
      for (int d = 0; d < 4; d++) rank_extent[d] = serial ? dims[d] : grouped ? dims[d] / grid[d] : 1;

      auto corner = [&](int rank) {
        coords_t c = rank_coords[rank];
        for (int d = 0; d < 4; d++) c[d] -= c[d] % rank_extent[d];
        return c;
      };

      const int me = comm_rank();
      const auto my_corner = corner(me);
      if (serial) {
        my_leader = 0;
      } else {
        auto it = std::find(rank_coords.begin(), rank_coords.end(), my_corner);
        if (it == rank_coords.end())
          errorQuda("No rank has the coordinates %d %d %d %d", my_corner[0], my_corner[1], my_corner[2], my_corner[3]);
        my_leader = static_cast<int>(it - rank_coords.begin());
      }
      leads = my_leader == me;
      if (leads)
        for (int rank = 0; rank < static_cast<int>(rank_coords.size()); rank++)
          if (serial || corner(rank) == my_corner) my_ranks.push_back(rank);

      if (serial)
        all_leaders = {0};
      else
        for (int rank = 0; rank < static_cast<int>(rank_coords.size()); rank++)
          if (corner(rank) == rank_coords[rank]) all_leaders.push_back(rank);
    }

    namespace
    {
      // the size of the staging buffer of a leader: the largest share of the region of the other ranks it handles
      size_t staging_bytes(const Geometry &g, const Partitions &partitions)
      {
        size_t bytes = 0;
        for (int rank : partitions.ranks())
          if (rank != comm_rank()) bytes = std::max(bytes, g.local_bytes(partitions.coords(rank)));
        return bytes;
      }

      /**
         @brief Checksum, byte swap and unpack one block of the rank at c, read into its buffer in file order
       */
      void unpack_block(Checksum &sum, const Geometry &g, const coords_t &c, int block, std::vector<char> &buffer,
                        const SiteUnpack &unpack)
      {
        const auto &layout = g.layout;
        if (layout.checksum == ChecksumType::SCIDAC) checksum_rows(sum, g, c, block, buffer.data());
        if (layout.swap) byte_swap(buffer.data(), buffer.size() / layout.word_bytes, layout.word_bytes);
        if (layout.checksum == ChecksumType::MILC) checksum_rows(sum, g, c, block, buffer.data());
        g.for_each_site(
          c, [&](size_t index, size_t pos) { unpack(block, index, buffer.data() + pos * layout.site_bytes); });
      }

      /**
         @brief Pack, checksum and byte swap one block of the rank at c into its buffer, in file order
       */
      void pack_block(Checksum &sum, const Geometry &g, const coords_t &c, int block, std::vector<char> &buffer,
                      const SitePack &pack)
      {
        const auto &layout = g.layout;
        g.for_each_site(c,
                        [&](size_t index, size_t pos) { pack(block, index, buffer.data() + pos * layout.site_bytes); });
        if (layout.checksum == ChecksumType::MILC) checksum_rows(sum, g, c, block, buffer.data());
        if (layout.swap) byte_swap(buffer.data(), buffer.size() / layout.word_bytes, layout.word_bytes);
        if (layout.checksum == ChecksumType::SCIDAC) checksum_rows(sum, g, c, block, buffer.data());
      }

      /**
         @brief Read or write one block of every rank's box of the region, collectively through MPI-IO
       */
      void access_box(const File &file, const Geometry &g, const coords_t &c, int block, char *buffer, bool write)
      {
        if (!file.mpi_handle()) errorQuda("%s was not opened through MPI-IO", file.name().c_str());
        const auto b = g.local_box(c);
        coords_t region, lo, extent;
        for (int d = 0; d < 4; d++) {
          region[d] = static_cast<int>(g.region.extent(d));
          lo[d] = b.lo[d] - g.region.lo[d];
          extent[d] = static_cast<int>(b.extent(d));
        }
        const size_t offset = g.layout.offset + block * g.region.volume() * g.layout.site_bytes;
        mpi_access(*file.mpi_handle(), write, buffer, offset, region, lo, extent, g.layout.site_bytes);
      }

      coords_t my_coords() { return {comm_coord(0), comm_coord(1), comm_coord(2), comm_coord(3)}; }
    } // namespace

    Checksum read_data(const std::string &filename, const DataLayout &layout, const int *X,
                       const Partitions &partitions, const SiteUnpack &unpack, const Hypercube *region)
    {
      return read_data({Geometry(layout, X, region).whole_region(filename)}, layout, X, partitions, unpack, region);
    }

    Checksum read_data(const std::vector<FilePart> &parts, const DataLayout &layout, const int *X,
                       const Partitions &partitions, const SiteUnpack &unpack, const Hypercube *region)
    {
      Geometry g(layout, X, region);
      const int my_rank = comm_rank();
      const auto &me = partitions.coords(my_rank);
      std::vector<char> buffer(g.local_bytes(me));
      Checksum sum;

      const auto &ranks = partitions.ranks();
      FileCache files(parts, File::Mode::READ);
      std::vector<char> staging(staging_bytes(g, partitions));
      for (int block = 0; block < layout.n_block; block++) {
        for (int rank : ranks) {
          if (rank == my_rank) {
            read_rows(files, parts, g, me, block, buffer.data());
          } else {
            read_rows(files, parts, g, partitions.coords(rank), block, staging.data());
            send(staging.data(), g.local_bytes(partitions.coords(rank)), rank);
          }
        }
        if (!partitions.is_leader()) recv(buffer.data(), buffer.size(), partitions.leader());
        unpack_block(sum, g, me, block, buffer, unpack);
      }

      return combine(sum);
    }

    Checksum read_data(const File &file, const DataLayout &layout, const int *X, const SiteUnpack &unpack,
                       const Hypercube *region)
    {
      Geometry g(layout, X, region);
      const auto me = my_coords();
      std::vector<char> buffer(g.local_bytes(me));
      Checksum sum;
      for (int block = 0; block < layout.n_block; block++) {
        access_box(file, g, me, block, buffer.data(), false);
        unpack_block(sum, g, me, block, buffer, unpack);
      }
      return combine(sum);
    }

    Checksum write_data(const std::string &filename, const DataLayout &layout, const int *X,
                        const Partitions &partitions, const SitePack &pack, const Hypercube *region)
    {
      return write_data({Geometry(layout, X, region).whole_region(filename)}, layout, X, partitions, pack, region);
    }

    Checksum write_data(const std::vector<FilePart> &parts, const DataLayout &layout, const int *X,
                        const Partitions &partitions, const SitePack &pack, const Hypercube *region)
    {
      Geometry g(layout, X, region);
      const int my_rank = comm_rank();
      const auto &me = partitions.coords(my_rank);
      std::vector<char> buffer(g.local_bytes(me));
      Checksum sum;

      // the files have been created, with everything but the data
      comm_barrier();

      const auto &ranks = partitions.ranks();
      FileCache files(parts, File::Mode::WRITE);
      std::vector<char> staging(staging_bytes(g, partitions));
      for (int block = 0; block < layout.n_block; block++) {
        pack_block(sum, g, me, block, buffer, pack);
        if (!partitions.is_leader()) send(buffer.data(), buffer.size(), partitions.leader());
        for (int rank : ranks) {
          if (rank == my_rank) {
            write_rows(files, parts, g, me, block, buffer.data());
          } else {
            recv(staging.data(), g.local_bytes(partitions.coords(rank)), rank);
            write_rows(files, parts, g, partitions.coords(rank), block, staging.data());
          }
        }
      }
      files.close();

      // every rank's data is in the files before anyone reads them
      comm_barrier();

      return combine(sum);
    }

    Checksum write_data(const File &file, const DataLayout &layout, const int *X, const SitePack &pack,
                        const Hypercube *region)
    {
      Geometry g(layout, X, region);
      const auto me = my_coords();
      std::vector<char> buffer(g.local_bytes(me));
      Checksum sum;
      for (int block = 0; block < layout.n_block; block++) {
        pack_block(sum, g, me, block, buffer, pack);
        access_box(file, g, me, block, buffer.data(), true);
      }
      return combine(sum);
    }

  } // namespace io

} // namespace quda
