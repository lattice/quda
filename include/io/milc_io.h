#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <io/gauge_io.h>

namespace quda
{

  namespace milc
  {

    /**
       @brief Description of the data section of a MILC file: one or
       more blocks (e.g., eigenvectors), each holding the sites of the
       global lattice in lexicographic order with x running fastest
     */
    struct DataLayout {
      size_t offset = 0;                       /** Byte offset of the first block */
      size_t site_bytes = 0;                   /** Bytes per site */
      size_t word_bytes = 4;                   /** Size of the floating-point words, for byte swapping */
      int n_block = 1;                         /** Number of blocks, stored one after another */
      QudaParity parity = QUDA_INVALID_PARITY; /** If even or odd, only sites of that parity are stored */
    };

    /**
       @brief The two MILC checksums over the 32-bit words of the data section
     */
    struct Checksum {
      uint32_t sum29 = 0;
      uint32_t sum31 = 0;
      bool operator==(const Checksum &other) const { return sum29 == other.sum29 && sum31 == other.sum31; }
      bool operator!=(const Checksum &other) const { return !(*this == other); }
    };

    /**
       @brief Called for each local site with the site's data in native byte order.
       @param[in] block The block (vector) index
       @param[in] index The QUDA host checkerboard index of the site: even
       sites first for full fields, or the index within the parity when
       the layout stores a single parity
       @param[in] site The site's data, DataLayout::site_bytes long
     */
    using SiteUnpack = std::function<void(int block, size_t index, const char *site)>;

    /**
       @brief Called for each local site to fill in the site's data in native byte order
     */
    using SitePack = std::function<void(int block, size_t index, char *site)>;

    /**
       @brief Simple wrapper around a POSIX file descriptor, raising errorQuda on failure
     */
    class File
    {
      int fd = -1;
      std::string filename;

    public:
      enum class Mode { READ, WRITE, CREATE };

      File(const std::string &filename, Mode mode);
      ~File();
      File(const File &) = delete;
      File &operator=(const File &) = delete;

      void read(void *data, size_t bytes, size_t offset) const;
      void write(const void *data, size_t bytes, size_t offset) const;
      size_t size() const;

      /**
         @brief Close the file, raising an error if that fails (which can
         report failed writes on network file systems)
       */
      void close();
    };

    /**
       @brief Reverse the byte order of an array of words
       @param[in,out] data The array
       @param[in] n_words Number of words
       @param[in] word_bytes Size of each word (4 or 8)
     */
    void byte_swap(void *data, size_t n_words, size_t word_bytes);

    inline uint32_t byte_swap(uint32_t word) { return __builtin_bswap32(word); }

    /**
       @brief Read the start of a file on rank 0 and broadcast it, along with the file size
       @param[in] filename The file
       @param[out] data Buffer for the first bytes of the file
       @param[in] bytes Number of bytes to read
       @return The size of the file in bytes
     */
    size_t read_header(const std::string &filename, void *data, size_t bytes);

    /**
       @brief The global lattice dimensions
       @param[in] X The local lattice dimensions
     */
    std::array<int, 4> global_dims(const int *X);

    /**
       @brief Size in bytes of the data section described by a layout
     */
    size_t data_bytes(const DataLayout &layout, const std::array<int, 4> &dims);

    /**
       @brief Read the data section of a file.  Each rank receives its
       own sites, in parallel mode by reading them itself and in serial
       mode from rank 0.
       @param[in] filename The file
       @param[in] layout Layout of the data section
       @param[in] X The local lattice dimensions
       @param[in] swap Whether the file is in the opposite byte order
       @param[in] io_mode Serial or parallel
       @param[in] unpack Called for each local site of each block
       @return The checksum of the data section, combined over all ranks
     */
    Checksum read_data(const std::string &filename, const DataLayout &layout, const int *X, bool swap,
                       QudaFileIOMode io_mode, const SiteUnpack &unpack);

    /**
       @brief Write the data section of a file, in native byte order.
       The file must already exist (rank 0 creates it and writes the
       header beforehand).
       @return The checksum of the data section, combined over all ranks
     */
    Checksum write_data(const std::string &filename, const DataLayout &layout, const int *X, QudaFileIOMode io_mode,
                        const SitePack &pack);

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
       or "key value"; surrounding quotes are removed from the values.
       @return The keys and values, or nothing if there is no file
     */
    std::optional<std::map<std::string, std::string>> read_info(const std::string &filename);

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
     */
    void save_gauge(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                    const FileIOParam &param);

  } // namespace milc

} // namespace quda
