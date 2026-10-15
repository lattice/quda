#pragma once

// Format-neutral storage engine: moves the sites of a lattice field between a file and each rank's
// host memory, through the leaders of the I/O partitions, and computes the file's checksum on the way

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>
#include <enum_quda.h>

namespace quda
{

  namespace io
  {

    /**
       @brief How a file's checksum is computed.  Both rotate a 32-bit
       value left by i % 29 and i % 31 and XOR the results into two
       sums.
     */
    enum class ChecksumType {
      MILC,  /** The 32-bit words of the data in native byte order; i counts words */
      SCIDAC /** The CRC-32 of each site's bytes as stored; i is the site's global lexicographic index */
    };

    /**
       @brief Description of the data section of a file: one or more
       blocks (e.g., eigenvectors), each holding the sites of the
       global lattice in lexicographic order with x running fastest
     */
    struct DataLayout {
      size_t offset = 0;                       /** Byte offset of the first block */
      size_t site_bytes = 0;                   /** Bytes per site */
      size_t word_bytes = 4;                   /** Size of the floating-point words, for byte swapping */
      int n_block = 1;                         /** Number of blocks, stored one after another */
      QudaParity parity = QUDA_INVALID_PARITY; /** If even or odd, only sites of that parity are stored */
      bool swap = false;                       /** Whether the file's byte order is the opposite of the host's */
      ChecksumType checksum = ChecksumType::MILC;
    };

    /**
       @brief The two checksums of a file's data: sum29 and sum31 for
       MILC files, suma and sumb for SciDAC files
     */
    struct Checksum {
      uint32_t sum29 = 0;
      uint32_t sum31 = 0;
      bool operator==(const Checksum &other) const { return sum29 == other.sum29 && sum31 == other.sum31; }
      bool operator!=(const Checksum &other) const { return !(*this == other); }
    };

    /**
       @brief The two checksums as "%x %x", as MILC .info files and messages print them
     */
    std::string checksum_str(const Checksum &sum);

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
       @brief Whether the host is little endian
     */
    constexpr bool host_little_endian() { return __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__; }

    /**
       @brief The CRC-32 of zlib, gzip and PNG (reflected polynomial
       0xedb88320, initial value and final XOR 0xffffffff)
       @param[in] data The data
       @param[in] bytes Number of bytes
       @param[in] crc The CRC of any preceding data, to continue a computation
     */
    uint32_t crc32(const void *data, size_t bytes, uint32_t crc = 0);

    /**
       @brief The most bytes handed to the communications layer in one
       message or broadcast: the MPI and QMP backends pass byte counts
       to MPI as an int
     */
    constexpr size_t max_message_bytes = size_t(1) << 30;

    /**
       @brief Broadcast from rank 0, in pieces of at most max_message_bytes
     */
    void broadcast(void *data, size_t bytes);

    /**
       @brief Broadcast a string from rank 0, resizing it on the other ranks
     */
    void broadcast(std::string &data);

    /**
       @brief Broadcast a vector from rank 0, resizing it on the other ranks
     */
    template <typename T> void broadcast(std::vector<T> &data)
    {
      static_assert(std::is_trivially_copyable_v<T>, "broadcast copies the elements' bytes");
      size_t n = data.size();
      broadcast(&n, sizeof(n));
      data.resize(n);
      broadcast(data.data(), n * sizeof(T));
    }

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
       @brief Check an I/O partition grid against the process grid and
       log how it applies.  A grid that does not divide the process
       grid is an error in either mode; a valid grid is ignored in
       serial mode.
       @param[in] io_mode Serial or parallel
       @param[in] grid Number of partitions in each dimension, or all zeros for one per rank
       @param[in] dims The process grid
       @return Whether the grid groups the ranks: false for a zero grid and in serial mode
     */
    bool check_partition_grid(QudaFileIOMode io_mode, const std::array<int, 4> &grid, const std::array<int, 4> &dims);

    /**
       @brief The I/O partitions: blocks of the process grid, each led
       by the rank at its lowest corner (as MILC's io_node), which is
       the only rank of its block to open the file and reads or writes
       the data of every rank in the block.  Serial mode is a single
       partition led by rank 0, whatever the grid; in parallel mode a
       zero grid means one partition per rank.  Constructing one is
       collective.
     */
    class Partitions
    {
      std::vector<std::array<int, 4>> rank_coords;
      std::array<int, 4> extent; // ranks per partition in each dimension
      int my_leader;
      bool leads;
      std::vector<int> my_ranks;

    public:
      /**
         @param[in] io_mode Serial or parallel
         @param[in] grid Number of partitions in each dimension, or all zeros for one per rank
       */
      Partitions(QudaFileIOMode io_mode, const std::array<int, 4> &grid);

      /** The leader of this rank's partition */
      int leader() const { return my_leader; }

      /** Whether this rank leads its partition */
      bool is_leader() const { return leads; }

      /** On a leader, the ranks of its partition in ascending order, itself included; empty elsewhere */
      const std::vector<int> &ranks() const { return my_ranks; }

      /** The coordinates of a rank in the process grid */
      const std::array<int, 4> &coords(int rank) const { return rank_coords[rank]; }
    };

    /**
       @brief Read the data section of a file.  Each partition's leader
       reads the sites of every rank in its partition and sends them
       to their ranks.
       @param[in] filename The file
       @param[in] layout Layout of the data section
       @param[in] X The local lattice dimensions
       @param[in] partitions Which ranks open the file, and for which ranks they read
       @param[in] unpack Called for each local site of each block
       @return The checksum of the data section, combined over all ranks
     */
    Checksum read_data(const std::string &filename, const DataLayout &layout, const int *X,
                       const Partitions &partitions, const SiteUnpack &unpack);

    /**
       @brief Write the data section of a file.  The file must already
       exist (rank 0 creates it and writes everything else beforehand).
       Each partition's leader receives the sites of every rank in its
       partition and writes them.
       @return The checksum of the data section, combined over all ranks
     */
    Checksum write_data(const std::string &filename, const DataLayout &layout, const int *X,
                        const Partitions &partitions, const SitePack &pack);

    /**
       @brief Report a checksum mismatch: a warning if allowed, otherwise an error
     */
    void report_checksum_mismatch(const std::string &message, bool allow);

    /**
       @brief Compare the checksum computed from a file's data with the one the file stores, reporting a mismatch
       @param[in] allow Whether a mismatch is a warning rather than an error
       @return Whether they match
     */
    bool check_checksum(const std::string &filename, const Checksum &computed, const Checksum &stored, bool allow);

  } // namespace io

} // namespace quda
