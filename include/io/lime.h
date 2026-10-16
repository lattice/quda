#pragma once

// The LIME container used by SciDAC and ILDG files: a sequence of
// records, each a 144-byte big-endian header followed by the record's
// data, padded with zeros to a multiple of 8 bytes.  Records are grouped
// into messages by the header's message-begin (MB) and message-end (ME)
// flags.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <io/site_io.h>

namespace quda
{

  namespace lime
  {

    constexpr uint32_t magic = 0x456789ab;
    constexpr uint16_t version = 1;
    constexpr size_t header_bytes = 144;
    constexpr size_t type_bytes = 128;

    /**
       @brief A record found in a LIME file
     */
    struct Record {
      std::string type;
      size_t data_offset = 0; /** Where the record's data starts */
      size_t data_bytes = 0;  /** Length of the data, without padding */
      bool mb = false;        /** First record of a message */
      bool me = false;        /** Last record of a message */
    };

    /**
       @brief The number of zero bytes that follow a record's data
     */
    constexpr size_t padding(size_t bytes) { return (8 - bytes % 8) % 8; }

    /**
       @brief List the records of a LIME file.  Rank 0 reads the record
       headers and broadcasts the list.  Errors if the file is not a
       LIME file or is truncated.
     */
    std::vector<Record> read_records(const std::string &filename);

    /**
       @brief List the records of an open LIME file on the calling rank alone, with the same checks as read_records
       @param[in] file The file
       @param[in] filename Its name, for messages
     */
    std::vector<Record> list_records(const io::File &file, const std::string &filename);

    /**
       @brief The largest record read_string accepts.  Metadata records
       are XML of at most a few kilobytes; the cap keeps a corrupt or
       unexpected record from being allocated on every rank.
     */
    constexpr size_t max_string_bytes = size_t(16) << 20;

    /**
       @brief Read the data of a (small) record as a string, without
       any trailing NUL characters.  Rank 0 reads it and broadcasts it.
       Errors if the record is longer than max_string_bytes.
     */
    std::string read_string(const std::string &filename, const Record &record);

    /**
       @brief Read the data of several (small) records as strings, as
       read_string does, with one open of the file on rank 0 and one
       broadcast
     */
    std::vector<std::string> read_strings(const std::string &filename, const std::vector<Record> &records);

    /**
       @brief Writes records one after another to a file.  Used on rank 0 only.
     */
    class Writer
    {
      const io::File &file;
      size_t offset;

    public:
      /**
         @param[in] file The file
         @param[in] offset Where the next record starts
       */
      Writer(const io::File &file, size_t offset = 0) : file(file), offset(offset) { }

      /**
         @brief Where the next record starts
       */
      size_t end() const { return offset; }

      /**
         @brief Write a record's header and the padding after its data,
         leaving room for the data, which is written separately
         @return The offset of the record's data
       */
      size_t reserve(const std::string &type, size_t bytes, bool mb, bool me);

      /**
         @brief Write a record
       */
      void write(const std::string &type, const void *data, size_t bytes, bool mb, bool me);

      /**
         @brief Write a string record, including its terminating NUL, as SciDAC files do
       */
      void write(const std::string &type, const std::string &data, bool mb, bool me)
      {
        write(type, data.c_str(), data.size() + 1, mb, me);
      }
    };

  } // namespace lime

} // namespace quda
