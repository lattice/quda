#include <cstring>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/lime.h>

namespace quda
{

  namespace lime
  {

    namespace
    {
      constexpr unsigned char mb_flag = 0x80;
      constexpr unsigned char me_flag = 0x40;

      uint64_t get_big_endian(const unsigned char *p, int bytes)
      {
        uint64_t value = 0;
        for (int i = 0; i < bytes; i++) value = (value << 8) | p[i];
        return value;
      }

      void put_big_endian(unsigned char *p, uint64_t value, int bytes)
      {
        for (int i = bytes - 1; i >= 0; i--, value >>= 8) p[i] = value & 0xff;
      }

      // the fixed-size form in which rank 0 broadcasts the list of records
      struct RecordData {
        char type[type_bytes + 1];
        uint64_t data_offset;
        uint64_t data_bytes;
        int mb;
        int me;
      };
    } // namespace

    std::vector<Record> read_records(const std::string &filename)
    {
      std::vector<RecordData> list;
      if (comm_rank() == 0) {
        io::File file(filename, io::File::Mode::READ);
        const size_t size = file.size();
        for (size_t offset = 0; offset < size;) {
          unsigned char header[header_bytes];
          if (size - offset < header_bytes)
            errorQuda("%s is truncated: %lu bytes after the last LIME record", filename.c_str(), size - offset);
          file.read(header, header_bytes, offset);
          if (get_big_endian(header, 4) != magic)
            errorQuda("%s is not a LIME file: no LIME record header at byte %lu", filename.c_str(), offset);

          RecordData r = {};
          memcpy(r.type, header + 16, type_bytes);
          r.data_offset = offset + header_bytes;
          r.data_bytes = get_big_endian(header + 8, 8);
          r.mb = (header[6] & mb_flag) != 0;
          r.me = (header[6] & me_flag) != 0;
          if (r.data_bytes > size - r.data_offset)
            errorQuda("%s is truncated: record %s at byte %lu has %lu bytes of data, but the file ends after %lu",
                      filename.c_str(), r.type, offset, r.data_bytes, size - r.data_offset);
          list.push_back(r);
          offset = r.data_offset + r.data_bytes + padding(r.data_bytes);
        }
      }
      io::broadcast(list);

      std::vector<Record> records(list.size());
      for (size_t i = 0; i < list.size(); i++)
        records[i] = {list[i].type, list[i].data_offset, list[i].data_bytes, list[i].mb != 0, list[i].me != 0};
      return records;
    }

    std::string read_string(const std::string &filename, const Record &record)
    {
      if (record.data_bytes > max_string_bytes)
        errorQuda("Record %s at byte %lu of %s has %lu bytes, more than the %lu bytes allowed for a metadata record",
                  record.type.c_str(), record.data_offset - header_bytes, filename.c_str(), record.data_bytes,
                  max_string_bytes);

      std::string data;
      if (comm_rank() == 0) {
        data.resize(record.data_bytes);
        io::File file(filename, io::File::Mode::READ);
        file.read(data.data(), data.size(), record.data_offset);
      }
      io::broadcast(data);
      auto end = data.find_last_not_of('\0');
      data.resize(end == std::string::npos ? 0 : end + 1);
      return data;
    }

    size_t Writer::reserve(const std::string &type, size_t bytes, bool mb, bool me)
    {
      if (type.size() >= type_bytes) errorQuda("LIME record type %s is too long", type.c_str());

      unsigned char header[header_bytes] = {};
      put_big_endian(header, magic, 4);
      put_big_endian(header + 4, version, 2);
      header[6] = (mb ? mb_flag : 0) | (me ? me_flag : 0);
      put_big_endian(header + 8, bytes, 8);
      memcpy(header + 16, type.c_str(), type.size());
      file.write(header, header_bytes, offset);

      const size_t data_offset = offset + header_bytes;
      const unsigned char zeros[8] = {};
      if (padding(bytes) > 0) file.write(zeros, padding(bytes), data_offset + bytes);

      offset = data_offset + bytes + padding(bytes);
      return data_offset;
    }

    void Writer::write(const std::string &type, const void *data, size_t bytes, bool mb, bool me)
    {
      auto data_offset = reserve(type, bytes, mb, me);
      file.write(data, bytes, data_offset);
    }

  } // namespace lime

} // namespace quda
