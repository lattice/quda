#include <cstdlib>
#include <sstream>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/host_field.h>
#include <io/scidac_io.h>

namespace quda
{

  namespace scidac
  {

    namespace
    {
      constexpr int n_dim = 4;
      constexpr int matrix_reals = 18;

      long int_value(const std::string &xml, const char *tag, const std::string &filename, const char *record)
      {
        auto value = xml_value(xml, tag);
        if (!value) errorQuda("The %s record of %s has no <%s> element", record, filename.c_str(), tag);
        return std::strtol(value->c_str(), nullptr, 10);
      }

      uint32_t hex_value(const std::string &xml, const char *tag, const std::string &filename)
      {
        auto value = xml_value(xml, tag);
        if (!value) errorQuda("The %s record of %s has no <%s> element", checksum, filename.c_str(), tag);
        return static_cast<uint32_t>(std::strtoul(value->c_str(), nullptr, 16));
      }

      void read_usqcd_info(const std::string &user_xml, GaugeInfo &info)
      {
        auto usqcd = xml_value(user_xml, "usqcdInfo");
        if (!usqcd) return;
        auto plaq = xml_value(*usqcd, "plaq");
        auto linktr = xml_value(*usqcd, "linktr");
        if (!plaq || !linktr) return;
        info.has_usqcd_info = true;
        info.plaq = std::strtod(plaq->c_str(), nullptr);
        info.linktr = std::strtod(linktr->c_str(), nullptr);
      }
    } // namespace

    GaugeInfo load_gauge(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                         const FileIOParam &param)
    {
      io::HostField field(gauge, n_dim, prec, matrix_reals);

      auto records = lime::read_records(filename);
      if (records.empty()) errorQuda("%s has no LIME records", filename.c_str());
      auto messages = group_messages(records);

      // ILDG files written by other codes need not have the SciDAC records
      const bool native = records[0].type == private_file_xml;

      // the field is in the first message with binary data
      const std::vector<lime::Record> *message = nullptr;
      const lime::Record *data = nullptr;
      for (auto &m : messages) {
        data = find_record(m, binary_data);
        if (!data) data = find_record(m, ildg_binary_data);
        if (data) {
          message = &m;
          break;
        }
      }
      if (!data) errorQuda("%s has no %s or %s record", filename.c_str(), binary_data, ildg_binary_data);

      const lime::Record *format_record = find_record(*message, ildg_format);
      if (!native && !format_record) format_record = find_record(records, ildg_format);

      GaugeInfo info;
      info.ildg = format_record != nullptr;
      std::array<int, n_dim> dims;
      long typesize = 0, datacount = 0;

      if (native) {
        auto file_info = lime::read_string(filename, records[0]);
        if (int_value(file_info, "spacetime", filename, private_file_xml) != n_dim)
          errorQuda("%s is not a 4-d lattice field file", filename.c_str());
        std::istringstream dims_str(xml_value(file_info, "dims").value_or(""));
        for (auto &n : dims)
          if (!(dims_str >> n))
            errorQuda("Cannot read the dimensions in the %s record of %s", private_file_xml, filename.c_str());
        auto volfmt = int_value(file_info, "volfmt", filename, private_file_xml);
        if (volfmt != 0)
          errorQuda("%s is a %s file (volfmt %ld); only single files are supported", filename.c_str(),
                    volfmt == 1   ? "multifile" :
                      volfmt == 2 ? "partfile" :
                                    "volume format",
                    volfmt);

        auto private_record = find_record(*message, private_record_xml);
        if (!private_record) errorQuda("%s has no %s record for its binary data", filename.c_str(), private_record_xml);
        auto record_info = lime::read_string(filename, *private_record);
        // version 1.0 records give <globaldata>, version 1.1 records <recordtype>; 0 means a lattice field
        auto recordtype = xml_value(record_info, "recordtype");
        if (!recordtype) recordtype = xml_value(record_info, "globaldata");
        if (recordtype && std::strtol(recordtype->c_str(), nullptr, 10) != 0)
          errorQuda("The binary data of %s is not a lattice field (record type %s)", filename.c_str(),
                    recordtype->c_str());
        info.file_prec
          = xml_value(record_info, "precision").value_or("") == "F" ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
        typesize = int_value(record_info, "typesize", filename, private_record_xml);
        datacount = int_value(record_info, "datacount", filename, private_record_xml);

        if (auto user_record = find_record(*message, record_xml))
          read_usqcd_info(lime::read_string(filename, *user_record), info);
      } else {
        if (!format_record)
          errorQuda("%s is neither a SciDAC file (its first record is %s) nor an ILDG file (it has no %s record)",
                    filename.c_str(), records[0].type.c_str(), ildg_format);
        auto format = lime::read_string(filename, *format_record);
        auto bits = int_value(format, "precision", filename, ildg_format);
        if (bits != 32 && bits != 64) errorQuda("Unsupported ILDG precision %ld in %s", bits, filename.c_str());
        info.file_prec = bits == 64 ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
        const char *dim_tags[] = {"lx", "ly", "lz", "lt"};
        for (int d = 0; d < n_dim; d++) dims[d] = int_value(format, dim_tags[d], filename, ildg_format);
        typesize = matrix_reals * info.file_prec;
        datacount = n_dim;
      }

      auto N = io::global_dims(X);
      if (dims != N)
        errorQuda("%s has dimensions %d %d %d %d, but the lattice is %d %d %d %d", filename.c_str(), dims[0], dims[1],
                  dims[2], dims[3], N[0], N[1], N[2], N[3]);
      if (datacount != n_dim)
        errorQuda("The datacount %ld of %s does not match the expected number of fields %d", datacount,
                  filename.c_str(), n_dim);
      if (typesize != matrix_reals * info.file_prec)
        errorQuda("The typesize %ld of %s does not match the expected datasize %d", typesize, filename.c_str(),
                  matrix_reals * info.file_prec);

      // binary data is big endian, with the four links of a site as row-major 3x3 complex matrices
      auto layout = field.layout(info.file_prec, data->data_offset, io::host_little_endian(), io::ChecksumType::SCIDAC);
      if (data->data_bytes != io::data_bytes(layout, N))
        errorQuda("The binary data of %s is %lu bytes, but should be %lu bytes", filename.c_str(), data->data_bytes,
                  io::data_bytes(layout, N));

      info.sum = io::read_data(filename, layout, X, io::Partitions(param.io_mode, param.io_partition_grid),
                               field.unpacker(info.file_prec));

      // as in QIO, a SciDAC file must have a checksum, while ILDG files from other codes have none
      if (native) {
        const lime::Record *checksum_record = nullptr;
        for (auto r = data; r < message->data() + message->size() && !checksum_record; r++)
          if (r->type == checksum) checksum_record = r;
        if (!checksum_record) errorQuda("%s has no %s record after its binary data", filename.c_str(), checksum);
        auto xml = lime::read_string(filename, *checksum_record);
        io::Checksum stored = {hex_value(xml, "suma", filename), hex_value(xml, "sumb", filename)};
        info.has_checksum = true;
        info.checksum_matches = io::check_checksum(filename, info.sum, stored, param.allow_checksum_mismatch);
      }

      return info;
    }

    io::Checksum save_gauge(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                            const FileIOParam &param, const GaugeInfo &info)
    {
      // checks the grid before rank 0 truncates the file
      const io::Partitions partitions(param.io_mode, param.io_partition_grid);
      io::HostField field(gauge, n_dim, prec, matrix_reals);
      const QudaPrecision file_prec = prec;
      const int typesize = matrix_reals * file_prec;
      const auto N = io::global_dims(X);
      const auto datatype = gauge_datatype(file_prec, info.ildg);
      auto layout = field.layout(file_prec, 0, io::host_little_endian(), io::ChecksumType::SCIDAC);
      const size_t bytes = io::data_bytes(layout, N);

      // rank 0 writes everything but the binary data and the checksum, which follows it
      uint64_t data_offset = 0;
      if (comm_rank() == 0) {
        io::File file(filename, io::File::Mode::CREATE);
        lime::Writer writer(file);
        writer.write(private_file_xml, file_info_xml(N, 0), true, false);
        writer.write(file_xml, std::string("Dummy user file XML"), false, true);
        writer.write(private_record_xml, record_info_xml(datatype, file_prec, 3, 0, typesize, n_dim), true, false);
        writer.write(record_xml,
                     info.has_usqcd_info ?
                       usqcd_info_xml(info.plaq, info.linktr) :
                       quda_record_xml(matrix_reals, datatype, QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, 3, 0),
                     false, false);
        if (info.ildg) {
          writer.write(ildg_format, ildg_format_xml(file_prec, N), false, false);
          if (!info.ildg_lfn.empty()) writer.write(ildg_data_lfn, info.ildg_lfn, false, false);
        }
        data_offset = writer.reserve(info.ildg ? ildg_binary_data : binary_data, bytes, false, false);
        file.close();
      }
      comm_broadcast(&data_offset, sizeof(data_offset));
      layout.offset = data_offset;

      auto sum = io::write_data(filename, layout, X, partitions, field.packer(file_prec));

      if (comm_rank() == 0) {
        io::File file(filename, io::File::Mode::WRITE);
        lime::Writer writer(file, data_offset + bytes + lime::padding(bytes));
        writer.write(checksum, checksum_xml(sum.sum29, sum.sum31), false, true);
        file.close();
      }
      return sum;
    }

  } // namespace scidac

} // namespace quda
