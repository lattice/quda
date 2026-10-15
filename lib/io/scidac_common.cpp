#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <unistd.h>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/scidac_io.h>

namespace quda
{

  namespace scidac
  {

    namespace
    {
      constexpr const char *xml_declaration = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";

      std::string element(const std::string &tag, const std::string &value)
      {
        return "<" + tag + ">" + value + "</" + tag + ">";
      }

      std::string element(const std::string &tag, int value) { return element(tag, std::to_string(value)); }

      std::string hex(uint32_t value)
      {
        char buf[16];
        snprintf(buf, sizeof(buf), "%x", value);
        return buf;
      }

      /**
         @brief The date as QIO writes it: asctime() in UTC, with its newline replaced by " UTC"
       */
      std::string record_date()
      {
        time_t now = time(nullptr);
        struct tm utc;
        char buf[32] = {};
        if (!gmtime_r(&now, &utc) || !asctime_r(&utc, buf)) return "";
        std::string date(buf);
        if (!date.empty() && date.back() == '\n') date.back() = ' ';
        return date + "UTC";
      }

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

      bool single_parity(QudaParity parity) { return parity == QUDA_EVEN_PARITY || parity == QUDA_ODD_PARITY; }

      /**
         @brief The lattice of a file holding a field with local dimensions X (of the full field) and the given parity
       */
      std::array<int, 4> file_dims(const int *X, QudaParity parity)
      {
        auto N = io::global_dims(X);
        if (single_parity(parity)) N[0] /= 2;
        return N;
      }

      io::DataLayout field_layout(const io::HostField &field, QudaPrecision file_prec, size_t offset, QudaParity parity)
      {
        // binary data is big endian
        auto layout = field.layout(file_prec, offset, io::host_little_endian(), io::ChecksumType::SCIDAC);
        layout.parity = single_parity(parity) ? parity : QUDA_INVALID_PARITY;
        return layout;
      }
    } // namespace

    std::string file_info_xml(const std::array<int, 4> &dims, int volfmt)
    {
      std::string dims_str;
      for (auto n : dims) dims_str += std::to_string(n) + " ";
      return xml_declaration
        + element("scidacFile",
                  element("version", "1.1") + element("spacetime", 4) + element("dims", dims_str)
                    + element("volfmt", volfmt));
    }

    std::string record_info_xml(const std::string &datatype, QudaPrecision prec, int colors, int spins, int typesize,
                                int datacount)
    {
      std::string tags = element("version", "1.1") + element("date", record_date()) + element("recordtype", 0)
        + element("datatype", datatype) + element("precision", prec == QUDA_DOUBLE_PRECISION ? "D" : "F");
      if (colors != 0) tags += element("colors", colors);
      if (spins != 0) tags += element("spins", spins);
      if (typesize != 0) tags += element("typesize", typesize);
      if (datacount != 0) tags += element("datacount", datacount);
      return xml_declaration + element("scidacRecord", tags);
    }

    std::string checksum_xml(uint32_t suma, uint32_t sumb)
    {
      return xml_declaration
        + element("scidacChecksum", element("version", "1.0") + element("suma", hex(suma)) + element("sumb", hex(sumb)));
    }

    std::string ildg_format_xml(QudaPrecision prec, const std::array<int, 4> &dims)
    {
      return std::string(xml_declaration)
        + "<ildgFormat xmlns=\"http://www.lqcd.org/ildg\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "
          "xsi:schemaLocation=\"http://www.lqcd.org/ildg/filefmt.xsd\">"
        + element("version", "1.0") + element("field", "su3gauge") + element("precision", 8 * prec)
        + element("lx", dims[0]) + element("ly", dims[1]) + element("lz", dims[2]) + element("lt", dims[3])
        + "</ildgFormat>";
    }

    std::string gauge_datatype(QudaPrecision file_prec, bool ildg)
    {
      const char *p = file_prec == QUDA_DOUBLE_PRECISION ? "D" : "F";
      // QIO only writes the ILDG records for the standard USQCD (or QDP) color matrix datatypes
      return ildg ? std::string("USQCD_") + p + "3_ColorMatrix" : std::string("QUDA_") + p + "Nc3_GaugeField";
    }

    std::string vector_datatype(QudaPrecision file_prec, int nSpin, int nColor)
    {
      return std::string("QUDA_") + (file_prec == QUDA_DOUBLE_PRECISION ? "D" : "F") + "Ns" + std::to_string(nSpin)
        + "Nc" + std::to_string(nColor) + "_ColorSpinorField";
    }

    std::string quda_record_xml(int len, const std::string &datatype, QudaSiteSubset subset, QudaParity parity,
                                int nColor, int nSpin)
    {
      std::string name;
      switch (len) {
      case 6: name = "qudaStaggeredColorSpinorField"; break; // SU(3) staggered
      case 18: name = "qudaGaugeFieldFile"; break;           // SU(3) gauge field
      case 24: name = "qudaWilsonColorSpinorField"; break;   // SU(3) Wilson
      default: name = "qudaMGColorSpinorField"; break;       // MG color spinor vector
      }

      // if parity+even, it's a half-x-dim even only vector
      // if parity+odd, it's a half-x-dim odd only vector
      // if full+even, it's a full vector with only even sites filled, odd are zero
      // if full+odd, it's a full vector with only odd sites filled, even are zero
      // if full+full, it's a full vector with all sites filled (either a full ColorSpinorField or a GaugeField)
      std::string info = element("subset", subset == QUDA_PARITY_SITE_SUBSET ? "parity" : "full");
      info += element("parity", parity == QUDA_EVEN_PARITY ? "even" : parity == QUDA_ODD_PARITY ? "odd" : "full");
      info += element("nColor", nColor) + element("nSpin", nSpin);

      return xml_declaration
        + element(name, element("version", "BETA") + element("type", datatype) + element("info", info));
    }

    std::string usqcd_info_xml(double plaq, double linktr)
    {
      char plaq_str[32], linktr_str[32];
      snprintf(plaq_str, sizeof(plaq_str), "%.16e", plaq);
      snprintf(linktr_str, sizeof(linktr_str), "%.16e", linktr);
      // QIO's XML parser does not handle nested tags of the same name, so QUDA's own record XML (which has an <info>
      // tag) cannot be embedded in the <info> field
      return xml_declaration
        + element("usqcdInfo",
                  element("version", "1.0") + element("plaq", plaq_str) + element("linktr", linktr_str)
                    + element("info", "QUDA"));
    }

    std::optional<std::string> xml_value(const std::string &xml, const std::string &tag)
    {
      // whether an opening tag <tag> or <tag attributes...> starts at pos
      auto opens = [&](size_t pos) {
        if (xml.compare(pos, tag.size() + 1, "<" + tag) != 0) return false;
        auto next = pos + tag.size() + 1;
        return next < xml.size() && (xml[next] == '>' || xml[next] == '/' || isspace(xml[next]));
      };
      const std::string close = "</" + tag + ">";

      auto start = xml.find('<');
      while (start != std::string::npos && !opens(start)) start = xml.find('<', start + 1);
      if (start == std::string::npos) return std::nullopt;
      auto content = xml.find('>', start);
      if (content == std::string::npos) return std::nullopt;
      if (xml[content - 1] == '/') return std::string();
      content++;

      int depth = 1;
      for (auto pos = xml.find('<', content); pos != std::string::npos; pos = xml.find('<', pos + 1)) {
        if (opens(pos)) {
          auto end = xml.find('>', pos);
          if (end != std::string::npos && xml[end - 1] != '/') depth++;
        } else if (xml.compare(pos, close.size(), close) == 0 && --depth == 0) {
          return xml.substr(content, pos - content);
        }
      }
      return std::nullopt;
    }

    std::vector<std::vector<lime::Record>> group_messages(const std::vector<lime::Record> &records)
    {
      std::vector<std::vector<lime::Record>> messages;
      bool open = false;
      for (auto &r : records) {
        if (r.mb || !open) messages.emplace_back();
        messages.back().push_back(r);
        open = !r.me;
      }
      return messages;
    }

    const lime::Record *find_record(const std::vector<lime::Record> &message, const std::string &type)
    {
      for (auto &r : message)
        if (r.type == type) return &r;
      return nullptr;
    }

    FieldRecords find_field(const std::string &filename)
    {
      FieldRecords f;
      int partfile_name = 0;
      if (comm_rank() == 0)
        partfile_name = access(filename.c_str(), F_OK) != 0 && access((filename + ".vol0000").c_str(), F_OK) == 0;
      comm_broadcast(&partfile_name, sizeof(partfile_name));
      f.file = partfile_name ? filename + ".vol0000" : filename;
      const auto &file = f.file;

      auto records = lime::read_records(file);
      if (records.empty()) errorQuda("%s has no LIME records", file.c_str());
      auto messages = group_messages(records);
      f.native = records[0].type == private_file_xml;

      std::string file_info;
      if (f.native) {
        file_info = lime::read_string(file, records[0]);
        auto volfmt = int_value(file_info, "volfmt", file, private_file_xml);
        if (volfmt == 2)
          errorQuda("%s is a partfile (volfmt 2), which only the external QIO backend reads so far: load it with the "
                    "qio-external format",
                    file.c_str());
        if (volfmt != 0) errorQuda("%s has volume format %ld; only single files are supported", file.c_str(), volfmt);
      }

      // the field is in the first message with binary data
      const std::vector<lime::Record> *message = nullptr;
      size_t data_index = 0;
      for (auto &m : messages) {
        for (auto i = 0u; i < m.size(); i++) {
          if (m[i].type != binary_data && m[i].type != ildg_binary_data) continue;
          if (!message) {
            message = &m;
            data_index = i;
          }
          f.n_binary++;
          break;
        }
      }
      if (!message) errorQuda("%s has no %s or %s record", file.c_str(), binary_data, ildg_binary_data);
      f.data = (*message)[data_index];
      for (auto i = data_index + 1; i < message->size() && !f.checksum; i++)
        if ((*message)[i].type == checksum) f.checksum = (*message)[i];

      auto &m = f.metadata;
      const lime::Record *format_record = find_record(*message, ildg_format);
      if (!f.native && !format_record) format_record = find_record(records, ildg_format);
      m.ildg = format_record != nullptr;

      if (f.native) {
        if (int_value(file_info, "spacetime", file, private_file_xml) != 4)
          errorQuda("%s is not a 4-d lattice field file", file.c_str());
        std::istringstream dims_str(xml_value(file_info, "dims").value_or(""));
        for (auto &n : f.dims)
          if (!(dims_str >> n))
            errorQuda("Cannot read the dimensions in the %s record of %s", private_file_xml, file.c_str());

        auto private_record = find_record(*message, private_record_xml);
        if (!private_record) errorQuda("%s has no %s record for its binary data", file.c_str(), private_record_xml);
        auto record_info = lime::read_string(file, *private_record);
        // version 1.0 records give <globaldata>, version 1.1 records <recordtype>; 0 means a lattice field
        auto recordtype = xml_value(record_info, "recordtype");
        if (!recordtype) recordtype = xml_value(record_info, "globaldata");
        if (recordtype && std::strtol(recordtype->c_str(), nullptr, 10) != 0)
          errorQuda("The binary data of %s is not a lattice field (record type %s)", file.c_str(), recordtype->c_str());
        m.datatype = xml_value(record_info, "datatype").value_or("");
        m.prec = xml_value(record_info, "precision").value_or("") == "F" ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
        // QIO omits colors and spins when they are zero
        m.colors = std::strtol(xml_value(record_info, "colors").value_or("0").c_str(), nullptr, 10);
        m.spins = std::strtol(xml_value(record_info, "spins").value_or("0").c_str(), nullptr, 10);
        m.typesize = int_value(record_info, "typesize", file, private_record_xml);
        m.datacount = int_value(record_info, "datacount", file, private_record_xml);

        if (auto user_record = find_record(*message, record_xml)) m.user_xml = lime::read_string(file, *user_record);
      } else {
        if (!format_record)
          errorQuda("%s is neither a SciDAC file (its first record is %s) nor an ILDG file (it has no %s record)",
                    file.c_str(), records[0].type.c_str(), ildg_format);
        auto format = lime::read_string(file, *format_record);
        auto bits = int_value(format, "precision", file, ildg_format);
        if (bits != 32 && bits != 64) errorQuda("Unsupported ILDG precision %ld in %s", bits, file.c_str());
        m.prec = bits == 64 ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
        const char *dim_tags[] = {"lx", "ly", "lz", "lt"};
        for (int d = 0; d < 4; d++) f.dims[d] = int_value(format, dim_tags[d], file, ildg_format);
        // ILDG files hold gauge fields: four 3x3 complex matrices per site
        m.typesize = 18 * m.prec;
        m.datacount = 4;
      }

      return f;
    }

    FieldChecksum read_field(const FieldRecords &records, const io::HostField &field, const int *X, QudaParity parity,
                             const FileIOParam &param)
    {
      const auto &file = records.file;
      const auto &m = records.metadata;
      const auto N = file_dims(X, parity);
      const auto &D = records.dims;
      if (D != N)
        errorQuda("%s has dimensions %d %d %d %d, but the field's lattice is %d %d %d %d%s", file.c_str(), D[0], D[1],
                  D[2], D[3], N[0], N[1], N[2], N[3],
                  single_parity(parity) ? " (half the global x extent, as single-parity fields are stored)" : "");
      if (m.typesize != static_cast<int>(field.item_bytes(m.prec)))
        errorQuda("The typesize %d of %s does not match the expected datasize %lu", m.typesize, file.c_str(),
                  field.item_bytes(m.prec));

      auto layout = field_layout(field, m.prec, records.data.data_offset, parity);
      const auto bytes = io::data_bytes(layout, io::global_dims(X));
      if (records.data.data_bytes != bytes)
        errorQuda("The binary data of %s is %lu bytes, but should be %lu bytes", file.c_str(), records.data.data_bytes,
                  bytes);

      FieldChecksum c;
      c.sum
        = io::read_data(file, layout, X, io::Partitions(param.io_mode, param.io_partition_grid), field.unpacker(m.prec));

      // as in QIO, a SciDAC file must have a checksum, while ILDG files from other codes have none
      if (records.native) {
        if (!records.checksum) errorQuda("%s has no %s record after its binary data", file.c_str(), checksum);
        auto xml = lime::read_string(file, *records.checksum);
        io::Checksum stored = {hex_value(xml, "suma", file), hex_value(xml, "sumb", file)};
        c.has_checksum = true;
        c.checksum_matches = io::check_checksum(file, c.sum, stored, param.allow_checksum_mismatch);
      }
      return c;
    }

    io::Checksum write_field(const std::string &filename, const FieldMetadata &metadata, const io::HostField &field,
                             const int *X, QudaParity parity, const FileIOParam &param)
    {
      // checks the grid before rank 0 truncates the file
      const io::Partitions partitions(param.io_mode, param.io_partition_grid);
      const auto &m = metadata;
      const auto N = file_dims(X, parity);
      auto layout = field_layout(field, m.prec, 0, parity);
      const size_t bytes = io::data_bytes(layout, io::global_dims(X));

      // rank 0 writes everything but the binary data and the checksum, which follows it
      uint64_t data_offset = 0;
      if (comm_rank() == 0) {
        io::File file(filename, io::File::Mode::CREATE);
        lime::Writer writer(file);
        writer.write(private_file_xml, file_info_xml(N, 0), true, false);
        writer.write(file_xml, std::string("Dummy user file XML"), false, true);
        writer.write(private_record_xml,
                     record_info_xml(m.datatype, m.prec, m.colors, m.spins, m.typesize, m.datacount), true, false);
        writer.write(record_xml, m.user_xml, false, false);
        if (m.ildg) {
          writer.write(ildg_format, ildg_format_xml(m.prec, N), false, false);
          if (!m.ildg_lfn.empty()) writer.write(ildg_data_lfn, m.ildg_lfn, false, false);
        }
        data_offset = writer.reserve(m.ildg ? ildg_binary_data : binary_data, bytes, false, false);
        file.close();
      }
      comm_broadcast(&data_offset, sizeof(data_offset));
      layout.offset = data_offset;

      auto sum = io::write_data(filename, layout, X, partitions, field.packer(m.prec));

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
