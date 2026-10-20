#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <dirent.h>
#include <unistd.h>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/scidac_io.h>

extern const char *gitversion;

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

      std::string format(const char *fmt, double value)
      {
        char buf[64];
        snprintf(buf, sizeof(buf), fmt, value);
        return buf;
      }

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

    std::string volume_name(const std::string &filename, int n)
    {
      char suffix[24];
      snprintf(suffix, sizeof(suffix), ".vol%04d", n);
      return filename + suffix;
    }

    namespace
    {
      // whether the suffix of a volume name after ".vol" is a volume number: QIO writes %04d of an int
      bool volume_number(const std::string &digits)
      {
        return digits.size() >= 4 && digits.size() <= 9
          && std::all_of(digits.begin(), digits.end(), [](unsigned char c) { return std::isdigit(c); });
      }

      // the partfile a file named as a partfile volume belongs to, or an empty string
      std::string partfile_of_volume(const std::string &file)
      {
        const auto pos = file.rfind(".vol");
        return pos != std::string::npos && volume_number(file.substr(pos + 4)) ? file.substr(0, pos) : std::string();
      }
    } // namespace

    std::string hypercube_str(const io::Hypercube &box)
    {
      char buf[128];
      snprintf(buf, sizeof(buf), "%d %d %d %d .. %d %d %d %d", box.lower[0], box.lower[1], box.lower[2], box.lower[3],
               box.upper[0], box.upper[1], box.upper[2], box.upper[3]);
      return buf;
    }

    namespace
    {
      // the record types of QIO's private record XML
      constexpr int field_recordtype = 0;
      constexpr int hypercube_recordtype = 2;

      // an integer list as QIO writes it, each integer followed by a space
      std::string int_list(const std::array<int, 4> &values)
      {
        std::string s;
        for (auto v : values) s += std::to_string(v) + " ";
        return s;
      }
    } // namespace

    std::string record_info_xml(const std::string &datatype, QudaPrecision prec, int colors, int spins, int typesize,
                                int datacount, const std::optional<io::Hypercube> &hypercube)
    {
      std::string tags = element("version", "1.1") + element("date", record_date())
        + element("recordtype", hypercube ? hypercube_recordtype : field_recordtype);
      if (hypercube)
        tags += element("spacetime", 4) + element("hyperlower", int_list(hypercube->lower))
          + element("hyperupper", int_list(hypercube->upper));
      tags += element("datatype", datatype) + element("precision", prec == QUDA_DOUBLE_PRECISION ? "D" : "F");
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

    std::string quda_file_xml(const std::string &content)
    {
      std::string tags = element("version", "1.0");
      if (gitversion) tags += element("commit", gitversion);
      return xml_declaration + element("qudaFile", tags + element("content", content));
    }

    std::string vector_datatype(QudaPrecision file_prec, int nSpin, int nColor, bool legacy)
    {
      const std::string p = file_prec == QUDA_DOUBLE_PRECISION ? "D" : "F";
      if (!legacy && nColor == 3 && nSpin == 1) return color_vector_datatype(file_prec);
      if (!legacy && nColor == 3 && nSpin == 4) return "USQCD_" + p + "3_DiracFermion";
      return "QUDA_" + p + "Ns" + std::to_string(nSpin) + "Nc" + std::to_string(nColor) + "_ColorSpinorField";
    }

    std::string color_vector_datatype(QudaPrecision file_prec)
    {
      return std::string("USQCD_") + (file_prec == QUDA_DOUBLE_PRECISION ? "D" : "F") + "3_ColorVector";
    }

    bool spins_match(const std::string &datatype, int spins, int nSpin)
    {
      if (spins == nSpin) return true;
      return spins == 0 && nSpin == 1
        && (datatype == color_vector_datatype(QUDA_SINGLE_PRECISION)
            || datatype == color_vector_datatype(QUDA_DOUBLE_PRECISION));
    }

    namespace
    {
      // the element name of QUDA's record XML
      const char *quda_record_name(int len)
      {
        switch (len) {
        case 6: return "qudaStaggeredColorSpinorField"; // SU(3) staggered
        case 18: return "qudaGaugeFieldFile";           // SU(3) gauge field
        case 24: return "qudaWilsonColorSpinorField";   // SU(3) Wilson
        default: return "qudaMGColorSpinorField";       // MG color spinor vector
        }
      }

      // the content of the <info> element of QUDA's record XML
      std::string quda_record_info(QudaSiteSubset subset, QudaParity parity, int nColor, int nSpin,
                                   std::optional<double> eval, std::optional<double> eval_imag,
                                   std::optional<double> resid, const std::string &sign_convention = "")
      {
        // if parity+even, it's a half-x-dim even only vector
        // if parity+odd, it's a half-x-dim odd only vector
        // if full+even, it's a full vector with only even sites filled, odd are zero
        // if full+odd, it's a full vector with only odd sites filled, even are zero
        // if full+full, it's a full vector with all sites filled (either a full ColorSpinorField or a GaugeField)
        std::string info = element("subset", subset == QUDA_PARITY_SITE_SUBSET ? "parity" : "full");
        info += element("parity", parity == QUDA_EVEN_PARITY ? "even" : parity == QUDA_ODD_PARITY ? "odd" : "full");
        info += element("nColor", nColor) + element("nSpin", nSpin);
        // with the precision of MILC's eigenvector records
        if (eval) info += element("eigVal", format("%.18e", *eval));
        if (eval_imag) info += element("eigValImag", format("%.18e", *eval_imag));
        if (resid) info += element("resid", format("%e", *resid));
        if (!sign_convention.empty()) info += element("staggeredSignConvention", sign_convention);
        return info;
      }
    } // namespace

    std::string quda_record_xml(int len, QudaSiteSubset subset, QudaParity parity, int nColor, int nSpin,
                                std::optional<double> eval, std::optional<double> eval_imag,
                                std::optional<double> resid, const std::string &sign_convention)
    {
      const auto info = quda_record_info(subset, parity, nColor, nSpin, eval, eval_imag, resid, sign_convention);
      return xml_declaration + element(quda_record_name(len), element("version", "1.0") + element("info", info));
    }

    std::string legacy_quda_record_xml(int len, const std::string &datatype, QudaSiteSubset subset, QudaParity parity,
                                       int nColor, int nSpin, std::optional<double> eval, std::optional<double> resid)
    {
      const auto info = quda_record_info(subset, parity, nColor, nSpin, eval, {}, resid);
      return xml_declaration
        + element(quda_record_name(len), element("version", "BETA") + element("type", datatype) + element("info", info));
    }

    std::string milc_eigen_file_xml(int n_vectors)
    {
      return xml_declaration
        + element("info", element("title", "KS eigenvalues and vectors") + element("Nvecs", n_vectors));
    }

    std::string milc_eigen_record_xml(double eval, double resid)
    {
      return xml_declaration
        + element("info", element("eigVal", format("%.18e", eval)) + element("resid", format("%e", resid)));
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

    namespace
    {
      // whether <filename> and the first volume of a partfile of that name exist, decided on rank 0
      std::array<int, 2> files_named(const std::string &filename)
      {
        std::array<int, 2> exists = {0, 0};
        if (comm_rank() == 0) {
          exists[0] = access(filename.c_str(), F_OK) == 0;
          exists[1] = access(volume_name(filename, 0).c_str(), F_OK) == 0;
        }
        comm_broadcast(exists.data(), sizeof(exists));
        return exists;
      }
    } // namespace

    bool partfile_named(const std::string &filename)
    {
      const auto exists = files_named(filename);
      return !exists[0] && exists[1];
    }

    namespace
    {
      /**
         @brief The box of a hypercube record from its private record XML: <hyperlower> and <hyperupper> of four
         integers each, within the lattice, after a <spacetime> of 4 if there is one
       */
      io::Hypercube parse_hypercube(const std::string &record_info, const std::array<int, 4> &dims,
                                    const std::string &what)
      {
        auto spacetime = xml_value(record_info, "spacetime");
        if (spacetime && std::strtol(spacetime->c_str(), nullptr, 10) != 4)
          errorQuda("%s is a hypercube record with <spacetime>%s</spacetime>, but only 4-d records are supported",
                    what.c_str(), spacetime->c_str());
        io::Hypercube box;
        for (auto [tag, bound] : {std::pair {"hyperlower", &box.lower}, std::pair {"hyperupper", &box.upper}}) {
          auto value = xml_value(record_info, tag);
          if (!value) errorQuda("%s is a hypercube record without a <%s> element", what.c_str(), tag);
          std::istringstream in(*value);
          std::string extra;
          for (auto &n : *bound)
            if (!(in >> n)) errorQuda("The <%s> of %s, a hypercube record, is not 4 integers", tag, what.c_str());
          if (in >> extra) errorQuda("The <%s> of %s, a hypercube record, is not 4 integers", tag, what.c_str());
        }
        for (int d = 0; d < 4; d++)
          if (box.lower[d] < 0 || box.lower[d] > box.upper[d] || box.upper[d] >= dims[d])
            errorQuda("%s is a hypercube record of the sites %s, which are not a box of the lattice %d %d %d %d",
                      what.c_str(), hypercube_str(box).c_str(), dims[0], dims[1], dims[2], dims[3]);
        return box;
      }
    } // namespace

    void check_no_hypercube(const FieldRecords &records, int n)
    {
      const auto count = n < 0 ? records.fields.size() : std::min(records.fields.size(), static_cast<size_t>(n));
      for (auto k = 0u; k < count; k++) {
        const auto &box = records.fields[k].metadata.hypercube;
        if (box)
          errorQuda("Binary record %u of %s is a hypercube record (sites %s), which only staggered source files hold",
                    k, records.file.c_str(), hypercube_str(*box).c_str());
      }
    }

    FieldRecords find_field(const std::string &filename)
    {
      FieldRecords f;
      f.filename = filename;
      const auto exists = files_named(filename);
      const bool partfile_name = !exists[0] && exists[1];
      if (exists[0] && exists[1])
        warningQuda("Loading %s rather than the partfile volumes %s...: as in QIO, a file of the name given comes "
                    "first, and one of them may be left over from an earlier save",
                    filename.c_str(), volume_name(filename, 0).c_str());
      f.file = partfile_name ? volume_name(filename, 0) : filename;
      const auto &file = f.file;

      auto records = lime::read_records(file);
      if (records.empty()) errorQuda("%s has no LIME records", file.c_str());
      auto messages = group_messages(records);
      f.native = records[0].type == private_file_xml;

      const auto partfile = partfile_of_volume(file);
      if (records[0].type == sitelist && !partfile.empty())
        errorQuda("%s is a volume of a partfile: load the partfile by the name %s", file.c_str(), partfile.c_str());

      // each field is a message with binary data; ILDG files from other codes hold one gauge field
      struct Message {
        const std::vector<lime::Record> *records;
        size_t data;
      };
      std::vector<Message> found;
      for (auto &m : messages) {
        for (auto i = 0u; i < m.size(); i++) {
          if (m[i].type != binary_data && m[i].type != ildg_binary_data) continue;
          found.push_back({&m, i});
          break;
        }
      }
      if (found.empty()) errorQuda("%s has no %s or %s record", file.c_str(), binary_data, ildg_binary_data);
      if (!f.native) found.resize(1);

      // rank 0 reads every metadata record at once
      std::vector<lime::Record> wanted;
      auto want = [&](const lime::Record *r) {
        if (!r) return -1;
        wanted.push_back(*r);
        return static_cast<int>(wanted.size()) - 1;
      };
      const int file_info_index = f.native ? want(&records[0]) : -1;
      const int user_file_index = f.native ? want(find_record(records, file_xml)) : -1;
      struct Strings {
        int private_record, user_record, checksum, ildg_format;
      };
      std::vector<Strings> indices;
      std::vector<bool> has_ildg_format;
      for (auto &m : found) {
        const auto &message = *m.records;
        const lime::Record *sum = nullptr;
        for (auto i = m.data + 1; i < message.size() && !sum; i++)
          if (message[i].type == checksum) sum = &message[i];
        const lime::Record *format_record = find_record(message, ildg_format);
        if (!f.native && !format_record) format_record = find_record(records, ildg_format);
        has_ildg_format.push_back(format_record != nullptr);
        // SciDAC files give the metadata in their private records; ILDG files from other codes only in ildg-format
        if (f.native)
          indices.push_back(
            {want(find_record(message, private_record_xml)), want(find_record(message, record_xml)), want(sum), -1});
        else
          indices.push_back({-1, -1, -1, want(format_record)});
      }
      const auto strings = lime::read_strings(file, wanted);
      auto text = [&](int index) { return index < 0 ? std::string() : strings[index]; };

      std::string file_info;
      if (f.native) {
        file_info = text(file_info_index);
        f.user_file_xml = text(user_file_index);
        f.volfmt = int_value(file_info, "volfmt", file, private_file_xml);
        if (f.volfmt == partfile_volfmt && !partfile_name) {
          if (!partfile.empty())
            errorQuda("%s is the first volume of a partfile: load the partfile by the name %s", file.c_str(),
                      partfile.c_str());
          errorQuda("%s is the first volume of a partfile (volfmt 2), but partfile volumes are named %s", file.c_str(),
                    volume_name(file, 0).c_str());
        }
        if (f.volfmt != single_file_volfmt && f.volfmt != partfile_volfmt)
          errorQuda("%s has volume format %d; only single files and partfiles are supported", file.c_str(), f.volfmt);
      }

      if (f.native) {
        if (int_value(file_info, "spacetime", file, private_file_xml) != 4)
          errorQuda("%s is not a 4-d lattice field file", file.c_str());
        std::istringstream dims_str(xml_value(file_info, "dims").value_or(""));
        for (auto &n : f.dims)
          if (!(dims_str >> n))
            errorQuda("Cannot read the dimensions in the %s record of %s", private_file_xml, file.c_str());
      }

      for (auto k = 0u; k < found.size(); k++) {
        FieldRecord r;
        r.index = k;
        r.data = (*found[k].records)[found[k].data];
        auto &m = r.metadata;
        m.ildg = has_ildg_format[k];
        const auto &index = indices[k];
        const std::string what = found.size() > 1 ? "binary record " + std::to_string(k) + " of " + file : file;

        if (f.native) {
          if (index.private_record < 0)
            errorQuda("%s has no %s record for its binary data", what.c_str(), private_record_xml);
          const auto record_info = text(index.private_record);
          // version 1.0 records give <globaldata>, version 1.1 records <recordtype>; 0 means a lattice field, 2 a
          // hypercube (a box of the lattice)
          auto recordtype = xml_value(record_info, "recordtype");
          if (!recordtype) recordtype = xml_value(record_info, "globaldata");
          const long type = recordtype ? std::strtol(recordtype->c_str(), nullptr, 10) : field_recordtype;
          if (type == hypercube_recordtype)
            m.hypercube = parse_hypercube(record_info, f.dims, what);
          else if (type != field_recordtype)
            errorQuda("The binary data of %s is not a lattice field (record type %s)", what.c_str(), recordtype->c_str());
          m.datatype = xml_value(record_info, "datatype").value_or("");
          m.prec
            = xml_value(record_info, "precision").value_or("") == "F" ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
          // QIO omits colors and spins when they are zero
          m.colors = std::strtol(xml_value(record_info, "colors").value_or("0").c_str(), nullptr, 10);
          m.spins = std::strtol(xml_value(record_info, "spins").value_or("0").c_str(), nullptr, 10);
          m.typesize = int_value(record_info, "typesize", what, private_record_xml);
          m.datacount = int_value(record_info, "datacount", what, private_record_xml);
          m.user_xml = text(index.user_record);
          if (index.checksum >= 0) {
            const auto xml = text(index.checksum);
            r.checksum = io::Checksum {hex_value(xml, "suma", what), hex_value(xml, "sumb", what)};
          }
        } else {
          if (index.ildg_format < 0)
            errorQuda("%s is neither a SciDAC file (its first record is %s) nor an ILDG file (it has no %s record)",
                      file.c_str(), records[0].type.c_str(), ildg_format);
          const auto format = text(index.ildg_format);
          auto bits = int_value(format, "precision", file, ildg_format);
          if (bits != 32 && bits != 64) errorQuda("Unsupported ILDG precision %ld in %s", bits, file.c_str());
          m.prec = bits == 64 ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
          const char *dim_tags[] = {"lx", "ly", "lz", "lt"};
          for (int d = 0; d < 4; d++) f.dims[d] = int_value(format, dim_tags[d], file, ildg_format);
          // ILDG files hold gauge fields: four 3x3 complex matrices per site
          m.typesize = 18 * m.prec;
          m.datacount = 4;
        }
        f.fields.push_back(r);
      }

      return f;
    }

    namespace
    {
      using coords_t = std::array<int, 4>;

      // site lists are read and written in pieces of this size
      constexpr size_t sitelist_chunk_bytes = size_t(8) << 20;

      uint64_t lattice_volume(const coords_t &S) { return static_cast<uint64_t>(S[0]) * S[1] * S[2] * S[3]; }

      uint64_t lattice_index(const coords_t &x, const coords_t &S)
      {
        return x[0] + S[0] * (x[1] + S[1] * (x[2] + static_cast<uint64_t>(S[2]) * x[3]));
      }

      coords_t lattice_coords(uint64_t index, const coords_t &S)
      {
        coords_t x;
        for (int d = 0; d < 4; d++) {
          x[d] = index % S[d];
          index /= S[d];
        }
        return x;
      }

      /**
         @brief Call f(first, n) for each x row of a box, in lexicographic
         order, where first is the global lexicographic index of the row's
         first site and n its length
       */
      template <typename F> void for_each_box_row(const io::FilePart &box, const coords_t &S, F &&f)
      {
        const auto &P = box.corner;
        const auto &E = box.extent;
        for (int t = 0; t < E[3]; t++)
          for (int z = 0; z < E[2]; z++)
            for (int y = 0; y < E[1]; y++) f(lattice_index({P[0], P[1] + y, P[2] + z, P[3] + t}, S), E[0]);
      }

      /**
         @brief The bytes per entry of a volume's site list: as in QIO,
         32 bits if every index of the volume fits, else 64
       */
      size_t sitelist_width(const io::FilePart &box, const coords_t &S)
      {
        coords_t last;
        for (int d = 0; d < 4; d++) last[d] = box.corner[d] + box.extent[d] - 1;
        return lattice_index(last, S) <= UINT32_MAX ? 4 : 8;
      }

      void write_sitelist(const io::File &file, size_t offset, const io::FilePart &box, const coords_t &S)
      {
        const size_t width = sitelist_width(box, S);
        std::vector<unsigned char> chunk;
        chunk.reserve(sitelist_chunk_bytes + width * box.extent[0]);
        auto flush = [&]() {
          file.write(chunk.data(), chunk.size(), offset);
          offset += chunk.size();
          chunk.clear();
        };
        for_each_box_row(box, S, [&](uint64_t first, int n) {
          for (int x = 0; x < n; x++)
            for (int b = static_cast<int>(width) - 1; b >= 0; b--) chunk.push_back(((first + x) >> (8 * b)) & 0xff);
          if (chunk.size() >= sitelist_chunk_bytes) flush();
        });
        if (!chunk.empty()) flush();
      }

      /**
         @brief The partfile volumes in the directory of a file: <filename>.vol followed by four or more digits, in
         ascending order of their numbers
       */
      std::vector<std::string> list_volumes(const std::string &filename)
      {
        const auto slash = filename.rfind('/');
        const std::string dir = slash == std::string::npos ? "." : slash == 0 ? "/" : filename.substr(0, slash);
        const std::string path = slash == std::string::npos ? "" : filename.substr(0, slash + 1);
        const std::string prefix = filename.substr(path.size()) + ".vol";

        DIR *d = opendir(dir.c_str());
        if (!d) errorQuda("Cannot list the directory %s to find the volumes of %s", dir.c_str(), filename.c_str());
        std::vector<std::pair<long, std::string>> found;
        while (auto entry = readdir(d)) {
          const std::string name = entry->d_name;
          if (name.compare(0, prefix.size(), prefix) != 0) continue;
          const auto digits = name.substr(prefix.size());
          if (!volume_number(digits)) continue;
          found.emplace_back(std::strtol(digits.c_str(), nullptr, 10), path + name);
        }
        closedir(d);

        std::sort(found.begin(), found.end());
        std::vector<std::string> volumes;
        for (auto &f : found) volumes.push_back(f.second);
        return volumes;
      }

      // what rank 0 broadcasts about each volume, besides its records' data offsets
      struct VolumeBox {
        coords_t corner;
        coords_t extent;
      };

      /**
         @brief Find the binary records of a partfile volume, and check
         that its site list is a box of the lattice in lexicographic order
         @param[in] name The volume
         @param[in] data_type The type of the binary data records, as in the first volume
         @param[in] n_records The number of binary records in the first volume
         @param[in] first_volume The first volume's name, for messages
         @param[in] site_bytes Size of a site
         @param[in] S The lattice of the file
         @param[out] offsets The data offsets of the volume's binary records are appended
       */
      VolumeBox scan_volume(const std::string &name, const std::string &data_type, size_t n_records,
                            const std::string &first_volume, size_t site_bytes, const coords_t &S,
                            std::vector<uint64_t> &offsets)
      {
        io::File file(name, io::File::Mode::READ);
        const auto records = lime::list_records(file, name);
        const lime::Record *list = nullptr;
        std::vector<const lime::Record *> data_records;
        for (auto &r : records) {
          if (!list && r.type == sitelist) list = &r;
          if (r.type == data_type) data_records.push_back(&r);
        }
        if (!list) errorQuda("The partfile volume %s has no %s record", name.c_str(), sitelist);
        if (data_records.empty()) errorQuda("The partfile volume %s has no %s record", name.c_str(), data_type.c_str());
        if (data_records.size() != n_records)
          errorQuda("The partfile volume %s holds %lu %s records, but %s holds %lu", name.c_str(), data_records.size(),
                    data_type.c_str(), first_volume.c_str(), n_records);
        const auto data = data_records[0];
        for (auto i = 1u; i < data_records.size(); i++)
          if (data_records[i]->data_bytes != data->data_bytes)
            errorQuda("Binary record %u of the partfile volume %s has %lu bytes, but its first has %lu", i,
                      name.c_str(), data_records[i]->data_bytes, data->data_bytes);
        for (auto r : data_records) offsets.push_back(r->data_offset);

        const size_t n = data->data_bytes / site_bytes;
        if (n == 0 || data->data_bytes % site_bytes != 0)
          errorQuda("The binary data of %s is %lu bytes, which is not a whole number of sites of %lu bytes",
                    name.c_str(), data->data_bytes, site_bytes);
        const size_t width = list->data_bytes / n;
        if ((width != 4 && width != 8) || list->data_bytes % n != 0)
          errorQuda(
            "The site list of %s has %lu bytes for the %lu sites of its binary data, rather than 4 or 8 per site",
            name.c_str(), list->data_bytes, n);

        auto entry = [&](const unsigned char *p) {
          uint64_t v = 0;
          for (size_t b = 0; b < width; b++) v = (v << 8) | p[b];
          return v;
        };
        auto not_a_box = [&](const char *reason, uint64_t k, uint64_t found, uint64_t expected) {
          errorQuda("The site list of the partfile volume %s is not a box of the lattice in lexicographic order (%s: "
                    "entry %lu is %ld, expected %ld); only block-decomposed partfiles are supported",
                    name.c_str(), reason, k, static_cast<long>(found), static_cast<long>(expected));
        };

        // the box follows from the first and last sites
        unsigned char ends[2][8];
        file.read(ends[0], width, list->data_offset);
        file.read(ends[1], width, list->data_offset + (n - 1) * width);
        const uint64_t first = entry(ends[0]), last = entry(ends[1]);
        if (last >= lattice_volume(S)) not_a_box("outside the lattice", n - 1, last, lattice_volume(S) - 1);
        if (last < first) not_a_box("not ascending", n - 1, last, first);
        VolumeBox box = {lattice_coords(first, S), {}};
        const auto hi = lattice_coords(last, S);
        size_t box_volume = 1;
        for (int d = 0; d < 4; d++) {
          box.extent[d] = hi[d] - box.corner[d] + 1;
          if (box.extent[d] <= 0) not_a_box("its last site is not the box's upper corner", n - 1, last, first);
          box_volume *= box.extent[d];
        }
        if (box_volume != n)
          errorQuda("The site list of the partfile volume %s spans a box of %lu sites but holds %lu; only "
                    "block-decomposed partfiles are supported",
                    name.c_str(), box_volume, n);

        std::vector<unsigned char> chunk(std::min(sitelist_chunk_bytes, n * width));
        size_t k = 0, pos = 0, filled = 0;
        for_each_box_row({name, 0, box.corner, box.extent}, S, [&](uint64_t row, int length) {
          for (int x = 0; x < length; x++, k++, pos += width) {
            if (pos == filled) {
              filled = std::min(chunk.size(), (n - k) * width);
              file.read(chunk.data(), filled, list->data_offset + k * width);
              pos = 0;
            }
            const uint64_t v = entry(chunk.data() + pos);
            if (v != row + x) not_a_box("not the box's sites in order", k, v, row + x);
          }
        });
        return box;
      }

      /**
         @brief Save fields as a partfile; see write_fields
       */
      std::vector<io::Checksum> write_partfile(const std::string &filename, const std::string &user_file_xml,
                                               const std::vector<FieldMetadata> &metadata,
                                               const std::vector<io::HostField> &fields, const int *X,
                                               QudaParity parity, const FileIOParam &param)
      {
        for (auto &m : metadata)
          if (m.ildg) errorQuda("%s: ILDG files cannot be saved as partfiles", filename.c_str());
        const auto &grid = param.partfile_grid;
        const bool virtual_grid = grid != coords_t {};
        // checks the grid before any volume is truncated
        const io::Partitions partitions(param.io_mode, param.io_partition_grid, !virtual_grid);
        const auto S = file_dims(X, parity);
        std::vector<io::DataLayout> layouts;
        for (auto k = 0u; k < fields.size(); k++)
          layouts.push_back(field_layout(fields[k], metadata[k].prec, 0, parity));

        // each volume holds a box of the lattice, and is created by its writer
        std::vector<io::FilePart> parts;
        std::vector<int> writers;
        if (virtual_grid) {
          for (int d = 0; d < 4; d++)
            if (grid[d] <= 0 || S[d] % grid[d] != 0)
              errorQuda("The partfile grid %d %d %d %d does not divide the lattice %d %d %d %d", grid[0], grid[1],
                        grid[2], grid[3], S[0], S[1], S[2], S[3]);
          coords_t E, b;
          for (int d = 0; d < 4; d++) E[d] = S[d] / grid[d];
          for (b[0] = 0; b[0] < grid[0]; b[0]++)
            for (b[1] = 0; b[1] < grid[1]; b[1]++)
              for (b[2] = 0; b[2] < grid[2]; b[2]++)
                for (b[3] = 0; b[3] < grid[3]; b[3]++) {
                  coords_t corner;
                  for (int d = 0; d < 4; d++) corner[d] = b[d] * E[d];
                  parts.push_back({volume_name(filename, static_cast<int>(parts.size())), 0, corner, E});
                  writers.push_back(0);
                }
        } else {
          coords_t L = {X[0], X[1], X[2], X[3]};
          if (single_parity(parity)) L[0] /= 2;
          for (int leader : partitions.leaders()) {
            io::FilePart p = {volume_name(filename, leader), 0, {}, {}};
            for (int d = 0; d < 4; d++) {
              p.corner[d] = partitions.coords(leader)[d] * L[d];
              p.extent[d] = partitions.extent()[d] * L[d];
            }
            parts.push_back(p);
            writers.push_back(leader);
          }
          if (writers[0] != 0)
            errorQuda("Rank 0 does not lead its I/O partition, so the partfile %s would have no first volume",
                      filename.c_str());
        }
        logQuda(QUDA_VERBOSE, "Saving %s as a partfile of %lu volumes\n", filename.c_str(), parts.size());

        // every volume but the first has a site list and then the records' binary data, so the data offsets are
        // known to all: offsets[k][i] is that of record k in volume i
        auto list_bytes = [&](const io::FilePart &p) { return p.volume() * sitelist_width(p, S); };
        auto data_bytes = [&](const io::FilePart &p, size_t k) { return p.volume() * layouts[k].site_bytes; };
        std::vector<std::vector<uint64_t>> offsets(fields.size(), std::vector<uint64_t>(parts.size()));
        for (auto i = 1u; i < parts.size(); i++) {
          uint64_t next = lime::header_bytes + list_bytes(parts[i]) + lime::padding(list_bytes(parts[i]));
          for (auto k = 0u; k < fields.size(); k++) {
            offsets[k][i] = next + lime::header_bytes;
            next = offsets[k][i] + data_bytes(parts[i], k) + lime::padding(data_bytes(parts[i], k));
          }
        }

        // the first volume holds the metadata, as in QIO, and rank 0 appends each record's metadata to it; each
        // record of the others is a message of its own
        std::optional<io::File> first_file;
        std::optional<lime::Writer> first_writer;
        for (auto i = 0u; i < parts.size(); i++) {
          if (writers[i] != comm_rank()) continue;
          const auto &p = parts[i];
          if (i == 0) {
            first_file.emplace(p.filename, io::File::Mode::CREATE);
            first_writer.emplace(*first_file);
            first_writer->write(private_file_xml, file_info_xml(S, partfile_volfmt), true, false);
            write_sitelist(*first_file, first_writer->reserve(sitelist, list_bytes(p), false, false), p, S);
            first_writer->write(file_xml, user_file_xml, false, true);
            continue;
          }
          io::File file(p.filename, io::File::Mode::CREATE);
          lime::Writer writer(file);
          write_sitelist(file, writer.reserve(sitelist, list_bytes(p), true, true), p, S);
          for (auto k = 0u; k < fields.size(); k++) {
            const auto offset = writer.reserve(binary_data, data_bytes(p, k), true, true);
            if (offset != offsets[k][i])
              errorQuda("Binary record %u of %s starts at byte %lu rather than at byte %lu, where the other ranks "
                        "expect it",
                        k, p.filename.c_str(), offset, offsets[k][i]);
          }
          file.close();
        }

        std::vector<io::Checksum> sums;
        for (auto k = 0u; k < fields.size(); k++) {
          const auto &m = metadata[k];
          uint64_t first_offset = 0;
          if (comm_rank() == 0) {
            first_writer->write(private_record_xml,
                                record_info_xml(m.datatype, m.prec, m.colors, m.spins, m.typesize, m.datacount), true,
                                false);
            first_writer->write(record_xml, m.user_xml, false, false);
            first_offset = first_writer->reserve(binary_data, data_bytes(parts[0], k), false, false);
          }
          comm_broadcast(&first_offset, sizeof(first_offset));
          auto record_parts = parts;
          record_parts[0].offset = first_offset;
          for (auto i = 1u; i < parts.size(); i++) record_parts[i].offset = offsets[k][i];

          auto sum = io::write_data(record_parts, layouts[k], X, partitions, fields[k].packer(m.prec));
          if (comm_rank() == 0) first_writer->write(checksum, checksum_xml(sum.sum29, sum.sum31), false, true);
          sums.push_back(sum);
        }
        if (first_file) first_file->close();
        return sums;
      }
    } // namespace

    std::vector<std::vector<io::FilePart>> find_volumes(const FieldRecords &records, size_t site_bytes)
    {
      const auto &S = records.dims;
      const size_t n_records = records.fields.size();
      std::vector<VolumeBox> boxes;
      std::vector<uint64_t> offsets; // the data offsets of each volume's records, one volume after another
      std::string names;             // each followed by a newline
      if (comm_rank() == 0) {
        const auto volumes = list_volumes(records.filename);
        for (auto &v : volumes) {
          boxes.push_back(scan_volume(v, records.fields[0].data.type, n_records, records.file, site_bytes, S, offsets));
          names += v + "\n";
        }

        uint64_t sites = 0;
        for (auto i = 0u; i < boxes.size(); i++) {
          sites
            += static_cast<uint64_t>(boxes[i].extent[0]) * boxes[i].extent[1] * boxes[i].extent[2] * boxes[i].extent[3];
          for (auto j = 0u; j < i; j++) {
            bool overlap = true;
            for (int d = 0; d < 4; d++)
              overlap = overlap
                && std::max(boxes[i].corner[d], boxes[j].corner[d])
                  < std::min(boxes[i].corner[d] + boxes[i].extent[d], boxes[j].corner[d] + boxes[j].extent[d]);
            if (overlap)
              errorQuda("The partfile volumes %s and %s hold some of the same sites: one may be left over from an "
                        "earlier save with another rank layout or I/O partition grid",
                        volumes[j].c_str(), volumes[i].c_str());
          }
        }
        if (sites != lattice_volume(S))
          errorQuda("The %lu volumes of the partfile %s hold %lu of the %lu sites of the lattice %d %d %d %d: a volume "
                    "is missing",
                    volumes.size(), records.filename.c_str(), sites, lattice_volume(S), S[0], S[1], S[2], S[3]);
      }
      io::broadcast(boxes);
      io::broadcast(offsets);
      io::broadcast(names);

      std::vector<std::vector<io::FilePart>> parts(n_records);
      std::istringstream in(names);
      for (auto i = 0u; i < boxes.size(); i++) {
        std::string name;
        std::getline(in, name);
        for (auto k = 0u; k < n_records; k++)
          parts[k].push_back({name, offsets[i * n_records + k], boxes[i].corner, boxes[i].extent});
      }
      logQuda(QUDA_VERBOSE, "Found %lu volumes of the partfile %s, with %lu binary records each\n", boxes.size(),
              records.filename.c_str(), n_records);
      return parts;
    }

    std::vector<FieldChecksum> read_fields(const FieldRecords &records, const std::vector<io::HostField> &fields,
                                           const int *X, QudaParity parity, const FileIOParam &param)
    {
      const auto &file = records.file;
      if (fields.size() > records.fields.size())
        errorQuda("%s holds %lu fields, fewer than the %lu to be read", file.c_str(), records.fields.size(),
                  fields.size());
      const auto N = file_dims(X, parity);
      const auto &D = records.dims;
      if (D != N)
        errorQuda("%s has dimensions %d %d %d %d, but the field's lattice is %d %d %d %d%s", file.c_str(), D[0], D[1],
                  D[2], D[3], N[0], N[1], N[2], N[3],
                  single_parity(parity) ? " (half the global x extent, as single-parity fields are stored)" : "");

      auto what = [&](size_t k) {
        const auto index = records.fields[k].index;
        return records.fields.size() > 1 || index > 0 ? "binary record " + std::to_string(index) + " of " + file : file;
      };
      std::vector<io::DataLayout> layouts;
      for (auto k = 0u; k < fields.size(); k++) {
        const auto &r = records.fields[k];
        const auto &m = r.metadata;
        if (m.typesize != static_cast<int>(fields[k].item_bytes(m.prec)))
          errorQuda("The typesize %d of %s does not match the expected datasize %lu", m.typesize, what(k).c_str(),
                    fields[k].item_bytes(m.prec));
        layouts.push_back(field_layout(fields[k], m.prec, r.data.data_offset, parity));
      }

      const io::Partitions partitions(param.io_mode, param.io_partition_grid);
      const bool partfile = records.volfmt == partfile_volfmt;
      for (auto k = 0u; partfile && k < fields.size(); k++)
        if (records.fields[k].metadata.hypercube)
          errorQuda("%s is a hypercube record in a partfile, which is not supported", what(k).c_str());
      std::vector<std::vector<io::FilePart>> volumes;
      if (partfile && !fields.empty()) {
        for (auto k = 1u; k < layouts.size(); k++)
          if (layouts[k].site_bytes != layouts[0].site_bytes)
            errorQuda("The sites of %s have %lu bytes, but those of its first binary record %lu", what(k).c_str(),
                      layouts[k].site_bytes, layouts[0].site_bytes);
        volumes = find_volumes(records, layouts[0].site_bytes);
      }

      std::vector<FieldChecksum> sums;
      for (auto k = 0u; k < fields.size(); k++) {
        const auto &r = records.fields[k];
        const auto unpack = fields[k].unpacker(r.metadata.prec);
        FieldChecksum c;
        if (partfile) {
          c.sum = io::read_data(volumes[k], layouts[k], X, partitions, unpack);
        } else {
          const auto &box = r.metadata.hypercube;
          const auto bytes = box ? box->volume() * layouts[k].site_bytes : io::data_bytes(layouts[k], io::global_dims(X));
          if (r.data.data_bytes != bytes)
            errorQuda("The binary data of %s is %lu bytes, but should be %lu bytes%s", what(k).c_str(),
                      r.data.data_bytes, bytes, box ? (" for the sites " + hypercube_str(*box)).c_str() : "");
          c.sum = io::read_data(file, layouts[k], X, partitions, unpack, box ? &*box : nullptr);
        }

        // as in QIO, a SciDAC file must have a checksum, while ILDG files from other codes have none
        if (records.native) {
          if (!r.checksum) errorQuda("%s has no %s record after its binary data", what(k).c_str(), checksum);
          c.has_checksum = true;
          c.checksum_matches = io::check_checksum(what(k), c.sum, *r.checksum, param.allow_checksum_mismatch);
        }
        sums.push_back(c);
      }
      return sums;
    }

    FieldChecksum read_field(const FieldRecords &records, const io::HostField &field, const int *X, QudaParity parity,
                             const FileIOParam &param)
    {
      return read_fields(records, {field}, X, parity, param)[0];
    }

    std::vector<io::Checksum> write_fields(const std::string &filename, const std::string &user_file_xml,
                                           const std::vector<FieldMetadata> &metadata,
                                           const std::vector<io::HostField> &fields, const int *X, QudaParity parity,
                                           const FileIOParam &param)
    {
      if (metadata.size() != fields.size())
        errorQuda("%lu metadata records for %lu fields in %s", metadata.size(), fields.size(), filename.c_str());

      // as QIO, a partfile on one rank is a single file
      if (param.partfile && (comm_size() > 1 || param.partfile_grid != std::array<int, 4> {})) {
        for (auto &m : metadata)
          if (m.hypercube) errorQuda("%s: hypercube records cannot be saved in partfiles", filename.c_str());
        return write_partfile(filename, user_file_xml, metadata, fields, X, parity, param);
      }

      // checks the grid before rank 0 truncates the file
      const io::Partitions partitions(param.io_mode, param.io_partition_grid);
      const auto N = file_dims(X, parity);

      // rank 0 writes everything but the binary data, appending each record's checksum after its data
      std::optional<io::File> file;
      std::optional<lime::Writer> writer;
      if (comm_rank() == 0) {
        file.emplace(filename, io::File::Mode::CREATE);
        writer.emplace(*file);
        writer->write(private_file_xml, file_info_xml(N, 0), true, false);
        writer->write(file_xml, user_file_xml, false, true);
      }

      std::vector<io::Checksum> sums;
      for (auto k = 0u; k < fields.size(); k++) {
        const auto &m = metadata[k];
        auto layout = field_layout(fields[k], m.prec, 0, parity);
        const auto box = m.hypercube ? &*m.hypercube : nullptr;
        const size_t bytes = box ? box->volume() * layout.site_bytes : io::data_bytes(layout, io::global_dims(X));
        uint64_t data_offset = 0;
        if (comm_rank() == 0) {
          writer->write(private_record_xml,
                        record_info_xml(m.datatype, m.prec, m.colors, m.spins, m.typesize, m.datacount, m.hypercube),
                        true, false);
          writer->write(record_xml, m.user_xml, false, false);
          if (m.ildg) {
            writer->write(ildg_format, ildg_format_xml(m.prec, N), false, false);
            if (!m.ildg_lfn.empty()) writer->write(ildg_data_lfn, m.ildg_lfn, false, false);
          }
          data_offset = writer->reserve(m.ildg ? ildg_binary_data : binary_data, bytes, false, false);
        }
        comm_broadcast(&data_offset, sizeof(data_offset));
        layout.offset = data_offset;

        auto sum = io::write_data(filename, layout, X, partitions, fields[k].packer(m.prec), box);
        if (comm_rank() == 0) writer->write(checksum, checksum_xml(sum.sum29, sum.sum31), false, true);
        sums.push_back(sum);
      }
      if (file) file->close();
      return sums;
    }

  } // namespace scidac

} // namespace quda
