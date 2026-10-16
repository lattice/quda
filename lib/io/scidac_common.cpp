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
      f.filename = filename;
      int exists[2] = {0, 0}; // <filename>, and the first volume of a partfile of that name
      if (comm_rank() == 0) {
        exists[0] = access(filename.c_str(), F_OK) == 0;
        exists[1] = access(volume_name(filename, 0).c_str(), F_OK) == 0;
      }
      comm_broadcast(exists, sizeof(exists));
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

      std::string file_info;
      if (f.native) {
        file_info = lime::read_string(file, records[0]);
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

      // what rank 0 broadcasts about each volume
      struct VolumeBox {
        uint64_t offset;
        coords_t corner;
        coords_t extent;
      };

      /**
         @brief Find the binary data of a partfile volume, and check that
         its site list is a box of the lattice in lexicographic order
         @param[in] name The volume
         @param[in] data_type The type of the binary data record, as in the first volume
         @param[in] site_bytes Size of a site
         @param[in] S The lattice of the file
       */
      VolumeBox scan_volume(const std::string &name, const std::string &data_type, size_t site_bytes, const coords_t &S)
      {
        io::File file(name, io::File::Mode::READ);
        const auto records = lime::list_records(file, name);
        const lime::Record *list = nullptr, *data = nullptr;
        for (auto &r : records) {
          if (!list && r.type == sitelist) list = &r;
          if (!data && r.type == data_type) data = &r;
        }
        if (!list) errorQuda("The partfile volume %s has no %s record", name.c_str(), sitelist);
        if (!data) errorQuda("The partfile volume %s has no %s record", name.c_str(), data_type.c_str());

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
        VolumeBox box = {data->data_offset, lattice_coords(first, S), {}};
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
         @brief Save a field as a partfile; see write_field
       */
      io::Checksum write_partfile(const std::string &filename, const FieldMetadata &m, const io::HostField &field,
                                  const int *X, QudaParity parity, const FileIOParam &param)
      {
        if (m.ildg) errorQuda("%s: ILDG files cannot be saved as partfiles", filename.c_str());
        const auto &grid = param.partfile_grid;
        const bool virtual_grid = grid != coords_t {};
        // checks the grid before any volume is truncated
        const io::Partitions partitions(param.io_mode, param.io_partition_grid, !virtual_grid);
        const auto S = file_dims(X, parity);
        const auto layout = field_layout(field, m.prec, 0, parity);

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

        // every volume but the first has a site list and the binary data, so its data offset is known to all
        auto list_bytes = [&](const io::FilePart &p) { return p.volume() * sitelist_width(p, S); };
        auto data_bytes = [&](const io::FilePart &p) { return p.volume() * layout.site_bytes; };
        for (auto i = 1u; i < parts.size(); i++)
          parts[i].offset = 2 * lime::header_bytes + list_bytes(parts[i]) + lime::padding(list_bytes(parts[i]));

        // the first volume holds the metadata, as in QIO; each record of the others is a message of its own
        for (auto i = 0u; i < parts.size(); i++) {
          if (writers[i] != comm_rank()) continue;
          auto &p = parts[i];
          const bool first = i == 0;
          io::File file(p.filename, io::File::Mode::CREATE);
          lime::Writer writer(file);
          if (first) writer.write(private_file_xml, file_info_xml(S, partfile_volfmt), true, false);
          write_sitelist(file, writer.reserve(sitelist, list_bytes(p), !first, !first), p, S);
          if (first) {
            writer.write(file_xml, std::string("Dummy user file XML"), false, true);
            writer.write(private_record_xml,
                         record_info_xml(m.datatype, m.prec, m.colors, m.spins, m.typesize, m.datacount), true, false);
            writer.write(record_xml, m.user_xml, false, false);
          }
          const auto offset = writer.reserve(binary_data, data_bytes(p), !first, !first);
          if (!first && offset != p.offset)
            errorQuda(
              "The binary data of %s starts at byte %lu rather than at byte %lu, where the other ranks expect it",
              p.filename.c_str(), offset, p.offset);
          p.offset = offset;
          file.close();
        }
        uint64_t first_offset = parts[0].offset;
        comm_broadcast(&first_offset, sizeof(first_offset));
        parts[0].offset = first_offset;

        auto sum = io::write_data(parts, layout, X, partitions, field.packer(m.prec));

        if (comm_rank() == 0) {
          const auto &p = parts[0];
          io::File file(p.filename, io::File::Mode::WRITE);
          lime::Writer writer(file, p.offset + data_bytes(p) + lime::padding(data_bytes(p)));
          writer.write(checksum, checksum_xml(sum.sum29, sum.sum31), false, true);
          file.close();
        }
        return sum;
      }
    } // namespace

    std::vector<io::FilePart> find_volumes(const FieldRecords &records, size_t site_bytes)
    {
      const auto &S = records.dims;
      std::vector<VolumeBox> boxes;
      std::string names; // each followed by a newline
      if (comm_rank() == 0) {
        const auto volumes = list_volumes(records.filename);
        for (auto &v : volumes) {
          boxes.push_back(scan_volume(v, records.data.type, site_bytes, S));
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
      io::broadcast(names);

      std::vector<io::FilePart> parts;
      std::istringstream in(names);
      for (auto &b : boxes) {
        io::FilePart p = {"", b.offset, b.corner, b.extent};
        std::getline(in, p.filename);
        parts.push_back(p);
      }
      logQuda(QUDA_VERBOSE, "Found %lu volumes of the partfile %s\n", parts.size(), records.filename.c_str());
      return parts;
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
      const io::Partitions partitions(param.io_mode, param.io_partition_grid);
      FieldChecksum c;
      if (records.volfmt == partfile_volfmt) {
        auto parts = find_volumes(records, layout.site_bytes);
        c.sum = io::read_data(parts, layout, X, partitions, field.unpacker(m.prec));
      } else {
        const auto bytes = io::data_bytes(layout, io::global_dims(X));
        if (records.data.data_bytes != bytes)
          errorQuda("The binary data of %s is %lu bytes, but should be %lu bytes", file.c_str(),
                    records.data.data_bytes, bytes);
        c.sum = io::read_data(file, layout, X, partitions, field.unpacker(m.prec));
      }

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
      // as QIO, a partfile on one rank is a single file
      if (param.partfile && (comm_size() > 1 || param.partfile_grid != std::array<int, 4> {}))
        return write_partfile(filename, metadata, field, X, parity, param);

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
