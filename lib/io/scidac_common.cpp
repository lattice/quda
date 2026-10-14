#include <cctype>
#include <cstdio>
#include <ctime>
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

  } // namespace scidac

} // namespace quda
