#include <cstdlib>
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

    void set_gauge_record_xml(GaugeMetadata &metadata, const std::string &user_xml)
    {
      metadata.user_record_xml = user_xml;
      auto usqcd = xml_value(user_xml, "usqcdInfo");
      metadata.has_usqcd_info = usqcd.has_value();
      metadata.usqcd_info = usqcd ? xml_value(*usqcd, "info") : std::nullopt;
    }

    GaugeInfo load_gauge(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                         const FileIOParam &param)
    {
      io::HostField field(gauge, n_dim, prec, matrix_reals);
      auto records = find_field(filename);
      check_no_hypercube(records, 1);
      const auto &m = records.fields[0].metadata;
      if (m.datacount != n_dim)
        errorQuda("The datacount %d of %s does not match the expected number of fields %d", m.datacount,
                  records.file.c_str(), n_dim);

      GaugeInfo info;
      info.ildg = m.ildg;
      info.file_prec = m.prec;
      if (records.native) read_usqcd_info(m.user_xml, info);
      info.metadata.user_file_xml = records.user_file_xml;
      set_gauge_record_xml(info.metadata, m.user_xml);
      info.metadata.ildg_lfn = m.ildg_lfn;
      info.metadata.record_date = m.date;

      // the four links of a site are row-major 3x3 complex matrices
      auto sum = read_field(records, field, X, QUDA_INVALID_PARITY, param);
      info.sum = sum.sum;
      info.has_checksum = sum.has_checksum;
      info.checksum_matches = sum.checksum_matches;
      return info;
    }

    GaugeRecords gauge_records(QudaPrecision file_prec, const GaugeInfo &info)
    {
      GaugeRecords g;
      if (!info.user_file_xml.empty())
        g.user_file_xml = info.user_file_xml;
      else
        g.user_file_xml = info.legacy_strings ? legacy_file_xml : quda_file_xml("gauge field");
      auto &m = g.record;
      const std::string p = file_prec == QUDA_DOUBLE_PRECISION ? "D" : "F";
      // QIO only writes the ILDG records for the USQCD (or QDP) color matrix datatypes
      m.datatype = info.legacy_strings && !info.ildg ? "QUDA_" + p + "Nc3_GaugeField" : "USQCD_" + p + "3_ColorMatrix";
      m.prec = file_prec;
      m.colors = 3;
      m.typesize = matrix_reals * file_prec;
      m.datacount = n_dim;
      m.date = info.date;
      if (info.user_record_xml)
        m.user_xml = *info.user_record_xml;
      else if (info.has_usqcd_info)
        m.user_xml = usqcd_info_xml(info.plaq, info.linktr, info.usqcd_info);
      else if (info.legacy_strings)
        m.user_xml = legacy_quda_record_xml(matrix_reals, m.datatype, QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, 3, 0);
      else
        m.user_xml = quda_record_xml(matrix_reals, QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, 3, 0);
      m.ildg = info.ildg;
      if (!info.ildg_lfn.empty()) m.ildg_lfn = info.ildg_lfn;
      return g;
    }

    io::Checksum save_gauge(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                            const FileIOParam &param, const GaugeInfo &info)
    {
      io::HostField field(gauge, n_dim, prec, matrix_reals);
      const auto g = gauge_records(prec, info);
      return write_fields(filename, g.user_file_xml, {g.record}, {field}, X, QUDA_INVALID_PARITY, param)[0];
    }

  } // namespace scidac

} // namespace quda
