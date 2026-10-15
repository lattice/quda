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

    GaugeInfo load_gauge(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                         const FileIOParam &param)
    {
      io::HostField field(gauge, n_dim, prec, matrix_reals);
      auto records = find_field(filename);
      const auto &m = records.metadata;
      if (m.datacount != n_dim)
        errorQuda("The datacount %d of %s does not match the expected number of fields %d", m.datacount,
                  records.file.c_str(), n_dim);

      GaugeInfo info;
      info.ildg = m.ildg;
      info.file_prec = m.prec;
      if (records.native) read_usqcd_info(m.user_xml, info);

      // the four links of a site are row-major 3x3 complex matrices
      auto sum = read_field(records, field, X, QUDA_INVALID_PARITY, param);
      info.sum = sum.sum;
      info.has_checksum = sum.has_checksum;
      info.checksum_matches = sum.checksum_matches;
      return info;
    }

    io::Checksum save_gauge(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                            const FileIOParam &param, const GaugeInfo &info)
    {
      io::HostField field(gauge, n_dim, prec, matrix_reals);
      FieldMetadata m;
      m.datatype = gauge_datatype(prec, info.ildg);
      m.prec = prec;
      m.colors = 3;
      m.typesize = matrix_reals * prec;
      m.datacount = n_dim;
      m.user_xml = info.has_usqcd_info ?
        usqcd_info_xml(info.plaq, info.linktr) :
        quda_record_xml(matrix_reals, m.datatype, QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY, 3, 0);
      m.ildg = info.ildg;
      m.ildg_lfn = info.ildg_lfn;
      return write_field(filename, m, field, X, QUDA_INVALID_PARITY, param);
    }

  } // namespace scidac

} // namespace quda
