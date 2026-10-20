#include <cstdlib>
#include <util_quda.h>
#include <io/scidac_io.h>

namespace quda
{

  namespace scidac
  {

    namespace
    {
      // the file types of USQCD KS propagator files (QIO's qioxml_usqcd_ksprop.h)
      constexpr const char *vv_pairs = "USQCD_ColorVector_Source_Sink_Pairs";
      constexpr const char *c1v3 = "USQCD_ColorVector_ScalarSource_ThreeSink";
      constexpr const char *cv_pairs = "USQCD_ColorVector_ScalarSource_Sink_Pairs";

      // the elements of USQCD KS propagator files: the file XML, a source record's and a solution record's XML
      constexpr const char *ksprop_file_tag = "usqcdKSPropFile";
      constexpr const char *source_tag = "usqcdSourceInfo";
      constexpr const char *solution_tag = "usqcdKSPropInfo";

      // the reals of a site of a 3-color vector and of a complex field
      constexpr int color_vector_reals = 6;
      constexpr int complex_reals = 2;

      std::string complex_datatype(QudaPrecision prec)
      {
        return std::string("QLA_") + (prec == QUDA_DOUBLE_PRECISION ? "D" : "F") + "_Complex";
      }

      bool is_complex(const std::string &datatype)
      {
        return datatype == complex_datatype(QUDA_SINGLE_PRECISION) || datatype == complex_datatype(QUDA_DOUBLE_PRECISION);
      }

      // the content of a record XML's source element: usqcdSourceInfo, or the older usqcdKSPropInfo, as MILC reads them
      std::optional<std::string> source_content(const std::string &xml)
      {
        auto content = xml_value(xml, source_tag);
        return content ? content : xml_value(xml, solution_tag);
      }

      const char *prec_str(QudaPrecision prec) { return prec == QUDA_DOUBLE_PRECISION ? "D" : "F"; }

      void check_color_vector(const FieldMetadata &m, const std::string &what)
      {
        const auto datatype = color_vector_datatype(m.prec);
        if (m.datatype != datatype)
          errorQuda("%s has the datatype %s, but a color-vector record of precision %s has %s", what.c_str(),
                    m.datatype.c_str(), prec_str(m.prec), datatype.c_str());
        if (m.colors != 3) errorQuda("%s has %d colors, but a color-vector record has 3", what.c_str(), m.colors);
        if (m.datacount != 1)
          errorQuda("%s has datacount %d, but a color-vector record has 1", what.c_str(), m.datacount);
        if (m.typesize != color_vector_reals * m.prec)
          errorQuda("%s has typesize %d, but a color-vector record of precision %s has %d", what.c_str(), m.typesize,
                    prec_str(m.prec), color_vector_reals * m.prec);
      }

      // the content of an element that MILC's decoders require, or an error
      std::string required(const std::string &xml, const char *tag, const std::string &what)
      {
        auto value = xml_value(xml, tag);
        if (!value) errorQuda("%s has no <%s>, which MILC requires", what.c_str(), tag);
        return *value;
      }

      std::string record_str(size_t k, const std::string &file)
      {
        return "Binary record " + std::to_string(k) + " of " + file;
      }

      void check_propagator(KSFile &f)
      {
        const auto &records = f.records;
        const auto &file = records.file;
        const auto content = *xml_value(records.user_file_xml, ksprop_file_tag);
        const std::string what = "The file XML of the USQCD KS propagator file " + file;
        required(content, "version", what);
        const auto type = required(content, "type", what);
        f.file_info = required(content, "info", what);
        if (type == c1v3 || type == cv_pairs)
          errorQuda("%s is a USQCD KS propagator file of type %s, which is not supported; only %s (VV pairs) is",
                    file.c_str(), type.c_str(), vv_pairs);
        if (type != vv_pairs)
          errorQuda("%s is a USQCD KS propagator file of the unknown type %s; only %s (VV pairs) is supported",
                    file.c_str(), type.c_str(), vv_pairs);

        const auto n = records.fields.size();
        if (n % 2 != 0)
          errorQuda("%s holds %lu binary records, but a file of source/solution pairs holds an even number",
                    file.c_str(), n);
        for (auto k = 0u; k < n; k++) {
          const auto &m = records.fields[k].metadata;
          const bool source = k % 2 == 0;
          const std::string what = record_str(k, file) + ", the " + (source ? "source" : "solution") + " of pair "
            + std::to_string(k / 2) + ",";
          check_color_vector(m, what);
          auto xml = xml_value(m.user_xml, source ? source_tag : solution_tag);
          if (!xml)
            errorQuda("%s has no %s record XML: %s", what.c_str(), source ? source_tag : solution_tag,
                      m.user_xml.c_str());
          required(*xml, "version", what);
          f.info[k] = required(*xml, "info", what);
          if (source) continue;
          if (m.hypercube)
            errorQuda("%s is a hypercube record (sites %s), but solutions hold the whole lattice", what.c_str(),
                      hypercube_str(*m.hypercube).c_str());
          const auto color = required(*xml, "color", what);
          if (std::strtol(color.c_str(), nullptr, 10) != static_cast<long>(k / 2))
            errorQuda("%s has <color>%s</color>, but the solution of pair %u must have color %u, as in MILC",
                      what.c_str(), color.c_str(), k / 2, k / 2);
        }
      }

      void check_vector_source(KSFile &f)
      {
        const auto &records = f.records;
        f.file_info = records.user_file_xml;
        for (auto k = 0u; k < records.fields.size(); k++) {
          const auto &m = records.fields[k].metadata;
          const auto what = record_str(k, records.file);
          check_color_vector(m, what);
          const auto content = *source_content(m.user_xml);
          f.info[k] = xml_value(content, "info").value_or("");
          // as MILC's check_color, an absent color counts as 0
          const long color = std::strtol(xml_value(content, "color").value_or("0").c_str(), nullptr, 10);
          if (color != static_cast<long>(k))
            errorQuda("%s has color %ld, but record %u of a staggered source file must have color %u, as in MILC (a "
                      "record without <color> counts as color 0)",
                      what.c_str(), color, k, k);
        }
      }

      void check_complex_source(KSFile &f)
      {
        const auto &records = f.records;
        f.file_info = records.user_file_xml;
        const auto &m = records.fields[0].metadata;
        const auto what = "The complex-field record of " + records.file;
        // as MILC's read_complex_scidac_xml, which checks only the typesize and the datacount
        if (m.datatype != complex_datatype(m.prec))
          errorQuda("%s has the datatype %s, but its precision is %s", what.c_str(), m.datatype.c_str(),
                    prec_str(m.prec));
        if (m.typesize != complex_reals * m.prec)
          errorQuda("%s has typesize %d, but %s has %d", what.c_str(), m.typesize, m.datatype.c_str(),
                    complex_reals * m.prec);
        if (m.datacount != 1) errorQuda("%s has datacount %d, but a complex field has 1", what.c_str(), m.datacount);
        f.info[0] = xml_value(m.user_xml, "info").value_or("");
        if (records.fields.size() > 1)
          warningQuda("%s holds %lu binary records; only the first, a complex field, is loaded", records.file.c_str(),
                      records.fields.size());
      }
    } // namespace

    std::optional<KSFileType> ks_file_type(const FieldRecords &records)
    {
      if (!records.native) return std::nullopt;
      const auto &file_xml = records.user_file_xml;
      if (xml_value(file_xml, ksprop_file_tag)) return KSFileType::Propagator;
      if (xml_value(file_xml, "usqcdPropFile"))
        errorQuda("%s is a USQCD Dirac propagator file (usqcdPropFile), which is not implemented", records.file.c_str());
      if (is_complex(records.fields[0].metadata.datatype)) return KSFileType::ComplexSource;
      // the record XML tells MILC's source files from vector files, whose datatype they share; find_ks_file then
      // checks the datatypes
      for (auto &r : records.fields)
        if (!source_content(r.metadata.user_xml)) return std::nullopt;
      // MILC writes Dirac sources (save_*_scidac_w_source) with the same record XML
      for (auto &r : records.fields) {
        const auto &datatype = r.metadata.datatype;
        if (datatype == "USQCD_F3_DiracFermion" || datatype == "USQCD_D3_DiracFermion")
          errorQuda("%s is a MILC Dirac source file (binary record %lu is a %s source), which is not supported; "
                    "staggered source files hold USQCD_[FD]3_ColorVector records",
                    records.file.c_str(), r.index, datatype.c_str());
      }
      return KSFileType::VectorSource;
    }

    KSFile find_ks_file(const std::string &filename)
    {
      if (partfile_named(filename))
        errorQuda("%s is a partfile (%s is its first volume): staggered source and propagator partfiles are not "
                  "supported",
                  filename.c_str(), volume_name(filename, 0).c_str());
      KSFile f;
      f.records = find_field(filename);
      const auto type = ks_file_type(f.records);
      if (!type)
        errorQuda("%s is not a staggered source or KS propagator file (a USQCD KS propagator file, or one of MILC's "
                  "color-vector or complex-field source files); QUDA's and MILC's eigenvector files load through "
                  "VectorIO",
                  f.records.file.c_str());
      f.type = *type;
      f.info.resize(f.records.fields.size());
      switch (f.type) {
      case KSFileType::Propagator: check_propagator(f); break;
      case KSFileType::VectorSource: check_vector_source(f); break;
      case KSFileType::ComplexSource: check_complex_source(f); break;
      }
      return f;
    }

  } // namespace scidac

} // namespace quda
