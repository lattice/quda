#include <algorithm>
#include <array>
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
      /**
         @brief The local dimensions of the full field that holds a set of vectors
       */
      std::array<int, 4> full_dims(const int *X, QudaSiteSubset subset)
      {
        std::array<int, 4> full = {X[0], X[1], X[2], X[3]};
        if (subset == QUDA_PARITY_SITE_SUBSET) full[0] *= 2;
        return full;
      }

      /**
         @brief The parity of the sites a file holds: either for single-parity vectors, whose files have half the x
         extent and the same layout whichever parity the vectors have, or none for full vectors
       */
      QudaParity file_parity(QudaSiteSubset subset, QudaParity parity)
      {
        if (subset != QUDA_PARITY_SITE_SUBSET) return QUDA_INVALID_PARITY;
        return parity == QUDA_ODD_PARITY ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;
      }

      constexpr const char *milc_eigen_title = "KS eigenvalues and vectors";

      // the reals of a 3-color vector at a site
      constexpr int milc_vector_reals = 6;
    } // namespace

    VectorFile find_vectors(const std::string &filename)
    {
      VectorFile v;
      v.records = find_field(filename);
      const auto &records = v.records;
      const auto &file = records.file;
      if (!records.native)
        errorQuda("%s is an ILDG file without the SciDAC records, so it holds no vectors", file.c_str());

      const auto &file_xml = records.user_file_xml;
      if (xml_value(file_xml, "usqcdKSPropFile"))
        errorQuda("%s is a USQCD KS propagator file, which cannot be loaded yet", file.c_str());

      // as MILC's parse_file_xml_Nvec, which reads a file without a positive <Nvecs> as QUDA's
      const long n_vecs = std::strtol(xml_value(file_xml, "Nvecs").value_or("0").c_str(), nullptr, 10);
      if (xml_value(file_xml, "title").value_or("") == milc_eigen_title && n_vecs > 0) {
        v.strings = VectorStrings::MILC;
        if (xml_value(file_xml, "format").value_or("") == "Packed")
          errorQuda("%s holds MILC's packed eigenvectors, which are not supported", file.c_str());
        for (auto k = 0u; k < records.fields.size(); k++) {
          const auto &m = records.fields[k].metadata;
          const auto datatype = color_vector_datatype(m.prec);
          const char *prec = m.prec == QUDA_DOUBLE_PRECISION ? "D" : "F";
          if (m.datatype != datatype)
            errorQuda("Record %u of the MILC eigenvector file %s has the datatype %s, but MILC's eigenvectors of "
                      "precision %s have %s",
                      k, file.c_str(), m.datatype.c_str(), prec, datatype.c_str());
          if (m.datacount != 1)
            errorQuda("Record %u of the MILC eigenvector file %s has datacount %d, but MILC's eigenvectors have 1", k,
                      file.c_str(), m.datacount);
          if (m.colors != 3)
            errorQuda("Record %u of the MILC eigenvector file %s has %d colors, but MILC's eigenvectors have 3", k,
                      file.c_str(), m.colors);
          if (m.typesize != milc_vector_reals * m.prec)
            errorQuda("Record %u of the MILC eigenvector file %s has typesize %d, but MILC's eigenvectors of "
                      "precision %s have %d",
                      k, file.c_str(), m.typesize, prec, milc_vector_reals * m.prec);
        }
        if (n_vecs != static_cast<long>(records.fields.size()))
          warningQuda("The MILC eigenvector file %s gives <Nvecs> %ld, but holds %lu eigenvectors, which count, as in "
                      "MILC",
                      file.c_str(), n_vecs, records.fields.size());
      }

      for (auto &r : records.fields) {
        auto eval = xml_value(r.metadata.user_xml, "eigVal");
        if (!eval) {
          v.evals.clear();
          break;
        }
        auto imag = xml_value(r.metadata.user_xml, "eigValImag");
        v.evals.emplace_back(std::strtod(eval->c_str(), nullptr), imag ? std::strtod(imag->c_str(), nullptr) : 0.0);
      }
      return v;
    }

    std::vector<io::HostField> vector_fields(const VectorFile &file, const io::HostField &field, int n_vectors)
    {
      const auto &records = file.records.fields;
      const auto &name = file.records.file;
      const int Ls = field.items() / n_vectors;
      const int datacount = records[0].metadata.datacount;

      if (records.size() == 1 && datacount > Ls) {
        if (datacount % Ls != 0)
          errorQuda("The datacount %d of %s is not a whole number of vectors of %d items", datacount, name.c_str(), Ls);
        if (n_vectors > datacount / Ls)
          errorQuda("%s holds %d vectors in one record, fewer than the %d to be loaded", name.c_str(), datacount / Ls,
                    n_vectors);
        return {field.extend(datacount)};
      }

      if (n_vectors > static_cast<int>(records.size()))
        errorQuda("%s holds %lu vectors, fewer than the %d to be loaded", name.c_str(), records.size(), n_vectors);
      std::vector<io::HostField> fields;
      for (int i = 0; i < n_vectors; i++) {
        if (records[i].metadata.datacount != Ls)
          errorQuda("Binary record %d of %s has datacount %d, but holds one vector of %d items, as every record of a "
                    "file of several",
                    i, name.c_str(), records[i].metadata.datacount, Ls);
        fields.push_back(field.slice(i * Ls, Ls));
      }
      return fields;
    }

    std::vector<FieldChecksum> load_vectors(const VectorFile &file, const std::vector<io::HostField> &fields,
                                            const int *X, QudaSiteSubset subset, QudaParity parity, int nColor,
                                            int nSpin, const FileIOParam &param)
    {
      const auto &records = file.records;
      // MILC's records have no spins, and were checked to have 3 colors
      if (file.strings == VectorStrings::QUDA) {
        for (auto k = 0u; k < fields.size(); k++) {
          const auto &m = records.fields[k].metadata;
          if (spins_match(m.datatype, m.spins, nSpin) && m.colors == nColor) continue;
          warningQuda("The file %s has %d spins and %d colors, but the vectors have %d and %d", records.file.c_str(),
                      m.spins, m.colors, nSpin, nColor);
          break;
        }
      }

      const auto X_full = full_dims(X, subset);
      return read_fields(records, fields, X_full.data(), file_parity(subset, parity), param);
    }

    VectorRecords vector_records(QudaPrecision prec, int n_vectors, int Ls, QudaSiteSubset subset, QudaParity parity,
                                 int nColor, int nSpin, const VectorSaveInfo &info)
    {
      auto value = [](const std::vector<double> &x, int i) {
        return x.empty() ? std::optional<double>() : std::optional<double>(x[i]);
      };
      VectorRecords v;
      if (info.strings == VectorStrings::MILC) {
        if (info.interleaved) errorQuda("MILC's eigenvector format has one record per vector");
        if (nSpin != 1 || nColor != 3 || Ls != 1)
          errorQuda("MILC's eigenvector format holds 4-d vectors of 1 spin and 3 colors, not of %d spins, %d colors "
                    "and %d slices",
                    nSpin, nColor, Ls);
        if (info.evals.empty()) errorQuda("MILC's eigenvector format needs the eigenvalues");
        v.user_file_xml = milc_eigen_file_xml(n_vectors);
        for (int i = 0; i < n_vectors; i++) {
          FieldMetadata m;
          m.datatype = color_vector_datatype(prec);
          m.prec = prec;
          m.colors = 3;
          m.typesize = milc_vector_reals * prec;
          m.datacount = 1;
          m.user_xml = milc_eigen_record_xml(info.evals[i], value(info.resids, i).value_or(0.0));
          v.records.push_back(m);
        }
        return v;
      }

      const bool legacy = info.legacy_strings || info.interleaved;
      const int reals = 2 * nSpin * nColor;
      v.user_file_xml = legacy ? legacy_file_xml :
                                 quda_file_xml(n_vectors == 1 ? "1 vector" : std::to_string(n_vectors) + " vectors");
      FieldMetadata m;
      m.datatype = vector_datatype(prec, nSpin, nColor, legacy);
      m.prec = prec;
      m.colors = nColor;
      m.spins = m.datatype == color_vector_datatype(prec) ? 0 : nSpin;
      m.typesize = reals * prec;
      if (info.interleaved) {
        m.datacount = n_vectors * Ls;
        m.user_xml = legacy_quda_record_xml(reals, m.datatype, subset, parity, nColor, nSpin);
        v.records.push_back(m);
        return v;
      }
      m.datacount = Ls;
      const bool complex_evals
        = std::any_of(info.evals_imag.begin(), info.evals_imag.end(), [](double x) { return x != 0.0; });
      for (int i = 0; i < n_vectors; i++) {
        if (legacy)
          m.user_xml = legacy_quda_record_xml(reals, m.datatype, subset, parity, nColor, nSpin, value(info.evals, i),
                                              value(info.resids, i));
        else
          m.user_xml = quda_record_xml(reals, subset, parity, nColor, nSpin, value(info.evals, i),
                                       complex_evals ? value(info.evals_imag, i) : std::nullopt, value(info.resids, i));
        v.records.push_back(m);
      }
      return v;
    }

    std::vector<io::Checksum> save_vectors(const std::string &filename, const io::HostField &field, int n_vectors,
                                           const int *X, QudaSiteSubset subset, QudaParity parity, int nColor,
                                           int nSpin, const VectorSaveInfo &info, const FileIOParam &param)
    {
      const int Ls = field.items() / n_vectors;
      const auto v = vector_records(field.precision(), n_vectors, Ls, subset, parity, nColor, nSpin, info);
      std::vector<io::HostField> fields;
      if (info.interleaved) {
        fields.push_back(field);
      } else {
        for (int i = 0; i < n_vectors; i++) fields.push_back(field.slice(i * Ls, Ls));
      }

      const auto X_full = full_dims(X, subset);
      return write_fields(filename, v.user_file_xml, v.records, fields, X_full.data(), file_parity(subset, parity),
                          param);
    }

  } // namespace scidac

} // namespace quda
