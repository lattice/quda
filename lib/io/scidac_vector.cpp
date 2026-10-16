#include <array>
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
    } // namespace

    std::pair<QudaPrecision, FieldChecksum> load_vectors(const std::string &filename, const io::HostField &field,
                                                         const int *X, QudaSiteSubset subset, QudaParity parity,
                                                         int nColor, int nSpin, const FileIOParam &param)
    {
      const int Nvec = field.items();
      auto records = find_field(filename);
      const auto &file = records.file;
      const auto &m = records.metadata;
      if (!records.native)
        errorQuda("%s is an ILDG file without the SciDAC records, so it holds no vectors", file.c_str());
      if (m.datacount != Nvec) {
        if (records.n_binary > 1)
          errorQuda(
            "%s holds %d binary records with %d vectors each (MILC writes one vector per record), but only files "
            "with all %d vectors in one record are supported so far",
            file.c_str(), records.n_binary, m.datacount, Nvec);
        errorQuda("The datacount %d of %s does not match the expected number of vectors %d", m.datacount, file.c_str(),
                  Nvec);
      }
      if (m.spins != nSpin)
        warningQuda("The file %s has %d spins, but the vectors have %d", file.c_str(), m.spins, nSpin);
      if (m.colors != nColor)
        warningQuda("The file %s has %d colors, but the vectors have %d", file.c_str(), m.colors, nColor);

      const auto X_full = full_dims(X, subset);
      return {m.prec, read_field(records, field, X_full.data(), file_parity(subset, parity), param)};
    }

    io::Checksum save_vectors(const std::string &filename, const io::HostField &field, const int *X,
                              QudaSiteSubset subset, QudaParity parity, int nColor, int nSpin, const FileIOParam &param)
    {
      if (param.partfile)
        errorQuda("The native QIO backend cannot write partfiles yet: save %s with the qio-external format, or as a "
                  "single file",
                  filename.c_str());

      const int reals = 2 * nSpin * nColor;
      const QudaPrecision prec = field.precision();
      FieldMetadata m;
      m.datatype = vector_datatype(prec, nSpin, nColor);
      m.prec = prec;
      m.colors = nColor;
      m.spins = nSpin;
      m.typesize = reals * prec;
      m.datacount = field.items();
      m.user_xml = quda_record_xml(reals, m.datatype, subset, parity, nColor, nSpin);

      const auto X_full = full_dims(X, subset);
      return write_field(filename, m, field, X_full.data(), file_parity(subset, parity), param);
    }

  } // namespace scidac

} // namespace quda
