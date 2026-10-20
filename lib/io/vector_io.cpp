#include <algorithm>
#include <array>
#include <type_traits>
#include <color_spinor_field.h>
#include <comm_quda.h>
#include <io/qio_field.h>
#include <io/scidac_io.h>
#include <io/vector_io.h>
#include <blas_quda.h>
#include <staggered_sign_convention.h>
#include <timer.h>

namespace quda
{

  namespace
  {
    void check_vector_format(QudaFileFormat format)
    {
      switch (format) {
      case QUDA_FILE_FORMAT_QIO: break;
      case QUDA_FILE_FORMAT_MILC: errorQuda("MILC vector I/O is not yet supported"); break;
      case QUDA_FILE_FORMAT_ILDG: errorQuda("The ILDG format only supports gauge fields"); break;
      default: errorQuda("Unexpected file format %d", format);
      }
    }

    /**
       @brief The host field of a set of host vectors in space-spin-color
       order.  The files hold 4-d fields, so each item is a 4-d vector:
       a whole vector, or one fifth-dimension slice s of a 5-d vector.
       Where each item's sites are in memory depends on the field:

       - 4-d vectors (Wilson, clover, twisted mass, staggered, MG
         near-null vectors), full or single parity: one array per
         vector.  A full field is stored [parity][4-d checkerboard
         site], which is the host index, even sites first.
       - 5-d vectors (domain wall, Mobius), single parity, as the
         eigenvectors of their even-odd preconditioned operators are:
         stored [s][4-d checkerboard site], so slice s is one block at s
         times the 4-d half volume.
       - 5-d vectors, full: either fields of the unpreconditioned
         operator, or single-parity ones saved or loaded with parity
         inflation (through a full temporary).  Stored [parity][s][4-d
         checkerboard site], so slice s's sites are split between
         Even() and Odd(), each at s times the 4-d half volume: two
         arrays per slice.

       Which half holds a slice's 4-d even sites depends on the
       preconditioning.  With 4-d preconditioning (QUDA_4D_PC: Mobius,
       4-d domain wall, and every 4-d operator) the parity is that of
       x + y + z + t, so they are in Even().  With 5-d preconditioning
       (QUDA_5D_PC, which QUDA uses only for QUDA_DOMAIN_WALL_DSLASH)
       the parity is that of x + y + z + t + s, so for odd s they are in
       Odd(), and the two halves are swapped.
     */
    template <typename Field> io::HostField host_field(const std::vector<Field *> &vecs)
    {
      using ptr_t = std::conditional_t<std::is_const_v<Field>, const void *, void *>;
      using byte_t = std::conditional_t<std::is_const_v<Field>, const char *, char *>;
      const Field &v0 = *vecs[0];
      const int Ls = v0.Ndim() == 5 ? v0.X(4) : 1;
      const int n = vecs.size() * Ls;
      const int reals = 2 * v0.Nspin() * v0.Ncolor();
      const size_t site_bytes = reals * v0.Precision();

      if (Ls > 1 && v0.SiteSubset() == QUDA_FULL_SITE_SUBSET) {
        const size_t half_volume = v0.VolumeCB() / Ls;
        std::vector<ptr_t> even(n), odd(n);
        for (auto i = 0u; i < vecs.size(); i++) {
          for (int s = 0; s < Ls; s++) {
            const bool swap = v0.PCType() == QUDA_5D_PC && s % 2 == 1;
            auto &e = swap ? vecs[i]->Odd() : vecs[i]->Even();
            auto &o = swap ? vecs[i]->Even() : vecs[i]->Odd();
            even[i * Ls + s] = e.template data<byte_t>() + s * half_volume * site_bytes;
            odd[i * Ls + s] = o.template data<byte_t>() + s * half_volume * site_bytes;
          }
        }
        return io::HostField(even.data(), odd.data(), n, v0.Precision(), reals, half_volume);
      }

      const size_t stride = v0.Volume() / Ls * site_bytes;
      std::vector<ptr_t> arrays(n);
      for (auto i = 0u; i < vecs.size(); i++)
        for (int s = 0; s < Ls; s++) arrays[i * Ls + s] = vecs[i]->template data<byte_t>() + s * stride;
      return io::HostField(arrays.data(), n, v0.Precision(), reals);
    }

    /**
       @brief The host field of single-parity 4-d host vectors as that
       parity of the full lattice, for files of the full lattice: the
       other parity's sites are skipped on load and saved as zeros
     */
    template <typename Field> io::HostField host_field_in_full(const std::vector<Field *> &vecs, QudaParity parity)
    {
      using ptr_t = std::conditional_t<std::is_const_v<Field>, const void *, void *>;
      const Field &v0 = *vecs[0];
      const int n = vecs.size();
      std::vector<ptr_t> even(n, nullptr), odd(n, nullptr);
      for (int i = 0; i < n; i++) (parity == QUDA_ODD_PARITY ? odd : even)[i] = vecs[i]->data();
      return io::HostField(even.data(), odd.data(), n, v0.Precision(), 2 * v0.Nspin() * v0.Ncolor(), v0.Volume());
    }

    /**
       @brief The local dimensions of the lattice a file holds a set of vectors on
       @param[in] v A vector
       @param[in] full Whether the file holds a single-parity vector on the full lattice
     */
    std::array<int, 4> file_dims(const ColorSpinorField &v, bool full)
    {
      std::array<int, 4> X = {v.X(0), v.X(1), v.X(2), v.X(3)};
      if (full && v.SiteSubset() == QUDA_PARITY_SITE_SUBSET) X[0] *= 2;
      return X;
    }

    // how the vectors were found in a file or saved, for messages
    const char *layout_str(scidac::VectorStrings strings, bool interleaved)
    {
      if (strings == scidac::VectorStrings::MILC) return "in MILC's eigenvector format";
      return interleaved ? "in QUDA's format, all in one record" : "in QUDA's format, one record per vector";
    }

    // the checksums of a set of records, for messages
    template <typename Sum> std::string checksums_str(const std::vector<Sum> &sums)
    {
      if (sums.size() == 1) return io::checksum_str(sums[0]);
      return "of " + std::to_string(sums.size()) + " records";
    }

#ifdef HAVE_QIO
    std::vector<scidac::FieldChecksum> load_qio(const std::string &filename, const scidac::VectorFile &file,
                                                const std::vector<io::HostField> &fields, const int *X,
                                                QudaSiteSubset subset, int nColor, int nSpin, const FileIOParam &param)
    {
      auto sums = read_spinor_field(filename.c_str(), fields, X, subset, nColor, nSpin,
                                    file.strings == scidac::VectorStrings::MILC, param.io_mode, param.io_partition_grid);
      // QIO does not report the checksums stored in the file
      for (auto k = 0u; k < sums.size(); k++)
        if (!sums[k].checksum_matches)
          io::report_checksum_mismatch("Checksum mismatch in binary record " + std::to_string(k) + " of " + filename
                                         + ": computed " + io::checksum_str(sums[k].sum),
                                       param.allow_checksum_mismatch);
      return sums;
    }

    std::vector<io::Checksum> save_qio(const std::string &filename, const io::HostField &field, int n_vectors,
                                       const int *X, QudaSiteSubset subset, QudaParity parity, int nColor, int nSpin,
                                       const scidac::VectorSaveInfo &info, const FileIOParam &param)
    {
      const int Ls = field.items() / n_vectors;
      const auto records = scidac::vector_records(field.precision(), n_vectors, Ls, subset, parity, nColor, nSpin, info);
      std::vector<io::HostField> fields;
      for (int i = 0; i < n_vectors; i++) fields.push_back(field.slice(i * Ls, Ls));
      return write_spinor_field(filename.c_str(), records, fields, X, subset, param.partfile, param.io_mode,
                                param.io_partition_grid);
    }
#else
    std::vector<scidac::FieldChecksum> load_qio(const std::string &, const scidac::VectorFile &,
                                                const std::vector<io::HostField> &, const int *, QudaSiteSubset, int,
                                                int, const FileIOParam &)
    {
      errorQuda("The external QIO backend has not been enabled");
      return {};
    }

    std::vector<io::Checksum> save_qio(const std::string &, const io::HostField &, int, const int *, QudaSiteSubset,
                                       QudaParity, int, int, const scidac::VectorSaveInfo &, const FileIOParam &)
    {
      errorQuda("The external QIO backend has not been enabled");
      return {};
    }
#endif

    const char *backend_str(const FileIOParam &param)
    {
      return param.external_qio ? " with the external QIO library" : "";
    }

    // log the staggered sign convention that the records of a file state, if any
    void log_sign_convention(const scidac::VectorFile &file, const std::string &filename)
    {
      std::vector<std::string> conventions;
      for (auto &f : file.records.fields)
        conventions.push_back(scidac::xml_value(f.metadata.user_xml, "staggeredSignConvention").value_or(""));
      if (std::all_of(conventions.begin(), conventions.end(), [](auto &s) { return s.empty(); })) return;
      if (std::any_of(conventions.begin(), conventions.end(), [&](auto &s) { return s != conventions[0]; })) {
        warningQuda("The records of %s state different staggered sign conventions", filename.c_str());
        return;
      }
      logQuda(QUDA_SUMMARIZE, "The vectors of %s are in %s's staggered sign convention\n", filename.c_str(),
              conventions[0].c_str());
    }
  } // namespace

  VectorIO::VectorIO(const std::string &filename, const FileIOParam &param) : filename(filename), param(param)
  {
    if (filename.empty())
      errorQuda("No eigenspace input file defined (filename = %s, parity_inflate = %d", filename.c_str(),
                param.parity_inflate);
  }

  void VectorIO::load(cvector_ref<ColorSpinorField> &vecs, std::vector<complex_t> *evals)
  {
    check_vector_format(resolve_load_format(filename, param.format));

    const ColorSpinorField &v0 = vecs[0];
    const int Nvec = vecs.size();
    const QudaPrecision load_prec = v0.Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : v0.Precision();
    if (v0.Ndim() != 4 && v0.Ndim() != 5) errorQuda("Unexpected field dimension %d", v0.Ndim());
    const bool single_parity = v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET;
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Start loading %04d vectors from %s\n", Nvec, filename.c_str());

    const auto file = scidac::find_vectors(filename);
    log_sign_convention(file, filename);
    const bool milc = file.strings == scidac::VectorStrings::MILC;
    if (milc && (v0.Nspin() != 1 || v0.Ncolor() != 3 || v0.Ndim() != 4))
      errorQuda("%s is in MILC's eigenvector format, which holds 4-d vectors of 1 spin and 3 colors, but the vectors "
                "have %d spins, %d colors and %d dimensions",
                filename.c_str(), v0.Nspin(), v0.Ncolor(), v0.Ndim());
    // MILC's files hold the full lattice, from which a single-parity field takes its parity whatever parity_inflate
    // says; QUDA's hold single-parity fields inflated, or on half the lattice
    const bool parity_in_full = milc && single_parity;
    const bool inflate = param.parity_inflate && single_parity && !milc;

    auto spinor_parity = v0.SuggestedParity();
    if ((inflate || parity_in_full) && spinor_parity != QUDA_EVEN_PARITY && spinor_parity != QUDA_ODD_PARITY)
      errorQuda("When loading single parity vectors, the suggested parity must be set.");

    std::vector<ColorSpinorField> tmp(Nvec);
    bool create_tmp = load_prec != v0.Precision() || inflate || v0.Location() == QUDA_CUDA_FIELD_LOCATION;

    if (create_tmp) {
      ColorSpinorParam csParam(vecs[0]);
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.setPrecision(load_prec);
      csParam.location = QUDA_CPU_FIELD_LOCATION;
      csParam.create = QUDA_NULL_FIELD_CREATE;
      if (inflate) {
        csParam.x[0] *= 2;
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
      }
      for (int i = 0; i < Nvec; i++) tmp[i] = ColorSpinorField(csParam);
    }

    // the file formats hold 4-d fields, so 5-d vectors are split into their 4-d slices
    const int Ls = v0.Ndim() == 5 ? v0.X(4) : 1;
    std::vector<ColorSpinorField *> V(Nvec);
    for (int i = 0; i < Nvec; i++) V[i] = create_tmp ? &tmp[i] : &vecs[i];
    const auto field = parity_in_full ? host_field_in_full(V, spinor_parity) : host_field(V);
    const auto fields = scidac::vector_fields(file, field, Nvec);

    // the lattice of the file: that of the inflated temporaries, or the full lattice of MILC's files
    const auto X = file_dims(*V[0], parity_in_full);
    const auto subset = (inflate || parity_in_full) ? QUDA_FULL_SITE_SUBSET : v0.SiteSubset();

    quda::host_timer_t host_timer;
    host_timer.start();

    auto sums = param.external_qio ?
      load_qio(filename, file, fields, X.data(), subset, v0.Ncolor(), v0.Nspin(), param) :
      scidac::load_vectors(file, fields, X.data(), subset, spinor_parity, v0.Ncolor(), v0.Nspin(), param);

    host_timer.stop();
    bool verified = true;
    for (auto &s : sums) verified = verified && s.checksum_matches;
    std::vector<io::Checksum> checksums;
    for (auto &s : sums) checksums.push_back(s.sum);
    const auto &first = file.records.fields[0].metadata;
    const bool interleaved = file.records.fields.size() == 1 && first.datacount > Ls;
    logQuda(QUDA_VERBOSE, "Loaded %d %s precision vector fields from %s%s, %s; checksums %s %s\n", Nvec * Ls,
            first.prec == QUDA_DOUBLE_PRECISION ? "double" : "single", filename.c_str(), backend_str(param),
            layout_str(file.strings, interleaved), checksums_str(checksums).c_str(),
            verified ? "verified" : "NOT verified");
    for (auto k = 0u; checksums.size() > 1 && k < checksums.size(); k++)
      logQuda(QUDA_DEBUG_VERBOSE, "Checksums of binary record %u: %s\n", k, io::checksum_str(checksums[k]).c_str());
    logQuda(QUDA_SUMMARIZE, "Time spent loading vectors from %s = %g secs\n", filename.c_str(), host_timer.last());

    if (evals && file.evals.size() >= static_cast<size_t>(Nvec)) {
      if (evals->size() < static_cast<size_t>(Nvec)) evals->resize(Nvec);
      for (int i = 0; i < Nvec; i++) (*evals)[i] = complex_t(file.evals[i].real(), file.evals[i].imag());
      logQuda(QUDA_VERBOSE, "Loaded %d eigenvalues from %s%s\n", Nvec, filename.c_str(),
              milc ? ", in MILC's convention (of the massless operator -D_eo D_oe)" : "");
    }

    if (create_tmp) {
      if (!inflate) {
        for (int i = 0; i < Nvec; i++) vecs[i] = tmp[i];
      } else {
        for (int i = 0; i < Nvec; i++) vecs[i] = spinor_parity == QUDA_EVEN_PARITY ? tmp[i].Even() : tmp[i].Odd();
      }
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done loading vectors\n");
  }

  void VectorIO::save(cvector_ref<const ColorSpinorField> &vecs, QudaPrecision prec, uint32_t size,
                      const std::vector<complex_t> *evals, const std::vector<real_t> *resids)
  {
    check_vector_format(resolve_save_format(param.format));

    const bool milc = param.milc_eigenvectors;
    const ColorSpinorField &v0 = vecs[0];
    const int Nvec = (size != 0 && size < vecs.size()) ? size : vecs.size();
    if (v0.Ndim() != 4 && v0.Ndim() != 5) errorQuda("Unexpected field dimension %d", v0.Ndim());
    if (milc && (v0.Nspin() != 1 || v0.Ncolor() != 3 || v0.Ndim() != 4))
      errorQuda("MILC's eigenvector format (milc_eigenvectors) holds 4-d vectors of 1 spin and 3 colors, but the "
                "vectors have %d spins, %d colors and %d dimensions; QUDA's default format saves them",
                v0.Nspin(), v0.Ncolor(), v0.Ndim());
    if (milc && !evals) errorQuda("MILC's eigenvector format (milc_eigenvectors) needs the eigenvalues");
    if (milc && param.interleaved_vectors) errorQuda("MILC's eigenvector format has one record per vector");
    if (param.staggered_sign_convention && (v0.Nspin() != 1 || v0.Ndim() != 4))
      errorQuda("A staggered sign convention is recorded only for 4-d staggered vectors (1 spin), but the vectors have "
                "%d spins and %d dimensions",
                v0.Nspin(), v0.Ndim());
    if (param.interleaved_vectors && param.external_qio)
      errorQuda("Only the native backend saves vectors interleaved in one record");
    if (evals && evals->size() < static_cast<size_t>(Nvec))
      errorQuda("%lu eigenvalues for %d vectors", evals->size(), Nvec);
    if (resids && resids->size() < static_cast<size_t>(Nvec))
      errorQuda("%lu residuals for %d vectors", resids->size(), Nvec);

    if (prec < QUDA_SINGLE_PRECISION && prec != QUDA_INVALID_PRECISION) errorQuda("Unsupported precision %d", prec);
    const QudaPrecision save_prec = prec != QUDA_INVALID_PRECISION ? prec :
      v0.Precision() < QUDA_SINGLE_PRECISION                       ? QUDA_SINGLE_PRECISION :
                                                                     v0.Precision();

    // MILC's files are always on the full lattice; QUDA's hold single-parity fields inflated, or on half the lattice
    const bool single_parity = v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET;
    const bool parity_in_full = milc && single_parity;
    const bool inflate = param.parity_inflate && single_parity && !milc;
    bool create_tmp = save_prec != v0.Precision() || inflate || v0.Location() == QUDA_CUDA_FIELD_LOCATION;
    auto spinor_parity = v0.SuggestedParity();
    if ((inflate || parity_in_full) && spinor_parity != QUDA_EVEN_PARITY && spinor_parity != QUDA_ODD_PARITY)
      errorQuda("When saving single parity vectors, the suggested parity must be set.");
    std::vector<ColorSpinorField> tmp(Nvec);

    if (create_tmp) {
      ColorSpinorParam csParam(vecs[0]);
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.setPrecision(save_prec);
      csParam.location = QUDA_CPU_FIELD_LOCATION;

      if (!inflate) {
        // We're good, copy as is.
        csParam.create = QUDA_NULL_FIELD_CREATE;
        for (int i = 0; i < Nvec; i++) {
          tmp[i] = ColorSpinorField(csParam);
          tmp[i] = vecs[i];
        }
      } else {                                      // QUDA_PARITY_SITE_SUBSET
        csParam.x[0] *= 2;                          // corrects for the factor of two in the X direction
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET; // create a full-parity field.
        csParam.create = QUDA_ZERO_FIELD_CREATE;    // to explicitly zero the odd sites.
        for (int i = 0; i < Nvec; i++) {
          tmp[i] = ColorSpinorField(csParam);

          // copy the single parity only eigen/singular vector into the even components of the full parity vector
          blas::copy(spinor_parity == QUDA_EVEN_PARITY ? tmp[i].Even() : tmp[i].Odd(), vecs[i]);
        }
      }
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      if (param.partfile)
        printfQuda("Start saving %d vectors to %s in PARTFILE format\n", Nvec, filename.c_str());
      else
        printfQuda("Start saving %d vectors to %s in SINGLEFILE format\n", Nvec, filename.c_str());
    }

    // the file formats hold 4-d fields, so 5-d vectors are split into their 4-d slices
    const int Ls = v0.Ndim() == 5 ? v0.X(4) : 1;
    std::vector<const ColorSpinorField *> V(Nvec);
    for (int i = 0; i < Nvec; i++) V[i] = create_tmp ? &tmp[i] : &vecs[i];
    const auto field = parity_in_full ? host_field_in_full(V, spinor_parity) : host_field(V);

    // the lattice of the file: that of the inflated temporaries, or the full lattice of MILC's files
    const auto X = file_dims(*V[0], parity_in_full);
    const auto subset = (inflate || parity_in_full) ? QUDA_FULL_SITE_SUBSET : v0.SiteSubset();

    scidac::VectorSaveInfo info;
    info.strings = milc ? scidac::VectorStrings::MILC : scidac::VectorStrings::QUDA;
    info.interleaved = param.interleaved_vectors;
    info.legacy_strings = param.legacy_strings;
    for (int i = 0; evals && i < Nvec; i++) {
      info.evals.push_back(static_cast<double>((*evals)[i].real()));
      info.evals_imag.push_back(static_cast<double>((*evals)[i].imag()));
    }
    for (int i = 0; resids && i < Nvec; i++) info.resids.push_back(static_cast<double>((*resids)[i]));
    if (param.staggered_sign_convention)
      info.sign_convention = get_staggered_sign_convention_str(*param.staggered_sign_convention);

    quda::host_timer_t host_timer;
    host_timer.start();

    auto sums = param.external_qio ?
      save_qio(filename, field, Nvec, X.data(), subset, spinor_parity, v0.Ncolor(), v0.Nspin(), info, param) :
      scidac::save_vectors(filename, field, Nvec, X.data(), subset, spinor_parity, v0.Ncolor(), v0.Nspin(), info, param);
    // rank 0 appends the checksum record after the other ranks are done
    comm_barrier();

    host_timer.stop();
    const std::string convention
      = info.sign_convention.empty() ? "" : ", in " + info.sign_convention + "'s staggered sign convention";
    logQuda(QUDA_SUMMARIZE, "Saved %d vectors to %s %s (%lu records)%s%s\n", Nvec, filename.c_str(),
            layout_str(info.strings, info.interleaved), sums.size(), backend_str(param), convention.c_str());
    logQuda(QUDA_VERBOSE, "Saved %d %s precision vector fields to %s%s; checksums %s\n", Nvec * Ls,
            save_prec == QUDA_DOUBLE_PRECISION ? "double" : "single", filename.c_str(), backend_str(param),
            checksums_str(sums).c_str());
    for (auto k = 0u; sums.size() > 1 && k < sums.size(); k++)
      logQuda(QUDA_DEBUG_VERBOSE, "Checksums of binary record %u: %s\n", k, io::checksum_str(sums[k]).c_str());
    logQuda(QUDA_SUMMARIZE, "Time spent saving vectors to %s = %g secs\n", filename.c_str(), host_timer.last());

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done saving vectors\n");
  }

} // namespace quda
