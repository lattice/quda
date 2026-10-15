#include <color_spinor_field.h>
#include <comm_quda.h>
#include <io/qio_field.h>
#include <io/scidac_io.h>
#include <io/vector_io.h>
#include <blas_quda.h>
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

#ifdef HAVE_QIO
    std::pair<QudaPrecision, scidac::FieldChecksum> load_qio(const std::string &filename, void *V[], QudaPrecision prec,
                                                             const int *X, QudaSiteSubset subset, QudaParity parity,
                                                             int nColor, int nSpin, int Nvec, const FileIOParam &param)
    {
      auto result = read_spinor_field(filename.c_str(), V, prec, X, subset, parity, nColor, nSpin, Nvec, param.io_mode,
                                      param.io_partition_grid);
      // QIO does not report the checksums stored in the file
      const auto &sum = result.second;
      if (!sum.checksum_matches)
        io::report_checksum_mismatch("Checksum mismatch in " + filename + ": computed " + io::checksum_str(sum.sum),
                                     param.allow_checksum_mismatch);
      return result;
    }

    io::Checksum save_qio(const std::string &filename, const void *V[], QudaPrecision prec, const int *X,
                          QudaSiteSubset subset, QudaParity parity, int nColor, int nSpin, int Nvec,
                          const FileIOParam &param)
    {
      return write_spinor_field(filename.c_str(), V, prec, X, subset, parity, nColor, nSpin, Nvec, param.partfile,
                                param.io_mode, param.io_partition_grid);
    }
#else
    std::pair<QudaPrecision, scidac::FieldChecksum> load_qio(const std::string &, void *[], QudaPrecision, const int *,
                                                             QudaSiteSubset, QudaParity, int, int, int,
                                                             const FileIOParam &)
    {
      errorQuda("The external QIO backend has not been enabled");
      return {};
    }

    io::Checksum save_qio(const std::string &, const void *[], QudaPrecision, const int *, QudaSiteSubset, QudaParity,
                          int, int, int, const FileIOParam &)
    {
      errorQuda("The external QIO backend has not been enabled");
      return {};
    }
#endif

    const char *backend_str(const FileIOParam &param)
    {
      return param.external_qio ? " with the external QIO library" : "";
    }
  } // namespace

  VectorIO::VectorIO(const std::string &filename, const FileIOParam &param) : filename(filename), param(param)
  {
    if (filename.empty())
      errorQuda("No eigenspace input file defined (filename = %s, parity_inflate = %d", filename.c_str(),
                param.parity_inflate);
  }

  void VectorIO::load(cvector_ref<ColorSpinorField> &vecs, std::vector<complex_t> *)
  {
    check_vector_format(resolve_load_format(filename, param.format));

    const bool parity_inflate = param.parity_inflate;
    const ColorSpinorField &v0 = vecs[0];
    const int Nvec = vecs.size();
    const QudaPrecision load_prec = v0.Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : v0.Precision();

    auto spinor_parity = v0.SuggestedParity();
    if (v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET && parity_inflate && spinor_parity != QUDA_EVEN_PARITY
        && spinor_parity != QUDA_ODD_PARITY)
      errorQuda("When loading single parity vectors, the suggested parity must be set.");
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Start loading %04d vectors from %s\n", Nvec, filename.c_str());

    std::vector<ColorSpinorField> tmp(Nvec);
    bool create_tmp = load_prec != v0.Precision() || (v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET && parity_inflate)
      || v0.Location() == QUDA_CUDA_FIELD_LOCATION;

    if (create_tmp) {
      ColorSpinorParam csParam(vecs[0]);
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.setPrecision(load_prec);
      csParam.location = QUDA_CPU_FIELD_LOCATION;
      csParam.create = QUDA_NULL_FIELD_CREATE;
      if (csParam.siteSubset == QUDA_PARITY_SITE_SUBSET && parity_inflate) {
        csParam.x[0] *= 2;
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
      }
      for (int i = 0; i < Nvec; i++) tmp[i] = ColorSpinorField(csParam);
    }

    if (v0.Ndim() == 4 || v0.Ndim() == 5) {
      // since QIO routines presently assume we have 4-d fields, we need to convert to array of 4-d fields
      auto Ls = v0.Ndim() == 5 ? v0.X(4) : 1;
      auto V4 = v0.Volume() / Ls;
      if (v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET && parity_inflate) V4 *= 2;
      // the vectors passed on are in load_prec, either the temporary fields or vecs itself
      auto stride = V4 * v0.Ncolor() * v0.Nspin() * 2 * load_prec;
      std::vector<void *> V(Nvec * Ls);
      for (int i = 0; i < Nvec; i++) {
        auto &v = create_tmp ? tmp[i] : vecs[i];
        for (int j = 0; j < Ls; j++) { V[i * Ls + j] = v.data<char *>() + j * stride; }
      }

      // if we're loading inflated vectors, we need to grab the spinor size from `tmp` instead of `v0`.
      auto spinor_X = (create_tmp && parity_inflate) ? tmp[0].X() : v0.X();
      auto spinor_site_subset = (create_tmp && parity_inflate) ? tmp[0].SiteSubset() : v0.SiteSubset();

      // time loading
      quda::host_timer_t host_timer;
      host_timer.start(); // start the timer

      auto [file_prec, sum] = param.external_qio ?
        load_qio(filename, V.data(), load_prec, spinor_X, spinor_site_subset, spinor_parity, v0.Ncolor(), v0.Nspin(),
                 Nvec * Ls, param) :
        scidac::load_vectors(filename, V.data(), load_prec, spinor_X, spinor_site_subset, spinor_parity, v0.Ncolor(),
                             v0.Nspin(), Nvec * Ls, param);

      host_timer.stop(); // stop the timer
      logQuda(QUDA_VERBOSE, "Loaded %d %s precision vector fields from %s%s; checksums %s %s\n", Nvec * Ls,
              file_prec == QUDA_DOUBLE_PRECISION ? "double" : "single", filename.c_str(), backend_str(param),
              io::checksum_str(sum.sum).c_str(), sum.checksum_matches ? "verified" : "NOT verified");
      logQuda(QUDA_SUMMARIZE, "Time spent loading vectors from %s = %g secs\n", filename.c_str(), host_timer.last());
    } else {
      errorQuda("Unexpected field dimension %d", v0.Ndim());
    }

    if (create_tmp) {
      if (v0.SiteSubset() == QUDA_FULL_SITE_SUBSET || !parity_inflate) {
        for (int i = 0; i < Nvec; i++) vecs[i] = tmp[i];
      } else {
        for (int i = 0; i < Nvec; i++) vecs[i] = spinor_parity == QUDA_EVEN_PARITY ? tmp[i].Even() : tmp[i].Odd();
      }
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done loading vectors\n");
  }

  void VectorIO::save(cvector_ref<const ColorSpinorField> &vecs, QudaPrecision prec, uint32_t size,
                      const std::vector<complex_t> *)
  {
    check_vector_format(resolve_save_format(param.format));

    const bool parity_inflate = param.parity_inflate;
    const bool partfile = param.partfile;
    const ColorSpinorField &v0 = vecs[0];
    const int Nvec = (size != 0 && size < vecs.size()) ? size : vecs.size();
    if (prec < QUDA_SINGLE_PRECISION && prec != QUDA_INVALID_PRECISION) errorQuda("Unsupported precision %d", prec);
    const QudaPrecision save_prec = prec != QUDA_INVALID_PRECISION ? prec :
      v0.Precision() < QUDA_SINGLE_PRECISION                       ? QUDA_SINGLE_PRECISION :
                                                                     v0.Precision();

    bool create_tmp = save_prec != v0.Precision() || (v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET && parity_inflate)
      || v0.Location() == QUDA_CUDA_FIELD_LOCATION;
    auto spinor_parity = v0.SuggestedParity();
    if (v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET && parity_inflate && spinor_parity != QUDA_EVEN_PARITY
        && spinor_parity != QUDA_ODD_PARITY)
      errorQuda("When loading single parity vectors, the suggested parity must be set.");
    std::vector<ColorSpinorField> tmp(Nvec);

    if (create_tmp) {
      ColorSpinorParam csParam(vecs[0]);
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.setPrecision(save_prec);
      csParam.location = QUDA_CPU_FIELD_LOCATION;

      if (csParam.siteSubset == QUDA_FULL_SITE_SUBSET || !parity_inflate) {
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
      if (partfile)
        printfQuda("Start saving %d vectors to %s in PARTFILE format\n", Nvec, filename.c_str());
      else
        printfQuda("Start saving %d vectors to %s in SINGLEFILE format\n", Nvec, filename.c_str());
    }

    if (v0.Ndim() == 4 || v0.Ndim() == 5) {
      // since QIO routines presently assume we have 4-d fields, we need to convert to array of 4-d fields
      auto Ls = v0.Ndim() == 5 ? v0.X(4) : 1;
      auto V4 = v0.Volume() / Ls;
      if (v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET && parity_inflate) V4 *= 2;
      // the vectors passed on are in save_prec, either the temporary fields or vecs itself
      auto stride = V4 * v0.Ncolor() * v0.Nspin() * 2 * save_prec;
      std::vector<const void *> V(Nvec * Ls);
      for (int i = 0; i < Nvec; i++) {
        auto &v = create_tmp ? tmp[i] : vecs[i];
        for (int j = 0; j < Ls; j++) { V[i * Ls + j] = v.data<const char *>() + j * stride; }
      }

      // if we performed parity inflation, we need to grab the spinor size from `tmp` instead of `v0`.
      auto spinor_X = (create_tmp && parity_inflate) ? tmp[0].X() : v0.X();
      auto spinor_site_subset = (create_tmp && parity_inflate) ? tmp[0].SiteSubset() : v0.SiteSubset();

      // time saving
      quda::host_timer_t host_timer;
      host_timer.start(); // start the timer

      auto sum = param.external_qio ? save_qio(filename, V.data(), save_prec, spinor_X, spinor_site_subset,
                                               spinor_parity, v0.Ncolor(), v0.Nspin(), Nvec * Ls, param) :
                                      scidac::save_vectors(filename, V.data(), save_prec, spinor_X, spinor_site_subset,
                                                           spinor_parity, v0.Ncolor(), v0.Nspin(), Nvec * Ls, param);
      // rank 0 appends the checksum record after the other ranks are done
      comm_barrier();

      host_timer.stop(); // stop the timer
      logQuda(QUDA_VERBOSE, "Saved %d %s precision vector fields to %s%s; checksums %s\n", Nvec * Ls,
              save_prec == QUDA_DOUBLE_PRECISION ? "double" : "single", filename.c_str(), backend_str(param),
              io::checksum_str(sum).c_str());
      logQuda(QUDA_SUMMARIZE, "Time spent saving vectors to %s = %g secs\n", filename.c_str(), host_timer.last());
    } else {
      errorQuda("Unexpected field dimension %d", v0.Ndim());
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done saving vectors\n");
  }

} // namespace quda
