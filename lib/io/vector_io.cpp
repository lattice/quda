#include <color_spinor_field.h>
#include <io/qio_field.h>
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
      case QUDA_FILE_FORMAT_QIO:
#ifndef HAVE_QIO
        errorQuda("QIO support has not been enabled");
#endif
        break;
      case QUDA_FILE_FORMAT_MILC: errorQuda("MILC vector I/O is not yet supported"); break;
      case QUDA_FILE_FORMAT_ILDG: errorQuda("The ILDG format only supports gauge fields"); break;
      default: errorQuda("Unexpected file format %d", format);
      }
    }

#ifdef HAVE_QIO
    void load_qio(const std::string &filename, void *V[], QudaPrecision prec, const int *X, QudaSiteSubset subset,
                  QudaParity parity, int nColor, int nSpin, int Nvec, QudaFileIOMode io_mode)
    {
      read_spinor_field(filename.c_str(), V, prec, X, subset, parity, nColor, nSpin, Nvec, io_mode);
    }

    void save_qio(const std::string &filename, const void *V[], QudaPrecision prec, const int *X, QudaSiteSubset subset,
                  QudaParity parity, int nColor, int nSpin, int Nvec, bool partfile, QudaFileIOMode io_mode)
    {
      write_spinor_field(filename.c_str(), V, prec, X, subset, parity, nColor, nSpin, Nvec, partfile, io_mode);
    }
#else
    void load_qio(const std::string &, void *[], QudaPrecision, const int *, QudaSiteSubset, QudaParity, int, int, int,
                  QudaFileIOMode)
    {
      errorQuda("QIO support has not been enabled");
    }

    void save_qio(const std::string &, const void *[], QudaPrecision, const int *, QudaSiteSubset, QudaParity, int, int,
                  int, bool, QudaFileIOMode)
    {
      errorQuda("QIO support has not been enabled");
    }
#endif
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
      auto stride = V4 * v0.Ncolor() * v0.Nspin() * 2 * v0.Precision();
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

      load_qio(filename, V.data(), v0.Precision(), spinor_X, spinor_site_subset, spinor_parity, v0.Ncolor(), v0.Nspin(),
               Nvec * Ls, param.io_mode);

      host_timer.stop(); // stop the timer
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
      auto stride = V4 * v0.Ncolor() * v0.Nspin() * 2 * v0.Precision();
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

      save_qio(filename, V.data(), save_prec, spinor_X, spinor_site_subset, spinor_parity, v0.Ncolor(), v0.Nspin(),
               Nvec * Ls, partfile, param.io_mode);

      host_timer.stop(); // stop the timer
      logQuda(QUDA_SUMMARIZE, "Time spent saving vectors to %s = %g secs\n", filename.c_str(), host_timer.last());
    } else {
      errorQuda("Unexpected field dimension %d", v0.Ndim());
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done saving vectors\n");
  }

} // namespace quda
