#include <color_spinor_field.h>
#include <qio_field.h>
#include <vector_io.h>
#include <blas_quda.h>
#include <timer.h>

namespace quda
{

  VectorIO::VectorIO(const std::string &filename, bool parity_inflate, bool partfile) :
    filename(filename), parity_inflate(parity_inflate), partfile(partfile)
  {
    if (strcmp(filename.c_str(), "") == 0)
      errorQuda("No eigenspace input file defined (filename = %s, parity_inflate = %d", filename.c_str(), parity_inflate);
  }

  void VectorIO::load(cvector_ref<ColorSpinorField> &vecs)
  {
    const ColorSpinorField &v0 = vecs[0];
    const int Nvec = vecs.size();
    const QudaPrecision load_prec = v0.Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : v0.Precision();

    auto spinor_parity = v0.SuggestedParity();
    if (v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET && parity_inflate &&
        spinor_parity != QUDA_EVEN_PARITY && spinor_parity != QUDA_ODD_PARITY)
      errorQuda("When loading single parity vectors, the suggested parity must be set.");
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Start loading %04d vectors from %s\n", Nvec, filename.c_str());

    std::vector<ColorSpinorField> tmp(Nvec);
    bool create_tmp = load_prec != v0.Precision() || (v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET && parity_inflate) ||
      v0.Location() == QUDA_CUDA_FIELD_LOCATION;

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
      auto spinor_X = parity_inflate ? tmp[0].X() : v0.X();
      auto spinor_site_subset = parity_inflate ? tmp[0].SiteSubset() : v0.SiteSubset();

      // time loading
      quda::host_timer_t host_timer;
      host_timer.start(); // start the timer

      read_spinor_field(filename.c_str(), V.data(), v0.Precision(), spinor_X, spinor_site_subset, spinor_parity,
                        v0.Ncolor(), v0.Nspin(), Nvec * Ls, 0, nullptr);

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

  void VectorIO::save(cvector_ref<const ColorSpinorField> &vecs, QudaPrecision prec, uint32_t size)
  {
    const ColorSpinorField &v0 = vecs[0];
    const int Nvec = (size != 0 && size < vecs.size()) ? size : vecs.size();
    if (prec < QUDA_SINGLE_PRECISION && prec != QUDA_INVALID_PRECISION) errorQuda("Unsupported precision %d", prec);
    const QudaPrecision save_prec = prec != QUDA_INVALID_PRECISION ? prec :
      v0.Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : v0.Precision();

    bool create_tmp = save_prec != v0.Precision() || (v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET && parity_inflate) ||
      v0.Location() == QUDA_CUDA_FIELD_LOCATION;
    auto spinor_parity = v0.SuggestedParity();
    if (v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET && parity_inflate &&
        spinor_parity != QUDA_EVEN_PARITY && spinor_parity != QUDA_ODD_PARITY)
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
      } else { // QUDA_PARITY_SITE_SUBSET
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
      auto spinor_X = parity_inflate ? tmp[0].X() : v0.X();
      auto spinor_site_subset = parity_inflate ? tmp[0].SiteSubset() : v0.SiteSubset();

      // time saving
      quda::host_timer_t host_timer;
      host_timer.start(); // start the timer

      write_spinor_field(filename.c_str(), V.data(), save_prec, spinor_X, spinor_site_subset, spinor_parity,
                         v0.Ncolor(), v0.Nspin(), Nvec * Ls, 0, nullptr, partfile);

      host_timer.stop(); // stop the timer
      logQuda(QUDA_SUMMARIZE, "Time spent saving vectors to %s = %g secs\n", filename.c_str(), host_timer.last());
    } else {
      errorQuda("Unexpected field dimension %d", v0.Ndim());
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done saving vectors\n");
  }

} // namespace quda
