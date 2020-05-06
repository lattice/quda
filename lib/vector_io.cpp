#include <color_spinor_field.h>
#include <qio_field.h>
#include <vector_io.h>
#include <blas_quda.h>

namespace quda
{

  VectorIO::VectorIO(const std::string &filename, bool parity_inflate) :
    filename(filename),
    parity_inflate(parity_inflate)
  {
    if (strcmp(filename.c_str(), "") == 0) { errorQuda("No eigenspace input file defined."); }
  }

  void VectorIO::load(std::vector<ColorSpinorField *> &vecs)
  {
#ifdef HAVE_QIO
    const int Nvec = vecs.size();
    auto spinor_parity = vecs[0]->SuggestedParity();
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Start loading %04d vectors from %s\n", Nvec, filename.c_str());

    std::vector<ColorSpinorField *> tmp;
    tmp.reserve(Nvec);
    if (vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
      ColorSpinorParam csParam(*vecs[0]);
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.setPrecision(vecs[0]->Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : vecs[0]->Precision());
      csParam.location = QUDA_CPU_FIELD_LOCATION;
      csParam.create = QUDA_NULL_FIELD_CREATE;
      if (csParam.siteSubset == QUDA_PARITY_SITE_SUBSET && parity_inflate) {
        csParam.x[0] *= 2;
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
      }
      for (int i = 0; i < Nvec; i++) { tmp.push_back(ColorSpinorField::Create(csParam)); }
    } else {
      ColorSpinorParam csParam(*vecs[0]);
      if (csParam.siteSubset == QUDA_PARITY_SITE_SUBSET && parity_inflate) {
        csParam.x[0] *= 2;
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
        for (int i = 0; i < Nvec; i++) { tmp.push_back(ColorSpinorField::Create(csParam)); }
      } else {
        for (int i = 0; i < Nvec; i++) { tmp.push_back(vecs[i]); }
      }
    }

    if (vecs[0]->Ndim() == 4 || vecs[0]->Ndim() == 5) {
      // since QIO routines presently assume we have 4-d fields, we need to convert to array of 4-d fields
      auto Ls = vecs[0]->Ndim() == 5 ? tmp[0]->X(4) : 1;
      auto V4 = tmp[0]->Volume() / Ls;
      auto stride = V4 * tmp[0]->Ncolor() * tmp[0]->Nspin() * 2 * tmp[0]->Precision();
      void **V = static_cast<void **>(safe_malloc(Nvec * Ls * sizeof(void *)));
      for (int i = 0; i < Nvec; i++) {
        for (int j = 0; j < Ls; j++) { V[i * Ls + j] = static_cast<char *>(tmp[i]->V()) + j * stride; }
      }

      read_spinor_field(filename.c_str(), &V[0], tmp[0]->Precision(), tmp[0]->X(), tmp[0]->SiteSubset(), spinor_parity,
                        tmp[0]->Ncolor(), tmp[0]->Nspin(), Nvec * Ls, 0, (char **)0);

      host_free(V);
    } else {
      errorQuda("Unexpected field dimension %d", vecs[0]->Ndim());
    }

    if (vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {

      ColorSpinorParam csParam(*vecs[0]);
      if (csParam.siteSubset == QUDA_FULL_SITE_SUBSET || !parity_inflate) {
        for (int i = 0; i < Nvec; i++) {
          *vecs[i] = *tmp[i];
          delete tmp[i];
        }
      } else {
        // Create a temporary single-parity CPU field
        csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        csParam.setPrecision(vecs[0]->Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : vecs[0]->Precision());
        csParam.location = QUDA_CPU_FIELD_LOCATION;
        csParam.create = QUDA_NULL_FIELD_CREATE;

        ColorSpinorField *tmp_intermediate = ColorSpinorField::Create(csParam);

        for (int i = 0; i < Nvec; i++) {
          if (spinor_parity == QUDA_EVEN_PARITY)
            blas::copy(*tmp_intermediate, tmp[i]->Even());
          else if (spinor_parity == QUDA_ODD_PARITY)
            blas::copy(*tmp_intermediate, tmp[i]->Odd());
          else
            errorQuda("When loading single parity vectors, the suggested parity must be set.");

          *vecs[i] = *tmp_intermediate;
          delete tmp[i];
        }

        delete tmp_intermediate;
      }
    } else if (vecs[0]->Location() == QUDA_CPU_FIELD_LOCATION && vecs[0]->SiteSubset() == QUDA_PARITY_SITE_SUBSET) {
      for (int i = 0; i < Nvec; i++) {
        if (spinor_parity == QUDA_EVEN_PARITY)
          blas::copy(*vecs[i], tmp[i]->Even());
        else if (spinor_parity == QUDA_ODD_PARITY)
          blas::copy(*vecs[i], tmp[i]->Odd());
        else
          errorQuda("When loading single parity vectors, the suggested parity must be set.");

        delete tmp[i];
      }
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done loading vectors\n");
#else
    errorQuda("\nQIO library was not built.\n");
#endif
  }

  void VectorIO::save(const std::vector<ColorSpinorField *> &vecs)
  {
#ifdef HAVE_QIO
    const int Nvec = vecs.size();
    std::vector<ColorSpinorField *> tmp;
    tmp.reserve(Nvec);
    auto spinor_parity = vecs[0]->SuggestedParity();
    if (vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
      ColorSpinorParam csParam(*vecs[0]);
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.setPrecision(vecs[0]->Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : vecs[0]->Precision());
      csParam.location = QUDA_CPU_FIELD_LOCATION;

      if (csParam.siteSubset == QUDA_FULL_SITE_SUBSET || !parity_inflate) {
        // We're good, copy as is.
        csParam.create = QUDA_NULL_FIELD_CREATE;
        for (int i = 0; i < Nvec; i++) {
          tmp.push_back(ColorSpinorField::Create(csParam));
          *tmp[i] = *vecs[i];
        }
      } else { // QUDA_PARITY_SITE_SUBSET
        csParam.create = QUDA_NULL_FIELD_CREATE;

        // intermediate host single-parity field
        ColorSpinorField *tmp_intermediate = ColorSpinorField::Create(csParam);

        csParam.x[0] *= 2;                          // corrects for the factor of two in the X direction
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET; // create a full-parity field.
        csParam.create = QUDA_ZERO_FIELD_CREATE;    // to explicitly zero the odd sites.
        for (int i = 0; i < Nvec; i++) {
          tmp.push_back(ColorSpinorField::Create(csParam));

          // copy the single parity eigen/singular vector into an
          // intermediate device-side vector
          *tmp_intermediate = *vecs[i];

          // copy the single parity only eigen/singular vector into the even components of the full parity vector
          if (spinor_parity == QUDA_EVEN_PARITY)
            blas::copy(tmp[i]->Even(), *tmp_intermediate);
          else if (spinor_parity == QUDA_ODD_PARITY)
            blas::copy(tmp[i]->Odd(), *tmp_intermediate);
          else
            errorQuda("When saving single parity vectors, the suggested parity must be set.");
        }
        delete tmp_intermediate;
      }
    } else {
      ColorSpinorParam csParam(*vecs[0]);
      if (csParam.siteSubset == QUDA_PARITY_SITE_SUBSET && parity_inflate) {
        csParam.x[0] *= 2;
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        for (int i = 0; i < Nvec; i++) {
          tmp.push_back(ColorSpinorField::Create(csParam));
          if (spinor_parity == QUDA_EVEN_PARITY)
            blas::copy(tmp[i]->Even(), *vecs[i]);
          else if (spinor_parity == QUDA_ODD_PARITY)
            blas::copy(tmp[i]->Odd(), *vecs[i]);
          else
            errorQuda("When saving single parity vectors, the suggested parity must be set.");
        }
      } else {
        for (int i = 0; i < Nvec; i++) { tmp.push_back(vecs[i]); }
      }
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Start saving %d vectors to %s\n", Nvec, filename.c_str());

    if (vecs[0]->Ndim() == 4 || vecs[0]->Ndim() == 5) {
      // since QIO routines presently assume we have 4-d fields, we need to convert to array of 4-d fields
      auto Ls = vecs[0]->Ndim() == 5 ? tmp[0]->X(4) : 1;
      auto V4 = tmp[0]->Volume() / Ls;
      auto stride = V4 * tmp[0]->Ncolor() * tmp[0]->Nspin() * 2 * tmp[0]->Precision();
      void **V = static_cast<void **>(safe_malloc(Nvec * Ls * sizeof(void *)));
      for (int i = 0; i < Nvec; i++) {
        for (int j = 0; j < Ls; j++) { V[i * Ls + j] = static_cast<char *>(tmp[i]->V()) + j * stride; }
      }

      write_spinor_field(filename.c_str(), &V[0], tmp[0]->Precision(), tmp[0]->X(), tmp[0]->SiteSubset(), spinor_parity,
                         tmp[0]->Ncolor(), tmp[0]->Nspin(), Nvec * Ls, 0, (char **)0);

      host_free(V);
    } else {
      errorQuda("Unexpected field dimension %d", vecs[0]->Ndim());
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done saving vectors\n");
    if (vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION
        || (vecs[0]->Location() == QUDA_CPU_FIELD_LOCATION && vecs[0]->SiteSubset() == QUDA_PARITY_SITE_SUBSET)) {
      for (int i = 0; i < Nvec; i++) delete tmp[i];
    }
#else
    errorQuda("\nQIO library was not built.\n");
#endif
  }

} // namespace quda
