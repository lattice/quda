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

      // time loading
      quda::host_timer_t host_timer;
      host_timer.start(); // start the timer

      read_spinor_field(filename.c_str(), V.data(), v0.Precision(), v0.X(), v0.SiteSubset(),
                        spinor_parity, v0.Ncolor(), v0.Nspin(), Nvec * Ls, 0, nullptr);

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

  void VectorIO::loadProp(vector_ref<ColorSpinorField> &vecs)
  {
    const ColorSpinorField &v0 = vecs[0];	  

    if (vecs.size() != 12) errorQuda("Must have 12 vectors in propagator, passed %lu", vecs.size());
    const int Nvec = vecs.size();
    auto spinor_parity = v0.SuggestedParity();
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Start loading %04d vectors from %s\n", Nvec, filename.c_str());

    std::vector<ColorSpinorField> tmp(Nvec);

    if (vecs[0].Location() == QUDA_CUDA_FIELD_LOCATION) {
      ColorSpinorParam csParam(v0);
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.setPrecision(v0.Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : v0.Precision());
      csParam.location = QUDA_CPU_FIELD_LOCATION;
      csParam.create = QUDA_NULL_FIELD_CREATE;
      if (csParam.siteSubset == QUDA_PARITY_SITE_SUBSET && parity_inflate) {
        csParam.x[0] *= 2;
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
      }
      for (int i = 0; i < Nvec; i++) { 
	 tmp[i] = ColorSpinorField(csParam); 
      }
    } else {
      ColorSpinorParam csParam(v0);
      if (csParam.siteSubset == QUDA_PARITY_SITE_SUBSET && parity_inflate) {
        csParam.x[0] *= 2;
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
        for (int i = 0; i < Nvec; i++) { tmp[i] = ColorSpinorField(csParam); }
      } else {
        for (int i = 0; i < Nvec; i++) { 
	  tmp[i] = ColorSpinorField(csParam);	
	  tmp[i] = vecs[i]; 
	}
      }
    }

    if (v0.Ndim() == 4 || v0.Ndim() == 5) {
      // since QIO routines presently assume we have 4-d fields, we need to convert to array of 4-d fields
      auto Ls = v0.Ndim() == 5 ? tmp[0].X(4) : 1;
      auto V4 = tmp[0].Volume() / Ls;
      auto stride = V4 * tmp[0].Ncolor() * tmp[0].Nspin() * 2 * tmp[0].Precision();
      void **V = static_cast<void **>(safe_malloc(Nvec * Ls * sizeof(void *)));
      for (int i = 0; i < Nvec; i++) {
        for (int j = 0; j < Ls; j++) { V[i * Ls + j] = tmp[i].data<char*>() + j * stride; }
      }

      //read_propagator_field(filename.c_str(), &V[0], tmp[0]->Precision(), tmp[0]->X(), tmp[0]->SiteSubset(),
      //                      spinor_parity, tmp[0]->Ncolor(), tmp[0]->Nspin(), Nvec / 12, 0, (char **)0);

      host_free(V);
    } else {
      errorQuda("Unexpected field dimension %d", vecs[0].Ndim());
    }

    if (v0.Location() == QUDA_CUDA_FIELD_LOCATION) {

      ColorSpinorParam csParam(v0);
      if (csParam.siteSubset == QUDA_FULL_SITE_SUBSET || !parity_inflate) {
        for (int i = 0; i < Nvec; i++) {
          vecs[i] = tmp[i];
        }
      } else {
        // Create a temporary single-parity CPU field
        csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        csParam.setPrecision(v0.Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : v0.Precision());
        csParam.location = QUDA_CPU_FIELD_LOCATION;
        csParam.create = QUDA_NULL_FIELD_CREATE;

        ColorSpinorField tmp_intermediate = ColorSpinorField(csParam);

        for (int i = 0; i < Nvec; i++) {
          if (spinor_parity == QUDA_EVEN_PARITY)
            blas::copy(tmp_intermediate, tmp[i].Even());
          else if (spinor_parity == QUDA_ODD_PARITY)
            blas::copy(tmp_intermediate, tmp[i].Odd());
          else
            errorQuda("When loading single parity vectors, the suggested parity must be set.");

          vecs[i] = tmp_intermediate;
        }
      }
    } else if (v0.Location() == QUDA_CPU_FIELD_LOCATION && v0.SiteSubset() == QUDA_PARITY_SITE_SUBSET) {
      for (int i = 0; i < Nvec; i++) {
        if (spinor_parity == QUDA_EVEN_PARITY)
          blas::copy(vecs[i], tmp[i].Even());
        else if (spinor_parity == QUDA_ODD_PARITY)
          blas::copy(vecs[i], tmp[i].Odd());
        else
          errorQuda("When loading single parity vectors, the suggested parity must be set.");
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

      // time saving
      quda::host_timer_t host_timer;
      host_timer.start(); // start the timer

      write_spinor_field(filename.c_str(), V.data(), save_prec, v0.X(), v0.SiteSubset(), spinor_parity, v0.Ncolor(),
                         v0.Nspin(), Nvec * Ls, 0, nullptr, partfile);

      host_timer.stop(); // stop the timer
      logQuda(QUDA_SUMMARIZE, "Time spent saving vectors to %s = %g secs\n", filename.c_str(), host_timer.last());
    } else {
      errorQuda("Unexpected field dimension %d", v0.Ndim());
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done saving vectors\n");
  }

  void VectorIO::saveProp(cvector_ref<ColorSpinorField> &vecs)
  {
    if (vecs.size() != 12) errorQuda("Must have 12 vectors in propagator, passed %lu", vecs.size());

    const int Nvec = vecs.size();
    std::vector<ColorSpinorField> tmp(Nvec);
    auto spinor_parity = vecs[0].SuggestedParity();
    if (vecs[0].Location() == QUDA_CUDA_FIELD_LOCATION) {
      ColorSpinorParam csParam(vecs[0]);
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.setPrecision(vecs[0].Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : vecs[0].Precision());
      csParam.location = QUDA_CPU_FIELD_LOCATION;

      if (csParam.siteSubset == QUDA_FULL_SITE_SUBSET || !parity_inflate) {
        // We're good, copy as is.
        csParam.create = QUDA_NULL_FIELD_CREATE;
        for (int i = 0; i < Nvec; i++) {
          tmp[i] = ColorSpinorField(csParam);
          tmp[i] = vecs[i];
        }
      } else { // QUDA_PARITY_SITE_SUBSET
        csParam.create = QUDA_NULL_FIELD_CREATE;

        // intermediate host single-parity field
        ColorSpinorField tmp_intermediate = ColorSpinorField(csParam);

        csParam.x[0] *= 2;                          // corrects for the factor of two in the X direction
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET; // create a full-parity field.
        csParam.create = QUDA_ZERO_FIELD_CREATE;    // to explicitly zero the odd sites.
        for (int i = 0; i < Nvec; i++) {
          tmp[i] = ColorSpinorField(csParam);

          // copy the single parity eigen/singular vector into an
          // intermediate device-side vector
          tmp_intermediate = vecs[i];

          // copy the single parity only eigen/singular vector into the even components of the full parity vector
          if (spinor_parity == QUDA_EVEN_PARITY)
            blas::copy(tmp[i].Even(), tmp_intermediate);
          else if (spinor_parity == QUDA_ODD_PARITY)
            blas::copy(tmp[i].Odd(), tmp_intermediate);
          else
            errorQuda("When saving single parity vectors, the suggested parity must be set.");
        }
      }
    } else {
      ColorSpinorParam csParam(vecs[0]);
      if (csParam.siteSubset == QUDA_PARITY_SITE_SUBSET && parity_inflate) {
        csParam.x[0] *= 2;
        csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        for (int i = 0; i < Nvec; i++) {
          tmp[i] = ColorSpinorField(csParam);
          if (spinor_parity == QUDA_EVEN_PARITY)
            blas::copy(tmp[i].Even(), vecs[i]);
          else if (spinor_parity == QUDA_ODD_PARITY)
            blas::copy(tmp[i].Odd(), vecs[i]);
          else
            errorQuda("When saving single parity vectors, the suggested parity must be set.");
        }
      } else {      
        for (int i = 0; i < Nvec; i++) { 
          tmp[i] = ColorSpinorField(csParam); 
	  tmp[i] = vecs[i]; 
	}
      }
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Start saving %d vectors to %s\n", Nvec, filename.c_str());

    if (vecs[0].Ndim() == 4 || vecs[0].Ndim() == 5) {
      // since QIO routines presently assume we have 4-d fields, we need to convert to array of 4-d fields
      auto Ls = vecs[0].Ndim() == 5 ? tmp[0].X(4) : 1;
      auto V4 = tmp[0].Volume() / Ls;
      auto stride = V4 * tmp[0].Ncolor() * tmp[0].Nspin() * 2 * tmp[0].Precision();
      void **V = static_cast<void **>(safe_malloc(Nvec * Ls * sizeof(void *)));
      for (int i = 0; i < Nvec; i++) {
        for (int j = 0; j < Ls; j++) { V[i * Ls + j] = tmp[i].data<char*>() + j * stride; }
      }

      //write_propagator_field(filename.c_str(), &V[0], tmp[0]->Precision(), tmp[0]->X(), tmp[0]->SiteSubset(),
      //                       spinor_parity, tmp[0]->Ncolor(), tmp[0]->Nspin(), (Nvec) / 12, 0, (char **)0);

      host_free(V);
    } else {
      errorQuda("Unexpected field dimension %d", vecs[0].Ndim());
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Done saving vectors\n");
  }

  void VectorIO::downPrec(cvector_ref<ColorSpinorField> &vecs_high_prec,
                          vector_ref<ColorSpinorField> &vecs_low_prec, const QudaPrecision low_prec)
  {
    if (low_prec >= vecs_high_prec[0].Precision()) {
      errorQuda("Attempting to down-prec from precision %d to %d", vecs_high_prec[0].Precision(), low_prec);
    }
#if 0
    ColorSpinorParam csParamClone(vecs_high_prec[0]);
    csParamClone.create = QUDA_REFERENCE_FIELD_CREATE;
    csParamClone.setPrecision(low_prec);
    for (unsigned int i = 0; i < vecs_high_prec.size(); i++) {
      vecs_low_prec[i] = vecs_high_prec[i].CreateAlias(csParamClone);
    }
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("Vector space successfully down copied from prec %d to prec %d\n", vecs_high_prec[0].Precision(),
                 vecs_low_prec[0].Precision());
    }
#endif    
  }

} // namespace quda
