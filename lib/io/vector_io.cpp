#include <type_traits>
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

#ifdef HAVE_QIO
    std::pair<QudaPrecision, scidac::FieldChecksum> load_qio(const std::string &filename, const io::HostField &field,
                                                             const int *X, QudaSiteSubset subset, QudaParity parity,
                                                             int nColor, int nSpin, const FileIOParam &param)
    {
      auto result = read_spinor_field(filename.c_str(), field, X, subset, parity, nColor, nSpin, param.io_mode,
                                      param.io_partition_grid);
      // QIO does not report the checksums stored in the file
      const auto &sum = result.second;
      if (!sum.checksum_matches)
        io::report_checksum_mismatch("Checksum mismatch in " + filename + ": computed " + io::checksum_str(sum.sum),
                                     param.allow_checksum_mismatch);
      return result;
    }

    io::Checksum save_qio(const std::string &filename, const io::HostField &field, const int *X, QudaSiteSubset subset,
                          QudaParity parity, int nColor, int nSpin, const FileIOParam &param)
    {
      return write_spinor_field(filename.c_str(), field, X, subset, parity, nColor, nSpin, param.partfile,
                                param.io_mode, param.io_partition_grid);
    }
#else
    std::pair<QudaPrecision, scidac::FieldChecksum> load_qio(const std::string &, const io::HostField &, const int *,
                                                             QudaSiteSubset, QudaParity, int, int, const FileIOParam &)
    {
      errorQuda("The external QIO backend has not been enabled");
      return {};
    }

    io::Checksum save_qio(const std::string &, const io::HostField &, const int *, QudaSiteSubset, QudaParity, int, int,
                          const FileIOParam &)
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
      // the file formats hold 4-d fields, so 5-d vectors are split into their 4-d slices
      auto Ls = v0.Ndim() == 5 ? v0.X(4) : 1;
      std::vector<ColorSpinorField *> V(Nvec);
      for (int i = 0; i < Nvec; i++) V[i] = create_tmp ? &tmp[i] : &vecs[i];
      const auto field = host_field(V);

      // if we're loading inflated vectors, we need to grab the spinor size from `tmp` instead of `v0`.
      auto spinor_X = (create_tmp && parity_inflate) ? tmp[0].X() : v0.X();
      auto spinor_site_subset = (create_tmp && parity_inflate) ? tmp[0].SiteSubset() : v0.SiteSubset();

      // time loading
      quda::host_timer_t host_timer;
      host_timer.start(); // start the timer

      auto [file_prec, sum] = param.external_qio ?
        load_qio(filename, field, spinor_X, spinor_site_subset, spinor_parity, v0.Ncolor(), v0.Nspin(), param) :
        scidac::load_vectors(filename, field, spinor_X, spinor_site_subset, spinor_parity, v0.Ncolor(), v0.Nspin(),
                             param);

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
      // the file formats hold 4-d fields, so 5-d vectors are split into their 4-d slices
      auto Ls = v0.Ndim() == 5 ? v0.X(4) : 1;
      std::vector<const ColorSpinorField *> V(Nvec);
      for (int i = 0; i < Nvec; i++) V[i] = create_tmp ? &tmp[i] : &vecs[i];
      const auto field = host_field(V);

      // if we performed parity inflation, we need to grab the spinor size from `tmp` instead of `v0`.
      auto spinor_X = (create_tmp && parity_inflate) ? tmp[0].X() : v0.X();
      auto spinor_site_subset = (create_tmp && parity_inflate) ? tmp[0].SiteSubset() : v0.SiteSubset();

      // time saving
      quda::host_timer_t host_timer;
      host_timer.start(); // start the timer

      auto sum = param.external_qio ?
        save_qio(filename, field, spinor_X, spinor_site_subset, spinor_parity, v0.Ncolor(), v0.Nspin(), param) :
        scidac::save_vectors(filename, field, spinor_X, spinor_site_subset, spinor_parity, v0.Ncolor(), v0.Nspin(),
                             param);
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
