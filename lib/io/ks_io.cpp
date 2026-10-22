#include <algorithm>
#include <array>
#include <cstring>
#include <set>
#include <color_spinor_field.h>
#include <comm_quda.h>
#include <util_quda.h>
#include <io/ks_io.h>
#include <io/qio_field.h>
#include <io/scidac_io.h>
#include <timer.h>

namespace quda
{

  namespace
  {
    // the reals of a site of a 3-color vector and of a complex field
    constexpr int color_vector_reals = 6;
    constexpr int complex_reals = 2;

    void check_ks_format(const std::string &filename, const FileIOParam &param)
    {
      if (param.format == QUDA_FILE_FORMAT_MILC || param.format == QUDA_FILE_FORMAT_ILDG)
        errorQuda("Cannot load %s in the %s format: staggered sources and propagators are SciDAC files",
                  filename.c_str(), get_file_format_str(param.format));
      const auto format = resolve_load_format(filename, param.format);
      if (format != QUDA_FILE_FORMAT_QIO)
        errorQuda("%s is a %s file, but staggered sources and propagators are SciDAC files", filename.c_str(),
                  get_file_format_str(format));
    }

    const char *subset_str(const ColorSpinorField &f)
    {
      if (f.SiteSubset() == QUDA_FULL_SITE_SUBSET) return "full";
      switch (f.SuggestedParity()) {
      case QUDA_EVEN_PARITY: return "even-parity";
      case QUDA_ODD_PARITY: return "odd-parity";
      default: return "single-parity (without a parity)";
      }
    }

    // the local dimensions of a field's full lattice
    std::array<int, 4> full_dims(const ColorSpinorField &f)
    {
      return {f.X(0) * (f.SiteSubset() == QUDA_PARITY_SITE_SUBSET ? 2 : 1), f.X(1), f.X(2), f.X(3)};
    }

    /**
       @brief Check that fields are 4-d staggered fields, full or of a set parity, on one lattice
       @param[in] dims The local dimensions of the full lattice, set from the first field checked
     */
    void check_ks_fields(cvector_ref<ColorSpinorField> &fields, const char *what, std::optional<std::array<int, 4>> &dims)
    {
      for (auto i = 0u; i < fields.size(); i++) {
        const ColorSpinorField &f = fields[i];
        const bool parity_set = f.SuggestedParity() == QUDA_EVEN_PARITY || f.SuggestedParity() == QUDA_ODD_PARITY;
        if (f.Ndim() != 4 || f.Nspin() != 1 || f.Ncolor() != 3
            || (f.SiteSubset() == QUDA_PARITY_SITE_SUBSET && !parity_set))
          errorQuda("Staggered sources and solutions are 4-d fields of 1 spin and 3 colors, full or of the even or odd "
                    "parity, but %s %u is a %s field of %d dimensions, %d spins and %d colors",
                    what, i, subset_str(f), f.Ndim(), f.Nspin(), f.Ncolor());
        const auto X = full_dims(f);
        if (!dims) dims = X;
        if (X != *dims)
          errorQuda("%s %u is on the local lattice %d %d %d %d, but the other fields on %d %d %d %d", what, i, X[0],
                    X[1], X[2], X[3], (*dims)[0], (*dims)[1], (*dims)[2], (*dims)[3]);
      }
    }

    /**
       @brief The fields to load into, as zeroed host fields in space-spin-color order in single or double
       precision: the fields themselves where they are such fields, otherwise temporaries that are copied back
     */
    class HostCopies
    {
      cvector_ref<ColorSpinorField> &fields;
      std::vector<ColorSpinorField> tmp;
      std::vector<ColorSpinorField *> host;

    public:
      HostCopies(cvector_ref<ColorSpinorField> &fields) : fields(fields), tmp(fields.size()), host(fields.size())
      {
        for (auto i = 0u; i < fields.size(); i++) {
          ColorSpinorField &f = fields[i];
          if (f.Precision() >= QUDA_SINGLE_PRECISION && f.Location() == QUDA_CPU_FIELD_LOCATION
              && f.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
            std::memset(f.data(), 0, f.Bytes());
            host[i] = &f;
            continue;
          }
          ColorSpinorParam param(f);
          param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
          param.setPrecision(std::max(f.Precision(), QUDA_SINGLE_PRECISION));
          param.location = QUDA_CPU_FIELD_LOCATION;
          param.create = QUDA_ZERO_FIELD_CREATE;
          tmp[i] = ColorSpinorField(param);
          host[i] = &tmp[i];
        }
      }

      ColorSpinorField &operator[](size_t i) { return *host[i]; }

      size_t size() const { return host.size(); }

      void copy_back()
      {
        for (auto i = 0u; i < host.size(); i++)
          if (host[i] == &tmp[i]) fields[i] = tmp[i];
      }
    };

    /**
       @brief The host field of one item that a record is read into: the whole of a full host field, or the
       field's parity of the full lattice; with reals 2 and stride 6, color 0 of each site
     */
    io::HostField item(ColorSpinorField &h, int reals = color_vector_reals, int stride = 0)
    {
      void *data = h.data();
      if (h.SiteSubset() == QUDA_FULL_SITE_SUBSET) return io::HostField(&data, 1, h.Precision(), reals, stride);
      void *none = nullptr;
      const bool odd = h.SuggestedParity() == QUDA_ODD_PARITY;
      return io::HostField(odd ? &none : &data, odd ? &data : &none, 1, h.Precision(), reals, h.Volume(), stride);
    }

#ifdef HAVE_QIO
    /**
       @brief A host field that holds no sites, for a record that is read but not loaded
     */
    io::HostField no_item(const scidac::FieldMetadata &m)
    {
      void *none = nullptr;
      return io::HostField(&none, 1, QUDA_DOUBLE_PRECISION, m.typesize / m.prec);
    }

    std::vector<scidac::FieldChecksum> load_qio(const std::string &filename, const scidac::KSFile &file,
                                                const std::vector<int> &records, const std::vector<io::HostField> &fields,
                                                const int *X, const FileIOParam &param)
    {
      // QIO reads the records in turn, so those not to be loaded are read into fields without sites
      std::vector<io::HostField> all;
      for (int k = 0, i = 0; k <= records.back(); k++)
        all.push_back(k == records[i] ? fields[i++] : no_item(file.records.fields[k].metadata));
      const auto read = read_ks_fields(filename.c_str(), all, X, param.io_mode, param.io_partition_grid);
      std::vector<scidac::FieldChecksum> sums;
      for (int k : records) {
        // QIO does not report the checksums stored in the file
        if (!read[k].checksum_matches)
          io::report_checksum_mismatch("Checksum mismatch in binary record " + std::to_string(k) + " of " + filename
                                         + ": computed " + io::checksum_str(read[k].sum),
                                       param.allow_checksum_mismatch);
        sums.push_back(read[k]);
      }
      return sums;
    }
#else
    std::vector<scidac::FieldChecksum> load_qio(const std::string &, const scidac::KSFile &, const std::vector<int> &,
                                                const std::vector<io::HostField> &, const int *, const FileIOParam &)
    {
      errorQuda("The external QIO backend has not been enabled");
      return {};
    }
#endif

    /**
       @brief Check the lattice of a staggered file and the sizes of its first records, as the native backend reads
       them, so that the external backend gives the same messages
     */
    void check_sizes(const scidac::KSFile &file, int n_records, const int *X)
    {
      const auto &records = file.records;
      const auto N = io::global_dims(X);
      const auto &D = records.dims;
      if (D != N)
        errorQuda("%s has dimensions %d %d %d %d, but the field's lattice is %d %d %d %d", records.file.c_str(), D[0],
                  D[1], D[2], D[3], N[0], N[1], N[2], N[3]);
      for (int k = 0; k < n_records; k++) {
        const auto &r = records.fields[k];
        const auto &box = r.metadata.hypercube;
        const size_t sites = box ? box->volume() : static_cast<size_t>(N[0]) * N[1] * N[2] * N[3];
        const size_t bytes = sites * r.metadata.typesize * r.metadata.datacount;
        if (r.data.data_bytes != bytes)
          errorQuda("The binary data of binary record %d of %s is %lu bytes, but should be %lu bytes%s", k,
                    records.file.c_str(), r.data.data_bytes, bytes,
                    box ? (" for the sites " + scidac::hypercube_str(*box)).c_str() : "");
      }
    }

    /**
       @brief Read records of a staggered file, in ascending order, each into its host field, on the full lattice
     */
    std::vector<scidac::FieldChecksum> read_records(const std::string &filename, const scidac::KSFile &file,
                                                    const std::vector<int> &records,
                                                    const std::vector<io::HostField> &fields, const int *X,
                                                    const FileIOParam &param)
    {
      check_sizes(file, records.back() + 1, X);
      if (param.external_qio) return load_qio(filename, file, records, fields, X, param);
      auto selected = file.records;
      selected.fields.clear();
      for (int k : records) selected.fields.push_back(file.records.fields[k]);
      return scidac::read_fields(selected, fields, X, QUDA_INVALID_PARITY, param);
    }

    /**
       @brief Copy color 0 of the first field into color c of field c, for each further field
     */
    template <typename Dst, typename Src> void copy_color(ColorSpinorField &dst, const ColorSpinorField &src, int c)
    {
      auto d = dst.data<Dst *>();
      auto s = src.data<const Src *>();
      for (size_t x = 0; x < src.Volume(); x++)
        for (int j = 0; j < complex_reals; j++)
          d[x * color_vector_reals + complex_reals * c + j] = s[x * color_vector_reals + j];
    }

    void promote_complex(HostCopies &host)
    {
      const auto &src = host[0];
      for (auto c = 1u; c < host.size(); c++) {
        auto &dst = host[c];
        const bool d = dst.Precision() == QUDA_DOUBLE_PRECISION, s = src.Precision() == QUDA_DOUBLE_PRECISION;
        if (d && s) copy_color<double, double>(dst, src, c);
        if (d && !s) copy_color<double, float>(dst, src, c);
        if (!d && s) copy_color<float, double>(dst, src, c);
        if (!d && !s) copy_color<float, float>(dst, src, c);
      }
    }

    /**
       @brief The boxes of a set of records, for messages: how many hold the full lattice, time slices (and
       which) and other boxes
     */
    std::string boxes_str(const std::vector<std::optional<io::Hypercube>> &boxes, const std::array<int, 4> &N,
                          const char *what)
    {
      size_t full = 0;
      std::set<int> slices;
      size_t n_slices = 0;
      std::vector<std::string> others;
      for (auto &b : boxes) {
        if (!b) {
          full++;
        } else if (b->lower[3] == b->upper[3] && b->lower[0] == 0 && b->lower[1] == 0 && b->lower[2] == 0
                   && b->upper[0] == N[0] - 1 && b->upper[1] == N[1] - 1 && b->upper[2] == N[2] - 1) {
          n_slices++;
          slices.insert(b->lower[3]);
        } else {
          others.push_back(scidac::hypercube_str(*b));
        }
      }
      auto count = [&](size_t n, const std::string &kind) {
        return std::to_string(n) + " " + kind + " " + what + (n == 1 ? " record" : " records");
      };
      std::vector<std::string> parts;
      if (full > 0) parts.push_back(count(full, "full-lattice"));
      if (n_slices > 0) {
        std::string t;
        for (int s : slices) t += (t.empty() ? "" : ", ") + std::to_string(s);
        parts.push_back(count(n_slices, "time-slice") + " (t = " + t + ")");
      }
      if (!others.empty()) {
        std::string sites;
        for (auto &o : others) sites += (sites.empty() ? "" : "; ") + o;
        parts.push_back(count(others.size(), "hypercube") + " (sites " + sites + ")");
      }
      std::string s;
      for (auto i = 0u; i < parts.size(); i++) s += (i == 0 ? "" : i + 1 == parts.size() ? " and " : ", ") + parts[i];
      return s;
    }

    const char *backend_str(const FileIOParam &param)
    {
      return param.external_qio ? " with the external QIO library" : "";
    }

    void log_checksums(const std::string &filename, const std::vector<int> &records,
                       const std::vector<scidac::FieldChecksum> &sums)
    {
      for (auto i = 0u; i < sums.size(); i++)
        logQuda(QUDA_VERBOSE, "Checksums of binary record %d of %s: %s %s\n", records[i], filename.c_str(),
                io::checksum_str(sums[i].sum).c_str(), sums[i].checksum_matches ? "verified" : "NOT verified");
    }

    // the lattice of a file, for messages
    std::array<int, 4> global_dims(const std::array<int, 4> &X) { return io::global_dims(X.data()); }
  } // namespace

  KSSourceIO::KSSourceIO(const std::string &filename, const FileIOParam &param) : filename(filename), param(param)
  {
    if (filename.empty()) errorQuda("No staggered source file given");
  }

  KSFileInfo KSSourceIO::load(cvector_ref<ColorSpinorField> &sources)
  {
    check_ks_format(filename, param);
    const int n = sources.size();
    if (n == 0) errorQuda("No sources to load from %s", filename.c_str());
    std::optional<std::array<int, 4>> X;
    check_ks_fields(sources, "Source", X);

    quda::host_timer_t host_timer;
    host_timer.start();
    const auto file = scidac::find_ks_file(filename);
    const auto &fields = file.records.fields;
    KSFileInfo info;
    info.type = file.type;
    info.file_info = file.file_info;

    std::vector<int> records;
    switch (file.type) {
    case KSFileType::VectorSource:
      if (n > static_cast<int>(fields.size()))
        errorQuda("The staggered source file %s holds %lu sources, fewer than the %d to be loaded",
                  file.records.file.c_str(), fields.size(), n);
      for (int i = 0; i < n; i++) records.push_back(i);
      break;
    case KSFileType::Propagator:
      if (n > static_cast<int>(fields.size() / 2))
        errorQuda("The USQCD KS propagator file %s holds %lu sources, fewer than the %d to be loaded",
                  file.records.file.c_str(), fields.size() / 2, n);
      for (int i = 0; i < n; i++) records.push_back(2 * i);
      break;
    case KSFileType::ComplexSource:
      if (n > 3)
        errorQuda("The complex-field file %s is promoted into one of 3 colors for each source, so it loads into at "
                  "most 3 sources, not %d",
                  file.records.file.c_str(), n);
      for (int i = 1; i < n; i++)
        if (sources[i].SiteSubset() != sources[0].SiteSubset()
            || (sources[i].SiteSubset() == QUDA_PARITY_SITE_SUBSET
                && sources[i].SuggestedParity() != sources[0].SuggestedParity()))
          errorQuda(
            "The sources of a complex-field file must all be full or all of one parity, but source %d is %s and "
            "source 0 %s",
            i, subset_str(sources[i]), subset_str(sources[0]));
      records.push_back(0);
      break;
    }

    HostCopies host(sources);
    std::vector<io::HostField> items;
    if (file.type == KSFileType::ComplexSource)
      items.push_back(item(host[0], complex_reals, color_vector_reals));
    else
      for (int i = 0; i < n; i++) items.push_back(item(host[i]));
    const auto sums = read_records(filename, file, records, items, X->data(), param);
    if (file.type == KSFileType::ComplexSource) promote_complex(host);
    host.copy_back();

    info.file_prec = fields[records[0]].metadata.prec;
    std::vector<std::optional<io::Hypercube>> record_boxes;
    for (int k : records) record_boxes.push_back(fields[k].metadata.hypercube);
    for (int i = 0; i < n; i++) {
      const int k = records[file.type == KSFileType::ComplexSource ? 0 : i];
      info.boxes.push_back(fields[k].metadata.hypercube);
      info.source_info.push_back(file.info[k]);
    }
    host_timer.stop();

    const auto N = global_dims(*X);
    const std::string boxes = boxes_str(record_boxes, N, "source");
    switch (file.type) {
    case KSFileType::VectorSource:
      logQuda(QUDA_SUMMARIZE, "Loaded %d sources from the staggered source file %s%s, %s\n", n, filename.c_str(),
              backend_str(param), boxes.c_str());
      break;
    case KSFileType::Propagator:
      logQuda(QUDA_SUMMARIZE, "Loaded %d sources from the USQCD KS propagator file %s (VV pairs)%s, %s\n", n,
              filename.c_str(), backend_str(param), boxes.c_str());
      break;
    case KSFileType::ComplexSource: {
      const std::string colors = n == 1 ? "color 0" : "colors 0-" + std::to_string(n - 1);
      logQuda(QUDA_SUMMARIZE, "Loaded %d sources from the complex-field file %s (%s)%s, promoted to %s, %s\n", n,
              filename.c_str(), fields[0].metadata.datatype.c_str(), backend_str(param), colors.c_str(), boxes.c_str());
      break;
    }
    }
    log_checksums(filename, records, sums);
    logQuda(QUDA_VERBOSE, "Time spent loading sources from %s = %g secs\n", filename.c_str(), host_timer.last());
    return info;
  }

  KSPropagatorIO::KSPropagatorIO(const std::string &filename, const FileIOParam &param) :
    filename(filename), param(param)
  {
    if (filename.empty()) errorQuda("No staggered propagator file given");
  }

  KSFileInfo KSPropagatorIO::load(cvector_ref<ColorSpinorField> &solutions, cvector_ref<ColorSpinorField> &sources)
  {
    check_ks_format(filename, param);
    const int n = solutions.size();
    if (n == 0) errorQuda("No solutions to load from %s", filename.c_str());
    const bool with_sources = sources.size() > 0;
    if (with_sources && static_cast<int>(sources.size()) != n)
      errorQuda("%lu sources for the %d solutions to load from %s: give none or one per solution", sources.size(), n,
                filename.c_str());
    std::optional<std::array<int, 4>> X;
    check_ks_fields(solutions, "Solution", X);
    check_ks_fields(sources, "Source", X);

    quda::host_timer_t host_timer;
    host_timer.start();
    const auto file = scidac::find_ks_file(filename);
    if (file.type != KSFileType::Propagator)
      errorQuda("%s is a staggered source file, not a USQCD KS propagator file; load it with KSSourceIO",
                file.records.file.c_str());
    const auto &fields = file.records.fields;
    if (n > static_cast<int>(fields.size() / 2))
      errorQuda("The USQCD KS propagator file %s holds %lu source/solution pairs, fewer than the %d to be loaded",
                file.records.file.c_str(), fields.size() / 2, n);

    HostCopies host_solutions(solutions);
    HostCopies host_sources(sources);
    std::vector<int> records;
    std::vector<io::HostField> items;
    for (int i = 0; i < n; i++) {
      if (with_sources) {
        records.push_back(2 * i);
        items.push_back(item(host_sources[i]));
      }
      records.push_back(2 * i + 1);
      items.push_back(item(host_solutions[i]));
    }
    const auto sums = read_records(filename, file, records, items, X->data(), param);
    host_solutions.copy_back();
    host_sources.copy_back();

    KSFileInfo info;
    info.type = file.type;
    info.file_info = file.file_info;
    info.file_prec = fields[records[0]].metadata.prec;
    for (int i = 0; i < n; i++) {
      info.solution_info.push_back(file.info[2 * i + 1]);
      if (!with_sources) continue;
      info.boxes.push_back(fields[2 * i].metadata.hypercube);
      info.source_info.push_back(file.info[2 * i]);
    }
    host_timer.stop();

    if (with_sources) {
      logQuda(QUDA_SUMMARIZE,
              "Loaded %d sources and %d solutions from the USQCD KS propagator file %s (VV pairs)%s, %s\n", n, n,
              filename.c_str(), backend_str(param), boxes_str(info.boxes, global_dims(*X), "source").c_str());
    } else {
      logQuda(QUDA_SUMMARIZE,
              "Loaded %d solutions from the USQCD KS propagator file %s (VV pairs)%s, without their sources\n", n,
              filename.c_str(), backend_str(param));
    }
    log_checksums(filename, records, sums);
    logQuda(QUDA_VERBOSE, "Time spent loading the propagator from %s = %g secs\n", filename.c_str(), host_timer.last());
    return info;
  }

} // namespace quda
