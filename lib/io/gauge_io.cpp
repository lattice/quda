#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>
#include <util_quda.h>
#include <timer.h>
#include <comm_quda.h>
#include <gauge_field.h>
#include <gauge_tools.h>
#include <pgauge_monte.h>
#include <io/gauge_io.h>
#include <io/milc_io.h>
#include <io/qio_field.h>
#include <io/reunitarize.h>
#include <io/scidac_io.h>
#include <io/site_io.h>

namespace quda
{

  namespace
  {
    // the QDP arrays of a host field, one per direction
    struct HostArrays {
      void *data[4];
      explicit HostArrays(const GaugeField &u)
      {
        for (int d = 0; d < 4; d++) data[d] = u.data(d);
      }
    };

    /**
       @brief Parameters for a reference field of general links with periodic boundaries on the QDP arrays of a
       host field
     */
    GaugeFieldParam host_field_param(const GaugeField &host)
    {
      QudaGaugeParam gauge_param = newQudaGaugeParam();
      for (int d = 0; d < 4; d++) gauge_param.X[d] = host.X()[d];
      gauge_param.type = QUDA_GENERAL_LINKS;
      gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
      gauge_param.cpu_prec = host.Precision();
      gauge_param.t_boundary = QUDA_PERIODIC_T;
      gauge_param.anisotropy = 1.0;
      gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

      GaugeFieldParam field_param(gauge_param, host.raw_pointer());
      field_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      return field_param;
    }

    void check_field(const GaugeField &u)
    {
      if (u.Ncolor() != 3) errorQuda("Gauge field files hold 3 colors, not %d", u.Ncolor());
      if (u.Geometry() != QUDA_VECTOR_GEOMETRY)
        errorQuda("Gauge field files hold one link per direction, not geometry %d", u.Geometry());
      switch (u.LinkType()) {
      case QUDA_SU3_LINKS:
      case QUDA_GENERAL_LINKS:
      case QUDA_THREE_LINKS:
      case QUDA_SMEARED_LINKS: break;
      default: errorQuda("Gauge field files hold links, not fields of link type %d", u.LinkType());
      }
    }

    // the precision of the host copy of a field the formats take: at least single
    QudaPrecision host_precision(QudaPrecision prec)
    {
      return prec < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : prec;
    }

    lat_dim_t interior_dims(const GaugeField &u)
    {
      lat_dim_t X = u.X();
      if (u.GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED)
        for (int d = 0; d < 4; d++) X[d] -= 2 * u.R()[d];
      return X;
    }

    /**
       @brief The staggered phases and t boundary sign that the links of a field hold, which a conversion removes
       on save and applies on load (see GaugeIO)
     */
    struct LinkConventions {
      bool phases = false; /** Whether the links hold staggered phases */
      QudaStaggeredPhase phase_type = QUDA_STAGGERED_PHASE_NO;
      bool sign = false;       /** Whether the links hold the antiperiodic t boundary sign */
      bool long_links = false; /** Long links: the sign on the last three time slices */

      bool any() const { return phases || sign; }

      std::string str() const
      {
        std::string s;
        if (phases) s = std::string("the ") + get_staggered_phase_str(phase_type) + " staggered phases";
        if (phases && sign) s += " and ";
        if (sign)
          s += long_links ? "the antiperiodic t boundary sign of the last three time slices" :
                            "the antiperiodic t boundary sign of the last time slice";
        return s;
      }

      static const char *get_staggered_phase_str(QudaStaggeredPhase type)
      {
        switch (type) {
        case QUDA_STAGGERED_PHASE_MILC: return "MILC";
        case QUDA_STAGGERED_PHASE_TIFR: return "TIFR";
        case QUDA_STAGGERED_PHASE_CHROMA: return "CHROMA";
        default: return "unknown";
        }
      }
    };

    LinkConventions link_conventions(const GaugeField &u, bool convert)
    {
      LinkConventions c;
      if (!convert) return c;
      const auto type = u.StaggeredPhase();
      const bool has_type
        = type == QUDA_STAGGERED_PHASE_MILC || type == QUDA_STAGGERED_PHASE_TIFR || type == QUDA_STAGGERED_PHASE_CHROMA;
      c.phases = u.StaggeredPhaseApplied();
      c.phase_type = type;
      c.long_links = u.LinkType() == QUDA_THREE_LINKS;
      // applyStaggeredPhase applies the boundary sign with the phases
      c.sign = u.TBoundary() == QUDA_ANTI_PERIODIC_T && (!has_type || c.phases);

      if (c.phases) {
        if (!has_type)
          errorQuda("Only MILC, TIFR and CHROMA staggered phases can be converted, not phase type %d", type);
        if (u.iMu() != 0.0)
          errorQuda("The staggered phases of a field with an imaginary chemical potential (i_mu = %g) cannot be "
                    "converted",
                    static_cast<double>(u.iMu()));
        if (c.long_links && type != QUDA_STAGGERED_PHASE_MILC)
          errorQuda("Only MILC staggered phases can be converted on long links, not %s", c.get_staggered_phase_str(type));
        // applyStaggeredPhase takes the phases from local coordinates, which give the global phases only for even
        // local extents
        const std::vector<int> depends = type == QUDA_STAGGERED_PHASE_MILC ? std::vector<int> {0, 1, 3} :
          type == QUDA_STAGGERED_PHASE_TIFR                                ? std::vector<int> {1, 2, 3} :
                                                                             std::vector<int> {0, 1, 2};
        const auto X = interior_dims(u);
        for (int d : depends)
          if (X[d] % 2 != 0)
            errorQuda("The %s staggered phases cannot be converted with the odd local extent %d in dimension %d",
                      c.get_staggered_phase_str(type), X[d], d);
      }
      return c;
    }

    // the sign of a link in direction dir at global coordinates g from the staggered phases (as getPhase in
    // kernels/gauge_phase.cuh defines them, without the boundary)
    int staggered_phase(QudaStaggeredPhase type, int dir, const int g[4])
    {
      int n = 0;
      switch (type) {
      case QUDA_STAGGERED_PHASE_MILC: {
        const int exponents[4] = {g[3], g[3] + g[0], g[3] + g[0] + g[1], 0};
        n = exponents[dir];
      } break;
      case QUDA_STAGGERED_PHASE_TIFR: {
        const int exponents[4] = {3 + g[3] + g[2] + g[1], 2 + g[3] + g[2], 1 + g[3], 0};
        n = exponents[dir];
      } break;
      case QUDA_STAGGERED_PHASE_CHROMA: {
        const int exponents[4] = {1, 1 + g[0], 1 + g[0] + g[1], 1 + g[0] + g[1] + g[2]};
        n = exponents[dir];
      } break;
      default: errorQuda("Unexpected staggered phase type %d", type);
      }
      return n % 2 == 0 ? 1 : -1;
    }

    /**
       @brief Multiply the links of a host QDP field by the signs of a set of conventions, which applies them to
       plain links and removes them from links that hold them.  None of the three conventions' phase of a link in
       direction mu depends on x_mu, so a long link's phase, the product of the phases of its three links, is that
       of one link.
     */
    template <typename Float> void apply_conventions(GaugeField &view, const LinkConventions &c)
    {
      const auto &X = view.X();
      const size_t volume = view.Volume();
      const size_t half = view.VolumeCB();
      const int T = X[3] * comm_dim(3);
      const int sign_slices = c.long_links ? 3 : 1;
      int offset[4];
      for (int d = 0; d < 4; d++) offset[d] = comm_coord(d) * X[d];

      for (int dir = 0; dir < 4; dir++) {
        auto u = view.data<Float *>(dir);
#pragma omp parallel for
        for (size_t r = 0; r < volume; r++) {
          int x[4], g[4];
          size_t rest = r;
          for (int d = 0; d < 4; d++) {
            x[d] = rest % X[d];
            rest /= X[d];
            g[d] = x[d] + offset[d];
          }
          int s = c.phases ? staggered_phase(c.phase_type, dir, g) : 1;
          if (c.sign && dir == 3 && g[3] >= T - sign_slices) s = -s;
          if (s > 0) continue;
          const size_t parity = (x[0] + x[1] + x[2] + x[3]) % 2;
          Float *link = u + (parity * half + r / 2) * 18;
          for (int i = 0; i < 18; i++) link[i] = -link[i];
        }
      }
    }

    void apply_conventions(GaugeField &view, const LinkConventions &c)
    {
      if (view.Precision() == QUDA_DOUBLE_PRECISION)
        apply_conventions<double>(view, c);
      else
        apply_conventions<float>(view, c);
    }

    /**
       @brief The parameters of the host field the formats take for a field: QDP order, no reconstruct, no ghost
       zone, the interior of an extended field
     */
    GaugeFieldParam file_view_param(const GaugeField &u, QudaPrecision prec)
    {
      GaugeFieldParam param(u);
      param.location = QUDA_CPU_FIELD_LOCATION;
      param.create = QUDA_NULL_FIELD_CREATE;
      param.reconstruct = QUDA_RECONSTRUCT_NO;
      param.setPrecision(prec);
      param.order = QUDA_QDP_GAUGE_ORDER;
      param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      param.pad = 0;
      param.x = interior_dims(u);
      param.r = {};
      return param;
    }

    // whether a field is the host field the formats take in the given precision
    bool is_file_view(const GaugeField &u, QudaPrecision prec)
    {
      return u.Location() == QUDA_CPU_FIELD_LOCATION && u.Order() == QUDA_QDP_GAUGE_ORDER && u.Precision() == prec
        && u.Reconstruct() == QUDA_RECONSTRUCT_NO && u.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED;
    }

    /**
       @brief Copy a field's links into a host file view, the interior of an extended field
     */
    void copy_to_file_view(GaugeField &view, const GaugeField &u)
    {
      if (u.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
        view.copy(u);
      } else if (u.Location() == QUDA_CPU_FIELD_LOCATION) {
        copyExtendedGauge(view, u, QUDA_CPU_FIELD_LOCATION);
      } else {
        GaugeFieldParam param(u);
        param.create = QUDA_NULL_FIELD_CREATE;
        param.reconstruct = QUDA_RECONSTRUCT_NO;
        param.setPrecision(u.Precision(), true);
        param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
        param.pad = 0;
        param.x = interior_dims(u);
        param.r = {};
        GaugeField interior(param);
        copyExtendedGauge(interior, u, QUDA_CUDA_FIELD_LOCATION);
        view.copy(interior);
      }
    }

    /**
       @brief Parameters for a native device field with the same contents as a host field
     */
    GaugeFieldParam device_field_param(const GaugeFieldParam &host_param)
    {
      GaugeFieldParam field_param = host_param;
      field_param.create = QUDA_NULL_FIELD_CREATE;
      field_param.reconstruct = QUDA_RECONSTRUCT_NO;
      field_param.setPrecision(host_param.Precision(), true);
      field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
      field_param.location = QUDA_CUDA_FIELD_LOCATION;
      return field_param;
    }

    /**
       @brief Compute the average plaquettes (all, spatial and temporal) and the link trace, normalized as Re Tr / 3
     */
    void compute_plaquettes_linktr(const GaugeField &view, GaugeObservables &obs)
    {
      auto field_param = host_field_param(view);
      GaugeField host(field_param);
      GaugeField device(device_field_param(field_param));
      device.copy(host);

      // getLinkTrace averages Tr U over the links of a field without a halo
      obs.linktr = getLinkTrace(device).real() / 3.0;

      lat_dim_t R;
      for (int d = 0; d < 4; d++) R[d] = comm_dim_partitioned(d) ? 1 : 0;
      std::unique_ptr<GaugeField> extended(createExtendedGauge(device, R));
      const auto plaq = plaquette(*extended);
      obs.plaq = static_cast<double>(plaq[0]);
      obs.plaq_spatial = static_cast<double>(plaq[1]);
      obs.plaq_temporal = static_cast<double>(plaq[2]);
    }

    void compute_plaq_linktr(const GaugeField &view, double &plaq, double &linktr)
    {
      GaugeObservables obs;
      compute_plaquettes_linktr(view, obs);
      plaq = obs.plaq;
      linktr = obs.linktr;
    }

    /**
       @brief MILC's NERSC checksum of a host gauge field (generic/nersc_cksum.c): the 32-bit patterns of the real
       and imaginary parts of rows 0 and 1 of every link, each converted to float, summed modulo 2^32
     */
    uint32_t nersc_checksum(const GaugeField &view)
    {
      const QudaPrecision prec = view.Precision();
      if (prec != QUDA_DOUBLE_PRECISION && prec != QUDA_SINGLE_PRECISION)
        errorQuda("The NERSC checksum needs a double or single precision host gauge field, not precision %d", prec);
      const size_t volume = view.Volume();
      const HostArrays gauge(view);
      // rows 0 and 1 of a row-major 3x3 complex matrix are its first 12 reals
      constexpr int row_reals = 12;

      // integer sums don't depend on the order, so the result doesn't depend on the threads or the ranks
      uint64_t sum = 0;
      for (int dir = 0; dir < 4; dir++) {
        const auto d = static_cast<const double *>(gauge.data[dir]);
        const auto f = static_cast<const float *>(gauge.data[dir]);
#pragma omp parallel for reduction(+ : sum)
        for (size_t s = 0; s < volume; s++) {
          for (int i = 0; i < row_reals; i++) {
            const float value = prec == QUDA_DOUBLE_PRECISION ? static_cast<float>(d[s * 18 + i]) : f[s * 18 + i];
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            sum += bits;
          }
        }
      }
      size_t total = sum;
      comm_allreduce_sum(total);
      return static_cast<uint32_t>(total);
    }

#ifdef HAVE_QIO
    scidac::GaugeInfo load_qio(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                               const FileIOParam &param)
    {
      scidac::GaugeInfo info;
      read_gauge_field(filename.c_str(), const_cast<void **>(gauge), prec, X, param.io_mode, param.io_partition_grid,
                       info);
      // QIO does not report the checksums stored in the file
      if (info.has_checksum && !info.checksum_matches)
        io::report_checksum_mismatch("Checksum mismatch in " + filename + ": computed " + io::checksum_str(info.sum),
                                     param.allow_checksum_mismatch);
      return info;
    }

    io::Checksum save_qio(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                          const FileIOParam &param, const scidac::GaugeInfo &info)
    {
      return write_gauge_field(filename.c_str(), const_cast<void **>(gauge), prec, X, param.partfile, param.io_mode,
                               param.io_partition_grid, info);
    }
#else
    scidac::GaugeInfo load_qio(const std::string &, void *const[], QudaPrecision, const int *, const FileIOParam &)
    {
      errorQuda("The external QIO backend has not been enabled");
      return {};
    }

    io::Checksum save_qio(const std::string &, const void *const[], QudaPrecision, const int *, const FileIOParam &,
                          const scidac::GaugeInfo &)
    {
      errorQuda("The external QIO backend has not been enabled");
      return {};
    }
#endif

    /**
       @brief Log what was loaded, for every format and backend
     */
    void log_loaded(const std::string &filename, const GaugeFileInfo &info, bool external)
    {
      const char *kind = info.format == QUDA_FILE_FORMAT_MILC ? "a MILC v5" : info.ildg ? "an ILDG" : "a SciDAC";
      logQuda(QUDA_SUMMARIZE, "Loaded %s %s precision gauge field from %s%s%s%s; checksums %s %s\n", kind,
              info.file_prec == QUDA_DOUBLE_PRECISION ? "double" : "single", filename.c_str(),
              external ? " with the external QIO library" : "", info.byte_swapped ? " (byte swapped)" : "",
              info.has_info_file ? " and its .info file" : "", io::checksum_str({info.sum29, info.sum31}).c_str(),
              !info.has_checksum      ? "(the file has none)" :
                info.checksum_matches ? "verified" :
                                        "NOT verified");
    }

    /**
       @brief Load a QIO or ILDG gauge field, checking the ILDG records and any usqcdInfo record
     */
    GaugeFileInfo load_lime(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                            const FileIOParam &param, bool require_ildg)
    {
      auto info = param.external_qio ? load_qio(filename, gauge, prec, X, param) :
                                       scidac::load_gauge(filename, gauge, prec, X, param);

      if (require_ildg && !info.ildg) errorQuda("File %s does not have the ILDG records", filename.c_str());

      GaugeFileInfo file_info;
      file_info.format = require_ildg ? QUDA_FILE_FORMAT_ILDG : QUDA_FILE_FORMAT_QIO;
      file_info.file_prec = info.file_prec;
      file_info.ildg = info.ildg;
      file_info.has_plaq_linktr = info.has_usqcd_info;
      file_info.plaq = info.plaq;
      file_info.linktr = info.linktr;
      file_info.has_checksum = info.has_checksum;
      file_info.checksum_matches = info.checksum_matches;
      file_info.sum29 = info.sum.sum29;
      file_info.sum31 = info.sum.sum31;
      file_info.metadata = info.metadata;
      return file_info;
    }

    /**
       @brief Log the metadata a load found, beyond what log_loaded gives
     */
    void log_metadata(const std::string &filename, const GaugeMetadata &m)
    {
      if (getVerbosity() < QUDA_VERBOSE) return;
      if (m.record_date)
        logQuda(QUDA_VERBOSE, "Record date of %s: \"%s\"\n", filename.c_str(), io::excerpt(*m.record_date).c_str());
      if (m.ildg_lfn)
        logQuda(QUDA_VERBOSE, "ILDG LFN of %s: \"%s\"\n", filename.c_str(), io::excerpt(*m.ildg_lfn).c_str());
      if (m.has_usqcd_info) {
        const std::string info
          = m.usqcd_info ? "with " + std::to_string(m.usqcd_info->size()) + " bytes of <info>" : "without <info>";
        logQuda(QUDA_VERBOSE, "%s has a usqcdInfo record %s\n", filename.c_str(), info.c_str());
      }
      if (!m.milc_info.empty()) logQuda(QUDA_VERBOSE, "%s.info has %lu lines\n", filename.c_str(), m.milc_info.size());
    }

    /**
       @brief Compare the plaquette and link trace recorded in a file's metadata with those of the loaded field
     */
    void check_plaq_linktr(const std::string &filename, const GaugeField &view, GaugeFileInfo &info)
    {
      const QudaPrecision prec = view.Precision();
      double plaq, linktr;
      compute_plaq_linktr(view, plaq, linktr);

      // MILC .info files print these values to six decimal places, and values computed from data
      // that has passed through single precision only agree to single precision
      double tol = info.format == QUDA_FILE_FORMAT_MILC                            ? 1e-5 :
        (info.file_prec == QUDA_DOUBLE_PRECISION && prec == QUDA_DOUBLE_PRECISION) ? 1e-12 :
                                                                                     1e-6;
      const char *source = info.format == QUDA_FILE_FORMAT_MILC ? ".info file" : "usqcdInfo record";

      info.plaq_linktr_matches = std::abs(plaq - info.plaq) <= tol && std::abs(linktr - info.linktr) <= tol;
      if (!info.plaq_linktr_matches)
        warningQuda("Mismatch with the %s of %s: plaquette %.16e, link trace %.16e; computed %.16e, %.16e", source,
                    filename.c_str(), info.plaq, info.linktr, plaq, linktr);
      else
        logQuda(QUDA_SUMMARIZE, "%s plaquette %.16e and link trace %.16e verified\n", source, info.plaq, info.linktr);
    }

    /**
       @brief Project the links of a host gauge field onto SU(3) with QUDA's polar projection, on the device
     */
    void project_polar(GaugeField &view)
    {
      auto field_param = host_field_param(view);
      GaugeField host(field_param);
      GaugeField device(device_field_param(field_param));
      device.copy(host);

      int *num_failures_h = static_cast<int *>(pool_host_pinned_malloc(sizeof(int)));
      int *num_failures_d = static_cast<int *>(get_mapped_device_pointer(num_failures_h));
      *num_failures_h = 0;
      projectSU3(device, device.toleranceSU3(), num_failures_d);
      size_t failures = *num_failures_h;
      pool_host_pinned_free(num_failures_h);
      comm_allreduce_sum(failures);
      if (failures > 0) errorQuda("The polar projection onto SU(3) failed for %lu links", failures);

      host.copy(device);
    }

    /**
       @brief Measure the deviation from unitarity of a loaded gauge field, and reunitarize it if requested
     */
    void reunitarize(const std::string &filename, GaugeField &view, QudaGaugeReunitarize method, GaugeFileInfo &info)
    {
      const QudaPrecision prec = view.Precision();
      const HostArrays gauge(view);
      const bool requested = method != QUDA_GAUGE_REUNITARIZE_NONE;
      const bool less_precise = info.file_prec < prec;
      if (!requested && !less_precise && getVerbosity() < QUDA_VERBOSE) return;

      if (method == QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_2 && prec != QUDA_DOUBLE_PRECISION)
        errorQuda("Reunitarizing as MILC's double-precision builds do needs a double-precision host gauge field");

      const size_t volume = view.Volume();

      // MILC's single-precision builds hold the links in single precision, and check them after rounding
      const bool round = method == QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_1 && prec == QUDA_DOUBLE_PRECISION;
      if (round) {
        for (int dir = 0; dir < 4; dir++) {
          auto u = static_cast<double *>(gauge.data[dir]);
#pragma omp parallel for
          for (size_t i = 0; i < volume * 18; i++) u[i] = static_cast<float>(u[i]);
        }
      }

      auto deviation = [&]() {
        double d = io::unitarity_deviation(gauge.data, prec, volume);
        comm_allreduce_max(d);
        return d;
      };
      info.unitarity_deviation = deviation();

      if (!requested) {
        const QudaVerbosity level = less_precise ? QUDA_SUMMARIZE : QUDA_VERBOSE;
        logQuda(level, "Largest deviation from unitarity of the links from %s: %.2e%s\n", filename.c_str(),
                info.unitarity_deviation,
                less_precise ? " (the file is less precise than the host field; set reunitarize to reunitarize it)" : "");
        return;
      }

      if (info.unitarity_deviation > io::milc_unitarity_tolerance)
        errorQuda("The gauge field from %s deviates from unitarity by up to %.2e, more than MILC's tolerance %.0e",
                  filename.c_str(), info.unitarity_deviation, io::milc_unitarity_tolerance);

      // MILC's single-precision builds check unitarity but never reunitarize (reload_lattice in generic/io_helpers.c)
      if (method == QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_1) {
        logQuda(QUDA_SUMMARIZE,
                "Gauge field from %s%s checked as MILC's single-precision builds do: largest deviation from "
                "unitarity %.2e; not reunitarized\n",
                filename.c_str(), round ? " rounded to single precision and" : "", info.unitarity_deviation);
        return;
      }

      switch (method) {
      case QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_2:
        io::reunitarize_gram_schmidt(gauge.data, prec, volume);
        break;
      case QUDA_GAUGE_REUNITARIZE_POLAR: project_polar(view); break;
      default: errorQuda("Unexpected reunitarization method %d", method);
      }

      info.reunitarized_deviation = deviation();
      logQuda(QUDA_SUMMARIZE,
              "Reunitarized the gauge field from %s (%s): largest deviation from unitarity %.2e changed to %.2e\n",
              filename.c_str(), get_gauge_reunitarize_str(method), info.unitarity_deviation, info.reunitarized_deviation);
    }

    /**
       @brief Save a QIO or ILDG gauge field, with a usqcdInfo record if requested, and the metadata given
     */
    io::Checksum save_lime(const std::string &filename, const GaugeField &view, const FileIOParam &param, bool ildg,
                           const GaugeMetadata &metadata)
    {
      const HostArrays gauge(view);
      const QudaPrecision prec = view.Precision();
      const int *X = view.X().data;
      scidac::GaugeInfo info;
      info.legacy_strings = param.legacy_strings;
      info.ildg = ildg;
      if (ildg) {
        if (metadata.ildg_lfn && !metadata.ildg_lfn->empty())
          info.ildg_lfn = *metadata.ildg_lfn;
        else
          info.ildg_lfn = param.ildg_lfn.empty() ? filename : param.ildg_lfn;
      }
      info.user_file_xml = metadata.user_file_xml;
      info.date = metadata.record_date;
      if (!metadata.has_usqcd_info && !metadata.user_record_xml.empty()) {
        info.user_record_xml = metadata.user_record_xml;
        if (param.write_usqcd_info)
          logQuda(QUDA_VERBOSE, "The record XML given for %s is written as it is, so no usqcdInfo record is\n",
                  filename.c_str());
      } else if (param.write_usqcd_info) {
        compute_plaq_linktr(view, info.plaq, info.linktr);
        info.has_usqcd_info = true;
        if (metadata.usqcd_info) info.usqcd_info = *metadata.usqcd_info;
        logQuda(QUDA_SUMMARIZE, "usqcdInfo plaquette %.16e, link trace %.16e\n", info.plaq, info.linktr);
      } else if (metadata.has_usqcd_info) {
        logQuda(QUDA_VERBOSE, "write_usqcd_info is not set, so the usqcdInfo record given for %s is not written\n",
                filename.c_str());
      }
      return param.external_qio ? save_qio(filename, gauge.data, prec, X, param, info) :
                                  scidac::save_gauge(filename, gauge.data, prec, X, param, info);
    }

    /**
       @brief The host field the formats save from for a field: the field itself when it already is one and needs
       no conversion, otherwise a copy with the conversion's signs removed
     */
    class SaveView
    {
      std::optional<GaugeField> copy;
      const GaugeField *view;

    public:
      SaveView(const GaugeField &u, QudaPrecision prec, const LinkConventions &c) : view(&u)
      {
        if (is_file_view(u, prec) && !c.any()) return;
        copy.emplace(file_view_param(u, prec));
        copy_to_file_view(*copy, u);
        if (c.any()) apply_conventions(*copy, c);
        view = &*copy;
      }
      const GaugeField &operator*() const { return *view; }
    };
  } // namespace

  GaugeObservables gauge_observables(const GaugeField &u, bool convert_link_conventions)
  {
    check_field(u);
    const auto c = link_conventions(u, convert_link_conventions);
    if (c.any()) printfQuda("Computing the observables of the plain links: removed %s\n", c.str().c_str());
    const SaveView view(u, host_precision(u.Precision()), c);

    GaugeObservables obs;
    compute_plaquettes_linktr(*view, obs);
    obs.nersc_checksum = nersc_checksum(*view);
    return obs;
  }

  GaugeIO::GaugeIO(const std::string &filename, const FileIOParam &param) : filename(filename), param(param)
  {
    if (filename.empty()) errorQuda("No gauge field file defined");
  }

  GaugeFileInfo GaugeIO::load(GaugeField &u)
  {
    check_field(u);
    if (u.GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED)
      errorQuda("Loading %s into an extended gauge field is not supported", filename.c_str());
    const auto c = link_conventions(u, param.convert_link_conventions);
    const QudaPrecision prec = host_precision(u.Precision());

    auto format = resolve_load_format(filename, param.format);
    logQuda(QUDA_SUMMARIZE, "Start loading %s gauge field from %s\n", get_file_format_str(format), filename.c_str());

    // a host field with a ghost zone is loaded through a copy, which exchanges it
    std::optional<GaugeField> copy;
    const bool in_place = is_file_view(u, prec) && u.GhostExchange() == QUDA_GHOST_EXCHANGE_NO;
    if (!in_place) copy.emplace(file_view_param(u, prec));
    GaugeField &view = in_place ? u : *copy;
    const HostArrays gauge(view);
    const int *X = view.X().data;

    host_timer_t host_timer;
    host_timer.start();

    GaugeFileInfo info;
    switch (format) {
    case QUDA_FILE_FORMAT_QIO: info = load_lime(filename, gauge.data, prec, X, param, false); break;
    case QUDA_FILE_FORMAT_ILDG: info = load_lime(filename, gauge.data, prec, X, param, true); break;
    case QUDA_FILE_FORMAT_MILC: info = milc::load_gauge(filename, gauge.data, prec, X, param); break;
    default: errorQuda("Unexpected file format %d", format);
    }

    host_timer.stop();
    log_loaded(filename, info, param.external_qio && format != QUDA_FILE_FORMAT_MILC);
    log_metadata(filename, info.metadata);
    logQuda(QUDA_SUMMARIZE, "Time spent loading gauge field from %s = %g secs\n", filename.c_str(), host_timer.last());

    if (info.has_plaq_linktr) check_plaq_linktr(filename, view, info);
    reunitarize(filename, view, param.reunitarize, info);

    if (c.any()) {
      apply_conventions(view, c);
      printfQuda("Applied %s to the links loaded from %s\n", c.str().c_str(), filename.c_str());
    }
    if (!in_place) u.copy(view);
    return info;
  }

  void GaugeIO::save(const GaugeField &u, QudaPrecision prec, const GaugeMetadata &metadata)
  {
    check_field(u);
    if (prec != QUDA_INVALID_PRECISION && prec != QUDA_DOUBLE_PRECISION && prec != QUDA_SINGLE_PRECISION)
      errorQuda("%s: gauge fields are saved from double or single precision, not precision %d", filename.c_str(), prec);
    const QudaPrecision save_prec = prec != QUDA_INVALID_PRECISION ? prec : host_precision(u.Precision());

    auto format = resolve_save_format(param.format);
    if (param.partfile && format == QUDA_FILE_FORMAT_ILDG)
      errorQuda("%s: ILDG files cannot be saved as partfiles", filename.c_str());
    if (param.partfile && format == QUDA_FILE_FORMAT_MILC)
      errorQuda("%s: MILC's gauge format has no partfiles", filename.c_str());
    const auto c = link_conventions(u, param.convert_link_conventions);
    logQuda(QUDA_SUMMARIZE, "Start saving %s gauge field to %s%s\n", get_file_format_str(format), filename.c_str(),
            param.partfile ? " as a partfile" : "");

    const SaveView view(u, save_prec, c);
    if (c.any()) printfQuda("Removed %s from the links saved to %s\n", c.str().c_str(), filename.c_str());

    host_timer_t host_timer;
    host_timer.start();

    io::Checksum sum;
    switch (format) {
    case QUDA_FILE_FORMAT_QIO: sum = save_lime(filename, *view, param, false, metadata); break;
    case QUDA_FILE_FORMAT_ILDG: sum = save_lime(filename, *view, param, true, metadata); break;
    case QUDA_FILE_FORMAT_MILC: {
      const HostArrays gauge(*view);
      sum = milc::save_gauge(filename, gauge.data, save_prec, (*view).X().data, param, metadata);
    } break;
    default: errorQuda("Unexpected file format %d", format);
    }
    // rank 0 finishes the file (checksum record or header, .info file) after the other ranks are done
    comm_barrier();

    host_timer.stop();
    const bool external = param.external_qio && format != QUDA_FILE_FORMAT_MILC;
    logQuda(QUDA_SUMMARIZE, "Saved %s gauge field to %s%s; checksums %s\n", get_file_format_str(format),
            filename.c_str(), external ? " with the external QIO library" : "", io::checksum_str(sum).c_str());
    logQuda(QUDA_SUMMARIZE, "Time spent saving gauge field to %s = %g secs\n", filename.c_str(), host_timer.last());
  }

} // namespace quda
