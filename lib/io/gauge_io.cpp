#include <cmath>
#include <cstdio>
#include <memory>
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
    /**
       @brief Parameters for a reference field wrapping host arrays in QDP order
     */
    GaugeFieldParam host_field_param(const void *const gauge[], QudaPrecision prec, const int *X)
    {
      QudaGaugeParam gauge_param = newQudaGaugeParam();
      for (int d = 0; d < 4; d++) gauge_param.X[d] = X[d];
      gauge_param.type = QUDA_GENERAL_LINKS;
      gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
      gauge_param.cpu_prec = prec;
      gauge_param.t_boundary = QUDA_PERIODIC_T;
      gauge_param.anisotropy = 1.0;
      gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

      GaugeFieldParam field_param(gauge_param, const_cast<void **>(gauge));
      field_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      return field_param;
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
       @brief Compute the average plaquette and link trace, both normalized as Re Tr / 3
     */
    void compute_plaq_linktr(const void *const gauge[], QudaPrecision prec, const int *X, double &plaq, double &linktr)
    {
      auto field_param = host_field_param(gauge, prec, X);
      GaugeField host(field_param);
      GaugeField device(device_field_param(field_param));
      device.copy(host);

      // getLinkTrace averages Tr U over the links of a field without a halo
      linktr = getLinkTrace(device).real() / 3.0;

      lat_dim_t R;
      for (int d = 0; d < 4; d++) R[d] = comm_dim_partitioned(d) ? 1 : 0;
      std::unique_ptr<GaugeField> extended(createExtendedGauge(device, R));
      plaq = static_cast<double>(plaquette(*extended)[0]);
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
      return write_gauge_field(filename.c_str(), const_cast<void **>(gauge), prec, X, param.io_mode,
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
      return file_info;
    }

    /**
       @brief Compare the plaquette and link trace recorded in a file's metadata with those of the loaded field
     */
    void check_plaq_linktr(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                           GaugeFileInfo &info)
    {
      double plaq, linktr;
      compute_plaq_linktr(gauge, prec, X, plaq, linktr);

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
    void project_polar(void *const gauge[], QudaPrecision prec, const int *X)
    {
      auto field_param = host_field_param(gauge, prec, X);
      GaugeField host(field_param);
      GaugeField device(device_field_param(field_param));
      device.copy(host);

      int *num_failures_h = static_cast<int *>(pool_host_pinned_malloc(sizeof(int)));
      int *num_failures_d = static_cast<int *>(get_mapped_device_pointer(num_failures_h));
      *num_failures_h = 0;
      projectSU3(device, device.toleranceSU3(), num_failures_d);
      int failures = *num_failures_h;
      pool_host_pinned_free(num_failures_h);
      comm_allreduce_int(failures);
      if (failures > 0) errorQuda("The polar projection onto SU(3) failed for %d links", failures);

      host.copy(device);
    }

    /**
       @brief Measure the deviation from unitarity of a loaded gauge field, and reunitarize it if requested
     */
    void reunitarize(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                     QudaGaugeReunitarize method, GaugeFileInfo &info)
    {
      const bool requested = method != QUDA_GAUGE_REUNITARIZE_NONE;
      const bool less_precise = info.file_prec < prec;
      if (!requested && !less_precise && getVerbosity() < QUDA_VERBOSE) return;

      if (method == QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_2 && prec != QUDA_DOUBLE_PRECISION)
        errorQuda("Reunitarizing as MILC's double-precision builds do needs a double-precision host gauge field");

      const size_t volume = static_cast<size_t>(X[0]) * X[1] * X[2] * X[3];

      // MILC's single-precision builds hold the links in single precision, and check them after rounding
      const bool round = method == QUDA_GAUGE_REUNITARIZE_MILC_GRAM_SCHMIDT_PRECISION_1 && prec == QUDA_DOUBLE_PRECISION;
      if (round) {
        for (int dir = 0; dir < 4; dir++) {
          auto u = static_cast<double *>(gauge[dir]);
#pragma omp parallel for
          for (size_t i = 0; i < volume * 18; i++) u[i] = static_cast<float>(u[i]);
        }
      }

      auto deviation = [&]() {
        double d = io::unitarity_deviation(gauge, prec, volume);
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
        io::reunitarize_gram_schmidt(gauge, prec, volume);
        break;
      case QUDA_GAUGE_REUNITARIZE_POLAR: project_polar(gauge, prec, X); break;
      default: errorQuda("Unexpected reunitarization method %d", method);
      }

      info.reunitarized_deviation = deviation();
      logQuda(QUDA_SUMMARIZE,
              "Reunitarized the gauge field from %s (%s): largest deviation from unitarity %.2e changed to %.2e\n",
              filename.c_str(), get_gauge_reunitarize_str(method), info.unitarity_deviation, info.reunitarized_deviation);
    }

    /**
       @brief Save a QIO or ILDG gauge field, with a usqcdInfo record if requested
     */
    io::Checksum save_lime(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                           const FileIOParam &param, bool ildg)
    {
      scidac::GaugeInfo info;
      info.ildg = ildg;
      if (ildg) info.ildg_lfn = param.ildg_lfn.empty() ? filename : param.ildg_lfn;
      if (param.write_usqcd_info) {
        compute_plaq_linktr(gauge, prec, X, info.plaq, info.linktr);
        info.has_usqcd_info = true;
        logQuda(QUDA_SUMMARIZE, "usqcdInfo plaquette %.16e, link trace %.16e\n", info.plaq, info.linktr);
      }
      return param.external_qio ? save_qio(filename, gauge, prec, X, param, info) :
                                  scidac::save_gauge(filename, gauge, prec, X, param, info);
    }
  } // namespace

  GaugeIO::GaugeIO(const std::string &filename, const FileIOParam &param) : filename(filename), param(param)
  {
    if (filename.empty()) errorQuda("No gauge field file defined");
  }

  GaugeFileInfo GaugeIO::load(void *const gauge[], QudaPrecision prec, const int *X)
  {
    auto format = resolve_load_format(filename, param.format);
    logQuda(QUDA_SUMMARIZE, "Start loading %s gauge field from %s\n", get_file_format_str(format), filename.c_str());

    host_timer_t host_timer;
    host_timer.start();

    GaugeFileInfo info;
    switch (format) {
    case QUDA_FILE_FORMAT_QIO: info = load_lime(filename, gauge, prec, X, param, false); break;
    case QUDA_FILE_FORMAT_ILDG: info = load_lime(filename, gauge, prec, X, param, true); break;
    case QUDA_FILE_FORMAT_MILC: info = milc::load_gauge(filename, gauge, prec, X, param); break;
    default: errorQuda("Unexpected file format %d", format);
    }

    host_timer.stop();
    log_loaded(filename, info, param.external_qio && format != QUDA_FILE_FORMAT_MILC);
    logQuda(QUDA_SUMMARIZE, "Time spent loading gauge field from %s = %g secs\n", filename.c_str(), host_timer.last());

    if (info.has_plaq_linktr) check_plaq_linktr(filename, gauge, prec, X, info);
    reunitarize(filename, gauge, prec, X, param.reunitarize, info);
    return info;
  }

  void GaugeIO::save(const void *const gauge[], QudaPrecision prec, const int *X)
  {
    auto format = resolve_save_format(param.format);
    logQuda(QUDA_SUMMARIZE, "Start saving %s gauge field to %s\n", get_file_format_str(format), filename.c_str());

    host_timer_t host_timer;
    host_timer.start();

    io::Checksum sum;
    switch (format) {
    case QUDA_FILE_FORMAT_QIO: sum = save_lime(filename, gauge, prec, X, param, false); break;
    case QUDA_FILE_FORMAT_ILDG: sum = save_lime(filename, gauge, prec, X, param, true); break;
    case QUDA_FILE_FORMAT_MILC: sum = milc::save_gauge(filename, gauge, prec, X, param); break;
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
