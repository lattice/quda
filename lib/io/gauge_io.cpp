#include <cmath>
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

namespace quda
{

  namespace
  {
    /**
       @brief Compute the average plaquette and link trace, both normalized as Re Tr / 3
     */
    void compute_plaq_linktr(const void *const gauge[], QudaPrecision prec, const int *X, double &plaq, double &linktr)
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
      GaugeField host(field_param);

      field_param.create = QUDA_NULL_FIELD_CREATE;
      field_param.reconstruct = QUDA_RECONSTRUCT_NO;
      field_param.setPrecision(prec, true);
      field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
      field_param.location = QUDA_CUDA_FIELD_LOCATION;
      GaugeField device(field_param);
      device.copy(host);

      // getLinkTrace averages Tr U over the links of a field without a halo
      linktr = getLinkTrace(device).real() / 3.0;

      lat_dim_t R;
      for (int d = 0; d < 4; d++) R[d] = comm_dim_partitioned(d) ? 1 : 0;
      std::unique_ptr<GaugeField> extended(createExtendedGauge(device, R));
      plaq = static_cast<double>(plaquette(*extended)[0]);
    }

#ifdef HAVE_QIO
    void load_qio(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                  QudaFileIOMode io_mode, QIOGaugeInfo &info)
    {
      read_gauge_field(filename.c_str(), const_cast<void **>(gauge), prec, X, io_mode, info);
    }

    void save_qio(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                  QudaFileIOMode io_mode, const QIOGaugeInfo &info)
    {
      write_gauge_field(filename.c_str(), const_cast<void **>(gauge), prec, X, io_mode, info);
    }
#else
    void load_qio(const std::string &, void *const[], QudaPrecision, const int *, QudaFileIOMode, QIOGaugeInfo &)
    {
      errorQuda("QIO support has not been enabled");
    }

    void save_qio(const std::string &, const void *const[], QudaPrecision, const int *, QudaFileIOMode,
                  const QIOGaugeInfo &)
    {
      errorQuda("QIO support has not been enabled");
    }
#endif

    /**
       @brief Load a QIO or ILDG gauge field, checking the ILDG records and any usqcdInfo record
     */
    GaugeFileInfo load_lime(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                            const FileIOParam &param, bool require_ildg)
    {
      QIOGaugeInfo info;
      load_qio(filename, gauge, prec, X, param.io_mode, info);

      if (require_ildg && !info.ildg) errorQuda("File %s does not have the ILDG records", filename.c_str());
      logQuda(QUDA_SUMMARIZE, "Loaded %s %s gauge field\n", info.ildg ? "an ILDG" : "a SciDAC",
              info.file_prec == QUDA_DOUBLE_PRECISION ? "double precision" : "single precision");

      GaugeFileInfo file_info;
      file_info.format = require_ildg ? QUDA_FILE_FORMAT_ILDG : QUDA_FILE_FORMAT_QIO;
      file_info.file_prec = info.file_prec;
      file_info.ildg = info.ildg;
      file_info.has_plaq_linktr = info.has_usqcd_info;
      file_info.plaq = info.plaq;
      file_info.linktr = info.linktr;
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
       @brief Save a QIO or ILDG gauge field, with a usqcdInfo record if requested
     */
    void save_lime(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                   const FileIOParam &param, bool ildg)
    {
      QIOGaugeInfo info;
      info.ildg = ildg;
      if (ildg) info.ildg_lfn = param.ildg_lfn.empty() ? filename : param.ildg_lfn;
      if (param.write_usqcd_info) {
        compute_plaq_linktr(gauge, prec, X, info.plaq, info.linktr);
        info.has_usqcd_info = true;
        logQuda(QUDA_SUMMARIZE, "usqcdInfo plaquette %.16e, link trace %.16e\n", info.plaq, info.linktr);
      }
      save_qio(filename, gauge, prec, X, param.io_mode, info);
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
    logQuda(QUDA_SUMMARIZE, "Time spent loading gauge field from %s = %g secs\n", filename.c_str(), host_timer.last());

    if (info.has_plaq_linktr) check_plaq_linktr(filename, gauge, prec, X, info);
    return info;
  }

  void GaugeIO::save(const void *const gauge[], QudaPrecision prec, const int *X)
  {
    auto format = resolve_save_format(param.format);
    logQuda(QUDA_SUMMARIZE, "Start saving %s gauge field to %s\n", get_file_format_str(format), filename.c_str());

    host_timer_t host_timer;
    host_timer.start();

    switch (format) {
    case QUDA_FILE_FORMAT_QIO: save_lime(filename, gauge, prec, X, param, false); break;
    case QUDA_FILE_FORMAT_ILDG: save_lime(filename, gauge, prec, X, param, true); break;
    case QUDA_FILE_FORMAT_MILC: milc::save_gauge(filename, gauge, prec, X, param); break;
    default: errorQuda("Unexpected file format %d", format);
    }

    host_timer.stop();
    logQuda(QUDA_SUMMARIZE, "Time spent saving gauge field to %s = %g secs\n", filename.c_str(), host_timer.last());
  }

} // namespace quda
