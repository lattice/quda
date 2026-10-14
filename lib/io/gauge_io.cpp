#include <util_quda.h>
#include <timer.h>
#include <io/gauge_io.h>
#include <io/qio_field.h>

namespace quda
{

  namespace
  {
#ifdef HAVE_QIO
    void load_qio(const std::string &filename, void *const gauge[], QudaPrecision prec, const int *X,
                  QudaFileIOMode io_mode)
    {
      read_gauge_field(filename.c_str(), const_cast<void **>(gauge), prec, X, io_mode);
    }

    void save_qio(const std::string &filename, const void *const gauge[], QudaPrecision prec, const int *X,
                  QudaFileIOMode io_mode)
    {
      write_gauge_field(filename.c_str(), const_cast<void **>(gauge), prec, X, io_mode);
    }
#else
    void load_qio(const std::string &, void *const[], QudaPrecision, const int *, QudaFileIOMode)
    {
      errorQuda("QIO support has not been enabled");
    }

    void save_qio(const std::string &, const void *const[], QudaPrecision, const int *, QudaFileIOMode)
    {
      errorQuda("QIO support has not been enabled");
    }
#endif
  } // namespace

  GaugeIO::GaugeIO(const std::string &filename, const FileIOParam &param) : filename(filename), param(param)
  {
    if (filename.empty()) errorQuda("No gauge field file defined");
  }

  void GaugeIO::load(void *const gauge[], QudaPrecision prec, const int *X)
  {
    auto format = resolve_load_format(filename, param.format);
    logQuda(QUDA_SUMMARIZE, "Start loading %s gauge field from %s\n", get_file_format_str(format), filename.c_str());

    host_timer_t host_timer;
    host_timer.start();

    switch (format) {
    case QUDA_FILE_FORMAT_QIO: load_qio(filename, gauge, prec, X, param.io_mode); break;
    case QUDA_FILE_FORMAT_MILC: errorQuda("MILC gauge field I/O is not yet supported"); break;
    case QUDA_FILE_FORMAT_ILDG: errorQuda("ILDG gauge field I/O is not yet supported"); break;
    default: errorQuda("Unexpected file format %d", format);
    }

    host_timer.stop();
    logQuda(QUDA_SUMMARIZE, "Time spent loading gauge field from %s = %g secs\n", filename.c_str(), host_timer.last());
  }

  void GaugeIO::save(const void *const gauge[], QudaPrecision prec, const int *X)
  {
    auto format = resolve_save_format(param.format);
    logQuda(QUDA_SUMMARIZE, "Start saving %s gauge field to %s\n", get_file_format_str(format), filename.c_str());

    host_timer_t host_timer;
    host_timer.start();

    switch (format) {
    case QUDA_FILE_FORMAT_QIO: save_qio(filename, gauge, prec, X, param.io_mode); break;
    case QUDA_FILE_FORMAT_MILC: errorQuda("MILC gauge field I/O is not yet supported"); break;
    case QUDA_FILE_FORMAT_ILDG: errorQuda("ILDG gauge field I/O is not yet supported"); break;
    default: errorQuda("Unexpected file format %d", format);
    }

    host_timer.stop();
    logQuda(QUDA_SUMMARIZE, "Time spent saving gauge field to %s = %g secs\n", filename.c_str(), host_timer.last());
  }

} // namespace quda
