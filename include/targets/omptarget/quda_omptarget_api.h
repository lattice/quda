#pragma once

namespace quda {
  namespace target {
    namespace omptarget {
      int qudaSetupLaunchParameter(const TuneParam &);
      void set_runtime_error(int error, const char *api_func, const char *func, const char *file,
                             const char *line, bool allow_error = false);
    }
  }
}
