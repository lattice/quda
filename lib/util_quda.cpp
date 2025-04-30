#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stack>
#include <sstream>
#include <sys/time.h>

#include <enum_quda.h>
#include <util_quda.h>
#include <malloc_quda.h>
#include <tune_quda.h>

using namespace quda;

static const size_t MAX_PREFIX_SIZE = 100;

static QudaVerbosity verbosity_ = QUDA_SUMMARIZE;
static char prefix_[MAX_PREFIX_SIZE] = "";
static FILE *outfile_ = stdout;

static const int MAX_BUFFER_SIZE = 1000;
static char buffer_[MAX_BUFFER_SIZE] = "";

QudaVerbosity getVerbosity() { return verbosity_; }
char *getOutputPrefix() { return prefix_; }
FILE *getOutputFile() { return outfile_; }

void setVerbosity(QudaVerbosity verbosity)
{
  verbosity_ = verbosity;
}

bool getRankVerbosity() {
  static bool init = false;
  static bool rank_verbosity = false;
  static char *rank_verbosity_env = getenv("QUDA_RANK_VERBOSITY");

  if (!init && rank_verbosity_env) { // set the policies to tune for explicitly
    std::stringstream rank_list(rank_verbosity_env);

    int rank_;
    while (rank_list >> rank_) {
      if (comm_rank() == rank_ || rank_ == -1) rank_verbosity = true;
      if (rank_list.peek() == ',') rank_list.ignore();
    }
  } else if (!init) {
    rank_verbosity = comm_rank() == 0 ? true : false; // default is process 0 only
  }
  init = true;

  return rank_verbosity;
}

static bool tune = true;

// default has autotuning enabled but can be overridden with the QUDA_ENABLE_TUNING environment variable
bool getTuning()
{
  static bool init = false;
  if (!init) {
    char *enable_tuning = getenv("QUDA_ENABLE_TUNING");
    if (!enable_tuning || strcmp(enable_tuning, "0") != 0) {
      tune = true;
    } else {
      tune = false;
    }
    init = true;
  }

  return tune;
}

void setTuning(bool tuning)
{
  // first check if tuning is disabled, in which case we do nothing
  static bool init = false;
  static bool tune_disable = false;
  if (!init) {
    char *enable_tuning = getenv("QUDA_ENABLE_TUNING");
    tune_disable = (enable_tuning && strcmp(enable_tuning, "0") == 0);
    init = true;
  }
  if (!tune_disable) tune = tuning;
}

static std::stack<bool> tstack;

void pushTuning(bool tuning)
{
  tstack.push(getTuning());
  setTuning(tuning);
}

void popTuning()
{
  if (tstack.empty()) errorQuda("popTuning() called with empty stack");
  setTuning(tstack.top());
  tstack.pop();
}

void setOutputPrefix(const char *prefix)
{
  strncpy(prefix_, prefix, MAX_PREFIX_SIZE);
  prefix_[MAX_PREFIX_SIZE-1] = '\0';
}

void setOutputFile(FILE *outfile)
{
  outfile_ = outfile;
}

static std::stack<QudaVerbosity> vstack;

void pushVerbosity(QudaVerbosity verbosity)
{
  vstack.push(getVerbosity());
  setVerbosity(verbosity);

  if (vstack.size() > 15) {
    warningQuda("Verbosity stack contains %u elements.  Is there a missing popVerbosity() somewhere?",
		static_cast<unsigned int>(vstack.size()));
  }
}

void popVerbosity()
{
  if (vstack.empty()) {
    errorQuda("popVerbosity() called with empty stack");
  }
  setVerbosity(vstack.top());
  vstack.pop();
}

static std::stack<char *> pstack;

void pushOutputPrefix(const char *prefix)
{
  // backup current prefix onto the stack
  char *prefix_backup = (char *)safe_malloc(MAX_PREFIX_SIZE * sizeof(char));
  strncpy(prefix_backup, getOutputPrefix(), MAX_PREFIX_SIZE);
  pstack.push(prefix_backup);

  // set new prefix
  setOutputPrefix(prefix);

  if (pstack.size() > 15) {
    warningQuda("Verbosity stack contains %u elements.  Is there a missing popOutputPrefix() somewhere?",
                static_cast<unsigned int>(vstack.size()));
  }
}

void popOutputPrefix()
{
  if (pstack.empty()) { errorQuda("popOutputPrefix() called with empty stack"); }

  // recover prefix from stack
  char *prefix_restore = pstack.top();
  setOutputPrefix(prefix_restore);
  host_free(prefix_restore);
  pstack.pop();
}

char *getPrintBuffer() { return buffer_; }

const char *getOmpThreadStr()
{
  static std::string omp_thread_string;
  static bool init = false;
  if (!init) {
#ifdef QUDA_OPENMP
    omp_thread_string = std::string("omp_threads=" + std::to_string(omp_get_max_threads()) + ",");
#endif
    init = true;
  }
  return omp_thread_string.c_str();
}

void errorQuda_(const char *func, const char *file, int line, ...)
{
  fprintf(getOutputFile(), " (rank %d, host %s, %s:%d in %s())\n", comm_rank_global(), comm_hostname(), file, line, func);
  fprintf(getOutputFile(), "%s       last kernel called was (name=%s,volume=%s,aux=%s)\n", getOutputPrefix(),
          quda::getLastTuneKey().name, quda::getLastTuneKey().volume, quda::getLastTuneKey().aux);
  fflush(getOutputFile());
  quda::saveTuneCache(true);
  comm_abort(1);
}

namespace quda
{

  unsigned int get_max_multi_rhs()
  {
    static bool init = false;
    static int max = MAX_MULTI_RHS;

    if (!init) {
      char *max_str = getenv("QUDA_MAX_MULTI_RHS");
      if (max_str) {
        max = atoi(max_str);
        if (max <= 0) errorQuda("QUDA_MAX_MULTI_RHS=%d cannot be negative", max);
        if (max > MAX_MULTI_RHS)
          errorQuda("QUDA_MAX_MULTI_RHS=%d cannot be greater than CMake set value %u", max, MAX_MULTI_RHS);
        printfQuda("QUDA_MAX_MULTI_RHS set to %d\n", max);
      }
      init = true;
    }

    return max;
  }

} // namespace quda
