#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stack>
#include <sys/time.h>

#include <enum_quda.h>
#include <util_quda.h>
#include <sstream>

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

// default has autotuning enabled but can be overridden with the QUDA_ENABLE_TUNING environment variable
QudaTune getTuning() {
  static bool init = false;
  static QudaTune tune = QUDA_TUNE_YES;

  if (!init) {
    char *enable_tuning = getenv("QUDA_ENABLE_TUNING");
    if (!enable_tuning || strcmp(enable_tuning,"0")!=0) {
      tune = QUDA_TUNE_YES;
    } else {
      tune = QUDA_TUNE_NO;
    }
    init = true;
  }

  return tune;
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

  if (vstack.size() > 10) {
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

char *getPrintBuffer() { return buffer_; }

char* getOmpThreadStr() {
  static char omp_thread_string[128];
  static bool init = false;
  if (!init) {
    strcpy(omp_thread_string,",omp_threads=");
    char *omp_threads = getenv("OMP_NUM_THREADS");
    strcat(omp_thread_string, omp_threads ? omp_threads : "1");
    init = true;
  }
  return omp_thread_string;
}
