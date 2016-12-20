#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stack>
#include <sys/time.h>

#include <enum_quda.h>
#include <util_quda.h>


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
