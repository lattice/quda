#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sys/time.h>

#include <enum_quda.h>
#include <util_quda.h>


static const size_t MAX_PREFIX_SIZE = 100;

static QudaVerbosity verbosity_ = QUDA_SUMMARIZE;
static char prefix_[MAX_PREFIX_SIZE] = "";
static FILE *outfile_ = stdout;

QudaVerbosity getVerbosity() { return verbosity_; }
char *getOutputPrefix() { return prefix_; }
FILE *getOutputFile() { return outfile_; }

void setVerbosity(const QudaVerbosity verbosity)
{
  verbosity_ = verbosity;
};

void setOutputPrefix(const char *prefix)
{
  strncpy(prefix_, prefix, MAX_PREFIX_SIZE);
  prefix_[MAX_PREFIX_SIZE-1] = '\0';
};

void setOutputFile(FILE *outfile)
{
  outfile_ = outfile;
};
