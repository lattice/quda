#include <stdlib.h>
#include <sys/time.h>

#include <util_quda.h>

static struct timeval startTime;

void stopwatchStart() {
  gettimeofday(&startTime, NULL);
}

double stopwatchReadSeconds() {
  struct timeval endTime;
  gettimeofday(&endTime, NULL);

  long ds = endTime.tv_sec - startTime.tv_sec;
  long dus = endTime.tv_usec - startTime.tv_usec;
  return ds + 0.000001*dus;
}
