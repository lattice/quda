#ifndef _UTIL_QUDA_H
#define _UTIL_QUDA_H

#include <quda_internal.h>

#ifdef __cplusplus
extern "C" {
#endif
  
  void stopwatchStart();
  double stopwatchReadSeconds();

#ifdef __cplusplus
}
#endif

#endif // _UTIL_QUDA_H
