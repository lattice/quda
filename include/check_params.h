#pragma once

#include <quda_internal.h>
#include <quda.h>

#include <float.h>
#define INVALID_INT QUDA_INVALID_ENUM
#define INVALID_DOUBLE DBL_MIN

enum paramType { INIT, CHECK, PRINT };

// define macro to carry out the appropriate action for a given parameter
//#if defined INIT_PARAM
#define P(param, x, val, type) { if(type == INIT) { param->x = val; }	\
    else if(type == CHECK) {						\
      if (param->x == val) errorQuda("Parameter " #x " undefined");	\
    }									\
    else if(type == PRINT) {						\
      if ((double)val == INVALID_DOUBLE) {				\
	printfQuda(#x " = %g\n", (double)param->x);			\
      }									\
      else printfQuda(#x " = %d\n", (int)param->x);			\
    }									\
  }


void parseQudaGaugeParam(QudaGaugeParam *param, paramType type);
void checkGaugeParam(QudaGaugeParam *param);

void parseQudaEigParam(QudaEigParam *param, paramType type);
void checkEigParam(QudaEigParam *param);

void parseQudaCloverParam(QudaInvertParam *param, paramType type);
void checkCloverParam(QudaInvertParam *param);

void parseQudaInvertParam(QudaInvertParam *param, paramType type);
void checkInvertParam(QudaInvertParam *param, void *out_ptr=nullptr, void *in_ptr=nullptr);

void parseQudaMultigridParam(QudaMultigridParam *param, paramType type);
void checkMultigridParam(QudaMultigridParam *param);

void parseQudaGaugeObservableParam(QudaGaugeObservableParam *param, paramType type);
void checkGaugeObservableParam(QudaGaugeObservableParam *param);

void parseQudaCublasParam(QudaCublasParam *param, paramType type);
void checkCublasParam(QudaCublasParam *param);
