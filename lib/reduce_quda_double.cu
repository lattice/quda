#define REDUCE_SUFFIX double
#define REDUCE_PREC QUDA_DOUBLE_PRECISION
#define REDUCE_STORE double
#include "reduce_quda_inst.hpp"
#undef REDUCE_STORE
#undef REDUCE_PREC
#undef REDUCE_SUFFIX
