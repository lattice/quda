#define REDUCE_SUFFIX single
#define REDUCE_PREC QUDA_SINGLE_PRECISION
#define REDUCE_STORE float
#include "reduce_quda_inst.hpp"
#undef REDUCE_STORE
#undef REDUCE_PREC
#undef REDUCE_SUFFIX
