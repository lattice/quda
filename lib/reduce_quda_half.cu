#define REDUCE_SUFFIX half
#define REDUCE_PREC QUDA_HALF_PRECISION
#define REDUCE_STORE short
#include "reduce_quda_inst.hpp"
#undef REDUCE_STORE
#undef REDUCE_PREC
#undef REDUCE_SUFFIX
