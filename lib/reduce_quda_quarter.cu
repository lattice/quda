#define REDUCE_SUFFIX quarter
#define REDUCE_PREC QUDA_QUARTER_PRECISION
#define REDUCE_STORE int8_t
#include "reduce_quda_inst.hpp"
#undef REDUCE_STORE
#undef REDUCE_PREC
#undef REDUCE_SUFFIX
