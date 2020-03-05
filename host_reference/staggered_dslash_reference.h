
#ifndef _STAGGERED_QUDA_DSLASH_REF_H
#define _STAGGERED_QUDA_DSLASH_REF_H
#include <blas_reference.h>
#include <quda_internal.h>
#include "color_spinor_field.h"

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;

void setDims(int *);

void staggered_dslash(cpuColorSpinorField *out, void **fatlink, void **longlink, void **ghost_fatlink,
    void **ghost_longlink, cpuColorSpinorField *in, int oddBit, int daggerBit, QudaPrecision sPrecision,
    QudaPrecision gPrecision, QudaDslashType dslash_type);

void matdagmat(cpuColorSpinorField *out, void **fatlink, void **longlink, void **ghost_fatlink, void **ghost_longlink,
    cpuColorSpinorField *in, double mass, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision,
    cpuColorSpinorField *tmp, QudaParity parity, QudaDslashType dslash_type);

#endif // _QUDA_DLASH_REF_H
