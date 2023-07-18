#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;

void setDims(int *);

void staggeredDslash(ColorSpinorField &out, void **fatlink, void **longlink, void **ghost_fatlink,
                     void **ghost_longlink, const ColorSpinorField &in, int oddBit, int daggerBit,
                     QudaPrecision sPrecision, QudaPrecision gPrecision, QudaDslashType dslash_type,
		     bool use_ghost);

void staggeredMatDagMat(ColorSpinorField &out, void **fatlink, void **longlink, void **ghost_fatlink,
                        void **ghost_longlink, const ColorSpinorField &in, double mass, int dagger_bit,
                        QudaPrecision sPrecision, QudaPrecision gPrecision, ColorSpinorField &tmp, QudaParity parity,
                        QudaDslashType dslash_type, bool use_ghost);

// Versions of the above functions that take in cpuGaugeField
void staggeredDslash(ColorSpinorField &out, cpuGaugeField *fatlink, cpuGaugeField *longlink,
                     const ColorSpinorField &in, int oddBit, int daggerBit, QudaDslashType dslash_type,
                     bool use_ghost);

void staggeredMatDagMat(ColorSpinorField &out, cpuGaugeField *fatlink, cpuGaugeField *longlink,
                        const ColorSpinorField &in, double mass, int dagger_bit, ColorSpinorField &tmp,
                        QudaParity parity, QudaDslashType dslash_type, bool use_ghost);

// Local operator routine that handles creating "extended" ColorSpinorFields, injecting/extracting
// spinor values, etc
void staggeredMatDagMatLocal(ColorSpinorField &out, cpuGaugeField *fatlink, cpuGaugeField *longlink,
                        const ColorSpinorField &in, double mass, int dagger_bit,
                        QudaParity parity, QudaDslashType dslash_type);

