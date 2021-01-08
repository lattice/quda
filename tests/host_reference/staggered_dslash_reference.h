#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;

void setDims(int *);

template <typename sFloat, typename gFloat>
void staggeredDslashReference(sFloat *res, gFloat **fatlink, gFloat **longlink, gFloat **ghostFatlink,
                              gFloat **ghostLonglink, sFloat *spinorField, sFloat **fwd_nbr_spinor,
                              sFloat **back_nbr_spinor, int oddBit, int daggerBit, int nSrc, QudaDslashType dslash_type);

void staggeredDslash(ColorSpinorField *out, void **fatlink, void **longlink, void **ghost_fatlink,
                     void **ghost_longlink, ColorSpinorField *in, int oddBit, int daggerBit, QudaPrecision sPrecision,
                     QudaPrecision gPrecision, QudaDslashType dslash_type);

void staggeredMatDagMat(ColorSpinorField *out, void **fatlink, void **longlink, void **ghost_fatlink,
                        void **ghost_longlink, ColorSpinorField *in, double mass, int dagger_bit,
                        QudaPrecision sPrecision, QudaPrecision gPrecision, ColorSpinorField *tmp, QudaParity parity,
                        QudaDslashType dslash_type);
