#pragma once 

#include <blas_reference.h>
#include <quda_internal.h>
#include <color_spinor_field.h>

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;

void setDims(int *);

void staggered_dslash(ColorSpinorField *out, void **fatlink, void **longlink, void **ghost_fatlink,
    void **ghost_longlink, ColorSpinorField *in, int oddBit, int daggerBit, QudaPrecision sPrecision,
    QudaPrecision gPrecision, QudaDslashType dslash_type);

void matdagmat(ColorSpinorField *out, void **fatlink, void **longlink, void **ghost_fatlink, void **ghost_longlink,
	       ColorSpinorField *in, double mass, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision,
	       ColorSpinorField *tmp, QudaParity parity, QudaDslashType dslash_type);

void verifyStaggeredInversion(ColorSpinorField *tmp, ColorSpinorField *ref, ColorSpinorField *in, ColorSpinorField *out, 
			      void *qdp_fatlink[], void *qdp_longlink[], void **ghost_fatlink, void **ghost_longlink, 
			      QudaGaugeParam &gauge_param, QudaInvertParam &inv_param);
