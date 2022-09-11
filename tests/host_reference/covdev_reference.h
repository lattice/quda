#pragma once
#include <quda_internal.h>
#include "color_spinor_field.h"

using namespace quda;

void setDims(int *);

void covdev_dslash(void *res, void **link, void *spinorField, int oddBit, int daggerBit, int mu,
                   QudaPrecision sPrecision, QudaPrecision gPrecision);
void covdev_dslash_mg4dir(ColorSpinorField &out, void **link, void **ghostLink, const ColorSpinorField &in, int oddBit,
                          int daggerBit, int mu, QudaPrecision sPrecision, QudaPrecision gPrecision);

void mat(void *out, void **link, void *in, int daggerBit, int mu, QudaPrecision sPrecision, QudaPrecision gPrecision);

void matdagmat(void *out, void **link, void *in, int dagger_bit, int mu, QudaPrecision sPrecision,
               QudaPrecision gPrecision, void *tmp, QudaParity parity);

void mat_mg4dir(ColorSpinorField &out, void **link, void **ghostLink, const ColorSpinorField &in, int daggerBit, int mu,
                QudaPrecision sPrecision, QudaPrecision gPrecision);
void matdagmat_mg4dir(ColorSpinorField &out, void **link, void **ghostLink, const ColorSpinorField &in, int dagger_bit,
                      int mu, QudaPrecision sPrecision, QudaPrecision gPrecision, ColorSpinorField &tmp,
                      QudaParity parity);
