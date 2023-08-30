#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;

void setDims(int *);

template <typename real_t>
void staggeredDslashReference(real_t *res, real_t **fatlink, real_t **longlink, real_t **ghostFatlink,
                              real_t **ghostLonglink, real_t *spinorField, real_t **fwd_nbr_spinor,
                              real_t **back_nbr_spinor, int oddBit, int daggerBit, int nSrc, QudaDslashType dslash_type);

void stag_dslash(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
                 int oddBit, int daggerBit, QudaDslashType dslash_type);

void stag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
              double mass, int daggerBit, QudaDslashType dslash_type);

void stag_matpc(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
                double mass, int dagger_bit, ColorSpinorField &tmp, QudaParity parity, QudaDslashType dslash_type);
