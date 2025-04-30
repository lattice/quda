#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;

void setDims(int *);

/**
 * @brief Apply even-odd or odd-even component of a staggered-type dslash
 *
 * @param[out] out Host output rhs
 * @param[in] fat_link Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
 * @param[in] long_link Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
 * @param[in] in Host input spinor
 * @param[in] oddBit 0 for D_eo, 1 for D_oe
 * @param[in] daggerBit 0 for the regular operator, 1 for the dagger operator
 * @param[in] dslash_type Dslash type
 * @param[in] laplace3D Whether we applying the 3-d variant of the
 * Laplace operator.  A value of 4 is the regular 4-d operator, and a
 * value < 4 implies a 3-d operator with the value designating the
 * orthogonal dimension (e.g., the dimension not enabled).
 */
void stag_dslash(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
                 const ColorSpinorField &in, int oddBit, int daggerBit, QudaDslashType dslash_type, int laplace3D);

/**
 * @brief Apply the full parity staggered-type dslash
 *
 * @param[out] out Host output rhs
 * @param[in] fat_link Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
 * @param[in] long_link Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
 * @param[in] in Host input spinor
 * @param[in] mass Mass for the dslash operator
 * @param[in] daggerBit 0 for the regular operator, 1 for the dagger operator
 * @param[in] dslash_type Dslash type
 * @param[in] laplace3D Whether we applying the 3-d variant of the
 * Laplace operator.  A value of 4 is the regular 4-d operator, and a
 * value < 4 implies a 3-d operator with the value designating the
 * orthogonal dimension (e.g., the dimension not enabled).
 */
void stag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
              const ColorSpinorField &in, double mass, int daggerBit, QudaDslashType dslash_type, int laplace3D);

/**
 * @brief Apply the full parity staggered-type matdag_mat
 *
 * @param[out] out Host output rhs
 * @param[in] fat_link Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
 * @param[in] long_link Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
 * @param[in] in Host input spinor
 * @param[in] mass Mass for the dslash operator
 * @param[in] daggerBit 0 for the regular operator, 1 for the dagger operator
 * @param[in] dslash_type Dslash type
 * @param[in] laplace3D Whether we applying the 3-d variant of the
 * Laplace operator.  A value of 4 is the regular 4-d operator, and a
 * value < 4 implies a 3-d operator with the value designating the
 * orthogonal dimension (e.g., the dimension not enabled).
 */
void stag_matdag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
                     const ColorSpinorField &in, double mass, int daggerBit, QudaDslashType dslash_type, int laplace3D);

/**
 * @brief Apply the even-even or odd-odd preconditioned staggered dslash
 *
 * @param[out] out Host output rhs
 * @param[in] fat_link Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
 * @param[in] long_link Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
 * @param[in] in Host input spinor
 * @param[in] mass Mass for the dslash operator
 * @param[in] dagger_bit 0 for the regular operator, 1 for the dagger operator --- irrelevant for the HPD preconditioned operator
 * @param[in] parity Parity of preconditioned dslash
 * @param[in] dslash_type Dslash type
 * @param[in] laplace3D Whether we applying the 3-d variant of the
 * Laplace operator.  A value of 4 is the regular 4-d operator, and a
 * value < 4 implies a 3-d operator with the value designating the
 * orthogonal dimension (e.g., the dimension not enabled).
 */
void stag_matpc(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
                const ColorSpinorField &in, double mass, int dagger_bit, QudaParity parity, QudaDslashType dslash_type,
                int laplace3D);
