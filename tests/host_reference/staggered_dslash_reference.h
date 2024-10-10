#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;

void setDims(int *);

/**
 * @brief Base host routine to apply the even-odd or odd-even component of a staggered-type dslash
 *
 * @tparam real_t Datatype used in the host dslash
 * @param res Host output result
 * @param fatlink Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
 * @param longlink Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
 * @param ghostFatlink Ghost zones for the host fat links
 * @param ghostLonglink Ghost zones for the host long links
 * @param spinorField Host input spinor
 * @param fwd_nbr_spinor Forward ghost zones for the host input spinor
 * @param back_nbr_spinor Backwards ghost zones for the host input spinor
 * @param oddBit 0 for D_eo, 1 for D_oe
 * @param daggerBit 0 for the regular operator, 1 for the dagger operator
 * @param dslash_type Dslash type
 */
template <typename real_t>
void staggeredDslashReference(real_t *res, real_t **fatlink, real_t **longlink, real_t **ghostFatlink,
                              real_t **ghostLonglink, real_t *spinorField, real_t **fwd_nbr_spinor,
                              real_t **back_nbr_spinor, int oddBit, int daggerBit, QudaDslashType dslash_type,
                              int laplace3D);

/**
 * @brief Apply even-odd or odd-even component of a staggered-type dslash
 *
 * @param out Host output rhs
 * @param fat_link Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
 * @param long_link Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
 * @param in Host input spinor
 * @param oddBit 0 for D_eo, 1 for D_oe
 * @param daggerBit 0 for the regular operator, 1 for the dagger operator
 * @param dslash_type Dslash type
 */
void stag_dslash(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
                 const ColorSpinorField &in, int oddBit, int daggerBit, QudaDslashType dslash_type, int laplace3D);

/**
 * @brief Apply the full parity staggered-type dslash
 *
 * @param out Host output rhs
 * @param fat_link Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
 * @param long_link Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
 * @param in Host input spinor
 * @param mass Mass for the dslash operator
 * @param daggerBit 0 for the regular operator, 1 for the dagger operator
 * @param dslash_type Dslash type
 */
void stag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
              const ColorSpinorField &in, double mass, int daggerBit, QudaDslashType dslash_type, int laplace3D);

/**
 * @brief Apply the full parity staggered-type matdag_mat
 *
 * @param out Host output rhs
 * @param fat_link Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
 * @param long_link Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
 * @param in Host input spinor
 * @param mass Mass for the dslash operator
 * @param daggerBit 0 for the regular operator, 1 for the dagger operator
 * @param dslash_type Dslash type
 */
void stag_matdag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
                     const ColorSpinorField &in, double mass, int daggerBit, QudaDslashType dslash_type, int laplace3D);

/**
 * @brief Apply the even-even or odd-odd preconditioned staggered dslash
 *
 * @param out Host output rhs
 * @param fat_link Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
 * @param long_link Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
 * @param in Host input spinor
 * @param mass Mass for the dslash operator
 * @param dagger_bit 0 for the regular operator, 1 for the dagger operator --- irrelevant for the HPD preconditioned operator
 * @param parity Parity of preconditioned dslash
 * @param dslash_type Dslash type
 */
void stag_matpc(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link,
                const ColorSpinorField &in, double mass, int dagger_bit, QudaParity parity, QudaDslashType dslash_type,
                int laplace3D);
