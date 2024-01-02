#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

static constexpr bool is_multi_gpu_build() {
#ifdef MULTI_GPU
  return true;
#else
  return false;
#endif
}

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;

void setDims(int *);

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
  * @param comm_override Override array used for local operators, elements are "1" for comms as usual, "0" to use PBC instead, default all 1
  */
void stag_dslash(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
                 int oddBit, int daggerBit, QudaDslashType dslash_type, std::array<int, 4> comm_override = {1, 1, 1, 1});

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
  * @param comm_override Override array used for local operators, elements are "1" for comms as usual, "0" to use PBC instead, default all 1
  */
void stag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
              double mass, int daggerBit, QudaDslashType dslash_type, std::array<int, 4> comm_override = {1, 1, 1, 1});

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
  * @param comm_override Override array used for local operators, elements are "1" for comms as usual, "0" to use PBC instead, default all 1
  */
void stag_matdag_mat(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
              double mass, int daggerBit, QudaDslashType dslash_type, std::array<int, 4> comm_override = {1, 1, 1, 1});

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
  * @param comm_override Override array used for local operators, elements are "1" for comms as usual, "0" to use PBC instead, default all 1
  */
void stag_matpc(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
                double mass, int dagger_bit, QudaParity parity, QudaDslashType dslash_type, std::array<int, 4> comm_override = {1, 1, 1, 1});

/**
  * @brief Apply the local version of the full parity staggered-type dslash
  *
  * @param out Host output rhs
  * @param fat_link Extended fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
  * @param long_link Extended long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
  * @param in Host input spinor
  * @param mass Mass for the dslash operator
  * @param daggerBit 0 for the regular operator, 1 for the dagger operator
  * @param dslash_type Dslash type
  */
void stag_mat_local(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
              double mass, int daggerBit, QudaDslashType dslash_type);

/**
  * @brief Apply the local version of the full parity staggered-type matdag_mat
  *
  * @param out Host output rhs
  * @param fat_link Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
  * @param long_link Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
  * @param in Host input spinor
  * @param mass Mass for the dslash operator
  * @param daggerBit 0 for the regular operator, 1 for the dagger operator
  * @param dslash_type Dslash type
  */
void stag_matdag_mat_local(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
              double mass, int daggerBit, QudaDslashType dslash_type);

/**
  * @brief Apply the local version of the even-even or odd-odd preconditioned staggered dslash
  *
  * @param out Host output rhs
  * @param fat_link Fat links for an asqtad dslash, or the gauge links for a staggered or Laplace dslash
  * @param long_link Long links for an asqtad dslash, or an empty GaugeField for staggered or Laplace dslash
  * @param in Host input spinor
  * @param mass Mass for the dslash operator
  * @param dagger_bit 0 for the regular operator, 1 for the dagger operator --- irrelevant for the HPD preconditioned operator
  * @param parity Parity of preconditioned dslash
  * @param dslash_type Dslash type
  * @param comm_override Override array used for local operators, elements are "1" for comms as usual, "0" to use PBC instead, default all 1
  */
void stag_matpc_local(ColorSpinorField &out, const GaugeField &fat_link, const GaugeField &long_link, const ColorSpinorField &in,
                double mass, int dagger_bit, QudaParity parity, QudaDslashType dslash_type);
