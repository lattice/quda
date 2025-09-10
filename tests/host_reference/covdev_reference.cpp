#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <quda_internal.h>
#include <quda.h>
#include <blas_quda.h>

#include "host_utils.h"
#include "index_utils.hpp"
#include "misc.h"
#include "covdev_reference.h"
#include "dslash_reference.h"
#include "util_quda.h"

// covdevReference()
//
// if oddBit is zero: calculate even parity spinor elements (using odd parity spinor)
// if oddBit is one:  calculate odd parity spinor elements
//
// if daggerBit is zero: perform ordinary covariant derivative operator
// if daggerBit is one:  perform hermitian covariant derivative operator
//

template <typename real_t>
void covdevReference(real_t *res, real_t **link, real_t **ghostLink, const ColorSpinorField &in, int oddBit,
                     int daggerBit, int mu)
{
  auto fwd_nbr_spinor = reinterpret_cast<real_t **>(in.fwdGhostFaceBuffer);
  auto back_nbr_spinor = reinterpret_cast<real_t **>(in.backGhostFaceBuffer);

  const int my_spinor_site_size = in.Nspin() == 1 ? stag_spinor_site_size : spinor_site_size;
  int muDagger = mu * 2 + daggerBit;

#pragma omp parallel for
  for (int i = 0; i < Vh * my_spinor_site_size; i++) res[i] = 0.0;

  real_t *linkEven[4], *linkOdd[4];
  real_t *ghostLinkEven[4], *ghostLinkOdd[4];

  for (int dir = 0; dir < 4; dir++) {
    linkEven[dir] = link[dir];
    linkOdd[dir] = link[dir] + Vh * gauge_site_size;

    ghostLinkEven[dir] = ghostLink[dir];
    ghostLinkOdd[dir] = ghostLink[dir] + (faceVolume[dir] / 2) * gauge_site_size;
  }

#pragma omp parallel for
  for (int sid = 0; sid < Vh; sid++) {
    int offset = my_spinor_site_size * sid;

    const real_t *lnk = gaugeLink(sid, muDagger, oddBit, linkEven, linkOdd, ghostLinkEven, ghostLinkOdd, 1, 1);
    const real_t *spinor = spinorNeighbor(sid, muDagger, oddBit, static_cast<const real_t *>(in.data()), fwd_nbr_spinor,
                                          back_nbr_spinor, 1, 1, my_spinor_site_size);

    std::vector<real_t> gaugedSpinor(my_spinor_site_size);

    if (daggerBit) {
      for (int s = 0; s < in.Nspin(); s++) su3Tmul(&gaugedSpinor[s * 6], lnk, &spinor[s * 6]);
    } else {
      for (int s = 0; s < in.Nspin(); s++) su3Mul(&gaugedSpinor[s * 6], lnk, &spinor[s * 6]);
    }
    sum(&res[offset], &res[offset], gaugedSpinor.data(), spinor_site_size);

  } // 4-d volume
}

void covdev_dslash(ColorSpinorField &out, const GaugeField &link, const ColorSpinorField &in, int oddBit, int daggerBit,
                   int mu, QudaPrecision sPrecision, QudaPrecision gPrecision)
{
  if (sPrecision != gPrecision) errorQuda("Spinor and gauge field precision do not match");

  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (oddBit == QUDA_EVEN_PARITY) {
    otherparity = QUDA_ODD_PARITY;
  } else if (oddBit == QUDA_ODD_PARITY) {
    otherparity = QUDA_EVEN_PARITY;
  } else {
    errorQuda("full parity not supported");
  }
  const int nFace = 1;

  in.exchangeGhost(otherparity, nFace, daggerBit);

  void *data[4] = {link.data(0), link.data(1), link.data(2), link.data(3)};
  void *ghostLink[4] = {link.Ghost()[0].data(), link.Ghost()[1].data(), link.Ghost()[2].data(), link.Ghost()[3].data()};

  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    covdevReference((double *)out.data(), reinterpret_cast<double **>(data), (double **)ghostLink, in, oddBit,
                    daggerBit, mu);
  } else {
    covdevReference((float *)out.data(), reinterpret_cast<float **>(data), (float **)ghostLink, in, oddBit, daggerBit,
                    mu);
  }
}

template <typename real_t>
void Mat(ColorSpinorField &out, const GaugeField &link, const ColorSpinorField &in, int daggerBit, int mu)
{
  void *data[4] = {link.data(0), link.data(1), link.data(2), link.data(3)};
  void *ghostLink[4] = {link.Ghost()[0].data(), link.Ghost()[1].data(), link.Ghost()[2].data(), link.Ghost()[3].data()};

  const int nFace = 1;
  {
    auto &inEven = in.Even();
    auto &outOdd = out.Odd();

    inEven.exchangeGhost(QUDA_EVEN_PARITY, nFace, daggerBit);
    covdevReference(reinterpret_cast<real_t *>(outOdd.data()), reinterpret_cast<real_t **>(data),
                    reinterpret_cast<real_t **>(ghostLink), in.Even(), 1, daggerBit, mu);
  }

  {
    auto &inOdd = in.Odd();
    auto &outEven = out.Even();

    inOdd.exchangeGhost(QUDA_ODD_PARITY, nFace, daggerBit);
    covdevReference(reinterpret_cast<real_t *>(outEven.data()), reinterpret_cast<real_t **>(data),
                    reinterpret_cast<real_t **>(ghostLink), in.Odd(), 0, daggerBit, mu);
  }
}

void mat(ColorSpinorField &out, const GaugeField &link, const ColorSpinorField &in, int dagger_bit, int mu)
{
  if (checkPrecision(in, out, link) == QUDA_DOUBLE_PRECISION) {
    Mat<double>(out, link, in, dagger_bit, mu);
  } else {
    Mat<float>(out, link, in, dagger_bit, mu);
  }
}

void matdagmat(ColorSpinorField &out, const GaugeField &link, const ColorSpinorField &in, int dagger_bit, int mu,
               QudaPrecision sPrecision, QudaPrecision gPrecision, ColorSpinorField &tmp, QudaParity parity)
{
  // assert sPrecision and gPrecision must be the same
  if (sPrecision != gPrecision) errorQuda("Spinor precision and gPrecison is not the same");

  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (parity == QUDA_EVEN_PARITY) {
    otherparity = QUDA_ODD_PARITY;
  } else if (parity == QUDA_ODD_PARITY) {
    otherparity = QUDA_EVEN_PARITY;
  } else {
    errorQuda("full parity not supported");
  }

  covdev_dslash(tmp, link, in, otherparity, dagger_bit, mu, sPrecision, gPrecision);

  covdev_dslash(out, link, tmp, parity, dagger_bit, mu, sPrecision, gPrecision);
}
