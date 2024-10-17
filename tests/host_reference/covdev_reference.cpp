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

template <typename Float>
void covdevReference(Float *res, Float **link, const Float *spinorField, int oddBit, int daggerBit, int mu)
{
  for (auto i = 0lu; i < Vh * spinor_site_size; i++) res[i] = 0.0;

  Float *linkEven[4], *linkOdd[4];

  for (int dir = 0; dir < 4; dir++) {
    linkEven[dir] = link[dir];
    linkOdd[dir] = link[dir] + Vh * gauge_site_size;
  }

  for (int sid = 0; sid < Vh; sid++) {
    auto offset = spinor_site_size * sid;

    Float gaugedSpinor[spinor_site_size];

    const Float *lnk = gaugeLink(sid, mu, oddBit, linkEven, linkOdd, 1);
    const Float *spinor = spinorNeighbor(sid, mu, oddBit, spinorField, 1);

    if (daggerBit) {
      for (int s = 0; s < 4; s++) su3Tmul(&gaugedSpinor[s * 6], lnk, &spinor[s * 6]);
    } else {
      for (int s = 0; s < 4; s++) su3Mul(&gaugedSpinor[s * 6], lnk, &spinor[s * 6]);
    }

    sum(&res[offset], &res[offset], gaugedSpinor, spinor_site_size);
  } // 4-d volume
}

void covdev_dslash(void *res, void **link, void *spinorField, int oddBit, int daggerBit, int mu,
                   QudaPrecision sPrecision, QudaPrecision gPrecision)
{
  if (sPrecision != gPrecision)
    errorQuda("Spinor and gauge field precision do not match");

  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    covdevReference((double *)res, (double **)link, (double *)spinorField, oddBit, daggerBit, mu);
  } else {
    covdevReference((float *)res, (float **)link, (float *)spinorField, oddBit, daggerBit, mu);
  }
}

template <typename Float>
void Mat(ColorSpinorField &out, const GaugeField &link, const ColorSpinorField &in, int daggerBit, int mu)
{
  // full dslash operator
  void *data[4] = {link.data(0), link.data(1), link.data(2), link.data(3)};
  covdevReference(reinterpret_cast<Float *>(out.Odd().data()), reinterpret_cast<Float **>(data),
                  reinterpret_cast<Float *>(in.Even().data()), 1, daggerBit, mu);
  covdevReference(reinterpret_cast<Float *>(out.Even().data()), reinterpret_cast<Float **>(data),
                  reinterpret_cast<Float *>(in.Odd().data()), 0, daggerBit, mu);
}

void mat(ColorSpinorField &out, const GaugeField &link, const ColorSpinorField &in, int dagger_bit, int mu)
{
  if (checkPrecision(in, out, link) == QUDA_DOUBLE_PRECISION) {
    Mat<double>(out, link, in, dagger_bit, mu);
  } else {
    Mat<float>(out, link, in, dagger_bit, mu);
  }
}

template <typename Float>
void Matdagmat(Float *out, Float **link, Float *in, int daggerBit, int mu, Float *tmp, QudaParity parity)
{
  switch (parity) {
  case QUDA_EVEN_PARITY: {
    Float *inEven = in;
    Float *outEven = out;
    covdevReference(tmp, link, inEven, 1, daggerBit, mu);
    covdevReference(outEven, link, tmp, 0, daggerBit, mu);
    break;
  }
  case QUDA_ODD_PARITY: {
    Float *inOdd = in;
    Float *outOdd = out;
    covdevReference(tmp, link, inOdd, 0, daggerBit, mu);
    covdevReference(outOdd, link, tmp, 1, daggerBit, mu);
    break;
  }

  default: fprintf(stderr, "ERROR: invalid parity in %s,line %d\n", __FUNCTION__, __LINE__); break;
  }
}

void matdagmat(void *out, void **link, void *in, int dagger_bit, int mu, QudaPrecision sPrecision,
               QudaPrecision gPrecision, void *tmp, QudaParity parity)
{
  if (sPrecision != gPrecision)
    errorQuda("Spinor and gauge field precision do not match");

  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    Matdagmat((double *)out, (double **)link, (double *)in, dagger_bit, mu, (double *)tmp, parity);
  } else {
    Matdagmat((float *)out, (float **)link, (float *)in, dagger_bit, mu, (float *)tmp, parity);
  }
}

#ifdef MULTI_GPU

template <typename sFloat, typename gFloat>
void covdevReference_mg4dir(sFloat *res, gFloat **link, gFloat **ghostLink, const ColorSpinorField &in, int oddBit,
                            int daggerBit, int mu)
{
  auto fwd_nbr_spinor = reinterpret_cast<sFloat **>(in.fwdGhostFaceBuffer);
  auto back_nbr_spinor = reinterpret_cast<sFloat **>(in.backGhostFaceBuffer);

  const int my_spinor_site_size = in.Nspin() == 1 ? stag_spinor_site_size : spinor_site_size;

  for (int i = 0; i < Vh * my_spinor_site_size; i++) res[i] = 0.0;

  gFloat *linkEven[4], *linkOdd[4];
  gFloat *ghostLinkEven[4], *ghostLinkOdd[4];

  for (int dir = 0; dir < 4; dir++) {
    linkEven[dir] = link[dir];
    linkOdd[dir] = link[dir] + Vh * gauge_site_size;

    ghostLinkEven[dir] = ghostLink[dir];
    ghostLinkOdd[dir] = ghostLink[dir] + (faceVolume[dir] / 2) * gauge_site_size;
  }

  for (int sid = 0; sid < Vh; sid++) {
    int offset = my_spinor_site_size * sid;

    const gFloat *lnk = gaugeLink(sid, mu, oddBit, linkEven, linkOdd, ghostLinkEven, ghostLinkOdd, 1, 1);
    const sFloat *spinor = spinorNeighbor(sid, mu, oddBit, static_cast<const sFloat *>(in.data()), fwd_nbr_spinor,
                                          back_nbr_spinor, 1, 1, my_spinor_site_size);

    std::vector<sFloat> gaugedSpinor(my_spinor_site_size);

    if (daggerBit) {
      for (int s = 0; s < in.Nspin(); s++) su3Tmul(&gaugedSpinor[s * 6], lnk, &spinor[s * 6]);
    } else {
      for (int s = 0; s < in.Nspin(); s++) su3Mul(&gaugedSpinor[s * 6], lnk, &spinor[s * 6]);
    }
    sum(&res[offset], &res[offset], gaugedSpinor.data(), spinor_site_size);
  } // 4-d volume
}

void covdev_dslash_mg4dir(ColorSpinorField &out, const GaugeField &link, const ColorSpinorField &in, int oddBit,
                          int daggerBit, int mu, QudaPrecision sPrecision, QudaPrecision gPrecision)
{
  if (sPrecision != gPrecision)
    errorQuda("Spinor and gauge field precision do not match");

  QudaParity otherparity = QUDA_INVALID_PARITY;
  if (oddBit == QUDA_EVEN_PARITY) {
    otherparity = QUDA_ODD_PARITY;
  } else if (oddBit == QUDA_ODD_PARITY) {
    otherparity = QUDA_EVEN_PARITY;
  } else {
    errorQuda("ERROR: full parity not supported in function %s", __FUNCTION__);
  }
  const int nFace = 1;

  in.exchangeGhost(otherparity, nFace, daggerBit);

  void *data[4] = {link.data(0), link.data(1), link.data(2), link.data(3)};
  void *ghostLink[4] = {link.Ghost()[0].data(), link.Ghost()[1].data(), link.Ghost()[2].data(), link.Ghost()[3].data()};

  if (sPrecision == QUDA_DOUBLE_PRECISION) {
    covdevReference_mg4dir((double *)out.data(), reinterpret_cast<double **>(data), (double **)ghostLink, in, oddBit,
                            daggerBit, mu);
  } else {
    covdevReference_mg4dir((float *)out.data(), reinterpret_cast<float **>(data), (float **)ghostLink, in, oddBit,
                            daggerBit, mu);
  }
}

template <typename Float>
void Mat_mg4dir(ColorSpinorField &out, const GaugeField &link, const ColorSpinorField &in, int daggerBit, int mu)
{
  void *data[4] = {link.data(0), link.data(1), link.data(2), link.data(3)};
  void *ghostLink[4] = {link.Ghost()[0].data(), link.Ghost()[1].data(), link.Ghost()[2].data(), link.Ghost()[3].data()};

  const int nFace = 1;
  {
    auto &inEven = in.Even();
    auto &outOdd = out.Odd();

    inEven.exchangeGhost(QUDA_EVEN_PARITY, nFace, daggerBit);
    covdevReference_mg4dir(reinterpret_cast<Float *>(outOdd.data()), reinterpret_cast<Float **>(data),
                           reinterpret_cast<Float **>(ghostLink), in.Even(), 1, daggerBit, mu);
  }

  {
    auto &inOdd = in.Odd();
    auto &outEven = out.Even();

    inOdd.exchangeGhost(QUDA_ODD_PARITY, nFace, daggerBit);
    covdevReference_mg4dir(reinterpret_cast<Float *>(outEven.data()), reinterpret_cast<Float **>(data),
                           reinterpret_cast<Float **>(ghostLink), in.Odd(), 0, daggerBit, mu);
  }
}

void mat_mg4dir(ColorSpinorField &out, const GaugeField &link, const ColorSpinorField &in, int dagger_bit, int mu)
{
  if (checkPrecision(in, out, link) == QUDA_DOUBLE_PRECISION) {
    Mat_mg4dir<double>(out, link, in, dagger_bit, mu);
  } else {
    Mat_mg4dir<float>(out, link, in, dagger_bit, mu);
  }
}

void matdagmat_mg4dir(ColorSpinorField &out, const GaugeField &link, const ColorSpinorField &in, int dagger_bit, int mu,
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

  covdev_dslash_mg4dir(tmp, link, in, otherparity, dagger_bit, mu, sPrecision, gPrecision);

  covdev_dslash_mg4dir(out, link, tmp, parity, dagger_bit, mu, sPrecision, gPrecision);
}

#endif
