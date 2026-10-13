#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#include <quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <instantiate.h>

#include "command_line_params.h"
#include "gauge_utils.h"
#include "host_utils.h"
#include "misc.h"
#include "covdev_reference.h"
#include "staggered_dslash_reference.h"
#include "test.h"

using namespace quda;

// One complex per color. n_ev is already a command-line parameter.
constexpr int nColor = 3;
constexpr int n_current_ev = 2;

#include "exact_current_test_gtest.hpp"

namespace {

void **alloc_links(size_t bytes)
{
  void **links = new void *[4];
  for (int dir = 0; dir < 4; dir++) links[dir] = safe_malloc(bytes);
  return links;
}

void free_links(void **links)
{
  if (!links) return;
  for (int dir = 0; dir < 4; dir++) host_free(links[dir]);
  delete[] links;
}

// y.Odd() = scale * x.Odd(). axy, not axpy: the odd MCD result is replaced.
template <typename real>
void copy_odd(ColorSpinorField &y, const ColorSpinorField &x, real scale)
{
  const int n = (y.Volume() / 2) * nColor;
  auto *dst = reinterpret_cast<std::complex<real> *>(y.Odd().data());
  auto *src = reinterpret_cast<const std::complex<real> *>(x.Odd().data());
  for (int i = 0; i < n; i++) dst[i] = scale * src[i];
}

// G5 is +1 on even sites and -1 on odd sites. G1 is the identity.
template <typename real> void apply_g5(ColorSpinorField &x) { copy_odd<real>(x, x, static_cast<real>(-1)); }

template <typename real>
std::complex<real> site_dot(const ColorSpinorField &a, const ColorSpinorField &b, int site)
{
  auto *aa = reinterpret_cast<const std::complex<real> *>(a.data());
  auto *bb = reinterpret_cast<const std::complex<real> *>(b.data());
  std::complex<real> dot = 0;
  for (int c = 0; c < nColor; c++) dot += std::conj(aa[nColor * site + c]) * bb[nColor * site + c];
  return dot;
}

template <typename real>
void accumulate_imag(real *jlow, const ColorSpinorField &left, const ColorSpinorField &right, int mu, size_t base,
                     int vol_cb, double zscale)
{
  const real z = static_cast<real>(zscale);
  for (int x = 0; x < vol_cb; x++) {
    jlow[base + 4 * x + mu] += z * site_dot<real>(left, right, static_cast<int>(base == 0 ? x : vol_cb + x)).imag();
  }
}

void zscales(const double *eval, const double *mass, int nmasses, int i, double &zscale, double &zscale2)
{
  zscale = 0;
  zscale2 = 0;
  switch (nmasses) {
  case 1: zscale = 1.0 / (eval[i] + 4.0 * mass[0] * mass[0]); break;
  case 2: {
    const double dl = eval[i] + 4.0 * mass[0] * mass[0];
    const double ds = eval[i] + 4.0 * mass[1] * mass[1];
    zscale = 4.0 * (mass[1] * mass[1] - mass[0] * mass[0]) / (dl * ds);
    break;
  }
  case 3: {
    const double du = eval[i] + 4.0 * mass[0] * mass[0];
    const double dd = eval[i] + 4.0 * mass[1] * mass[1];
    const double ds = eval[i] + 4.0 * mass[2] * mass[2];
    zscale = 4.0 * (mass[1] * mass[1] - mass[0] * mass[0]) / (du * dd);
    zscale2 = 4.0 * (mass[2] * mass[2] - mass[0] * mass[0]) / (du * ds);
    break;
  }
  default: errorQuda("Unexpected nmasses %d", nmasses);
  }
}

template <typename real>
void host_current(real *jlow, real *jlow2, const GaugeField &fat, const GaugeField &lng, const GaugeField &shift,
                  std::vector<ColorSpinorField> &evec_even, std::vector<ColorSpinorField> &evec_odd, const double *eval,
                  const double *mass, int nmasses)
{
  ColorSpinorParam full(evec_even[0]);
  full.siteSubset = QUDA_FULL_SITE_SUBSET;
  full.x[0] *= 2;
  full.create = QUDA_ZERO_FIELD_CREATE;
  ColorSpinorField gr0(full);
  ColorSpinorField gr_mu(full);
  ColorSpinorField evec(full);

  ColorSpinorParam parity(evec_even[0]);
  parity.create = QUDA_ZERO_FIELD_CREATE;
  ColorSpinorField dslash_out(parity);

  const int vol_cb = gr0.Volume() / 2;
  const size_t odd_base = 2 * gr0.Volume();

  for (int i = 0; i < n_current_ev; i++) {
    // D_oe. Even sites of gr0 stay zero.
    memset(dslash_out.data(), 0, dslash_out.Bytes());
    stag_dslash(dslash_out, fat, lng, evec_even[i], QUDA_ODD_PARITY, 0, QUDA_ASQTAD_DSLASH, 4);
    memcpy(gr0.Odd().data(), dslash_out.data(), dslash_out.Bytes());

    memcpy(evec.Even().data(), evec_even[i].data(), evec_even[i].Bytes());
    memcpy(evec.Odd().data(), evec_odd[i].data(), evec_odd[i].Bytes());

    double zscale = 0, zscale2 = 0;
    zscales(eval, mass, nmasses, i, zscale, zscale2);

    for (int mu = 0; mu < 4; mu++) {
      mat(gr_mu, shift, gr0, 0, mu);
      copy_odd<real>(gr_mu, gr0, static_cast<real>(-1));
      apply_g5<real>(gr_mu);
      accumulate_imag<real>(jlow, evec, gr_mu, mu, 0, vol_cb, -zscale);
      if (nmasses == 3) accumulate_imag<real>(jlow2, evec, gr_mu, mu, 0, vol_cb, -zscale2);

      mat(gr_mu, shift, evec, 0, mu);
      apply_g5<real>(gr_mu); // ax(-1) on the odd sites
      apply_g5<real>(gr_mu); // spin-taste G5
      accumulate_imag<real>(jlow, gr0, gr_mu, mu, odd_base, vol_cb, +zscale);
      if (nmasses == 3) accumulate_imag<real>(jlow2, gr0, gr_mu, mu, odd_base, vol_cb, +zscale2);
    }
  }
}

template <typename real>
double max_abs_diff(const real *a, const real *b, size_t n)
{
  double deviation = 0;
  for (size_t i = 0; i < n; i++) {
    double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (std::isnan(diff)) return std::numeric_limits<double>::infinity();
    deviation = std::max(deviation, diff);
  }
  return deviation;
}

ColorSpinorParam parity_param(const QudaGaugeParam &gauge_param, QudaPrecision prec)
{
  ColorSpinorParam cs;
  cs.nColor = nColor;
  cs.nSpin = 1;
  cs.nDim = 4;
  for (int d = 0; d < 4; d++) cs.x[d] = gauge_param.X[d];
  cs.x[0] /= 2;
  cs.setPrecision(prec);
  cs.pad = 0;
  cs.siteSubset = QUDA_PARITY_SITE_SUBSET;
  cs.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  cs.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  cs.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  cs.pc_type = QUDA_4D_PC;
  cs.create = QUDA_ZERO_FIELD_CREATE;
  cs.location = QUDA_CPU_FIELD_LOCATION;
  return cs;
}

} // namespace

double exact_current_test(test_t param)
{
  const QudaPrecision prec = ::testing::get<0>(param);
  const int nmasses = ::testing::get<1>(param);

  for (int d = 0; d < 4; d++) {
    if (dim[d] % 2) errorQuda("exact_current_test requires even dimensions, dim[%d] = %d", d, dim[d]);
  }

  cpu_prec = prec;
  cuda_prec = prec;
  cuda_prec_sloppy = prec;
  cuda_prec_precondition = prec;
  cuda_prec_eigensolver = prec;
  host_gauge_data_type_size = (prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  dslash_type = QUDA_ASQTAD_DSLASH;

  QudaGaugeParam fat_param = newQudaGaugeParam();
  setStaggeredGaugeParam(fat_param);
  fat_param.cpu_prec = prec;
  fat_param.cuda_prec = prec;
  fat_param.cuda_prec_sloppy = prec;
  fat_param.cuda_prec_precondition = prec;
  fat_param.cuda_prec_eigensolver = prec;
  fat_param.cuda_prec_refinement_sloppy = prec;
  fat_param.reconstruct = QUDA_RECONSTRUCT_NO;
  fat_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  fat_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;
  fat_param.reconstruct_eigensolver = QUDA_RECONSTRUCT_NO;
  fat_param.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;
  fat_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  fat_param.location = QUDA_CPU_FIELD_LOCATION;
  fat_param.type = QUDA_ASQTAD_FAT_LINKS;
  fat_param.tadpole_coeff = 1.0;

  const size_t link_bytes = V * gauge_site_size * host_gauge_data_type_size;
  void **fat = alloc_links(link_bytes);
  void **lng = alloc_links(link_bytes);
  void **shift = alloc_links(link_bytes);

  // Unit fat and long links, with the same staggered phases the host dslash reference expects.
  constructIdentityGaugeField(fat, prec);
  applyGaugeFieldScaling(fat, Vh, fat_param, prec);
  applyGaugeFieldScaling_long(fat, Vh, fat_param, QUDA_STAGGERED_DSLASH, prec);
  constructIdentityGaugeField(lng, prec);
  applyGaugeFieldScaling(lng, Vh, fat_param, prec);
  applyGaugeFieldScaling_long(lng, Vh, fat_param, QUDA_ASQTAD_DSLASH, prec);

  QudaGaugeParam shift_param = fat_param;
  shift_param.type = QUDA_WILSON_LINKS;
  constructQudaGaugeField(shift, GaugeFieldConstructionType::UNIT_GAUGE, shift_param, prec);

  GaugeFieldParam fat_field_param(fat_param, fat, QUDA_ASQTAD_FAT_LINKS);
  fat_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  GaugeField cpu_fat(fat_field_param);

  GaugeFieldParam long_field_param(fat_param, lng, QUDA_ASQTAD_LONG_LINKS);
  long_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  GaugeField cpu_long(long_field_param);

  GaugeFieldParam shift_field_param(shift_param, shift, QUDA_WILSON_LINKS);
  shift_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  GaugeField cpu_shift(shift_field_param);

  loadGaugeQuda(fat, &fat_param);
  QudaGaugeParam long_param = fat_param;
  long_param.type = QUDA_ASQTAD_LONG_LINKS;
  loadGaugeQuda(lng, &long_param);
  shift_param.make_resident_gauge = true;
  shift_param.use_resident_gauge = false;
  loadGaugeQuda(shift, &shift_param);

  auto cs = parity_param(fat_param, prec);
  std::vector<ColorSpinorField> evec_even, evec_odd;
  evec_even.reserve(n_current_ev);
  evec_odd.reserve(n_current_ev);
  for (int i = 0; i < n_current_ev; i++) {
    evec_even.emplace_back(cs);
    evec_odd.emplace_back(cs);
    evec_even[i].Source(QUDA_RANDOM_SOURCE);
    evec_odd[i].Source(QUDA_RANDOM_SOURCE);
  }

  ColorSpinorParam gpu(cs);
  gpu.setPrecision(prec, prec, true);
  gpu.location = QUDA_CUDA_FIELD_LOCATION;
  gpu.create = QUDA_ZERO_FIELD_CREATE;
  std::vector<ColorSpinorField> cuda_even, cuda_odd;
  cuda_even.reserve(n_current_ev);
  cuda_odd.reserve(n_current_ev);
  std::vector<void *> even_ptr(n_current_ev), odd_ptr(n_current_ev);
  for (int i = 0; i < n_current_ev; i++) {
    cuda_even.emplace_back(gpu);
    cuda_odd.emplace_back(gpu);
    cuda_even[i] = evec_even[i];
    cuda_odd[i] = evec_odd[i];
    even_ptr[i] = cuda_even[i].data();
    odd_ptr[i] = cuda_odd[i].data();
  }

  const double eval[n_current_ev] = {0.3, 1.7};
  const double mass[3] = {0.05, 0.10, 0.20};
  // evec_even is a parity field, so the full volume is twice its volume.
  const size_t full_volume = 2 * static_cast<size_t>(evec_even[0].Volume());
  const size_t current_len = 4 * full_volume;

  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  inv_param.dslash_type = QUDA_ASQTAD_DSLASH;
  inv_param.cpu_prec = prec;
  inv_param.cuda_prec = prec;
  inv_param.cuda_prec_sloppy = prec;
  inv_param.cuda_prec_precondition = prec;
  inv_param.cuda_prec_eigensolver = prec;
  inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.solution_type = QUDA_MATPC_SOLUTION;
  inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;

  const int X[4] = {fat_param.X[0], fat_param.X[1], fat_param.X[2], fat_param.X[3]};

  double deviation = 0;
  if (prec == QUDA_DOUBLE_PRECISION) {
    std::vector<double> jlow(current_len, 0), ref(current_len, 0);
    std::vector<double> jlow2(current_len, 0), ref2(current_len, 0);
    host_current<double>(ref.data(), ref2.data(), cpu_fat, cpu_long, cpu_shift, evec_even, evec_odd, eval, mass, nmasses);
    exactCurrentQuda(even_ptr.data(), odd_ptr.data(), eval, n_current_ev, mass, nmasses, &inv_param, X, jlow.data(),
                     nmasses == 3 ? jlow2.data() : nullptr);
    deviation = max_abs_diff(jlow.data(), ref.data(), current_len);
    if (nmasses == 3) deviation = std::max(deviation, max_abs_diff(jlow2.data(), ref2.data(), current_len));
  } else {
    std::vector<float> jlow(current_len, 0), ref(current_len, 0);
    std::vector<float> jlow2(current_len, 0), ref2(current_len, 0);
    host_current<float>(ref.data(), ref2.data(), cpu_fat, cpu_long, cpu_shift, evec_even, evec_odd, eval, mass, nmasses);
    exactCurrentQuda(even_ptr.data(), odd_ptr.data(), eval, n_current_ev, mass, nmasses, &inv_param, X, jlow.data(),
                     nmasses == 3 ? jlow2.data() : nullptr);
    deviation = max_abs_diff(jlow.data(), ref.data(), current_len);
    if (nmasses == 3) deviation = std::max(deviation, max_abs_diff(jlow2.data(), ref2.data(), current_len));
  }

  printfQuda("exact current deviation = %e, tolerance = %e (nmasses = %d, prec = %s)\n", deviation, getTolerance(prec),
             nmasses, get_prec_str(prec));

  freeGaugeQuda();
  cpu_fat = {};
  cpu_long = {};
  cpu_shift = {};
  free_links(fat);
  free_links(lng);
  free_links(shift);
  return deviation;
}

struct exact_current_tester : quda_test {
  exact_current_tester(int argc, char **argv) : quda_test("Exact Current Test", argc, argv) { }
};

int main(int argc, char **argv)
{
  dslash_type = QUDA_ASQTAD_DSLASH;

  exact_current_tester test(argc, argv);
  test.init();
  return test.execute();
}
