#include <array>
#include <complex>
#include <cmath>
#include <memory>
#include <vector>

#include "force_utils.hpp"
#include "gauge_force_reference.h"
#include "gauge_smear_reference.h"
#include "host_utils.h"
#include "instantiate_host.hpp"
#include "misc.h"

namespace {

template <typename real_t> using matrix = Matrix<3, std::complex<real_t>>;

/**
 * @brief Check whether a matrix and its inverse satisfy the SU(3) tolerance.
 *
 * @param[in] inv Inverse of the matrix.
 * @param[in] u Matrix to check.
 * @param[in] tol Unitarity tolerance.
 */
template <typename real_t> bool is_unitary(const matrix<real_t> &inv, const matrix<real_t> &u, real_t tol)
{
  const auto identity = conj(u) * u;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (std::abs(u(i, j).real() - inv(j, i).real()) > tol
          || std::abs(u(i, j).imag() + inv(j, i).imag()) > tol)
        return false;
    }
    if (std::abs(identity(i, i).real() - static_cast<real_t>(1.0)) > tol || std::abs(identity(i, i).imag()) > tol)
      return false;
    for (int j = 0; j < i; j++) {
      if (std::abs(identity(i, j).real()) > tol || std::abs(identity(i, j).imag()) > tol
          || std::abs(identity(j, i).real()) > tol || std::abs(identity(j, i).imag()) > tol)
        return false;
    }
  }
  return true;
}

/**
 * @brief Project a matrix onto SU(3) using polar decomposition.
 *
 * @param[in,out] u Matrix to project.
 * @param[in] tol Unitarity tolerance.
 */
template <typename real_t> void polar_su3(matrix<real_t> &u, real_t tol)
{
  auto out = u;
  auto inv = u.inverse();
  for (int i = 0; !is_unitary(inv, out, tol) && i < 100; i++) {
    out = static_cast<real_t>(0.5) * (out + conj(inv));
    inv = out.inverse();
  }

  const auto det = out.determinant();
  const auto mod = std::pow(std::norm(det), static_cast<real_t>(-1.0 / 6.0));
  u = std::polar(mod, -std::arg(det) / static_cast<real_t>(3.0)) * out;
}

/**
 * @brief Compute the forward and backward APE staples for a link.
 *
 * @param[in] links Extended input gauge links.
 * @param[in] i Local link index.
 * @param[in] nu Link direction.
 * @param[in] dir_ignore Direction excluded from smearing.
 * @param[in] anisotropy Temporal anisotropy.
 * @param[in] lat Extended lattice metadata.
 * @return APE staple.
 */
template <typename real_t> matrix<real_t> staple(const matrix<real_t> *const *links, size_t i, int nu, int dir_ignore,
                                                 double anisotropy, const lattice_t &lat)
{
  matrix<real_t> out;
  for (int mu = 0; mu < 4; mu++) {
    if (mu == nu || mu == dir_ignore) continue;

    real_t coeff;
    coeff = 1.0;
    if (mu == 3) coeff = anisotropy * anisotropy;
    std::array<int, 4> dx {};
    const auto &u1 = links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
    dx[mu]++;
    const auto &u2 = links[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
    dx[mu]--;
    dx[nu]++;
    const auto &u3 = links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
    out += coeff * u1 * u2 * conj(u3);

    dx = {};
    dx[mu]--;
    const auto &u4 = links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
    const auto &u5 = links[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
    dx[nu]++;
    const auto &u6 = links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
    out += coeff * conj(u4) * u5 * u6;
  }
  return out;
}

/**
 * @brief Access a link at a local site in an extended gauge field.
 *
 * @param[in] links Extended gauge links.
 * @param[in] i Local link index.
 * @param[in] dir Link direction.
 * @param[in] lat Extended lattice metadata.
 * @return Link at the local site.
 */
template <typename real_t>
const matrix<real_t> &centered_link(const matrix<real_t> *const *links, size_t i, int dir, const lattice_t &lat)
{
  std::array<int, 4> dx {};
  return links[dir][gf_neighborIndexFullLattice(i, dx.data(), lat)];
}

/**
 * @brief Apply one APE smearing step to a host gauge field.
 *
 * @param[out] out Smeared gauge field.
 * @param[in] in Extended input gauge field.
 * @param[in] alpha APE smearing coefficient.
 * @param[in] dir_ignore Direction excluded from smearing.
 * @param[in] anisotropy Temporal anisotropy.
 * @param[in] lat Extended lattice metadata.
 */
template <typename real_t> struct APESmear {
  void operator()(quda::GaugeField &out, const quda::GaugeField &in, double alpha, int dir_ignore,
                  double anisotropy, const lattice_t &lat)
  {
    using link = matrix<real_t>;
    const auto input_ptrs = in.data_array<void *>();
    const auto output_ptrs = out.data_array<void *>();
    auto input = reinterpret_cast<const link *const *>(input_ptrs.data);
    auto output = reinterpret_cast<link *const *>(output_ptrs.data);
    const int ape_dim = dir_ignore == 4 ? 4 : 3;
    const auto alpha_ = static_cast<real_t>(alpha);
    const auto anisotropy_ = static_cast<real_t>(anisotropy);
    const auto scale = alpha_ / static_cast<real_t>(2 * (ape_dim - 1));
    const auto tolerance = static_cast<real_t>(in.toleranceSU3());

#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      for (int d = 0; d < ape_dim; d++) {
        const int nu = d + (d >= dir_ignore);
        const auto &u = centered_link(input, i, nu, lat);
        auto test_u = (static_cast<real_t>(1.0) - alpha_) * Identity<3, std::complex<real_t>>()()
          + scale * staple(input, i, nu, dir_ignore, anisotropy_, lat) * conj(u);
        polar_su3(test_u, tolerance);
        output[nu][i] = test_u * u;
      }
    }
  }
};

/**
 * @brief Compute a staple from links stored in separate direction fields.
 *
 * @param[in] gauge_mu Link field used for parallel links.
 * @param[in] gauge_nu Link field used for perpendicular links.
 * @param[in] i Local link index.
 * @param[in] mu Parallel direction.
 * @param[in] nu Perpendicular direction.
 * @param[in] lat Extended lattice metadata.
 * @return Forward and backward staple.
 */
template <typename real_t>
matrix<real_t> mixed_staple(const matrix<real_t> *const *gauge_mu, const matrix<real_t> *const *gauge_nu, size_t i,
                            int mu, int nu, const lattice_t &lat)
{
  std::array<int, 4> dx {};
  const auto &a = gauge_nu[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[nu]++;
  const auto &b = gauge_mu[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[nu]--;
  dx[mu]++;
  const auto &c = gauge_nu[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  auto out = a * b * conj(c);

  dx = {};
  dx[nu]--;
  const auto &d = gauge_nu[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  const auto &e = gauge_mu[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[mu]++;
  const auto &f = gauge_nu[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  out += conj(d) * e * f;
  return out;
}

/**
 * @brief Compute an oriented gauge path starting at a local link site.
 *
 * @param[in] links Extended gauge links.
 * @param[in] i Local link index.
 * @param[in] path Signed link directions, encoded as plus or minus (direction + 1).
 * @param[in] lat Extended lattice metadata.
 * @return Ordered path product.
 */
template <typename real_t, size_t N>
matrix<real_t> path_product(const matrix<real_t> *const *links, size_t i, const std::array<int, N> &path,
                            const lattice_t &lat)
{
  auto product = Identity<3, std::complex<real_t>>()();
  std::array<int, 4> dx {};
  for (const auto signed_dir : path) {
    const int dir = std::abs(signed_dir) - 1;
    if (signed_dir > 0) {
      product = product * links[dir][gf_neighborIndexFullLattice(i, dx.data(), lat)];
      dx[dir]++;
    } else {
      dx[dir]--;
      product = product * conj(links[dir][gf_neighborIndexFullLattice(i, dx.data(), lat)]);
    }
  }
  return product;
}

/**
 * @brief Compute the 1x2 and 2x1 rectangle sum around a link.
 *
 * @param[in] links Extended gauge links.
 * @param[in] i Local link index.
 * @param[in] nu Link direction.
 * @param[in] dir_ignore Direction excluded from the sum.
 * @param[in] anisotropy Temporal anisotropy.
 * @param[in] lat Extended lattice metadata.
 * @return Rectangle sum.
 */
template <typename real_t>
matrix<real_t> rectangle(const matrix<real_t> *const *links, size_t i, int nu, int dir_ignore, double anisotropy,
                         const lattice_t &lat)
{
  matrix<real_t> out;
  for (int mu = 0; mu < 4; mu++) {
    if (mu == nu || mu == dir_ignore) continue;
    real_t coeff;
    coeff = 1.0;
    if (mu == 3) coeff = anisotropy * anisotropy;
    const int m = mu + 1;
    const int n = nu + 1;
    out += coeff * path_product(links, i, std::array<int, 5> {-n, m, n, n, -m}, lat);
    out += coeff * path_product(links, i, std::array<int, 5> {m, n, n, -m, -n}, lat);
    out += coeff * path_product(links, i, std::array<int, 5> {m, m, n, -m, -m}, lat);
    out += coeff * path_product(links, i, std::array<int, 5> {-n, -m, n, n, m}, lat);
    out += coeff * path_product(links, i, std::array<int, 5> {-m, n, n, m, -n}, lat);
    out += coeff * path_product(links, i, std::array<int, 5> {-m, -m, n, m, m}, lat);
  }
  return out;
}

/**
 * @brief Exponentiate a stout generator and update a link.
 *
 * @param[in] omega Stout generator before Hermitian projection.
 * @param[in] u Input link.
 * @return Updated link.
 */
template <typename real_t> matrix<real_t> stout_update(matrix<real_t> omega, const matrix<real_t> &u)
{
  make_herm(omega);
  return exponentiate_iQ(omega) * u;
}

/**
 * @brief Apply one Stout or over-improved Stout step to a host gauge field.
 *
 * @param[out] out Smeared gauge field.
 * @param[in] in Extended input gauge field.
 * @param[in] rho Stout coefficient.
 * @param[in] epsilon Over-improvement coefficient.
 * @param[in] dir_ignore Direction excluded from smearing.
 * @param[in] anisotropy Temporal anisotropy.
 * @param[in] over_improved Select the rectangle-improved action.
 * @param[in] lat Extended lattice metadata.
 */
template <typename real_t> struct StoutSmear {
  void operator()(quda::GaugeField &out, const quda::GaugeField &in, double rho, double epsilon, int dir_ignore,
                  double anisotropy, bool over_improved, const lattice_t &lat)
  {
    using link = matrix<real_t>;
    const auto input_ptrs = in.data_array<void *>();
    const auto output_ptrs = out.data_array<void *>();
    auto input = reinterpret_cast<const link *const *>(input_ptrs.data);
    auto output = reinterpret_cast<link *const *>(output_ptrs.data);
    const int smear_dim = dir_ignore == 4 ? 4 : 3;
    const auto rho_ = static_cast<real_t>(rho);
    const auto epsilon_ = static_cast<real_t>(epsilon);
    const auto anisotropy_ = static_cast<real_t>(anisotropy);
    const auto staple_coeff = static_cast<real_t>(rho_ * (5.0 - 2.0 * epsilon_) / 3.0);
    const auto rectangle_coeff = static_cast<real_t>(rho_ * (1.0 - epsilon_) / 12.0);

#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      for (int d = 0; d < smear_dim; d++) {
        const int nu = d + (d >= dir_ignore);
        const auto &u = centered_link(input, i, nu, lat);
        auto omega = rho_ * staple(input, i, nu, dir_ignore, anisotropy_, lat);
        if (over_improved) {
          omega = (staple_coeff * staple(input, i, nu, dir_ignore, anisotropy_, lat)
                   - rectangle_coeff * rectangle(input, i, nu, dir_ignore, anisotropy_, lat));
        }
        output[nu][i] = stout_update(omega * conj(u), u);
      }
    }
  }
};

/**
 * @brief Produce the first HYP tensor level from thin links.
 *
 * @param[out] out HYP tensors indexed by excluded direction.
 * @param[in] in Extended input gauge field.
 * @param[in] alpha HYP level coefficient.
 * @param[in] dir_ignore Direction excluded for three-dimensional HYP.
 * @param[in] lat Extended lattice metadata.
 */
template <typename real_t> struct HYPLevel1 {
  void operator()(quda::GaugeField *const *out, const quda::GaugeField &in, double alpha, int dir_ignore,
                  const lattice_t &lat)
  {
    using link = matrix<real_t>;
    const auto input_ptrs = in.data_array<void *>();
    auto input = reinterpret_cast<const link *const *>(input_ptrs.data);
    std::array<std::array<link *, 4>, 4> output;
    for (int slot = 0; slot < 4; slot++) {
      const auto ptrs = out[slot]->data_array<void *>();
      auto field = reinterpret_cast<link *const *>(ptrs.data);
      for (int dir = 0; dir < 4; dir++) output[slot][dir] = field[dir];
    }
    const auto alpha_ = static_cast<real_t>(alpha);
    const auto identity = Identity<3, std::complex<real_t>>()();
    const auto tolerance = static_cast<real_t>(in.toleranceSU3());

#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      for (int mu = 0; mu < 4; mu++) {
        if (mu == dir_ignore) continue;
        const auto &u = centered_link(input, i, mu, lat);
        for (int nu = 0; nu < 4; nu++) {
          if (nu == mu || nu == dir_ignore) continue;
          auto test_u = (static_cast<real_t>(1.0) - alpha_) * identity
            + static_cast<real_t>(alpha_ / 2) * mixed_staple(input, input, i, mu, nu, lat) * conj(u);
          polar_su3(test_u, tolerance);
          output[nu][mu][i] = test_u * u;
        }
      }
    }
  }
};

/**
 * @brief Produce the second HYP tensor level for four-dimensional HYP.
 *
 * @param[out] out HYP tensors indexed by excluded direction.
 * @param[in] in First-level extended HYP tensors.
 * @param[in] thin Extended thin gauge field.
 * @param[in] alpha HYP level coefficient.
 * @param[in] lat Extended lattice metadata.
 */
template <typename real_t> struct HYPLevel2 {
  void operator()(quda::GaugeField *const *out, quda::GaugeField *const *in, const quda::GaugeField &thin,
                  double alpha, const lattice_t &lat)
  {
    using link = matrix<real_t>;
    std::array<std::array<const link *, 4>, 4> input;
    std::array<std::array<link *, 4>, 4> output;
    for (int slot = 0; slot < 4; slot++) {
      const auto input_ptrs = in[slot]->data_array<void *>();
      const auto output_ptrs = out[slot]->data_array<void *>();
      auto input_field = reinterpret_cast<const link *const *>(input_ptrs.data);
      auto output_field = reinterpret_cast<link *const *>(output_ptrs.data);
      for (int dir = 0; dir < 4; dir++) {
        input[slot][dir] = input_field[dir];
        output[slot][dir] = output_field[dir];
      }
    }
    const auto thin_ptrs = thin.data_array<void *>();
    auto thin_links = reinterpret_cast<const link *const *>(thin_ptrs.data);
    const auto alpha_ = static_cast<real_t>(alpha);
    const auto identity = Identity<3, std::complex<real_t>>()();
    const auto tolerance = static_cast<real_t>(thin.toleranceSU3());

#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      for (int mu = 0; mu < 4; mu++) {
        const auto &u = centered_link(thin_links, i, mu, lat);
        for (int nu = 0; nu < 4; nu++) {
          if (nu == mu) continue;
          link staple_sum;
          for (int rho = 0; rho < 4; rho++) {
            if (rho == mu || rho == nu) continue;
            int sigma = 0;
            while (sigma == mu || sigma == nu || sigma == rho) sigma++;
            staple_sum += mixed_staple(input[sigma].data(), input[sigma].data(), i, mu, rho, lat);
          }
          auto test_u = (static_cast<real_t>(1.0) - alpha_) * identity
            + static_cast<real_t>(alpha_ / 4) * staple_sum * conj(u);
          polar_su3(test_u, tolerance);
          output[nu][mu][i] = test_u * u;
        }
      }
    }
  }
};

/**
 * @brief Apply the final HYP projection from a tensor level.
 *
 * @param[out] out Smeared gauge field.
 * @param[in] tensors Extended HYP tensors indexed by excluded direction.
 * @param[in] thin Extended thin gauge field.
 * @param[in] alpha HYP level coefficient.
 * @param[in] dir_ignore Direction excluded for three-dimensional HYP.
 * @param[in] lat Extended lattice metadata.
 */
template <typename real_t> struct HYPFinal {
  void operator()(quda::GaugeField &out, quda::GaugeField *const *tensors, const quda::GaugeField &thin, double alpha,
                  int dir_ignore, const lattice_t &lat)
  {
    using link = matrix<real_t>;
    std::array<std::array<const link *, 4>, 4> input;
    for (int slot = 0; slot < 4; slot++) {
      const auto ptrs = tensors[slot]->data_array<void *>();
      auto field = reinterpret_cast<const link *const *>(ptrs.data);
      for (int dir = 0; dir < 4; dir++) input[slot][dir] = field[dir];
    }
    const auto thin_ptrs = thin.data_array<void *>();
    const auto output_ptrs = out.data_array<void *>();
    auto thin_links = reinterpret_cast<const link *const *>(thin_ptrs.data);
    auto output = reinterpret_cast<link *const *>(output_ptrs.data);
    const int hyp_dim = dir_ignore == 4 ? 4 : 3;
    const auto alpha_ = static_cast<real_t>(alpha);
    const auto scale = static_cast<real_t>(alpha_ / (hyp_dim == 4 ? 6 : 4));
    const auto identity = Identity<3, std::complex<real_t>>()();
    const auto tolerance = static_cast<real_t>(thin.toleranceSU3());

#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      for (int mu = 0; mu < 4; mu++) {
        if (mu == dir_ignore) continue;
        link staple_sum;
        for (int nu = 0; nu < 4; nu++) {
          if (nu == mu || nu == dir_ignore) continue;
          if (hyp_dim == 4) {
            staple_sum += mixed_staple(input[nu].data(), input[mu].data(), i, mu, nu, lat);
          } else {
            int rho = 0;
            while (rho == mu || rho == nu || rho == dir_ignore) rho++;
            staple_sum += mixed_staple(input[rho].data(), input[rho].data(), i, mu, nu, lat);
          }
        }
        const auto &u = centered_link(thin_links, i, mu, lat);
        auto test_u = (static_cast<real_t>(1.0) - alpha_) * identity + scale * staple_sum * conj(u);
        polar_su3(test_u, tolerance);
        output[mu][i] = test_u * u;
      }
    }
  }
};

/**
 * @brief Apply a one-step HYP smear using temporary tensor fields.
 *
 * @param[out] out Smeared gauge field.
 * @param[in] in Extended input gauge field.
 * @param[in] gauge_param Gauge field parameters used for temporary fields.
 * @param[in] R Extended border radius.
 * @param[in] smear_param HYP parameters.
 * @param[in] lat Extended lattice metadata.
 */
void hyp_smear(quda::GaugeField &out, const quda::GaugeField &in, QudaGaugeParam &gauge_param,
               const quda::lat_dim_t &R, const QudaGaugeSmearParam &smear_param, const lattice_t &lat)
{
  quda::GaugeFieldParam field_param(out);
  field_param.location = QUDA_CPU_FIELD_LOCATION;
  field_param.create = QUDA_NULL_FIELD_CREATE;
  std::array<std::unique_ptr<quda::GaugeField>, 4> level1;
  std::array<std::unique_ptr<quda::GaugeField>, 4> level2;
  std::array<quda::GaugeField *, 4> level1_ptr {};
  std::array<quda::GaugeField *, 4> level2_ptr {};
  for (int slot = 0; slot < 4; slot++) {
    level1[slot] = std::make_unique<quda::GaugeField>(field_param);
    level2[slot] = std::make_unique<quda::GaugeField>(field_param);
    level1_ptr[slot] = level1[slot].get();
    level2_ptr[slot] = level2[slot].get();
  }

  instantiate_host<HYPLevel1>(in.Precision(), level1_ptr.data(), in, smear_param.alpha3, smear_param.dir_ignore, lat);
  std::array<quda::GaugeField *, 4> level1_ex {};
  for (int slot = 0; slot < 4; slot++)
    level1_ex[slot] = quda::createExtendedGauge(level1[slot]->data_array().data, gauge_param, R);

  if (smear_param.dir_ignore == 4) {
    instantiate_host<HYPLevel2>(in.Precision(), level2_ptr.data(), level1_ex.data(), in, smear_param.alpha2, lat);
    std::array<quda::GaugeField *, 4> level2_ex {};
    for (int slot = 0; slot < 4; slot++)
      level2_ex[slot] = quda::createExtendedGauge(level2[slot]->data_array().data, gauge_param, R);
    instantiate_host<HYPFinal>(in.Precision(), out, level2_ex.data(), in, smear_param.alpha1, smear_param.dir_ignore, lat);
    for (auto field : level2_ex) delete field;
  } else {
    instantiate_host<HYPFinal>(in.Precision(), out, level1_ex.data(), in, smear_param.alpha2, smear_param.dir_ignore, lat);
  }
  for (auto field : level1_ex) delete field;
}

template <typename real_t>
matrix<real_t> flow_action(const matrix<real_t> *const *links, size_t i, int dir, QudaGaugeSmearType type,
                           double anisotropy, const lattice_t &lat)
{
  if (type == QUDA_GAUGE_SMEAR_WILSON_FLOW) return staple(links, i, dir, 4, anisotropy, lat);
  return static_cast<real_t>(5.0 / 3.0) * staple(links, i, dir, 4, anisotropy, lat)
    - static_cast<real_t>(1.0 / 12.0) * rectangle(links, i, dir, 4, anisotropy, lat);
}

template <typename real_t> using flow_temp = std::vector<std::array<matrix<real_t>, 4>>;

template <typename real_t> struct WFlowW1 {
  void operator()(quda::GaugeField &out, const quda::GaugeField &in, flow_temp<real_t> &temp, real_t epsilon,
                  real_t anisotropy, QudaGaugeSmearType type, const lattice_t &lat)
  {
    using link = matrix<real_t>;
    const auto input_ptrs = in.data_array<void *>();
    const auto output_ptrs = out.data_array<void *>();
    auto input = reinterpret_cast<const link *const *>(input_ptrs.data);
    auto output = reinterpret_cast<link *const *>(output_ptrs.data);
#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      for (int dir = 0; dir < 4; dir++) {
        const auto &u = centered_link(input, i, dir, lat);
        const auto z = flow_action(input, i, dir, type, anisotropy, lat) * conj(u);
        temp[i][dir] = z;
        output[dir][i] = stout_update(static_cast<real_t>(1.0 / 4.0) * epsilon * z, u);
      }
    }
  }
};

template <typename real_t> struct WFlowW2 {
  void operator()(quda::GaugeField &out, const quda::GaugeField &in, flow_temp<real_t> &temp, real_t epsilon,
                  real_t anisotropy, QudaGaugeSmearType type, const lattice_t &lat)
  {
    using link = matrix<real_t>;
    const auto input_ptrs = in.data_array<void *>();
    const auto output_ptrs = out.data_array<void *>();
    auto input = reinterpret_cast<const link *const *>(input_ptrs.data);
    auto output = reinterpret_cast<link *const *>(output_ptrs.data);
#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      for (int dir = 0; dir < 4; dir++) {
        const auto &u = centered_link(input, i, dir, lat);
        auto z = static_cast<real_t>(8.0 / 9.0) * flow_action(input, i, dir, type, anisotropy, lat) * conj(u);
        z = z - static_cast<real_t>(17.0 / 36.0) * temp[i][dir];
        temp[i][dir] = z;
        output[dir][i] = stout_update(epsilon * z, u);
      }
    }
  }
};

template <typename real_t> struct WFlowVt {
  void operator()(quda::GaugeField &out, const quda::GaugeField &in, const flow_temp<real_t> &temp, real_t epsilon,
                  real_t anisotropy, QudaGaugeSmearType type, const lattice_t &lat)
  {
    using link = matrix<real_t>;
    const auto input_ptrs = in.data_array<void *>();
    const auto output_ptrs = out.data_array<void *>();
    auto input = reinterpret_cast<const link *const *>(input_ptrs.data);
    auto output = reinterpret_cast<link *const *>(output_ptrs.data);
#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      for (int dir = 0; dir < 4; dir++) {
        const auto &u = centered_link(input, i, dir, lat);
        auto z = static_cast<real_t>(3.0 / 4.0) * flow_action(input, i, dir, type, anisotropy, lat) * conj(u);
        z = z - temp[i][dir];
        output[dir][i] = stout_update(epsilon * z, u);
      }
    }
  }
};

/**
 * @brief Apply one fourth-order Wilson-flow Runge--Kutta stage.
 *
 * @param[out] out Flowed gauge field.
 * @param[in] in Extended input gauge field.
 * @param[in,out] temp Stored flow generators from prior stages.
 * @param[in] epsilon Flow step size.
 * @param[in] anisotropy Temporal anisotropy.
 * @param[in] type Wilson or Symanzik flow action.
 * @param[in] lat Extended lattice metadata.
 * @param[in] coeff_a Coefficient on the stored generator.
 * @param[in] coeff_b Coefficient on the current generator update.
 * @param[in] get_stored Whether to subtract the stored generator.
 * @param[in] do_store Whether to store the current generator.
 */
template <typename real_t> struct WFlowRK4Step {
  void operator()(quda::GaugeField &out, const quda::GaugeField &in, flow_temp<real_t> &temp, real_t epsilon,
                  real_t anisotropy, QudaGaugeSmearType type, const lattice_t &lat, double coeff_a, double coeff_b,
                  bool get_stored, bool do_store)
  {
    using link = matrix<real_t>;
    const auto input_ptrs = in.data_array<void *>();
    const auto output_ptrs = out.data_array<void *>();
    auto input = reinterpret_cast<const link *const *>(input_ptrs.data);
    auto output = reinterpret_cast<link *const *>(output_ptrs.data);
#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      for (int dir = 0; dir < 4; dir++) {
        const auto &u = centered_link(input, i, dir, lat);
        auto z = flow_action(input, i, dir, type, anisotropy, lat) * conj(u);
        if (get_stored) z -= static_cast<real_t>(coeff_a) * temp[i][dir];
        if (do_store) temp[i][dir] = z;
        output[dir][i] = stout_update(static_cast<real_t>(coeff_b) * epsilon * z, u);
      }
    }
  }
};

template <typename real_t> struct WFlowRK3 {
  void operator()(quda::GaugeField &out, const quda::GaugeField &in, QudaGaugeParam &gauge_param,
                  const quda::lat_dim_t &R, const QudaGaugeSmearParam &smear_param, const lattice_t &lat)
  {
    const auto epsilon = static_cast<real_t>(smear_param.epsilon);
    const auto anisotropy = static_cast<real_t>(smear_param.smear_anisotropy);
    const auto type = smear_param.smear_type;
    flow_temp<real_t> temp(lat.volume);

    quda::GaugeFieldParam field_param(out);
    field_param.location = QUDA_CPU_FIELD_LOCATION;
    field_param.create = QUDA_NULL_FIELD_CREATE;
    quda::GaugeField stage_two(field_param);

    WFlowW1<real_t>{}(out, in, temp, epsilon, anisotropy, type, lat);
    auto stage_one_ex = quda::createExtendedGauge(out.data_array().data, gauge_param, R);
    WFlowW2<real_t>{}(stage_two, *stage_one_ex, temp, epsilon, anisotropy, type, lat);
    delete stage_one_ex;
    auto stage_two_ex = quda::createExtendedGauge(stage_two.data_array().data, gauge_param, R);
    WFlowVt<real_t>{}(out, *stage_two_ex, temp, epsilon, anisotropy, type, lat);
    delete stage_two_ex;
  }
};

template <typename real_t> struct WFlowRK4 {
  void operator()(quda::GaugeField &out, const quda::GaugeField &in, QudaGaugeParam &gauge_param,
                  const quda::lat_dim_t &R, const QudaGaugeSmearParam &smear_param, const lattice_t &lat)
  {
    struct rk4_stage {
      double coeff_a;
      double coeff_b;
      bool get_stored;
      bool do_store;
    };
    static constexpr std::array<rk4_stage, 6> stages {{
      {0.0, 0.032918605146, false, true},
      {0.737101392796, 0.823256998200, true, true},
      {1.634740794341, 0.381530948900, true, true},
      {0.744739003780, 0.200092213184, true, true},
      {1.469897351522, 1.718581042715, true, true},
      {2.813971388035, 0.27, true, false},
    }};

    const auto epsilon = static_cast<real_t>(smear_param.epsilon);
    const auto anisotropy = static_cast<real_t>(smear_param.smear_anisotropy);
    const auto type = smear_param.smear_type;
    flow_temp<real_t> temp(lat.volume);

    quda::GaugeFieldParam field_param(out);
    field_param.location = QUDA_CPU_FIELD_LOCATION;
    field_param.create = QUDA_NULL_FIELD_CREATE;
    quda::GaugeField stage(field_param);

    WFlowRK4Step<real_t>{}(out, in, temp, epsilon, anisotropy, type, lat, stages[0].coeff_a, stages[0].coeff_b,
                            stages[0].get_stored, stages[0].do_store);
    auto out_ex = quda::createExtendedGauge(out.data_array().data, gauge_param, R);
    WFlowRK4Step<real_t>{}(stage, *out_ex, temp, epsilon, anisotropy, type, lat, stages[1].coeff_a, stages[1].coeff_b,
                            stages[1].get_stored, stages[1].do_store);
    delete out_ex;
    auto stage_ex = quda::createExtendedGauge(stage.data_array().data, gauge_param, R);
    WFlowRK4Step<real_t>{}(out, *stage_ex, temp, epsilon, anisotropy, type, lat, stages[2].coeff_a, stages[2].coeff_b,
                            stages[2].get_stored, stages[2].do_store);
    delete stage_ex;
    out_ex = quda::createExtendedGauge(out.data_array().data, gauge_param, R);
    WFlowRK4Step<real_t>{}(stage, *out_ex, temp, epsilon, anisotropy, type, lat, stages[3].coeff_a, stages[3].coeff_b,
                            stages[3].get_stored, stages[3].do_store);
    delete out_ex;
    stage_ex = quda::createExtendedGauge(stage.data_array().data, gauge_param, R);
    WFlowRK4Step<real_t>{}(out, *stage_ex, temp, epsilon, anisotropy, type, lat, stages[4].coeff_a, stages[4].coeff_b,
                            stages[4].get_stored, stages[4].do_store);
    delete stage_ex;
    out_ex = quda::createExtendedGauge(out.data_array().data, gauge_param, R);
    WFlowRK4Step<real_t>{}(stage, *out_ex, temp, epsilon, anisotropy, type, lat, stages[5].coeff_a, stages[5].coeff_b,
                            stages[5].get_stored, stages[5].do_store);
    delete out_ex;
    out.copy(stage);
  }
};

/**
 * @brief Apply one third-order Wilson-flow integration cycle.
 *
 * @param[out] out Flowed gauge field.
 * @param[in] in Extended input gauge field.
 * @param[in] gauge_param Gauge field parameters used for temporary fields.
 * @param[in] R Extended border radius.
 * @param[in] smear_param Flow parameters.
 * @param[in] lat Extended lattice metadata.
 */
void wflow_smear_rk3(quda::GaugeField &out, const quda::GaugeField &in, QudaGaugeParam &gauge_param,
                     const quda::lat_dim_t &R, const QudaGaugeSmearParam &smear_param, const lattice_t &lat)
{
  instantiate_host<WFlowRK3>(in.Precision(), out, in, gauge_param, R, smear_param, lat);
}

/**
 * @brief Apply one fourth-order Wilson-flow integration cycle.
 *
 * @param[out] out Flowed gauge field.
 * @param[in] in Extended input gauge field.
 * @param[in] gauge_param Gauge field parameters used for temporary fields.
 * @param[in] R Extended border radius.
 * @param[in] smear_param Flow parameters.
 * @param[in] lat Extended lattice metadata.
 */
void wflow_smear_rk4(quda::GaugeField &out, const quda::GaugeField &in, QudaGaugeParam &gauge_param,
                     const quda::lat_dim_t &R, const QudaGaugeSmearParam &smear_param, const lattice_t &lat)
{
  instantiate_host<WFlowRK4>(in.Precision(), out, in, gauge_param, R, smear_param, lat);
}

} // namespace

void gauge_smear_reference(quda::GaugeField &out, const quda::GaugeField &in, const QudaGaugeSmearParam &smear_param)
{
  if (smear_param.n_steps != 1) errorQuda("Host gauge-smear reference supports one step, received %u", smear_param.n_steps);
  if ((smear_param.smear_type == QUDA_GAUGE_SMEAR_WILSON_FLOW
       || smear_param.smear_type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW)
      && smear_param.rk_order != 3 && smear_param.rk_order != 4)
    errorQuda("Host flow reference supports third- and fourth-order RK, received order %d", smear_param.rk_order);

  auto reference_param = smear_param;
  if (reference_param.smear_type == QUDA_GAUGE_SMEAR_APE || reference_param.smear_type == QUDA_GAUGE_SMEAR_STOUT) {
    if (reference_param.dir_ignore < 0) reference_param.dir_ignore = 3;
  } else if (reference_param.smear_type == QUDA_GAUGE_SMEAR_OVRIMP_STOUT
             || reference_param.smear_type == QUDA_GAUGE_SMEAR_HYP) {
    if (reference_param.dir_ignore < 0 || reference_param.dir_ignore > 3) reference_param.dir_ignore = 4;
  }

  quda::lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  auto input_ex = quda::createExtendedGauge(in.data_array().data, gauge_param, R);
  lattice_t lat(*input_ex);

  out.copy(in);
  switch (reference_param.smear_type) {
  case QUDA_GAUGE_SMEAR_APE:
    instantiate_host<APESmear>(in.Precision(), out, *input_ex, reference_param.alpha, reference_param.dir_ignore,
                                reference_param.smear_anisotropy, lat);
    break;
  case QUDA_GAUGE_SMEAR_STOUT:
    instantiate_host<StoutSmear>(in.Precision(), out, *input_ex, reference_param.rho, reference_param.epsilon,
                                  reference_param.dir_ignore, reference_param.smear_anisotropy, false, lat);
    break;
  case QUDA_GAUGE_SMEAR_OVRIMP_STOUT:
    instantiate_host<StoutSmear>(in.Precision(), out, *input_ex, reference_param.rho, reference_param.epsilon,
                                  reference_param.dir_ignore, reference_param.smear_anisotropy, true, lat);
    break;
  case QUDA_GAUGE_SMEAR_HYP:
    hyp_smear(out, *input_ex, gauge_param, R, reference_param, lat);
    break;
  case QUDA_GAUGE_SMEAR_WILSON_FLOW:
  case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW:
    if (reference_param.rk_order == 4)
      wflow_smear_rk4(out, *input_ex, gauge_param, R, reference_param, lat);
    else
      wflow_smear_rk3(out, *input_ex, gauge_param, R, reference_param, lat);
    break;
  default: errorQuda("Unsupported host gauge smear type %d", reference_param.smear_type);
  }
  delete input_ex;
}
