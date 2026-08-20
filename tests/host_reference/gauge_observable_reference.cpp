#include <array>
#include <complex>

#include "force_utils.hpp"
#include "gauge_force_reference.h"
#include "gauge_observable_reference.h"
#include "host_utils.h"
#include "instantiate_host.hpp"
#include "misc.h"

template <typename real_t> using matrix = Matrix<3, std::complex<real_t>>;

template <typename real_t>
double plaquette(const matrix<real_t> *const *links, size_t i, int mu, int nu, const lattice_t &lat)
{
  std::array<int, 4> dx {};
  const auto &u1 = links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[mu]++;
  const auto &u2 = links[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[mu]--;
  dx[nu]++;
  const auto &u3 = links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[nu]--;
  const auto &u4 = links[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  return trace(u1 * u2 * conj(u3) * conj(u4)).real();
}

template <typename real_t>
double rectangle(const matrix<real_t> *const *links, size_t i, int mu, int nu, const lattice_t &lat)
{
  std::array<int, 4> dx {};
  auto mu_long = links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[mu]++;
  mu_long = mu_long * links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[mu]++;
  mu_long = mu_long * links[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[mu]--;
  dx[nu]++;
  mu_long = mu_long * conj(links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)]);
  dx[mu]--;
  mu_long = mu_long * conj(links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)]);
  dx[nu]--;
  mu_long = mu_long * conj(links[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)]);

  dx = {};
  auto nu_long = links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[mu]++;
  nu_long = nu_long * links[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[nu]++;
  nu_long = nu_long * links[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)];
  dx[mu]--;
  dx[nu]++;
  nu_long = nu_long * conj(links[mu][gf_neighborIndexFullLattice(i, dx.data(), lat)]);
  dx[nu]--;
  nu_long = nu_long * conj(links[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)]);
  dx[nu]--;
  nu_long = nu_long * conj(links[nu][gf_neighborIndexFullLattice(i, dx.data(), lat)]);

  return trace(mu_long + nu_long).real();
}

template <typename real_t> struct PlaquetteReference {
  void operator()(std::array<double, 2> &result, const quda::GaugeField &u, const lattice_t &lat)
  {
    const auto ptrs = u.data_array<void *>();
    const auto links = reinterpret_cast<const matrix<real_t> *const *>(ptrs.data);
    double spatial = 0.0;
    double temporal = 0.0;

#pragma omp parallel for reduction(+ : spatial, temporal)
    for (size_t i = 0; i < lat.volume; i++) {
      for (int mu = 0; mu < 3; mu++) {
        for (int nu = mu + 1; nu < 3; nu++) spatial += plaquette(links, i, mu, nu, lat);
        temporal += plaquette(links, i, mu, 3, lat);
      }
    }

    result = {spatial, temporal};
  }
};

template <typename real_t> struct PlaquetteRectangleReferenceCompute {
  void operator()(std::array<double, 4> &result, const quda::GaugeField &u, const lattice_t &lat)
  {
    const auto ptrs = u.data_array<void *>();
    const auto links = reinterpret_cast<const matrix<real_t> *const *>(ptrs.data);
    double spatial_plaquette = 0.0;
    double temporal_plaquette = 0.0;
    double spatial_rectangle = 0.0;
    double temporal_rectangle = 0.0;

#pragma omp parallel for reduction(+ : spatial_plaquette, temporal_plaquette, spatial_rectangle, temporal_rectangle)
    for (size_t i = 0; i < lat.volume; i++) {
      for (int mu = 0; mu < 3; mu++) {
        for (int nu = mu + 1; nu < 3; nu++) {
          spatial_plaquette += plaquette(links, i, mu, nu, lat);
          spatial_rectangle += rectangle(links, i, mu, nu, lat);
        }
        temporal_plaquette += plaquette(links, i, mu, 3, lat);
        temporal_rectangle += rectangle(links, i, mu, 3, lat);
      }
    }

    result = {spatial_plaquette, temporal_plaquette, spatial_rectangle, temporal_rectangle};
  }
};

std::array<double, 3> plaquette_reference(const quda::GaugeField &u)
{
  quda::lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  auto u_ex = quda::createExtendedGauge(u.data_array().data, gauge_param, R);
  lattice_t lat(*u_ex);

  std::array<double, 2> sums {};
  instantiate_host<PlaquetteReference>(u.Precision(), sums, *u_ex, lat);
  quda::comm_allreduce_sum(sums[0]);
  quda::comm_allreduce_sum(sums[1]);

  const double normalization = 9.0 * lat.volume * quda::comm_size();
  const double spatial = sums[0] / normalization;
  const double temporal = sums[1] / normalization;
  delete u_ex;
  return {0.5 * (spatial + temporal), spatial, temporal};
}

PlaquetteRectangleReference plaquette_rectangle_reference(const quda::GaugeField &u)
{
  quda::lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  auto u_ex = quda::createExtendedGauge(u.data_array().data, gauge_param, R);
  lattice_t lat(*u_ex);

  std::array<double, 4> sums {};
  instantiate_host<PlaquetteRectangleReferenceCompute>(u.Precision(), sums, *u_ex, lat);
  for (auto &sum : sums) quda::comm_allreduce_sum(sum);

  const double plaquette_normalization = 9.0 * lat.volume * quda::comm_size();
  const double rectangle_normalization = 18.0 * lat.volume * quda::comm_size();
  const double spatial_plaquette = sums[0] / plaquette_normalization;
  const double temporal_plaquette = sums[1] / plaquette_normalization;
  const double spatial_rectangle = sums[2] / rectangle_normalization;
  const double temporal_rectangle = sums[3] / rectangle_normalization;
  delete u_ex;

  return {{0.5 * (spatial_plaquette + temporal_plaquette), spatial_plaquette, temporal_plaquette},
          {0.5 * (spatial_rectangle + temporal_rectangle), spatial_rectangle, temporal_rectangle}};
}
