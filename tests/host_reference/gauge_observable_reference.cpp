#include <array>
#include <complex>
#include <vector>

#include <comm_quda.h>
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

template <typename real_t> struct PolyakovLoopReferenceCompute {
  std::array<double, 2> operator()(const quda::GaugeField &u)
  {
    const int x0 = u.X()[0];
    const int x1 = u.X()[1];
    const int x2 = u.X()[2];
    const int x3 = u.X()[3];
    const size_t volume = u.Volume();
    const size_t spatial_volume = volume / x3;
    const auto ptrs = u.data_array<void *>();
    const auto links = reinterpret_cast<const matrix<real_t> *const *>(ptrs.data);

    std::vector<matrix<real_t>> local_product(spatial_volume);
#pragma omp parallel for
    for (size_t spatial_index = 0; spatial_index < spatial_volume; spatial_index++) {
      size_t index = spatial_index;
      const int x = index % x0;
      index /= x0;
      const int y = index % x1;
      const int z = index / x1;
      const int parity = (x + y + z) & 1;
      const size_t half_index = ((z * x1 + y) * (x0 / 2)) + x / 2;

      auto product = Identity<3, std::complex<real_t>>()();
      for (int t = 0; t < x3; t++) {
        const size_t temporal_half_index = half_index + static_cast<size_t>(t) * x0 * x1 * x2 / 2;
        const int temporal_parity = parity ^ (t & 1);
        const size_t temporal_index = temporal_half_index + temporal_parity * volume / 2;
        product = product * links[3][temporal_index];
      }
      local_product[spatial_index] = product;
    }

    const int temporal_ranks = quda::comm_dim(3);
    const int temporal_coordinate = quda::comm_coord(3);
    std::vector<std::vector<matrix<real_t>>> rank_product(temporal_ranks, std::vector<matrix<real_t>>(spatial_volume));
    rank_product[temporal_coordinate] = local_product;

    if (temporal_ranks > 1) {
      std::array<std::vector<matrix<real_t>>, 2> buffer {std::vector<matrix<real_t>>(spatial_volume),
                                                         std::vector<matrix<real_t>>(spatial_volume)};
      buffer[0] = local_product;
      const size_t bytes = spatial_volume * sizeof(matrix<real_t>);
      std::array<quda::MsgHandle *, 2> receive;
      std::array<quda::MsgHandle *, 2> send;
      for (int i = 0; i < 2; i++) {
        receive[i] = quda::comm_declare_receive_relative(buffer[i].data(), 3, 1, bytes);
        send[i] = quda::comm_declare_send_relative(buffer[i].data(), 3, -1, bytes);
      }

      int send_buffer = 0;
      int receive_buffer = 1;
      for (int step = 1; step < temporal_ranks; step++) {
        quda::comm_start(receive[receive_buffer]);
        quda::comm_start(send[send_buffer]);
        quda::comm_wait(receive[receive_buffer]);
        quda::comm_wait(send[send_buffer]);
        rank_product[(temporal_coordinate + step) % temporal_ranks] = buffer[receive_buffer];
        std::swap(send_buffer, receive_buffer);
      }
      for (int i = 0; i < 2; i++) {
        quda::comm_free(receive[i]);
        quda::comm_free(send[i]);
      }
    }

    double real = 0.0;
    double imaginary = 0.0;
#pragma omp parallel for reduction(+ : real, imaginary)
    for (size_t spatial_index = 0; spatial_index < spatial_volume; spatial_index++) {
      auto product = Identity<3, std::complex<real_t>>()();
      for (int t = 0; t < temporal_ranks; t++) product = product * rank_product[t][spatial_index];
      const auto value = trace(product);
      real += value.real();
      imaginary += value.imag();
    }

    quda::comm_allreduce_sum(real);
    quda::comm_allreduce_sum(imaginary);
    const double normalization = spatial_volume * quda::comm_size();
    return {real / normalization, imaginary / normalization};
  }
};

template <typename real_t> struct LinkDeterminantTraceReferenceCompute {
  void operator()(std::array<double, 6> &result, const quda::GaugeField &u)
  {
    const auto ptrs = u.data_array<void *>();
    const auto links = reinterpret_cast<const matrix<real_t> *const *>(ptrs.data);
    double determinant_real = 0.0;
    double determinant_imaginary = 0.0;
    double trace_real = 0.0;
    double trace_imaginary = 0.0;
    double determinant_scale = 0.0;
    double trace_scale = 0.0;

#pragma omp parallel for reduction(+ : determinant_real, determinant_imaginary, trace_real, trace_imaginary,           \
                                     determinant_scale, trace_scale)
    for (size_t i = 0; i < u.Volume(); i++) {
      for (int dir = 0; dir < 4; dir++) {
        const auto determinant = links[dir][i].determinant();
        const auto link_trace = trace(links[dir][i]);
        determinant_real += determinant.real();
        determinant_imaginary += determinant.imag();
        trace_real += link_trace.real();
        trace_imaginary += link_trace.imag();
        determinant_scale += std::abs(determinant);
        trace_scale += std::abs(link_trace);
      }
    }

    result = {determinant_real, determinant_imaginary, trace_real, trace_imaginary, determinant_scale, trace_scale};
  }
};

template <typename real_t>
const matrix<real_t> &shifted_link(const matrix<real_t> *const *links, size_t i, int dir,
                                   const std::array<int, 4> &shift, const lattice_t &lat)
{
  auto dx = shift;
  return links[dir][gf_neighborIndexFullLattice(i, dx.data(), lat)];
}

template <typename real_t>
matrix<real_t> clover_fmunu(const matrix<real_t> *const *links, size_t i, int mu, int nu, const lattice_t &lat)
{
  auto link = [&](int dir, int delta_mu, int delta_nu) -> const matrix<real_t> & {
    std::array<int, 4> dx {};
    dx[mu] = delta_mu;
    dx[nu] = delta_nu;
    return shifted_link(links, i, dir, dx, lat);
  };

  auto f = link(mu, 0, 0) * link(nu, 1, 0) * conj(link(mu, 0, 1)) * conj(link(nu, 0, 0));
  f += link(nu, 0, 0) * conj(link(mu, -1, 1)) * conj(link(nu, -1, 0)) * link(mu, -1, 0);
  f += conj(link(nu, 0, -1)) * link(mu, 0, -1) * link(nu, 1, -1) * conj(link(mu, 0, 0));
  f += conj(link(mu, -1, 0)) * conj(link(nu, -1, -1)) * link(mu, -1, -1) * link(nu, 0, -1);
  return static_cast<real_t>(0.125) * (f - conj(f));
}

template <typename real_t> struct FieldStrengthReferenceCompute {
  void operator()(quda::GaugeField &out, const quda::GaugeField &u, const lattice_t &lat)
  {
    const auto input_ptrs = u.data_array<void *>();
    auto links = reinterpret_cast<const matrix<real_t> *const *>(input_ptrs.data);
    const auto output_ptrs = out.data_array<void *>();
    auto fmunu = reinterpret_cast<matrix<real_t> *const *>(output_ptrs.data);
    constexpr std::array<std::array<int, 2>, 6> directions {{{1, 0}, {2, 0}, {2, 1}, {3, 0}, {3, 1}, {3, 2}}};

#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      for (int component = 0; component < 6; component++)
        fmunu[component][i] = clover_fmunu(links, i, directions[component][0], directions[component][1], lat);
    }
  }
};

template <typename real_t> struct FieldStrengthObservableReferenceCompute {
  FieldStrengthObservableReference operator()(const quda::GaugeField &fmunu)
  {
    const auto ptrs = fmunu.data_array<void *>();
    auto field = reinterpret_cast<const matrix<real_t> *const *>(ptrs.data);
    const auto identity = Identity<3, std::complex<real_t>>()();
    constexpr double q_norm = -1.0 / (4.0 * M_PI * M_PI);
    double spatial_energy = 0.0;
    double temporal_energy = 0.0;
    double qcharge = 0.0;
    double qcharge_scale = 0.0;
    std::vector<double> density(fmunu.Volume());

#pragma omp parallel for reduction(+ : spatial_energy, temporal_energy, qcharge, qcharge_scale)
    for (size_t i = 0; i < fmunu.Volume(); i++) {
      std::array<matrix<real_t>, 6> traceless;
      for (int component = 0; component < 6; component++) {
        traceless[component]
          = field[component][i] - static_cast<real_t>(1.0 / 3.0) * trace(field[component][i]) * identity;
        const double contribution = -trace(traceless[component] * traceless[component]).real();
        if (component < 3)
          spatial_energy += contribution;
        else
          temporal_energy += contribution;
      }
      const double q_site = q_norm
        * (trace(traceless[0] * traceless[5]).real() - trace(traceless[1] * traceless[4]).real()
           + trace(traceless[2] * traceless[3]).real());
      density[i] = q_site;
      qcharge += q_site;
      qcharge_scale += std::abs(q_site);
    }

    quda::comm_allreduce_sum(spatial_energy);
    quda::comm_allreduce_sum(temporal_energy);
    quda::comm_allreduce_sum(qcharge);
    quda::comm_allreduce_sum(qcharge_scale);
    const double volume = fmunu.Volume() * quda::comm_size();
    spatial_energy /= volume;
    temporal_energy /= volume;
    return {
      {spatial_energy + temporal_energy, spatial_energy, temporal_energy}, qcharge, std::move(density), qcharge_scale};
  }
};

template <typename real_t> struct ProjectSU3Reference {
  int operator()(quda::GaugeField &u)
  {
    const auto ptrs = u.data_array<void *>();
    auto links = reinterpret_cast<matrix<real_t> *const *>(ptrs.data);
    const auto tolerance = static_cast<real_t>(u.toleranceSU3());
    int failures = 0;

#pragma omp parallel for reduction(+ : failures)
    for (size_t i = 0; i < u.Volume(); i++) {
      for (int dir = 0; dir < 4; dir++) {
        auto &link = links[dir][i];
        polar_su3(link, tolerance);
        if (!is_unitary(link.inverse(), link, tolerance)) failures++;
      }
    }

    return failures;
  }
};

int project_su3_reference(quda::GaugeField &u)
{
  return instantiate_host_reduce<ProjectSU3Reference, int>(u.Precision(), u);
}

std::array<double, 3> plaquette_reference(const quda::GaugeField &u)
{
  quda::lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  gauge_param.cpu_prec = u.Precision();
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
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
  gauge_param.cpu_prec = u.Precision();
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
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

std::array<double, 2> polyakov_loop_reference(const quda::GaugeField &u)
{
  return instantiate_host_reduce<PolyakovLoopReferenceCompute, std::array<double, 2>>(u.Precision(), u);
}

LinkDeterminantTraceReference link_determinant_trace_reference(const quda::GaugeField &u)
{
  std::array<double, 6> sums {};
  instantiate_host<LinkDeterminantTraceReferenceCompute>(u.Precision(), sums, u);
  for (auto &sum : sums) quda::comm_allreduce_sum(sum);

  const double normalization = 4.0 * u.Volume() * quda::comm_size();
  return {{sums[0] / normalization, sums[1] / normalization},
          {sums[2] / normalization, sums[3] / normalization},
          sums[4] / normalization,
          sums[5] / normalization};
}

void compute_fmunu_reference(quda::GaugeField &fmunu, const quda::GaugeField &u)
{
  quda::lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  gauge_param.cpu_prec = u.Precision();
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  auto u_ex = quda::createExtendedGauge(u.data_array().data, gauge_param, R);
  lattice_t lat(*u_ex);
  instantiate_host<FieldStrengthReferenceCompute>(u.Precision(), fmunu, *u_ex, lat);
  delete u_ex;
}

FieldStrengthObservableReference field_strength_observable_reference(const quda::GaugeField &fmunu)
{
  return instantiate_host_reduce<FieldStrengthObservableReferenceCompute, FieldStrengthObservableReference>(
    fmunu.Precision(), fmunu);
}
