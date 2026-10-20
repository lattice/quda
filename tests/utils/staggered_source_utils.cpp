#include <cmath>
#include <complex>
#include <random>
#include <string>

#include <comm_quda.h>
#include <util_quda.h>

#include "host_utils.h"
#include "staggered_source_utils.h"

const char *get_ks_source_type_str(KSSourceType type)
{
  switch (type) {
  case KSSourceType::RANDOM: return "random";
  case KSSourceType::LOAD: return "load";
  case KSSourceType::POINT: return "point";
  case KSSourceType::CORNER_WALL: return "corner-wall";
  case KSSourceType::CORNER_WALL_X: return "corner-wall-x";
  case KSSourceType::CORNER_WALL_Y: return "corner-wall-y";
  case KSSourceType::CORNER_WALL_XY: return "corner-wall-xy";
  case KSSourceType::CORNER_WALL_Z: return "corner-wall-z";
  case KSSourceType::CORNER_WALL_ZX: return "corner-wall-zx";
  case KSSourceType::CORNER_WALL_YZ: return "corner-wall-yz";
  case KSSourceType::CORNER_WALL_XYZ: return "corner-wall-xyz";
  case KSSourceType::EVEN_WALL: return "even-wall";
  case KSSourceType::EVENANDODD_WALL: return "evenandodd-wall";
  case KSSourceType::EVENMINUSODD_WALL: return "evenminusodd-wall";
  case KSSourceType::RANDOM_COMPLEX_WALL: return "random-complex-wall";
  case KSSourceType::RANDOM_COLOR_WALL: return "random-color-wall";
  }
  return "invalid";
}

const char *get_ks_source_subset_str(KSSourceSubset subset)
{
  switch (subset) {
  case KSSourceSubset::FULL: return "full";
  case KSSourceSubset::CORNER: return "corner";
  }
  return "invalid";
}

bool is_built_ks_source(KSSourceType type) { return type != KSSourceType::RANDOM && type != KSSourceType::LOAD; }

bool is_complex_field_ks_source(KSSourceType type)
{
  return is_built_ks_source(type) && type != KSSourceType::RANDOM_COLOR_WALL;
}

bool is_random_ks_source(KSSourceType type)
{
  return type == KSSourceType::RANDOM_COMPLEX_WALL || type == KSSourceType::RANDOM_COLOR_WALL;
}

namespace
{
  using complex_t = std::complex<double>;
  using coords_t = std::array<int, 4>;

  // the global coordinates of the site at host index h of a full host field on the local lattice X (QUDA's
  // checkerboard order, even sites first; the local extents are even, so local and global parities agree)
  coords_t global_coords(size_t h, const coords_t &X)
  {
    const size_t half = static_cast<size_t>(X[0]) * X[1] * X[2] * X[3] / 2;
    const int parity = h >= half;
    const size_t cb = h % half;
    coords_t x;
    const size_t za = cb / (X[0] / 2);
    x[1] = za % X[1];
    x[2] = (za / X[1]) % X[2];
    x[3] = za / (static_cast<size_t>(X[1]) * X[2]);
    x[0] = 2 * (cb % (X[0] / 2)) + ((x[1] + x[2] + x[3] + parity) & 1);
    for (int d = 0; d < 4; d++) x[d] += quda::comm_coord(d) * X[d];
    return x;
  }

  int corner_wall_r0(KSSourceType type) { return static_cast<int>(type) - static_cast<int>(KSSourceType::CORNER_WALL); }

  bool is_corner_wall(KSSourceType type)
  {
    const int r0 = corner_wall_r0(type);
    return r0 >= 0 && r0 < 8;
  }

  // the value of a deterministic complex-field source at a site of its slice (generic/quark_source.c)
  double deterministic_value(const KSSourceParam &param, const coords_t &g)
  {
    const bool even = (g[0] + g[1] + g[2] + g[3]) % 2 == 0;
    if (param.type == KSSourceType::POINT) return g == param.origin ? 1.0 : 0.0;
    if (is_corner_wall(param.type)) {
      const int r0 = corner_wall_r0(param.type);
      return g[0] % 2 == (r0 & 1) && g[1] % 2 == ((r0 & 2) >> 1) && g[2] % 2 == ((r0 & 4) >> 2) ? 1.0 : 0.0;
    }
    switch (param.type) {
    case KSSourceType::EVEN_WALL: return even ? 1.0 : 0.0;
    case KSSourceType::EVENANDODD_WALL: return 1.0;
    case KSSourceType::EVENMINUSODD_WALL: return even ? 1.0 : -1.0;
    default: errorQuda("%s is not a deterministic source", get_ks_source_type_str(param.type));
    }
    return 0.0;
  }

  double gaussian(host_rng_t &rng)
  {
    std::normal_distribution<double> dist {0.0, 1.0};
    return dist(rng);
  }

  // subset_mask_c / subset_mask_v: the corner subset drops the sites of the slice with an odd x, y or z
  bool masked(const KSSourceParam &param, const coords_t &g)
  {
    return param.subset == KSSourceSubset::CORNER && (g[0] % 2 != 0 || g[1] % 2 != 0 || g[2] % 2 != 0);
  }

  /**
     insert_c_mom / insert_v_mom: z exp(i theta), in MILC's order of operations, so that the values are MILC's bit for
     bit on hosts that don't contract to fused multiply-adds
   */
  complex_t momentum_phase(const KSSourceParam &param, const coords_t &N, const coords_t &g, complex_t z)
  {
    const auto &mom = param.momentum;
    if (mom[0] == 0 && mom[1] == 0 && mom[2] == 0) return z;
    const double pi = M_PI;
    const double px = 2 * pi * mom[0] / N[0];
    const double py = 2 * pi * mom[1] / N[1];
    const double pz = 2 * pi * mom[2] / N[2];
    const auto &o = param.origin;
    const double th = px * (g[0] - o[0]) + py * (g[1] - o[1]) + pz * (g[2] - o[2]);
    const double yr = cos(th), yi = sin(th);
    return {z.real() * yr - z.imag() * yi, z.real() * yi + z.imag() * yr};
  }

  template <typename real_t>
  void build(std::vector<quda::ColorSpinorField> &sources, const KSSourceParam &param, const coords_t &X,
             const coords_t &N)
  {
    const size_t volume = static_cast<size_t>(X[0]) * X[1] * X[2] * X[3];
    const int t0 = param.type == KSSourceType::POINT ? param.origin[3] : param.t0;
    const int n = sources.size();
    std::vector<real_t *> data(n);
    for (int i = 0; i < n; i++) {
      sources[i].zero();
      data[i] = sources[i].data<real_t *>();
    }

    const bool color_wall = param.type == KSSourceType::RANDOM_COLOR_WALL;
    for (size_t h = 0; h < volume; h++) {
      const auto g = global_coords(h, X);
      if (t0 != ks_source_all_t_slices && g[3] != t0) continue;

      if (color_wall) {
        for (int i = 0; i < n; i++) {
          std::array<complex_t, 3> v;
          for (auto &c : v) {
            const double re = gaussian(host_source_rand[h]);
            c = {re, gaussian(host_source_rand[h])};
          }
          double magsq = 0.0;
          for (auto &c : v) magsq += c.real() * c.real() + c.imag() * c.imag();
          const double x = 1.0 / sqrt(magsq);
          for (auto &c : v) c = {x * c.real(), x * c.imag()};
          if (masked(param, g)) v = {};
          for (int c = 0; c < 3; c++) {
            const auto z = momentum_phase(param, N, g, v[c]);
            data[i][h * 6 + 2 * c] = z.real();
            data[i][h * 6 + 2 * c + 1] = z.imag();
          }
        }
        continue;
      }

      complex_t z;
      if (param.type == KSSourceType::RANDOM_COMPLEX_WALL) {
        const double re = gaussian(host_source_rand[h]);
        z = {re, gaussian(host_source_rand[h])};
        const double x = 1.0 / sqrt(z.real() * z.real() + z.imag() * z.imag());
        z = {x * z.real(), x * z.imag()};
        if (masked(param, g)) z = 0.0;
      } else {
        z = deterministic_value(param, g);
      }
      z = momentum_phase(param, N, g, z);
      for (int c = 0; c < n; c++) {
        data[c][h * 6 + 2 * c] = z.real();
        data[c][h * 6 + 2 * c + 1] = z.imag();
      }
    }
  }
} // namespace

void constructKSSources(std::vector<quda::ColorSpinorField> &sources, const KSSourceParam &param)
{
  using namespace quda;
  const char *type = get_ks_source_type_str(param.type);
  if (!is_built_ks_source(param.type)) errorQuda("%s sources are not built by constructKSSources", type);
  if (sources.empty()) return;

  coords_t X = {sources[0].X(0), sources[0].X(1), sources[0].X(2), sources[0].X(3)};
  for (auto i = 0u; i < sources.size(); i++) {
    const auto &s = sources[i];
    if (s.Location() != QUDA_CPU_FIELD_LOCATION || s.SiteSubset() != QUDA_FULL_SITE_SUBSET || s.Nspin() != 1
        || s.Ncolor() != 3 || s.Ndim() != 4 || s.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER
        || (s.Precision() != QUDA_DOUBLE_PRECISION && s.Precision() != QUDA_SINGLE_PRECISION))
      errorQuda("Source %u of %lu is not a full 4-d staggered host field in space-spin-color order, of double or "
                "single precision, so MILC's %s source cannot be built into it",
                i, sources.size(), type);
    for (int d = 0; d < 4; d++)
      if (s.X(d) != X[d]) errorQuda("Source %u of %lu has another lattice than source 0", i, sources.size());
  }

  coords_t N;
  for (int d = 0; d < 4; d++) N[d] = X[d] * comm_dim(d);
  if (param.t0 < ks_source_all_t_slices || param.t0 >= N[3])
    errorQuda("The time slice t0 = %d of the %s source is outside the lattice (0 to %d, or %d for every slice)",
              param.t0, type, N[3] - 1, ks_source_all_t_slices);
  for (int d = 0; d < 4; d++)
    if (param.origin[d] < 0 || param.origin[d] >= N[d])
      errorQuda("The origin %d %d %d %d of the %s source is outside the %d %d %d %d lattice", param.origin[0],
                param.origin[1], param.origin[2], param.origin[3], type, N[0], N[1], N[2], N[3]);
  if (is_complex_field_ks_source(param.type) && sources.size() > 3)
    errorQuda("A %s source has 3 colors, one per source, but %lu sources were asked for", type, sources.size());
  if (param.subset == KSSourceSubset::CORNER && !is_random_ks_source(param.type))
    errorQuda("The corner subset applies to the random walls only, as in MILC, not to %s sources", type);

  if (is_random_ks_source(param.type)) {
    const size_t volume = static_cast<size_t>(X[0]) * X[1] * X[2] * X[3];
    if (host_source_rand.size() != volume || host_source_seed != param.seed) initSourceRand(param.seed);
  }

  if (sources[0].Precision() == QUDA_DOUBLE_PRECISION) {
    for (auto &s : sources)
      if (s.Precision() != QUDA_DOUBLE_PRECISION) errorQuda("The sources have different precisions");
    build<double>(sources, param, X, N);
  } else {
    for (auto &s : sources)
      if (s.Precision() != QUDA_SINGLE_PRECISION) errorQuda("The sources have different precisions");
    build<float>(sources, param, X, N);
  }

  auto list = [](const auto &v, int n) {
    std::string s;
    for (int i = 0; i < n; i++) s += (i ? " " : "") + std::to_string(v[i]);
    return s;
  };
  std::string where;
  if (param.type == KSSourceType::POINT)
    where = "at " + list(param.origin, 4);
  else if (param.t0 == ks_source_all_t_slices)
    where = "on every time slice";
  else
    where = "on time slice " + std::to_string(param.t0);
  const auto &p = param.momentum;
  if (p[0] != 0 || p[1] != 0 || p[2] != 0) where += ", momentum " + list(p, 3) + " from " + list(param.origin, 3);
  if (param.subset == KSSourceSubset::CORNER) where += ", corner subset";
  if (is_random_ks_source(param.type)) where += ", seed " + std::to_string(param.seed);
  logQuda(QUDA_SUMMARIZE, "Built %lu %s sources (MILC's definition) %s\n", sources.size(), type, where.c_str());
}
