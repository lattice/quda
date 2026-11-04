#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <comm_quda.h>
#include <io/gauge_io.h>

#include "test.h"
#include "gauge_utils.h"
#include "staggered_gauge_utils.h"

// The unit links that the staggered tests' gauge construction paths generate, checked link by link against their
// expected values computed from the global coordinates: c times the identity, with c the field's scale times MILC's
// phases as the test utilities apply them (eta_x = (-1)^t, eta_y = (-1)^(t+x), eta_z = (-1)^(t+x+y)), times the
// antiperiodic sign on the t links of the last global t slice, or of the last three for asqtad long links, each
// applied once.  The loading paths load plain unit links (periodic, no phases) saved by GaugeIO, and must give
// exactly what the generating paths give; the Wilson-type path (constructHostGaugeField) applies only the sign on
// the last slice.

namespace
{
  enum class Path { STAGGERED, ASQTAD, ASQTAD_FAT_LONG, LAPLACE, INPUT_NO_PHASE, INPUT_MILC_PHASE, WILSON };

  // how the unit field is asked for: generated, or loaded from a file of plain unit links
  enum class UnitInput { UNIT_GAUGE, GAUGE_INPUT_UNIT, LOAD_GAUGE, GAUGE_INPUT_LOAD };

  const char *get_path_str(Path path)
  {
    switch (path) {
    case Path::STAGGERED: return "staggered";
    case Path::ASQTAD: return "asqtad";
    case Path::ASQTAD_FAT_LONG: return "asqtad_fat_long";
    case Path::LAPLACE: return "laplace";
    case Path::INPUT_NO_PHASE: return "input_no_phase";
    case Path::INPUT_MILC_PHASE: return "input_milc_phase";
    case Path::WILSON: return "wilson";
    }
    return "invalid";
  }

  const char *get_input_str(UnitInput input)
  {
    switch (input) {
    case UnitInput::UNIT_GAUGE: return "unit_gauge";
    case UnitInput::GAUGE_INPUT_UNIT: return "gauge_input_unit";
    case UnitInput::LOAD_GAUGE: return "load_gauge";
    case UnitInput::GAUGE_INPUT_LOAD: return "gauge_input_load";
    }
    return "invalid";
  }

  // plain unit links, saved once per run for the loading paths and removed at the end
  const char *unit_file = "staggered_gauge_input_unit.lat";
  bool unit_file_saved = false;

  struct Expected {
    double scale;
    bool phases;
    int boundary_slices;
  };

  // double-precision host links in QDP order
  struct HostLinks {
    std::array<std::vector<double>, 4> data;
    std::array<void *, 4> ptr;
    HostLinks()
    {
      for (int d = 0; d < 4; d++) {
        data[d].assign(V * gauge_site_size, 0.0);
        ptr[d] = data[d].data();
      }
    }
  };

  // the number of links on this rank that differ from the expected ones, printing the first few
  size_t wrong_links(const std::string &name, const HostLinks &u, Expected expected, QudaTboundary t_boundary)
  {
    constexpr double tol = 1e-12;
    int L[4], N[4], offset[4];
    for (int d = 0; d < 4; d++) {
      L[d] = Z[d];
      N[d] = L[d] * quda::comm_dim(d);
      offset[d] = L[d] * quda::comm_coord(d);
    }

    size_t wrong = 0;
    for (int t = 0; t < L[3]; t++)
      for (int z = 0; z < L[2]; z++)
        for (int y = 0; y < L[1]; y++)
          for (int x = 0; x < L[0]; x++) {
            const int g[4] = {offset[0] + x, offset[1] + y, offset[2] + z, offset[3] + t};
            const size_t lex = x + L[0] * (y + static_cast<size_t>(L[1]) * (z + static_cast<size_t>(L[2]) * t));
            const size_t index = (g[0] + g[1] + g[2] + g[3]) % 2 == 0 ? lex / 2 : (lex + V) / 2;
            for (int dir = 0; dir < 4; dir++) {
              double c = expected.scale;
              if (expected.phases && dir < 3 && (g[3] + (dir > 0 ? g[0] : 0) + (dir > 1 ? g[1] : 0)) % 2) c = -c;
              if (dir == 3 && t_boundary == QUDA_ANTI_PERIODIC_T && g[3] >= N[3] - expected.boundary_slices) c = -c;

              const double *link = u.data[dir].data() + index * gauge_site_size;
              double deviation = 0.0;
              for (int row = 0; row < 3; row++)
                for (int col = 0; col < 3; col++) {
                  deviation = std::max(deviation, std::abs(link[2 * (3 * row + col)] - (row == col ? c : 0.0)));
                  deviation = std::max(deviation, std::abs(link[2 * (3 * row + col) + 1]));
                }
              if (!(deviation <= tol)) {
                if (wrong++ < 3)
                  printf("rank %d: %s link dir %d at global (%d %d %d %d): m00 = (%g, %g), expected %g times the "
                         "identity, largest deviation %.3e\n",
                         quda::comm_rank(), name.c_str(), dir, g[0], g[1], g[2], g[3], link[0], link[1], c, deviation);
              }
            }
          }
    return wrong;
  }

  // every rank passes or fails alike
  void expect_links(const std::string &name, const HostLinks &u, Expected expected, QudaTboundary t_boundary)
  {
    size_t wrong = wrong_links(name, u, expected, t_boundary);
    quda::comm_allreduce_sum(wrong);
    EXPECT_EQ(wrong, 0u) << name << " links differ from the expected ones";
  }

  void save_unit_file(const QudaGaugeParam &gauge_param)
  {
    if (unit_file_saved) return;
    HostLinks u;
    constructIdentityGaugeField(u.ptr.data(), QUDA_DOUBLE_PRECISION);
    quda::GaugeIO(unit_file).save(plainHostGaugeFieldReference(u.ptr.data(), QUDA_DOUBLE_PRECISION, gauge_param.X));
    unit_file_saved = true;
  }

  class UnitFileEnvironment : public ::testing::Environment
  {
  public:
    void TearDown() override
    {
      if (!unit_file_saved) return;
      quda::comm_barrier();
      if (quda::comm_rank() == 0) std::remove(unit_file);
    }
  };
} // namespace

using test_t = ::testing::tuple<Path, UnitInput, QudaTboundary>;

class StaggeredGaugeInputTest : public ::testing::TestWithParam<test_t>
{
};

TEST_P(StaggeredGaugeInputTest, signs)
{
  const auto [path, input, t_boundary] = GetParam();
  if (path == Path::LAPLACE && !is_enabled_laplace()) GTEST_SKIP();
  // constructHostGaugeField ignores --gauge-input
  if (path == Path::WILSON && (input == UnitInput::GAUGE_INPUT_UNIT || input == UnitInput::GAUGE_INPUT_LOAD))
    GTEST_SKIP();
  const bool load = input == UnitInput::LOAD_GAUGE || input == UnitInput::GAUGE_INPUT_LOAD;

  // the options the construction paths read
  latfile.clear();
  unit_gauge = input == UnitInput::UNIT_GAUGE;
  gauge_input = input == UnitInput::GAUGE_INPUT_UNIT ? "unit" : input == UnitInput::GAUGE_INPUT_LOAD ? "load" : "";
  gauge_input_width_explicit = false;
  fermion_t_boundary = t_boundary;
  dslash_type = path == Path::ASQTAD || path == Path::ASQTAD_FAT_LONG ? QUDA_ASQTAD_DSLASH :
    path == Path::LAPLACE                                             ? QUDA_LAPLACE_DSLASH :
                                                                        QUDA_STAGGERED_DSLASH;
  compute_fatlong = path == Path::ASQTAD_FAT_LONG;
  tadpole_factor = 1.0;
  eps_naik = 0.0;
  n_naiks = 1;

  cpu_prec = QUDA_DOUBLE_PRECISION;
  host_gauge_data_type_size = sizeof(double);
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  if (path == Path::WILSON)
    setWilsonGaugeParam(gauge_param);
  else
    setStaggeredGaugeParam(gauge_param);
  gauge_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  if (load) {
    save_unit_file(gauge_param);
    latfile = unit_file;
  }

  if (path == Path::WILSON) {
    HostLinks u;
    constructHostGaugeField(u.ptr.data(), gauge_param, 0, nullptr);
    expect_links("wilson", u, {1.0, false, 1}, t_boundary);
  } else if (path == Path::INPUT_NO_PHASE || path == Path::INPUT_MILC_PHASE) {
    const bool milc = path == Path::INPUT_MILC_PHASE;
    HostLinks u;
    constructHostGaugeInputField(u.ptr.data(), gauge_param, 0, nullptr, GaugeInputMode::HAAR,
                                 milc ? QUDA_STAGGERED_PHASE_MILC : QUDA_STAGGERED_PHASE_NO);
    expect_links("input", u, {1.0, milc, 1}, t_boundary);
  } else {
    HostLinks thin, fat, lng;
    constructStaggeredHostGaugeField(thin.ptr.data(), lng.ptr.data(), fat.ptr.data(), gauge_param, 0, nullptr, false);
    const bool phases = path != Path::LAPLACE;
    // HISQ's fat link of unit links with MILC's phases, around whose plaquettes the phases multiply to -1:
    // 1 + 3/8 + 3/8 + 1/8 - 3/4 from the one-link, 3-, 5- and 7-link staples and the Lepage term
    const double fat_scale = path == Path::ASQTAD_FAT_LONG ? 1.125 : 1.0;
    // a loaded field's long links aren't built without --compute-fat-long, so they keep their zeros
    const double long_scale = path == Path::ASQTAD_FAT_LONG || (path == Path::ASQTAD && !load) ? -1.0 / 24.0 : 0.0;
    expect_links("thin", thin, {1.0, phases, 1}, t_boundary);
    expect_links("fat", fat, {fat_scale, phases, 1}, t_boundary);
    expect_links("long", lng, {long_scale, phases, 3}, t_boundary);
  }
}

int main(int argc, char **argv)
{
  quda_test test("Staggered Gauge Input Test", argc, argv);
  test.init();
  ::testing::AddGlobalTestEnvironment(new UnitFileEnvironment);
  return test.execute();
}

INSTANTIATE_TEST_SUITE_P(StaggeredGaugeInput, StaggeredGaugeInputTest,
                         ::testing::Combine(::testing::Values(Path::STAGGERED, Path::ASQTAD, Path::ASQTAD_FAT_LONG,
                                                              Path::LAPLACE, Path::INPUT_NO_PHASE,
                                                              Path::INPUT_MILC_PHASE, Path::WILSON),
                                            ::testing::Values(UnitInput::UNIT_GAUGE, UnitInput::GAUGE_INPUT_UNIT,
                                                              UnitInput::LOAD_GAUGE, UnitInput::GAUGE_INPUT_LOAD),
                                            ::testing::Values(QUDA_PERIODIC_T, QUDA_ANTI_PERIODIC_T)),
                         [](testing::TestParamInfo<test_t> param) {
                           std::string name = get_path_str(::testing::get<0>(param.param));
                           name += std::string("_") + get_input_str(::testing::get<1>(param.param));
                           name += ::testing::get<2>(param.param) == QUDA_PERIODIC_T ? "_periodic" : "_antiperiodic";
                           return name;
                         });
