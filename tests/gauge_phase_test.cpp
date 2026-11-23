#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <quda.h>
#include <comm_quda.h>
#include <gauge_field.h>
#include <instantiate.h>

#include "host_utils.h"
#include "gauge_utils.h"
#include "misc.h"
#include "test.h"

// The staggered phases that GaugeField::applyStaggeredPhase and removeStaggeredPhase apply to device fields,
// against the test utilities' host phases (applyGaugeStaggeredPhase) and t boundary sign (applyGaugeTemporalBCs,
// through applyGaugeFieldScaling): the MILC, TIFR or CHROMA convention, times the antiperiodic sign on the t links
// of the last global t slice.  The links are Haar-random, and the fields must agree exactly.  Extended fields are
// checked as the gauge force and gauge path computations use them: removing the phases from an extended copy of a
// phased field must give the extended copy of the field without phases, halo included.  Through the interface,
// phased links with the antiperiodic sign are loaded with the phases removed: labelled periodic, as MILC's
// observables load them, the sign stays and the Polyakov loop is the negative of that of the links without phases;
// labelled antiperiodic, the sign is removed with the phases and the Polyakov loop is theirs.  The gauge force of
// phased links labelled antiperiodic, as MILC's gauge force loads them, is that of the links without phases.

namespace
{

  /**
     @brief Parameters for a gauge field of the test's lattice in double or single precision, without
     reconstruct or ghost zone: QDP order on the host, native order on the device
     @param[in] location Host or device
     @param[in] prec Precision of the field
     @param[in] phase Staggered phase type the field records
     @param[in] t_boundary t boundary the field records
     @return The parameters
   */
  quda::GaugeFieldParam field_param(QudaFieldLocation location, QudaPrecision prec, QudaStaggeredPhase phase,
                                    QudaTboundary t_boundary)
  {
    QudaGaugeParam gauge_param = newQudaGaugeParam();
    setWilsonGaugeParam(gauge_param);
    gauge_param.cpu_prec = prec;
    gauge_param.t_boundary = t_boundary;
    gauge_param.staggered_phase_type = phase;
    gauge_param.staggered_phase_applied = false;
    quda::GaugeFieldParam param(gauge_param);
    param.location = location;
    param.create = QUDA_NULL_FIELD_CREATE;
    param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    param.reconstruct = QUDA_RECONSTRUCT_NO;
    param.setPrecision(prec, location == QUDA_CUDA_FIELD_LOCATION);
    return param;
  }

  /**
     @brief A Haar-random host gauge field of the test's lattice, without phases or boundary sign
     @param[in] prec Precision of the field
     @param[in] t_boundary The t boundary the field records, which GaugeField::copy requires of the fields it is
     copied to and from; without reconstruct, the label doesn't change the links
     @return The field
   */
  quda::GaugeField random_field(QudaPrecision prec, QudaTboundary t_boundary)
  {
    QudaGaugeParam gauge_param = newQudaGaugeParam();
    setWilsonGaugeParam(gauge_param);
    gauge_param.cpu_prec = prec;
    gauge_param.t_boundary = QUDA_PERIODIC_T;
    gauge_param.anisotropy = 1.0;
    quda::GaugeField u(field_param(QUDA_CPU_FIELD_LOCATION, prec, QUDA_STAGGERED_PHASE_NO, t_boundary));
    constructHostGaugeInputField(u, gauge_param, 0, nullptr, GaugeInputMode::HAAR);
    return u;
  }

  /**
     @brief A host field's links times the phases of a convention and the t boundary sign, applied with the test
     utilities
     @param[in] u The links, in QDP order
     @param[in] type MILC, TIFR or CHROMA
     @param[in] t_boundary Periodic or antiperiodic: whether the t links of the last global t slice change sign
     @return The phased links, as a host field in QDP order recording t_boundary
   */
  quda::GaugeField host_phased(const quda::GaugeField &u, QudaStaggeredPhase type, QudaTboundary t_boundary)
  {
    quda::GaugeField phased(field_param(QUDA_CPU_FIELD_LOCATION, u.Precision(), QUDA_STAGGERED_PHASE_NO, t_boundary));
    for (int dir = 0; dir < 4; dir++) memcpy(phased.data(dir), u.data(dir), u.Volume() * 18 * u.Precision());
    applyGaugeStaggeredPhase(phased, type);

    // with no anisotropy and no gauge fixing, applyGaugeFieldScaling only applies the t boundary sign
    QudaGaugeParam gauge_param = newQudaGaugeParam();
    setWilsonGaugeParam(gauge_param);
    gauge_param.anisotropy = 1.0;
    gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
    gauge_param.t_boundary = t_boundary;
    auto arrays = phased.data_array();
    applyGaugeFieldScaling(arrays.data, static_cast<int>(phased.VolumeCB()), gauge_param, phased.Precision());
    return phased;
  }

  /**
     @brief Expect two host fields in QDP order to hold the same links, with every rank passing or failing alike.
     Collective.
     @param[in] result The field to check
     @param[in] msg_result Its label in messages
     @param[in] expected The reference
     @param[in] msg_expected Its label in messages
   */
  void expect_links(const quda::GaugeField &result, const std::string &msg_result, const quda::GaugeField &expected,
                    const std::string &msg_expected)
  {
    int fail = compare_link_fields(result, msg_result, expected, msg_expected, 0.0).first ? 0 : 1;
    quda::comm_allreduce_max(fail);
    EXPECT_EQ(fail, 0) << msg_result << " links differ from the " << msg_expected << " links";
  }

  /**
     @brief Whether two fields hold the same links on every rank, compared on the host in QDP order, printing the
     first few that differ.  Collective.
     @param[in] a A field
     @param[in] b Another, with the same parameters
     @return Whether they are equal
   */
  bool same_links(const quda::GaugeField &a, const quda::GaugeField &b)
  {
    quda::GaugeFieldParam host_param(a);
    host_param.location = QUDA_CPU_FIELD_LOCATION;
    host_param.create = QUDA_NULL_FIELD_CREATE;
    host_param.order = QUDA_QDP_GAUGE_ORDER;
    quda::GaugeField ha(host_param), hb(host_param);
    ha.copy(a);
    hb.copy(b);

    // sites by their coordinates in the field, halo included; R is the halo width of an extended field
    const auto &E = ha.X();
    const auto &R = ha.R();
    const size_t volume = ha.Volume();
    const size_t site_bytes = 18 * ha.Precision();
    size_t wrong = 0;
    for (int t = 0; t < E[3]; t++)
      for (int z = 0; z < E[2]; z++)
        for (int y = 0; y < E[1]; y++)
          for (int x = 0; x < E[0]; x++) {
            const size_t lex = x + E[0] * (y + static_cast<size_t>(E[1]) * (z + static_cast<size_t>(E[2]) * t));
            const size_t index = (x + y + z + t) % 2 == 0 ? lex / 2 : (lex + volume) / 2;
            for (int dir = 0; dir < 4; dir++) {
              const auto pa = static_cast<const char *>(ha.data(dir)) + index * site_bytes;
              const auto pb = static_cast<const char *>(hb.data(dir)) + index * site_bytes;
              if (memcmp(pa, pb, site_bytes) != 0 && wrong++ < 4)
                printf("rank %d: link dir %d at (%d %d %d %d) in the field, (%d %d %d %d) in the interior, differs\n",
                       quda::comm_rank(), dir, x, y, z, t, x - R[0], y - R[1], z - R[2], t - R[3]);
            }
          }
    quda::comm_allreduce_sum(wrong);
    return wrong == 0;
  }

  /**
     @brief Gauge parameters for loading host links in QDP order through the interface, without reconstruct, in
     one precision throughout
     @param[in] prec The precision of the links, on the host and the device
     @param[in] type The staggered phase convention the links record
     @param[in] t_boundary The t boundary the links record
     @param[in] applied Whether the links record their phases as applied
     @return The parameters
   */
  QudaGaugeParam load_param(QudaPrecision prec, QudaStaggeredPhase type, QudaTboundary t_boundary, bool applied)
  {
    QudaGaugeParam gauge_param = newQudaGaugeParam();
    setWilsonGaugeParam(gauge_param);
    gauge_param.cpu_prec = gauge_param.cuda_prec = gauge_param.cuda_prec_sloppy = prec;
    gauge_param.cuda_prec_precondition = gauge_param.cuda_prec_eigensolver = prec;
    gauge_param.cuda_prec_refinement_sloppy = prec;
    gauge_param.reconstruct = gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_precondition = gauge_param.reconstruct_eigensolver = QUDA_RECONSTRUCT_NO;
    gauge_param.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;
    gauge_param.anisotropy = 1.0;
    gauge_param.t_boundary = t_boundary;
    gauge_param.staggered_phase_type = type;
    gauge_param.staggered_phase_applied = applied;
    return gauge_param;
  }

  /**
     @brief The temporal Polyakov loop of host links loaded through the interface, with their phases removed
     if they are applied, as gaugeObservablesQuda computes it.  Collective.
     @param[in] u The links, in QDP order
     @param[in] type The staggered phase convention the links record
     @param[in] t_boundary The t boundary the links record
     @param[in] applied Whether the links hold their phases, which are then removed
     @return The real and imaginary parts
   */
  std::array<double, 2> polyakov_loop(const quda::GaugeField &u, QudaStaggeredPhase type, QudaTboundary t_boundary,
                                      bool applied)
  {
    auto gauge_param = load_param(u.Precision(), type, t_boundary, applied);
    auto arrays = u.data_array();
    loadGaugeQuda(arrays.data, &gauge_param);
    QudaGaugeObservableParam obs_param = newQudaGaugeObservableParam();
    obs_param.compute_polyakov_loop = QUDA_BOOLEAN_TRUE;
    obs_param.remove_staggered_phase = applied ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    gaugeObservablesQuda(&obs_param);
    freeGaugeQuda();
    return {obs_param.ploop[0], obs_param.ploop[1]};
  }

  /**
     @brief The gauge force of host links, as computeGaugeForceQuda computes it from the plaquette staples and two
     rectangles of each direction, with the links' phases removed if they are applied.  Collective.
     @param[in] u The links, in QDP order
     @param[in] type The staggered phase convention the links record
     @param[in] t_boundary The t boundary the links record
     @param[in] applied Whether the links hold their phases, which are then removed
     @return The momentum, as a host field in QDP order
   */
  quda::GaugeField gauge_force(const quda::GaugeField &u, QudaStaggeredPhase type, QudaTboundary t_boundary, bool applied)
  {
    // as gauge_path_test's: per direction, the six plaquette staples and two rectangles, 7 - d being backwards in d
    const int paths[4][8][5]
      = {{{1, 7, 6}, {6, 7, 1}, {2, 7, 5}, {5, 7, 2}, {3, 7, 4}, {4, 7, 3}, {0, 1, 7, 7, 6}, {1, 7, 7, 6, 0}},
         {{2, 6, 5}, {5, 6, 2}, {3, 6, 4}, {4, 6, 3}, {0, 6, 7}, {7, 6, 0}, {1, 2, 6, 6, 5}, {2, 6, 6, 5, 1}},
         {{3, 5, 4}, {4, 5, 3}, {0, 5, 7}, {7, 5, 0}, {1, 5, 6}, {6, 5, 1}, {2, 3, 5, 5, 4}, {3, 5, 5, 4, 2}},
         {{0, 4, 7}, {7, 4, 0}, {1, 4, 6}, {6, 4, 1}, {2, 4, 5}, {5, 4, 2}, {3, 0, 4, 4, 7}, {0, 4, 4, 7, 3}}};
    constexpr int num_paths = 8;
    int length[num_paths] = {3, 3, 3, 3, 3, 3, 5, 5};
    double coeff[num_paths] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.5, 2.6};
    std::array<std::array<std::vector<int>, num_paths>, 4> path_data;
    std::array<std::array<int *, num_paths>, 4> path_ptrs;
    std::array<int **, 4> input_path_buf;
    for (int dir = 0; dir < 4; dir++) {
      for (int i = 0; i < num_paths; i++) {
        path_data[dir][i].assign(paths[dir][i], paths[dir][i] + length[i]);
        path_ptrs[dir][i] = path_data[dir][i].data();
      }
      input_path_buf[dir] = path_ptrs[dir].data();
    }

    auto gauge_param = load_param(u.Precision(), type, t_boundary, applied);
    quda::GaugeFieldParam mom_param(gauge_param, nullptr, QUDA_ASQTAD_MOM_LINKS);
    mom_param.location = QUDA_CPU_FIELD_LOCATION;
    mom_param.create = QUDA_ZERO_FIELD_CREATE;
    quda::GaugeField mom(mom_param);
    auto mom_arrays = mom.data_array();
    auto arrays = u.data_array();
    computeGaugeForceQuda(mom_arrays.data, arrays.data, input_path_buf.data(), length, coeff, num_paths, 6, 0.3,
                          &gauge_param);
    return mom;
  }

  /**
     @brief The largest difference between two momentum fields over all ranks.  Collective.
     @tparam real double or float
     @param[in] a A host momentum field in QDP order
     @param[in] b Another, with the same parameters
     @return The largest absolute difference of any real
   */
  template <typename real> double max_mom_deviation(const quda::GaugeField &a, const quda::GaugeField &b)
  {
    const size_t reals = a.Volume() * a.Reconstruct();
    double dev = 0.0;
    for (int dir = 0; dir < 4; dir++) {
      const auto pa = a.data<const real *>(dir);
      const auto pb = b.data<const real *>(dir);
      for (size_t i = 0; i < reals; i++) dev = std::max(dev, std::abs(static_cast<double>(pa[i]) - pb[i]));
    }
    quda::comm_allreduce_max(dev);
    return dev;
  }

  /**
     @param[in] type A staggered phase convention
     @return Its name in test names
   */
  const char *get_phase_str(QudaStaggeredPhase type)
  {
    switch (type) {
    case QUDA_STAGGERED_PHASE_MILC: return "milc";
    case QUDA_STAGGERED_PHASE_TIFR: return "tifr";
    case QUDA_STAGGERED_PHASE_CHROMA: return "chroma";
    default: return "invalid";
    }
  }

} // namespace

using test_t = ::testing::tuple<QudaPrecision, QudaStaggeredPhase, QudaTboundary>;

class GaugePhaseTest : public ::testing::TestWithParam<test_t>
{
};

// the device phases and t boundary sign are those of the test utilities, and removing them restores the links
TEST_P(GaugePhaseTest, apply_remove)
{
  const auto [prec, type, t_boundary] = GetParam();
  if (!quda::is_enabled(prec)) GTEST_SKIP();
  const auto u = random_field(prec, t_boundary);
  quda::GaugeField device(field_param(QUDA_CUDA_FIELD_LOCATION, prec, type, t_boundary));
  device.copy(u);

  // the copy takes u's phase type, so the device field is given its own
  device.applyStaggeredPhase(type);
  EXPECT_TRUE(device.StaggeredPhaseApplied());
  quda::GaugeField host(field_param(QUDA_CPU_FIELD_LOCATION, prec, type, t_boundary));
  host.copy(device);
  expect_links(host, "device phased", host_phased(u, type, t_boundary), "host phased");

  device.removeStaggeredPhase();
  EXPECT_FALSE(device.StaggeredPhaseApplied());
  host.copy(device);
  expect_links(host, "device unphased", u, "original");
}

// removing the phases from an extended copy of a phased field gives the extended copy of the field without phases
TEST_P(GaugePhaseTest, extended_remove)
{
  const auto [prec, type, t_boundary] = GetParam();
  if (!quda::is_enabled(prec)) GTEST_SKIP();
  const auto u = random_field(prec, t_boundary);
  quda::GaugeField plain(field_param(QUDA_CUDA_FIELD_LOCATION, prec, type, t_boundary));
  plain.copy(u);
  quda::GaugeField phased(field_param(QUDA_CUDA_FIELD_LOCATION, prec, type, t_boundary));
  phased.copy(u);
  phased.applyStaggeredPhase(type);

  // the halo widths of the interface's extended fields, and of width 1, whose sum is odd when an odd number of
  // directions is partitioned, so that a site's parity label differs from the parity of its coordinates
  for (int width : {2, 1}) {
    quda::lat_dim_t R;
    for (int d = 0; d < 4; d++) R[d] = width * quda::comm_dim_partitioned(d);
    std::unique_ptr<quda::GaugeField> expected(quda::createExtendedGauge(plain, R));
    std::unique_ptr<quda::GaugeField> extended(quda::createExtendedGauge(phased, R));
    extended->removeStaggeredPhase();
    EXPECT_TRUE(same_links(*extended, *expected)) << "halo width " << width;
  }
}

using interface_test_t = ::testing::tuple<QudaPrecision, QudaStaggeredPhase>;

class GaugePhaseInterfaceTest : public ::testing::TestWithParam<interface_test_t>
{
};

// phased links with the antiperiodic sign, labelled periodic as MILC's observables load them: removing the phases
// leaves the sign, so the Polyakov loop is the negative of that of the links without phases
TEST_P(GaugePhaseInterfaceTest, polyakov_periodic_label)
{
  const auto [prec, type] = GetParam();
  if (!quda::is_enabled(prec)) GTEST_SKIP();
  const auto u = random_field(prec, QUDA_ANTI_PERIODIC_T);
  const auto plain = polyakov_loop(u, QUDA_STAGGERED_PHASE_NO, QUDA_PERIODIC_T, false);
  const auto phased = polyakov_loop(host_phased(u, type, QUDA_ANTI_PERIODIC_T), type, QUDA_PERIODIC_T, true);
  for (int i = 0; i < 2; i++) EXPECT_DOUBLE_EQ(phased[i], -plain[i]) << (i == 0 ? "real part" : "imaginary part");
}

// the same links labelled antiperiodic: the sign is removed with the phases, so the Polyakov loop is that of the
// links without phases
TEST_P(GaugePhaseInterfaceTest, polyakov_antiperiodic_label)
{
  const auto [prec, type] = GetParam();
  if (!quda::is_enabled(prec)) GTEST_SKIP();
  const auto u = random_field(prec, QUDA_ANTI_PERIODIC_T);
  const auto plain = polyakov_loop(u, QUDA_STAGGERED_PHASE_NO, QUDA_PERIODIC_T, false);
  const auto phased = polyakov_loop(host_phased(u, type, QUDA_ANTI_PERIODIC_T), type, QUDA_ANTI_PERIODIC_T, true);
  for (int i = 0; i < 2; i++) EXPECT_DOUBLE_EQ(phased[i], plain[i]) << (i == 0 ? "real part" : "imaginary part");
}

// the gauge force of phased links labelled antiperiodic, as MILC's gauge force loads them, is that of the links
// without phases: a sign on every t link of a time slice cancels in loops that don't wind around the lattice
TEST_P(GaugePhaseInterfaceTest, gauge_force)
{
  const auto [prec, type] = GetParam();
  if (!quda::is_enabled(prec)) GTEST_SKIP();
  const auto u = random_field(prec, QUDA_ANTI_PERIODIC_T);
  const auto plain = gauge_force(u, QUDA_STAGGERED_PHASE_NO, QUDA_PERIODIC_T, false);
  const auto phased = gauge_force(host_phased(u, type, QUDA_ANTI_PERIODIC_T), type, QUDA_ANTI_PERIODIC_T, true);
  const double dev = prec == QUDA_DOUBLE_PRECISION ? max_mom_deviation<double>(phased, plain) :
                                                     max_mom_deviation<float>(phased, plain);
  EXPECT_EQ(dev, 0.0);

  quda::GaugeFieldParam zero_param(plain);
  zero_param.create = QUDA_ZERO_FIELD_CREATE;
  const quda::GaugeField zero(zero_param);
  const double size
    = prec == QUDA_DOUBLE_PRECISION ? max_mom_deviation<double>(plain, zero) : max_mom_deviation<float>(plain, zero);
  EXPECT_GT(size, 0.0) << "the force is zero, so the comparison shows nothing";
}

int main(int argc, char **argv)
{
  quda_test test("Gauge phase test", argc, argv);
  test.init();
  return test.execute();
}

using ::testing::Combine;
using ::testing::Values;

INSTANTIATE_TEST_SUITE_P(Phase, GaugePhaseTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 Values(QUDA_STAGGERED_PHASE_MILC, QUDA_STAGGERED_PHASE_TIFR, QUDA_STAGGERED_PHASE_CHROMA),
                                 Values(QUDA_PERIODIC_T, QUDA_ANTI_PERIODIC_T)),
                         [](testing::TestParamInfo<test_t> param) {
                           std::string name = get_prec_str(::testing::get<0>(param.param));
                           name += std::string("_") + get_phase_str(::testing::get<1>(param.param));
                           return name
                             + (::testing::get<2>(param.param) == QUDA_PERIODIC_T ? "_periodic" : "_antiperiodic");
                         });

INSTANTIATE_TEST_SUITE_P(Interface, GaugePhaseInterfaceTest,
                         Combine(Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                 Values(QUDA_STAGGERED_PHASE_MILC, QUDA_STAGGERED_PHASE_TIFR,
                                        QUDA_STAGGERED_PHASE_CHROMA)),
                         [](testing::TestParamInfo<interface_test_t> param) {
                           return std::string(get_prec_str(::testing::get<0>(param.param))) + "_"
                             + get_phase_str(::testing::get<1>(param.param));
                         });
