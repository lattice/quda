#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <color_spinor_field.h>

#include "host_rng.hpp"

/**
 * @brief The sources of staggered_invert_test: QUDA's random sources, sources loaded from a file, or MILC's staggered
 * sources (ks_spectrum's generic/quark_source.c), built from global coordinates.  The corner walls are MILC's
 * corner_wall_<r0>, whose sites have x, y and z parities equal to bits 0, 1 and 2 of r0.
 */
enum class KSSourceType : int {
  RANDOM,
  LOAD,
  POINT,
  CORNER_WALL,
  CORNER_WALL_X,
  CORNER_WALL_Y,
  CORNER_WALL_XY,
  CORNER_WALL_Z,
  CORNER_WALL_ZX,
  CORNER_WALL_YZ,
  CORNER_WALL_XYZ,
  EVEN_WALL,
  EVENANDODD_WALL,
  EVENMINUSODD_WALL,
  RANDOM_COMPLEX_WALL,
  RANDOM_COLOR_WALL
};

/**
 * @brief MILC's subset of a random wall: every site of the time slice, or those with x, y and z even
 */
enum class KSSourceSubset : int { FULL, CORNER };

/**
 * @brief The value of t0 that selects every time slice (MILC's ALL_T_SLICES)
 */
constexpr int ks_source_all_t_slices = -1;

struct KSSourceParam {
  KSSourceType type = KSSourceType::POINT;
  int t0 = 0;                       // the time slice of the walls, or ks_source_all_t_slices
  std::array<int, 4> origin = {};   // the point of a point source, and the origin of the momentum phase
  std::array<int, 3> momentum = {}; // in units of 2 pi / the global extent
  KSSourceSubset subset = KSSourceSubset::FULL;
};

/**
 * @brief The generators of MILC's random walls: one per site of a local lattice, separate from host_rand, each seeded
 * from the seed and the site's global lexicographic index (initSourceRand, host_utils.h), so that the walls don't
 * depend on the rank layout
 */
class KSSourceRNG
{
  std::vector<host_rng_t> rng;
  uint64_t seed_;
  std::array<int, 4> X_;

public:
  /**
   * @brief Initialize the generators; needs to be called after comms are initialized
   * @param[in] seed The seed
   * @param[in] X The local lattice dimensions of the fields to be built
   */
  KSSourceRNG(uint64_t seed, const std::array<int, 4> &X);

  uint64_t seed() const { return seed_; }
  const std::array<int, 4> &X() const { return X_; }
  host_rng_t &operator[](size_t h) { return rng[h]; }
};

const char *get_ks_source_type_str(KSSourceType type);
const char *get_ks_source_subset_str(KSSourceSubset subset);

/**
 * @brief Whether the type is one of MILC's sources, built by constructKSSources
 */
bool is_built_ks_source(KSSourceType type);

/**
 * @brief Whether the type is one of MILC's complex-field sources, of which source c holds the field in color c
 */
bool is_complex_field_ks_source(KSSourceType type);

/**
 * @brief Whether the type is one of MILC's random walls
 */
bool is_random_ks_source(KSSourceType type);

/**
 * @brief Build MILC's staggered source of the given type into host fields, as ks_spectrum defines it: the source is
 * built on slice t0, then (random walls only) masked to the subset, then multiplied by exp(i theta) on slice t0, with
 * theta = 2 pi sum_i p_i (x_i - origin_i) / N_i over x, y and z.  Source c of a complex-field type holds the field in
 * color c (at most 3 sources); each source of random_color_wall is a new draw.  Off the slice the sources are zero.
 * The random walls draw from rng, on the fields' lattice, and leave it advanced, so later builds give new values.
 * @param[out] sources Host fields, full, 1 spin, 3 colors, double or single precision, space-spin-color order
 * @param[in] param The source
 * @param[in,out] rng The generators of the random walls (not used, and may be null, for the other types)
 */
void constructKSSources(std::vector<quda::ColorSpinorField> &sources, const KSSourceParam &param,
                        KSSourceRNG *rng = nullptr);

/**
 * @brief Convert staggered fields between QUDA's and MILC's sign conventions, by negating their odd sites (odd global
 * x + y + z + t)
 *
 * QUDA's staggered operator is eps M_MILC eps, with eps = (-1)^(x+y+z+t), so MILC's solution of M_MILC x = b is
 * eps M_QUDA^{-1} (eps b).  Apply it to b (and to an initial guess from MILC) before a QUDA solve, and to the
 * solution and to b after it; the operation is its own inverse.  eps is QUDA's staggered gamma5
 * (DiracMatrix::applyGamma5 for the unpreconditioned staggered and asqtad operators).
 *
 * @param[in,out] fields Full 4-d fields of 1 spin, not composite, on the device or on the host in double or single
 * precision
 */
void convertStaggeredSignConvention(std::vector<quda::ColorSpinorField> &fields);
