#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <color_spinor_field.h>

/**
   The sources of staggered_invert_test: QUDA's random sources, sources loaded from a file, or MILC's staggered
   sources (ks_spectrum's generic/quark_source.c), built from global coordinates.  The corner walls are MILC's
   corner_wall_<r0>, whose sites have x, y and z parities equal to bits 0, 1 and 2 of r0.
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

/** MILC's subset of a random wall: every site of the time slice, or those with x, y and z even */
enum class KSSourceSubset : int { FULL, CORNER };

/** The value of t0 that selects every time slice (MILC's ALL_T_SLICES) */
constexpr int ks_source_all_t_slices = -1;

struct KSSourceParam {
  KSSourceType type = KSSourceType::POINT;
  int t0 = 0;                       // the time slice of the walls, or ks_source_all_t_slices
  std::array<int, 4> origin = {};   // the point of a point source, and the origin of the momentum phase
  std::array<int, 3> momentum = {}; // in units of 2 pi / the global extent
  KSSourceSubset subset = KSSourceSubset::FULL;
  uint64_t seed = 1234; // of the source generator, for the random walls
};

const char *get_ks_source_type_str(KSSourceType type);
const char *get_ks_source_subset_str(KSSourceSubset subset);

/** Whether the type is one of MILC's sources, built by constructKSSources */
bool is_built_ks_source(KSSourceType type);

/** Whether the type is one of MILC's complex-field sources, of which source c holds the field in color c */
bool is_complex_field_ks_source(KSSourceType type);

/** Whether the type is one of MILC's random walls */
bool is_random_ks_source(KSSourceType type);

/**
   @brief Build MILC's staggered source of the given type into host fields, as ks_spectrum defines it: the source is
   built on slice t0, then (random walls only) masked to the subset, then multiplied by exp(i theta) on slice t0, with
   theta = 2 pi sum_i p_i (x_i - origin_i) / N_i over x, y and z.  Source c of a complex-field type holds the field in
   color c (at most 3 sources); each source of random_color_wall is a new draw.  Off the slice the sources are zero.
   The random walls draw from host_source_rand (initialized with param.seed on the fields' lattice on first use, or
   when the seed or the lattice changes) and leave it advanced, so later builds give new values.  Returning to an
   earlier seed reinitializes its streams: seeds A, B, A repeat A's first draws.
   @param[out] sources Host fields, full, 1 spin, 3 colors, double or single precision, space-spin-color order
   @param[in] param The source
 */
void constructKSSources(std::vector<quda::ColorSpinorField> &sources, const KSSourceParam &param);
