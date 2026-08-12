#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <complex>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <color_spinor_field.h>

#include <comm_quda.h>
#include <qio_field.h>

#include "host_utils.h"
#include "command_line_params.h"
#include "misc.h"

/**
   Per-configuration correlator measurement driver for correlator
   distribution studies.

   For each configuration <prefix>_cfg_<n>.lime in the requested range,
   this driver computes twelve propagator components from a single
   point source (optionally at a random spatial site per configuration),
   contracts all sixteen DeGrand-Rossi meson channels and the nucleon
   two-point function at zero momentum, and appends one row per
   (channel, timeslice) to a plain-text dataset.  No averaging is
   performed anywhere: the per-configuration correlators ARE the data.
 */

// Names of the DR meson channels in the G_idx order used by the FT
// contraction kernels (see get_dr_gm_i in gamma.cuh): vectors,
// pseudo-vectors, scalar, pseudoscalar, tensors
static const char *meson_channel_names[16] = {"G1",  "G2",  "G3",  "G4",  "G5G1", "G5G2", "G5G3", "G5G4",
                                              "1",   "G5",  "S12", "S13", "S14",  "S23",  "S24",  "S34"};

void display_test_info()
{
  printfQuda("running the following measurement:\n");
  printfQuda("correlator distribution driver, dslash_type %s\n", get_dslash_str(dslash_type));
  printfQuda("configs %s_cfg_[%d:%d:%d]\n", corrdist_config_prefix.c_str(), corrdist_config_start, corrdist_config_end,
             corrdist_config_step);
  printfQuda("S_dimension T_dimension kappa\n");
  printfQuda("%d/%d/%d        %d        %e\n", xdim, ydim, zdim, tdim, kappa);
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

// Set a unit point source at global coordinates x for spin-color
// component dil in an even-odd, space-spin-color ordered host field.
template <typename Float> void make_point_source(void *v, const int *const X, const int *const x, int dil)
{
  size_t local_floats = (size_t)V * 24;
  memset(v, 0, local_floats * sizeof(Float));

  int lx[4];
  for (int d = 0; d < 4; d++) {
    lx[d] = x[d] - quda::comm_coord(d) * X[d];
    if (lx[d] < 0 || lx[d] >= X[d]) return; // source not on this rank
  }

  int sindx = lx[0] + X[0] * (lx[1] + X[1] * (lx[2] + X[2] * lx[3]));
  int parity = (lx[0] + lx[1] + lx[2] + lx[3]) & 1;
  int cb_idx = sindx / 2;
  size_t off = 24 * ((size_t)Vh * parity + cb_idx) + 2 * dil;
  static_cast<Float *>(v)[off] = 1.0;
}

int main(int argc, char **argv)
{
  auto app = make_app();
  add_propagator_option_group(app);
  add_corrdist_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  setQudaPrecisions();
  initComms(argc, argv, gridsize_from_cmdline);
  display_test_info();
  initQuda(device_ordinal);
  initRand();

  if (corrdist_config_prefix.size() == 0) errorQuda("--corrdist-config-prefix must be given");
  if (corrdist_out.size() == 0) errorQuda("--corrdist-out must be given");
  if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH)
    errorQuda("Unsupported dslash type %s; use wilson or clover", get_dslash_str(dslash_type));

  std::array<int, 4> X = {xdim, ydim, zdim, tdim};
  setDims(X.data());

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    inv_param.compute_clover = 1;
    inv_param.compute_clover_inverse = 1;
  }

  // Host gauge field
  void *gauge[4];
  for (int dir = 0; dir < 4; dir++) gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);

  // Host spinor fields: 12 sources and 12 solutions
  quda::ColorSpinorParam cs_param;
  constructWilsonTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  size_t spinor_bytes = (size_t)V * 24 * (cs_param.Precision() == QUDA_SINGLE_PRECISION ? 4 : 8);
  constexpr int nprop = 12;
  std::vector<void *> source(nprop), prop(nprop);
  for (int i = 0; i < nprop; i++) {
    source[i] = safe_malloc(spinor_bytes);
    prop[i] = safe_malloc(spinor_bytes);
  }

  const size_t Lt = tdim * quda::comm_dim(3);
  constexpr int n_mom = 1;
  const int mom[4] = {0, 0, 0, 0};
  const QudaFFTSymmType fft_type[4] = {QUDA_FFT_SYMM_EO, QUDA_FFT_SYMM_EO, QUDA_FFT_SYMM_EO, QUDA_FFT_SYMM_EO};

  // meson results: [t][16][2], nucleon results: [t][16][2]
  std::vector<double> meson_result(Lt * 16 * 2), nucleon_result(Lt * 16 * 2);

  // open the output file on rank 0; append so measurement can be resumed
  std::ofstream out_file;
  if (quda::comm_rank() == 0) {
    bool fresh = !std::ifstream(corrdist_out).good();
    out_file.open(corrdist_out, std::ios::app);
    if (!out_file) errorQuda("Failed to open output file %s", corrdist_out.c_str());
    if (fresh) out_file << "# cfg src_x src_y src_z src_t channel t re im\n";
  }

  int n_measured = 0;
  for (int cfg = corrdist_config_start; cfg <= corrdist_config_end; cfg += corrdist_config_step) {
    std::string fname = corrdist_config_prefix + "_cfg_" + std::to_string(cfg) + ".lime";
    if (!std::ifstream(fname).good()) {
      printfQuda("Skipping missing configuration %s\n", fname.c_str());
      continue;
    }

    read_gauge_field(fname.c_str(), gauge, gauge_param.cpu_prec, gauge_param.X, 0, (char **)0);
    loadGaugeQuda((void *)gauge, &gauge_param);
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) loadCloverQuda(nullptr, nullptr, &inv_param);

    // Sources per configuration: spread evenly across timeslices (maximally
    // separated, so samples decorrelate) with spatial positions drawn
    // deterministically per (configuration, source). Statistics for the
    // distribution analysis are sized by cumulant errors, not spectroscopy,
    // so extra sources per config substitute for extra HMC.
    for (int s = 0; s < corrdist_num_sources; s++) {

    int src[4] = {prop_source_position[0][0], prop_source_position[0][1], prop_source_position[0][2],
                  prop_source_position[0][3]};
    src[3] = (src[3] + (s * (int)Lt) / corrdist_num_sources) % (int)Lt;
    if (corrdist_random_source) {
      srand(2718281 + 131 * cfg + s); // deterministic per (configuration, source), identical on all ranks
      for (int d = 0; d < 3; d++) src[d] = rand() % (X[d] * quda::comm_dim(d));
    }

    // twelve solves
    for (int dil = 0; dil < nprop; dil++) {
      if (cs_param.Precision() == QUDA_SINGLE_PRECISION)
        make_point_source<float>(source[dil], X.data(), src, dil);
      else
        make_point_source<double>(source[dil], X.data(), src, dil);
      invertQuda(prop[dil], source[dil], &inv_param);
    }

    // zero-momentum meson contractions, all 16 DR channels
    std::fill(meson_result.begin(), meson_result.end(), 0.0);
    void *meson_ptr = meson_result.data();
    contractFTQuda(prop.data(), prop.data(), &meson_ptr, QUDA_CONTRACT_TYPE_DR_FT_T, (void *)&cs_param, 3, X.data(),
                   src, n_mom, mom, fft_type);

    // nucleon contraction (u = d for degenerate light quarks)
    std::fill(nucleon_result.begin(), nucleon_result.end(), 0.0);
    void *nucleon_ptr = nucleon_result.data();
    baryonContractFTQuda(prop.data(), prop.data(), &nucleon_ptr, QUDA_CONTRACT_TYPE_BARYON_NUCLEON_FT_T,
                         (void *)&cs_param, X.data(), src, n_mom, mom, fft_type);

    if (quda::comm_rank() == 0) {
      char line[256];
      for (int G = 0; G < 16; G++) {
        // sign convention from moving gamma_5 through the insertion,
        // matching established practice; makes the pion (G5) positive
        double sign = G < 8 ? -1.0 : 1.0;
        for (size_t t = 0; t < Lt; t++) {
          // shift so that t is the source-sink separation
          size_t ts = (t + src[3]) % Lt;
          snprintf(line, sizeof(line), "%d %d %d %d %d %s %zu %+.16e %+.16e\n", cfg, src[0], src[1], src[2], src[3],
                   meson_channel_names[G], t, sign * meson_result[2 * (16 * ts + G)],
                   sign * meson_result[2 * (16 * ts + G) + 1]);
          out_file << line;
        }
      }
      // nucleon: write the positive- and negative-parity projected
      // traces Tr[P+- C] with P+- = (1 +- gamma_4)/2; in the DeGrand-Rossi
      // basis Tr[gamma_4 C] = C_02 + C_13 + C_20 + C_31
      for (size_t t = 0; t < Lt; t++) {
        size_t ts = (t + src[3]) % Lt;
        std::complex<double> tr(0.0, 0.0), tr_g4(0.0, 0.0);
        auto elem = [&](int i, int j) {
          return std::complex<double>(nucleon_result[2 * (16 * ts + 4 * i + j)],
                                      nucleon_result[2 * (16 * ts + 4 * i + j) + 1]);
        };
        for (int i = 0; i < 4; i++) tr += elem(i, i);
        tr_g4 = elem(0, 2) + elem(1, 3) + elem(2, 0) + elem(3, 1);
        std::complex<double> pos = 0.5 * (tr + tr_g4), neg = 0.5 * (tr - tr_g4);
        snprintf(line, sizeof(line), "%d %d %d %d %d N_pos %zu %+.16e %+.16e\n", cfg, src[0], src[1], src[2], src[3], t,
                 pos.real(), pos.imag());
        out_file << line;
        snprintf(line, sizeof(line), "%d %d %d %d %d N_neg %zu %+.16e %+.16e\n", cfg, src[0], src[1], src[2], src[3], t,
                 neg.real(), neg.imag());
        out_file << line;
      }
      out_file.flush();
    }

    printfQuda("Measured configuration %d source %d/%d (%d %d %d %d)\n", cfg, s + 1, corrdist_num_sources, src[0],
               src[1], src[2], src[3]);
    } // source loop

    freeGaugeQuda();
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) freeCloverQuda();
    n_measured++;
  }

  printfQuda("Correlator distribution measurement complete: %d configurations -> %s\n", n_measured,
             corrdist_out.c_str());

  if (quda::comm_rank() == 0) out_file.close();
  for (int i = 0; i < nprop; i++) {
    host_free(source[i]);
    host_free(prop[i]);
  }
  for (int dir = 0; dir < 4; dir++) host_free(gauge[dir]);

  endQuda();
  finalizeComms();

  return 0;
}
