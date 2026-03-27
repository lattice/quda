#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <type_traits>

#include "host_utils.h"
#include "force_utils.hpp"
#include "index_utils.hpp"
#include "instantiate_host.hpp"
#include "misc.h"
#include "hisq_force_reference.h"

extern int Z[4];
extern int V;
extern int Vh;

template <typename real_t> struct ComputeLinkOrderedOuterProduct {
  void operator()(const void *const src_, quda::GaugeField &dest, size_t nhops)
  {
    auto src = reinterpret_cast<const su3_vector<real_t> *>(src_);

#pragma omp parallel for
    for (int i = 0; i < V; ++i) {
      int dx[4];
      for (int dir = 0; dir < 4; ++dir) {
        dx[3] = dx[2] = dx[1] = dx[0] = 0;
        dx[dir] = nhops;
        int nbr_idx = neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
        auto hw = src + nbr_idx;
        auto p = get_su3_matrix<real_t>(dest, i, dir);
        su3_projector(hw, &src[i], p);
      } // dir
    }   // i
  }
};

void computeLinkOrderedOuterProduct(void *src, quda::GaugeField &dst, size_t nhops)
{
  instantiate_host<ComputeLinkOrderedOuterProduct>(dst.Precision(), src, dst, nhops);
}

#define RETURN_IF_ERR                                                                                                  \
  if (err) return;

extern int Vh;
extern int Vh_ex;

template <int N> struct Sign {
  static const int result = 1;
};

template <> struct Sign<1> {
  static const int result = -1;
};

template <typename real_t_> class LoadStore
{
  using real_t = real_t_;
  using complex = std::complex<real_t>;
  using matrix = Matrix<3, complex>;

private:
  const int volume;
  const int half_volume;

public:
  LoadStore(int vol) : volume(vol), half_volume(vol / 2) { }

  void loadMatrixFromField(const real_t *const field, int oddBit, int half_lattice_index, matrix *const mat) const;

  void loadMatrixFromField(const real_t *const *const field, int oddBit, int dir, int half_lattice_index,
                           matrix *const mat) const;

  void loadMatrixFromField(const matrix *const field, int oddBit, int half_lattice_index, matrix *const mat) const;

  void loadMatrixFromField(const matrix *const *const field, int oddBit, int dir, int half_lattice_index,
                           matrix *const mat) const;

  void storeMatrixToField(const matrix &mat, int oddBit, int half_lattice_index, real_t *const field) const;

  void storeMatrixToField(const matrix &mat, int oddBit, int half_lattice_index, matrix *const field) const;

  void addMatrixToField(const matrix &mat, int oddBit, int half_lattice_index, real_t coeff, real_t *const) const;

  void addMatrixToField(const matrix &mat, int oddBit, int half_lattice_index, real_t coeff, matrix *const) const;

  void addMatrixToField(const matrix &mat, int oddBit, int dir, int half_lattice_index, real_t coeff,
                        real_t *const *const) const;

  void addMatrixToField(const matrix &mat, int oddBit, int dir, int half_lattice_index, real_t coeff,
                        matrix *const *const) const;

  void storeMatrixToMomentumField(const matrix &mat, int oddBit, int dir, int half_lattice_index, real_t coeff,
                                  real_t *const) const;
  real_t getData(const real_t *const *const field, int idx, int dir, int oddBit, int offset, int hfv) const;
  void addData(real_t *const *const field, int idx, int dir, int oddBit, int offset, real_t, int hfv) const;
  int half_idx_conversion_ex2normal(int half_lattice_index, const int *dim, int oddBit) const;
  int half_idx_conversion_normal2ex(int half_lattice_index, const int *dim, int oddBit) const;
};

template <typename real_t>
int LoadStore<real_t>::half_idx_conversion_ex2normal(int half_lattice_index_ex, const int *dim, int oddBit) const
{
  int X1 = dim[0];
  int X2 = dim[1];
  int X3 = dim[2];
  // int X4=dim[3];

  int E1 = dim[0] + 4;
  int E2 = dim[1] + 4;
  int E3 = dim[2] + 4;
  // int E4=dim[3]+4;
  int E1h = E1 / 2;

  int sid = half_lattice_index_ex;

  int za = sid / E1h;
  int x1h = sid - za * E1h;
  int zb = za / E2;
  int x2 = za - zb * E2;
  int x4 = zb / E3;
  int x3 = zb - x4 * E3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  int x1 = 2 * x1h + x1odd;

  int idx = ((x4 - 2) * X3 * X2 * X1 + (x3 - 2) * X2 * X1 + (x2 - 2) * X1 + (x1 - 2)) / 2;
  return idx;
}

template <class real_t>
int LoadStore<real_t>::half_idx_conversion_normal2ex(int half_lattice_index, const int *dim, int oddBit) const
{
  int X1 = dim[0];
  int X2 = dim[1];
  int X3 = dim[2];
  // int X4=dim[3];
  int X1h = X1 / 2;

  int E1 = dim[0] + 4;
  int E2 = dim[1] + 4;
  int E3 = dim[2] + 4;
  // int E4=dim[3]+4;
  // int E1h=E1/2;

  int sid = half_lattice_index;

  int za = sid / X1h;
  int x1h = sid - za * X1h;
  int zb = za / X2;
  int x2 = za - zb * X2;
  int x4 = zb / X3;
  int x3 = zb - x4 * X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  int x1 = 2 * x1h + x1odd;

  int idx = ((x4 + 2) * E3 * E2 * E1 + (x3 + 2) * E2 * E1 + (x2 + 2) * E1 + (x1 + 2)) / 2;

  return idx;
}

template <class real_t>
void LoadStore<real_t>::loadMatrixFromField(const typename LoadStore<real_t>::matrix *const field, int oddBit,
                                            int half_lattice_index, typename LoadStore<real_t>::matrix *const mat) const
{
  int hfv = (is_multi_gpu()) ? Vh_ex : Vh;
  *mat = field[oddBit * hfv + half_lattice_index];
}

template <class real_t>
void LoadStore<real_t>::loadMatrixFromField(const typename LoadStore<real_t>::matrix *const *const field, int oddBit,
                                            int dir, int half_lattice_index,
                                            typename LoadStore<real_t>::matrix *const mat) const
{
  int hfv = (is_multi_gpu()) ? Vh_ex : Vh;
  *mat = field[dir][oddBit * hfv + half_lattice_index];
}

template <class real_t>
void LoadStore<real_t>::storeMatrixToField(const typename LoadStore<real_t>::matrix &mat, int oddBit,
                                           int half_lattice_index, typename LoadStore<real_t>::matrix *const field) const
{
  int hfv = (is_multi_gpu()) ? Vh_ex : Vh;
  field[oddBit * hfv + half_lattice_index] = mat;
}

template <class real_t>
void LoadStore<real_t>::addMatrixToField(const typename LoadStore<real_t>::matrix &mat, int oddBit,
                                         int half_lattice_index, real_t coeff,
                                         typename LoadStore<real_t>::matrix *const field) const
{
  int hfv = (is_multi_gpu()) ? Vh_ex : Vh;
  field[oddBit * hfv + half_lattice_index] += coeff * mat;
}

template <class real_t>
void LoadStore<real_t>::addMatrixToField(const typename LoadStore<real_t>::matrix &mat, int oddBit, int dir,
                                         int half_lattice_index, real_t coeff,
                                         typename LoadStore<real_t>::matrix *const *const field) const
{
  int hfv = (is_multi_gpu()) ? Vh_ex : Vh;
  field[dir][oddBit * hfv + half_lattice_index] += coeff * mat;
}

template <class real_t>
void LoadStore<real_t>::storeMatrixToMomentumField(const typename LoadStore<real_t>::matrix &mat, int oddBit, int dir,
                                                   int half_lattice_index, real_t coeff, real_t *const field) const
{
  real_t *const mom_field = field + ((oddBit * half_volume + half_lattice_index) * 4 + dir) * 10;
  mom_field[0] = (mat(0, 1).real() - mat(1, 0).real()) * 0.5 * coeff;
  mom_field[1] = (mat(0, 1).imag() + mat(1, 0).imag()) * 0.5 * coeff;

  mom_field[2] = (mat(0, 2).real() - mat(2, 0).real()) * 0.5 * coeff;
  mom_field[3] = (mat(0, 2).imag() + mat(2, 0).imag()) * 0.5 * coeff;

  mom_field[4] = (mat(1, 2).real() - mat(2, 1).real()) * 0.5 * coeff;
  mom_field[5] = (mat(1, 2).imag() + mat(2, 1).imag()) * 0.5 * coeff;

  const real_t temp = (mat(0, 0).imag() + mat(1, 1).imag() + mat(2, 2).imag()) * 0.3333333333333333333;
  mom_field[6] = (mat(0, 0).imag() - temp) * coeff;
  mom_field[7] = (mat(1, 1).imag() - temp) * coeff;
  mom_field[8] = (mat(2, 2).imag() - temp) * coeff;
  mom_field[9] = 0.0;
}

template <int oddBit> struct Locator {
private:
  int local_dim[4];
  int volume;
  int half_index;
  int full_index;
  int full_coord[4];
  void getCoordsFromHalfIndex(int half_index, int coord[4]);
  void getCoordsFromFullIndex(int full_index, int coord[4]);
  void cache(int half_lattice_index); // caches the half-lattice index, full-lattice index,
                                      // and full-lattice coordinates

public:
  Locator(const int dim[4]);
  int getFullFromHalfIndex(int half_lattice_index);
  int getNeighborFromFullIndex(int full_lattice_index, int dir, int *err = NULL);
};

template <int oddBit> Locator<oddBit>::Locator(const int dim[4]) : half_index(-1), full_index(-1)
{
  volume = 1;
  for (int dir = 0; dir < 4; ++dir) {
    local_dim[dir] = dim[dir];
    volume *= local_dim[dir];
  }
}

// Store the half_lattice index, works out and stores the full lattice index
// and the coordinates.
template <int oddBit> void Locator<oddBit>::getCoordsFromHalfIndex(int half_lattice_index, int coord[4])
{
  int E1 = local_dim[0] + (is_multi_gpu() ? 4 : 0);
  int E2 = local_dim[1] + (is_multi_gpu() ? 4 : 0);
  int E3 = local_dim[2] + (is_multi_gpu() ? 4 : 0);
  // int E4 = local_dim[3] + (is_multi_gpu() ? 4 : 0);
  int E1h = E1 / 2;

  int z1 = half_lattice_index / E1h;
  int x1h = half_lattice_index - z1 * E1h;
  int z2 = z1 / E2;
  coord[1] = z1 - z2 * E2;
  coord[3] = z2 / E3;
  coord[2] = z2 - coord[3] * E3;
  int x1odd = (coord[1] + coord[2] + coord[3] + oddBit) & 1;
  coord[0] = 2 * x1h + x1odd;
}

template <int oddBit> void Locator<oddBit>::getCoordsFromFullIndex(int full_lattice_index, int coord[4])
{
  int D1 = local_dim[0] + (is_multi_gpu() ? 4 : 0);
  int D2 = local_dim[1] + (is_multi_gpu() ? 4 : 0);
  int D3 = local_dim[2] + (is_multi_gpu() ? 4 : 0);
  // int D4 = local_dim[3] + (is_multi_gpu() ? 4 : 0);
  // int D1h = D1 / 2;

  int z1 = full_lattice_index / D1;
  coord[0] = full_lattice_index - z1 * D1;
  int z2 = z1 / D2;
  coord[1] = z1 - z2 * D2;
  coord[3] = z2 / D3;
  coord[2] = z2 - coord[3] * D3;
}

template <int oddBit> void Locator<oddBit>::cache(int half_lattice_index)
{
  half_index = half_lattice_index;
  getCoordsFromHalfIndex(half_lattice_index, full_coord);
  int x1odd = (full_coord[1] + full_coord[2] + full_coord[3] + oddBit) & 1;
  full_index = 2 * half_lattice_index + x1odd;
}

template <int oddBit> int Locator<oddBit>::getFullFromHalfIndex(int half_lattice_index)
{
  if (half_index != half_lattice_index) cache(half_lattice_index);
  return full_index;
}

// From full index return the neighbouring full index
template <int oddBit> int Locator<oddBit>::getNeighborFromFullIndex(int full_lattice_index, int dir, int *err)
{
  if (err) *err = 0;

  int coord[4];
  int neighbor_index = 0;
  getCoordsFromFullIndex(full_lattice_index, coord);
  if constexpr (is_multi_gpu()) {
    int E1 = local_dim[0] + 4;
    int E2 = local_dim[1] + 4;
    int E3 = local_dim[2] + 4;
    int E4 = local_dim[3] + 4;
    switch (dir) {
    case 0: //+X
      neighbor_index = full_lattice_index + 1;
      if (err && (coord[0] == E1 - 1)) *err = 1;
      break;
    case 1: //+Y
      neighbor_index = full_lattice_index + E1;
      if (err && (coord[1] == E2 - 1)) *err = 1;
      break;
    case 2: //+Z
      neighbor_index = full_lattice_index + E2 * E1;
      if (err && (coord[2] == E3 - 1)) *err = 1;
      break;
    case 3: //+T
      neighbor_index = full_lattice_index + E3 * E2 * E1;
      if (err && (coord[3] == E4 - 1)) *err = 1;
      break;
    case 7: //-X
      neighbor_index = full_lattice_index - 1;
      if (err && (coord[0] == 0)) *err = 1;
      break;
    case 6: //-Y
      neighbor_index = full_lattice_index - E1;
      if (err && (coord[1] == 0)) *err = 1;
      break;
    case 5: //-Z
      neighbor_index = full_lattice_index - E2 * E1;
      if (err && (coord[2] == 0)) *err = 1;
      break;
    case 4: //-T
      neighbor_index = full_lattice_index - E3 * E2 * E1;
      if (err && (coord[3] == 0)) *err = 1;
      break;
    default: errorQuda("Neighbor index could not be determined"); break;
    } // switch(dir)
  } else {
    switch (dir) {
    case 0:
      neighbor_index = (coord[0] == local_dim[0] - 1) ? full_lattice_index + 1 - local_dim[0] : full_lattice_index + 1;
      break;
    case 1:
      neighbor_index = (coord[1] == local_dim[1] - 1) ? full_lattice_index + local_dim[0] * (1 - local_dim[1]) :
                                                        full_lattice_index + local_dim[0];
      break;
    case 2:
      neighbor_index = (coord[2] == local_dim[2] - 1) ?
        full_lattice_index + local_dim[0] * local_dim[1] * (1 - local_dim[2]) :
        full_lattice_index + local_dim[0] * local_dim[1];
      break;
    case 3:
      neighbor_index = (coord[3] == local_dim[3] - 1) ?
        full_lattice_index + local_dim[0] * local_dim[1] * local_dim[2] * (1 - local_dim[3]) :
        full_lattice_index + local_dim[0] * local_dim[1] * local_dim[2];
      break;
    case 7: neighbor_index = (coord[0] == 0) ? full_lattice_index - 1 + local_dim[0] : full_lattice_index - 1; break;
    case 6:
      neighbor_index
        = (coord[1] == 0) ? full_lattice_index - local_dim[0] * (1 - local_dim[1]) : full_lattice_index - local_dim[0];
      break;
    case 5:
      neighbor_index = (coord[2] == 0) ? full_lattice_index - local_dim[0] * local_dim[1] * (1 - local_dim[2]) :
                                         full_lattice_index - local_dim[0] * local_dim[1];
      break;
    case 4:
      neighbor_index = (coord[3] == 0) ?
        full_lattice_index - local_dim[0] * local_dim[1] * local_dim[2] * (1 - local_dim[3]) :
        full_lattice_index - local_dim[0] * local_dim[1] * local_dim[2];
      break;
    default: errorQuda("Neighbor index could not be determined"); break;
    } // switch(dir)
    if (err) *err = 0;
  }
  return neighbor_index;
}

template <class real_t> struct PathCoefficients {
  real_t one;
  real_t three;
  real_t five;
  real_t seven;
  real_t naik;
  real_t lepage;
};

template <typename real_t> struct HisqStaplesForce {
  using complex = std::complex<real_t>;
  using matrix = Matrix<3, complex>;

  template <int oddBit>
  void computeOneLinkSite(const int dim[4], int half_lattice_index, const matrix *const *const oprod, int sig,
                          real_t coeff, const LoadStore<real_t> &ls, matrix *const *const output)
  {
    if (GOES_FORWARDS(sig)) {
      matrix colorMatW;

      int idx = is_multi_gpu() ? ls.half_idx_conversion_normal2ex(half_lattice_index, dim, oddBit) : half_lattice_index;

      ls.loadMatrixFromField(oprod, oddBit, sig, idx, &colorMatW);
      ls.addMatrixToField(colorMatW, oddBit, sig, idx, coeff, output);
    }
  }

  void computeOneLinkField(const int dim[4], const matrix *const *const oprod, int sig, real_t coeff,
                           matrix *const *const output)
  {
    int volume = 1;
    for (int dir = 0; dir < 4; ++dir) volume *= dim[dir];
    const int half_volume = volume / 2;
    LoadStore<real_t> ls(volume);

#pragma omp parallel for
    for (int site = 0; site < half_volume; ++site) { computeOneLinkSite<0>(dim, site, oprod, sig, coeff, ls, output); }
// Loop over odd lattice sites
#pragma omp parallel for
    for (int site = 0; site < half_volume; ++site) { computeOneLinkSite<1>(dim, site, oprod, sig, coeff, ls, output); }
  }

  template <int oddBit>
  void computeMiddleLinkSite(int half_lattice_index, const int dim[4], matrix *const oprod, const matrix *const Qprev,
                             const matrix *const *const link, int sig, int mu, real_t coeff, const LoadStore<real_t> &ls,
                             matrix *const Pmu, matrix *const P3, matrix *const Qmu, matrix *const *const newOprod)
  {
    const bool mu_positive = (GOES_FORWARDS(mu)) ? true : false;
    const bool sig_positive = (GOES_FORWARDS(sig)) ? true : false;

    Locator<oddBit> locator(dim);
    int point_b, point_c, point_d;
    int ad_link_nbr_idx, ab_link_nbr_idx, bc_link_nbr_idx;
    int X = locator.getFullFromHalfIndex(half_lattice_index);

    int err;
    int new_mem_idx = locator.getNeighborFromFullIndex(X, OPP_DIR(mu), &err);
    RETURN_IF_ERR;
    point_d = new_mem_idx >> 1;
    // getNeighborFromFullIndex will work on any site on the lattice, odd or even
    new_mem_idx = locator.getNeighborFromFullIndex(new_mem_idx, sig, &err);
    RETURN_IF_ERR;
    point_c = new_mem_idx >> 1;

    new_mem_idx = locator.getNeighborFromFullIndex(X, sig);
    RETURN_IF_ERR;
    point_b = new_mem_idx >> 1;

    ad_link_nbr_idx = (mu_positive) ? point_d : half_lattice_index;
    bc_link_nbr_idx = (mu_positive) ? point_c : point_b;
    ab_link_nbr_idx = (sig_positive) ? half_lattice_index : point_b;

    matrix ab_link, bc_link, ad_link;
    matrix colorMatW, colorMatX, colorMatY, colorMatZ;

    if (sig_positive) {
      ls.loadMatrixFromField(link, oddBit, sig, ab_link_nbr_idx, &ab_link);
    } else {
      ls.loadMatrixFromField(link, 1 - oddBit, OPP_DIR(sig), ab_link_nbr_idx, &ab_link);
    }

    if (mu_positive) {
      ls.loadMatrixFromField(link, oddBit, mu, bc_link_nbr_idx, &bc_link);
    } else {
      ls.loadMatrixFromField(link, 1 - oddBit, OPP_DIR(mu), bc_link_nbr_idx, &bc_link);
    }

    if (Qprev == NULL) {
      if (sig_positive) {
        ls.loadMatrixFromField(reinterpret_cast<const matrix *const *>(oprod), 1 - oddBit, sig, point_d, &colorMatY);
      } else {
        ls.loadMatrixFromField(reinterpret_cast<const matrix *const *>(oprod), oddBit, OPP_DIR(sig), point_c, &colorMatY);
        colorMatY = conj(colorMatY);
      }
    } else { // Qprev != NULL
      ls.loadMatrixFromField(oprod, oddBit, point_c, &colorMatY);
    }

    colorMatW = (!mu_positive) ? bc_link * colorMatY : conj(bc_link) * colorMatY;
    if (Pmu) ls.storeMatrixToField(colorMatW, 1 - oddBit, point_b, Pmu);

    colorMatY = (sig_positive) ? ab_link * colorMatW : conj(ab_link) * colorMatW;
    ls.storeMatrixToField(colorMatY, oddBit, half_lattice_index, P3);

    if (mu_positive) {
      ls.loadMatrixFromField(link, 1 - oddBit, mu, ad_link_nbr_idx, &ad_link);
    } else {
      ls.loadMatrixFromField(link, oddBit, OPP_DIR(mu), ad_link_nbr_idx, &ad_link);
      ad_link = conj(ad_link);
    }

    if (Qprev == NULL) {
      if (sig_positive) colorMatY = colorMatW * ad_link;
      if (Qmu) ls.storeMatrixToField(ad_link, oddBit, half_lattice_index, Qmu);
    } else { // Qprev != NULL
      if (Qmu || sig_positive) {
        ls.loadMatrixFromField(Qprev, 1 - oddBit, point_d, &colorMatY);
        colorMatX = colorMatY * ad_link;
      }
      if (Qmu) ls.storeMatrixToField(colorMatX, oddBit, half_lattice_index, Qmu);
      if (sig_positive) colorMatY = colorMatW * colorMatX;
    }

    if (sig_positive) ls.addMatrixToField(colorMatY, oddBit, sig, half_lattice_index, coeff, newOprod);
  } // computeMiddleLinkSite

  void computeMiddleLinkField(const int dim[4], matrix *const oprod, const matrix *const Qprev,
                              const matrix *const *const link, int sig, int mu, real_t coeff, matrix *const Pmu,
                              matrix *const P3, matrix *const Qmu, matrix *const *const newOprod)
  {

    int volume = 1;
    for (int dir = 0; dir < 4; ++dir) volume *= dim[dir];
    const int loop_count = is_multi_gpu() ? Vh_ex : volume / 2;

    // loop over the lattice volume
    // To keep the code as close to the GPU code as possible, we'll
    // loop over the even sites first and then the odd sites
    LoadStore<real_t> ls(volume);
#pragma omp parallel for
    for (int site = 0; site < loop_count; ++site) {
      computeMiddleLinkSite<0>(site, dim, oprod, Qprev, link, sig, mu, coeff, ls, Pmu, P3, Qmu, newOprod);
    }
    // Loop over odd lattice sites
#pragma omp parallel for
    for (int site = 0; site < loop_count; ++site) {
      computeMiddleLinkSite<1>(site, dim, oprod, Qprev, link, sig, mu, coeff, ls, Pmu, P3, Qmu, newOprod);
    }
  }

  template <int oddBit>
  void computeSideLinkSite(int half_lattice_index, const int dim[4], const matrix *const P3, const matrix *const Qprod,
                           const matrix *const *const link, int sig, int mu, real_t coeff, real_t accumu_coeff,
                           const LoadStore<real_t> &ls, matrix *const shortP, matrix *const *const newOprod)
  {

    const bool mu_positive = (GOES_FORWARDS(mu)) ? true : false;
    const bool sig_positive = (GOES_FORWARDS(sig)) ? true : false;

    Locator<oddBit> locator(dim);
    int point_d;
    int ad_link_nbr_idx;
    int X = locator.getFullFromHalfIndex(half_lattice_index);

    int err;
    int new_mem_idx = locator.getNeighborFromFullIndex(X, OPP_DIR(mu), &err);
    RETURN_IF_ERR;
    point_d = new_mem_idx >> 1;
    ad_link_nbr_idx = (mu_positive) ? point_d : half_lattice_index;

    matrix ad_link;
    matrix colorMatW, colorMatX, colorMatY;

    ls.loadMatrixFromField(P3, oddBit, half_lattice_index, &colorMatY);

    if (shortP) {
      if (mu_positive) {
        ad_link_nbr_idx = point_d;
        ls.loadMatrixFromField(link, 1 - oddBit, mu, ad_link_nbr_idx, &ad_link);
      } else {
        ad_link_nbr_idx = half_lattice_index;
        ls.loadMatrixFromField(link, oddBit, OPP_DIR(mu), ad_link_nbr_idx, &ad_link);
      }
      colorMatW = (mu_positive) ? ad_link * colorMatY : conj(ad_link) * colorMatY;
      ls.addMatrixToField(colorMatW, 1 - oddBit, point_d, accumu_coeff, shortP);
    } // if(shortP)

    real_t mycoeff = ((sig_positive && oddBit) || (!sig_positive && !oddBit)) ? coeff : -coeff;

    if (Qprod) {
      ls.loadMatrixFromField(Qprod, 1 - oddBit, point_d, &colorMatX);
      if (mu_positive) {
        colorMatW = colorMatY * colorMatX;
        if (!oddBit) { mycoeff = -mycoeff; }
        ls.addMatrixToField(colorMatW, 1 - oddBit, mu, point_d, mycoeff, newOprod);
      } else {
        colorMatW = conj(colorMatX) * conj(colorMatY);
        if (oddBit) { mycoeff = -mycoeff; }
        ls.addMatrixToField(colorMatW, oddBit, OPP_DIR(mu), half_lattice_index, mycoeff, newOprod);
      }
    }

    if (!Qprod) {
      if (mu_positive) {
        if (!oddBit) { mycoeff = -mycoeff; }
        ls.addMatrixToField(colorMatY, 1 - oddBit, mu, point_d, mycoeff, newOprod);
      } else {
        if (oddBit) { mycoeff = -mycoeff; }
        colorMatW = conj(colorMatY);
        ls.addMatrixToField(colorMatW, oddBit, OPP_DIR(mu), half_lattice_index, mycoeff, newOprod);
      }
    } // if !(Qprod)

  } // computeSideLinkSite

  void computeSideLinkField(const int dim[4], const matrix *const P3, const matrix *const Qprod,
                            const matrix *const *const link, int sig, int mu, real_t coeff, real_t accumu_coeff,
                            matrix *const shortP, matrix *const *const newOprod)
  {
    // Need some way of setting half_volume
    int volume = 1;
    for (int dir = 0; dir < 4; ++dir) volume *= dim[dir];
    const int loop_count = is_multi_gpu() ? Vh_ex : volume / 2;
    LoadStore<real_t> ls(volume);

#pragma omp parallel for
    for (int site = 0; site < loop_count; ++site) {
      computeSideLinkSite<0>(site, dim, P3, Qprod, link, sig, mu, coeff, accumu_coeff, ls, shortP, newOprod);
    }

#pragma omp parallel for
    for (int site = 0; site < loop_count; ++site) {
      computeSideLinkSite<1>(site, dim, P3, Qprod, link, sig, mu, coeff, accumu_coeff, ls, shortP, newOprod);
    }
  }

  template <int oddBit>
  void computeAllLinkSite(int half_lattice_index, const int dim[4], const matrix *const oprod, const matrix *const Qprev,
                          const matrix *const *const link, int sig, int mu, real_t coeff, real_t accumu_coeff,
                          const LoadStore<real_t> &ls, matrix *const shortP, matrix *const *const newOprod)
  {
    const bool mu_positive = (GOES_FORWARDS(mu)) ? true : false;
    const bool sig_positive = (GOES_FORWARDS(sig)) ? true : false;

    matrix ab_link, bc_link, ad_link;
    matrix colorMatW, colorMatX, colorMatY, colorMatZ;

    int ab_link_nbr_idx, point_b, point_c, point_d;

    Locator<oddBit> locator(dim);
    int X = locator.getFullFromHalfIndex(half_lattice_index);

    int err;
    int new_mem_idx = locator.getNeighborFromFullIndex(X, OPP_DIR(mu), &err);
    RETURN_IF_ERR;
    point_d = new_mem_idx >> 1;

    new_mem_idx = locator.getNeighborFromFullIndex(new_mem_idx, sig, &err);
    RETURN_IF_ERR;
    point_c = new_mem_idx >> 1;

    new_mem_idx = locator.getNeighborFromFullIndex(X, sig, &err);
    RETURN_IF_ERR;
    point_b = new_mem_idx >> 1;
    ab_link_nbr_idx = (sig_positive) ? half_lattice_index : point_b;

    real_t mycoeff = ((sig_positive && oddBit) || (!sig_positive && !oddBit)) ? coeff : -coeff;

    if (mu_positive) {
      ls.loadMatrixFromField(Qprev, 1 - oddBit, point_d, &colorMatX);
      ls.loadMatrixFromField(link, 1 - oddBit, mu, point_d, &ad_link);
      // compute point_c
      ls.loadMatrixFromField(oprod, oddBit, point_c, &colorMatY);
      ls.loadMatrixFromField(link, oddBit, mu, point_c, &bc_link);
      colorMatZ = conj(bc_link) * colorMatY; // okay

      if (sig_positive) {
        colorMatY = colorMatX * ad_link;
        colorMatW = colorMatZ * colorMatY;
        ls.addMatrixToField(colorMatW, oddBit, sig, half_lattice_index, Sign<oddBit>::result * mycoeff, newOprod);
      }

      if (sig_positive) {
        ls.loadMatrixFromField(link, oddBit, sig, ab_link_nbr_idx, &ab_link);
      } else {
        ls.loadMatrixFromField(link, 1 - oddBit, OPP_DIR(sig), ab_link_nbr_idx, &ab_link);
      }
      colorMatY = (sig_positive) ? ab_link * colorMatZ : conj(ab_link) * colorMatZ; // okay
      colorMatW = colorMatY * colorMatX;
      ls.addMatrixToField(colorMatW, 1 - oddBit, mu, point_d, -Sign<oddBit>::result * mycoeff, newOprod);
      colorMatW = ad_link * colorMatY;
      ls.addMatrixToField(colorMatW, 1 - oddBit, point_d, accumu_coeff, shortP); // Warning! Need to check this!
    } else {                                                                     // negative mu
      mu = OPP_DIR(mu);
      ls.loadMatrixFromField(Qprev, 1 - oddBit, point_d, &colorMatX);
      ls.loadMatrixFromField(link, oddBit, mu, half_lattice_index, &ad_link);
      ls.loadMatrixFromField(oprod, oddBit, point_c, &colorMatY);
      ls.loadMatrixFromField(link, 1 - oddBit, mu, point_b, &bc_link);

      if (sig_positive) colorMatW = colorMatX * conj(ad_link);
      colorMatZ = bc_link * colorMatY;
      if (sig_positive) {
        colorMatY = colorMatZ * colorMatW;
        ls.addMatrixToField(colorMatY, oddBit, sig, half_lattice_index, Sign<oddBit>::result * mycoeff, newOprod);
      }

      if (sig_positive) {
        ls.loadMatrixFromField(link, oddBit, sig, ab_link_nbr_idx, &ab_link);
      } else {
        ls.loadMatrixFromField(link, 1 - oddBit, OPP_DIR(sig), ab_link_nbr_idx, &ab_link);
      }

      colorMatY = (sig_positive) ? ab_link * colorMatZ : conj(ab_link) * colorMatZ; // 611
      colorMatW = conj(colorMatX) * conj(colorMatY);

      ls.addMatrixToField(colorMatW, oddBit, mu, half_lattice_index, Sign<oddBit>::result * mycoeff, newOprod);
      colorMatW = conj(ad_link) * colorMatY;
      ls.addMatrixToField(colorMatW, 1 - oddBit, point_d, accumu_coeff, shortP);
    } // end mu
  }   // allLinkKernel

  void computeAllLinkField(const int dim[4], const matrix *const oprod, const matrix *const Qprev,
                           const matrix *const *const link, int sig, int mu, real_t coeff, real_t accumu_coeff,
                           matrix *const shortP, matrix *const *const newOprod)
  {
    int volume = 1;
    for (int dir = 0; dir < 4; ++dir) volume *= dim[dir];
    const int loop_count = is_multi_gpu() ? Vh_ex : volume / 2;

    LoadStore<real_t> ls(volume);
#pragma omp parallel for
    for (int site = 0; site < loop_count; ++site) {

      computeAllLinkSite<0>(site, dim, oprod, Qprev, link, sig, mu, coeff, accumu_coeff, ls, shortP, newOprod);
    }

    for (int site = 0; site < loop_count; ++site) {
      computeAllLinkSite<1>(site, dim, oprod, Qprev, link, sig, mu, coeff, accumu_coeff, ls, shortP, newOprod);
    }
  }

  void operator()(const int dim[4], PathCoefficients<double> staple_coeff, void **oprod_, void **link_, void **tempmat_,
                  void **newOprod_)
  {
    auto oprod = reinterpret_cast<matrix **>(oprod_);
    auto link = reinterpret_cast<matrix **>(link_);
    auto tempmat = reinterpret_cast<matrix **>(tempmat_);
    auto newOprod = reinterpret_cast<matrix **>(newOprod_);

    real_t OneLink, ThreeSt, FiveSt, SevenSt, Lepage, coeff;

    OneLink = staple_coeff.one;
    ThreeSt = staple_coeff.three;
    FiveSt = staple_coeff.five;
    SevenSt = staple_coeff.seven;
    Lepage = staple_coeff.lepage;

    auto Pmu = tempmat[0];
    auto P3 = tempmat[1];
    auto P5 = tempmat[2];
    auto Pnumu = tempmat[3];
    auto Qmu = tempmat[4];
    auto Qnumu = tempmat[5];

    for (int sig = 0; sig < 4; ++sig) { computeOneLinkField(dim, oprod, sig, OneLink, newOprod); }

    // sig labels the net displacement of the staple
    for (int sig = 0; sig < 8; ++sig) {
      for (int mu = 0; mu < 8; ++mu) {
        if (mu == sig || mu == OPP_DIR(sig)) continue;

        // no, that first cast isn't a typo.
        computeMiddleLinkField(dim, reinterpret_cast<matrix *>(oprod), static_cast<matrix *>(NULL), link, sig, mu,
                               -ThreeSt, Pmu, P3, Qmu, newOprod);

        for (int nu = 0; nu < 8; ++nu) {
          if (nu == mu || nu == OPP_DIR(mu) || nu == sig || nu == OPP_DIR(sig)) continue;

          computeMiddleLinkField(dim, Pmu, Qmu, link, sig, nu, staple_coeff.five, Pnumu, P5, Qnumu, newOprod);

          for (int rho = 0; rho < 8; ++rho) {
            if (rho == sig || rho == OPP_DIR(sig) || rho == mu || rho == OPP_DIR(mu) || rho == nu || rho == OPP_DIR(nu)) {
              continue;
            }

            if (FiveSt != 0)
              coeff = SevenSt / FiveSt;
            else
              coeff = 0;
            computeAllLinkField(dim, Pnumu, Qnumu, link, sig, rho, staple_coeff.seven, coeff, P5, newOprod);

          } // rho

          // 5-staple: side link
          if (ThreeSt != 0)
            coeff = FiveSt / ThreeSt;
          else
            coeff = 0;
          computeSideLinkField(dim, P5, Qmu, link, sig, nu, -FiveSt, coeff, P3, newOprod);

        } // nu

        // lepage
        if (staple_coeff.lepage != 0.) {
          computeMiddleLinkField(dim, Pmu, Qmu, link, sig, mu, Lepage, static_cast<matrix *>(NULL), P5,
                                 static_cast<matrix *>(NULL), newOprod);

          if (ThreeSt != 0)
            coeff = Lepage / ThreeSt;
          else
            coeff = 0;
          computeSideLinkField(dim, P5, Qmu, link, sig, mu, -Lepage, coeff, P3, newOprod);
        } // lepage != 0

        computeSideLinkField(dim, P3, static_cast<matrix *>(NULL), link, sig, mu, ThreeSt, 0.,
                             static_cast<matrix *>(NULL), newOprod);
      } // mu
    }   // sig

    // Need also to compute the one-link contribution
  }
};

void hisqStaplesForceCPU(const double *path_coeff, quda::GaugeField &oprod, quda::GaugeField &link,
                         quda::GaugeField *newOprod)
{
  auto precision = quda::checkPrecision(oprod, link, *newOprod);
  int X_[4];
  for (int d = 0; d < 4; d++) X_[d] = oprod.X()[d] - 2 * oprod.R()[d];

  uint64_t len = is_multi_gpu() ? (2 * Vh_ex) : (X_[0] * X_[1] * X_[2] * X_[3]);

  PathCoefficients<double> act_path_coeff;
  act_path_coeff.one = path_coeff[0];
  act_path_coeff.naik = path_coeff[1];
  act_path_coeff.three = path_coeff[2];
  act_path_coeff.five = path_coeff[3];
  act_path_coeff.seven = path_coeff[4];
  act_path_coeff.lepage = path_coeff[5];

  // temporary LatticeColorMatrix
  void *tempmat[6];
  for (int i = 0; i < 6; i++) { tempmat[i] = safe_malloc(len * 18 * precision); }

  instantiate_host<HisqStaplesForce>(precision, X_, act_path_coeff, oprod.data_array().data, link.data_array().data,
                                     tempmat, newOprod->data_array().data);

  for (int i = 0; i < 6; ++i) { host_free(tempmat[i]); }
}

template <class real_t> struct ComputeLongLinkField {
  using complex = std::complex<real_t>;
  using matrix = Matrix<3, complex>;

  template <int oddBit>
  void computeLongLinkSite(int half_lattice_index, const int dim[4], const matrix *const *const oprod,
                           const matrix *const *const link, int sig, real_t coeff, const LoadStore<real_t> &ls,
                           matrix *const *const output)
  {
    if (GOES_FORWARDS(sig)) {

      Locator<oddBit> locator(dim);

      matrix ab_link, bc_link, de_link, ef_link;
      matrix colorMatU, colorMatV, colorMatW, colorMatX, colorMatY, colorMatZ;

      int point_a, point_b, point_c, point_d, point_e;
      int idx = is_multi_gpu() ? ls.half_idx_conversion_normal2ex(half_lattice_index, dim, oddBit) : half_lattice_index;

      int X = locator.getFullFromHalfIndex(idx);
      point_c = idx;

      int new_mem_idx = locator.getNeighborFromFullIndex(X, sig);
      point_d = new_mem_idx >> 1;

      new_mem_idx = locator.getNeighborFromFullIndex(new_mem_idx, sig);
      point_e = new_mem_idx >> 1;

      new_mem_idx = locator.getNeighborFromFullIndex(X, OPP_DIR(sig));
      point_b = new_mem_idx >> 1;

      new_mem_idx = locator.getNeighborFromFullIndex(new_mem_idx, OPP_DIR(sig));
      point_a = new_mem_idx >> 1;

      ls.loadMatrixFromField(link, oddBit, sig, point_a, &ab_link);
      ls.loadMatrixFromField(link, 1 - oddBit, sig, point_b, &bc_link);
      ls.loadMatrixFromField(link, 1 - oddBit, sig, point_d, &de_link);
      ls.loadMatrixFromField(link, oddBit, sig, point_e, &ef_link);

      ls.loadMatrixFromField(oprod, oddBit, sig, point_c, &colorMatZ);
      ls.loadMatrixFromField(oprod, 1 - oddBit, sig, point_b, &colorMatY);
      ls.loadMatrixFromField(oprod, oddBit, sig, point_a, &colorMatX);

      colorMatV = de_link * ef_link * colorMatZ - de_link * colorMatY * bc_link + colorMatX * ab_link * bc_link;

      ls.addMatrixToField(colorMatV, oddBit, sig, point_c, coeff, output);
    }
  }

  void operator()(const int dim[4], const void *const *const oprod_, const void *const *const link_, int sig,
                  real_t coeff, void *const *const output_)
  {
    auto oprod = reinterpret_cast<const matrix *const *>(oprod_);
    auto link = reinterpret_cast<const matrix *const *>(link_);
    auto output = reinterpret_cast<matrix *const *>(output_);

    int volume = 1;
    for (int dir = 0; dir < 4; ++dir) volume *= dim[dir];
    const int half_volume = volume / 2;

    LoadStore<real_t> ls(volume);
#pragma omp parallel for
    for (int site = 0; site < half_volume; ++site) {
      computeLongLinkSite<0>(site, dim, oprod, link, sig, coeff, ls, output);
    }
    // Loop over odd lattice sites
#pragma omp parallel for
    for (int site = 0; site < half_volume; ++site) {
      computeLongLinkSite<1>(site, dim, oprod, link, sig, coeff, ls, output);
    }
  }
};

void hisqLongLinkForceCPU(double coeff, quda::GaugeField &oprod, quda::GaugeField &link, quda::GaugeField *newOprod)
{
  auto precision = quda::checkPrecision(oprod, link, *newOprod);
  int X_[4];
  for (int d = 0; d < 4; d++) X_[d] = oprod.X()[d] - 2 * oprod.R()[d];

  for (int sig = 0; sig < 4; ++sig) {
    instantiate_host<ComputeLongLinkField>(precision, X_, oprod.data_array().data, link.data_array().data, sig, coeff,
                                           newOprod->data_array().data);
  } // sig
}

template <class real_t> struct CompleteForceField {
  using complex = std::complex<real_t>;
  using matrix = Matrix<3, complex>;

  template <int oddBit>
  void completeForceSite(int half_lattice_index, const int dim[4], const matrix *const *const oprod,
                         const matrix *const *const link, int sig, const LoadStore<real_t> &ls, real_t *const mom)
  {
    matrix colorMatX, colorMatY, linkW;

    int idx = is_multi_gpu() ? ls.half_idx_conversion_normal2ex(half_lattice_index, dim, oddBit) : half_lattice_index;

    ls.loadMatrixFromField(link, oddBit, sig, idx, &linkW);
    ls.loadMatrixFromField(oprod, oddBit, sig, idx, &colorMatX);

    const real_t coeff = (oddBit) ? -1 : 1;
    colorMatY = linkW * colorMatX;

    ls.storeMatrixToMomentumField(colorMatY, oddBit, sig, half_lattice_index, coeff, mom);
  }

  void operator()(const int dim[4], const void *const *const oprod_, const void *const *const link_, int sig,
                  void *const mom_)
  {
    auto oprod = reinterpret_cast<const matrix *const *>(oprod_);
    auto link = reinterpret_cast<const matrix *const *>(link_);
    auto mom = reinterpret_cast<real_t *>(mom_);

    int volume = dim[0] * dim[1] * dim[2] * dim[3];
    const int half_volume = volume / 2;
    LoadStore<real_t> ls(volume);

#pragma omp parallel for
    for (int site = 0; site < half_volume; ++site) { completeForceSite<0>(site, dim, oprod, link, sig, ls, mom); }
#pragma omp parallel for
    for (int site = 0; site < half_volume; ++site) { completeForceSite<1>(site, dim, oprod, link, sig, ls, mom); }
  }
};

void hisqCompleteForceCPU(quda::GaugeField &oprod, quda::GaugeField &link, quda::GaugeField *mom)
{
  auto precision = quda::checkPrecision(oprod, link, *mom);
  int X_[4];
  for (int d = 0; d < 4; d++) X_[d] = oprod.X()[d] - 2 * oprod.R()[d];

  for (int sig = 0; sig < 4; ++sig) {
    instantiate_host<CompleteForceField>(precision, X_, oprod.data_array().data, link.data_array().data, sig,
                                         mom->data());
  } // loop over sig
}
