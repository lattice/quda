
#include <dirac_quda.h>
#include <blas_quda.h>
#include <cmath>

namespace quda
{

  struct PolynomialBasisParams {

    /** What basis polynomial are we using */
    QudaPolynomialBasis basis;

    /** What order polynomial are we generating? */
    int n_order;

    /** How often are we re-normalizing the vector to avoid overflow
        Set to 0 to never normalize */
    int normalize_freq;

    /** Chebyshev basis: slope of mapping onto [-1,1] */
    double m_map;

    /** Chebyshev basis: intercept of mapping onto [-1,1] */
    double b_map;

    /** Vector of temporary vectors; need 2 for power iterations, 4 for chebyshev basis */
    std::vector<ColorSpinorField_ref> tmp_vectors;

    PolynomialBasisParams() :
      basis(QUDA_INVALID_BASIS),
      n_order(0),
      normalize_freq(0),
      m_map(0.0),
      b_map(0.0),
      tmp_vectors() { }

    static double compute_m_map(double lambda_min, double lambda_max) { return 2. / (lambda_max - lambda_min); }
    static double compute_b_map(double lambda_min, double lambda_max) { return - (lambda_max + lambda_min) / (lambda_max - lambda_min); }

    static void check_params(const PolynomialBasisParams& params, bool skip_temporaries_check = false) {
      if (params.n_order < 0) errorQuda("Invalid polynomial order %d", params.n_order);
      if (params.normalize_freq < 0) errorQuda("Invalid rescale frequency %d", params.normalize_freq);

      // the temporary check does not need to be done for CA basis generation
      if (!skip_temporaries_check) {
        switch (params.basis) {
          case QUDA_POWER_BASIS:
            if (params.tmp_vectors.size() < 2) errorQuda("Invalid temporary vector count %lu for power basis, expected 2", params.tmp_vectors.size());
            break;
          case QUDA_CHEBYSHEV_BASIS:
            if (params.m_map < 0) errorQuda("Invalid m_map %e, implies lambda_min >= lambda_max", params.m_map);
            if (params.tmp_vectors.size() < 4) errorQuda("Invalid temporary vector count %lu for Chebyshev basis, expected 4", params.tmp_vectors.size());
            break;
          default: errorQuda("Polynomial basis is unspecified");
        }
      }
    }
  };

  /**
    @brief Apply a polynomial basis to a starting vector, optionally saving with some frequency
    @param[in] diracm Dirac matrix used for the polynomial
    @param[out] output_vecs Output vectors, which may be more than one if the save frquency is non-zero
    @param[in] start_vec Starting vector for polynomial application
    @param[in] poly_params Parameters for the polynomial application
    @param[in] save_freq Frequency with which vectors are saved to output_vecs
    @param[in] args Parameter pack of ColorSpinorFields used as temporaries passed to Dirac
  */
  template <typename... Args>
  void applyMatrixPolynomial(const DiracMatrix &diracm, std::vector<ColorSpinorField_ref> &output_vecs,
                                     const ColorSpinorField &start_vec, const PolynomialBasisParams &poly_params, int save_freq,
                                     Args &&...args)
  {
    PolynomialBasisParams::check_params(poly_params);
    // if the save_frequency is 0, make sure output_vecs is of length 1
    if (save_freq < 0)
      errorQuda("Invalid save frequency %lu", output_vecs.size());
    else if (save_freq == 0 && output_vecs.size() != 1)
      errorQuda("Invalid vector length %lu", output_vecs.size());
    else if (save_freq > 0 && poly_params.n_order % save_freq != 0)
      errorQuda("Saving frequency %d does not evenly divide into polynomial order %d", save_freq, poly_params.n_order);
    else if (save_freq > 0 && static_cast<int>(output_vecs.size()) != poly_params.n_order / save_freq)
      errorQuda("Invalid vector length %lu, expected %d", output_vecs.size(), poly_params.n_order / save_freq);

    int save_count = 0;

    if (poly_params.basis == QUDA_POWER_BASIS) {
      ColorSpinorField &tempvec1 = poly_params.tmp_vectors[0];
      ColorSpinorField &tempvec2 = poly_params.tmp_vectors[1];
      blas::copy(tempvec1, start_vec); // no-op if fieds alias

      for (int i = 1; i <= poly_params.n_order; i++) {
        diracm(tempvec2, tempvec1, args...);
        if (save_freq > 0 && i % save_freq == 0)
          blas::copy(output_vecs[save_count++], tempvec2);
        if (poly_params.normalize_freq > 0 && i % poly_params.normalize_freq == 0) {
          double tmp_nrm = sqrt(blas::norm2(tempvec2));
          logQuda(QUDA_VERBOSE, "Triggered rescale during matrix polynomial application; norm at rescale is %e\n", tmp_nrm);
          blas::ax(1.0 / tmp_nrm, tempvec2);
        }
        std::swap(tempvec1, tempvec2);
      }

      // if save_freq != 0, the last vector has already been saved into output_vecs
      if (save_freq == 0) blas::copy(output_vecs[0], tempvec1);
    } else if (poly_params.basis == QUDA_CHEBYSHEV_BASIS) {

      ColorSpinorField &pk = poly_params.tmp_vectors[0];
      ColorSpinorField &pkm1 = poly_params.tmp_vectors[1];
      ColorSpinorField &pkm2 = poly_params.tmp_vectors[2];
      ColorSpinorField &Apkm1 = poly_params.tmp_vectors[3];

      int save_count = 0;

      // P0
      blas::copy(pk, start_vec);

      if (poly_params.n_order > 0) {
        // P1 = m Ap_0 + b p_0
        std::swap(pkm1, pk); // p_k -> p_{k - 1}
        diracm(Apkm1, pkm1, args...);
        blas::axpbyz(poly_params.m_map, Apkm1, poly_params.b_map, pkm1, pk);

        if (save_freq == 1) blas::copy(output_vecs[save_count++], pk);
        if (poly_params.normalize_freq == 1) {
          double tmp_nrm = sqrt(blas::norm2(pk));
	  logQuda(QUDA_VERBOSE, "Triggered rescale during matrix polynomial application; norm at rescale is %e\n", tmp_nrm);
          double tmp_inv_nrm = 1. / tmp_nrm;
	  blas::ax(tmp_inv_nrm, pk);
	  blas::ax(tmp_inv_nrm, pkm1);
	  blas::ax(tmp_inv_nrm, Apkm1);
        }

        if (poly_params.n_order > 1) {
          // Enter recursion relation
          for (int k = 2; k <= poly_params.n_order; k++) {
            std::swap(pkm2, pkm1); // p_{k - 1} -> p_{k-2}
            std::swap(pkm1, pk); // p_k -> p_{k-1}
            diracm(Apkm1, pkm1, args...); // compute A p_{k-1}
            blas::axpbypczw(2. * poly_params.m_map, Apkm1, 2. * poly_params.b_map, pkm1, -1., pkm2, pk);

            if (save_freq > 0 && k % save_freq == 0) blas::copy(output_vecs[save_count++], pk);
            // heuristic rescale to keep norms in check...
            if (poly_params.normalize_freq > 0 && k % poly_params.normalize_freq == 0) {
              double tmp_nrm = sqrt(blas::norm2(pk));
              logQuda(QUDA_VERBOSE, "Triggered rescale during matrix polynomial application; norm at rescale is %e\n", tmp_nrm);
              double tmp_inv_nrm = 1. / tmp_nrm;
	      blas::ax(tmp_inv_nrm, pk);
	      blas::ax(tmp_inv_nrm, pkm1);
	      blas::ax(tmp_inv_nrm, pkm2);
	      blas::ax(tmp_inv_nrm, Apkm1);
            }
          }
        }
      }

      // if save_freq != 0, the last vector has already been saved into output_vecs
      if (save_freq == 0) blas::copy(output_vecs[0], pk);
    } else {
      errorQuda("Invalid basis %d", poly_params.basis);
    }
  }

  /**
    @brief Apply a polynomial basis to a starting vector
    @param[in] diracm Dirac matrix used for the polynomial
    @param[out] output_vec Output vector
    @param[in] start_vec Starting vector for polynomial application
    @param[in] poly_params Parameters for the polynomial application
    @param[in] args Parameter pack of ColorSpinorFields used as temporaries passed to Dira
  */
  template <typename... Args>
  void applyMatrixPolynomial(const DiracMatrix &diracm, ColorSpinorField &output_vec,
                                     const ColorSpinorField &start_vec, const PolynomialBasisParams &poly_params,
                                     Args &&...args)
  {
    std::vector<ColorSpinorField_ref> output_vecs;
    output_vecs.emplace_back(std::ref(output_vec));
    applyMatrixPolynomial(diracm, output_vecs, start_vec, poly_params, 0, args...);
  }

  /**
     @brief Compute power iterations on a Dirac matrix
     @param[in] diracm Dirac matrix used for power iterations
     @param[in] start Starting rhs for power iterations; value preserved unless it aliases temporary vectors
     @param[in,out] poly_params Parameters for polynomial application, must correspond to power iterations
     @param[in] args Parameter pack of ColorSpinorFields used as temporary passed to Dirac
     @return Norm of final power iteration result
  */
  template <typename... Args>
  double performPowerIterations(const DiracMatrix &diracm, const ColorSpinorField &start,
                                        PolynomialBasisParams &poly_params, Args &&...args)
  {
    PolynomialBasisParams::check_params(poly_params);
    if (poly_params.basis != QUDA_POWER_BASIS) errorQuda("Invalid basis %d", poly_params.basis);

    auto tempvec1 = poly_params.tmp_vectors[0];
    auto tempvec2 = poly_params.tmp_vectors[1];

    applyMatrixPolynomial(diracm, tempvec1, start, poly_params, args...);

    // Get Rayleigh quotient
    double tmpnrm = sqrt(blas::norm2(tempvec1));
    blas::ax(1.0 / tmpnrm, tempvec1);
    diracm(tempvec2, tempvec1, args...);
    double lambda_max = sqrt(blas::norm2(tempvec2));
    logQuda(QUDA_VERBOSE, "Power iterations approximate max = %e\n", lambda_max);

    return lambda_max;
  }

  /**
    @brief Apply a polynomial basis to a starting vector, optionally saving with some frequency
    @param[in] diracm Dirac matrix used for the polynomial
    @param[out] Ap dirac matrix times the Krylov basis vectors
    @param[in,out] p Krylov basis vectors; assumes p[0] is in place
    @param[in] poly_params Parameters for the polynomial application
    @param[in] args Parameter pack of ColorSpinorFields used as temporaries passed to Dirac
  */
  template <typename... Args>
  void computeCAKrylovSpace(const DiracMatrix &diracm, std::vector<ColorSpinorField> &Ap,
                                    std::vector<ColorSpinorField> &p, PolynomialBasisParams poly_params, Args &&...args)
  {
    PolynomialBasisParams::check_params(poly_params, true);

    auto n_krylov = poly_params.n_order;

    // in some cases p or Ap may be larger
    if (static_cast<int>(p.size()) < n_krylov) errorQuda("Invalid p.size() %lu < n_krylov %d", p.size(), n_krylov);
    if (static_cast<int>(Ap.size()) < n_krylov) errorQuda("Invalid Ap.size() %lu < n_krylov %d", Ap.size(), n_krylov);

    if (poly_params.basis == QUDA_POWER_BASIS) {
      for (int k = 0; k < n_krylov; k++) {
        diracm(Ap[k], p[k], args...);
        if (k < (n_krylov - 1)) blas::copy(p[k + 1], Ap[k]); // no op if fields alias, which is often the case
      }
    } else if (poly_params.basis == QUDA_CHEBYSHEV_BASIS) {
      diracm(Ap[0], p[0], args...);

      if (n_krylov > 1) {
        // p_1 = m Ap_0 + b p_0
        blas::axpbyz(poly_params.m_map, Ap[0], poly_params.b_map, p[0], p[1]);
        diracm(Ap[1], p[1], args...);

        // Enter recursion relation
        if (n_krylov > 2) {
          // p_k = 2 m A[_{k-1} + 2 b p_{k-1} - p_{k-2}
          for (int k = 2; k < n_krylov; k++) {
            blas::axpbypczw(2. * poly_params.m_map, Ap[k - 1], 2. * poly_params.b_map, p[k - 1], -1., p[k - 2], p[k]);
            diracm(Ap[k], p[k], args...);
          }
        }
      }
    } else {
      errorQuda("Invalid basis %d", poly_params.basis);
    }
  }

} // namespace quda
