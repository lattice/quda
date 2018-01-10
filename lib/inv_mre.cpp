#include <invert_quda.h>
#include <blas_quda.h>

#include <Eigen/Dense>

namespace quda {

  MinResExt::MinResExt(DiracMatrix &mat, bool orthogonal, bool apply_mat, bool hermitian, TimeProfile &profile)
    : mat(mat), orthogonal(orthogonal), apply_mat(apply_mat), hermitian(hermitian), profile(profile){

  }

  MinResExt::~MinResExt() {

  }

#if 1

  /**
     @brief Solve the equation A p_k psi_k = b by minimizing the
     residual and using Eigen's SVD algorithm for numerical stability

     @param psi[out] Array of coefficients
     @param p[in] Search direction vectors
     @param q[in] Search direction vectors with the operator applied
  */
  template<bool hermitian>
  void solve(Complex *psi_, std::vector<ColorSpinorField*> &p, std::vector<ColorSpinorField*> &q, ColorSpinorField &b) {

    using namespace Eigen;
    typedef Matrix<Complex, Dynamic, Dynamic> matrix;
    typedef Matrix<Complex, Dynamic, 1> vector;

    vector phi(p.size()), psi(p.size());
    matrix A(p.size(),p.size());

    for (unsigned int i=0; i<p.size(); i++) phi(i) = blas::cDotProduct(*p[i], b);

    // Construct the matrix
    //MW us block blas here 
    for (unsigned int j=0; j<p.size(); j++) {
      if(hermitian){
        A(j,j) = blas::cDotProduct(*q[j], *p[j]);
        for (unsigned int k=j+1; k<p.size(); k++) {
          A(j,k) = blas::cDotProduct(*p[j], *q[k]);
          A(k,j) = conj(A(j,k));
        }
      } else {
        for (unsigned int k=0; k<p.size(); k++) {
          A(j,k) = blas::cDotProduct(*p[j], *q[k]);
        } 
      }
    }
    JacobiSVD<matrix> svd(A, ComputeThinU | ComputeThinV);
    psi = svd.solve(phi);

    for (unsigned int i=0; i<p.size(); i++) psi_[i] = psi(i);
  }

#else //old stuff

  /**
     @brief Solve the equation A p_k psi_k = b by minimizing the
     residual and using Gaussian elimination

     @param psi[out] Array of coefficients
     @param p[in] Search direction vectors
     @param q[in] Search direction vectors with the operator applied
  */
  void solve(Complex *psi, std::vector<ColorSpinorField*> &p, std::vector<ColorSpinorField*> &q, ColorSpinorField &b) {

    const int N = p.size();

    // Array to hold the matrix elements
    Complex **A = new Complex*[N];
    for (int i=0; i<N; i++) A[i] = new Complex[N];

    // Solution and source vectors
    Complex *phi = new Complex[N];

    // construct right hand side
    for (unsigned int i=0; i<p.size(); i++) phi[i] = blas::cDotProduct(*p[i], b);

    // Construct the matrix
    for (unsigned int j=0; j<p.size(); j++) {
      A[j][j] = blas::cDotProduct(*q[j], *p[j]);
      for (unsigned int k=j+1; k<p.size(); k++) {
	A[j][k] = blas::cDotProduct(*p[j], *q[k]);
	A[k][j] = conj(A[j][k]);
      }
    }

    // Gauss-Jordan elimination with partial pivoting
    for (int i=0; i<N; i++) {

      // Perform partial pivoting
      int k = i;
      for (int j=i+1; j<N; j++) if (abs(A[j][j]) > abs(A[k][k])) k = j;
      if (k != i) {
	std::swap<Complex>(phi[k], phi[i]);
	for (int j=0; j<N; j++) std::swap<Complex>(A[k][j], A[i][j]);
      }

      // Convert matrix to upper triangular form
      for (int j=i+1; j<N; j++) {
	Complex xp = A[j][i]/A[i][i];
	phi[j] -= xp * phi[i];
	for (int k=0; k<N; k++) A[j][k] -= xp * A[i][k];
      }
    }

    // Use Gaussian Elimination to solve equations and calculate initial guess
    for (int i=N-1; i>=0; i--) {
      psi[i] = 0.0;
      for (int j=i+1; j<N; j++) psi[i] += A[i][j] * psi[j];
      psi[i] = (phi[i]-psi[i])/A[i][i];
    }

    for (int j=0; j<N; j++) delete [] A[j];
    delete [] A;

    delete [] phi;
  }

#endif // old stuff


  /*
    We want to find the best initial guess of the solution of
    A x = b, and we have N previous solutions x_i.
    The method goes something like this:
    
    1. Orthonormalise the p_i and q_i
    2. Form the matrix G_ij = x_i^dagger A x_j
    3. Form the vector B_i = x_i^dagger b
    4. solve A_ij a_j  = B_i
    5. x = a_i p_i
  */
  void MinResExt::operator()(ColorSpinorField &x, ColorSpinorField &b, 
			     std::vector<ColorSpinorField*> p, std::vector<ColorSpinorField*> q) {

    profile.TPSTART(QUDA_PROFILE_INIT);

    const int N = p.size();

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Constructing minimum residual extrapolation with basis size %d\n", N);

    // if no guess is required, then set initial guess = 0
    if (N == 0) {
      blas::zero(x);
      return;
    }

    // Solution coefficient vectors
    Complex *alpha = new Complex[N];
    Complex *minus_alpha = new Complex[N];

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double b2 = blas::norm2(b);

    // Orthonormalise the vector basis
    if (orthogonal) {
      for (int i=0; i<N; i++) {
	double p2 = blas::norm2(*p[i]);
	blas::ax(1 / sqrt(p2), *p[i]);
	if (!apply_mat) blas::ax(1 / sqrt(p2), *q[i]);
	for (int j=i+1; j<N; j++) {
	  Complex xp = blas::cDotProduct(*p[i], *p[j]);
	  blas::caxpy(-xp, *p[i], *p[j]);
	  // if not applying the matrix below then orthongonalize q
	  if (!apply_mat) blas::caxpy(-xp, *q[i], *q[j]);
	}
      }
    }

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);


    // if operator hasn't already been applied then apply
    if (apply_mat) for (int i=0; i<N; i++) mat(*q[i], *p[i]);

    if (hermitian) {
      solve<true>(alpha, p, q, b);
    } else {
      solve<false>(alpha, p, q, b);
    }

    for (int i=0; i<N; i++) minus_alpha[i] = -alpha[i];

    blas::zero(x);
    std::vector<ColorSpinorField*> X, B;
    X.push_back(&x); B.push_back(&b);
    blas::caxpy(alpha, p, X);
    blas::caxpy(minus_alpha, q, B);

    double rsd = sqrt(blas::norm2(b) / b2 );
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("MinResExt: N = %d, |res| / |src| = %e\n", N, rsd);


    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    delete [] minus_alpha;
    delete [] alpha;

    profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  // Wrapper for the above
  void MinResExt::operator()(ColorSpinorField &x, ColorSpinorField &b, std::vector<std::pair<ColorSpinorField*,ColorSpinorField*> > basis) {
    std::vector<ColorSpinorField*> p(basis.size()), q(basis.size());
    for (unsigned int i=0; i<basis.size(); i++) { p[i] = basis[i].first; q[i] = basis[i].second; }
    (*this)(x, b, p, q);
  }



} // namespace quda
