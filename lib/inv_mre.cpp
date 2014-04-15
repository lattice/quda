#include <invert_quda.h>
#include <blas_quda.h>

namespace quda {

  MinResExt::MinResExt(DiracMatrix &mat, TimeProfile &profile) 
    : mat(mat), profile(profile){

  }

  MinResExt::~MinResExt() {

  }

  void MinResExt::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b, 
			     cudaColorSpinorField **p, cudaColorSpinorField **q, int N) {

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
    
    // if no guess is required, then set initial guess = 0
    if (N == 0) {
      zeroCuda(x);
      return;
    }

    double b2 = norm2(b);

    // Array to hold the matrix elements
    Complex **G = new Complex*[N];
    for (int i=0; i<N; i++) G[i] = new Complex[N];
    
    // Solution and source vectors
    Complex *alpha = new Complex[N];
    Complex *beta = new Complex[N];

    // Orthonormalise the vector basis
    for (int i=0; i<N; i++) {
      double p2 = norm2(*p[i]);
      axCuda(1 / sqrt(p2), *p[i]);
      for (int j=i+1; j<N; j++) {
	Complex xp = cDotProductCuda(*p[i], *p[j]);
	caxpyCuda(-xp, *p[i], *p[j]);
      }
    }

    // Perform sparse matrix multiplication and construct rhs
    for (int i=0; i<N; i++) {
      beta[i] = cDotProductCuda(*p[i], b);
      mat(*q[i], *p[i]);
      G[i][i] = reDotProductCuda(*q[i], *p[i]);
    }

    // Construct the matrix
    for (int j=0; j<N; j++) {
      for (int k=j+1; k<N; k++) {
	G[j][k] = cDotProductCuda(*p[j], *q[k]);
	G[k][j] = conj(G[j][k]);
      }
    }

    // Gauss-Jordan elimination with partial pivoting
    for (int i=0; i<N; i++) {

      // Perform partial pivoting
      int k = i;
      for (int j=i+1; j<N; j++) if (abs(G[j][j]) > abs(G[k][k])) k = j;
      if (k != i) {
	std::swap<Complex>(beta[k], beta[i]);
	for (int j=0; j<N; j++) std::swap<Complex>(G[k][j], G[i][j]);
      }

      // Convert matrix to upper triangular form
      for (int j=i+1; j<N; j++) {
	Complex xp = G[j][i]/G[i][i];
	beta[j] -= xp * beta[i];
	for (int k=0; k<N; k++) G[j][k] -= xp * G[i][k];
      }
    }

    // Use Gaussian Elimination to solve equations and calculate initial guess
    zeroCuda(x);
    for (int i=N-1; i>=0; i--) {
      alpha[i] = 0.0;
      for (int j=i+1; j<N; j++) alpha[i] += G[i][j] * alpha[j];
      alpha[i] = (beta[i]-alpha[i])/G[i][i];
      caxpyCuda(alpha[i], *p[i], x);
      caxpyCuda(-alpha[i], *q[i], b);
      //printfQuda("%d %e %e\n", i, real(alpha[i]), imag(alpha[i]));
    }

    double rsd = sqrt(norm2(b) / b2 );
    printfQuda("MinResExt: N = %d, |res| / |src| = %e\n", N, rsd);
    
    for (int j=0; j<N; j++) delete [] G[j];

    delete [] G;
    delete [] alpha;
    delete [] beta;
  }

} // namespace quda
