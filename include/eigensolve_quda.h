#pragma once

#include <quda.h>
#include <quda_internal.h>
#include <timer.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>
#include <eigen_helper.h>

namespace quda
{

  // Local enum for the LU axpy block type
  enum blockType { PENCIL, LOWER_TRI, UPPER_TRI };

  class EigenSolver
  {
    using range = std::pair<int, int>;

  protected:
    const DiracMatrix &mat;
    QudaEigParam *eig_param;
    TimeProfile &profile;

    // Problem parameters
    //------------------
    int n_ev;         /** Size of initial factorisation */
    int n_kr;         /** Size of Krylov space after extension */
    int n_conv;       /** Number of converged eigenvalues requested */
    int n_ev_deflate; /** Number of converged eigenvalues to use in deflation */
    double tol;       /** Tolerance on eigenvalues */
    bool reverse;     /** True if using polynomial acceleration */
    char spectrum[3]; /** Part of the spectrum to be computed */
    bool compute_svd; /** Compute the SVD if requested **/

    // Algorithm variables
    //--------------------
    bool converged;
    int restart_iter;
    int max_restarts;
    int max_ortho_attempts;
    int check_interval;
    int batched_rotate;
    int block_size;
    int ortho_block_size;
    int iter;
    int iter_converged;
    int iter_locked;
    int iter_keep;
    int num_converged;
    int num_locked;
    int num_keep;

    std::vector<double> residua;

    // Device side vector workspace
    std::vector<ColorSpinorField> r;
    std::vector<ColorSpinorField> d_vecs_tmp;

    QudaPrecision save_prec;

  public:
    /**
       @brief Constructor for base Eigensolver class
       @param eig_param MGParam struct that defines all meta data
       @param profile Timeprofile instance used to profile
    */
    EigenSolver(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile);

    /**
       Destructor for EigenSolver class.
    */
    virtual ~EigenSolver() = default;

    /**
       @return Whether the solver is only for Hermitian systems
     */
    virtual bool hermitian() = 0;

    /**
       @brief Computes the eigen decomposition for the operator passed to create.
       @param kSpace The converged eigenvectors
       @param evals The converged eigenvalues
     */
    virtual void operator()(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals) = 0;

    /**
       @brief Creates the eigensolver using the parameters given and the matrix.
       @param eig_param The eigensolver parameters
       @param mat The operator to solve
       @param profile Time Profile
     */
    static EigenSolver *create(QudaEigParam *eig_param, const DiracMatrix &mat, TimeProfile &profile);

    /**
       @brief Check for an initial guess. If none present, populate with rands, then
       orthonormalise
       @param[in] kSpace The Krylov space vectors
    */
    void prepareInitialGuess(std::vector<ColorSpinorField> &kSpace);

    /**
       @brief Check for a maximum of the Chebyshev operator
       @param[in] kSpace The Krylov space vectors
    */
    void checkChebyOpMax(std::vector<ColorSpinorField> &kSpace);

    /**
       @brief Extend the Krylov space
       @param[in] kSpace The Krylov space vectors
       @param[in] evals The eigenvalue array
    */
    void prepareKrylovSpace(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals);

    /**
       @brief Set the epsilon parameter
       @param[in] prec Precision of the solver instance
       @param[out] epsilon The deduced epsilon value
    */
    double setEpsilon(const QudaPrecision prec);

    /**
       @brief Query the eigensolver precision to stdout
       @param[in] prec Precision of the solver instance
    */
    void queryPrec(const QudaPrecision prec);

    /**
       @brief Dump the eigensolver parameters to stdout
    */
    void printEigensolverSetup();

    /**
       @brief Release memory, save eigenvectors, resize the Krylov space to its original dimension
       @param[in] kSpace The Krylov space vectors
       @param[in] evals The eigenvalue array
    */
    void cleanUpEigensolver(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals);

    /**
       @brief Promoted the specified matVec operation:
       M, Mdag, MMdag, MdagM to a Chebyshev polynomial
       @param[in] out Output spinor
       @param[in] in Input spinor
    */
    void chebyOp(cvector_ref<ColorSpinorField> &out, cvector_ref <const ColorSpinorField> &in);

    /**
       @brief Estimate the spectral radius of the operator for the max value of the
       Chebyshev polynomial
       @param[in] out Output spinor
       @param[in] in Input spinor
    */
    double estimateChebyOpMax(ColorSpinorField &out, ColorSpinorField &in);

    /**
       @brief Orthogonalise input vectors r against
       vector space v using hybrid modified Gram-Schmidt block-BLAS
       @param[in] v Vector space
       @param[in] r Vectors to be orthogonalised
       @param[in] i Ortho block size for Hybrid MGS (1 = modified, j = classical, 1 < i < j = hybrid)
       @param[in] j Use vectors v[0:j]
    */
    void blockOrthogonalizeHMGS(std::vector<ColorSpinorField> &v, std::vector<ColorSpinorField> &r, int i, int j);

    /**
      @brief Orthonormalise input vector space v using Hybrid Modified Gram-Schmidt blockBLAS
      @param[in] v Vector space
      @param[in] i Ortho block size for Hybrid MGS (1 = modified, j-1 = classical, 1 < i < j-1 = hybrid)
      @param[in] j Use vectors v[0:j-1]
   */
    void orthonormalizeHMGS(std::vector<ColorSpinorField> &v, int i, int j);

    /**
       @brief Check orthonormality of input vector space v
       @param[out] bool If all vectors are orthonormal to 1e-16 returns true,
       else false.
       @param[in] v Vector space
       @param[in] j Use vectors v[0:j-1]
    */
    bool orthoCheck(std::vector<ColorSpinorField> &v, int j);

    /**
       @brief Rotate the Krylov space
       @tparam T type that determines if we're using real- or complex-valued
       @param[in] kSpace the Krylov space
       @param[in] rot_array The rotation matrix
       @param[in] offset The position of the start of unused vectors in kSpace
       @param[in] dim The number of rows in the rotation array
       @param[in] keep The number of columns in the rotation array
       @param[in] locked The number of locked vectors in kSpace
       @param[in] profile Time profiler
    */
    template <typename T>
    void rotateVecs(std::vector<ColorSpinorField> &kSpace, const std::vector<T> &rot_array,
                    int offset, int dim, int keep, int locked, TimeProfile &profile);

    /**
       @brief Permute the vector space using the permutation matrix.
       @param[in/out] kSpace The current Krylov space
       @param[in] mat Eigen object storing the pivots
       @param[in] size The size of the (square) permutation matrix
    */
    void permuteVecs(std::vector<ColorSpinorField> &kSpace, MatrixXi &mat, int size);

    /**
       @brief Rotate part of kSpace
       @tparam T type that determines if we're using real- or complex-valued
       @param[in/out] kSpace The current Krylov space
       @param[in] array The real rotation matrix
       @param[in] rank row rank of array
       @param[in] is Start of i index
       @param[in] ie End of i index
       @param[in] js Start of j index
       @param[in] je End of j index
       @param[in] blockType Type of caxpy(_U/L) to perform
       @param[in] je End of j index
       @param[in] offset Position of extra vectors in kSpace
       @param[in] offset Position of extra vectors in kSpace
    */
    template <typename T, typename matrix_t>
    void blockRotate(std::vector<ColorSpinorField> &kSpace, matrix_t &array, int rank,
                     const range &i, const range &j, blockType b_type, int offset);

    /**
       @brief Copy temp part of kSpace, zero out for next use
       @param[in/out] kSpace The current Krylov space
       @param[in] js Start of j index
       @param[in] je End of j index
       @param[in] offset Position of extra vectors in kSpace
    */
    void blockReset(std::vector<ColorSpinorField> &kSpace, int js, int je, int offset);

    /**
       @brief Deflate a set of source vectors with a given eigenspace
       @param[in,out] sol The resulting deflated vector set
       @param[in] src The source vector set we are deflating
       @param[in] evecs The eigenvectors to use in deflation
       @param[in] evals The eigenvalues to use in deflation
       @param[in] accumulate Whether to preserve the sol vector content prior to accumulating
    */
    void deflate(cvector_ref<ColorSpinorField> &sol, cvector_ref<const ColorSpinorField> &src,
                 cvector_ref<const ColorSpinorField> &evecs, const std::vector<Complex> &evals,
                 bool accumulate = false) const;

    /**
       @brief Deflate a set of source vectors with a set of left and
       right singular vectors
       @param[in,out] sol The resulting deflated vector set
       @param[in] src The source vector set we are deflating
       @param[in] evecs The singular vectors to use in deflation
       @param[in] evals The singular values to use in deflation
       @param[in] accumulate Whether to preserve the sol vector content prior to accumulating
    */
    void deflateSVD(cvector_ref<ColorSpinorField> &sol, cvector_ref<const ColorSpinorField> &vec,
                    cvector_ref<const ColorSpinorField> &evecs, const std::vector<Complex> &evals,
                    bool accumulate = false) const;

    /**
       @brief Computes Left/Right SVD from pre computed Right/Left
       @param[in] evecs Computed eigenvectors of NormOp
       @param[in] evals Computed eigenvalues of NormOp
    */
    void computeSVD(std::vector<ColorSpinorField> &evecs, std::vector<Complex> &evals);

    /**
       @brief Compute eigenvalues and their residiua
       @param[in] mat Matrix operator
       @param[in] evecs The eigenvectors
       @param[in] evals The eigenvalues
       @param[in] size The number of eigenvalues to compute
    */
    void computeEvals(std::vector<ColorSpinorField> &evecs, std::vector<Complex> &evals,
                      int size);

    /**
       @brief Compute eigenvalues and their residiua.  This variant compute the number of converged eigenvalues.
       @param[in] mat Matrix operator
       @param[in] evecs The eigenvectors
       @param[in] evals The eigenvalues
    */
    void computeEvals(std::vector<ColorSpinorField> &evecs, std::vector<Complex> &evals)
    {
      computeEvals(evecs, evals, n_conv);
    }

    /**
       @brief Load and check eigenpairs from file
       @param[in] mat Matrix operator
       @param[in] eig_vecs The eigenvectors to save
       @param[in] file The filename to save
    */
    void loadFromFile(std::vector<ColorSpinorField> &eig_vecs, std::vector<Complex> &evals);

    /**
       @brief Sort array the first n elements of x according to spec_type, y comes along for the ride
       @param[in] spec_type The spectrum type (Largest/Smallest)(Modulus/Imaginary/Real)
       @param[in] n The number of elements to sort
       @param[in] x The array to sort
       @param[in] y An array whose elements will be permuted in tandem with x
    */
    void sortArrays(QudaEigSpectrumType spec_type, int n, std::vector<Complex> &x, std::vector<Complex> &y);

    /**
       @brief Sort array the first n elements of x according to spec_type, y comes along for the ride
       Overloaded version with real x
       @param[in] spec_type The spectrum type (Largest/Smallest)(Modulus/Imaginary/Real)
       @param[in] n The number of elements to sort
       @param[in] x The array to sort
       @param[in] y An array whose elements will be permuted in tandem with x
    */
    void sortArrays(QudaEigSpectrumType spec_type, int n, std::vector<double> &x, std::vector<Complex> &y);

    /**
       @brief Sort array the first n elements of x according to spec_type, y comes along for the ride
       Overloaded version with real y
       @param[in] spec_type The spectrum type (Largest/Smallest)(Modulus/Imaginary/Real)
       @param[in] n The number of elements to sort
       @param[in] x The array to sort
       @param[in] y An array whose elements will be permuted in tandem with x
    */
    void sortArrays(QudaEigSpectrumType spec_type, int n, std::vector<Complex> &x, std::vector<double> &y);

    /**
       @brief Sort array the first n elements of x according to spec_type, y comes along for the ride
       Overloaded version with real x and real y
       @param[in] spec_type The spectrum type (Largest/Smallest)(Modulus/Imaginary/Real) that
       determines the sorting condition
       @param[in] n The number of elements to sort
       @param[in] x The array to sort
       @param[in] y An array whose elements will be permuted in tandem with x
    */
    void sortArrays(QudaEigSpectrumType spec_type, int n, std::vector<double> &x, std::vector<double> &y);

    /**
       @brief Sort array the first n elements of x according to spec_type, y comes along for the ride
       Overloaded version with complex x and integer y
       @param[in] spec_type The spectrum type (Largest/Smallest)(Modulus/Imaginary/Real) that
       determines the sorting condition
       @param[in] n The number of elements to sort
       @param[in] x The array to sort
       @param[in] y An array whose elements will be permuted in tandem with x
    */
    void sortArrays(QudaEigSpectrumType spec_type, int n, std::vector<Complex> &x, std::vector<int> &y);
  };

  /**
     @brief Thick Restarted Lanczos Method.
  */
  class TRLM : public EigenSolver
  {

  public:
    /**
       @brief Constructor for Thick Restarted Eigensolver class
       @param eig_param The eigensolver parameters
       @param mat The operator to solve
       @param profile Time Profile
    */
    TRLM(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile);

    /**
       @return Whether the solver is only for Hermitian systems
    */
    virtual bool hermitian() { return true; } /** TRLM is only for Hermitian systems */

    // Variable size matrix
    std::vector<double> ritz_mat;

    // Tridiagonal/Arrow matrix, fixed size.
    std::vector<double> alpha;
    std::vector<double> beta;

    /**
       @brief Compute eigenpairs
       @param[in] kSpace Krylov vector space
       @param[in] evals Computed eigenvalues
    */
    void operator()(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals);

    /**
       @brief Lanczos step: extends the Krylov space.
       @param[in] v Vector space
       @param[in] j Index of vector being computed
    */
    void lanczosStep(std::vector<ColorSpinorField> &v, int j);

    /**
       @brief Reorder the Krylov space by eigenvalue
       @param[in] kSpace the Krylov space
    */
    void reorder(std::vector<ColorSpinorField> &kSpace);

    /**
       @brief Get the eigendecomposition from the arrow matrix
    */
    void eigensolveFromArrowMat();

    /**
       @brief Rotate the Ritz vectors usinng the arrow matrix eigendecomposition
       @param[in] nKspace current Krylov space
    */
    void computeKeptRitz(std::vector<ColorSpinorField> &kSpace);

  };

  /**
     @brief Block Thick Restarted Lanczos Method.
  */
  class BLKTRLM : public TRLM
  {
  public:
    /**
       @brief Constructor for Thick Restarted Eigensolver class
       @param eig_param The eigensolver parameters
       @param mat The operator to solve
       @param profile Time Profile
    */
    BLKTRLM(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile);

    virtual bool hermitian() { return true; } /** (BLOCK)TRLM is only for Hermitian systems */

    // Variable size matrix
    std::vector<Complex> block_ritz_mat;

    /** Block Tridiagonal/Arrow matrix, fixed size. */
    std::vector<Complex> block_alpha;
    std::vector<Complex> block_beta;

    /** Temp storage used in blockLanczosStep, fixed size. */
    std::vector<Complex> jth_block;

    /** Size of blocks of data in alpha/beta */
    int block_data_length;

    /**
       @brief Compute eigenpairs
       @param[in] kSpace Krylov vector space
       @param[in] evals Computed eigenvalues
    */
    void operator()(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals);

    /**
       @brief block lanczos step: extends the Krylov space in block step
       @param[in] v Vector space
       @param[in] j Index of block of vectors being computed
    */
    void blockLanczosStep(std::vector<ColorSpinorField> &v, int j);

    /**
       @brief Get the eigendecomposition from the current block arrow matrix
    */
    void eigensolveFromBlockArrowMat();

    /**
       @brief Accumulate the R products of QR into the block beta array
       @param[in] k The QR iteration
       @param[in] arrow_offset The current block position
    */
    void updateBlockBeta(int k, int arrow_offset);

    /**
       @brief Rotate the Ritz vectors usinng the arrow matrix eigendecomposition
       Uses a complex ritz matrix
       @param[in] nKspace current Krylov space
    */
    void computeBlockKeptRitz(std::vector<ColorSpinorField> &kSpace);
  };

  /**
     @brief Implicitly Restarted Arnoldi Method.
  */
  class IRAM : public EigenSolver
  {

  public:
    std::vector<std::vector<Complex>> upperHess;
    std::vector<std::vector<Complex>> Qmat;
    std::vector<std::vector<Complex>> Rmat;

    /**
       @brief Constructor for Thick Restarted Eigensolver class
       @param eig_param The eigensolver parameters
       @param mat The operator to solve
       @param profile Time Profile
    */
    IRAM(const DiracMatrix &mat, QudaEigParam *eig_param, TimeProfile &profile);

    /**
       @return Whether the solver is only for Hermitian systems
    */
    virtual bool hermitian() { return false; } /** IRAM is for any linear system */

    /**
       @brief Compute eigenpairs
       @param[in] kSpace Krylov vector space
       @param[in] evals Computed eigenvalues
    */
    void operator()(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals);

    /**
       @brief Arnoldi step: extends the Krylov space by one vector
       @param[in] v Vector space
       @param[in] r Residual vector
       @param[in] beta Norm of residual vector
       @param[in] j Index of vector being computed
    */
    void arnoldiStep(std::vector<ColorSpinorField> &v, std::vector<ColorSpinorField> &r, double &beta, int j);

    /**
       @brief Get the eigendecomposition from the upper Hessenberg matrix via QR
       @param[in] evals Complex eigenvalues
       @param[in] beta Norm of residual (used to compute errors on eigenvalues)
    */
    void eigensolveFromUpperHess(std::vector<Complex> &evals, const double beta);

    /**
       @brief Rotate the Krylov space
       @param[in] v Vector space
       @param[in] keep The number of vectors to keep after rotation
    */
    void rotateBasis(std::vector<ColorSpinorField> &v, int keep);

    /**
       @brief Apply shifts to the upper Hessenberg matrix via QR decomposition
       @param[in] evals The shifts to apply
       @param[in] num_shifts The number of shifts to apply
    */
    void qrShifts(const std::vector<Complex> evals, const int num_shifts);

    /**
       @brief Apply One step of the the QR algorithm
       @param[in] Q The Q matrix
       @param[in] R The R matrix
    */
    void qrIteration(std::vector<std::vector<Complex>> &Q, std::vector<std::vector<Complex>> &R);

    /**
       @brief Reorder the Krylov space and eigenvalues
       @param[in] kSpace The Krylov space
       @param[in] evals the eigenvalues
       @param[in] spec_type The spectrum type (Largest/Smallest)(Modulus/Imaginary/Real) that
       determines the sorting condition
    */
    void reorder(std::vector<ColorSpinorField> &kSpace, std::vector<Complex> &evals,
                 const QudaEigSpectrumType spec_type);
  };

  /**
     arpack_solve()

     @brief The QUDA interface function. One passes two allocated arrays to
     hold the the eigenmode data, the problem matrix, the arpack
     parameters defining what problem is to be solves, and a container
     for QUDA data structure types.
     @param[out] h_evecs Host fields where the e-vectors will be copied to
     @param[out] h_evals Where the e-values will be copied to
     @param[in] mat An explicit construction of the problem matrix.
     @param[in] param Parameter container defining the how the matrix
     is to be solved.
     @param[in] eig_param Parameter structure for all QUDA eigensolvers
     @param[in,out] profile TimeProfile instance used for profiling
  */
  void arpack_solve(std::vector<ColorSpinorField> &h_evecs, std::vector<Complex> &h_evals, const DiracMatrix &mat,
                    QudaEigParam *eig_param, TimeProfile &profile);

} // namespace quda
