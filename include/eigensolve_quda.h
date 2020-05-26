#pragma once

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>

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
    int nEv;          /** Size of initial factorisation */
    int nKr;          /** Size of Krylov space after extension */
    int nConv;        /** Number of converged eigenvalues requested */
    double tol;       /** Tolerance on eigenvalues */
    bool reverse;     /** True if using polynomial acceleration */
    char spectrum[3]; /** Part of the spectrum to be computed */

    // Algorithm variables
    //--------------------
    bool converged;
    int restart_iter;
    int max_restarts;
    int check_interval;
    int batched_rotate;
    int block_size;
    int iter;
    int iter_converged;
    int iter_locked;
    int iter_keep;
    int num_converged;
    int num_locked;
    int num_keep;

    double *residua;

    // Device side vector workspace
    std::vector<ColorSpinorField *> r;
    std::vector<ColorSpinorField *> d_vecs_tmp;

    ColorSpinorField *tmp1;
    ColorSpinorField *tmp2;

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
    virtual ~EigenSolver();

    /**
       @return Whether the solver is only for Hermitian systems
     */
    virtual bool hermitian() = 0;

    /**
       @brief Computes the eigen decomposition for the operator passed to create.
       @param kSpace The converged eigenvectors
       @param evals The converged eigenvalues
     */
    virtual void operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals) = 0;

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
    void prepareInitialGuess(std::vector<ColorSpinorField *> &kSpace);

    /**
       @brief Check for a maximum of the Chebyshev operator
       @param[in] mat The problem operator
       @param[in] kSpace The Krylov space vectors
    */
    void checkChebyOpMax(const DiracMatrix &mat, std::vector<ColorSpinorField *> &kSpace);

    /**
       @brief Extend the Krylov space
       @param[in] kSpace The Krylov space vectors
       @param[in] evals The eigenvalue array
    */
    void prepareKrylovSpace(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals);

    /**
       @brief Set the epsilon parameter
       @param[in] prec Precision of the solver instance
       @param[out] epsilon The deduced epsilon value
    */
    double setEpsilon(const QudaPrecision prec);

    /**
       @brief Dump the eigensolver parameters to stdout
    */
    void printEigensolverSetup();

    /**
       @brief Release memory, save eigenvectors, resize the Krylov space to its original dimension
       @param[in] kSpace The Krylov space vectors
       @param[in] evals The eigenvalue array
    */
    void cleanUpEigensolver(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals);

    /**
       @brief Applies the specified matVec operation:
       M, Mdag, MMdag, MdagM
       @param[in] mat Matrix operator
       @param[in] out Output spinor
       @param[in] in Input spinor
    */
    void matVec(const DiracMatrix &mat, ColorSpinorField &out, const ColorSpinorField &in);

    /**
       @brief Promoted the specified matVec operation:
       M, Mdag, MMdag, MdagM to a Chebyshev polynomial
       @param[in] mat Matrix operator
       @param[in] out Output spinor
       @param[in] in Input spinor
    */
    void chebyOp(const DiracMatrix &mat, ColorSpinorField &out, const ColorSpinorField &in);

    /**
       @brief Estimate the spectral radius of the operator for the max value of the
       Chebyshev polynomial
       @param[in] mat Matrix operator
       @param[in] out Output spinor
       @param[in] in Input spinor
    */
    double estimateChebyOpMax(const DiracMatrix &mat, ColorSpinorField &out, ColorSpinorField &in);

    /**
       @brief Orthogonalise input vectors r against
       vector space v using block-BLAS
       @param[in] v Vector space
       @param[in] r Vectors to be orthogonalised
       @param[in] j Use vectors v[0:j]
       @param[in] s array of
    */
    void blockOrthogonalize(std::vector<ColorSpinorField *> v, std::vector<ColorSpinorField *> &r, int j);

    /**
       @brief Orthonormalise input vector space v using Modified Gram-Schmidt
       @param[in] v Vector space
       @param[in] j Use vectors v[0:j-1]
    */
    void orthonormalizeMGS(std::vector<ColorSpinorField *> &v, int j);

    /**
       @brief Check orthonormality of input vector space v
       @param[out] bool If all vectors are orthonormal to 1e-16 returns true,
       else false.
       @param[in] v Vector space
       @param[in] j Use vectors v[0:j-1]
    */
    bool orthoCheck(std::vector<ColorSpinorField *> v, int j);

    /**
       @brief Permute the vector space using the permutation matrix.
       @param[in/out] kSpace The current Krylov space
       @param[in] mat Eigen object storing the pivots
       @param[in] size The size of the (square) permutation matrix
    */
    void permuteVecs(std::vector<ColorSpinorField *> &kSpace, int *mat, int size);

    /**
       @brief Rotate part of kSpace
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
    */
    void blockRotate(std::vector<ColorSpinorField *> &kSpace, double *array, int rank, const range &i, const range &j, blockType b_type);

    /**
       @brief Rotate part of kSpace
       @param[in/out] kSpace The current Krylov space
       @param[in] array The complex rotation matrix
       @param[in] rank row rank of array
       @param[in] is Start of i index
       @param[in] ie End of i index
       @param[in] js Start of j index
       @param[in] je End of j index
       @param[in] blockType Type of caxpy(_U/L) to perform
       @param[in] offset Position of extra vectors in kSpace
    */

    void blockRotateComplex(std::vector<ColorSpinorField *> &kSpace, Complex *array, int rank, const range &i,
                            const range &j, blockType b_type, int offset);

    /**
       @brief Copy temp part of kSpace, zero out for next use
       @param[in/out] kSpace The current Krylov space
       @param[in] js Start of j index
       @param[in] je End of j index
       @param[in] offset Position of extra vectors in kSpace
    */
    void blockReset(std::vector<ColorSpinorField *> &kSpace, int js, int je, int offset);

    /**
       @brief Deflate a set of source vectors with a given eigenspace
       @param[in] sol The resulting deflated vector set
       @param[in] src The source vector set we are deflating
       @param[in] evecs The eigenvectors to use in deflation
       @param[in] evals The eigenvalues to use in deflation
       @param[in] accumulate Whether to preserve the sol vector content prior to accumulating
    */
    void deflate(std::vector<ColorSpinorField *> &sol, const std::vector<ColorSpinorField *> &src,
                 const std::vector<ColorSpinorField *> &evecs, const std::vector<Complex> &evals,
                 bool accumulate = false) const;

    /**
       @brief Deflate a given source vector with a given eigenspace.
       This is a wrapper variant for a single source vector.
       @param[in] sol The resulting deflated vector
       @param[in] src The source vector we are deflating
       @param[in] evecs The eigenvectors to use in deflation
       @param[in] evals The eigenvalues to use in deflation
       @param[in] accumulate Whether to preserve the sol vector content prior to accumulating
    */
    void deflate(ColorSpinorField &sol, const ColorSpinorField &src, const std::vector<ColorSpinorField *> &evecs,
                 const std::vector<Complex> &evals, bool accumulate = false)
    {
      // FIXME add support for mixed-precison dot product to avoid this copy
      if (src.Precision() != evecs[0]->Precision() && !tmp1) {
        ColorSpinorParam param(*evecs[0]);
        tmp1 = ColorSpinorField::Create(param);
      }
      ColorSpinorField *src_tmp = src.Precision() != evecs[0]->Precision() ? tmp1 : const_cast<ColorSpinorField *>(&src);
      blas::copy(*src_tmp, src); // no-op if these alias
      std::vector<ColorSpinorField *> src_ {src_tmp};
      std::vector<ColorSpinorField *> sol_ {&sol};
      deflate(sol_, src_, evecs, evals, accumulate);
    }

    /**
       @brief Deflate a set of source vectors with a set of left and
       right singular vectors
       @param[in] sol The resulting deflated vector set
       @param[in] src The source vector set we are deflating
       @param[in] evecs The singular vectors to use in deflation
       @param[in] evals The singular values to use in deflation
       @param[in] accumulate Whether to preserve the sol vector content prior to accumulating
    */
    void deflateSVD(std::vector<ColorSpinorField *> &sol, const std::vector<ColorSpinorField *> &vec,
                    const std::vector<ColorSpinorField *> &evecs, const std::vector<Complex> &evals,
                    bool accumulate = false) const;

    /**
       @brief Deflate a a given source vector with a given with a set of left and
       right singular vectors  This is a wrapper variant for a single source vector.
       @param[in] sol The resulting deflated vector set
       @param[in] src The source vector set we are deflating
       @param[in] evecs The singular vectors to use in deflation
       @param[in] evals The singular values to use in deflation
       @param[in] accumulate Whether to preserve the sol vector content prior to accumulating
    */
    void deflateSVD(ColorSpinorField &sol, const ColorSpinorField &src, const std::vector<ColorSpinorField *> &evecs,
                    const std::vector<Complex> &evals, bool accumulate = false)
    {
      // FIXME add support for mixed-precison dot product to avoid this copy
      if (src.Precision() != evecs[0]->Precision() && !tmp1) {
        ColorSpinorParam param(*evecs[0]);
        tmp1 = ColorSpinorField::Create(param);
      }
      ColorSpinorField *src_tmp = src.Precision() != evecs[0]->Precision() ? tmp1 : const_cast<ColorSpinorField *>(&src);
      blas::copy(*src_tmp, src); // no-op if these alias
      std::vector<ColorSpinorField *> src_ {src_tmp};
      std::vector<ColorSpinorField *> sol_ {&sol};
      deflateSVD(sol_, src_, evecs, evals, accumulate);
    }

    /**
       @brief Computes Left/Right SVD from pre computed Right/Left
       @param[in] mat Matrix operator
       @param[in] evecs Computed eigenvectors of NormOp
       @param[in] evals Computed eigenvalues of NormOp
    */
    void computeSVD(const DiracMatrix &mat, std::vector<ColorSpinorField *> &evecs, std::vector<Complex> &evals);

    /**
       @brief Compute eigenvalues and their residiua
       @param[in] mat Matrix operator
       @param[in] evecs The eigenvectors
       @param[in] evals The eigenvalues
       @param[in] size The number of eigenvalues to compute
    */
    void computeEvals(const DiracMatrix &mat, std::vector<ColorSpinorField *> &evecs, std::vector<Complex> &evals,
                      int size);

    /**
       @brief Compute eigenvalues and their residiua.  This variant compute the number of converged eigenvalues.
       @param[in] mat Matrix operator
       @param[in] evecs The eigenvectors
       @param[in] evals The eigenvalues
    */
    void computeEvals(const DiracMatrix &mat, std::vector<ColorSpinorField *> &evecs, std::vector<Complex> &evals)
    {
      computeEvals(mat, evecs, evals, nConv);
    }

    /**
       @brief Load and check eigenpairs from file
       @param[in] mat Matrix operator
       @param[in] eig_vecs The eigenvectors to save
       @param[in] file The filename to save
    */
    void loadFromFile(const DiracMatrix &mat, std::vector<ColorSpinorField *> &eig_vecs, std::vector<Complex> &evals);
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
       @brief Destructor for Thick Restarted Eigensolver class
    */
    virtual ~TRLM();

    virtual bool hermitian() { return true; } /** TRLM is only for Hermitian systems */

    // Variable size matrix
    std::vector<double> ritz_mat;

    // Tridiagonal/Arrow matrix, fixed size.
    double *alpha;
    double *beta;

    /**
       @brief Compute eigenpairs
       @param[in] kSpace Krylov vector space
       @param[in] evals Computed eigenvalues
    */
    void operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals);

    /**
       @brief Lanczos step: extends the Kylov space.
       @param[in] v Vector space
       @param[in] j Index of vector being computed
    */
    void lanczosStep(std::vector<ColorSpinorField *> v, int j);

    /**
       @brief Reorder the Krylov space by eigenvalue
       @param[in] kSpace the Krylov space
    */
    void reorder(std::vector<ColorSpinorField *> &kSpace);

    /**
       @brief Get the eigendecomposition from the arrow matrix
    */
    void eigensolveFromArrowMat();

    /**
       @brief Rotate the Ritz vectors usinng the arrow matrix eigendecomposition
       @param[in] nKspace current Krylov space
    */
    void computeKeptRitz(std::vector<ColorSpinorField *> &kSpace);

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

    /**
       @brief Destructor for Thick Restarted Eigensolver class
    */
    virtual ~BLKTRLM();

    virtual bool hermitian() { return true; } /** (BLOCK)TRLM is only for Hermitian systems */

    // Variable size matrix
    std::vector<Complex> block_ritz_mat;

    /** Block Tridiagonal/Arrow matrix, fixed size. */
    Complex *block_alpha;
    Complex *block_beta;

    /** Temp storage used in blockLanczosStep, fixed size. */
    Complex *jth_block;

    /** Size of blocks of data in alpha/beta */
    int block_data_length;

    /**
       @brief Compute eigenpairs
       @param[in] kSpace Krylov vector space
       @param[in] evals Computed eigenvalues
    */
    void operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals);

    /**
       @brief block lanczos step: extends the Kylov space in block step
       @param[in] v Vector space
       @param[in] j Index of block of vectors being computed
    */
    void blockLanczosStep(std::vector<ColorSpinorField *> v, int j);

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
    void computeBlockKeptRitz(std::vector<ColorSpinorField *> &kSpace);
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
  void arpack_solve(std::vector<ColorSpinorField *> &h_evecs, std::vector<Complex> &h_evals, const DiracMatrix &mat,
                    QudaEigParam *eig_param, TimeProfile &profile);

} // namespace quda
