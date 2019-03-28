#pragma once

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>

namespace quda {

  class EigenSolver {
    
  protected:
    QudaEigParam *eig_param;
    TimeProfile profile;
    
  public:
    EigenSolver(QudaEigParam *eig_param, TimeProfile &profile);
    virtual ~EigenSolver();
    
    virtual void operator()(std::vector<ColorSpinorField*> &kSpace,
			    std::vector<Complex> &evals) = 0;
    
    static EigenSolver* create(QudaEigParam *eig_param, const Dirac &mat, TimeProfile &profile);
    
    /**
       @brief Applies the specified matVec operation:
       M, Mdag, MMdag, MdagM
       @param[in] mat Matrix operator
       @param[in] out Output spinor
       @param[in] in Input spinor
    */
    
    void matVec(const Dirac &mat,
		ColorSpinorField &out,
		const ColorSpinorField &in);
    
    /**
       @brief Promoted the specified matVec operation:
       M, Mdag, MMdag, MdagM to a Chebyshev polynomial
       @param[in] mat Matrix operator
       @param[in] out Output spinor
       @param[in] in Input spinor
    */
    void chebyOp(const Dirac &mat,
		 ColorSpinorField &out,
		 const ColorSpinorField &in);

    /**
       @brief Orthogonalise input vector r against
       vector space v
       @param[in] v Vector space
       @param[in] r Vector to be orthogonalised
       @param[in] j Number of vectors in v to orthogonalise against
    */
    void orthogonalise(std::vector<ColorSpinorField*> v,
		       std::vector<ColorSpinorField*> r,
		       int j);

    /**
       @brief Orthogonalise input vector r against
       vector space v using block-BLAS
       @param[in] v Vector space
       @param[in] r Vector to be orthogonalised
       @param[in] j Number of vectors in v to orthogonalise against
    */
    void blockOrthogonalise(std::vector<ColorSpinorField*> v,
			    std::vector<ColorSpinorField*> r,
			    int j);

    /**
       @brief Deflate vector with Eigenvectors
       @param[in] vec_defl The deflated vector
       @param[in] vec The input vector
       @param[in] evecs The eigenvectors to use in deflation
       @param[in] evals The eigenvalues to use in deflation
    */    
    void deflate(std::vector<ColorSpinorField*> vec_defl,
		 std::vector<ColorSpinorField*> vec,
		 std::vector<ColorSpinorField*> evecs,
		 std::vector<Complex> evals);
    
    /**
       @brief Compute eigenvalues and their residiua
       @param[in] mat Matrix operator
       @param[in] evals The eigenvalues
       @param[out] residua The residua of the eigenpair
       @param[in] evecs The eigenvectors
       @param[in] tmp A workspace vector
    */        
    void computeEvals(const Dirac &mat,
		      std::vector<Complex> &evals,
		      double *residua,
		      std::vector<ColorSpinorField*> evecs,
		      std::vector<ColorSpinorField*> tmp);

    /**
       @brief Rotate eigenvectors by dense matrix
       @param[in] Qmat The dense rotation matrix
       @param[in] kSpace the vectors to be rotated
       @param[in] tmp Workspace vectors
    */            
    void basisRotateQ(Complex *Qmat,
		      std::vector<ColorSpinorField*> &kSpace,
		      std::vector<ColorSpinorField*> &tmp);
    
  };
  
  
  
  /**
     @brief Implicily Restarted Lanczos Method.
  */
  class IRLM : public EigenSolver {

  private:
    const Dirac &mat;
    
  public:
    IRLM(QudaEigParam *eig_param, const Dirac &mat, TimeProfile &profile);
    virtual ~IRLM();

    /**
       @brief Compute eigenpairs
       @param[in] kSpace Krylov vector space
       @param[in] evals Computed eigenvalues
       
    */
    void operator()(std::vector<ColorSpinorField*> &kSpace,
		    std::vector<Complex> &evals);
    
    /**
       @brief Lanczos step: extends the Kylov space.
       @param[in] v Vector space
       @param[in] r Current vector to add
       @param[in] evecs List of eigenvectors
       @param[in] locked List of converged eigenvectors
       @param[in] alpha Diagonal of tridiagonal
       @param[in] beta Subdiagonal of tridiagonal
       @param[in] j Index of last vector added       
    */
    void lanczosStep(std::vector<ColorSpinorField*> v,
		     std::vector<ColorSpinorField*> r,
		     std::vector<ColorSpinorField*> evecs,
		     bool *locked,
		     double *alpha, double *beta, int j);

    /**
       @brief Rotate eigenvectors by dense matrix dervied
       from the tridiagonal matrix in Lanczos
       @param[in] alpha Leading diagonal part of Lanczos tridiagonal matrix
       @param[in] beta Sub-leading diagonal part of Lanczos tridiagonal matrix
       @param[in] nEv The number of vectors to be rotated
       @param[in] kSpace the vectors to be rotated
       @param[in] tmp Workspace vectors
    */            
    void basisRotateTriDiag(double *alpha, double *beta, int nEv,
			    std::vector<ColorSpinorField*> &kSpace,
			    std::vector<ColorSpinorField*> &tmp);
        
    /**
       @brief Computes QR factorisation from the Lanczos tridiagonal matrix
       @param[in] Qmat Complex array for Q part of QR
       @param[in] alpha Leading diagonal part of Lanczos tridiagonal matrix
       @param[in] beta Sub-leading diagonal part of Lanczos tridiagonal matrix
       @param[in] inverse Project out the largest/smallest eigenvalues
    */    
    void computeQRfromTridiag(Complex *Qmat, double *alpha, double *beta, bool inverse);
    
    /**
       @brief Computes Left/Right SVD from pre computed Right/Left 
       @param[in] v Vector space
       @param[in] r Current vector to add
       @param[in] kSpace
       @param[in] evecs Computed eigenvectors of NormOp
       @param[in] evals Computed eigenvalues of NormOp
       @param[in] inverse Inverse sort if using PolyAcc       
    */
    void computeSVD(std::vector<ColorSpinorField*> &kSpace,
		    std::vector<ColorSpinorField*> &evecs,
		    std::vector<Complex> &evals,
		    bool inverse);
    
  };
  
  
  void arpack_solve(void *h_evecs, void *h_evals,
		    const Dirac &mat,
		    QudaEigParam *eig_param,
		    ColorSpinorParam *cpuParam);
  
}
