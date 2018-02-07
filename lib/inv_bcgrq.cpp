
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>
#include <limits>
#include <cmath>

#include <iostream>
#include <Eigen/Dense>
#include <cuda_profiler_api.h>

namespace quda{




// define this to use multi-functions, otherwise it'll
// do loops over dot products.
// this is more here for development convenience.
#ifdef BLOCKSOLVER
  #define BLOCKSOLVER_MULTIFUNCTIONS
  #define BLOCKSOLVE_DSLASH5D
#endif
//#define BLOCKSOLVER_VERBOSE

// Run algorithm with Q in high precision.
#define BLOCKSOLVER_PRECISE_Q

// Mathias' testing area for Pollock-Ribiere or however it's spelled.
//#define BLOCKSOLVER_ALTERNATIVE_BETA

// Explicitly reorthogonalize Q^\dagger P on reliable update.
//#define BLOCKSOLVER_EXPLICIT_QP_ORTHO
// Explicitly make pAp Hermitian every time it is computed.
//#define BLOCKSOLVER_EXPLICIT_PAP_HERMITIAN

// If defined, trigger a reliable updated whenever _any_ residual
// becomes small enough. Otherwise, trigger a reliable update
// when _all_ residuals become small enough (which is consistent with
// the algorithm stopping condition). Ultimately, this is using a
// min function versus a max function, so it's not a hard swap.
// #define BLOCKSOLVER_RELIABLE_POLICY_MIN

#ifdef BLOCKSOLVER_NVTX
#include "nvToolsExt.h"
static const uint32_t cg_nvtx_colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
static constexpr int cg_nvtx_num_colors = sizeof(cg_nvtx_colors)/sizeof(uint32_t);
#define PUSH_RANGE(name,cid) { \
    static int color_id = cid; \
    color_id = color_id%cg_nvtx_num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = cg_nvtx_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    eventAttrib.category = cid;\
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif



#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  using Eigen::Matrix;
  using Eigen::Map;
  using Eigen::RowMajor;
  using Eigen::Dynamic; 
#else 
  using Eigen::MatrixXcd;
#endif

// Matrix printing functions

template<typename Matrix>
inline void printmat(const char* label, const Matrix& mat)
{
#ifdef BLOCKSOLVER_VERBOSE
  printfQuda("\n%s\n", label);
  std::cout << mat;
  printfQuda("\n");
#endif
}

/**
     The following code is based on Kate's worker class in Multi-CG.
     
     This worker class is used to perform the update of X_sloppy,
     UNLESS a reliable update is triggered. X_sloppy is updated
     via a block caxpy: X_sloppy += P \alpha.
     We can accomodate multiple comms-compute overlaps
     by partitioning the block caxpy w/respect to P, because
     this doesn't require any memory shuffling of the dense, square
     matrix alpha. This results in improved strong scaling for
     blockCG. 

     See paragraphs 2 and 3 in the comments on the Worker class in
     Multi-CG for more remarks. 
*/
class BlockCGUpdate : public Worker {

    std::shared_ptr<ColorSpinorField> & x_sloppyp;
    std::shared_ptr<ColorSpinorField> & p_oldp; // double pointer because p participates in pointer swapping
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    Complex* alpha;
  #else
    MatrixXcd &alpha;
#endif

    /**
       How many RHS we're solving.
    */
    int n_rhs;

    /**
       How much to partition the shifted update. For now, we assume
       we always need to partition into two pieces (since BiCGstab-L
       should only be getting even/odd preconditioned operators).
    */
    int n_update; 
    
#ifndef BLOCKSOLVE_DSLASH5D
    /** 
       What X we're updating; only relevant when we're looping over rhs.
    */
    int curr_update;
#endif

  public:
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    BlockCGUpdate(std::shared_ptr<ColorSpinorField>& x_sloppyp, std::shared_ptr<ColorSpinorField> &  p_oldp, Complex* alpha) :
#else
    BlockCGUpdate(std::shared_ptr<ColorSpinorField>& x_sloppyp, std::shared_ptr<ColorSpinorField> &  p_oldp, MatrixXcd& alpha) :
#endif
      x_sloppyp(x_sloppyp), p_oldp(p_oldp), alpha(alpha), n_rhs(p_oldp->Components().size()),
      n_update( x_sloppyp->Nspin()==4 ? 4 : 2 )
#ifdef BLOCKSOLVE_DSLASH5D
    { ; }
#else
    { curr_update = 0; }
#endif
    
    ~BlockCGUpdate() { ; }
    

    // note that we can't set the stream parameter here so it is
    // ignored.  This is more of a future design direction to consider
    void apply(const cudaStream_t &stream) {    
      static int count = 0;

#ifdef BLOCKSOLVE_DSLASH5D
      // How many to update per apply.
      const int update_per_apply = n_rhs/n_update;

      // If the number of updates doesn't evenly divide into n_rhs, there's leftover.
      const int update_per_apply_on_last = n_rhs - n_update*update_per_apply;

      // Only update if there are things to apply.
      // Update 1 through n_count-1, as well as n_count if update_per_apply_blah_blah = 0.
      PUSH_RANGE("BLAS",2)
      if ((count != n_update-1 && update_per_apply != 0) || update_per_apply_on_last == 0)
      {
        std::vector<ColorSpinorField*> curr_p(p_oldp->Components().begin() + count*update_per_apply, p_oldp->Components().begin() + (count+1)*update_per_apply);

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        blas::caxpy(&alpha[count*update_per_apply*n_rhs], curr_p, x_sloppyp->Components());
#else
        for (int i = 0; i < update_per_apply; i++)
          for (int j = 0; j < n_rhs; j++)
            blas::caxpy(alpha(i+count*update_per_apply, j), *(curr_p[i]), x_sloppyp->Component(j));
#endif
      }
      else if (count == n_update-1) // we're updating the leftover.
      {
        std::vector<ColorSpinorField*> curr_p(p_oldp->Components().begin() + count*update_per_apply, p_oldp->Components().end());
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        blas::caxpy(&alpha[count*update_per_apply*n_rhs], curr_p, x_sloppyp->Components());
#else
        for (int i = 0; i < update_per_apply_on_last; i++)
          for (int j = 0; j < n_rhs; j++)
            blas::caxpy(alpha(i+count*update_per_apply,j), *(curr_p[i]), x_sloppyp->Component(j));
#endif
      }
      POP_RANGE
#else // BLOCKSOLVE_DSLASH5D
      if (count == 0)
      {
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        std::vector<ColorSpinorField*> curr_p;
        curr_p.push_back(&(pp)->Component(curr_update));
        blas::caxpy(&alpha[curr_update*n_rhs], curr_p, x_sloppyp->Components());
#else
        for (int j = 0; j < n_rhs; j++)
            blas::caxpy(alpha(curr_update,j), p_oldp->Component(curr_update), x_sloppyp->Component(j));
#endif
        if (++curr_update == n_rhs) curr_update = 0;
      }
#endif // BLOCKSOLVE_DSLASH5D
      
      if (++count == n_update) count = 0;
      
    }
    
  };

  // this is the Worker pointer that the dslash uses to launch the shifted updates
  namespace dslash {
    extern Worker* aux_worker;
  } 

// Code to check for reliable updates, copied from inv_bicgstab_quda.cpp
// Technically, there are ways to check both 'x' and 'r' for reliable updates...
// the current status in blockCG is to just look for reliable updates in 'r'.
int BlockCG::block_reliable(double &rNorm, double &maxrx, double &maxrr, const double &r2, const double &delta) {
  // reliable updates
  // rNorm = sqrt(r2);
  if (rNorm > maxrx) maxrx = rNorm;
  if (rNorm > maxrr) maxrr = rNorm;
  //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
  //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0
  int updateR = (rNorm < delta*maxrr) ? 1 : 0;
  
  //printf("reliable %d %e %e %e %e\n", updateR, rNorm, maxrx, maxrr, r2);

  return updateR;
}




template <int nsrc>
void BlockCG::solve_n(ColorSpinorField& x, ColorSpinorField& b) {

//  if (Location(x, b) != QUDA_CUDA_FIELD_LOCATION) errorQuda("Not supported");

  profile.TPSTART(QUDA_PROFILE_INIT);

  // Check to see that we're not trying to invert on a zero-field source
  //MW: it might be useful to check what to do here.
  double b2[QUDA_MAX_BLOCK_SRC];
  double b2avg=0;
  for(int i=0; i<nsrc; i++){
    b2[i]=blas::norm2(b.Component(i));
    b2avg += b2[i];
    if(b2[i] == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      errorQuda("Warning: inverting on zero-field source - undefined for block solver\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }
  }

  b2avg = b2avg / nsrc;

  ColorSpinorParam csParam(x);

  csParam.is_composite  = true;
  csParam.composite_dim = nsrc;
  csParam.nDim = 5;
  csParam.x[4] = 1;

  if (!init) {
    csParam.create = QUDA_COPY_FIELD_CREATE;
    rp = ColorSpinorField::CreateSmartPtr(b, csParam);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    yp = ColorSpinorField::CreateSmartPtr(b, csParam);
#ifdef BLOCKSOLVER_PRECISE_Q // high precision Q
    qp = ColorSpinorField::CreateSmartPtr(csParam);
    tmpp = ColorSpinorField::CreateSmartPtr(csParam);
#endif
    // sloppy fields
    csParam.setPrecision(param.precision_sloppy);
    x_sloppy_savedp = ColorSpinorField::CreateSmartPtr(csParam);
    pp = ColorSpinorField::CreateSmartPtr(csParam);
#ifdef BLOCKSOLVER_PRECISE_Q // we need an extra temporary p since we can't tmpp <-> qp <-> pp swap anymore.
    p_oldp = ColorSpinorField::CreateSmartPtr(csParam);
#else
    qp = ColorSpinorField::CreateSmartPtr(csParam); // We need a sloppy q.
    tmpp = ColorSpinorField::CreateSmartPtr(csParam);
#endif
    App = ColorSpinorField::CreateSmartPtr(csParam);
    tmp_matsloppyp = ColorSpinorField::CreateSmartPtr(csParam);
    init = true;

  }
  ColorSpinorField &r = *rp;
  ColorSpinorField &y = *yp;
  ColorSpinorField &x_sloppy_saved = *x_sloppy_savedp;
  ColorSpinorField &Ap = *App;
  ColorSpinorField &tmp_matsloppy = *tmp_matsloppyp;

  std::shared_ptr<ColorSpinorField> x_sloppyp; // Gets assigned below.

  r.ExtendLastDimension();
  y.ExtendLastDimension();
  x_sloppy_saved.ExtendLastDimension();
  Ap.ExtendLastDimension();
  pp->ExtendLastDimension();
#ifdef BLOCKSOLVER_PRECISE_Q
  p_oldp->ExtendLastDimension();
#endif
  qp->ExtendLastDimension();
  tmpp->ExtendLastDimension();
  tmp_matsloppy.ExtendLastDimension();

  csParam.setPrecision(param.precision_sloppy);
  // tmp2 only needed for multi-gpu Wilson-like kernels
  //ColorSpinorField *tmp2_p = !mat.isStaggered() ?
  //ColorSpinorField::Create(x, csParam) : &tmp;
  //ColorSpinorField &tmp2 = *tmp2_p;
  std::shared_ptr<ColorSpinorField> tmp2_p;// = nullptr;

  if(!mat.isStaggered()){
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    tmp2_p =  ColorSpinorField::CreateSmartPtr(x, csParam);
    tmp2_p->ExtendLastDimension();
  } else {
    tmp2_p = tmp_matsloppyp;
    }

  ColorSpinorField &tmp2 = *tmp2_p;

  // additional high-precision temporary if Wilson and mixed-precision
  csParam.setPrecision(param.precision);

  std::shared_ptr<ColorSpinorField> tmp3_p;// = nullptr;
  if(param.precision != param.precision_sloppy && !mat.isStaggered()){
    //csParam.create = QUDA_ZERO_FIELD_CREATE;
    tmp3_p =  ColorSpinorField::CreateSmartPtr(x, csParam); //ColorSpinorField::Create(csParam);
    tmp3_p->ExtendLastDimension();
  } else {
    tmp3_p = tmp_matsloppyp;
  }

  ColorSpinorField &tmp3 = *tmp3_p;

  // calculate residuals for all vectors
  //for(int i=0; i<nsrc; i++){
  //  mat(r.Component(i), x.Component(i), y.Component(i));
  //  blas::xmyNorm(b.Component(i), r.Component(i));
  //}
  // Step 2: R = AX - B, using Y as a temporary with the right precision.
#ifdef BLOCKSOLVE_DSLASH5D
  mat(r, x, y, tmp3);
#else
  for (int i = 0; i < nsrc; i++)
  {
    mat(r.Component(i), x.Component(i), y.Component(i), tmp3.Component(i));
    // blas::xpay(b.Component(i), -1.0, r.Component(i));
  }
#endif

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  blas::xpay(b, -1.0, r);
#else
  for (int i = 0; i < nsrc; i++)
  {
    blas::xpay(b.Component(i), -1.0, r.Component(i));
  }
#endif

  // Step 3: Y = X
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::copy(y, x);
#else
  for (int i = 0; i < nsrc; i++)
    blas::copy(y.Component(i), x.Component(i));
#endif

  // Step 4: Xs = 0
  // Set field aliasing according to whether
  // we're doing mixed precision or not. Based
  // on BiCGstab-L conventions.
  if (param.precision_sloppy == x.Precision() || !param.use_sloppy_partial_accumulator)
  {
    x_sloppyp = std::shared_ptr<ColorSpinorField>( &x, [](ColorSpinorField*){} );
    // x_sloppyp = std::shared_ptr<ColorSpinorField>(&x, ; // s_sloppy and x point to the same vector in memory.
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::zero(*x_sloppyp); // x_sloppy is zeroed out (and, by extension, so is x)
#else
    for (int i = 0; i < nsrc; i++)
      blas::zero(x_sloppyp->Component(i));
#endif
  }
  else
  {
    x_sloppyp = x_sloppy_savedp; // x_sloppy point to saved x_sloppy memory.
                                 // x_sloppy_savedp was already zero.
  }
  // No need to alias r---we need a separate q, which is analagous
  // to an 'r_sloppy' in other algorithms.

  // Syntatic sugar.
  ColorSpinorField &x_sloppy = *x_sloppyp;

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  // Set up eigen matrices here.
  // We need to do some goofing around with Eigen maps.
  // https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html

  // Allocate some raw memory for each matrix we need raw pointers for.
  Complex H_raw[nsrc*nsrc];
  Complex pAp_raw[nsrc*nsrc];
  Complex alpha_raw[nsrc*nsrc];
  Complex beta_raw[nsrc*nsrc];
  Complex Linv_raw[nsrc*nsrc];
  Complex Sdagger_raw[nsrc*nsrc];

  // Convenience. By default, Eigen matrices are column major.
  // We switch to row major because cDotProduct and
  // the multi-blas routines use row
  typedef Matrix<Complex, nsrc, nsrc, RowMajor> MatrixBCG;

  // Create maps. This forces the above pointers to be used under the hood.
  Map<MatrixBCG> H(H_raw, nsrc, nsrc);
  Map<MatrixBCG> pAp(pAp_raw, nsrc, nsrc);
  Map<MatrixBCG> alpha(alpha_raw, nsrc, nsrc);
  Map<MatrixBCG> beta(beta_raw, nsrc, nsrc);
  Map<MatrixBCG> Linv(Linv_raw, nsrc, nsrc);
  Map<MatrixBCG> Sdagger(Sdagger_raw, nsrc, nsrc);

  // Create other non-mapped matrices.
  MatrixBCG L = MatrixBCG::Zero(nsrc,nsrc);
  MatrixBCG C = MatrixBCG::Zero(nsrc,nsrc);
  MatrixBCG C_old = MatrixBCG::Zero(nsrc,nsrc);
  MatrixBCG S = MatrixBCG::Identity(nsrc,nsrc); // Step 10: S = I

#ifdef BLOCKSOLVER_VERBOSE
  Complex* pTp_raw = new Complex[nsrc*nsrc];
  Map<MatrixBCG> pTp(pTp_raw,nsrc,nsrc);
#endif
#else
  // Eigen Matrices instead of scalars
  MatrixXcd H = MatrixXcd::Zero(nsrc, nsrc);
  MatrixXcd alpha = MatrixXcd::Zero(nsrc,nsrc);
  MatrixXcd beta = MatrixXcd::Zero(nsrc,nsrc);
  MatrixXcd C = MatrixXcd::Zero(nsrc,nsrc);
  MatrixXcd C_old = MatrixXcd::Zero(nsrc,nsrc);
  MatrixXcd S = MatrixXcd::Identity(nsrc,nsrc); // Step 10: S = I
   MatrixXcd L = MatrixXcd::Zero(nsrc, nsrc);
  MatrixXcd Linv = MatrixXcd::Zero(nsrc, nsrc);
  MatrixXcd pAp = MatrixXcd::Identity(nsrc,nsrc);

// temporary workaround
   // MatrixXcd Sdagger = MatrixXcd::Identity(nsrc,nsrc);
   Complex Sdagger_raw[nsrc*nsrc];

   typedef Eigen::Matrix<Complex, nsrc, nsrc, Eigen::RowMajor> MatrixBCG;
   Eigen::Map<MatrixBCG> Sdagger(Sdagger_raw, nsrc, nsrc);
   

  #ifdef BLOCKSOLVER_VERBOSE
  MatrixXcd pTp =  MatrixXcd::Identity(nsrc,nsrc);
  #endif
#endif 

  // Step 5: H = (R)^\dagger R
  double r2avg=0;
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  blas::hDotProduct(H_raw, r.Components(), r.Components());

  for (int i = 0; i < nsrc; i++)
  {
    r2avg += H(i,i).real();
    printfQuda("r2[%i] %e\n", i, H(i,i).real());
  }
#else
  for(int i=0; i<nsrc; i++){
    for(int j=i; j < nsrc; j++){
      H(i,j) = blas::cDotProduct(r.Component(i),r.Component(j));
      if (i!=j) H(j,i) = std::conj(H(i,j));
      if (i==j) {
        r2avg += H(i,i).real();
        printfQuda("r2[%i] %e\n", i, H(i,i).real());
      }
    }
  }
#endif
  printmat("r2", H);

  //ColorSpinorParam cs5dParam(p);
  //cs5dParam.create = QUDA_REFERENCE_FIELD_CREATE;
  //cs5dParam.x[4] = nsrc;
  //cs5dParam.is_composite = false;

  //cudaColorSpinorField Ap5d(Ap,cs5dParam); 
  //cudaColorSpinorField p5d(p,cs5dParam); 
  //cudaColorSpinorField tmp5d(tmp,cs5dParam); 
  //cudaColorSpinorField tmp25d(tmp2,cs5dParam);  

  const bool use_heavy_quark_res =
  (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
  if(use_heavy_quark_res) errorQuda("ERROR: heavy quark residual not supported in block solver");

  // Create the worker class for updating x_sloppy. 
  // When we hit matSloppy, tmpp contains P.
#ifdef BLOCKSOLVER_PRECISE_Q

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  BlockCGUpdate blockcg_update(x_sloppyp, p_oldp, alpha_raw);
#else
  BlockCGUpdate blockcg_update(x_sloppyp, p_oldp, alpha);
#endif

#else // !BLOCKSOLVER_PRECISE_Q

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  BlockCGUpdate blockcg_update(x_sloppyp, tmpp, alpha_raw);
#else
  BlockCGUpdate blockcg_update(x_sloppyp, tmpp, alpha);
#endif

#endif

  profile.TPSTOP(QUDA_PROFILE_INIT);
  profile.TPSTART(QUDA_PROFILE_PREAMBLE);

  double stop[QUDA_MAX_BLOCK_SRC];

  for(int i = 0; i < nsrc; i++){
    stop[i] = stopping(param.tol, b2[i], param.residual_type);  // stopping condition of solver
  }

  profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
  profile.TPSTART(QUDA_PROFILE_COMPUTE);
  blas::flops = 0;

  int k = 0;

#ifdef BLOCKSOLVER_RELIABLE_POLICY_MIN
  double rNorm = 1e30; // reliable update policy is to use the smallest residual.
#else
  double rNorm = 0.0; // reliable update policy is to use the largest residual.
#endif

  PrintStats("Block-CG", k, r2avg / nsrc, b2avg, 0.);
  bool allconverged = true;
  bool converged[QUDA_MAX_BLOCK_SRC];
  for(int i=0; i<nsrc; i++){
    converged[i] = convergence(H(i,i).real(), 0., stop[i], param.tol_hq);
    allconverged = allconverged && converged[i];
#ifdef BLOCKSOLVER_RELIABLE_POLICY_MIN
    if (rNorm > sqrt(H(i,i).real())) rNorm = sqrt(H(i,i).real());
#else
    if (rNorm < sqrt(H(i,i).real())) rNorm = sqrt(H(i,i).real());
#endif
  }

  //double r0Norm = rNorm;
  double maxrx = rNorm;
  double maxrr = rNorm;
  double delta = param.delta;
  printfQuda("Reliable update delta = %.8f\n", delta);

  int rUpdate = 0;
  //int steps_since_reliable = 1;

  // Only matters for heavy quark residuals, which we don't have enabled
  // in blockCG (yet).
  //bool L2breakdown = false;

  // Step 6: L L^\dagger = H, Cholesky decomposition, L lower left triangular
  // Step 7: C = L^\dagger, C upper right triangular.
  // Set Linv = C.inverse() for convenience in the next step.
  L = H.llt().matrixL(); // retrieve factor L in the decomposition
  C = L.adjoint();
  std::cout << "RELUP " << C.norm() << " " << rNorm << std::endl;
  Linv = C.inverse();

  rNorm = C.norm();
  maxrx = C.norm();
  maxrr = C.norm();
  printfQuda("rNorm = %.8e on iteration %d!\n", rNorm, 0);

  #ifdef BLOCKSOLVER_VERBOSE
  std::cout << "r2\n " << H << std::endl;
  std::cout << "L\n " << L.adjoint() << std::endl;
  std::cout << "Linv = \n" << Linv << "\n";
  #endif

  // Step 8: finally set Q to thin QR decompsition of R.
  //blas::zero(*qp); // guaranteed to be zero at start.
#ifdef BLOCKSOLVER_PRECISE_Q

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  blas::caxpy_U(Linv_raw,r.Components(),qp->Components());
#else
  for(int i=0; i<nsrc; i++){
    for(int j=i;j<nsrc; j++){
      blas::caxpy(Linv(i,j), r.Component(i), qp->Component(j));
    }
  }
#endif

#else

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  blas::copy(*tmpp, r); // Need to do this b/c r is fine, q is sloppy, can't caxpy w/ x fine, y sloppy.
  blas::caxpy_U(Linv_raw,tmpp->Components(),qp->Components());  // C is upper triangular, so its inverse is.
#else
  for(int i=0; i<nsrc; i++){
    blas::copy(tmpp->Component(i), r.Component(i));
    for(int j=i;j<nsrc; j++){
      blas::caxpy(Linv(i,j), tmpp->Component(i), qp->Component(j));
    }
  }
#endif

#endif // BLOCKSOLVER_PRECISE_Q

  // Step 9: P = Q; additionally set P to thin QR decompoistion of r
  blas::copy(*pp, *qp);

#ifdef BLOCKSOLVER_VERBOSE
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  blas::hDotProduct(pTp_raw, pp->Components(), pp->Components());
#else
  for(int i=0; i<nsrc; i++){
    for(int j=0; j<nsrc; j++){
      pTp(i,j) = blas::cDotProduct(pp->Component(i), pp->Component(j));
    }
  }
#endif

  std::cout << " pTp  " << std::endl << pTp << std::endl;
  std::cout << " L " << std::endl << L.adjoint() << std::endl;
  std::cout << " C " << std::endl << C << std::endl;
  #endif

  // Step 10 was set S to the identity, but we took care of that
  // when we initialized all of the matrices. 

  bool just_reliable_updated = false; 
  cudaProfilerStart();
  while ( !allconverged && k < param.maxiter ) {
    // PUSH_RANGE("Dslash",1)
    //for(int i=0; i<nsrc; i++){
    // matSloppy(Ap.Component(i), p.Component(i), tmp.Component(i), tmp2.Component(i));  // tmp as tmp
    //}

    // Prepare to overlap some compute with comms.
    if (k > 0 && !just_reliable_updated)
    {
      dslash::aux_worker = &blockcg_update;
    }
    else
    {
      dslash::aux_worker = NULL;
      just_reliable_updated = false;
    }
    PUSH_RANGE("Dslash_sloppy",0)
    // Step 12: Compute Ap.
#ifdef BLOCKSOLVE_DSLASH5D
    matSloppy(Ap, *pp, tmp_matsloppy, tmp2);
#else
    for (int i = 0; i < nsrc; i++)
      matSloppy(Ap.Component(i), pp->Component(i), tmp_matsloppy.Component(i), tmp2.Component(i));
#endif
    POP_RANGE

    PUSH_RANGE("Reduction",1)
    // Step 13: calculate pAp = P^\dagger Ap
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::hDotProduct_Anorm(pAp_raw, pp->Components(), Ap.Components());
#else
    for(int i=0; i<nsrc; i++){
      for(int j=i; j < nsrc; j++){
        pAp(i,j) = blas::cDotProduct(pp->Component(i), Ap.Component(j));
        if (i!=j) pAp(j,i) = std::conj(pAp(i,j));
      }
    }
#endif
    POP_RANGE
    printmat("pAp", pAp);
    PUSH_RANGE("Eigen",3)
#ifdef BLOCKSOLVER_EXPLICIT_PAP_HERMITIAN
    H = 0.5*(pAp + pAp.adjoint().eval());
    pAp = H;
#endif

    // Step 14: Compute beta = pAp^(-1)
    // For convenience, we stick a minus sign on it now.
    beta = -pAp.inverse();

    // Step 15: Compute alpha = beta * C
    alpha = - beta * C;
    POP_RANGE

//MWREL: regular update

    // Step 16: update Xsloppy = Xsloppy + P alpha
    // This step now gets overlapped with the
    // comms in matSloppy. 
/*
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::caxpy(alpha_raw, *pp, x_sloppy);
#else
    for(int i = 0; i < nsrc; i++){
      for(int j = 0; j < nsrc; j++){
        blas::caxpy(alpha(i,j), pp->Component(i), x_sloppy.Component(j));
      }
    }
#endif
*/

    // Step 17: Update Q = Q - Ap beta (remember we already put the minus sign on beta)
    // update rSloppy
    PUSH_RANGE("BLAS",2)
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::caxpy(beta_raw, Ap, *qp);
#else
    for(int i=0; i<nsrc; i++){
      for(int j=0;j<nsrc; j++){
        blas::caxpy(beta(i,j), Ap.Component(i), qp->Component(j));
      }
    }
#endif

//MWALTBETA
    #ifdef BLOCKSOLVER_ALTERNATIVE_BETA
    #ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::zero(*tmpp);
    blas::caxpy(beta_raw, Ap, *tmpp);
    blas::hDotProduct(H_raw, qp->Components(), tmpp->Components());
    L = H.llt().matrixL();
    S = L.adjoint();
    // std::cout << "altS" << S;
    #endif
    #endif
    POP_RANGE

    PUSH_RANGE("Reduction",1)
    // Orthogonalize Q via a thin QR decomposition.
    // Step 18: H = Q^\dagger Q
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::hDotProduct(H_raw, qp->Components(), qp->Components());
#else
    printfQuda("Iteration %d\n",k);
    for(int i=0; i<nsrc; i++){
      for(int j=i; j < nsrc; j++){
        H(i,j) = blas::cDotProduct(qp->Component(i),qp->Component(j));
        //printfQuda("r2(%d,%d) = %.15e + I %.15e\n", i, j, real(r2(i,j)), imag(r2(i,j)));
        if (i!=j) H(j,i) = std::conj(H(i,j));
    }
      }
#endif
    printmat("r2", H);
  POP_RANGE
  PUSH_RANGE("Eigen",3)
    // Step 19: L L^\dagger = H; Cholesky decomposition of H, L is lower left triangular.
    L = H.llt().matrixL();// retrieve factor L  in the decomposition
    
    // Step 20: S = L^\dagger
    S = L.adjoint();
    // std::cout << "regS" << L.adjoint() << std::endl;

    // Step 21: Q = Q S^{-1}
    // This would be most "cleanly" implemented
    // with a block-cax, but that'd be a pain to implement.
    // instead, we accomplish it with a caxy, then a pointer
    // swap.

    Linv = S.inverse();
  POP_RANGE
  PUSH_RANGE("BLAS",2)
  blas::zero(*tmpp);
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::caxpy_U(Linv_raw, *qp, *tmpp); // tmp is acting as Q.
#else
    for(int i=0; i<nsrc; i++){
      for(int j=i;j<nsrc; j++){
        blas::caxpy(Linv(i,j), qp->Component(i), tmpp->Component(j));
      }
    }
#endif
    POP_RANGE

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    std::swap(qp, tmpp); // now Q actually is Q. tmp is the old Q.
#else
    // Technically, this is a 5D function that should
    // be split into a bunch of 4D functions... but it's
    // pointer swapping, that gets messy.
    std::swap(qp, tmpp); // now Q actually is Q. tmp is the old Q.
#endif

    PUSH_RANGE("Eigen",3)
    // Step 22: Back up C (we need to have it if we trigger a reliable update)
    C_old = C;

    // Step 23: Update C. C = S * C_old. This will get overridden if we
    // trigger a reliable update.
    C = S * C;

    // Step 24: calculate the residuals for all shifts. We use these
    // to determine if we need to do a reliable update, and if we do,
    // these values get recomputed.
#ifdef BLOCKSOLVER_RELIABLE_POLICY_MIN
    double r2 = 1e30; // reliable update policy is to use the smallest residual.
#else
    double r2 = 0.0; // reliable update policy is to use the largest residual.
#endif

    r2avg=0;
    for (int j=0; j<nsrc; j++ ){
      H(j,j) = C(0,j)*conj(C(0,j));
      for(int i=1; i < nsrc; i++)
        H(j,j) += C(i,j) * conj(C(i,j));
      r2avg += H(j,j).real();
#ifdef BLOCKSOLVER_RELIABLE_POLICY_MIN
      if (r2 > H(j,j).real()) r2 = H(j,j).real();
  #else
      if (r2 < H(j,j).real()) r2 = H(j,j).real();
  #endif
      }
    POP_RANGE
#ifdef BLOCKSOLVER_EXPLICIT_QP_ORTHO
    bool did_reliable = false;
#endif
    rNorm = C.norm();
    printfQuda("rNorm = %.8e on iteration %d!\n", rNorm, k);

    // reliable update
    if (block_reliable(rNorm, maxrx, maxrr, r2, delta))
    {
#ifdef BLOCKSOLVER_EXPLICIT_QP_ORTHO
      did_reliable = true;
#endif
      printfQuda("Triggered a reliable update on iteration %d!\n", k);
      // This was in the BiCGstab(-L) reliable updates, but I don't
      // think it's necessary... blas::xpy should support updating
      // y from a lower precision vector.
      //if (x.Precision() != x_sloppy.Precision())
      //{
      //  blas::copy(x, x_sloppy);
      //}

      // If we triggered a reliable update, we need
      // to do this X update now.
      // Step 16: update Xsloppy = Xsloppy + P alpha
      PUSH_RANGE("BLAS",1)
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::caxpy(alpha_raw, *pp, x_sloppy);
#else
      for(int i = 0; i < nsrc; i++){
        for(int j = 0; j < nsrc; j++){
          blas::caxpy(alpha(i,j), pp->Component(i), x_sloppy.Component(j));
    }
      }
#endif

      // Reliable updates step 2: Y = Y + X_s
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::xpy(x_sloppy, y);
#else
      for (int i = 0; i < nsrc; i++)
        blas::xpy(x_sloppy.Component(i), y.Component(i));
#endif
      POP_RANGE
      // Don't do aux work!
      dslash::aux_worker = NULL;

      PUSH_RANGE("Dslash",4)
      // Reliable updates step 4: R = AY - B, using X as a temporary with the right precision.
#ifdef BLOCKSOLVE_DSLASH5D
      mat(r, y, x, tmp3);
#else
      for (int i = 0; i < nsrc; i++)
        mat(r.Component(i), y.Component(i), x.Component(i), tmp3.Component(i));
#endif
      POP_RANGE
      PUSH_RANGE("BLAS",2)
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::xpay(b, -1.0, r);
#else
      for (int i = 0; i < nsrc; i++)
        blas::xpay(b.Component(i), -1.0, r.Component(i));
#endif

      // Reliable updates step 3: X_s = 0.
      // If x.Precision() == x_sloppy.Precision(), they refer
      // to the same pointer under the hood.
      // x gets used as a temporary in mat(r,y,x) above.
      // That's why we need to wait to zero 'x_sloppy' until here.
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::zero(x_sloppy);
#else
      for (int i = 0; i < nsrc; i++)
        blas::zero(x_sloppy.Component(i));
#endif
      POP_RANGE
      // Reliable updates step 5: H = (R)^\dagger R
      r2avg=0;
      PUSH_RANGE("Reduction",1)
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::hDotProduct(H_raw, r.Components(), r.Components());
      for (int i = 0; i < nsrc; i++)
      {
        r2avg += H(i,i).real();
        printfQuda("r2[%i] %e\n", i, H(i,i).real());
      }
#else
      for(int i=0; i<nsrc; i++){
        for(int j=i; j < nsrc; j++){
          H(i,j) = blas::cDotProduct(r.Component(i),r.Component(j));
          if (i!=j) H(j,i) = std::conj(H(i,j));
          if (i==j) {
            r2avg += H(i,i).real();
            printfQuda("r2[%i] %e\n", i, H(i,i).real());
          }
        }
      }
#endif
      POP_RANGE
      PUSH_RANGE("Eigen",3)
      printmat("reliable r2", H);

      // Reliable updates step 6: L L^\dagger = H, Cholesky decomposition, L lower left triangular
      // Reliable updates step 7: C = L^\dagger, C upper right triangular.
      // Set Linv = C.inverse() for convenience in the next step.
      L = H.llt().matrixL(); // retrieve factor L in the decomposition
      C = L.adjoint();
      Linv = C.inverse();
      POP_RANGE

#ifdef BLOCKSOLVER_VERBOSE
      std::cout << "r2\n " << H << std::endl;
      std::cout << "L\n " << L.adjoint() << std::endl;
#endif

      PUSH_RANGE("BLAS",2)
      // Reliable updates step 8: set Q to thin QR decompsition of R.
      blas::zero(*qp);

#ifdef BLOCKSOLVER_PRECISE_Q

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::caxpy_U(Linv_raw,r,*qp);
#else
      // temporary hack - use AC to pass matrix arguments to multiblas
      for(int i=0; i<nsrc; i++){
        for(int j=i;j<nsrc; j++){
          blas::caxpy(Linv(i,j), r.Component(i), qp->Component(j));
        }
      }
#endif

#else // !BLOCKSOLVER_PRECISE_Q

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::copy(*tmpp, r); // Need to do this b/c r is fine, q is sloppy, can't caxpy w/ x fine, y sloppy.
      blas::caxpy_U(Linv_raw,*tmpp,*qp);
#else
      // temporary hack - use AC to pass matrix arguments to multiblas
      for(int i=0; i<nsrc; i++){
        blas::copy(tmpp->Component(i), r.Component(i));
        for(int j=i;j<nsrc; j++){
          blas::caxpy(Linv(i,j), tmpp->Component(i), qp->Component(j));
        }
      }
#endif

#endif

      POP_RANGE
      PUSH_RANGE("Eigen",3)

      // Reliable updates step 9: Set S = C * C_old^{-1} (equation 6.1 in the blockCGrQ paper)
      // S = C * C_old.inverse();
#ifdef BLOCKSOLVER_VERBOSE
      std::cout << "reliable S " << S << std::endl;
#endif
      POP_RANGE

      // Reliable updates step 10: Recompute residuals, reset rNorm, etc.
#ifdef BLOCKSOLVER_RELIABLE_POLICY_MIN
      rNorm = 1e30; // reliable update policy is to use the smallest residual.
#else
      rNorm = 0.0; // reliable update policy is to use the largest residual.
#endif
      allconverged = true;
      for(int i=0; i<nsrc; i++){
        converged[i] = convergence(H(i,i).real(), 0., stop[i], param.tol_hq);
        allconverged = allconverged && converged[i];
#ifdef BLOCKSOLVER_RELIABLE_POLICY_MIN
        if (rNorm > sqrt(H(i,i).real())) rNorm = sqrt(H(i,i).real());
#else
        if (rNorm < sqrt(H(i,i).real())) rNorm = sqrt(H(i,i).real());
#endif
      }
      rNorm = C.norm();
      maxrx = rNorm;
      maxrr = rNorm;
      rUpdate++;

      just_reliable_updated = true; 

    } // end reliable.

    // Debug print of Q.
#ifdef BLOCKSOLVER_VERBOSE
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::hDotProduct(pTp_raw, qp->Components(), qp->Components());
#else
    for(int i=0; i<nsrc; i++){
      for(int j=0; j<nsrc; j++){
        pTp(i,j) = blas::cDotProduct(qp->Component(i), qp->Component(j));
      }
    }
#endif
    std::cout << " qTq " << std::endl << pTp << std::endl;
    std::cout <<  "QR" << S<<  std::endl << "QP " << S.inverse()*S << std::endl;;
    #endif

    // Step 28: P = Q + P S^\dagger
    // This would be best done with a cxpay,
    // but that's difficult for the same
    // reason a block cax is difficult.
    // Instead, we do it by a caxpyz + pointer swap.
    PUSH_RANGE("Eigen",3)
    Sdagger = S.adjoint();
    POP_RANGE
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    PUSH_RANGE("BLAS",2)
    if (just_reliable_updated) blas::caxpyz(Sdagger_raw,*pp,*qp,*tmpp); // tmp contains P.
    else blas::caxpyz_L(Sdagger_raw,*pp,*qp,*tmpp); // tmp contains P.
    POP_RANGE
#else
    PUSH_RANGE("BLAS",2)
//FIXME: check whether the implementation without BLOCKSOLVER_MULTIFUNCTIONS is correct
    if (just_reliable_updated) {
      // blas::caxpyz(Sdagger_raw,*pp,*qp,*tmpp); // tmp contains P.
      for (int j = 0; j < nsrc; j++){
        blas::copy(tmpp->Component(j), qp->Component(j));
    }
      for(int i=0; i<nsrc; i++){
        for(int j=0;j<nsrc; j++){
          blas::caxpy(Sdagger(i,j), pp->Component(i), tmpp->Component(j));
        }
      }
    } 
    else {
      // blas::caxpyz_L(Sdagger_raw,*pp,*qp,*tmpp); // tmp contains P.
      for (int j = 0; j < nsrc; j++){
        blas::copy(tmpp->Component(j), qp->Component(j));
        }
      for(int i=0; i<nsrc; i++){
        for(int j=0;j<=i; j++){
          blas::caxpy(Sdagger(i,j), pp->Component(i), tmpp->Component(j));
        }
      }
    }
    POP_RANGE
#endif

#ifdef BLOCKSOLVER_PRECISE_Q
    // Need to copy tmpp into p_oldp to reduce precision, then swap so p_oldp
    // actually holds the old p and pp actually holds the new one.
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::copy(*p_oldp, *tmpp);
#else
    for (int i = 0; i < nsrc; i++) {
      blas::copy(p_oldp->Component(i), tmpp->Component(i));
    }
#endif
    std::swap(pp,p_oldp);

#else 
    std::swap(pp,tmpp); // now P contains P, tmp now contains P_old
#endif

    // Done with step 28.

// THESE HAVE NOT BEEN RE-WRITTEN FOR
// HIGH PRECISION Q YET.
#ifdef BLOCKSOLVER_EXPLICIT_QP_ORTHO
    if (did_reliable)
    {
      // Let's try to explicitly restore Q^\dagger P = I.
      Complex O_raw[nsrc*nsrc];
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      Map<MatrixBCG> O(O_raw, nsrc, nsrc);
      blas::cDotProduct(O_raw, qp->Components(), pp->Components());
#else
      MatrixXcd O = MatrixXcd::Zero(nsrc, nsrc);
      for(int i=0; i<nsrc; i++){
        for(int j=0; j<nsrc; j++){
          O(i,j) = blas::cDotProduct(qp->Component(i), pp->Component(j));
        }
      }
#endif

      printfQuda("Current Q^\\dagger P:\n");
      std::cout << O << "\n";
      
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      O -= MatrixBCG::Identity(nsrc,nsrc);
#else
      O -= MatrixXcd::Identity(nsrc,nsrc);
#endif
      O = -O;
      std::cout << "BLAH\n" << O << "\n";
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::caxpy(O_raw, *qp, *pp);
#else
    // temporary hack
      for(int i=0; i<nsrc; i++){
        for(int j=0;j<nsrc; j++){
          blas::caxpy(O(i,j),qp->Component(i),pp->Component(j));
      }
    }
#endif
      

      // Check...
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::cDotProduct(O_raw, qp->Components(), pp->Components());
#else
      for(int i=0; i<nsrc; i++){
        for(int j=0; j<nsrc; j++){
          O(i,j) = blas::cDotProduct(qp->Component(i), pp->Component(j));
    }
      }
#endif
      printfQuda("Updated Q^\\dagger P:\n");
      std::cout << O << "\n";
    }
    // End test...
#endif // BLOCKSOLVER_EXPLICIT_QP_ORTHO


#ifdef BLOCKSOLVER_VERBOSE
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::hDotProduct(pTp_raw, pp->Components(), pp->Components());
#else
    for(int i=0; i<nsrc; i++){
      for(int j=0; j<nsrc; j++){
        pTp(i,j) = blas::cDotProduct(pp->Component(i), pp->Component(j));
      }
    }
#endif

    std::cout << " pTp " << std::endl << pTp << std::endl;
    std::cout <<  "S " << S<<  std::endl << "C " << C << std::endl;
#endif

    k++;
    PrintStats("Block-CG", k, r2avg / nsrc, b2avg, 0);
    // Step 29: update the convergence check. H will contain the right
    // thing whether or not we triggered a reliable update.
    allconverged = true;
    for(int i=0; i<nsrc; i++){
      converged[i] = convergence(H(i,i).real(), 0, stop[i], param.tol_hq);
      allconverged = allconverged && converged[i];
    }


  }

  // Because we're overlapping communication w/ comms, 
  // x_sloppy doesn't get updated until the next iteration
  // (unless we happened ot hit a reliable update on the
  // last iteration).
  // However, we converged... so we never hit the next iteration.
  // We need to take care of this last update now.
  // Step ??: update Xsloppy = Xsloppy + P alpha
#ifdef BLOCKSOLVER_PRECISE_Q
  // But remember p_oldp holds the old p.
  if (!just_reliable_updated)
  {
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::caxpy(alpha_raw, *p_oldp, x_sloppy);
#else
    // temporary hack using AC
    for(int i = 0; i < nsrc; i++){
      for(int j = 0; j < nsrc; j++){
        blas::caxpy(alpha(i,j), p_oldp->Component(i), x_sloppy.Component(j));
    }
    }
#endif
  }

#else // !BLOCKSOLVER_PRECISE_Q
  // But remember tmpp holds the old P. 
  if (!just_reliable_updated)
  {
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::caxpy(alpha_raw, *tmpp, x_sloppy);
#else
    // temporary hack using AC
    for(int i = 0; i < nsrc; i++){
      for(int j = 0; j < nsrc; j++){
        blas::caxpy(alpha(i,j), tmpp->Component(i), x_sloppy.Component(j));
  }
    }
#endif
  }
#endif // BLOCKSOLVER_PRECISE_Q

  // We've converged!
  // Step 27: Update Xs into Y.
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  blas::xpy(x_sloppy, y);
#else
  for (int i = 0; i < nsrc; i++)
    blas::xpy(x_sloppy.Component(i), y.Component(i));
#endif


  // And copy the final answer into X!
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  blas::copy(x, y);
#else
  for (int i = 0; i < nsrc; i++)
    blas::copy(x.Component(i), y.Component(i));
#endif

  profile.TPSTOP(QUDA_PROFILE_COMPUTE);
  profile.TPSTART(QUDA_PROFILE_EPILOGUE);

  param.secs = profile.Last(QUDA_PROFILE_COMPUTE);

  double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
  param.gflops = gflops;
  param.iter += k;

  { // temporary addition for SC'17
    comm_allreduce(&gflops);
    printfQuda("Block-CG(%d): Convergence in %d iterations, %f seconds, GFLOPS = %g\n", nsrc, k, param.secs, gflops / param.secs);
  }

  if (k == param.maxiter)
  warningQuda("Exceeded maximum iterations %d", param.maxiter);

  if (getVerbosity() >= QUDA_VERBOSE)
   printfQuda("Block-CG: ReliableUpdateInfo (delta,iter,rupdates): %f %i %i \n", delta, k, rUpdate);



  dslash::aux_worker = NULL;

  if (param.compute_true_res) {
  // compute the true residuals
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    mat(r, x, y, tmp3);
#else
    for (int i = 0; i < nsrc; i++)
    mat(r.Component(i), x.Component(i), y.Component(i), tmp3.Component(i));
#endif
    for (int i=0; i<nsrc; i++){
    param.true_res = sqrt(blas::xmyNorm(b.Component(i), r.Component(i)) / b2[i]);
    param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
    param.true_res_offset[i] = param.true_res;
    param.true_res_hq_offset[i] = param.true_res_hq;
    }
  }

  for (int i=0; i<nsrc; i++) {
    std::stringstream str;
    str << "Block-CG " << i;
    PrintSummary(str.str().c_str(), k, H(i,i).real(), b2[i]);
  }

  // reset the flops counters
  blas::flops = 0;
  mat.flops();
  matSloppy.flops();

  profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
  profile.TPSTART(QUDA_PROFILE_FREE);

  // if (&tmp3 != tmp_matsloppyp) delete tmp3_p;
  // if (&tmp2 != tmp_matsloppyp) delete tmp2_p;

  profile.TPSTOP(QUDA_PROFILE_FREE);

  return;
}


  BlockCG::BlockCG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
    BlockSolver(param, profile), mat(mat), matSloppy(matSloppy), yp(nullptr), rp(nullptr), App(nullptr), tmpp(nullptr),
    x_sloppy_savedp(nullptr), pp(nullptr), qp(nullptr), tmp_matsloppyp(nullptr), p_oldp(nullptr),
    init(false) {
#ifndef BLOCKSOLVER
  warningQuda("QUDA_BLOCKSOLVER not built with MULTI-BLAS / Dslash options Performance will be slow.");
#endif 
  }

  BlockCG::~BlockCG() {


    if ( init ) {
      for (auto pi : p) delete pi;
      // if (rp) delete rp;
      // if (yp) delete yp;
      // if (App) delete App;
      // if (tmpp) delete tmpp;
// #ifdef BLOCKSOLVER
//       if (x_sloppy_savedp) delete x_sloppy_savedp;
//       if (pp) delete pp;
// #ifdef BLOCKSOLVER_PRECISE_Q
//       if (p_oldp) delete p_oldp;
// #endif
//       if (qp) delete qp;
//       if (tmp_matsloppyp) delete tmp_matsloppyp;
 // #endif
      init = false;
    }
  }

void BlockCG::operator()(ColorSpinorField& x, ColorSpinorField& b) {



  if (param.num_src > QUDA_MAX_BLOCK_SRC)
    errorQuda("Requested number of right-hand sides %d exceeds max %d\n", param.num_src, QUDA_MAX_BLOCK_SRC);

  switch (param.num_src) {
  case  1: solve_n< 1>(x, b); break;
  case  2: solve_n< 2>(x, b); break;
  case  3: solve_n< 3>(x, b); break;
  case  4: solve_n< 4>(x, b); break;
  case  5: solve_n< 5>(x, b); break;
  case  6: solve_n< 6>(x, b); break;
  case  7: solve_n< 7>(x, b); break;
  case  8: solve_n< 8>(x, b); break;
  case  9: solve_n< 9>(x, b); break;
  case 10: solve_n<10>(x, b); break;
  case 11: solve_n<11>(x, b); break;
  case 12: solve_n<12>(x, b); break;
  case 13: solve_n<13>(x, b); break;
  case 14: solve_n<14>(x, b); break;
  case 15: solve_n<15>(x, b); break;
  case 16: solve_n<16>(x, b); break;
  case 24: solve_n<24>(x, b); break;
  case 32: solve_n<32>(x, b); break;
  case 48: solve_n<48>(x, b); break; 
  case 64: solve_n<64>(x, b); break;
  default:
    errorQuda("Block-CG with dimension %d not supported", param.num_src);
  }



}
}
