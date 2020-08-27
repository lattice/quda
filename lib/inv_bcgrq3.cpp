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

namespace quda
{


// define this to use multi-functions, otherwise it'll
// do loops over dot products.
// this is more here for development convenience.

// #define BLOCKSOLVER_MULTIFUNCTIONS
// #define BLOCKSOLVE_DSLASH5D
// #define BLOCKSOLVER_VERBOSE

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

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
  using Eigen::Dynamic;
  using Eigen::Map;
  using Eigen::Matrix;
  using Eigen::RowMajor;
#else
  using Eigen::MatrixXcd;
#endif

  // Matrix printing functions

  template <typename Matrix> inline void printmat(const char *label, const Matrix &mat)
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
  class BlockCGUpdate : public Worker
  {

    ColorSpinorFieldVector &x_sloppyp;
    ColorSpinorFieldVector &p_oldp; // double pointer because p participates in pointer swapping
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    Complex *alpha;
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
    BlockCGUpdate(ColorSpinorFieldVector &x_sloppyp, ColorSpinorFieldVector &p_oldp, Complex *alpha) :
#else
    BlockCGUpdate(ColorSpinorFieldVector &x_sloppyp, ColorSpinorFieldVector &p_oldp, MatrixXcd &alpha) :
#endif
      x_sloppyp(x_sloppyp),
      p_oldp(p_oldp),
      alpha(alpha),
      n_rhs(p_oldp.size()),
      n_update(x_sloppyp[0]->Nspin() == 4 ? 4 : 2)
#ifdef BLOCKSOLVE_DSLASH5D
    {
      if(x_sloppyp.size()==0) errorQuda("Xsloppyp is zero");
      ;
    }
#else
    {
      curr_update = 0;
    }
#endif

    ~BlockCGUpdate() { ; }

    // note that we can't set the stream parameter here so it is
    // ignored.  This is more of a future design direction to consider
    void apply(const cudaStream_t &stream)
    {
      static int count = 0;

#ifdef BLOCKSOLVE_DSLASH5D
      // How many to update per apply.
      const int update_per_apply = n_rhs / n_update;

      // If the number of updates doesn't evenly divide into n_rhs, there's leftover.
      const int update_per_apply_on_last = n_rhs - n_update * update_per_apply;

      // Only update if there are things to apply.
      // Update 1 through n_count-1, as well as n_count if update_per_apply_blah_blah = 0.
      if ((count != n_update - 1 && update_per_apply != 0) || update_per_apply_on_last == 0) {
        std::vector<ColorSpinorField *> curr_p(p_oldp.begin() + count * update_per_apply,
                                               p_oldp.begin() + (count + 1) * update_per_apply);

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        blas::caxpy(&alpha[count * update_per_apply * n_rhs], curr_p, x_sloppyp);
#else
        for (int i = 0; i < update_per_apply; i++)
          for (int j = 0; j < n_rhs; j++)
            blas::caxpy(alpha(i + count * update_per_apply, j), *(curr_p[i]), x_sloppyp[j]);
#endif
      } else if (count == n_update - 1) // we're updating the leftover.
      {
        std::vector<ColorSpinorField *> curr_p(p_oldp.begin() + count * update_per_apply, p_oldp.end());
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        blas::caxpy(&alpha[count * update_per_apply * n_rhs], curr_p, x_sloppyp);
#else
        for (int i = 0; i < update_per_apply_on_last; i++)
          for (int j = 0; j < n_rhs; j++)
            blas::caxpy(alpha(i + count * update_per_apply, j), *(curr_p[i]), x_sloppyp(j));
#endif
      }
#else // BLOCKSOLVE_DSLASH5D
      if (count == 0) {
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        std::vector<ColorSpinorField *> curr_p;
        curr_p.emplace_back(p_oldp[curr_update]);
        blas::caxpy(&alpha[curr_update * n_rhs], curr_p, x_sloppyp);
#else
        for (int j = 0; j < n_rhs; j++) blas::caxpy(alpha(curr_update, j), *p_oldp[curr_update], *x_sloppyp[j]);
#endif
        if (++curr_update == n_rhs) curr_update = 0;
      }
#endif // BLOCKSOLVE_DSLASH5D

      if (++count == n_update) count = 0;
    }
  };

  // this is the Worker pointer that the dslash uses to launch the shifted updates
  namespace dslash
  {
    extern Worker *aux_worker;
  }

  // Code to check for reliable updates, copied from inv_bicgstab_quda.cpp
  // Technically, there are ways to check both 'x' and 'r' for reliable updates...
  // the current status in blockCG is to just look for reliable updates in 'r'.
  int BlockCG::block_reliable(double &rNorm, double &maxrx, double &maxrr, const double &r2, const double &delta)
  {
    // reliable updates
    // rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    // int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
    // int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0
    int updateR = (rNorm < delta * maxrr) ? 1 : 0;

    // printf("reliable %d %e %e %e %e\n", updateR, rNorm, maxrx, maxrr, r2);

    return updateR;
  }

  template <int nsrc> void BlockCG::solve_n(ColorSpinorFieldVector &x, ColorSpinorFieldVector &b)
  {

    if (checkLocation(*x[0], *b[0]) != QUDA_CUDA_FIELD_LOCATION) errorQuda("Not supported");

    profile.TPSTART(QUDA_PROFILE_INIT);

    // Check to see that we're not trying to invert on a zero-field source
    // MW: it might be useful to check what to do here.
    double b2[QUDA_MAX_BLOCK_SRC];
    double b2avg = 0;
    for (int i = 0; i < nsrc; i++) {
      b2[i] = blas::norm2(*(b[i]));
      b2avg += b2[i];
      if (b2[i] == 0) {
        profile.TPSTOP(QUDA_PROFILE_INIT);
        errorQuda("Warning: inverting on zero-field source - undefined for block solver\n");
        x = b;
        param.true_res = 0.0;
        param.true_res_hq = 0.0;
        return;
      }
    }

    b2avg = b2avg / nsrc;

    ColorSpinorParam csParam(*x[0]);

    // csParam.is_composite = true;
    // csParam.composite_dim = nsrc;
    // csParam.nDim = 5;
    // csParam.x[4] = 1;

    if (!init) {
      for(int i=0; i < nsrc; i++){
      csParam.create = QUDA_NULL_FIELD_CREATE;
      rp.emplace_back(ColorSpinorField::Create(csParam));
      // csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp.emplace_back(ColorSpinorField::Create(csParam));
    
#ifdef BLOCKSOLVER_PRECISE_Q // high precision Q
      qp.emplace_back(ColorSpinorField::Create(csParam));
      tmpp.emplace_back(ColorSpinorField::Create(csParam));
#endif
      // sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      x_sloppy_savedp.emplace_back(ColorSpinorField::Create(csParam));
      pp.emplace_back(ColorSpinorField::Create(csParam));
#ifdef BLOCKSOLVER_PRECISE_Q // we need an extra temporary p since we can't tmpp <-> qp <-> pp swap anymore.
      p_oldp.emplace_back(ColorSpinorField::Create(csParam));
#else
      qp.emplace_back(ColorSpinorField::Create(csParam)); // We need a sloppy q.
      tmpp.emplace_back(ColorSpinorField::Create(csParam));
#endif
      App.emplace_back(ColorSpinorField::Create(csParam));
      tmp_matsloppyp.emplace_back(ColorSpinorField::Create(csParam));
      }
      init = true;
    }
    ColorSpinorFieldVector &r = rp;
    ColorSpinorFieldVector &y = yp;
    ColorSpinorFieldVector &Ap = App;
    ColorSpinorFieldVector &tmp_matsloppy = tmp_matsloppyp;

    ColorSpinorFieldVector x_sloppyp;
    x_sloppyp.resize(nsrc); // Gets assigned below.

    csParam.setPrecision(param.precision_sloppy);
    // tmp2 only needed for multi-gpu Wilson-like kernels
    // ColorSpinorField *tmp2_p = !mat.isStaggered() ?
    // ColorSpinorField::Create(x, csParam) : &tmp;
    // ColorSpinorField &tmp2 = *tmp2_p;
    ColorSpinorFieldVector tmp2_p; // = nullptr;

    // if (!mat.isStaggered()) {
    //   csParam.create = QUDA_ZERO_FIELD_CREATE;
    //   tmp2_p.resize(nsrc, ColorSpinorField::Create(*x[0], csParam));
    // } else {
      tmp2_p = tmp_matsloppyp;
    // }

    ColorSpinorFieldVector &tmp2 = tmp2_p;

    // additional high-precision temporary if Wilson and mixed-precision
    csParam.setPrecision(param.precision);

    ColorSpinorFieldVector tmp3_p; // = nullptr;
    // if (param.precision != param.precision_sloppy && !mat.isStaggered()) {
    //   // csParam.create = QUDA_ZERO_FIELD_CREATE;
    //   tmp3_p.emplace_back(ColorSpinorField::Create(*x[0], csParam)); // ColorSpinorField::Create(csParam);
    //   // tmp3_p->ExtendLastDimension();
    // } else {
      tmp3_p = tmp_matsloppyp;
    // }

    ColorSpinorFieldVector &tmp3 = tmp_matsloppyp;

    // Step 2: R = AX - B, using Y as a temporary with the right precision.
// #ifdef BLOCKSOLVE_DSLASH5D
    // mat(r, x, y, tmp3);
// #else
    // for (int i = 0; i < nsrc; i++) { mat(*rp[i], *x[i], *yp[i], *tmp3_p[i]); }
// #endif
    for (int i = 0; i < nsrc; i++) {
      // r2avg += H(i, i).real();
      // printfQuda("r2[%i] %e\n", i, H(i, i).real());
            // printfQuda("CHECK r2 %e\n",blas::norm2(*r[i]));
      mat(*r[i], *x[i], *y[i], *tmp3[i]);
      // printfQuda("CHECK r2 %e\n",blas::norm2(*r[i]));
    }

    // #ifdef BLOCKSOLVER_MULTIFUNCTIONS
    // blas::xpay(b, -1.0, r);
    // #else
    for (int i = 0; i < nsrc; i++) { blas::xpay(*b[i], -1.0, *r[i]); }
    // #endif

    // Step 3: Y = X
    // #ifdef BLOCKSOLVER_MULTIFUNCTIONS
    // blas::copy(y, x);
    // #else
    for (int i = 0; i < nsrc; i++) blas::copy(*y[i], *x[i]);
    // #endif

    // Step 4: Xs = 0
    // Set field aliasing according to whether
    // we're doing mixed precision or not. Based
    // on BiCGstab-L conventions.
    if (param.precision_sloppy == x[0]->Precision() || !param.use_sloppy_partial_accumulator) {
      
      // printfQuda("Go here ...\n");
      for(int i=0; i < nsrc; i++) x_sloppyp[i] = x[i];
      // x_sloppyp = std::shared_ptr<ColorSpinorField>(&x, ; // s_sloppy and x point to the same vector in memory.
      // #ifdef BLOCKSOLVER_MULTIFUNCTIONS
      // blas::zero(*x_sloppyp); // x_sloppy is zeroed out (and, by extension, so is x)
      // #else
      for (int i = 0; i < nsrc; i++) blas::zero(*x_sloppyp[i]);
      // #endif
    } else {
      // printfQuda("Go there ...\n");
      x_sloppyp = x_sloppy_savedp; // x_sloppy point to saved x_sloppy memory.
                                   // x_sloppy_savedp was already zero.
    }
    // No need to alias r---we need a separate q, which is analagous
    // to an 'r_sloppy' in other algorithms.

    // Syntatic sugar.
    ColorSpinorFieldVector &x_sloppy = x_sloppyp;

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    // Set up eigen matrices here.
    // We need to do some goofing around with Eigen maps.
    // https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html

    // Allocate some raw memory for each matrix we need raw pointers for.
    Complex H_raw[nsrc * nsrc];
    Complex pAp_raw[nsrc * nsrc];
    Complex alpha_raw[nsrc * nsrc];
    Complex beta_raw[nsrc * nsrc];
    Complex Linv_raw[nsrc * nsrc];
    Complex Sdagger_raw[nsrc * nsrc];

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
    MatrixBCG L = MatrixBCG::Zero(nsrc, nsrc);
    MatrixBCG C = MatrixBCG::Zero(nsrc, nsrc);
    MatrixBCG C_old = MatrixBCG::Zero(nsrc, nsrc);
    MatrixBCG S = MatrixBCG::Identity(nsrc, nsrc); // Step 10: S = I

#ifdef BLOCKSOLVER_VERBOSE
    Complex *pTp_raw = new Complex[nsrc * nsrc];
    Map<MatrixBCG> pTp(pTp_raw, nsrc, nsrc);
#endif
#else
    // Eigen Matrices instead of scalars
    MatrixXcd H = MatrixXcd::Zero(nsrc, nsrc);
    MatrixXcd alpha = MatrixXcd::Zero(nsrc, nsrc);
    MatrixXcd beta = MatrixXcd::Zero(nsrc, nsrc);
    MatrixXcd C = MatrixXcd::Zero(nsrc, nsrc);
    MatrixXcd C_old = MatrixXcd::Zero(nsrc, nsrc);
    MatrixXcd S = MatrixXcd::Identity(nsrc, nsrc); // Step 10: S = I
    MatrixXcd L = MatrixXcd::Zero(nsrc, nsrc);
    MatrixXcd Linv = MatrixXcd::Zero(nsrc, nsrc);
    MatrixXcd pAp = MatrixXcd::Identity(nsrc, nsrc);

    // temporary workaround
    // MatrixXcd Sdagger = MatrixXcd::Identity(nsrc,nsrc);
    Complex Sdagger_raw[nsrc * nsrc];

    typedef Eigen::Matrix<Complex, nsrc, nsrc, Eigen::RowMajor> MatrixBCG;
    Eigen::Map<MatrixBCG> Sdagger(Sdagger_raw, nsrc, nsrc);

#ifdef BLOCKSOLVER_VERBOSE
    MatrixXcd pTp = MatrixXcd::Identity(nsrc, nsrc);
#endif
#endif

    // Step 5: H = (R)^\dagger R
    double r2avg = 0;
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
    blas::hDotProduct(H_raw, r, r);

    for (int i = 0; i < nsrc; i++) {
      r2avg += H(i, i).real();
      printfQuda("r2[%i] %e\n", i, H(i, i).real());
      printfQuda("CHECK r2 %e\n",blas::norm2(*r[i]));
    }
#else
    for (int i = 0; i < nsrc; i++) {
      for (int j = i; j < nsrc; j++) {
        H(i, j) = blas::cDotProduct(*r[i], *r[j]);
        if (i != j) H(j, i) = std::conj(H(i, j));
        if (i == j) {
          r2avg += H(i, i).real();
          printfQuda("r2[%i] %e\n", i, H(i, i).real());
        }
      }
    }
#endif
    printmat("r2", H);

    // ColorSpinorParam cs5dParam(p);
    // cs5dParam.create = QUDA_REFERENCE_FIELD_CREATE;
    // cs5dParam.x[4] = nsrc;
    // cs5dParam.is_composite = false;

    // cudaColorSpinorField Ap5d(Ap,cs5dParam);
    // cudaColorSpinorField p5d(p,cs5dParam);
    // cudaColorSpinorField tmp5d(tmp,cs5dParam);
    // cudaColorSpinorField tmp25d(tmp2,cs5dParam);

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    if (use_heavy_quark_res) errorQuda("ERROR: heavy quark residual not supported in block solver");

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

    for (int i = 0; i < nsrc; i++) {
      stop[i] = stopping(param.tol, b2[i], param.residual_type); // stopping condition of solver
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
    for (int i = 0; i < nsrc; i++) {
      converged[i] = convergence(H(i, i).real(), 0., stop[i], param.tol_hq);
      allconverged = allconverged && converged[i];
#ifdef BLOCKSOLVER_RELIABLE_POLICY_MIN
      if (rNorm > sqrt(H(i, i).real())) rNorm = sqrt(H(i, i).real());
#else
      if (rNorm < sqrt(H(i, i).real())) rNorm = sqrt(H(i, i).real());
#endif
    }

    // double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double delta = param.delta;
    printfQuda("Reliable update delta = %.8f\n", delta);

    int rUpdate = 0;
    // int steps_since_reliable = 1;

    // Only matters for heavy quark residuals, which we don't have enabled
    // in blockCG (yet).
    // bool L2breakdown = false;

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
    for(int i=0; i < nsrc; i++){
    blas::zero(*qp[i]); // guaranteed to be zero at start.
  }
#ifdef BLOCKSOLVER_PRECISE_Q

// #ifdef BLOCKSOLVER_MULTIFUNCTIONS
//     blas::caxpy_U(Linv_raw, r, qp);
// #else
    for (int i = 0; i < nsrc; i++) {
      for (int j = i; j < nsrc; j++) { blas::caxpy(Linv(i, j), *r[i], *qp[j]); }
    }
// #endif

#else

// #ifdef BLOCKSOLVER_MULTIFUNCTIONS
//     blas::copy(*tmpp, r);              // Need to do this b/c r is fine, q is sloppy, can't caxpy w/ x fine, y sloppy.
//     blas::caxpy_U(Linv_raw, tmpp, qp); // C is upper triangular, so its inverse is.
// #else
    for (int i = 0; i < nsrc; i++) {
      blas::copy(*tmpp[i], *r[i]);
      for (int j = i; j < nsrc; j++) { blas::caxpy(Linv(i, j), *tmpp[i], *qp[j]); }
    }
// #endif

#endif // BLOCKSOLVER_PRECISE_Q

    // Step 9: P = Q; additionally set P to thin QR decompoistion of r
    for (int i = 0; i < nsrc; i++) blas::copy(*pp[i], *qp[i]);

#ifdef BLOCKSOLVER_VERBOSE
// #ifdef BLOCKSOLVER_MULTIFUNCTIONS
    // blas::hDotProduct(pTp_raw, pp, pp);
// #else
    for (int i = 0; i < nsrc; i++) {
      for (int j = 0; j < nsrc; j++) { pTp(i, j) = blas::cDotProduct(*pp[i], *pp[j]); }
    }
// #endif

    std::cout << " pTp  " << std::endl << pTp << std::endl;
    std::cout << " L " << std::endl << L.adjoint() << std::endl;
    std::cout << " C " << std::endl << C << std::endl;
#endif

    // Step 10 was set S to the identity, but we took care of that
    // when we initialized all of the matrices.

    bool just_reliable_updated = false;
    // cudaProfilerStart();
    while (!allconverged && k < param.maxiter) {
      // Prepare to overlap some compute with comms.
      if (k > 0 && !just_reliable_updated) {
        dslash::aux_worker = &blockcg_update;
      } else {
        dslash::aux_worker = NULL;
        just_reliable_updated = false;
      }
      // Step 12: Compute Ap.
// #ifdef BLOCKSOLVE_DSLASH5D
      // matSloppy(Ap, *pp, tmp_matsloppy, tmp2);
// #else
      for (int i = 0; i < nsrc; i++) matSloppy(*Ap[i], *pp[i], *tmp_matsloppy[i], *tmp2[i]);
// #endif

      // Step 13: calculate pAp = P^\dagger Ap
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::hDotProduct_Anorm(pAp_raw, pp, Ap);
#else
      for (int i = 0; i < nsrc; i++) {
        for (int j = i; j < nsrc; j++) {
          pAp(i, j) = blas::cDotProduct(*pp[i], *Ap[j]);
          if (i != j) pAp(j, i) = std::conj(pAp(i, j));
        }
      }
#endif
      printmat("pAp", pAp);
#ifdef BLOCKSOLVER_EXPLICIT_PAP_HERMITIAN
      H = 0.5 * (pAp + pAp.adjoint().eval());
      pAp = H;
#endif

      // Step 14: Compute beta = pAp^(-1)
      // For convenience, we stick a minus sign on it now.
      beta = -pAp.inverse();

      // Step 15: Compute alpha = beta * C
      alpha = -beta * C;

      // MWREL: regular update

      // Step 16: update Xsloppy = Xsloppy + P alpha
      // This step now gets overlapped with the
      // comms in matSloppy.

      // Step 17: Update Q = Q - Ap beta (remember we already put the minus sign on beta)
      // update rSloppy

// #ifdef BLOCKSOLVER_MULTIFUNCTIONS
      // blas::caxpy(beta_raw, Ap, qp);
// #else
      for (int i = 0; i < nsrc; i++) {
        for (int j = 0; j < nsrc; j++) { blas::caxpy(beta(i, j), *Ap[i], *qp[j]); }
      }
// #endif

      // MWALTBETA
#ifdef BLOCKSOLVER_ALTERNATIVE_BETA
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::zero(*tmpp);
      blas::caxpy(beta_raw, Ap, tmpp);
      blas::hDotProduct(H_raw, qp, tmpp);
      L = H.llt().matrixL();
      S = L.adjoint();
// std::cout << "altS" << S;
#endif
#endif
      // Orthogonalize Q via a thin QR decomposition.
      // Step 18: H = Q^\dagger Q
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::hDotProduct(H_raw, qp, qp);
#else
      printfQuda("Iteration %d\n", k);
      for (int i = 0; i < nsrc; i++) {
        for (int j = i; j < nsrc; j++) {
          H(i, j) = blas::cDotProduct(*qp[i], *qp[j]);
          // printfQuda("r2(%d,%d) = %.15e + I %.15e\n", i, j, real(r2(i,j)), imag(r2(i,j)));
          if (i != j) H(j, i) = std::conj(H(i, j));
        }
      }
#endif
      printmat("r2", H);
      // Step 19: L L^\dagger = H; Cholesky decomposition of H, L is lower left triangular.
      L = H.llt().matrixL(); // retrieve factor L  in the decomposition

      // Step 20: S = L^\dagger
      S = L.adjoint();

      // Step 21: Q = Q S^{-1}
      // This would be most "cleanly" implemented
      // with a block-cax, but that'd be a pain to implement.
      // instead, we accomplish it with a caxy, then a pointer
      // swap.

      Linv = S.inverse();
      for (int i = 0; i < nsrc; i++) blas::zero(*tmpp[i]);
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::caxpy_U(Linv_raw, qp, tmpp); // tmp is acting as Q.
#else
      for (int i = 0; i < nsrc; i++) {
        for (int j = i; j < nsrc; j++) { blas::caxpy(Linv(i, j), *qp[i], *tmpp[j]); }
      }
#endif

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      qp.swap(tmpp); // now Q actually is Q. tmp is the old Q.
#else
      // Technically, this is a 5D function that should
      // be split into a bunch of 4D functions... but it's
      // pointer swapping, that gets messy.
      qp.swap(tmpp); // now Q actually is Q. tmp is the old Q.
#endif

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
      double r2 = 0.0;     // reliable update policy is to use the largest residual.
#endif

      r2avg = 0;
      for (int j = 0; j < nsrc; j++) {
        H(j, j) = C(0, j) * conj(C(0, j));
        for (int i = 1; i < nsrc; i++) H(j, j) += C(i, j) * conj(C(i, j));
        r2avg += H(j, j).real();
#ifdef BLOCKSOLVER_RELIABLE_POLICY_MIN
        if (r2 > H(j, j).real()) r2 = H(j, j).real();
#else
        if (r2 < H(j, j).real()) r2 = H(j, j).real();
#endif
      }
#ifdef BLOCKSOLVER_EXPLICIT_QP_ORTHO
      bool did_reliable = false;
#endif
      rNorm = C.norm();
      printfQuda("rNorm = %.8e on iteration %d!\n", rNorm, k);

      // reliable update
      if (block_reliable(rNorm, maxrx, maxrr, r2, delta)) {
#ifdef BLOCKSOLVER_EXPLICIT_QP_ORTHO
        did_reliable = true;
#endif
        printfQuda("Triggered a reliable update on iteration %d!\n", k);
        // This was in the BiCGstab(-L) reliable updates, but I don't
        // think it's necessary... blas::xpy should support updating
        // y from a lower precision vector.
        // if (x.Precision() != x_sloppy.Precision())
        //{
        //  blas::copy(x, x_sloppy);
        //}

        // If we triggered a reliable update, we need
        // to do this X update now.
        // Step 16: update Xsloppy = Xsloppy + P alpha
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        blas::caxpy(alpha_raw, pp, x_sloppy);
#else
        for (int i = 0; i < nsrc; i++) {
          for (int j = 0; j < nsrc; j++) { blas::caxpy(alpha(i, j), *pp[i], *x_sloppy[j]); }
        }
#endif

        // Reliable updates step 2: Y = Y + X_s
        // #ifdef BLOCKSOLVER_MULTIFUNCTIONS
        //         blas::xpy(x_sloppy, y);
        // #else
        for (int i = 0; i < nsrc; i++) blas::xpy(*x_sloppy[i], *y[i]);
        // #endif
        // Don't do aux work!
        dslash::aux_worker = NULL;

        // Reliable updates step 4: R = AY - B, using X as a temporary with the right precision.
// #ifdef BLOCKSOLVE_DSLASH5D
        // mat(r, y, x, tmp3);
// #else
        for (int i = 0; i < nsrc; i++) mat(*r[i], *y[i], *x[i], *tmp3[i]);
// #endif

        // #ifdef BLOCKSOLVER_MULTIFUNCTIONS
        // blas::xpay(b, -1.0, r);
        // #else
        for (int i = 0; i < nsrc; i++) blas::xpay(*b[i], -1.0, *r[i]);
        // #endif

        // Reliable updates step 3: X_s = 0.
        // If x.Precision() == x_sloppy.Precision(), they refer
        // to the same pointer under the hood.
        // x gets used as a temporary in mat(r,y,x) above.
        // That's why we need to wait to zero 'x_sloppy' until here.
        // #ifdef BLOCKSOLVER_MULTIFUNCTIONS
        // blas::zero(x_sloppy);
        // #else
        for (int i = 0; i < nsrc; i++) blas::zero(*x_sloppy[i]);
        // #endif
        // Reliable updates step 5: H = (R)^\dagger R
        r2avg = 0;
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        blas::hDotProduct(H_raw, r, r);
        for (int i = 0; i < nsrc; i++) {
          r2avg += H(i, i).real();
          printfQuda("r2[%i] %e\n", i, H(i, i).real());
        }
#else
        for (int i = 0; i < nsrc; i++) {
          for (int j = i; j < nsrc; j++) {
            H(i, j) = blas::cDotProduct(*r[i], *r[j]);
            if (i != j) H(j, i) = std::conj(H(i, j));
            if (i == j) {
              r2avg += H(i, i).real();
              printfQuda("r2[%i] %e\n", i, H(i, i).real());
            }
          }
        }
#endif
        printmat("reliable r2", H);

        // Reliable updates step 6: L L^\dagger = H, Cholesky decomposition, L lower left triangular
        // Reliable updates step 7: C = L^\dagger, C upper right triangular.
        // Set Linv = C.inverse() for convenience in the next step.
        L = H.llt().matrixL(); // retrieve factor L in the decomposition
        C = L.adjoint();
        Linv = C.inverse();

#ifdef BLOCKSOLVER_VERBOSE
        std::cout << "r2\n " << H << std::endl;
        std::cout << "L\n " << L.adjoint() << std::endl;
#endif

        // Reliable updates step 8: set Q to thin QR decompsition of R.
        for (int i = 0; i < nsrc; i++) blas::zero(*qp[i]);

#ifdef BLOCKSOLVER_PRECISE_Q

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        blas::caxpy_U(Linv_raw, r, qp);
#else
        // temporary hack - use AC to pass matrix arguments to multiblas
        for (int i = 0; i < nsrc; i++) {
          for (int j = i; j < nsrc; j++) { blas::caxpy(Linv(i, j), *r[i], *qp[j]); }
        }
#endif

#else // !BLOCKSOLVER_PRECISE_Q

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        for (int i = 0; i < nsrc; i++) { blas::copy(*tmpp[i], *r[i]); }
        // blas::copy(*tmpp, r); // Need to do this b/c r is fine, q is sloppy, can't caxpy w/ x fine, y sloppy.
        blas::caxpy_U(Linv_raw, tmpp, qp);
#else
        // temporary hack - use AC to pass matrix arguments to multiblas
        for (int i = 0; i < nsrc; i++) {
          blas::copy(tmpp[i], r[i]);
          for (int j = i; j < nsrc; j++) { blas::caxpy(Linv(i, j), tmpp[i], qp[j]); }
        }
#endif

#endif

        // Reliable updates step 9: Set S = C * C_old^{-1} (equation 6.1 in the blockCGrQ paper)
        // S = C * C_old.inverse();
#ifdef BLOCKSOLVER_VERBOSE
        std::cout << "reliable S " << S << std::endl;
#endif

        // Reliable updates step 10: Recompute residuals, reset rNorm, etc.
#ifdef BLOCKSOLVER_RELIABLE_POLICY_MIN
        rNorm = 1e30; // reliable update policy is to use the smallest residual.
#else
        rNorm = 0.0; // reliable update policy is to use the largest residual.
#endif
        allconverged = true;
        for (int i = 0; i < nsrc; i++) {
          converged[i] = convergence(H(i, i).real(), 0., stop[i], param.tol_hq);
          allconverged = allconverged && converged[i];
#ifdef BLOCKSOLVER_RELIABLE_POLICY_MIN
          if (rNorm > sqrt(H(i, i).real())) rNorm = sqrt(H(i, i).real());
#else
          if (rNorm < sqrt(H(i, i).real())) rNorm = sqrt(H(i, i).real());
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
      blas::hDotProduct(pTp_raw, qp, qp);
#else
      for (int i = 0; i < nsrc; i++) {
        for (int j = 0; j < nsrc; j++) { pTp(i, j) = blas::cDotProduct(qp[i], qp[j]); }
      }
#endif
      std::cout << " qTq " << std::endl << pTp << std::endl;
      std::cout << "QR" << S << std::endl << "QP " << S.inverse() * S << std::endl;
      ;
#endif

      // Step 28: P = Q + P S^\dagger
      // This would be best done with a cxpay,
      // but that's difficult for the same
      // reason a block cax is difficult.
      // Instead, we do it by a caxpyz + pointer swap.
      Sdagger = S.adjoint();
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      if (just_reliable_updated)
        blas::caxpyz(Sdagger_raw, pp, qp, tmpp); // tmp contains P.
      else
        blas::caxpyz_L(Sdagger_raw, pp, qp, tmpp); // tmp contains P.
#else
      // FIXME: check whether the implementation without BLOCKSOLVER_MULTIFUNCTIONS is correct
      if (just_reliable_updated) {
        // blas::caxpyz(Sdagger_raw,*pp,*qp,*tmpp); // tmp contains P.
        for (int j = 0; j < nsrc; j++) { blas::copy(*tmpp[j], *qp[j]); }
        for (int i = 0; i < nsrc; i++) {
          for (int j = 0; j < nsrc; j++) { blas::caxpy(Sdagger(i, j), *pp[i], *tmpp[j]);
          }
        }
      } else {
        // blas::caxpyz_L(Sdagger_raw,*pp,*qp,*tmpp); // tmp contains P.
        for (int j = 0; j < nsrc; j++) { blas::copy(*tmpp[j], *qp[j]); }
        for (int i = 0; i < nsrc; i++) {
          for (int j = 0; j <= i; j++) { blas::caxpy(Sdagger(i, j), *pp[i], *tmpp[j]); }
        }
      }
#endif

#ifdef BLOCKSOLVER_PRECISE_Q
      // Need to copy tmpp into p_oldp to reduce precision, then swap so p_oldp
      // actually holds the old p and pp actually holds the new one.
      // #ifdef BLOCKSOLVER_MULTIFUNCTIONS
      //       blas::copy(*p_oldp, *tmpp);
      // #else
      for (int i = 0; i < nsrc; i++) { blas::copy(*p_oldp[i], *tmpp[i]); }
      // #endif
      pp.swap(p_oldp);

#else
      pp.swap(tmpp); // now P contains P, tmp now contains P_old
#endif

      // Done with step 28.

// THESE HAVE NOT BEEN RE-WRITTEN FOR
// HIGH PRECISION Q YET.
#ifdef BLOCKSOLVER_EXPLICIT_QP_ORTHO
      if (did_reliable) {
        // Let's try to explicitly restore Q^\dagger P = I.
        Complex O_raw[nsrc * nsrc];
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        Map<MatrixBCG> O(O_raw, nsrc, nsrc);
        blas::cDotProduct(O_raw, qp, pp;
#else
        MatrixXcd O = MatrixXcd::Zero(nsrc, nsrc);
        for (int i = 0; i < nsrc; i++) {
          for (int j = 0; j < nsrc; j++) { O(i, j) = blas::cDotProduct(qp[i], pp[j]); }
        }
#endif

        printfQuda("Current Q^\\dagger P:\n");
        std::cout << O << "\n";

#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        O -= MatrixBCG::Identity(nsrc, nsrc);
#else
        O -= MatrixXcd::Identity(nsrc, nsrc);
#endif
        O = -O;
        std::cout << "BLAH\n" << O << "\n";
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        blas::caxpy(O_raw, *qp, *pp);
#else
        // temporary hack
        for (int i = 0; i < nsrc; i++) {
          for (int j = 0; j < nsrc; j++) { blas::caxpy(O(i, j), qp[i], pp[j]); }
        }
#endif

        // Check...
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
        blas::cDotProduct(O_raw, qp, pp);
#else
        for (int i = 0; i < nsrc; i++) {
          for (int j = 0; j < nsrc; j++) { O(i, j) = blas::cDotProduct(qp[i], pp[j]); }
        }
#endif
        printfQuda("Updated Q^\\dagger P:\n");
        std::cout << O << "\n";
      }
      // End test...
#endif // BLOCKSOLVER_EXPLICIT_QP_ORTHO

#ifdef BLOCKSOLVER_VERBOSE
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::hDotProduct(pTp_raw, pp, pp );
#else
      for (int i = 0; i < nsrc; i++) {
        for (int j = 0; j < nsrc; j++) { pTp(i, j) = blas::cDotProduct(pp[i], pp[j]); }
      }
#endif

      std::cout << " pTp " << std::endl << pTp << std::endl;
      std::cout << "S " << S << std::endl << "C " << C << std::endl;
#endif

      k++;
      PrintStats("Block-CG", k, r2avg / nsrc, b2avg, 0);
      // Step 29: update the convergence check. H will contain the right
      // thing whether or not we triggered a reliable update.
      allconverged = true;
      for (int i = 0; i < nsrc; i++) {
        converged[i] = convergence(H(i, i).real(), 0, stop[i], param.tol_hq);
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
    if (!just_reliable_updated) {
// #ifdef BLOCKSOLVER_MULTIFUNCTIONS
      // blas::caxpy(alpha_raw, p_oldp, x_sloppy);
// #else
      // temporary hack using AC
      for (int i = 0; i < nsrc; i++) {
        for (int j = 0; j < nsrc; j++) { blas::caxpy(alpha(i, j), *p_oldp[i], *x_sloppy[j]); }
      }
// #endif
    }

#else // !BLOCKSOLVER_PRECISE_Q
    // But remember tmpp holds the old P.
    if (!just_reliable_updated) {
#ifdef BLOCKSOLVER_MULTIFUNCTIONS
      blas::caxpy(alpha_raw, tmpp, x_sloppy);
#else
      // temporary hack using AC
      for (int i = 0; i < nsrc; i++) {
        for (int j = 0; j < nsrc; j++) { blas::caxpy(alpha(i, j), tmpp[i], x_sloppy[j]); }
      }
#endif
    }
#endif // BLOCKSOLVER_PRECISE_Q

    // We've converged!
    // Step 27: Update Xs into Y.
// #ifdef BLOCKSOLVER_MULTIFUNCTIONS
//     blas::xpy(x_sloppy, y);
// #else
    for (int i = 0; i < nsrc; i++) blas::xpy(*x_sloppy[i], *y[i]);
// #endif

    // And copy the final answer into X!
// #ifdef BLOCKSOLVER_MULTIFUNCTIONS
//     blas::copy(x, y);
// #else
    for (int i = 0; i < nsrc; i++) blas::copy(*x[i], *y[i]);
// #endif

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);

    double gflops = (blas::flops + mat.flops() + matSloppy.flops()) * 1e-9;
    param.gflops = gflops;
    param.iter += k;

    { // temporary addition for SC'17
      comm_allreduce(&gflops);
      printfQuda("Block-CG(%d): Convergence in %d iterations, %f seconds, GFLOPS = %g\n", nsrc, k, param.secs,
                 gflops / param.secs);
    }

    if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("Block-CG: ReliableUpdateInfo (delta,iter,rupdates): %f %i %i \n", delta, k, rUpdate);

    dslash::aux_worker = NULL;

    if (param.compute_true_res) {
    // compute the true residuals
// #ifdef BLOCKSOLVE_DSLASH5D
      // mat(r, x, y, tmp3);
// #else
      for (int i = 0; i < nsrc; i++) mat(*r[i], *x[i], *y[i], *tmp3[i]);
// #endif
      for (int i = 0; i < nsrc; i++) {
        param.true_res = sqrt(blas::xmyNorm(*b[i], *r[i]) / b2[i]);
        param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(*x[i], *r[i]).z);
        param.true_res_offset[i] = param.true_res;
        param.true_res_hq_offset[i] = param.true_res_hq;
      }
    }

    for (int i = 0; i < nsrc; i++) {
      std::stringstream str;
      str << "Block-CG " << i;
      PrintSummary(str.str().c_str(), k, H(i, i).real(), b2[i]);
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
    MultiRhsSolver(param, profile),
    mat(mat),
    matSloppy(matSloppy),
    // yp(nullptr),
    // rp(nullptr),
    // App(nullptr),
    // tmpp(nullptr),
    // x_sloppy_savedp(nullptr),
    // pp(nullptr),
    // qp(nullptr),
    // tmp_matsloppyp(nullptr),
    // p_oldp(nullptr),
    init(false)
  {
  }

  BlockCG::~BlockCG()
  {
    if (init) {
      // for (auto pi : p) delete pi;
      init = false;
      for (auto v : rp) delete v;
      rp.clear();
      for (auto v : yp) delete v;
      yp.clear();
      for (auto v : qp) delete v;
      qp.clear();
      for (auto v : tmpp) delete v;
      tmpp.clear();
      for (auto v : x_sloppy_savedp) delete v;
      x_sloppy_savedp.clear();
      for (auto v : pp) delete v;
      pp.clear();
      for (auto v : p_oldp) delete v;
      p_oldp.clear();
      for (auto v : App) delete v;
      App.clear();
      for (auto v : tmp_matsloppyp) delete v;
      tmp_matsloppyp.clear();
    }
  }
    void BlockCG::operator()(ColorSpinorFieldVector &x, ColorSpinorFieldVector &b)
    {

      if (param.num_src > QUDA_MAX_BLOCK_SRC)
        errorQuda("Requested number of right-hand sides %d exceeds max %d\n", param.num_src, QUDA_MAX_BLOCK_SRC);

      switch (param.num_src) {
      case 1: solve_n<1>(x, b); break;
      // case 2: solve_n<2>(x, b); break;
      // case 3: solve_n<3>(x, b); break;
      case 4: solve_n<4>(x, b); break;
      // case 5: solve_n<5>(x, b); break;
      // case 6: solve_n<6>(x, b); break;
      // case 7: solve_n<7>(x, b); break;
      case 8: solve_n<8>(x, b); break;
      // case 9: solve_n<9>(x, b); break;
      // case 10: solve_n<10>(x, b); break;
      // case 11: solve_n<11>(x, b); break;
      // case 12: solve_n<12>(x, b); break;
      // case 13: solve_n<13>(x, b); break;
      // case 14: solve_n<14>(x, b); break;
      // case 15: solve_n<15>(x, b); break;
      case 16: solve_n<16>(x, b); break;
      // case 24: solve_n<24>(x, b); break;
      case 32: solve_n<32>(x, b); break;
      // case 48: solve_n<48>(x, b); break;
      // case 64: solve_n<64>(x, b); break;
      default: errorQuda("Block-CG with dimension %d not supported", param.num_src);
      }
    }
  } // namespace quda
