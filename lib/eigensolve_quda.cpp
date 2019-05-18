#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>

#include <quda_internal.h>
#include <quda_arpack_interface.h>
#include <eigensolve_quda.h>
#include <qio_field.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

bool flags = true;

namespace quda
{

  using namespace Eigen;

  // Eigensolver class
  //-----------------------------------------------------------------------------
  EigenSolver::EigenSolver(QudaEigParam *eig_param, TimeProfile &profile) : eig_param(eig_param), profile(profile)
  {

    // Timings for components of the eigensolver
    time_ = 0.0;
    time_e = 0.0;   // time in Eigen
    time_mv = 0.0;  // time in matVec
    time_mb = 0.0;  // time in multiblas
    time_svd = 0.0; // time to compute SVD

    // Problem parameters
    nEv = eig_param->nEv;
    nKr = eig_param->nKr;
    nConv = eig_param->nConv;
    tol = eig_param->tol;
    reverse = false;

    // Algorithm parameters
    converged = false;
    num_converged = 0;
    restart_iter = 0;
    max_restarts = eig_param->max_restarts;
    check_interval = eig_param->check_interval;

    // Sanity checks
    if (nKr <= nEv) errorQuda("nKr=%d is less than or equal to nEv=%d\n", nKr, nEv);
    if (nEv < nConv) errorQuda("nConv=%d is greater than nEv=%d\n", nConv, nEv);
    if (nEv == 0) errorQuda("nEv=0 passed to Eigensolver\n");
    if (nKr == 0) errorQuda("nKr=0 passed to Eigensolver\n");
    if (nConv == 0) errorQuda("nConv=0 passed to Eigensolver\n");

    // Convergence information arrays
    residua = new double[nKr];
    residua_old = new double[nKr];
    locked = new bool[nKr];
    for (int i = 0; i < nKr; i++) {
      residua[nKr] = 0.0;
      residua_old[nKr] = 0.0;
      locked[i] = false;
    }

    // Quda MultiBLAS freindly array
    Qmat = new Complex[nEv * nKr];

    // Part of the spectrum to be computed.
    spectrum = strdup("SR"); // Initialsed to stop the compiler warning.

    if (eig_param->use_poly_acc) {
      if (eig_param->spectrum == QUDA_SR_EIG_SPECTRUM)
        spectrum = strdup("LR");
      else if (eig_param->spectrum == QUDA_LR_EIG_SPECTRUM)
        spectrum = strdup("SR");
      else if (eig_param->spectrum == QUDA_SM_EIG_SPECTRUM)
        spectrum = strdup("LM");
      else if (eig_param->spectrum == QUDA_LM_EIG_SPECTRUM)
        spectrum = strdup("SM");
      else if (eig_param->spectrum == QUDA_SI_EIG_SPECTRUM)
        spectrum = strdup("LI");
      else if (eig_param->spectrum == QUDA_LI_EIG_SPECTRUM)
        spectrum = strdup("SI");
    } else {
      if (eig_param->spectrum == QUDA_SR_EIG_SPECTRUM)
        spectrum = strdup("SR");
      else if (eig_param->spectrum == QUDA_LR_EIG_SPECTRUM)
        spectrum = strdup("LR");
      else if (eig_param->spectrum == QUDA_SM_EIG_SPECTRUM)
        spectrum = strdup("SM");
      else if (eig_param->spectrum == QUDA_LM_EIG_SPECTRUM)
        spectrum = strdup("LM");
      else if (eig_param->spectrum == QUDA_SI_EIG_SPECTRUM)
        spectrum = strdup("SI");
      else if (eig_param->spectrum == QUDA_LI_EIG_SPECTRUM)
        spectrum = strdup("LI");
    }

    // Deduce whether to reverse the sorting
    const char *L = "L";
    const char *S = "S";
    if (strncmp(L, spectrum, 1) == 0 && !eig_param->use_poly_acc) {
      reverse = true;
    } else if (strncmp(S, spectrum, 1) == 0 && eig_param->use_poly_acc) {
      reverse = true;
    } else if (strncmp(L, spectrum, 1) == 0 && eig_param->use_poly_acc) {
      reverse = true;
    }

    // Ensure Eigensolver has correct params
    printfQuda("nConv %d\n", nConv);
    printfQuda("nEv %d\n", nEv);
    printfQuda("nKr %d\n", nKr);
    if (eig_param->use_poly_acc) {
      printfQuda("polyDeg %d\n", eig_param->poly_deg);
      printfQuda("a-min %f\n", eig_param->a_min);
      printfQuda("a-max %f\n", eig_param->a_max);
    }
  }

  // We bake the matrix operator 'mat' and the eigensolver parameters into the
  // eigensolver.
  EigenSolver *EigenSolver::create(QudaEigParam *eig_param, const Dirac &mat, TimeProfile &profile)
  {

    EigenSolver *eig_solver = nullptr;

    switch (eig_param->eig_type) {
    case QUDA_IMP_RST_LANCZOS:
      printfQuda("Creating IRLM eigensolver\n");
      eig_solver = new IRLM(eig_param, mat, profile);
      printfQuda("Done creating IRLM eigensolver\n");
      break;
    case QUDA_IMP_RST_ARNOLDI:
      printfQuda("Creating IRAM eigensolver\n");
      errorQuda("IRAM eigensolver not implemented\n");
      break;
    case QUDA_THICK_RST_LANCZOS:
      printfQuda("Creating TRLM eigensolver\n");
      errorQuda("TRLM eigensolver not implemented\n");
      break;
    default: errorQuda("Invalid eig solver type");
    }
    return eig_solver;
  }

  // Utilities and functions common to all Eigensolver instances
  //------------------------------------------------------------------------------

  void EigenSolver::matVec(const Dirac &mat, ColorSpinorField &out, const ColorSpinorField &in)
  {

    if (eig_param->use_norm_op && eig_param->use_dagger) {
      mat.MMdag(out, in);
      return;
    } else if (eig_param->use_norm_op && !eig_param->use_dagger) {
      mat.MdagM(out, in);
      return;
    } else if (!eig_param->use_norm_op && eig_param->use_dagger) {
      mat.Mdag(out, in);
      return;
    } else {
      mat.M(out, in);
      return;
    }
  }

  void EigenSolver::chebyOp(const Dirac &mat, ColorSpinorField &out, const ColorSpinorField &in)
  {

    // Just do a simple matVec if no poly acc is requested
    if (!eig_param->use_poly_acc) {
      time_ = -clock();
      matVec(mat, out, in);
      time_ += clock();
      time_mv += time_;
      return;
    }

    if (eig_param->poly_deg == 0) { errorQuda("Polynomial acceleration requested with zero polynomial degree"); }

    // Compute the polynomial accelerated operator.
    double delta, theta;
    double sigma, sigma1, sigma_old;
    double d1, d2, d3;

    double a = eig_param->a_min;
    double b = eig_param->a_max;

    delta = (b - a) / 2.0;
    theta = (b + a) / 2.0;

    sigma1 = -delta / theta;

    d1 = sigma1 / delta;
    d2 = 1.0;

    // out = d2 * in + d1 * out
    // C_1(x) = x
    time_ = -clock();
    matVec(mat, out, in);
    time_ += clock();
    time_mv += time_;

    time_ = -clock();
    blas::caxpby(d2, const_cast<ColorSpinorField &>(in), d1, out);
    if (eig_param->poly_deg == 1) return;

    // C_0 is the current 'in'  vector.
    // C_1 is the current 'out' vector.

    // Clone 'in' to two temporary vectors.
    ColorSpinorField *tmp1 = ColorSpinorField::Create(in);
    ColorSpinorField *tmp2 = ColorSpinorField::Create(in);

    blas::copy(*tmp1, in);
    blas::copy(*tmp2, out);
    time_ += clock();
    time_mb += time_;

    // Using Chebyshev polynomial recursion relation,
    // C_{m+1}(x) = 2*x*C_{m} - C_{m-1}

    sigma_old = sigma1;

    // construct C_{m+1}(x)
    for (int i = 2; i < eig_param->poly_deg; i++) {

      sigma = 1.0 / (2.0 / sigma1 - sigma_old);

      d1 = 2.0 * sigma / delta;
      d2 = -d1 * theta;
      d3 = -sigma * sigma_old;

      // mat*C_{m}(x)
      time_ = -clock();
      matVec(mat, out, *tmp2);
      time_ += clock();
      time_mv += time_;

      time_ = -clock();
      blas::ax(d3, *tmp1);
      Complex d1c(d1, 0.0);
      Complex d2c(d2, 0.0);
      blas::cxpaypbz(*tmp1, d2c, *tmp2, d1c, out);

      blas::copy(*tmp1, *tmp2);
      blas::copy(*tmp2, out);
      time_ += clock();
      time_mb += time_;

      sigma_old = sigma;
    }

    delete tmp1;
    delete tmp2;
  }

  // Orthogonalise r against V_[j]
  Complex EigenSolver::orthogonalise(std::vector<ColorSpinorField *> vecs, std::vector<ColorSpinorField *> rvec, int j)
  {
    time_ = -clock();
    Complex s(0.0, 0.0);
    Complex sum(0.0, 0.0);
    for (int i = 0; i < j; i++) {
      s = blas::cDotProduct(*vecs[i], *rvec[0]);
      sum += s;
      blas::caxpy(-s, *vecs[i], *rvec[0]);
    }
    time_ += clock();
    time_mb += time_;
    return sum;
  }

  // Orthogonalise r against V_[j]
  Complex EigenSolver::blockOrthogonalise(std::vector<ColorSpinorField *> vecs, std::vector<ColorSpinorField *> rvec,
                                          int j)
  {
    time_ = -clock();
    Complex *s = new Complex[j + 1];
    Complex sum(0.0, 0.0);
    std::vector<ColorSpinorField *> vecs_ptr;
    for (int i = 0; i < j + 1; i++) { vecs_ptr.push_back(vecs[i]); }
    // Block dot products stored in s.
    blas::cDotProduct(s, vecs_ptr, rvec);

    // Block orthogonalise
    for (int i = 0; i < j + 1; i++) {
      sum += s[i];
      s[i] *= -1.0;
    }
    blas::caxpy(s, vecs_ptr, rvec);

    delete s;
    time_ += clock();
    time_mb += time_;
    return sum;
  }

  // Deflate vec, place result in vec_defl
  void EigenSolver::deflate(std::vector<ColorSpinorField *> vec_defl, std::vector<ColorSpinorField *> vec,
                            std::vector<ColorSpinorField *> eig_vecs, std::vector<Complex> evals)
  {

    // number of evecs
    int n_defl = eig_param->nEv;

    printfQuda("Deflating %d vectors\n", n_defl);

    // Perform Sum_i V_i * (L_i)^{-1} * (V_i)^dag * vec = vec_defl
    // for all i computed eigenvectors and values.

    // Pointers to the required Krylov space vectors,
    // no extra memory is allocated.
    std::vector<ColorSpinorField *> eig_vecs_ptr;
    for (int i = 0; i < n_defl; i++) eig_vecs_ptr.push_back(eig_vecs[i]);

    // 1. Take block inner product: (V_i)^dag * vec = A_i
    Complex *s = new Complex[n_defl];
    blas::cDotProduct(s, eig_vecs_ptr, vec);

    // 2. Perform block caxpy: V_i * (L_i)^{-1} * A_i
    for (int i = 0; i < n_defl; i++) { s[i] /= evals[i].real(); }

    // 3. Accumulate sum vec_defl = Sum_i V_i * (L_i)^{-1} * A_i
    blas::zero(*vec_defl[0]);
    blas::caxpy(s, eig_vecs_ptr, vec_defl);

    // Orthonormality check in deflation.
    for (int i = 0; i < n_defl; i++)
      for (int j = 0; j <= i; j++) {
        // printfQuda("%d %d %.6e \n", i, j, blas::cDotProduct(*eig_vecs_ptr[i], *eig_vecs_ptr[j]).real() );
      }
  }

  void EigenSolver::computeEvals(const Dirac &mat, std::vector<Complex> &evals, int size)
  {

    for (int i = 0; i < size; i++) {

      // r = A * v_i
      time_ = -clock();
      matVec(mat, *r[0], *d_vecs_tmp[i]);
      time_ += clock();
      time_mv += time_;

      time_ = -clock();
      // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
      evals[i] = blas::cDotProduct(*d_vecs_tmp[i], *r[0]) / sqrt(blas::norm2(*d_vecs_tmp[i]));

      // Measure ||lambda_i*v_i - A*v_i||
      Complex n_unit(-1.0, 0.0);
      blas::caxpby(evals[i], *d_vecs_tmp[i], n_unit, *r[0]);
      residua[i] = sqrt(blas::norm2(*r[0]));
      time_ += clock();
      time_mb += time_;
    }
  }

  void EigenSolver::basisRotateQ(std::vector<ColorSpinorField *> &kSpace, int k)
  {

    time_ = -clock();

    std::vector<ColorSpinorField *> d_vecs_ptr;
    for (int i = 0; i < k; i++) {
      blas::zero(*d_vecs_tmp[i]);
      d_vecs_ptr.push_back(d_vecs_tmp[i]);
    }

    blas::caxpy(Qmat, kSpace, d_vecs_ptr);
    // Copy back to Krylov space array
    for (int i = 0; i < k; i++) { *kSpace[i] = *d_vecs_tmp[i]; }
    time_ += clock();
    time_mb += time_;
  }

  void EigenSolver::loadVectors(std::vector<ColorSpinorField *> &eig_vecs, std::string vec_infile)
  {

    //profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    //profile.TPSTART(QUDA_PROFILE_IO);

#ifdef HAVE_QIO
    const int Nvec = eig_vecs.size();
    if (strcmp(vec_infile.c_str(), "") != 0) {
      printfQuda("Start loading %04d vectors from %s\n", Nvec, vec_infile.c_str());

      std::vector<ColorSpinorField *> tmp;
      if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
        ColorSpinorParam csParam(*eig_vecs[0]);
        csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        csParam.setPrecision(eig_vecs[0]->Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION :
                                                                                eig_vecs[0]->Precision());
        csParam.location = QUDA_CPU_FIELD_LOCATION;
        csParam.create = QUDA_NULL_FIELD_CREATE;
        for (int i = 0; i < Nvec; i++) { tmp.push_back(ColorSpinorField::Create(csParam)); }
      } else {
        for (int i = 0; i < Nvec; i++) { tmp.push_back(eig_vecs[i]); }
      }

      void **V = static_cast<void **>(safe_malloc(Nvec * sizeof(void *)));
      for (int i = 0; i < Nvec; i++) {
        V[i] = tmp[i]->V();
        if (V[i] == NULL) { printfQuda("Could not allocate space for eigenVector[%d]\n", i); }
      }

      read_spinor_field(vec_infile.c_str(), &V[0], eig_vecs[0]->Precision(), eig_vecs[0]->X(), eig_vecs[0]->Ncolor(),
                        eig_vecs[0]->Nspin(), Nvec, 0, (char **)0);
      host_free(V);
      if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
        for (int i = 0; i < Nvec; i++) {
          *eig_vecs[i] = *tmp[i];
          delete tmp[i];
        }
      }

      printfQuda("Done loading vectors\n");
    } else {
      errorQuda("No eigenspace input file defined.");
    }
#else
    errorQuda("\nQIO library was not built.\n");
#endif
    //profile.TPSTOP(QUDA_PROFILE_IO);
    //profile.TPSTART(QUDA_PROFILE_COMPUTE);

    return;
  }

  void EigenSolver::saveVectors(std::vector<ColorSpinorField *> &eig_vecs, std::string vec_outfile)
  {
    
    //profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    //profile.TPSTART(QUDA_PROFILE_IO);
    
#ifdef HAVE_QIO
    const int Nvec = eig_vecs.size();
    std::vector<ColorSpinorField *> tmp;
    if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
      ColorSpinorParam csParam(*eig_vecs[0]);
      csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      csParam.setPrecision(eig_vecs[0]->Precision() < QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION :
                                                                              eig_vecs[0]->Precision());
      csParam.location = QUDA_CPU_FIELD_LOCATION;
      csParam.create = QUDA_NULL_FIELD_CREATE;
      for (int i = 0; i < Nvec; i++) {
        tmp.push_back(ColorSpinorField::Create(csParam));
        *tmp[i] = *eig_vecs[i];
      }
    } else {
      for (int i = 0; i < Nvec; i++) { tmp.push_back(eig_vecs[i]); }
    }

    printfQuda("Start saving %d vectors to %s\n", Nvec, vec_outfile.c_str());

    void **V = static_cast<void **>(safe_malloc(Nvec * sizeof(void *)));
    for (int i = 0; i < Nvec; i++) {
      V[i] = tmp[i]->V();
      if (V[i] == NULL) { printfQuda("Could not allocate space for eigenVector[%04d]\n", i); }
    }

    write_spinor_field(vec_outfile.c_str(), &V[0], eig_vecs[0]->Precision(), eig_vecs[0]->X(), eig_vecs[0]->Ncolor(),
                       eig_vecs[0]->Nspin(), Nvec, 0, (char **)0);

    host_free(V);
    printfQuda("Done saving vectors\n");
    if (eig_vecs[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
      for (int i = 0; i < Nvec; i++) delete tmp[i];
    }

#else
    errorQuda("\nQIO library was not built.\n");
#endif
    //profile.TPSTOP(QUDA_PROFILE_IO);
    //profile.TPSTART(QUDA_PROFILE_COMPUTE);

    return;
  }

  void EigenSolver::loadFromFile(const Dirac &mat, std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {

    printfQuda("Loading eigenvectors\n");
    loadVectors(kSpace, eig_param->vec_infile);

    // Create the device side residual vector by cloning
    // the kSpace passed to the function.
    ColorSpinorParam csParam(*kSpace[0]);
    r.push_back(ColorSpinorField::Create(csParam));

    // Error estimates (residua) given by ||A*vec - lambda*vec||
    computeEvals(mat, evals, nEv);
    for (int i = 0; i < nEv; i++) {
      printfQuda("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals[i].real(), evals[i].imag(), residua[i]);
    }

    delete r[0];
    return;
  }

  EigenSolver::~EigenSolver() {}
  //-----------------------------------------------------------------------------
  //-----------------------------------------------------------------------------

  // Implicitly Restarted Lanczos Method constructor
  IRLM::IRLM(QudaEigParam *eig_param, const Dirac &mat, TimeProfile &profile) :
    EigenSolver(eig_param, profile),
    mat(mat)
  {

    // Tridiagonal matrix
    alpha = new double[nKr];
    beta = new double[nKr];
    for (int i = 0; i < nKr; i++) {
      alpha[i] = 0.0;
      beta[i] = 0.0;
    }
  }

  void IRLM::operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {

    // Check to see if we are loading eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      loadFromFile(mat, kSpace, evals);
      return;
    }

    // Test for an initial guess
    double norm = sqrt(blas::norm2(*kSpace[0]));
    if (norm == 0) {
      printfQuda("Initial residual is zero. Populating with rands.\n");

      if (kSpace[0]->Location() == QUDA_CPU_FIELD_LOCATION) {
        kSpace[0]->Source(QUDA_RANDOM_SOURCE);
      } else {
        RNG *rng = new RNG(kSpace[0]->Volume(), 1234, kSpace[0]->X());
        rng->Init();
        spinorNoise(*kSpace[0], *rng, QUDA_NOISE_UNIFORM);
        rng->Release();
        delete rng;
      }
    }

    // Clone the kSpace params passed to the function.
    ColorSpinorParam csParam(*kSpace[0]);

    // r is a device side residual vector.
    r.push_back(ColorSpinorField::Create(csParam));
    // d_vecs_tmp are device side space for Lanczos Ritz vector basis rotation.
    for (int i = 0; i < nKr; i++) { d_vecs_tmp.push_back(ColorSpinorField::Create(csParam)); }

    // Normalise initial guess
    norm = sqrt(blas::norm2(*kSpace[0]));
    blas::ax(1.0 / norm, *kSpace[0]);
    //---------------------------------------------------------------------------

    // Begin Eigensolver computation
    //---------------------------------------------------------------------------
    printfQuda("*****************************\n");
    printfQuda("**** START IRLM SOLUTION ****\n");
    printfQuda("*****************************\n");

    double t1 = clock();

    // Define some literature consistent parameters
    int k = nEv;
    int m = nKr;
    int p = m - k;

    // Initial k step factorisation
    int step = 0;
    int iter = 0;
    for (step = 0; step < nEv; step++) lanczosStep(kSpace, step);
    iter += nEv;
    printfQuda("Initial %d step factorisation complete\n", nEv);

    // Loop over restart iterations.
    while (restart_iter < max_restarts && !converged) {

      for (step = k; step < m; step++) lanczosStep(kSpace, step);
      iter += p;
      printfQuda("Restart %d complete\n", restart_iter + 1);

      // alpha and beta will be updated in this function. The Q
      // matrix is put in Qmat for use in multiBLAS functions.
      computeQRfromTridiag(k);

      // Puts the rotated vectors into d_vecs_tmp.
      // Number of rotated vectors is k.
      basisRotateQ(kSpace, k);

      // Store the residual vector in the last vector of kSpace
      *kSpace[m - 1] = *r[0];

      // Convergence check
      if ((restart_iter + 1) % check_interval == 0) {

        // Constructs the rotation matrix from the tridiagonal
        // and puts the nEv rotated vectors into d_vecs_tmp.
        basisRotateTriDiag(kSpace, k);

        // Error estimates (residua) given by ||A*vec - lambda*vec||
        computeEvals(mat, evals, k);

        // Lock new eigenpairs
        for (int i = 0; i < k; i++) {
          if (residua[i] < tol && !locked[i]) {

            // This is a new eigenpair, lock it
            locked[i] = true;
            printfQuda("**** locking vector %d, converged eigenvalue = (%+.16e,%+.16e) resid %+.16e ****\n", i,
                       evals[i].real(), evals[i].imag(), residua[i]);

            num_converged++;
          }
        }

        printfQuda("%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter + 1);

        // Check for convergence
        bool all_locked = true;
        if (num_converged >= nConv) {
          for (int i = 0; i < nConv; i++) {
            if (reverse) {
              if (!locked[k - 1 - i]) all_locked = false;
            } else {
              if (!locked[i]) all_locked = false;
            }
          }

          if (all_locked) {
            // Transfer eigenvectors to kSpace array and exit IRLM
            for (int i = 0; i < nConv; i++) *kSpace[i] = *d_vecs_tmp[i];
            converged = true;
          }
        }
      }

      // Prepare the residual (located in kSpace[nKr]) for the next iteration
      time_ = -clock();
      // r_{k} = r_{m} * \sigma_{k}  |  \sigma_{k} = Q(m,k)
      blas::ax(Qmat[m * k - 1].real(), *kSpace[m - 1]);
      // blas::ax(sigma_k.real(), *kSpace[m-1]);
      // r_{k} += v_{k+1} * beta_{k}  |  beta_{k} = Tm(k+1,k)
      blas::axpy(beta[k - 1], *kSpace[k], *kSpace[m - 1]);

      // Construct the new starting vector
      // k = nEv + num_converged;
      double res_norm = sqrt(blas::norm2(*kSpace[m - 1]));
      blas::zero(*kSpace[k]);
      blas::axpy(1.0 / res_norm, *kSpace[m - 1], *kSpace[k]);
      beta[k - 1] = res_norm;
      time_ += clock();
      time_mb += time_;

      restart_iter++;
    }
    //---------------------------------------------------------------------------

    // Post computation report
    //---------------------------------------------------------------------------
    if (!converged) {
      printfQuda("IRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
                 "restart steps. "
                 "Please either increase nKr and nEv and/or extend the number of restarts\n",
                 nConv, nEv, nKr, max_restarts);
    } else {
      printfQuda("IRLM computed the requested %d vectors in %d restart steps and %d OP*x operations.\n", nConv,
                 restart_iter, iter);

      int idx = 0;
      for (int i = 0; i < nEv; i++) {
        if (reverse) idx = nEv - 1 - i;
        printfQuda("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[idx], 0.0, beta[idx]);
      }

      for (int i = 0; i < nEv; i++) {
        if (reverse) idx = nEv - 1 - i;
        printfQuda("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals[idx].real(), evals[idx].imag(),
                   residua[idx]);
      }

      // Compute SVD
      time_svd = -clock();
      if (eig_param->compute_svd) computeSVD(kSpace, d_vecs_tmp, evals, reverse);
      time_svd += clock();
    }

    double t2 = clock() - t1;
    double total;

    if (eig_param->compute_svd)
      total = (time_e + time_mv + time_mb + time_svd) / CLOCKS_PER_SEC;
    else
      total = (time_e + time_mv + time_mb) / CLOCKS_PER_SEC;
    printfQuda("Time to solve problem using IRLM = %e\n", t2 / CLOCKS_PER_SEC);
    printfQuda("Time spent using EIGEN           = %e  %.1f%%\n", time_e / CLOCKS_PER_SEC,
               100 * (time_e / CLOCKS_PER_SEC) / total);
    printfQuda("Time spent in matVec             = %e  %.1f%%\n", time_mv / CLOCKS_PER_SEC,
               100 * (time_mv / CLOCKS_PER_SEC) / total);
    printfQuda("Time spent in (multi)blas        = %e  %.1f%%\n", time_mb / CLOCKS_PER_SEC,
               100 * (time_mb / CLOCKS_PER_SEC) / total);
    if (eig_param->compute_svd)
      printfQuda("Time spent computing svd         = %e  %.1f%%\n", time_svd / CLOCKS_PER_SEC,
                 100 * (time_svd / CLOCKS_PER_SEC) / total);
    //---------------------------------------------------------------------------

    // Local clean-up
    delete r[0];
    for (int i = 0; i < nKr; i++) { delete d_vecs_tmp[i]; }

    // Only save if outfile is defined
    if (strcmp(eig_param->vec_outfile, "") != 0) {
      printfQuda("saving eigenvectors\n");
      saveVectors(kSpace, eig_param->vec_outfile);
    }

    printfQuda("*****************************\n");
    printfQuda("***** END IRLM SOLUTION *****\n");
    printfQuda("*****************************\n");
  }

  // Lanczos Member functions
  //---------------------------------------------------------------------------
  void IRLM::lanczosStep(std::vector<ColorSpinorField *> v, int j)
  {

    // Compute r = A * v_j - b_{j-i} * v_{j-1}
    // r = A * v_j

    time_ = -clock();
    // Purge locked eigenvectors from the residual
    // DMH: This step ensures that the next mat vec
    //     does not reintroduce components of the
    //     existing eigenspace

    int locked_size = 0;
    std::vector<ColorSpinorField *> d_vecs_locked;
    for (int i = 0; i < j; i++) {
      if (locked[i]) {
        d_vecs_locked.push_back(v[i]);
        locked_size++;
      }
    }
    if (locked_size > 0) {
      Complex *s = new Complex[locked_size];
      // Block dot products stored in s.
      std::vector<ColorSpinorField *> d_vec_vj;
      d_vec_vj.push_back(v[j]);
      blas::cDotProduct(s, d_vecs_locked, d_vec_vj);
      for (int i = 0; i < locked_size; i++) s[i] *= -1.0;
      blas::caxpy(s, d_vecs_locked, d_vec_vj);
      delete s;
    }

    time_ += clock();
    time_mb += time_;

    chebyOp(mat, *r[0], *v[j]);

    // Deflate locked eigenvectors from the residual
    // DMH: This step ensures that the new Krylov vector
    //     remains orthogonal to the found eigenspace
    //     We may resuse the pointers to locked eigenvectors.
    if (locked_size > 0) {
      Complex *s = new Complex[locked_size];
      // Block dot products stored in s.
      std::vector<ColorSpinorField *> d_vec_vj;
      d_vec_vj.push_back(r[0]);
      blas::cDotProduct(s, d_vecs_locked, d_vec_vj);
      for (int i = 0; i < locked_size; i++) s[i] *= -1.0;
      blas::caxpy(s, d_vecs_locked, d_vec_vj);
      delete s;
    }

    // a_j = v_j^dag * r
    alpha[j] = blas::reDotProduct(*v[j], *r[0]);

    // r = r - a_j * v_j
    blas::axpy(-alpha[j], *v[j], *r[0]);

    if (j > 0) {

      // r = r - b_{j-1} * v_{j-1}
      blas::axpy(-beta[j - 1], *v[j - 1], *r[0]);

      // Orthogonalise r against the Krylov space
      blockOrthogonalise(v, r, j);
    }

    // b_j = ||r||
    beta[j] = sqrt(blas::norm2(*r[0]));

    // Prepare next step.
    // v_{j+1} = r / b_j
    if (j < nKr - 1) {
      blas::zero(*v[j + 1]);
      blas::axpy(1.0 / beta[j], *r[0], *v[j + 1]);
    } else
      beta_m = beta[j];
  }

  void IRLM::computeQRfromTridiag(int k)
  {

    // Switch notation
    int m = nKr;
    int p = m - k;

    MatrixXd triDiag = MatrixXd::Zero(m, m);
    MatrixXd Q = MatrixXd::Identity(m, m);
    MatrixXd Qj = MatrixXd::Zero(m, m);
    MatrixXd TmMuI = MatrixXd::Zero(m, m);
    MatrixXd Id = MatrixXd::Identity(m, m);
    SelfAdjointEigenSolver<MatrixXd> eigensolver;

    time_ = -clock();

    // Construct Tridiagonal matrix T_{k,k}
    for (int i = 0; i < m; i++) {
      triDiag(i, i) = alpha[i];
      if (i < m - 1) {
        triDiag(i + 1, i) = beta[i];
        triDiag(i, i + 1) = beta[i];
      }
    }

    // Eigensolve the T_k matrix
    Tridiagonalization<MatrixXd> TD(triDiag);
    eigensolver.computeFromTridiagonal(TD.diagonal(), TD.subDiagonal());

    // Dump entire spectrum
    for (int i = 0; i < m; i++) {
      residua_old[i] = residua[i];
      residua[i] = beta_m * fabs(eigensolver.eigenvectors().col(i)[m - 1]);
    }
    // DMH: This is where the LR/SR spectrum is projected.
    //     For SR, we project out the (nKr - nEv) LARGEST eigenvalues
    //     For LR, we project out the (nKr - nEv) SMALLEST eigenvalues
    //     Eigen returns Eigenvalues from Hermitian matrices in ascending order.
    //     We take advantage of this here, but be mindful that Arnoldi
    //     type solves will require some degree of sorting.

    // Sort the eigenvalues in real ascending order
    double shifts[p];
    for (int j = 0; j < p; j++) {
      if (reverse)
        shifts[j] = eigensolver.eigenvalues()[j];
      else
        shifts[j] = eigensolver.eigenvalues()[j + k];
    }

    // DMH: This can be batch parallelised on the GPU.
    //     Will be useful when (nKr - nEv) is large.
    //     The N lots of QR decomp can be batched, and
    //     the Q_0 * Q_1 * ... * Q_{N-1} product
    //     can be done in pairs in parallel, like
    //     a sum reduction.
    //     (Q_0 * Q_1) * (Q_2 * Q_3) * ... * (Q_{N-2} * Q_{N-1})
    //     =  (Q_{0*1} * Q_{2*3}) * ... * (Q_{(N-2)(N-1)})
    //     ...
    //     =  Q_{0*1*2*3*...*(N-2)(N-1)}

    if (flags) printfQuda("Computing QR\n");

    for (int j = 0; j < p; j++) {

      // Apply the shift \mu_j
      TmMuI = triDiag;
      // Id is the identity matrix
      TmMuI -= shifts[j] * Id;

      // QR decomposition of Tm - \mu_i I
      HouseholderQR<MatrixXd> QR(TmMuI);

      // Retain Qj matrices as a product.
      Qj = QR.householderQ();
      Q = Q * Qj;

      // Update the Tridiag
      triDiag = Qj.adjoint() * triDiag;
      triDiag = triDiag * Qj;
    }

    // Place Q data in accessible arrays
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < m; j++) {
        Qmat[j * k + i].real(Q.col(i)[j]);
        Qmat[j * k + i].imag(0.0);
      }
    }

    // Update the triDiag
    for (int i = 0; i < m; i++) {
      alpha[i] = triDiag(i, i);
      if (i < m - 1) beta[i] = triDiag(i, i + 1);
    }

    time_ += clock();
    time_e += time_;

    if (flags) printfQuda("ComputeQR Complete\n");
  }

  void IRLM::basisRotateTriDiag(std::vector<ColorSpinorField *> &kSpace, int k)
  {

    // Eigen objects
    MatrixXd triDiag = MatrixXd::Identity(k, k);
    SelfAdjointEigenSolver<MatrixXd> eigensolver;
    Complex *Rmat_k = new Complex[k * k];

    time_ = -clock();

    // Construct the updated tridiag for the convergence check
    for (int i = 0; i < k; i++) {
      triDiag(i, i) = alpha[i];
      if (i < k - 1) {
        triDiag(i + 1, i) = beta[i];
        triDiag(i, i + 1) = beta[i];
      }
    }
    Tridiagonalization<MatrixXd> TD(triDiag);
    eigensolver.computeFromTridiagonal(TD.diagonal(), TD.subDiagonal());

    // Place data in accessible array
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < k; j++) {
        Rmat_k[j * k + i].real(eigensolver.eigenvectors().col(i)[j]);
        Rmat_k[j * k + i].imag(0.0);
      }
    }
    time_ += clock();
    time_e += time_;

    // Basis rotation
    // Pointers to the required Krylov space vectors,
    // no extra memory is allocated.
    time_ = -clock();
    std::vector<ColorSpinorField *> kSpace_ptr;
    std::vector<ColorSpinorField *> d_vecs_ptr;
    for (int i = 0; i < k; i++) {
      blas::zero(*d_vecs_tmp[i]);
      d_vecs_ptr.push_back(d_vecs_tmp[i]);
      kSpace_ptr.push_back(kSpace[i]);
    }

    blas::caxpy(Rmat_k, kSpace_ptr, d_vecs_ptr);
    time_ += clock();
    time_mb += time_;
  }

  void IRLM::computeSVD(std::vector<ColorSpinorField *> &kSpace, std::vector<ColorSpinorField *> &evecs,
                        std::vector<Complex> &evals, bool reverse)
  {

    printfQuda("Computing SVD of M%s\n", eig_param->use_dagger ? "dag" : "");

    // Switch to M (or Mdag) mat vec
    eig_param->use_norm_op = QUDA_BOOLEAN_NO;
    int nConv = eig_param->nConv;
    int nEv = eig_param->nEv;

    Complex sigma_tmp[nConv / 2];

    // Create a device side temp vector by cloning
    // the evecs passed to the function. We create a
    // vector of one element. In future, we may wish to
    // make a block eigensolver, so an std::vector structure
    // will be needed.

    std::vector<ColorSpinorField *> tmp;
    ColorSpinorParam csParam(*evecs[0]);
    tmp.push_back(ColorSpinorField::Create(csParam));

    for (int i = 0; i < nConv / 2; i++) {

      int idx = i;
      // If we used poly acc, the eigenpairs are in descending order.
      if (reverse) idx = nEv - 1 - i;

      // This function assumes that you have computed the eigenvectors
      // of MdagM, ie, the right SVD of M. The ith eigen vector in the array corresponds
      // to the ith right singular vector. We will sort the array as:
      //
      //     EV_array_MdagM = (Rev_0, Rev_1, ... , Rev_{n-1{)
      // to  SVD_array_M    = (Rsv_0, Lsv_0, Rsv_1, Lsv_1, ... ,
      //                       Rsv_{nEv/2-1}, Lsv_{nEv/2-1})
      //
      // We start at Rev_(n/2-1), compute Lsv_(n/2-1), then move the vectors
      // to the n-2 and n-1 positions in the array respectively.

      // As a cross check, we recompute the singular values from mat vecs rather
      // than make the direct relation (sigma_i)^2 = |lambda_i|
      //--------------------------------------------------------------------------
      Complex lambda = evals[idx];

      // M*Rev_i = M*Rsv_i = sigma_i Lsv_i
      matVec(mat, *tmp[0], *evecs[idx]);

      // sigma_i = sqrt(sigma_i (Lsv_i)^dag * sigma_i * Lsv_i )
      Complex sigma_sq = blas::cDotProduct(*tmp[0], *tmp[0]);
      sigma_tmp[i] = Complex(sqrt(sigma_sq.real()), sqrt(abs(sigma_sq.imag())));

      // Copy device Rsv[i] (EV[i]) to host SVD[2*i]
      *kSpace[2 * i] = *evecs[idx];

      // Normalise the Lsv: sigma_i Lsv_i -> Lsv_i
      double norm = sqrt(blas::norm2(*tmp[0]));
      blas::ax(1.0 / norm, *tmp[0]);
      // Copy Lsv[i] to SVD[2*i+1]
      *kSpace[2 * i + 1] = *evecs[idx];

      printfQuda("Sval[%04d] = %+.16e  %+.16e   sigma - sqrt(|lambda|) = %+.16e\n", i, sigma_tmp[i].real(),
                 sigma_tmp[i].imag(), sigma_tmp[i].real() - sqrt(abs(lambda.real())));
      //--------------------------------------------------------------------------
    }

    // Update the host evals array
    for (int i = 0; i < nConv / 2; i++) {
      evals[2 * i + 0] = sigma_tmp[i];
      evals[2 * i + 1] = sigma_tmp[i];
    }

    // Revert to MdagM (or MMdag) mat vec
    eig_param->use_norm_op = QUDA_BOOLEAN_YES;

    delete tmp[0];
  }

  // Destructor
  IRLM::~IRLM()
  {
    delete alpha;
    delete beta;
  }

  // ARPACK INTERAFCE ROUTINES
  //--------------------------------------------------------------------------

#ifdef ARPACK_LIB

  void arpackErrorHelpNAUPD();
  void arpackErrorHelpNEUPD();

#if (defined(QMP_COMMS) || defined(MPI_COMMS))
#include <mpi.h>
#endif

  void arpack_solve(void *h_evecs, void *h_evals, const Dirac &mat, QudaEigParam *eig_param, ColorSpinorParam *cpuParam)
  {

    // ARPACK logfile name
    char *arpack_logfile = eig_param->arpack_logfile;
    printfQuda("**** START ARPACK SOLUTION ****\n");
    printfQuda("Output directed to %s\n", arpack_logfile);

    // Create Eigensolver object for member function use
    TimeProfile profile("Dummy");
    EigenSolver *eig_solver = EigenSolver::create(eig_param, mat, profile);

    // Construct parameters and memory allocation
    //---------------------------------------------------------------------------------
    double time_ar = 0.0; // time in ARPACK
    double time_mv = 0.0; // time in QUDA mat vec + data transfer
    double time_ev = 0.0; // time in computing Eigenvectors

    // MPI objects
    int *fcomm_ = nullptr;
#if (defined(QMP_COMMS) || defined(MPI_COMMS))
    MPI_Fint mpi_comm_fort = MPI_Comm_c2f(MPI_COMM_WORLD);
    fcomm_ = static_cast<int *>(&mpi_comm_fort);
#endif

    // Determine local volume for memory allocations
    int local_dim[4];
    int local_vol = 1;
    for (int i = 0; i < 4; i++) {
      local_dim[i] = cpuParam->x[i];
      local_vol *= local_dim[i];
    }

    int nSpin = (eig_param->invert_param->dslash_type == QUDA_LAPLACE_DSLASH) ? 1 : 4;

    // all FORTRAN communication uses underscored
    int ido_ = 0;
    int info_ = 1; // if 0, use random vector. If 1, initial residual lives in resid_
    int *ipntr_ = (int *)malloc(14 * sizeof(int));
    int *iparam_ = (int *)malloc(11 * sizeof(int));
    int n_ = local_vol * nSpin * 3, nEv_ = eig_param->nEv, nKr_ = eig_param->nKr, ldv_ = local_vol * nSpin * 3,
        lworkl_ = (3 * nKr_ * nKr_ + 5 * nKr_) * 2, rvec_ = 1;
    int max_iter = eig_param->max_restarts * (nKr_ - nEv_) + nEv_;
    int *h_evals_sorted_idx = (int *)malloc(nKr_ * sizeof(int));

    // Assign values to ARPACK params
    iparam_[0] = 1;
    iparam_[2] = max_iter;
    iparam_[3] = 1;
    iparam_[6] = 1;

    // ARPACK problem type to be solved
    char howmny = 'P';
    char bmat = 'I';
    char *spectrum;
    spectrum = strdup("SR"); // Initialsed just to stop the compiler warning...

    if (eig_param->use_poly_acc) {
      if (eig_param->spectrum == QUDA_SR_EIG_SPECTRUM)
        spectrum = strdup("LR");
      else if (eig_param->spectrum == QUDA_LR_EIG_SPECTRUM)
        spectrum = strdup("SR");
      else if (eig_param->spectrum == QUDA_SM_EIG_SPECTRUM)
        spectrum = strdup("LM");
      else if (eig_param->spectrum == QUDA_LM_EIG_SPECTRUM)
        spectrum = strdup("SM");
      else if (eig_param->spectrum == QUDA_SI_EIG_SPECTRUM)
        spectrum = strdup("LI");
      else if (eig_param->spectrum == QUDA_LI_EIG_SPECTRUM)
        spectrum = strdup("SI");
    } else {
      if (eig_param->spectrum == QUDA_SR_EIG_SPECTRUM)
        spectrum = strdup("SR");
      else if (eig_param->spectrum == QUDA_LR_EIG_SPECTRUM)
        spectrum = strdup("LR");
      else if (eig_param->spectrum == QUDA_SM_EIG_SPECTRUM)
        spectrum = strdup("SM");
      else if (eig_param->spectrum == QUDA_LM_EIG_SPECTRUM)
        spectrum = strdup("LM");
      else if (eig_param->spectrum == QUDA_SI_EIG_SPECTRUM)
        spectrum = strdup("SI");
      else if (eig_param->spectrum == QUDA_LI_EIG_SPECTRUM)
        spectrum = strdup("LI");
    }

    bool reverse = true;
    const char *L = "L";
    const char *S = "S";
    if (strncmp(L, spectrum, 1) == 0 && !eig_param->use_poly_acc) {
      reverse = false;
    } else if (strncmp(S, spectrum, 1) == 0 && eig_param->use_poly_acc) {
      reverse = false;
    } else if (strncmp(L, spectrum, 1) == 0 && eig_param->use_poly_acc) {
      reverse = false;
    }

    double tol_ = eig_param->tol;
    double *mod_h_evals_sorted = (double *)malloc(nKr_ * sizeof(double));

    // Memory checks
    if ((mod_h_evals_sorted == nullptr) || (h_evals_sorted_idx == nullptr)) {
      errorQuda("eigenSolver: not enough memory for host eigenvalue sorting");
    }

    // ARPACK workspace
    Complex I(0.0, 1.0);
    Complex *resid_ = (Complex *)malloc(ldv_ * sizeof(Complex));

    // Use initial guess?
    if (info_ > 0)
      for (int a = 0; a < ldv_; a++) {
        resid_[a] = I;
        // printfQuda("(%e , %e)\n", real(resid_[a]), imag(resid_[a]));
      }

    Complex sigma_ = 0.0;
    Complex *w_workd_ = (Complex *)malloc(3 * ldv_ * sizeof(Complex));
    Complex *w_workl_ = (Complex *)malloc(lworkl_ * sizeof(Complex));
    Complex *w_workev_ = (Complex *)malloc(2 * nKr_ * sizeof(Complex));
    double *w_rwork_ = (double *)malloc(nKr_ * sizeof(double));
    int *select_ = (int *)malloc(nKr_ * sizeof(int));

    // Alias pointers
    Complex *h_evecs_ = nullptr;
    h_evecs_ = (Complex *)(double *)(h_evecs);
    Complex *h_evals_ = nullptr;
    h_evals_ = (Complex *)(double *)(h_evals);

    // Memory checks
    if ((iparam_ == nullptr) || (ipntr_ == nullptr) || (resid_ == nullptr) || (w_workd_ == nullptr)
        || (w_workl_ == nullptr) || (w_workev_ == nullptr) || (w_rwork_ == nullptr) || (select_ == nullptr)) {
      errorQuda("eigenSolver: not enough memory for ARPACK workspace.\n");
    }

    int iter_count = 0;

    bool allocate = true;
    ColorSpinorField *h_v = nullptr;
    ColorSpinorField *d_v = nullptr;
    ColorSpinorField *h_v2 = nullptr;
    ColorSpinorField *d_v2 = nullptr;
    ColorSpinorField *resid = nullptr;

    // ARPACK log routines
    // Code added to print the log of ARPACK
    int arpack_log_u = 9999;

#if (defined(QMP_COMMS) || defined(MPI_COMMS))

    if (arpack_logfile != NULL && (comm_rank() == 0)) {

      ARPACK(initlog)(&arpack_log_u, arpack_logfile, strlen(arpack_logfile));
      int msglvl0 = 9, msglvl3 = 9;
      ARPACK(pmcinitdebug)
      (&arpack_log_u, // logfil
       &msglvl3,      // mcaupd
       &msglvl3,      // mcaup2
       &msglvl0,      // mcaitr
       &msglvl3,      // mceigh
       &msglvl0,      // mcapps
       &msglvl0,      // mcgets
       &msglvl3       // mceupd
      );

      printfQuda("eigenSolver: Log info:\n");
      printfQuda("ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
      printfQuda("output is directed to %s\n", arpack_logfile);
    }
#else
    if (arpack_logfile != NULL) {

      ARPACK(initlog)(&arpack_log_u, arpack_logfile, strlen(arpack_logfile));
      int msglvl0 = 9, msglvl3 = 9;
      ARPACK(mcinitdebug)
      (&arpack_log_u, // logfil
       &msglvl3,      // mcaupd
       &msglvl3,      // mcaup2
       &msglvl0,      // mcaitr
       &msglvl3,      // mceigh
       &msglvl0,      // mcapps
       &msglvl0,      // mcgets
       &msglvl3       // mceupd
      );

      printfQuda("eigenSolver: Log info:\n");
      printfQuda("ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
      printfQuda("output is directed to %s\n", arpack_logfile);
    }

#endif

    // Start ARPACK routines
    //---------------------------------------------------------------------------------

    double t1, t2;

    do {

      t1 = -((double)clock());

      // Interface to arpack routines
      //----------------------------
#if (defined(QMP_COMMS) || defined(MPI_COMMS))

      ARPACK(pznaupd)
      (fcomm_, &ido_, &bmat, &n_, spectrum, &nEv_, &tol_, resid_, &nKr_, h_evecs_, &n_, iparam_, ipntr_, w_workd_,
       w_workl_, &lworkl_, w_rwork_, &info_, 1, 2);

      if (info_ != 0) {
        arpackErrorHelpNAUPD();
        errorQuda("\nError in pznaupd info = %d. Exiting.", info_);
      }
#else
      ARPACK(znaupd)
      (&ido_, &bmat, &n_, spectrum, &nEv_, &tol_, resid_, &nKr_, h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_,
       &lworkl_, w_rwork_, &info_, 1, 2);
      if (info_ != 0) {
        arpackErrorHelpNAUPD();
        errorQuda("\nError in znaupd info = %d. Exiting.", info_);
      }
#endif

      // If this is the first iteration, we allocate CPU and GPU memory for QUDA
      if (allocate) {

        // Fortran arrays start at 1. The C++ pointer is therefore the Fortran pointer
        // less one, hence ipntr[0] - 1 to specify the correct address.

        cpuParam->location = QUDA_CPU_FIELD_LOCATION;
        cpuParam->create = QUDA_REFERENCE_FIELD_CREATE;
        cpuParam->gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

        cpuParam->v = w_workd_ + (ipntr_[0] - 1);
        h_v = ColorSpinorField::Create(*cpuParam);
        // Adjust the position of the start of the array.
        cpuParam->v = w_workd_ + (ipntr_[1] - 1);
        h_v2 = ColorSpinorField::Create(*cpuParam);

        ColorSpinorParam cudaParam(*cpuParam);
        cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
        cudaParam.create = QUDA_ZERO_FIELD_CREATE;
        cudaParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
        cudaParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;

        d_v = ColorSpinorField::Create(cudaParam);
        d_v2 = ColorSpinorField::Create(cudaParam);
        resid = ColorSpinorField::Create(cudaParam);
        allocate = false;
      }

      if (ido_ == 99 || info_ == 1) break;

      if (ido_ == -1 || ido_ == 1) {

        t2 = -clock();

        *d_v = *h_v;
        // apply matrix-vector operation here:
        eig_solver->chebyOp(mat, *d_v2, *d_v);
        *h_v2 = *d_v2;

        t2 += clock();

        time_mv += t2;
      }

      t1 += clock();
      time_ar += t1;

      printfQuda("Arpack Iteration %s: %d\n", eig_param->use_poly_acc ? "(with poly acc) " : "", iter_count);
      iter_count++;

    } while (99 != ido_ && iter_count < max_iter);

    // Subspace calulated sucessfully. Compute nEv eigenvectors and values

    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_count, info_, ido_);
    printfQuda("Computing eigenvectors\n");

    time_ev = -clock();

    // Interface to arpack routines
    //----------------------------
#if (defined(QMP_COMMS) || defined(MPI_COMMS))
    ARPACK(pzneupd)
    (fcomm_, &rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_, w_workev_, &bmat, &n_, spectrum, &nEv_, &tol_,
     resid_, &nKr_, h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_, w_rwork_, &info_, 1, 1, 2);
    if (info_ == -15) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in pzneupd info = %d. You likely need to\n"
                "increase the maximum ARPACK iterations. Exiting.\n",
                info_);
    } else if (info_ != 0)
      errorQuda("\nError in pzneupd info = %d. Exiting.\n", info_);
#else
    ARPACK(zneupd)
    (&rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_, w_workev_, &bmat, &n_, spectrum, &nEv_, &tol_, resid_,
     &nKr_, h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_, w_rwork_, &info_, 1, 1, 2);
    if (info_ == -15) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in zneupd info = %d. You likely need to\n"
                "increase the maximum ARPACK iterations. Exiting.\n",
                info_);
    } else if (info_ != 0)
      errorQuda("\nError in zneupd info = %d. Exiting.\n", info_);
#endif

    // Print additional convergence information.
    if ((info_) == 1) {
      printfQuda("Maximum number of iterations reached.\n");
    } else {
      if (info_ == 3) {
        printfQuda("Error: No shifts could be applied during implicit\n");
        printfQuda("Error: Arnoldi update, try increasing nKr.\n");
      }
    }

#if (defined(QMP_COMMS) || defined(MPI_COMMS))

    if (comm_rank() == 0) {
      if (arpack_logfile != NULL) { ARPACK(finilog)(&arpack_log_u); }
    }
#else
    if (arpack_logfile != NULL) ARPACK(finilog)(&arpack_log_u);

#endif

    printfQuda("Checking eigenvalues\n");

    int nconv = iparam_[4];

    // Sort the eigenvalues in absolute ascending order
    std::vector<std::pair<double, int>> evals_sorted;
    for (int j = 0; j < nconv; j++) { evals_sorted.push_back(std::make_pair(h_evals_[j].real(), j)); }

    // Sort the array by value (first in the pair)
    // and the index (second in the pair) will come along
    // for the ride.
    std::sort(evals_sorted.begin(), evals_sorted.end());
    if (reverse) std::reverse(evals_sorted.begin(), evals_sorted.end());

    // print out the computed Ritz values and their error estimates
    for (int j = 0; j < nconv; j++) {
      printfQuda("RitzValue[%04d] = %+.16e %+.16e Residual: %+.16e\n", j, real(h_evals_[evals_sorted[j].second]),
                 imag(h_evals_[evals_sorted[j].second]), std::abs(*(w_workl_ + ipntr_[10] - 1 + evals_sorted[j].second)));
    }

    // Compute Eigenvalues from Eigenvectors.
    ColorSpinorField *h_v3 = NULL;
    int idx = 0;
    for (int i = 0; i < nconv; i++) {
      idx = nconv - 1 - evals_sorted[i].second;
      cpuParam->v = (Complex *)h_evecs_ + idx * ldv_;
      h_v3 = ColorSpinorField::Create(*cpuParam);

      // d_v = v
      *d_v = *h_v3;

      // d_v2 = M*v
      // eig_solver->matVec(*d_v2, *d_v);
      eig_solver->matVec(mat, *d_v2, *d_v);

      // lambda = v^dag * M*v
      h_evals_[idx] = blas::cDotProduct(*d_v, *d_v2);

      Complex unit(1.0, 0.0);
      Complex m_lambda(-real(h_evals_[idx]), -imag(h_evals_[idx]));

      // d_v = ||M*v - lambda*v||
      blas::caxpby(unit, *d_v2, m_lambda, *d_v);
      double L2norm = blas::norm2(*d_v);

      printfQuda("EigValue[%04d] = %+.16e  %+.16e  Residual: %.16e\n", i, real(h_evals_[idx]), imag(h_evals_[idx]),
                 sqrt(L2norm));

      delete h_v3;
    }

    time_ev += clock();

    double total = (time_ar + time_ev) / CLOCKS_PER_SEC;
    printfQuda("Time to solve problem using ARPACK         = %e\n", total);
    printfQuda("Time spent in ARPACK                       = %e  %.1f%%\n", (time_ar - time_mv) / CLOCKS_PER_SEC,
               100 * ((time_ar - time_mv) / CLOCKS_PER_SEC) / total);
    printfQuda("Time spent in QUDA (M*vec + data transfer) = %e  %.1f%%\n", time_mv / CLOCKS_PER_SEC,
               100 * (time_mv / CLOCKS_PER_SEC) / total);
    printfQuda("Time spent in computing Eigenvectors       = %e  %.1f%%\n", time_ev / CLOCKS_PER_SEC,
               100 * (time_ev / CLOCKS_PER_SEC) / total);

    // cleanup
    free(ipntr_);
    free(iparam_);
    free(mod_h_evals_sorted);
    free(h_evals_sorted_idx);
    free(resid_);
    free(w_workd_);
    free(w_workl_);
    free(w_workev_);
    free(w_rwork_);
    free(select_);
    free(spectrum);

    delete h_v;
    delete h_v2;
    delete d_v;
    delete d_v2;
    delete resid;

    return;
  }

  void arpackErrorHelpNAUPD()
  {
    printfQuda("Error help NAUPD\n");
    printfQuda("INFO Integer.  (INPUT/OUTPUT)\n");
    printfQuda("     If INFO .EQ. 0, a randomly initial residual vector is used.\n");
    printfQuda("     If INFO .NE. 0, RESID contains the initial residual vector,\n");
    printfQuda("                        possibly from a previous run.\n");
    printfQuda("     Error flag on output.\n");
    printfQuda("     =  0: Normal exit.\n");
    printfQuda("     =  1: Maximum number of iterations taken.\n");
    printfQuda("        All possible eigenvalues of OP has been found. IPARAM(5)\n");
    printfQuda("        returns the number of wanted converged Ritz values.\n");
    printfQuda("     =  2: No longer an informational error. Deprecated starting\n");
    printfQuda("        with release 2 of ARPACK.\n");
    printfQuda("     =  3: No shifts could be applied during a cycle of the\n");
    printfQuda("        Implicitly restarted Arnoldi iteration. One possibility\n");
    printfQuda("        is to increase the size of NCV relative to NEV.\n");
    printfQuda("        See remark 4 below.\n");
    printfQuda("     = -1: N must be positive.\n");
    printfQuda("     = -2: NEV must be positive.\n");
    printfQuda("     = -3: NCV-NEV >= 1 and less than or equal to N.\n");
    printfQuda("     = -4: The maximum number of Arnoldi update iteration\n");
    printfQuda("        must be greater than zero.\n");
    printfQuda("     = -5: WHICH must be 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'\n");
    printfQuda("     = -6: BMAT must be one of 'I' or 'G'.\n");
    printfQuda("     = -7: Length of private work array is not sufficient.\n");
    printfQuda("     = -8: Error return from LAPACK eigenvalue calculation;\n");
    printfQuda("     = -9: Starting vector is zero.\n");
    printfQuda("     = -10: IPARAM(7) must be 1,2,3.\n");
    printfQuda("     = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.\n");
    printfQuda("     = -12: IPARAM(1) must be equal to 0 or 1.\n");
    printfQuda("     = -9999: Could not build an Arnoldi factorization.\n");
    printfQuda("        User input error highly likely.  Please\n");
    printfQuda("        check actual array dimensions and layout.\n");
    printfQuda("        IPARAM(5) returns the size of the current Arnoldi\n");
    printfQuda("        factorization.\n");
  }

  void arpackErrorHelpNEUPD()
  {
    printfQuda("Error help NEUPD\n");
    printfQuda("INFO Integer.  (OUTPUT)\n");
    printfQuda("     Error flag on output.\n");
    printfQuda("     =  0: Normal exit.\n");
    printfQuda("     =  1: The Schur form computed by LAPACK routine csheqr\n");
    printfQuda("        could not be reordered by LAPACK routine ztrsen.\n");
    printfQuda("        Re-enter subroutine zneupd with IPARAM(5)=NCV and\n");
    printfQuda("        increase the size of the array D to have\n");
    printfQuda("        dimension at least dimension NCV and allocate at\n");
    printfQuda("        least NCV\n");
    printfQuda("        columns for Z. NOTE: Not necessary if Z and V share\n");
    printfQuda("        the same space. Please notify the authors if this\n");
    printfQuda("        error occurs.\n");
    printfQuda("     = -1: N must be positive.\n");
    printfQuda("     = -2: NEV must be positive.\n");
    printfQuda("     = -3: NCV-NEV >= 1 and less than or equal to N.\n");
    printfQuda("     = -5: WHICH must be 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'\n");
    printfQuda("     = -6: BMAT must be one of 'I' or 'G'.\n");
    printfQuda("     = -7: Length of private work WORKL array is inufficient.\n");
    printfQuda("     = -8: Error return from LAPACK eigenvalue calculation.\n");
    printfQuda("        This should never happened.\n");
    printfQuda("     = -9: Error return from calculation of eigenvectors.\n");
    printfQuda("        Informational error from LAPACK routine ztrevc.\n");
    printfQuda("     = -10: IPARAM(7) must be 1,2,3\n");
    printfQuda("     = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.\n");
    printfQuda("     = -12: HOWMNY = 'S' not yet implemented\n");
    printfQuda("     = -13: HOWMNY must be one of 'A' or 'P' if RVEC = .true.\n");
    printfQuda("     = -14: ZNAUPD did not find any eigenvalues to sufficient\n");
    printfQuda("        accuracy.\n");
    printfQuda("     = -15: ZNEUPD got a different count of the number of\n");
    printfQuda("        converged Ritz values than ZNAUPD got. This\n");
    printfQuda("        indicates the user probably made an error in\n");
    printfQuda("        passing data from ZNAUPD to ZNEUPD or that the\n");
    printfQuda("        data was modified before entering ZNEUPD\n");
  }

#else

  void arpack_solve(void *h_evecs, void *h_evals, const Dirac &mat, QudaEigParam *eig_param, ColorSpinorParam *cpuParam)
  {

    errorQuda("(P)ARPACK has not been enabled for this build");
  }

#endif

} // namespace quda
