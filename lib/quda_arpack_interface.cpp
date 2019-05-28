#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include <quda_internal.h>
#include <quda_arpack_interface.h>
#include <eigensolve_quda.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <util_quda.h>
#include <sys/time.h>

// ARPACK INTERAFCE ROUTINES
//--------------------------------------------------------------------------

namespace quda
{

#ifdef ARPACK_LIB

  void arpackErrorHelpNAUPD();
  void arpackErrorHelpNEUPD();

#if (defined(QMP_COMMS) || defined(MPI_COMMS))
#include <mpi.h>
#endif

  void arpack_solve(void *h_evecs, void *h_evals, const DiracMatrix &mat, QudaEigParam *eig_param,
                    ColorSpinorParam *cpuParam)
  {

    // ARPACK logfile name
    char *arpack_logfile = eig_param->arpack_logfile;
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("**** START ARPACK SOLUTION ****\n");
      printfQuda("Output directed to %s\n", arpack_logfile);
    }

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
    local_vol *= eig_param->invert_param->Ls;

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
      if (eig_param->spectrum == QUDA_SPECTRUM_SR_EIG)
        spectrum = strdup("LR");
      else if (eig_param->spectrum == QUDA_SPECTRUM_LR_EIG)
        spectrum = strdup("SR");
      else if (eig_param->spectrum == QUDA_SPECTRUM_SM_EIG)
        spectrum = strdup("LM");
      else if (eig_param->spectrum == QUDA_SPECTRUM_LM_EIG)
        spectrum = strdup("SM");
      else if (eig_param->spectrum == QUDA_SPECTRUM_SI_EIG)
        spectrum = strdup("LI");
      else if (eig_param->spectrum == QUDA_SPECTRUM_LI_EIG)
        spectrum = strdup("SI");
    } else {
      if (eig_param->spectrum == QUDA_SPECTRUM_SR_EIG)
        spectrum = strdup("SR");
      else if (eig_param->spectrum == QUDA_SPECTRUM_LR_EIG)
        spectrum = strdup("LR");
      else if (eig_param->spectrum == QUDA_SPECTRUM_SM_EIG)
        spectrum = strdup("SM");
      else if (eig_param->spectrum == QUDA_SPECTRUM_LM_EIG)
        spectrum = strdup("LM");
      else if (eig_param->spectrum == QUDA_SPECTRUM_SI_EIG)
        spectrum = strdup("SI");
      else if (eig_param->spectrum == QUDA_SPECTRUM_LI_EIG)
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

      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("eigenSolver: Log info:\n");
        printfQuda("ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
        printfQuda("output is directed to %s\n", arpack_logfile);
      }
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

      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("eigenSolver: Log info:\n");
        printfQuda("ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
        printfQuda("output is directed to %s\n", arpack_logfile);
      }
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

      if (getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("Arpack Iteration %s: %d\n", eig_param->use_poly_acc ? "(with poly acc) " : "", iter_count);
      iter_count++;

    } while (99 != ido_ && iter_count < max_iter);

    // Subspace calulated sucessfully. Compute nEv eigenvectors and values

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_count, info_, ido_);
      printfQuda("Computing eigenvectors\n");
    }

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
                "increase the maximum ARPACK iterations. Exiting.",
                info_);
    } else if (info_ != 0) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in pzneupd info = %d. Exiting.", info_);
    }
#else
    ARPACK(zneupd)
    (&rvec_, &howmny, select_, h_evals_, h_evecs_, &n_, &sigma_, w_workev_, &bmat, &n_, spectrum, &nEv_, &tol_, resid_,
     &nKr_, h_evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_, w_rwork_, &info_, 1, 1, 2);
    if (info_ == -15) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in zneupd info = %d. You likely need to\n"
                "increase the maximum ARPACK iterations. Exiting.",
                info_);
    } else if (info_ != 0) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in zneupd info = %d. Exiting.", info_);
#endif

    // Print additional convergence information.
    if ((info_) == 1) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Maximum number of iterations reached.\n");
    } else {
      if (info_ == 3) {
        errorQuda("ARPACK Error: No shifts could be applied during implicit\n");
        errorQuda("Arnoldi update.\n");
      }
    }

#if (defined(QMP_COMMS) || defined(MPI_COMMS))

    if (comm_rank() == 0) {
      if (arpack_logfile != NULL) { ARPACK(finilog)(&arpack_log_u); }
    }
#else
      if (arpack_logfile != NULL) ARPACK(finilog)(&arpack_log_u);

#endif

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Checking eigenvalues\n");

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
      if (getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("RitzValue[%04d] = %+.16e %+.16e Residual: %+.16e\n", j, real(h_evals_[j]), imag(h_evals_[j]),
                   std::abs(*(w_workl_ + ipntr_[10] - 1 + j)));
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
      eig_solver->matVec(mat, *d_v2, *d_v);

      // lambda = v^dag * M*v
      h_evals_[idx] = blas::cDotProduct(*d_v, *d_v2);

      Complex unit(1.0, 0.0);
      Complex m_lambda(-real(h_evals_[idx]), -imag(h_evals_[idx]));

      // d_v = ||M*v - lambda*v||
      blas::caxpby(unit, *d_v2, m_lambda, *d_v);
      double L2norm = blas::norm2(*d_v);

      if (getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("EigValue[%04d] = %+.16e  %+.16e  Residual: %.16e\n", i, real(h_evals_[idx]), imag(h_evals_[idx]),
                   sqrt(L2norm));

      delete h_v3;
    }

    time_ev += clock();

    double total = (time_ar + time_ev) / CLOCKS_PER_SEC;

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("Time to solve problem using ARPACK         = %e\n", total);
      printfQuda("Time spent in ARPACK                       = %e  %.1f%%\n", (time_ar - time_mv) / CLOCKS_PER_SEC,
                 100 * ((time_ar - time_mv) / CLOCKS_PER_SEC) / total);
      printfQuda("Time spent in QUDA (M*vec + data transfer) = %e  %.1f%%\n", time_mv / CLOCKS_PER_SEC,
                 100 * (time_mv / CLOCKS_PER_SEC) / total);
      printfQuda("Time spent in computing Eigenvectors       = %e  %.1f%%\n", time_ev / CLOCKS_PER_SEC,
                 100 * (time_ev / CLOCKS_PER_SEC) / total);
    }

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

  void arpack_solve(void *h_evecs, void *h_evals, const DiracMatrix &mat, QudaEigParam *eig_param,
                    ColorSpinorParam *cpuParam)
  {
    errorQuda("(P)ARPACK has not been enabled for this build");
  }
#endif

} // namespace quda
