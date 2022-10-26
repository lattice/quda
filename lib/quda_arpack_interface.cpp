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

// ARPACK INTERAFCE ROUTINES
//--------------------------------------------------------------------------

#if (defined(QMP_COMMS) || defined(MPI_COMMS))
#include <mpi.h>
#include "mpi_comm_handle.h"
#endif

namespace quda
{

#ifdef ARPACK_LIB

  void arpackErrorHelpNAUPD();
  void arpackErrorHelpNEUPD();

  void arpack_solve(std::vector<ColorSpinorField> &h_evecs, std::vector<Complex> &h_evals, const DiracMatrix &mat,
                    QudaEigParam *eig_param, TimeProfile &profile)
  {
    // Create Eigensolver object for member function use
    EigenSolver *eig_solver = EigenSolver::create(eig_param, mat, profile);

    profile.TPSTART(QUDA_PROFILE_INIT);

// ARPACK logfile name
#ifdef ARPACK_LOGGING
    char *arpack_logfile = eig_param->arpack_logfile;
#endif
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("**** START ARPACK SOLUTION ****\n");
#ifndef ARPACK_LOGGING
      printfQuda("Arpack logging not enabled.\n");
#else
      printfQuda("Output directed to %s\n", arpack_logfile);
#endif
    }

    // Construct parameters and memory allocation
    //---------------------------------------------------------------------------------

    // MPI objects
#if (defined(QMP_COMMS) || defined(MPI_COMMS))
    int *fcomm_ = nullptr;
    MPI_Fint mpi_comm_fort = MPI_Comm_c2f(get_mpi_handle());
    fcomm_ = static_cast<int *>(&mpi_comm_fort);
#endif

    // all FORTRAN communication uses underscored
    int ido_ = 0;
    int info_ = 1; // if 0, use random vector. If 1, initial residual lives in resid_
    std::vector<int> ipntr_(14);
    std::vector<int> iparam_(11);
    int n_ = h_evecs[0].Volume() * h_evecs[0].Nspin() * h_evecs[0].Ncolor();
    int n_ev_ = eig_param->n_ev;
    int n_kr_ = eig_param->n_kr;
    int ldv_ = h_evecs[0].Volume() * h_evecs[0].Nspin() * h_evecs[0].Ncolor();
    int lworkl_ = (3 * n_kr_ * n_kr_ + 5 * n_kr_) * 2;
    int rvec_ = 1;
    int max_iter = eig_param->max_restarts * (n_kr_ - n_ev_) + n_ev_;

    // Assign values to ARPACK params
    iparam_[0] = 1;
    iparam_[2] = max_iter;
    iparam_[3] = 1;
    iparam_[6] = 1;

    // ARPACK problem type to be solved
    char howmny = 'A';
    char bmat = 'I';
    char spectrum[3];

    // Part of the spectrum to be computed.
    switch (eig_param->spectrum) {
    case QUDA_SPECTRUM_SR_EIG: strcpy(spectrum, "SR"); break;
    case QUDA_SPECTRUM_LR_EIG: strcpy(spectrum, "LR"); break;
    case QUDA_SPECTRUM_SM_EIG: strcpy(spectrum, "SM"); break;
    case QUDA_SPECTRUM_LM_EIG: strcpy(spectrum, "LM"); break;
    case QUDA_SPECTRUM_SI_EIG: strcpy(spectrum, "SI"); break;
    case QUDA_SPECTRUM_LI_EIG: strcpy(spectrum, "LI"); break;
    default: errorQuda("Unexpected spectrum type %d", eig_param->spectrum);
    }

    if (strncmp("SR", spectrum, 2) == 0 && eig_param->use_poly_acc) {
      // Smallest real eigenvalues requested by user.
      // We will compute the largest eigenvaules of the polynomial
      // operator, then reverse the spectrum.
      spectrum[0] = 'L';
      spectrum[1] = 'R';
    }

    double tol_ = eig_param->tol;

    // ARPACK workspace
    Complex I(0.0, 1.0);
    std::vector<Complex> resid_(ldv_);

    // Use initial guess?
    if (info_ > 0) {
      for (int a = 0; a < ldv_; a++) resid_[a] = drand48();
    }

    Complex sigma_ = 0.0;
    std::vector<Complex> w_workd_(3 * ldv_);
    std::vector<Complex> w_workl_(lworkl_);
    std::vector<Complex> w_workev_(2 * n_kr_);
    std::vector<double> w_rwork_(n_kr_);
    std::vector<int> select_(n_kr_);

    std::vector<Complex> h_evecs_(n_kr_ * ldv_);
    std::vector<Complex> h_evals_(n_ev_);

    // create container wrapping the vectors returned from ARPACK
    ColorSpinorParam param(h_evecs[0]);
    param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.location = QUDA_CPU_FIELD_LOCATION;
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

    std::vector<ColorSpinorField> h_evecs_arpack(n_kr_);

    for (int i = 0; i < n_kr_; i++) {
      param.v = h_evecs_.data() + i * ldv_;
      h_evecs_arpack[i] = ColorSpinorField(param);
    }

    int iter_count = 0;

    bool allocate = true;
    ColorSpinorField h_v;
    ColorSpinorField d_v;
    ColorSpinorField h_v2;
    ColorSpinorField d_v2;
    ColorSpinorField resid;

#ifdef ARPACK_LOGGING
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
#endif

    profile.TPSTOP(QUDA_PROFILE_INIT);

    // Start ARPACK routines
    //---------------------------------------------------------------------------------

    do {

      profile.TPSTART(QUDA_PROFILE_ARPACK);

      // Interface to arpack routines
      //----------------------------
#if (defined(QMP_COMMS) || defined(MPI_COMMS))
      ARPACK(pznaupd)
      (fcomm_, &ido_, &bmat, &n_, spectrum, &n_ev_, &tol_, resid_.data(), &n_kr_, h_evecs_.data(), &n_, iparam_.data(),
       ipntr_.data(), w_workd_.data(), w_workl_.data(), &lworkl_, w_rwork_.data(), &info_, 1, 2);

      if (info_ != 0) {
        arpackErrorHelpNAUPD();
        errorQuda("\nError in pznaupd info = %d. Exiting.", info_);
      }
#else
      ARPACK(znaupd)
      (&ido_, &bmat, &n_, spectrum, &n_ev_, &tol_, resid_.data(), &n_kr_, h_evecs_.data(), &n_, iparam_.data(),
       ipntr_.data(), w_workd_.data(), w_workl_.data(), &lworkl_, w_rwork_.data(), &info_, 1, 2);
      if (info_ != 0) {
        arpackErrorHelpNAUPD();
        errorQuda("\nError in znaupd info = %d. Exiting.", info_);
      }
#endif

      profile.TPSTOP(QUDA_PROFILE_ARPACK);

      // If this is the first iteration, we allocate CPU and GPU memory for QUDA
      if (allocate) {
        ColorSpinorParam param(h_evecs[0]);
        param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        param.location = QUDA_CPU_FIELD_LOCATION;
        param.create = QUDA_REFERENCE_FIELD_CREATE;
        param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

        // Fortran arrays start at 1. The C++ pointer is therefore the Fortran pointer
        // less one, hence ipntr[0] - 1 to specify the correct address.
        param.v = w_workd_.data() + (ipntr_[0] - 1);
        h_v = ColorSpinorField(param);
        // Adjust the position of the start of the array.
        param.v = w_workd_.data() + (ipntr_[1] - 1);
        h_v2 = ColorSpinorField(param);

        // create device field temporaries
        param.location = QUDA_CUDA_FIELD_LOCATION;
        param.create = QUDA_ZERO_FIELD_CREATE;
        param.setPrecision(param.Precision(), param.Precision(), true);

        d_v = ColorSpinorField(param);
        d_v2 = ColorSpinorField(param);
        resid = ColorSpinorField(param);
        allocate = false;
      }

      if (ido_ == 99 || info_ == 1) break;

      if (ido_ == -1 || ido_ == 1) {

        profile.TPSTART(QUDA_PROFILE_D2H);

        d_v = h_v;

        profile.TPSTOP(QUDA_PROFILE_D2H);
        profile.TPSTART(QUDA_PROFILE_COMPUTE);

        // apply matrix-vector operation here:
        eig_solver->chebyOp(d_v2, d_v);

        profile.TPSTOP(QUDA_PROFILE_COMPUTE);
        profile.TPSTART(QUDA_PROFILE_H2D);

        h_v2 = d_v2;

        profile.TPSTOP(QUDA_PROFILE_H2D);
      }

      if (getVerbosity() >= QUDA_VERBOSE)
        printfQuda("Arpack Iteration %s: %d\n", eig_param->use_poly_acc ? "(with poly acc) " : "", iter_count);
      iter_count++;

    } while (99 != ido_ && iter_count < max_iter);

    // Subspace calulated sucessfully. Compute n_ev eigenvectors and values

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_count, info_, ido_);
      printfQuda("Computing eigenvectors\n");
    }

    profile.TPSTART(QUDA_PROFILE_ARPACK);

    // Interface to arpack routines
    //----------------------------
#if (defined(QMP_COMMS) || defined(MPI_COMMS))
    ARPACK(pzneupd)
    (fcomm_, &rvec_, &howmny, select_.data(), h_evals_.data(), h_evecs_.data(), &n_, &sigma_, w_workev_.data(), &bmat,
     &n_, spectrum, &n_ev_, &tol_, resid_.data(), &n_kr_, h_evecs_.data(), &n_, iparam_.data(), ipntr_.data(),
     w_workd_.data(), w_workl_.data(), &lworkl_, w_rwork_.data(), &info_, 1, 1, 2);
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
    (&rvec_, &howmny, select_.data(), h_evals_.data(), h_evecs_.data(), &n_, &sigma_, w_workev_.data(), &bmat, &n_,
     spectrum, &n_ev_, &tol_, resid_.data(), &n_kr_, h_evecs_.data(), &n_, iparam_.data(), ipntr_.data(),
     w_workd_.data(), w_workl_.data(), &lworkl_, w_rwork_.data(), &info_, 1, 1, 2);
    if (info_ == -15) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in zneupd info = %d. You likely need to\n"
                "increase the maximum ARPACK iterations. Exiting.",
                info_);
    } else if (info_ != 0) {
      arpackErrorHelpNEUPD();
      errorQuda("\nError in zneupd info = %d. Exiting.", info_);
    }
#endif

    profile.TPSTOP(QUDA_PROFILE_ARPACK);

    // Print additional convergence information.
    if ((info_) == 1) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Maximum number of iterations reached.\n");
    } else {
      if (info_ == 3) {
        errorQuda("ARPACK Error: No shifts could be applied during implicit\n");
        errorQuda("Arnoldi update.\n");
      }
    }
#ifdef ARPACK_LOGGING
#if (defined(QMP_COMMS) || defined(MPI_COMMS))
    if (comm_rank() == 0) {
      if (arpack_logfile != NULL) { ARPACK(finilog)(&arpack_log_u); }
    }
#else
    if (arpack_logfile != NULL) ARPACK(finilog)(&arpack_log_u);
#endif
#endif

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Checking eigenvalues\n");

    int nconv = iparam_[4];

    // Sort the eigenvalues. To do this we use the QUDA EigenSolver method, which
    // requires transferring data to std::vector arrays.
    std::vector<Complex> evals(nconv, 0.0);
    std::vector<int> arpack_index(nconv, 0.0);
    for (int i = 0; i < nconv; i++) {
      evals[i] = h_evals_[i];
      arpack_index[i] = i;
    }

    eig_solver->sortArrays(eig_param->spectrum, nconv, evals, arpack_index);

    // print out the computed Ritz values and their error estimates
    for (int i = 0; i < nconv; i++) {
      if (getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("RitzValue[%04d] = %+.16e %+.16e Residual: %+.16e\n", i, evals[i].real(), evals[i].imag(),
		   std::abs(*(w_workl_.data() + ipntr_[10] - 1 + arpack_index[i])));
    }

    // Compute singular/eigenvalues values from eigenvectors.
    if (eig_param->compute_svd) {
      printfQuda("Computing SVD\n");

      // This function assumes that you have computed the eigenvectors
      // of MdagM(MMdag), ie, the right(left) SVD of M. The ith eigen vector in the
      // array corresponds to the ith right(left) singular vector. We place the
      // computed left(right) singular vectors in the second half of the array. We
      // assume that right vectors are given and we compute the left.
      //
      // As a cross check, we recompute the singular values from mat vecs rather
      // than make the direct relation (sigma_i)^2 = |lambda_i|
      //--------------------------------------------------------------------------

      for (int i = 0; i < nconv; i++) {

	profile.TPSTART(QUDA_PROFILE_H2D);
	d_v = h_evecs_arpack[arpack_index[i]];
	profile.TPSTOP(QUDA_PROFILE_H2D);

	profile.TPSTART(QUDA_PROFILE_COMPUTE);
	// M*Rev_i = M*Rsv_i = sigma_i Lsv_i
	mat.Expose()->M(d_v2, d_v);
	// sigma_i = sqrt(sigma_i (Lsv_i)^dag * sigma_i * Lsv_i )
	double sigma_tmp = sqrt(blas::norm2(d_v2));
	// Normalise the Lsv: sigma_i Lsv_i -> Lsv_i
	blas::ax(1.0 / sigma_tmp, d_v2);
	profile.TPSTOP(QUDA_PROFILE_COMPUTE);

	if (getVerbosity() >= QUDA_SUMMARIZE)
	  printfQuda("Sval[%04d] = %+.16e sigma - sqrt(|lambda|) = %+.16e\n", i, sigma_tmp,
		     sigma_tmp - sqrt(abs(evals[i].real())));
      }
    } else {
      printfQuda("Computing Eigenvalues\n");
      for (int i = 0; i < nconv; i++) {

	profile.TPSTART(QUDA_PROFILE_D2H);
	d_v = h_evecs_arpack[arpack_index[i]];
	profile.TPSTOP(QUDA_PROFILE_D2H);

	profile.TPSTART(QUDA_PROFILE_COMPUTE);
	// d_v2 = M*v = lambda_measured * v
	mat(d_v2, d_v);
	// d_v = ||lambda_measured*v - lambda_arpack*v||
	blas::caxpby(Complex {1.0, 0.0}, d_v2, -evals[i], d_v);
	double L2norm = blas::norm2(d_v);
	profile.TPSTOP(QUDA_PROFILE_COMPUTE);

	if (getVerbosity() >= QUDA_SUMMARIZE)
	  printfQuda("Eval[%04d] = (%+.16e  %+.16e) ||%+.16e|| Residual: %.16e\n", i, evals[i].real(), evals[i].imag(),
		     abs(evals[i]), sqrt(L2norm));
      }
    }

    // copy back singular/eigenvectors and singular/eigenvalues using the sorting index
    if (eig_param->compute_svd) {
      for (int i = 0; i < nconv; i++) {

	profile.TPSTART(QUDA_PROFILE_H2D);
	d_v = h_evecs_arpack[arpack_index[i]];
	profile.TPSTOP(QUDA_PROFILE_H2D);

	// M*Rev_i = M*Rsv_i = sigma_i Lsv_i
	mat.Expose()->M(d_v2, d_v);

	// sigma_i = sqrt(sigma_i (Lsv_i)^dag * sigma_i * Lsv_i )
	double sigma_tmp = sqrt(blas::norm2(d_v2));

	// Normalise the Lsv: sigma_i Lsv_i -> Lsv_i
	blas::ax(1.0 / sigma_tmp, d_v2);

	h_evecs[i] = h_evecs_arpack[arpack_index[i]];
	profile.TPSTART(QUDA_PROFILE_D2H);
	h_evecs[i + nconv] = d_v2;
	profile.TPSTOP(QUDA_PROFILE_D2H);

	h_evals[i].real(sigma_tmp);
	h_evals[i].imag(0.0);
      }
    } else {
      for (int i = 0; i < nconv; i++) {
	h_evecs[i] = h_evecs_arpack[arpack_index[i]];
	h_evals[i] = evals[i];
      }
    }

    profile.TPSTART(QUDA_PROFILE_FREE);

    delete eig_solver;

    profile.TPSTOP(QUDA_PROFILE_FREE);
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

  void arpack_solve(std::vector<ColorSpinorField> &, std::vector<Complex> &, const DiracMatrix &, QudaEigParam *,
                    TimeProfile &)
  {
    errorQuda("(P)ARPACK has not been enabled for this build");
  }
#endif

} // namespace quda
