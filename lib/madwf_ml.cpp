#include <madwf_ml.h>
#include <madwf_transfer.h>

namespace quda
{

  MadwfAcc::MadwfAcc(const SolverParam &solve_param, TimeProfile &profile) :
    param(solve_param.madwf_param),
    mu(param.madwf_diagonal_suppressor),
    prec_precondition(solve_param.precision_precondition),
    profile(profile)
  {
    if (getVerbosity() >= QUDA_VERBOSE) {
      printfQuda("Launching MADWF accelerator ... \n");
      printfQuda("madwf_mu (low modes suppressor)                   = %.4f\n", param.madwf_diagonal_suppressor);
      printfQuda("madwf_ls (cheap Ls)                               = %d\n", param.madwf_ls);
      printfQuda("madwf_null_miniter                                = %d\n", param.madwf_null_miniter);
      printfQuda("madwf_null_tol                                    = %4.2e\n", param.madwf_null_tol);
      printfQuda("madwf_train_maxiter (max # of iters for training) = %d\n", param.madwf_train_maxiter);
    }
  }

  void MadwfAcc::fill_random(std::vector<transfer_float> &v)
  {
    static std::random_device rd;
    // the good rng
    static std::mt19937 rng(23ul * comm_rank());
    // The gaussian distribution
    static std::normal_distribution<double> n(0., 1.);

    for (auto &x : v) { x = 1e-1 * n(rng); }
  }

  void MadwfAcc::apply(Solver &base, ColorSpinorField &out, const ColorSpinorField &in)
  {
    madwf_ml::transfer_5d_hh(forward_tmp, in, device_param, false);
    base(backward_tmp, forward_tmp);
    madwf_ml::transfer_5d_hh(out, backward_tmp, device_param, true);

    blas::axpy(mu, in, out);
  }

  double MadwfAcc::cost(const DiracMatrix &ref, Solver &base, ColorSpinorField &out, const ColorSpinorField &in)
  {
    ColorSpinorParam csParam(in);
    ColorSpinorField tmp1(csParam);
    ColorSpinorField tmp2(csParam);

    apply(base, tmp1, in);
    ref(tmp2, tmp1);

    blas::copy(out, in);

    // M * T^ * A * T * phi - phi
    return blas::xmyNorm(tmp2, out);
  }

  void MadwfAcc::train(const DiracMatrix &ref, Solver &base, Solver &null, const ColorSpinorField &in,
                       bool tune_suppressor)
  {

    profile.TPSTART(QUDA_PROFILE_INIT);
    constexpr int complex_matrix_size = static_cast<int>(transfer_t); // spin by spin

    if (in.Ndim() != 5) { errorQuda("we need a 5 dimensional field for this."); }

    int Ls = in.X(4);
    int Ls_base = param.madwf_ls;
    size_t param_size = Ls * Ls_base * complex_matrix_size * 2;
    std::vector<transfer_float> host_param(param_size);

    if (param.madwf_param_load) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      profile.TPSTART(QUDA_PROFILE_IO);
      load_parameter(Ls, Ls_base);

      ColorSpinorParam csParam(in);
      csParam.x[4] = Ls_base;
      csParam.create = QUDA_NULL_FIELD_CREATE;
      csParam.setPrecision(prec_precondition);

      forward_tmp = ColorSpinorField(csParam);
      backward_tmp = ColorSpinorField(csParam);
      profile.TPSTOP(QUDA_PROFILE_IO);

      return;
    }

    ColorSpinorParam csParam(in);
    ColorSpinorField null_x(csParam);
    ColorSpinorField null_b(csParam);

    RNG rng(null_b, 2767);

    if (getVerbosity() >= QUDA_VERBOSE) { printfQuda("Generating Null Space Vectors ... \n"); }
    spinorNoise(null_b, rng, QUDA_NOISE_GAUSS);

    std::vector<ColorSpinorField> B(16);
    csParam.setPrecision(prec_precondition);
    for (auto &pB : B) { pB = ColorSpinorField(csParam); }

    profile.TPSTOP(QUDA_PROFILE_INIT);
    null.solve_and_collect(null_x, null_b, B, param.madwf_null_miniter, param.madwf_null_tol);
    profile.TPSTART(QUDA_PROFILE_INIT);
    for (auto &pb : B) { blas::ax(5e3 / sqrt(blas::norm2(pb)), pb); }

    commGlobalReductionPush(false);

    ColorSpinorField chi(csParam);
    ColorSpinorField tmp(csParam);
    ColorSpinorField theta(csParam);
    ColorSpinorField lambda(csParam);
    ColorSpinorField Mchi(csParam);

    double residual = 0.0;
    int count = 0;
    for (auto &phi : B) {
      residual += blas::norm2(phi);
      if (getVerbosity() >= QUDA_VERBOSE) {
        printfQuda("reference dslash norm %03d = %8.4e\n", count, blas::norm2(phi));
      }
      count++;
    }
    if (getVerbosity() >= QUDA_VERBOSE) { printfQuda("reference dslash norm = %8.4e\n", residual); }

    csParam.x[4] = Ls_base;
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    ColorSpinorField ATchi(csParam);
    ColorSpinorField ATphi(csParam);
    ColorSpinorField ADphi(csParam);

    ColorSpinorField ATMchi(csParam);

    forward_tmp = ColorSpinorField(csParam);
    backward_tmp = ColorSpinorField(csParam);

    fill_random(host_param);

    device_param.resize(param_size);
    device_param.from_host(host_param);

    device_container d1(param_size);
    device_container d2(param_size);
    device_container P(param_size);
    device_container D_old(param_size);

    double pmu = 0.0;

    transfer_float alpha;
    transfer_float b = 0.8;
    if (getVerbosity() >= QUDA_VERBOSE) {
      printfQuda("beta          = %.3f\n", b);
      printfQuda("training mu   = %.3f\n", mu);
    }

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_TRAINING);

    for (int iteration = 0; iteration < param.madwf_train_maxiter; iteration++) {

      device_container D(param_size);
      double dmu = 0.0;
      double chi2 = 0.0;
      std::array<double, 5> a = {};

      for (auto &phi : B) {
        chi2 += cost(ref, base, chi, phi);
        // ATx(ATphi, phi, T);
        madwf_ml::transfer_5d_hh(forward_tmp, phi, device_param, false);
        base(ATphi, forward_tmp);

        ref(Mchi, chi);

        // ATx(ATMchi, Mchi, T);
        madwf_ml::transfer_5d_hh(forward_tmp, Mchi, device_param, false);
        base(ATMchi, forward_tmp);

        // d1 = A * T * phi -x- M * chi
        madwf_ml::tensor_5d_hh(ATphi, Mchi, d1);
        // d2 = A * T * M * phi -x- phi
        madwf_ml::tensor_5d_hh(ATMchi, phi, d2);

        axpby(D, 2.0f, d1, 2.0f, d2);
        if (tune_suppressor) { dmu += 2.0 * blas::reDotProduct(Mchi, phi); }
      }

      axpby(P, (b - 1), P, (1 - b), D);
      if (tune_suppressor) { pmu = b * pmu + (1 - b) * dmu; }

      chi2 = 0.0;
      // line search
      for (auto &phi : B) {

        double ind_chi2 = cost(ref, base, chi, phi);
        chi2 += ind_chi2;

        // ATx(ATphi, phi, T);
        madwf_ml::transfer_5d_hh(forward_tmp, phi, device_param, false);
        base(ATphi, forward_tmp);

        // D' * A * T * phi
        madwf_ml::transfer_5d_hh(theta, ATphi, P, true);

        // ATx(ADphi, phi, P);
        madwf_ml::transfer_5d_hh(forward_tmp, phi, P, false);
        base(ADphi, forward_tmp);

        // T' * A * D * phi
        madwf_ml::transfer_5d_hh(tmp, ADphi, device_param, true);
        // theta
        blas::axpy(1.0, theta, tmp);
        if (tune_suppressor) { blas::axpy(pmu, phi, tmp); }

        ref(theta, tmp);

        // lambda = D' * A * D * phi
        madwf_ml::transfer_5d_hh(tmp, ADphi, P, true);

        ref(lambda, tmp);

        std::vector<Complex> dot(9);
        blas::cDotProduct(dot, {chi, theta, lambda}, {chi, theta, lambda});

        a[0] += dot[0].real();
        a[1] += -2.0 * dot[1].real();
        a[2] += dot[4].real() + 2.0 * dot[2].real();
        a[3] += -2.0 * dot[5].real();
        a[4] += dot[8].real();
      }

      std::array<double, 4> coeffs = {4.0 * a[4], 3.0 * a[3], 2.0 * a[2], a[1]};
      auto rs = cubic_formula(coeffs);

      alpha = 0;
      double root_min = poly4(a, 0);
      for (auto r : rs) {
        double eval = poly4(a, r);
        if (root_min > eval) {
          root_min = eval;
          alpha = r;
        }
      }

      axpby(device_param, 0.0f, device_param, -alpha, P);
      if (tune_suppressor) { mu -= alpha * pmu; }

      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("grad min iter %05d: %04d chi2 = %8.4e, chi2 %% = %8.4e, alpha = %+8.4e, mu = %+8.4e\n", comm_rank(),
                   iteration, chi2, chi2 / residual, alpha, mu);
      }
    }

    trained = true;

    if (getVerbosity() >= QUDA_VERBOSE) { printfQuda("Training finished ...\n"); }
    count = 0;
    for (auto &phi : B) {
      double ind_chi2 = cost(ref, base, chi, phi);
      double phi2 = blas::norm2(phi);
      if (getVerbosity() >= QUDA_VERBOSE) {
        printfQuda("chi2 %03d %% = %8.4e, phi2 = %8.4e\n", count, ind_chi2 / phi2, phi2);
      }
      count++;
    }

    // Broadcast the trained parameters
    host_param = device_param.to_host();
    comm_broadcast(host_param.data(), host_param.size() * sizeof(transfer_float));
    device_param.from_host(host_param);

    commGlobalReductionPop();
    profile.TPSTOP(QUDA_PROFILE_TRAINING);

    if (param.madwf_param_save) {
      profile.TPSTART(QUDA_PROFILE_IO);
      if (comm_rank() == 0) { save_parameter(Ls, Ls_base); } // Only rank zero write out to the disk
      comm_barrier();
      profile.TPSTOP(QUDA_PROFILE_IO);
    }
  }

  void MadwfAcc::save_parameter(int Ls, int Ls_base)
  {
    if (comm_rank() != 0) { errorQuda("Only rank zero writes out to disk"); } // Only rank zero write out to the disk
    std::vector<transfer_float> host_param = device_param.to_host();

    std::string save_param_path(param.madwf_param_outfile);
    char cstring[512];
    sprintf(cstring, "/madwf_trained_param_ls_%02d_%02d_mu_%.3f.dat", Ls, Ls_base, mu);
    save_param_path += std::string(cstring);
    FILE *fp = fopen(save_param_path.c_str(), "w");
    if (!fp) { errorQuda("Unable to open file %s\n", save_param_path.c_str()); }
    size_t fwrite_count = fwrite(host_param.data(), sizeof(transfer_float), host_param.size(), fp);
    fclose(fp);
    if (fwrite_count != host_param.size()) {
      errorQuda("Unable to write trained parameters to %s (%lu neq %lu).\n", save_param_path.c_str(), fwrite_count,
                host_param.size());
    }
    if (getVerbosity() >= QUDA_VERBOSE) { printfQuda("Trained parameters saved to %s ...\n", save_param_path.c_str()); }
  }

  void MadwfAcc::load_parameter(int Ls, int Ls_base)
  {
    constexpr int complex_matrix_size = static_cast<int>(transfer_t); // spin by spin
    size_t param_size = Ls * Ls_base * complex_matrix_size * 2;
    std::vector<transfer_float> host_param(param_size);

    char param_file_name[512];
    // Note that all ranks load from the same file.
    sprintf(param_file_name, "/madwf_trained_param_ls_%02d_%02d_mu_%.3f.dat", Ls, Ls_base, mu);
    std::string param_file_name_str(param_file_name);
    auto search_cache = host_training_param_cache.find(param_file_name_str);
    if (search_cache != host_training_param_cache.end()) {
      host_param = search_cache->second;
      if (getVerbosity() >= QUDA_VERBOSE) { printfQuda("Training params loaded from CACHE.\n"); }
    } else {
      // the parameter is not in cache: load from file system.
      std::string save_param_path(param.madwf_param_infile);
      save_param_path += param_file_name_str;
      FILE *fp = fopen(save_param_path.c_str(), "rb");
      if (!fp) { errorQuda("Unable to open file %s\n", save_param_path.c_str()); }
      size_t fread_count = fread(host_param.data(), sizeof(float), host_param.size(), fp);
      fclose(fp);
      if (fread_count != host_param.size()) {
        errorQuda("Unable to load training params from %s (%lu neq %lu).\n", save_param_path.c_str(), fread_count,
                  host_param.size());
      }
      host_training_param_cache.insert({param_file_name_str, host_param});
      printf("Rank %05d: Training params loaded from FILE %s ... \n", comm_rank(), save_param_path.c_str());
      comm_barrier();
      if (getVerbosity() >= QUDA_VERBOSE) { printfQuda("All ranks loaded.\n"); }
    }
    device_param.resize(param_size); // 2 for complex
    device_param.from_host(host_param);
    trained = true;
  }

  std::unordered_map<std::string, std::vector<float>> MadwfAcc::host_training_param_cache; // empty map

} // namespace quda
