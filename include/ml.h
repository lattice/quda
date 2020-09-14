#pragma once

#include <vector>
#include <random>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <madwf_ml.h>
#include <polynomial.h>

namespace quda
{

  struct MADWFacc {

    using TrainingFloat = float;

    // The parameters to be trained.
    using Tp = madwf_ml::TrainingParameter<TrainingFloat>;
    Tp device_param;
    
    // The diagonal component to suppress/lift the zero modes.
    double mu;
    int Ls_base;

    int null_maxiter;
    double null_tol;
    int train_maxiter;

    // persistent buffers for reuse.
    ColorSpinorField *forward_tmp = nullptr;
    ColorSpinorField *backward_tmp = nullptr;

    bool param_load;
    bool param_save;
    
    char param_infile[256];
    char param_outfile[256];

    // Has device_param been trained?
    bool trained = false;

    MADWFacc(const SolverParam &solve_param):
      mu(solve_param.madwf_diagonal_suppressor),
      Ls_base(solve_param.madwf_ls),
      null_maxiter(solve_param.madwf_null_maxiter),
      null_tol(solve_param.madwf_null_tol),
      train_maxiter(solve_param.madwf_train_maxiter),
      param_load(solve_param.madwf_param_load == QUDA_BOOLEAN_TRUE),
      param_save(solve_param.madwf_param_save == QUDA_BOOLEAN_TRUE)
    {
      strcpy(param_infile, solve_param.madwf_param_infile);
      strcpy(param_outfile, solve_param.madwf_param_outfile);
    
      printfQuda("MADWFacc:\n");
      printfQuda("madwf_mu = %.4f\n", mu);
      printfQuda("madwf_ls = %d\n", Ls_base);
      printfQuda("madwf_null_maxiter = %d\n", null_maxiter);
      printfQuda("madwf_null_tol = %.2f\n", null_tol);
      printfQuda("madwf_train_maxiter = %d\n", train_maxiter);
    
    }
    
    ~MADWFacc() {
      if (forward_tmp) { delete forward_tmp; } 
      if (backward_tmp) { delete backward_tmp; } 
    }

    void fill_random(std::vector<TrainingFloat> &v)
    {
      static std::random_device rd;
      // the good rng
      static std::mt19937 rng(23ul * comm_rank());
      // The gaussian distribution
      static std::normal_distribution<double> n(0., 1.);
  
      for (auto &x : v) { x = 1e-1 * n(rng); }
    }

    template <class Base>
    void apply(Base base, ColorSpinorField &out, const ColorSpinorField &in)
    {
      madwf_ml::transfer_5d_hh(*forward_tmp, in, device_param, false);
      base(*backward_tmp, *forward_tmp);
      madwf_ml::transfer_5d_hh(out, *backward_tmp, device_param, true);
      
      blas::axpy(mu, const_cast<ColorSpinorField &>(in), out);
    }

    // double calculate_chi(ColorSpinorField &out, const ColorSpinorField &in, const Tp &tp, double mu, int Ls_cheap);
    
    template <class Ref, class Base>
    double cost(const Ref &ref, Base base, ColorSpinorField &out, const ColorSpinorField &in)
    {
      
      ColorSpinorParam csParam(in);
      cudaColorSpinorField tmp1(csParam);
      cudaColorSpinorField tmp2(csParam);

      apply(base, tmp1, in);
      ref(tmp2, tmp1);

      blas::copy(out, in);

      // M * T^ * A * T * phi - phi
      return blas::xmyNorm(tmp2, out);

    }
    
    // void ATx(ColorSpinorField &out, const ColorSpinorField &in, const Tp &tp);

    template <class Ref, class Base, class Null>
    void train(const Ref &ref, Base base, Null &null, const ColorSpinorField &in) {
#if 1
      constexpr int complex_matrix_size = 16; // spin by spin
#else
      constexpr int complex_matrix_size = 2; // chiral
#endif   

      ColorSpinorParam csParam(in);
      // csParam.setPrecision(QUDA_HALF_PRECISION);
      cudaColorSpinorField null_rhs(csParam);
      cudaColorSpinorField null_x(csParam);
      
      std::vector<ColorSpinorField *> B(16);
      csParam.setPrecision(QUDA_HALF_PRECISION);
      for (auto &pB : B) {
        pB = new cudaColorSpinorField(csParam);
      }
      auto rng = new RNG(*B[0], 1234);
      rng->Init();

      printfQuda("Generating Null Space Vectors ... \n");
      spinorNoise(null_rhs, *rng, QUDA_NOISE_UNIFORM);
      static_cast<PreconCG &>(null)(null_x, null_rhs, B, 400);
      for (auto &pB : B) {
        printfQuda("pB norm2 = %12.8e\n", blas::norm2(null_rhs));
        blas::ax(5e3 / sqrt(blas::norm2(*pB)), *pB);
      }

      // global orthonormalization of the initial null-space vectors
      if(0) {
        for(int i=0; i<(int)B.size(); i++) {
          for (int j=0; j<i; j++) {
            Complex alpha = blas::cDotProduct(*B[j], *B[i]);// <j,i>
            blas::caxpy(-alpha, *B[j], *B[i]); // i - <j,i> j
          }
          double nrm2 = blas::norm2(*B[i]);
          if (nrm2 > 1e-16) { blas::ax(1.0 /sqrt(nrm2), *B[i]); } // i / <i,i>
          else errorQuda("\nCannot normalize %u vector\n", i);
        }
      }

      bool global_reduction = commGlobalReduction();
      commGlobalReductionSet(false);

      int Ls = in.X(4);
      int param_size = Ls * Ls_base * complex_matrix_size * 2;

      device_param.resize(param_size); // 2 for complex
      
      cudaColorSpinorField chi(csParam);
      cudaColorSpinorField tmp(csParam);
      cudaColorSpinorField theta(csParam);
      cudaColorSpinorField lambda(csParam);

      cudaColorSpinorField Mchi(csParam);

      double residual = 0.0;
      int count = 0;
      for (const auto &phi : B) {
        residual += blas::norm2(*phi);
        printfQuda("reference dslash norm %03d = %8.4e\n", count, blas::norm2(*phi));
        count++;
      }
      printfQuda("reference dslash norm = %8.4e\n", residual);

      csParam.x[4] = Ls_base;
      csParam.create = QUDA_ZERO_FIELD_CREATE;

      cudaColorSpinorField ATchi(csParam);
      cudaColorSpinorField ATphi(csParam);
      cudaColorSpinorField ADphi(csParam);

      cudaColorSpinorField ATMchi(csParam);

      forward_tmp = new cudaColorSpinorField(csParam);
      backward_tmp = new cudaColorSpinorField(csParam);

      std::vector<TrainingFloat> host_param(param_size);
      fill_random(host_param);
      
      device_param.resize(param_size);
      device_param.from_host(host_param);

      Tp d1(param_size);
      Tp d2(param_size);

      Tp P(param_size);

      Tp D_old(param_size);

      // double pmu = 0.0;

      // double old_chi2 = 0.0;
      TrainingFloat alpha;
      TrainingFloat b = 0.8;
      printfQuda("beta          = %.3f\n", b);
      printfQuda("training mu   = %.3f\n", mu);
      for (int iteration = 0; iteration < train_maxiter; iteration++) {

        Tp D(param_size);
        // double dmu = 0.0;

        double chi2 = 0.0;
        std::vector<double> a(5, 0.0);

        for (const auto &phi : B) {
          chi2 += cost(ref, base, chi, *phi);
          // ATx(ATphi, *phi, T);
          madwf_ml::transfer_5d_hh(*forward_tmp, *phi, device_param, false);
          
          base(ATphi, *forward_tmp);

          // inner_dslash(Mchi, chi);
          ref(Mchi, chi);
          
          // ATx(ATMchi, Mchi, T);
          madwf_ml::transfer_5d_hh(*forward_tmp, Mchi, device_param, false);
          
          base(ATMchi, *forward_tmp);

          // d1 = A * T * phi -x- M * chi
          madwf_ml::tensor_5d_hh(ATphi, Mchi, d1);
          // d2 = A * T * M * phi -x- phi
          madwf_ml::tensor_5d_hh(ATMchi, *phi, d2);

          madwf_ml::axpby(D, 2.0f, d1, 2.0f, d2);
          // dmu += 2.0 * reDotProduct(Mchi, *phi);
        }

        madwf_ml::axpby(P, (b - 1), P, (1 - b), D);
        // pmu = b * pmu + (1-b) * dmu;

        chi2 = 0.0;
        // line search
        for (const auto &phi : B) {

          double ind_chi2 = cost(ref, base, chi, *phi);
          chi2 += ind_chi2;

          // ATx(ATphi, *phi, T);
          madwf_ml::transfer_5d_hh(*forward_tmp, *phi, device_param, false);
          base(ATphi, *forward_tmp);

          // D' * A * T * phi
          madwf_ml::transfer_5d_hh(theta, ATphi, P, true);

          // ATx(ADphi, *phi, P);          
          madwf_ml::transfer_5d_hh(*forward_tmp, *phi, P, false);
          base(ADphi, *forward_tmp);

          // T' * A * D * phi
          madwf_ml::transfer_5d_hh(tmp, ADphi, device_param, true);
          // theta
          blas::axpy(1.0, theta, tmp);
          // axpy(pmu, *phi, tmp);
          
          // inner_dslash(theta, tmp);
          ref(theta, tmp);

          // lambda = D' * A * D * phi
          madwf_ml::transfer_5d_hh(tmp, ADphi, P, true);

          // inner_dslash(lambda, tmp);
          ref(lambda, tmp);

          std::vector<ColorSpinorField *> lhs {&chi, &theta, &lambda};
          std::vector<ColorSpinorField *> rhs {&chi, &theta, &lambda};
          Complex dot[9];
          blas::cDotProduct(dot, lhs, rhs);

          a[0] += dot[0].real();
          a[1] += -2.0 * dot[1].real();
          a[2] += dot[4].real() + 2.0 * dot[2].real();
          a[3] += -2.0 * dot[5].real();
          a[4] += dot[8].real();
        }

        double r[3] = {0.0, 0.0, 0.0};
        solve_deg3(4.0 * a[4], 3.0 * a[3], 2.0 * a[2], a[1], r[0], r[1], r[2]);

        // try the three roots
        double try_root[3];
        for (int i = 0; i < 3; i++) { try_root[i] = eval_deg4(a[4], a[3], a[2], a[1], a[0], r[i]); }

        if (try_root[0] < try_root[1] && try_root[0] < try_root[2]) {
          alpha = r[0];
        } else if (try_root[1] < try_root[2]) {
          alpha = r[1];
        } else {
          alpha = r[2];
        }
        madwf_ml::axpby(device_param, 0.0f, device_param, -alpha, P);
        // mu -= alpha * pmu;

        printfQuda("grad min iter %03d: %04d chi2 = %8.4e, chi2 %% = %8.4e, alpha = %+8.4e, mu = %+8.4e\n", comm_rank(),
                   iteration, chi2, chi2 / residual, alpha, mu);
      
        // if((chi2 - old_chi2) * (chi2 - old_chi2) / (ref * ref) / (old_chi2*old_chi2 / (ref*ref)) < 1e-10){ break; }
        // old_chi2 = chi2;
      }

      printfQuda("Training finished ...\n");
      count = 0;
      for (const auto &phi : B) {
        double ind_chi2 = cost(ref, base, chi, *phi);
        double phi2 = blas::norm2(*phi);
        printfQuda("chi2 %03d %% = %8.4e, phi2 = %8.4e\n", count, ind_chi2 / phi2, phi2);
        count++;
      }
/**
      tp = T.to_host();

      std::string save_param_path(getenv("QUDA_RESOURCE_PATH"));
      char cstring[512];
      // sprintf(cstring, "/training_param_rank_%03d_ls_%02d_%02d_mu_%.3f.dat", comm_rank(), Ls_in, Ls_cheap, mu);
      sprintf(cstring, "/training_param_rank_%05d_ls_%02d_%02d_mu_%.3f_it_%02d.dat", comm_rank(), Ls_in, Ls_cheap, mu, inner_iterations);
      save_param_path += std::string(cstring);
      FILE *fp = fopen(save_param_path.c_str(), "w");
      size_t fwrite_count = fwrite(tp.data(), sizeof(TrainingFloat), tp.size(), fp);
      fclose(fp);
      if (fwrite_count != tp.size()) {
        errorQuda("Unable to write training params to %s (%lu neq %lu).\n", save_param_path.c_str(), fwrite_count,
                  tp.size());
      }
      printfQuda("Training params saved to %s ...\n", save_param_path.c_str());
      
      
      double dummy_for_sync = 0.0;
      reduceDouble(dummy_for_sync);
      
      return;
*/
      /** ... */
      
      // Destroy all dynamically allocated stuff.
      rng->Release();
      delete rng;
      for (auto &pB : B) {
        delete pB;
      }
      commGlobalReductionSet(global_reduction);
    }

  };

} // namespace quda

